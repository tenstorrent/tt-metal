# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 model head for dots.ocr prefill: final RMSNorm -> LM head -> argmax.

This is the tail that runs after the decoder body at the end of prefill to
produce the next-token logits/id (greedy). The LM head is the big
``H -> vocab`` matmul; in TP4 it is column-parallel — each chip computes
``vocab/ndev`` logits, then the shards are all-gathered for a global argmax.

By default only the LAST token's logits are produced (the prefill -> first
decode-token hand-off), matching the M=1 (32-padded) head matmul seen in the
profiler.
"""

import ttnn

from models.experimental.dots_ocr_tp4.tt.common import (
    all_gather_last_dim,
    from_replicated_to_torch,
    mesh_num_devices,
    shard_to_mesh,
)
from models.experimental.dots_ocr_tp4.tt.rmsnorm import DotsOCRRMSNormTP4
from models.experimental.tt_symbiote.core.module import TTNNModule


class DotsOCRLMHeadTP4(TTNNModule):
    def __init__(self, mesh_device, config, weight_dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = config
        self.weight_dtype = weight_dtype
        self.num_devices = max(1, mesh_num_devices(mesh_device))
        self.norm = None
        self.lm_head_w = None
        self.vocab_size = None
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def from_torch(cls, mesh_device, config, torch_norm, torch_lm_head, weight_dtype=ttnn.bfloat16):
        m = cls(mesh_device, config, weight_dtype=weight_dtype)
        m.norm = DotsOCRRMSNormTP4.from_torch(mesh_device, torch_norm, eps=config.rms_norm_eps)

        w = torch_lm_head.weight.data  # [vocab, H]
        m.vocab_size = int(w.shape[0])
        nd = m.num_devices
        assert m.vocab_size % nd == 0, f"vocab {m.vocab_size} not divisible by {nd}"
        m.v_shard = m.vocab_size // nd  # per-chip vocab width (for distributed argmax)
        # Column-parallel: ttnn.linear wants [K=H, N=vocab]; shard N across chips.
        # (BFP8 weight was tried to cut the M=1 head-matmul bandwidth and was
        # speed-neutral -- the head matmul is not weight-bandwidth-bound -- so the
        # weight is kept at the requested dtype for accuracy.)
        m.lm_head_w = shard_to_mesh(w.t().contiguous(), mesh_device, dim=-1, dtype=weight_dtype)
        m.to_device(mesh_device)
        m._preprocessed_weight = True
        m._weights_on_device = True
        return m

    def forward(self, hidden: ttnn.Tensor, last_token_only: bool = True, return_token: bool = True, token_index=None):
        """hidden: replicated [B, S, H]. Returns (logits, token_ids_torch_or_None).

        logits: replicated [B, T, vocab]; T = 1 if last_token_only else S.

        ``token_index`` selects which sequence position to read when
        ``last_token_only`` (default = the last, S-1). When the sequence is
        right-padded to a tile multiple, pass the real last position here:
        causal attention makes the padded tail "future" tokens that don't
        affect earlier positions, so the logits at ``token_index`` are exact.
        """
        x = hidden
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze(x, 1)  # [B, 1, S, H]

        if last_token_only:
            B, _, S, H = (int(d) for d in x.shape)
            idx = (S - 1) if token_index is None else int(token_index)
            x = ttnn.slice(x, [0, 0, idx, 0], [B, 1, idx + 1, H])  # [B, 1, 1, H]

        x = self.norm.forward(x)

        # Per-chip logits over its vocab shard, then gather to full vocab.
        local_logits = ttnn.linear(
            x,
            self.lm_head_w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        logits = all_gather_last_dim(local_logits, self.mesh_device)
        ttnn.deallocate(local_logits)

        token_ids = None
        if return_token:
            tok = ttnn.argmax(logits, dim=-1, keepdim=False)  # uint32 over vocab
            token_ids = from_replicated_to_torch(tok, self.mesh_device)
            ttnn.deallocate(tok)

        return logits, token_ids

    def forward_token_dist(self, hidden: ttnn.Tensor, token_index=None):
        """Decode head via DISTRIBUTED argmax (the fast decode path).

        Each chip argmaxes only its local ``vocab/nd`` shard, so we avoid the
        full-vocab all-gather + full-vocab argmax that dominates decode (~40% of
        device time at M=1). Returns the per-chip ``(max_value, local_index)`` as
        two ``[1,1,1,1]`` device tensors (each chip holds its own); the caller
        gathers the nd candidates to host and computes
        ``global_token = winner_chip * v_shard + local_index[winner]``. Trace-
        capturable (no host readback here)."""
        x = hidden
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze(x, 1)  # [B, 1, S, H]
        B, _, S, H = (int(d) for d in x.shape)
        idx = (S - 1) if token_index is None else int(token_index)
        x = ttnn.slice(x, [0, 0, idx, 0], [B, 1, idx + 1, H], memory_config=ttnn.L1_MEMORY_CONFIG)

        x = self.norm.forward(x)
        local_logits = ttnn.linear(
            x,
            self.lm_head_w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )  # [1, 1, 1, v_shard] on each chip
        local_val = ttnn.max(local_logits, dim=-1, keepdim=True)  # [1,1,1,1] per chip
        local_idx = ttnn.argmax(local_logits, dim=-1, keepdim=True)  # [1,1,1,1] per chip (local idx)
        ttnn.deallocate(local_logits)
        return local_val, local_idx
