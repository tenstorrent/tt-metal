# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of the HunyuanImage-3.0 MoE router (gate).
#
# Mirrors the production inference path `HunyuanTopKGate.easy_topk`
# (see ref/moe/gate.py):
#     logits = x @ wg.T            # wg: [num_experts, hidden], bias-free
#     gates  = softmax(logits, dim=-1)         # over the expert axis
#     w, idx = topk(gates, moe_topk)
#     w      = w / clamp(w.sum(-1, keepdim=True), min=1e-8)   # if norm_topk_prob
#
# Notes:
# - The reference keeps the router projection + softmax in fp32 because bf16
#   ties can flip expert selection. Here the router runs end-to-end in bf16
#   (HiFi2 + fp32 dest accumulation on the matmul; PCC-validated against the
#   fp32 reference before relying on this in production).
# - ttnn.topk requires the reduced (last) dim to be a multiple of 64 and k a
#   power of two; num_experts=64 / moe_topk=8 satisfy both.

import ttnn
from models.common.lightweightmodule import LightweightModule

from ..cache import cache_file
from ..matmul_utils import l1_sharded_linear, to_interleaved_if_sharded
from ..parallel_utils import decode_mm_program_config


class HunyuanTtTopKGate(LightweightModule):
    """
    Single-device TTNN MoE router.

    Args:
        device:       TTNN device.
        hidden_size:  Model hidden size (e.g. 4096).
        num_experts:  Number of routed experts (e.g. 64).
        moe_topk:     Experts selected per token (e.g. 8).
        state_dict:   Model state_dict (plain torch tensors).
        weight_key:   Router weight key, with or without the ``.weight`` suffix,
                      e.g. ``model.layers.0.mlp.gate.wg``.
        norm_topk_prob: Renormalise the selected top-k weights (default True).
        weight_dtype: TTNN dtype for the router weight (default bfloat16).
    """

    def __init__(
        self,
        device,
        hidden_size: int,
        num_experts: int,
        moe_topk: int,
        state_dict: dict,
        weight_key: str,
        norm_topk_prob: bool = True,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.num_experts = num_experts
        self.moe_topk = moe_topk
        self.norm_topk_prob = norm_topk_prob
        self.weight_dtype = weight_dtype

        key = weight_key.removesuffix(".weight") + ".weight"
        w = state_dict[key]  # [num_experts, hidden]
        # ttnn.linear computes x @ weight, so store as [hidden, num_experts].
        w_t = w.transpose(0, 1).contiguous().float()
        is_mesh = device.__class__.__name__ == "MeshDevice"
        self.wg = ttnn.as_tensor(
            w_t,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
            cache_file_name=cache_file(weight_cache_path, key),
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor):
        """
        Args:
            x: TTNN tensor [B, S, H] in TILE_LAYOUT.
        Returns:
            (topk_weight, topk_index)
              topk_weight: [B, S, moe_topk]  routed probabilities (normalised)
              topk_index:  [B, S, moe_topk]  selected expert ids (uint)
        """
        # Upcast activations to match the fp32 router weight (mirrors the
        # reference, which casts hidden states to fp32 before the projection).
        if self.weight_dtype == ttnn.float32 and x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)

        # Router projection (K=4096, N=num_experts=64). Skinny-N: decode_mm picks
        # gather-K (mcast_in0=False) — BH 25.4us split-N → 17.7us gather-K nc=8
        # (test_matmul_shard_sweep / decode_mm skinny-N path). Returns None for Mt>=8.
        # allow_width_shard=False: MoE reuses this same `x` for every expert body.
        # The default width-shard path (taken when gate_pc is None, e.g. S=1 decode)
        # deallocates its activation input — which then TT_FATALs in expert linear.
        gate_pc = decode_mm_program_config(self.device, x.shape[-2], x.shape[-1], self.wg.shape[-1])
        logits = l1_sharded_linear(
            x,
            self.wg,
            compute_kernel_config=self.compute_kernel_config,
            program_config=gate_pc,
            allow_width_shard=False,
        )  # [B, S, num_experts]
        logits = to_interleaved_if_sharded(logits)

        gates = ttnn.softmax(logits, dim=-1)
        ttnn.deallocate(logits)

        # ttnn.topk requires bf16/bf8 input; only casts (and only then frees the
        # pre-cast tensor) if gates isn't already bf16 — gates_bf16 aliases gates
        # otherwise, so deallocating gates unconditionally would free it out from
        # under gates_bf16.
        if gates.get_dtype() != ttnn.bfloat16:
            gates_bf16 = ttnn.typecast(gates, ttnn.bfloat16)
            ttnn.deallocate(gates)
        else:
            gates_bf16 = gates

        topk_weight, topk_index = ttnn.topk(gates_bf16, self.moe_topk, dim=-1)
        ttnn.deallocate(gates_bf16)

        # Pad moe_topk up to the next full tile (32-wide) so every later
        # ttnn.sum(..., dim=-1) on these tensors — the denom sum below, and once
        # per expert in moe.py/moe_parallel.py's combine loop — sees an already
        # tile-aligned width and skips its internal FillPad (fill_pad.cpp only
        # fires when logical width != tile-rounded width). Weight padding is 0
        # (inert in any sum); index padding is an out-of-range sentinel
        # (num_experts, never a real expert id) so eq(topk_idx, gid) never
        # matches a padded slot.
        pad_amt = -self.moe_topk % 32
        if pad_amt:
            topk_weight = ttnn.pad(topk_weight, [(0, 0), (0, 0), (0, pad_amt)], value=0.0)
            topk_index = ttnn.pad(topk_index, [(0, 0), (0, 0), (0, pad_amt)], value=float(self.num_experts))

        if self.norm_topk_prob:
            denom = ttnn.sum(topk_weight, dim=-1, keepdim=True)  # [B, S, 1]
            denom = ttnn.clip(denom, 1e-8, float("inf"))
            topk_weight = ttnn.div(topk_weight, denom)
            ttnn.deallocate(denom)

        return topk_weight, topk_index
