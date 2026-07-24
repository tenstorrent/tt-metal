# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 text encoder (Qwen3-Embedding-0.6B) on TTNN — the prompt front-end.

Reference: `text_encoder(input_ids, ...).last_hidden_state` in the ACE-Step pipeline
(conditioning_embed.py). It is a standard **causal** Qwen3Model that turns tokenized prompt text
into `text_hidden_states [1, L, 1024]`, which the ConditionEncoder's text_projector consumes.

    x = embed_tokens[input_ids]            # vocab embedding lookup (host gather)
    for layer in layers:                   # 28 causal Qwen3 decoder layers
        x = layer(x, rope, causal_mask)
    x = norm(x)                            # final RMSNorm

Structurally identical to AceStepLyricEncoder / AceStepEncoderLayer (self_attn with q/k/v/o_proj +
per-head q/k-norm, SwiGLU MLP, pre-norms) — so we REUSE `AceStepEncoderLayer` + `RMSNorm1D`. The
only differences: (1) input is a vocab embedding lookup instead of a Linear, done on host (a gather
is not a device matmul); (2) attention is CAUSAL (verified against the reference), fed as an
additive causal mask to every layer's `attn_mask` (the attention op runs `is_causal=False` + mask);
(3) all layers are full-attention (no sliding window), hidden=1024, 28 layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.experimental.acestep.tt.encoder_layer import AceStepEncoderLayer

NEG_INF = -1e9


@dataclass
class AceStepTextEncoderConfig:
    # Vocab embedding table [vocab, hidden] (host lookup; kept on host as a torch tensor).
    embed_tokens: torch.Tensor
    # Final norm.
    norm_weight: LazyWeight
    # Per-layer AceStepEncoderLayerConfig (all full-attention, in order).
    layer_configs: list = field(default_factory=list)

    hidden_size: int = 1024
    eps: float = 1e-6
    mesh_device: ttnn.MeshDevice | None = None
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    def resolved(self) -> "AceStepTextEncoderConfig":
        if self.mesh_device is None:
            self.mesh_device = self.norm_weight.device
        if self.compute_kernel_config is None:
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        return self


class AceStepTextEncoder(LightweightModule):
    """forward(input_ids [1, L] torch.LongTensor, cos, sin) -> [1,1,L,hidden] text_hidden_states.

    The vocab embedding is a host gather (input_ids -> embeddings); the 28 causal layers run on
    device. A causal additive mask is built once per sequence length and fed to every layer.
    """

    def __init__(self, config: AceStepTextEncoderConfig):
        super().__init__()
        self.config = config.resolved()
        self.embed_tokens = config.embed_tokens  # host [vocab, hidden] (kept for lyric-embed lookup)
        # Resident on-device embedding table (ROW_MAJOR, as ttnn.embedding requires) so the vocab
        # gather runs on-device (trace-safe) instead of a host lookup + from_torch each call.
        self.embed_weight = ttnn.from_torch(
            config.embed_tokens.reshape(config.embed_tokens.shape[0], config.hidden_size),
            device=self.config.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.layers = [AceStepEncoderLayer(lc) for lc in config.layer_configs]
        self.norm = RMSNorm1D.from_config(RMSNorm1DConfig(weight=config.norm_weight, eps=config.eps))
        self._mask_cache: dict[int, ttnn.Tensor] = {}

    @classmethod
    def from_config(cls, config: AceStepTextEncoderConfig):
        return cls(config)

    def _causal_mask(self, seq_len: int) -> ttnn.Tensor:
        """Additive [1,1,seq,seq] causal mask (0 on/below diagonal, -inf above)."""
        if seq_len not in self._mask_cache:
            m = torch.triu(torch.full((seq_len, seq_len), NEG_INF), diagonal=1).reshape(1, 1, seq_len, seq_len)
            self._mask_cache[seq_len] = ttnn.from_torch(
                m, device=self.config.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
        return self._mask_cache[seq_len]

    def embed(self, input_ids) -> ttnn.Tensor:
        """On-device vocab gather -> device tensor [1,1,L,hidden].

        Accepts a torch LongTensor [L] or [1,L] (eager/test path — uploaded once at the input
        boundary, before any trace region), or an already-on-device ttnn index tensor [1,L] uint32
        (traced path). The gather itself is `ttnn.embedding` on-device.
        """
        if isinstance(input_ids, ttnn.Tensor):
            ids = input_ids
        else:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            ids = ttnn.from_torch(
                input_ids.to(torch.int32).reshape(1, input_ids.shape[-1]),
                device=self.config.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        L = ids.shape[-1]
        emb = ttnn.embedding(ids, self.embed_weight, layout=ttnn.TILE_LAYOUT)  # [1, L, hidden]
        return ttnn.reshape(emb, (1, 1, L, self.config.hidden_size))

    def forward(self, input_ids: torch.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        x = self.embed(input_ids)
        seq_len = x.shape[2]
        mask = self._causal_mask(seq_len)
        for layer in self.layers:
            x = layer.forward(x, cos, sin, attn_mask=mask)
        return self.norm.forward(x, mode="prefill")
