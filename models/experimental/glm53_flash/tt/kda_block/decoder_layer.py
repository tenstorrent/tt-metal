# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash KDA decoder block (``Glm5NextTextDecoderLayer`` with ``linear_attention``, layers ``kda_dense``
and ``kda_moe``): hyper-connection -> KDA -> mix -> hyper-connection -> FFN -> mix.

``hc`` = hc_mult (4) residual streams of ``D`` = hidden_size channels are carried as ``[B, S, hc, D]`` bf16 TILE
through the block, as in the DeepSeek-V4 blocks::

    post, comb, collapsed = attn_hc(streams)
    streams = mix_streams(post, comb, kda(input_layernorm(collapsed)), streams)
    post, comb, collapsed = ffn_hc(streams)
    streams = mix_streams(post, comb, ffn(post_attention_layernorm(collapsed)), streams)

Every op is a plain ttnn op or one of the Wormhole-capable experimental ops (``mhc_split_sinkhorn``, the KDA ops); no
weight goes through the DRAM prefetcher.

* :meth:`GLMKDADecoderLayer.decode` -- one token for each of ``B`` users, ``hidden_streams [B, 1, hc, D]``. Runs on
  Wormhole.
* :meth:`GLMKDADecoderLayer.forward` -- one user's prefill chunk, ``hidden_streams [1, T, hc, D]``. Needs
  ``ttnn.experimental.kda.prepare_chunk_recurrence``, which places 33 dataflow buffers per core and so does not fit
  Wormhole's 32 (see ``tests/test_kda.py::PREP_BLOCKER``).

The KDA state (per-user recurrent ``S`` and conv history) is a :class:`~models.experimental.glm53_flash.tt.kda.KDAStateCache`
from :meth:`allocate_state`, updated in place by both paths.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn

from models.experimental.glm53_flash.tt.ds_v4.common import DeepSeekV4Module, _HIFI4, _region
from models.experimental.glm53_flash.tt.ds_v4.streams import (
    _strip_prefix,
    load_norm_gamma,
    rows_from_tokens,
    tokens_from_rows,
)
from models.experimental.glm53_flash.tt.ds_v4.weight_cache import WeightCache, _as_cache
from models.experimental.glm53_flash.tt.kda import CHUNK, KDAStateCache
from models.experimental.glm53_flash.tt.kda_block.attention import build_kda
from models.experimental.glm53_flash.tt.kda_block.ffn import build_ffn
from models.experimental.glm53_flash.tt.kda_block.hyperconnection import GLMHyperConnection, mix_streams


class GLMKDADecoderLayer(DeepSeekV4Module):
    """ttnn port of a KDA ``Glm5NextTextDecoderLayer`` (see the module docstring).

    ``weights`` keys mirror the HF decoder-layer state_dict: ``self_attn.*`` (see
    :func:`~models.experimental.glm53_flash.tt.kda_block.attention.build_kda`), ``mlp.*``, ``attn_hc.{fn,base,scale}``,
    ``ffn_hc.{fn,base,scale}``, ``input_layernorm.weight`` and ``post_attention_layernorm.weight``; each value a torch
    tensor or a zero-arg thunk returning one.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        expert_dtype: ttnn.DataType = ttnn.bfloat4_b,
    ):
        """``weight_dtype`` is that of the wide KDA projections, the dense MLP and the shared expert; ``expert_dtype``
        that of the routed experts. The hyper-connection ``fn`` is fp32, the norms and low-rank gates bf16."""
        if config.layer_types[layer_idx] != "linear_attention":
            raise ValueError(
                f"layer {layer_idx} is {config.layer_types[layer_idx]!r}, not a KDA (linear_attention) layer"
            )
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.eps = config.rms_norm_eps
        self.is_moe = config.mlp_layer_types[layer_idx] == "sparse"
        cache = _as_cache(cache)

        self.self_attn = build_kda(
            config, _strip_prefix(weights, "self_attn"), device, cache=cache.sub("self_attn"), weight_dtype=weight_dtype
        )
        self.mlp = build_ffn(
            config,
            layer_idx,
            _strip_prefix(weights, "mlp"),
            device,
            cache=cache.sub("mlp"),
            weight_dtype=weight_dtype,
            expert_dtype=expert_dtype,
        )
        self.attn_hc = GLMHyperConnection(config, _strip_prefix(weights, "attn_hc"), device, cache=cache.sub("attn_hc"))
        self.ffn_hc = GLMHyperConnection(config, _strip_prefix(weights, "ffn_hc"), device, cache=cache.sub("ffn_hc"))
        self.input_layernorm_weight = load_norm_gamma(
            weights["input_layernorm.weight"], device, cache.file("input_layernorm")
        )
        self.post_attention_layernorm_weight = load_norm_gamma(
            weights["post_attention_layernorm.weight"], device, cache.file("post_attention_layernorm")
        )

    def allocate_state(self, slots: int) -> KDAStateCache:
        """Zeroed KDA state for ``slots`` users: the start of every user's sequence."""
        return self.self_attn.allocate_state(slots)

    def _norm(self, collapsed: ttnn.Tensor, gamma: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, S, 1, D]`` collapsed streams -> the RMSNormed sublayer input ``[1, 1, B*S, D]``."""
        rows = rows_from_tokens(collapsed)
        out = ttnn.rms_norm(rows, weight=gamma, epsilon=self.eps, compute_kernel_config=_HIFI4)
        ttnn.deallocate(rows)
        return out

    def _block(self, hidden_streams: ttnn.Tensor, attend, decode: bool) -> ttnn.Tensor:
        """The shared body: ``attend(normed [1, 1, T, D]) -> [1, 1, T, D]`` is the only part that differs."""
        b, s = hidden_streams.shape[0], hidden_streams.shape[1]
        with _region("ATTN_HC"):
            post, comb, collapsed = self.attn_hc(hidden_streams)
            normed = self._norm(collapsed, self.input_layernorm_weight)
            ttnn.deallocate(collapsed)
        with _region("KDA"):
            attn_out = attend(normed)
            ttnn.deallocate(normed)
        with _region("ATTN_MIX"):
            attn_tokens = tokens_from_rows(attn_out, b, s)
            streams = mix_streams(post, comb, attn_tokens, hidden_streams)
            for tensor in (post, comb, attn_out, attn_tokens):
                ttnn.deallocate(tensor)

        with _region("FFN_HC"):
            post, comb, collapsed = self.ffn_hc(streams)
            normed = self._norm(collapsed, self.post_attention_layernorm_weight)
            ttnn.deallocate(collapsed)
        with _region("MOE" if self.is_moe else "MLP"):
            mlp_out = self.mlp(normed, decode=decode)
            ttnn.deallocate(normed)
        with _region("FFN_MIX"):
            mlp_tokens = tokens_from_rows(mlp_out, b, s)
            out = mix_streams(post, comb, mlp_tokens, streams)
            for tensor in (post, comb, mlp_out, mlp_tokens, streams):
                ttnn.deallocate(tensor)
        return out

    def decode(
        self, hidden_streams: ttnn.Tensor, state: KDAStateCache, active: Optional[ttnn.Tensor] = None
    ) -> ttnn.Tensor:
        """One token for each user slot: ``hidden_streams`` ``[B, 1, hc, D]`` bf16 TILE with ``B == state.slots`` ->
        the new streams ``[B, 1, hc, D]``. ``state`` is advanced in place; ``active`` (optional ``[B, 1, 1, 1]`` fp32
        1/0) leaves inactive slots' state bit-identical. ``hidden_streams`` is left intact."""
        shape = tuple(hidden_streams.shape)
        if len(shape) != 4 or shape[1] != 1 or shape[2] != self.hc or shape[3] != self.hidden:
            raise ValueError(f"expected decode streams [B, 1, {self.hc}, {self.hidden}], got {shape}")
        if shape[0] != state.slots:
            raise ValueError(f"{shape[0]} decode rows for a state of {state.slots} slots")
        return self._block(hidden_streams, lambda normed: self.self_attn.decode(normed, state, active), decode=True)

    def forward(
        self,
        hidden_streams: ttnn.Tensor,
        state: KDAStateCache,
        slot: int = 0,
        valid_tokens: Optional[int] = None,
    ) -> ttnn.Tensor:
        """One prefill chunk of user ``slot``: ``hidden_streams`` ``[1, T, hc, D]`` bf16 TILE, ``T`` a multiple of 32,
        rows from ``valid_tokens`` (default ``T``) on are padding -> the new streams ``[1, T, hc, D]`` (padding rows
        unspecified). Continues from and replaces the slot's state."""
        shape = tuple(hidden_streams.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[2] != self.hc or shape[3] != self.hidden:
            raise ValueError(f"expected prefill streams [1, T, {self.hc}, {self.hidden}], got {shape}")
        t = shape[1]
        if t == 0 or t % CHUNK:
            raise ValueError(f"chunk length {t} must be a positive multiple of {CHUNK}")
        valid = t if valid_tokens is None else valid_tokens
        if not 0 < valid <= t:
            raise ValueError(f"valid_tokens {valid} outside (0, {t}]")
        actual_end = ttnn.from_torch(
            torch.tensor([valid], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        valid_end_f = ttnn.from_torch(
            torch.tensor([float(valid)], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        return self._block(
            hidden_streams,
            lambda normed: self.self_attn.prefill(normed, state, slot, actual_end=actual_end, valid_end_f=valid_end_f),
            decode=False,
        )
