# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the DFlash drafter (``MuseGlimmerAssistantModel``).

The drafter proposes ``block_size`` tokens from one forward pass; the target then
verifies them in a single wide forward.  Structurally it is a small (5-layer)
transformer, but it differs from the target decoder in four ways that each cause
a silent accuracy loss rather than a crash if ported by analogy:

1. **Plain RMSNorm, not centered.**  The target's text layers use
   ``MuseGlimmerTextCenteredRMSNorm`` (``x * (1 + w)``), which the target port
   pre-folds into ``_MuseGlimmerNorm``.  The drafter uses ordinary
   ``x * w``.  Reusing the target's norm module here would add a spurious ``+1``
   to every weight.

2. **Q and K/V come from different tensors, so QKV cannot be fused.**  Q is
   projected from the ``block_size`` window alone; K/V are projected from
   ``concat(context, window)``.  The target port fuses QKV into one matmul; doing
   that here would be wrong.

3. **Attention is bidirectional, not causal.**  ``is_causal`` is ``False`` and the
   mask is ``create_bidirectional_sliding_window_mask``, i.e. ``kv > q - window``
   with *no* upper bound.  The 16 window positions see each other in both
   directions.  A causal mask still runs and still produces plausible tokens - it
   just lowers the acceptance rate, which is invisible without measuring it.

4. **The context half of K/V is not re-normalised per layer.**  Each layer
   applies ``input_layernorm`` to the window hidden states only; the context
   tensor entering K/V is the encoder projection output, shared unchanged by all
   five layers.

The drafter also ships no ``embed_tokens`` and no ``lm_head``: input embeddings
come from the target's table via a plain lookup, and candidate logits come from
the target's ``lm_head``.  See ``tests/reference_dflash.py``, which asserts this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

DEFAULT_BLOCK_SIZE = 16


@dataclass(frozen=True)
class DFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    sliding_window: int
    rms_norm_eps: float
    rope_theta: float
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    max_position_embeddings: int

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def context_fan_in(self) -> int:
        """Last dim of ``context_hidden_states``: one target hidden per tapped layer."""
        return len(self.target_layer_ids) * self.hidden_size

    @property
    def sdpa_scale(self) -> float:
        return float(self.head_dim) ** -0.5


def config_from_hf(hf_config: Any) -> DFlashConfig:
    """Build the frozen config, validating everything the port depends on."""
    layer_types = list(hf_config.layer_types)
    if layer_types != ["sliding_attention"] * hf_config.num_hidden_layers:
        raise ValueError(f"drafter layer_types {layer_types!r}: the port assumes all-sliding")
    rope_theta = hf_config.rope_parameters["rope_theta"]
    return DFlashConfig(
        hidden_size=int(hf_config.hidden_size),
        intermediate_size=int(hf_config.intermediate_size),
        num_attention_heads=int(hf_config.num_attention_heads),
        num_key_value_heads=int(hf_config.num_key_value_heads),
        head_dim=int(hf_config.head_dim),
        num_hidden_layers=int(hf_config.num_hidden_layers),
        sliding_window=int(hf_config.sliding_window),
        rms_norm_eps=float(hf_config.rms_norm_eps),
        rope_theta=float(rope_theta),
        block_size=int(hf_config.block_size),
        mask_token_id=int(hf_config.mask_token_id),
        target_layer_ids=tuple(int(i) for i in hf_config.target_layer_ids),
        max_position_embeddings=int(hf_config.max_position_embeddings),
    )


def _to_device(
    tensor: torch.Tensor,
    *,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def rope_tables(position_ids: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """HF ``default`` RoPE cos/sin for arbitrary (not necessarily contiguous) positions.

    Mirrors ``MuseGlimmerAssistantRotaryEmbedding``: ``inv_freq`` over half the
    head dim, ``emb = cat(freqs, freqs)``, ``attention_scaling == 1.0``.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(position_ids.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def bidirectional_sliding_mask(
    q_positions: torch.Tensor, kv_positions: torch.Tensor, sliding_window: int, dtype: torch.dtype
) -> torch.Tensor:
    """Additive mask for ``create_bidirectional_sliding_window_mask``.

    The composed HF mask is ``bidirectional_mask_function AND
    sliding_window_overlay(w)``.  The first is unconditionally true, so the whole
    condition collapses to ``kv_idx > q_idx - sliding_window`` - a lower bound
    only.  There is deliberately no ``kv <= q`` term: that is what makes the
    diffusion window bidirectional.
    """
    allowed = kv_positions[None, :] > (q_positions[:, None] - sliding_window)
    mask = torch.zeros(allowed.shape, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.reshape(1, 1, *allowed.shape)


def build_context_hidden_states(
    taps: dict[int, ttnn.Tensor], target_layer_ids: tuple[int, ...], *, deallocate: bool = True
) -> ttnn.Tensor:
    """Assemble the drafter's context input from the target's tapped hidden states.

    HF builds it as ``torch.cat([hidden_states[i + 1] for i in target_layer_ids],
    dim=-1)``.  **Order matters**: ``encoder.fc`` is a single dense projection
    over the concatenated 33280-wide vector, so permuting the layers permutes
    that weight's input columns and silently produces a wrong - but entirely
    plausible - projection.
    """
    missing = [i for i in target_layer_ids if i not in taps]
    if missing:
        raise ValueError(f"missing tapped hidden states for layers {missing}")
    ordered = [taps[i] for i in target_layer_ids]
    out = ttnn.concat(ordered, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if deallocate:
        for tensor in ordered:
            ttnn.deallocate(tensor)
    return out


def build_noise_ids(anchor_token_id: int, block_size: int, mask_token_id: int) -> list[int]:
    """``[anchor] + [mask] * (block_size - 1)``.

    The anchor is the last token the *target* committed; the rest are mask tokens
    the drafter denoises.  Embeddings come from the target's table via a plain
    lookup - HF uses ``F.embedding(ids, weight)`` explicitly rather than the
    target's own embedding module, which for this checkpoint also applies a
    normaliser that must NOT be applied here.
    """
    return [int(anchor_token_id)] + [int(mask_token_id)] * (block_size - 1)


class DFlashDrafterCache:
    """Per-layer K/V for the *accepted context only*, mirroring ``DFlashCache``.

    HF appends the whole ``concat(context, window)`` to its cache each forward and
    then crops the window back off (``cache.crop(-block_size)``), leaving only
    context.  This holds the same state by construction: the window's K/V is
    computed per forward and never stored, so there is nothing to crop.

    Entries are post-``k_norm``, post-RoPE, which is what HF's ``cache.update()``
    receives.  ``positions`` tracks the absolute position of each cached row,
    because the sliding-window mask is built from absolute positions.
    """

    def __init__(self, num_layers: int) -> None:
        self.k: list[ttnn.Tensor | None] = [None] * num_layers
        self.v: list[ttnn.Tensor | None] = [None] * num_layers
        self.positions: torch.Tensor = torch.empty(0, dtype=torch.long)

    @property
    def length(self) -> int:
        return int(self.positions.numel())

    def append(self, layer_idx: int, key: ttnn.Tensor, value: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Concatenate on the sequence dim and return the full cached K/V."""
        if self.k[layer_idx] is None:
            self.k[layer_idx], self.v[layer_idx] = key, value
        else:
            merged_k = ttnn.concat([self.k[layer_idx], key], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            merged_v = ttnn.concat([self.v[layer_idx], value], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(self.k[layer_idx])
            ttnn.deallocate(self.v[layer_idx])
            ttnn.deallocate(key)
            ttnn.deallocate(value)
            self.k[layer_idx], self.v[layer_idx] = merged_k, merged_v
        return self.k[layer_idx], self.v[layer_idx]

    def note_positions(self, positions: torch.Tensor) -> None:
        self.positions = torch.cat([self.positions, positions.to(torch.long)])

    def release(self) -> None:
        for store in (self.k, self.v):
            for idx, tensor in enumerate(store):
                if tensor is not None:
                    ttnn.deallocate(tensor)
                    store[idx] = None
        self.positions = torch.empty(0, dtype=torch.long)


class _PlainNorm(LightweightModule):
    """``rms_norm(x) * w`` - the drafter's norm, *not* the target's centered variant."""

    def __init__(self, weight: ttnn.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class _DFlashMLP(LightweightModule):
    """SwiGLU: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, gate: ttnn.Tensor, up: ttnn.Tensor, down: ttnn.Tensor, activation_dtype: ttnn.DataType) -> None:
        super().__init__()
        self.gate = gate
        self.up = up
        self.down = down
        self.activation_dtype = activation_dtype

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.linear(x, self.gate, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.linear(x, self.up, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        activated = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        hidden = ttnn.mul(activated, up, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(activated)
        ttnn.deallocate(up)
        out = ttnn.linear(hidden, self.down, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden)
        return out


class _DFlashLayer(LightweightModule):
    """One drafter decoder layer."""

    def __init__(
        self,
        *,
        config: DFlashConfig,
        input_layernorm: _PlainNorm,
        post_attention_layernorm: _PlainNorm,
        wq: ttnn.Tensor,
        wk: ttnn.Tensor,
        wv: ttnn.Tensor,
        wo: ttnn.Tensor,
        q_norm_weight: ttnn.Tensor,
        k_norm_weight: ttnn.Tensor,
        mlp: _DFlashMLP,
        activation_dtype: ttnn.DataType,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight
        self.mlp = mlp
        self.activation_dtype = activation_dtype

    def _per_head_norm(self, tensor: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        """RMSNorm over ``head_dim`` with a learned scale (drafter QK-norm)."""
        shape = tuple(tensor.shape)
        flat = ttnn.reshape(tensor, (1, 1, shape[0] * shape[1] * shape[2], shape[3]))
        normed = ttnn.rms_norm(
            flat, weight=weight, epsilon=self.config.rms_norm_eps, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(flat)
        return ttnn.reshape(normed, shape)

    @staticmethod
    def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        negated = ttnn.neg(x2)
        ttnn.deallocate(x2)
        out = ttnn.concat([negated, x1], dim=-1)
        ttnn.deallocate(negated)
        ttnn.deallocate(x1)
        return out

    def _apply_rope(self, tensor: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        rotated = self._rotate_half(tensor)
        out = ttnn.add(
            ttnn.mul(tensor, cos, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.mul(rotated, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(rotated)
        return out

    def _split_heads(self, tensor: ttnn.Tensor, seq_len: int, num_heads: int) -> ttnn.Tensor:
        """``[1, 1, s, h*d] -> [1, h, s, d]``."""
        reshaped = ttnn.reshape(tensor, (1, seq_len, num_heads, self.config.head_dim))
        out = ttnn.permute(reshaped, (0, 2, 1, 3))
        ttnn.deallocate(reshaped)
        return out

    def _project_kv(self, source: ttnn.Tensor, seq_len: int, *, rope: tuple[ttnn.Tensor, ttnn.Tensor]):
        """``k_proj``/``v_proj`` + ``k_norm`` + RoPE for one K/V source.

        ``k_proj(cat(context, window)) == cat(k_proj(context), k_proj(window))``
        because it is linear, which is what lets the context half be cached while
        the window half is recomputed each forward.  ``v`` is deliberately not
        normed - only Q and K carry QK-norm.
        """
        config = self.config
        key = ttnn.linear(source, self.wk, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.linear(source, self.wv, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = self._split_heads(key, seq_len, config.num_key_value_heads)
        value = self._split_heads(value, seq_len, config.num_key_value_heads)
        key = self._per_head_norm(key, self.k_norm_weight)
        key = self._apply_rope(key, *rope)
        return key, value

    def _attend(self, query: ttnn.Tensor, key: ttnn.Tensor, value: ttnn.Tensor, mask: ttnn.Tensor, block: int):
        config = self.config
        key = ttnn.repeat_interleave(key, config.num_kv_groups, dim=1)
        value = ttnn.repeat_interleave(value, config.num_kv_groups, dim=1)
        key_t = ttnn.permute(key, (0, 1, 3, 2))
        ttnn.deallocate(key)
        scores = ttnn.matmul(query, key_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(key_t)
        scores = ttnn.mul(scores, config.sdpa_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores = ttnn.add(scores, mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        probs = ttnn.softmax(scores, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scores)
        attn = ttnn.matmul(probs, value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(probs)
        ttnn.deallocate(value)
        merged = ttnn.permute(attn, (0, 2, 1, 3))
        ttnn.deallocate(attn)
        merged = ttnn.reshape(merged, (1, 1, block, config.num_attention_heads * config.head_dim))
        out = ttnn.linear(merged, self.wo, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(merged)
        return out

    def _feed_forward(self, hidden_states: ttnn.Tensor, attn_out: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = ttnn.add(hidden_states, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)
        out = ttnn.add(hidden_states, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(hidden_states)
        return out

    def forward_cached(
        self,
        hidden_states: ttnn.Tensor,
        *,
        context: ttnn.Tensor | None,
        context_len: int,
        rope_ctx: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        rope_win: tuple[ttnn.Tensor, ttnn.Tensor],
        mask: ttnn.Tensor,
        cache: "DFlashDrafterCache",
        layer_idx: int,
    ) -> ttnn.Tensor:
        """Cached variant: append the new context K/V, recompute the window's."""
        config = self.config
        block = int(hidden_states.shape[2])
        normed = self.input_layernorm(hidden_states)

        if context is not None and context_len > 0:
            key_ctx, value_ctx = self._project_kv(context, context_len, rope=rope_ctx)
            cached_k, cached_v = cache.append(layer_idx, key_ctx, value_ctx)
        else:
            cached_k, cached_v = cache.k[layer_idx], cache.v[layer_idx]

        query = ttnn.linear(normed, self.wq, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = self._split_heads(query, block, config.num_attention_heads)
        query = self._per_head_norm(query, self.q_norm_weight)
        query = self._apply_rope(query, *rope_win)

        key_win, value_win = self._project_kv(normed, block, rope=rope_win)
        ttnn.deallocate(normed)

        # The cache tensors persist across iterations, so concat into new buffers
        # and never free the cached halves here.
        key = ttnn.concat([cached_k, key_win], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.concat([cached_v, value_win], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(key_win)
        ttnn.deallocate(value_win)

        attn_out = self._attend(query, key, value, mask, block)
        return self._feed_forward(hidden_states, attn_out)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        context: ttnn.Tensor,
        rope_q: tuple[ttnn.Tensor, ttnn.Tensor],
        rope_kv: tuple[ttnn.Tensor, ttnn.Tensor],
        mask: ttnn.Tensor,
    ) -> ttnn.Tensor:
        config = self.config
        block = int(hidden_states.shape[2])
        context_len = int(context.shape[2])
        kv_len = context_len + block

        normed = self.input_layernorm(hidden_states)

        # K/V see the projected context concatenated with this layer's normed window.
        # The context half is NOT re-normalised here - it is the encoder output, shared by every layer.
        kv_input = ttnn.concat([context, normed], dim=2)

        query = ttnn.linear(normed, self.wq, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.linear(kv_input, self.wk, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.linear(kv_input, self.wv, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(kv_input)
        ttnn.deallocate(normed)

        query = self._split_heads(query, block, config.num_attention_heads)
        key = self._split_heads(key, kv_len, config.num_key_value_heads)
        value = self._split_heads(value, kv_len, config.num_key_value_heads)

        query = self._per_head_norm(query, self.q_norm_weight)
        key = self._per_head_norm(key, self.k_norm_weight)

        query = self._apply_rope(query, *rope_q)
        key = self._apply_rope(key, *rope_kv)

        # GQA: 8 kv heads -> 32 query heads.
        key = ttnn.repeat_interleave(key, config.num_kv_groups, dim=1)
        value = ttnn.repeat_interleave(value, config.num_kv_groups, dim=1)

        key_t = ttnn.permute(key, (0, 1, 3, 2))
        ttnn.deallocate(key)
        scores = ttnn.matmul(query, key_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(query)
        ttnn.deallocate(key_t)
        scores = ttnn.mul(scores, config.sdpa_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores = ttnn.add(scores, mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        probs = ttnn.softmax(scores, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scores)

        attn = ttnn.matmul(probs, value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(probs)
        ttnn.deallocate(value)

        merged = ttnn.permute(attn, (0, 2, 1, 3))
        ttnn.deallocate(attn)
        merged = ttnn.reshape(merged, (1, 1, block, config.num_attention_heads * config.head_dim))
        attn_out = ttnn.linear(merged, self.wo, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(merged)

        hidden_states = ttnn.add(hidden_states, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        normed2 = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed2)
        ttnn.deallocate(normed2)
        out = ttnn.add(hidden_states, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(hidden_states)
        return out


class DFlashDrafter(LightweightModule):
    """The whole 5-layer drafter, including the context projection."""

    def __init__(
        self,
        *,
        config: DFlashConfig,
        mesh_device: ttnn.MeshDevice,
        encoder_fc: ttnn.Tensor,
        encoder_norm: _PlainNorm,
        layers: list[_DFlashLayer],
        final_norm: _PlainNorm,
        activation_dtype: ttnn.DataType,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.encoder_fc = encoder_fc
        self.encoder_norm = encoder_norm
        self.layers = layers
        self.final_norm = final_norm
        self.activation_dtype = activation_dtype

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config: Any,
        mesh_device: ttnn.MeshDevice,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> "DFlashDrafter":
        config = config_from_hf(hf_config)

        def linear_weight(name: str) -> torch.Tensor:
            # HF stores nn.Linear as [out, in]; ttnn.linear wants [in, out].
            return state_dict[name].to(torch.float32).transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)

        def plain_norm(name: str, dim: int) -> _PlainNorm:
            # Plain RMSNorm: the weight is used as-is. No (1 + w) folding here.
            weight = state_dict[name].to(torch.float32).reshape(1, 1, 1, dim)
            return _PlainNorm(
                _to_device(weight.to(torch.bfloat16), mesh_device=mesh_device, dtype=ttnn.bfloat16),
                config.rms_norm_eps,
            )

        def head_norm_weight(name: str) -> ttnn.Tensor:
            weight = state_dict[name].to(torch.float32).reshape(1, 1, 1, config.head_dim)
            return _to_device(weight.to(torch.bfloat16), mesh_device=mesh_device, dtype=ttnn.bfloat16)

        layers: list[_DFlashLayer] = []
        for idx in range(config.num_hidden_layers):
            prefix = f"layers.{idx}"
            layers.append(
                _DFlashLayer(
                    config=config,
                    input_layernorm=plain_norm(f"{prefix}.input_layernorm.weight", config.hidden_size),
                    post_attention_layernorm=plain_norm(
                        f"{prefix}.post_attention_layernorm.weight", config.hidden_size
                    ),
                    wq=_to_device(
                        linear_weight(f"{prefix}.self_attn.q_proj.weight"), mesh_device=mesh_device, dtype=weight_dtype
                    ),
                    wk=_to_device(
                        linear_weight(f"{prefix}.self_attn.k_proj.weight"), mesh_device=mesh_device, dtype=weight_dtype
                    ),
                    wv=_to_device(
                        linear_weight(f"{prefix}.self_attn.v_proj.weight"), mesh_device=mesh_device, dtype=weight_dtype
                    ),
                    wo=_to_device(
                        linear_weight(f"{prefix}.self_attn.o_proj.weight"), mesh_device=mesh_device, dtype=weight_dtype
                    ),
                    q_norm_weight=head_norm_weight(f"{prefix}.self_attn.q_norm.weight"),
                    k_norm_weight=head_norm_weight(f"{prefix}.self_attn.k_norm.weight"),
                    mlp=_DFlashMLP(
                        gate=_to_device(
                            linear_weight(f"{prefix}.mlp.gate_proj.weight"), mesh_device=mesh_device, dtype=weight_dtype
                        ),
                        up=_to_device(
                            linear_weight(f"{prefix}.mlp.up_proj.weight"), mesh_device=mesh_device, dtype=weight_dtype
                        ),
                        down=_to_device(
                            linear_weight(f"{prefix}.mlp.down_proj.weight"), mesh_device=mesh_device, dtype=weight_dtype
                        ),
                        activation_dtype=activation_dtype,
                    ),
                    activation_dtype=activation_dtype,
                )
            )

        return cls(
            config=config,
            mesh_device=mesh_device,
            encoder_fc=_to_device(linear_weight("encoder.fc.weight"), mesh_device=mesh_device, dtype=weight_dtype),
            encoder_norm=plain_norm("encoder.output_norm_enc.weight", config.hidden_size),
            layers=layers,
            final_norm=plain_norm("norm.weight", config.hidden_size),
            activation_dtype=activation_dtype,
        )

    def project_context(self, context_hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """``fc`` then ``output_norm_enc``: [1,1,T,5*H] -> [1,1,T,H]. Runs once per forward."""
        projected = ttnn.linear(
            context_hidden_states, self.encoder_fc, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = self.encoder_norm(projected)
        ttnn.deallocate(projected)
        return out

    def forward_cached(
        self,
        noise_embeds: ttnn.Tensor,
        new_context_hidden_states: ttnn.Tensor | None,
        *,
        context_positions: torch.Tensor,
        noise_positions: torch.Tensor,
        cache: DFlashDrafterCache,
    ) -> ttnn.Tensor:
        """One drafting step, reusing cached context K/V.

        Only the *newly accepted* context rows are passed; everything older is
        already in ``cache``.  This is what makes a drafting step cheap: without it
        each iteration would re-project the whole context through
        ``encoder.fc`` (33280 -> 6656), which alone costs more than the five
        decoder layers.

        Args:
            noise_embeds: ``[1, 1, block_size, hidden]``.
            new_context_hidden_states: ``[1, 1, num_new, 5 * hidden]`` or ``None``.
            context_positions: absolute positions of the new context rows.
            noise_positions: absolute positions of the ``block_size`` window slots.
            cache: mutated in place.
        """
        config = self.config
        block = int(noise_embeds.shape[2])
        context_len = 0 if new_context_hidden_states is None else int(new_context_hidden_states.shape[2])
        if context_len != int(context_positions.numel()):
            raise ValueError(f"context has {context_len} rows but {context_positions.numel()} positions were given")
        if int(noise_positions.numel()) != block:
            raise ValueError(f"noise_positions has {noise_positions.numel()} entries for a block of {block}")

        context = self.project_context(new_context_hidden_states) if context_len else None
        cache.note_positions(context_positions)
        kv_positions = torch.cat([cache.positions, noise_positions])

        def upload(tensor: torch.Tensor, seq_len: int) -> ttnn.Tensor:
            return _to_device(
                tensor.reshape(1, 1, seq_len, config.head_dim).to(torch.bfloat16),
                mesh_device=self.mesh_device,
                dtype=ttnn.bfloat16,
            )

        rope_ctx = None
        if context_len:
            cos, sin = rope_tables(context_positions, config.head_dim, config.rope_theta)
            rope_ctx = (upload(cos, context_len), upload(sin, context_len))
        cos_w, sin_w = rope_tables(noise_positions, config.head_dim, config.rope_theta)
        rope_win = (upload(cos_w, block), upload(sin_w, block))

        mask = _to_device(
            bidirectional_sliding_mask(noise_positions, kv_positions, config.sliding_window, torch.bfloat16),
            mesh_device=self.mesh_device,
            dtype=ttnn.bfloat16,
        )

        hidden = noise_embeds
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer.forward_cached(
                hidden,
                context=context,
                context_len=context_len,
                rope_ctx=rope_ctx,
                rope_win=rope_win,
                mask=mask,
                cache=cache,
                layer_idx=layer_idx,
            )

        if context is not None:
            ttnn.deallocate(context)
        for tensor in (*(rope_ctx or ()), *rope_win, mask):
            ttnn.deallocate(tensor)
        return self.final_norm(hidden)

    def forward(
        self,
        noise_embeds: ttnn.Tensor,
        context_hidden_states: ttnn.Tensor,
        *,
        position_ids: torch.Tensor,
    ) -> ttnn.Tensor:
        """Run the drafter.

        Args:
            noise_embeds: ``[1, 1, block_size, hidden]`` - the anchor token's
                embedding followed by ``block_size - 1`` mask-token embeddings,
                looked up in the *target's* table.
            context_hidden_states: ``[1, 1, T, 5 * hidden]`` - target hidden
                states at ``target_layer_ids``, concatenated on the last dim.
            position_ids: absolute positions of length ``T + block_size``.  Query
                positions are the last ``block_size`` entries, exactly as HF's
                ``apply_rotary_pos_emb`` slices them.

        Returns:
            ``[1, 1, block_size, hidden]`` - feed to the target's ``lm_head``,
            then drop position 0 to get ``block_size - 1`` candidate tokens.
        """
        config = self.config
        block = int(noise_embeds.shape[2])
        context_len = int(context_hidden_states.shape[2])
        if position_ids.numel() != context_len + block:
            raise ValueError(
                f"position_ids has {position_ids.numel()} entries; expected context {context_len} + block {block}"
            )

        cos_kv, sin_kv = rope_tables(position_ids, config.head_dim, config.rope_theta)
        # HF applies the *last* q_len rows of cos/sin to the queries.
        cos_q, sin_q = cos_kv[-block:], sin_kv[-block:]

        def upload(tensor: torch.Tensor, seq_len: int) -> ttnn.Tensor:
            return _to_device(
                tensor.reshape(1, 1, seq_len, config.head_dim).to(torch.bfloat16),
                mesh_device=self.mesh_device,
                dtype=ttnn.bfloat16,
            )

        rope_q = (upload(cos_q, block), upload(sin_q, block))
        rope_kv = (upload(cos_kv, context_len + block), upload(sin_kv, context_len + block))

        mask_torch = bidirectional_sliding_mask(
            position_ids[-block:], position_ids, config.sliding_window, torch.bfloat16
        )
        mask = _to_device(mask_torch, mesh_device=self.mesh_device, dtype=ttnn.bfloat16)

        context = self.project_context(context_hidden_states)

        hidden = noise_embeds
        for layer in self.layers:
            hidden = layer(hidden, context=context, rope_q=rope_q, rope_kv=rope_kv, mask=mask)

        ttnn.deallocate(context)
        for tensor in (*rope_q, *rope_kv, mask):
            ttnn.deallocate(tensor)
        return self.final_norm(hidden)
