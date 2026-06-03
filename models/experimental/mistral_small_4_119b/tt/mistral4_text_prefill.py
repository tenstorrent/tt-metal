# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Small-4-119B text prefill stack — purely TTNN, no PyTorch compute fallback.

``TtMistral4TextPrefillLogits`` implements the full language-model prefill:

    input_ids  →  embed_tokens  →  [N × TtMistral4DecoderLayer]
               →  final RMSNorm  →  lm_head  →  logits (torch tensor)

Each ``TtMistral4DecoderLayer`` contains:
    input_layernorm → TtMistral4Attention (MLA)
    residual add
    post_attention_layernorm → TtMistral4MoELayer (128 experts + shared)
    residual add

Supported call signature:
    model(input_ids, position_ids=..., position_embeddings=(cos, sin), mode="prefill")

``position_embeddings`` is the ``(cos, sin)`` tuple returned by HF's
``Mistral4RotaryEmbedding``, both of shape [1, 1, seq_len, qk_rope_head_dim=64].
If the tensors arrive as [1, seq_len, D] they are unsqueezed to [1, 1, seq_len, D].
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_small_4_119b.constants import (
    HIDDEN_SIZE,
    NORM_EPS,
    QK_ROPE_HEAD_DIM,
    TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_moe import TtMistral4MoELayer
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import (
    TtMistral4Attention,
    _load_norm_weight,
)


def _rms_norm(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    compute_kernel_config,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.rms_norm(
        x,
        weight=weight,
        epsilon=NORM_EPS,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )


# ── Sharded RMSNorm for the full-hidden hot path ───────────────────────────
# Default ttnn.rms_norm on an L1-interleaved [1,1,seq,HIDDEN_SIZE] tensor lands
# on a single core (~57 μs for HIDDEN_SIZE=4096). Width-sharding across 32 cores
# brings each norm to ~5 μs (+ ~1 μs each for the shard/unshard pair).
# `ttnn.rms_norm` auto-builds the sharded program config from `input.shard_spec()`,
# but we hand it the matching one explicitly so the program is cached.
_NORM_TILE = 32
_NORM_CORE_GRID_X = 8
_NORM_CORE_GRID_Y = 4  # 8 × 4 = 32 cores
_NORM_NUM_CORES = _NORM_CORE_GRID_X * _NORM_CORE_GRID_Y
assert (
    HIDDEN_SIZE % _NORM_NUM_CORES == 0
), f"HIDDEN_SIZE ({HIDDEN_SIZE}) must divide num_cores ({_NORM_NUM_CORES}) for width-sharded RMSNorm"
_NORM_SHARD_WIDTH = HIDDEN_SIZE // _NORM_NUM_CORES  # 128 cols/core
assert (
    _NORM_SHARD_WIDTH % _NORM_TILE == 0
), f"shard width ({_NORM_SHARD_WIDTH}) must be a multiple of tile width ({_NORM_TILE})"

# Configs cached by m_tiles so each unique seq_len build runs once.
_NORM_CFG_CACHE: dict = {}


def _norm_sharded_configs(m_tiles: int):
    """Return (input_memcfg, program_cfg) for a width-sharded RMSNorm at this M.

    m_tiles is the height of the input in tiles (= ceil(seq_len / 32)).
    """
    cached = _NORM_CFG_CACHE.get(m_tiles)
    if cached is not None:
        return cached
    input_memcfg = ttnn.create_sharded_memory_config(
        shape=(m_tiles * _NORM_TILE, _NORM_SHARD_WIDTH),
        core_grid=ttnn.CoreGrid(x=_NORM_CORE_GRID_X, y=_NORM_CORE_GRID_Y),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(_NORM_CORE_GRID_X, _NORM_CORE_GRID_Y),
        subblock_w=_NORM_SHARD_WIDTH // _NORM_TILE,  # 4 tiles
        block_h=m_tiles,
        block_w=_NORM_SHARD_WIDTH // _NORM_TILE,  # 4 tiles
        inplace=False,
    )
    cfg = (input_memcfg, program_cfg)
    _NORM_CFG_CACHE[m_tiles] = cfg
    return cfg


# Decode (M=1 tile) config kept as a module-level constant for existing
# decode call sites; same content as _norm_sharded_configs(1)[0].
_NORM_DECODE_PROGRAM_CFG = _norm_sharded_configs(1)[1]


def _rms_norm_sharded(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    compute_kernel_config,
) -> ttnn.Tensor:
    """Width-sharded RMSNorm for the full-hidden [..., HIDDEN_SIZE] case.

    Auto-picks the shard config from x's seq_len (x.shape[-2], rounded up to a
    tile). Returns L1-interleaved so downstream ops (residual add, attention
    QKV) don't have to know about the shard layout.
    """
    seq_len = x.shape[-2]
    m_tiles = (seq_len + _NORM_TILE - 1) // _NORM_TILE
    input_memcfg, program_cfg = _norm_sharded_configs(m_tiles)
    x_sh = ttnn.to_memory_config(x, input_memcfg)
    normed_sh = ttnn.rms_norm(
        x_sh,
        weight=weight,
        epsilon=NORM_EPS,
        program_config=program_cfg,
        memory_config=input_memcfg,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(x_sh)
    normed = ttnn.to_memory_config(normed_sh, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(normed_sh)
    return normed


# Backward-compat alias. Existing decode call sites pass M=1-tile-shaped inputs;
# the generalised _rms_norm_sharded handles that case identically.
_rms_norm_sharded_decode = _rms_norm_sharded


# ── Decoder layer ──────────────────────────────────────────────────────────


class TtMistral4DecoderLayer(LightweightModule):
    """Single Mistral4 decoder block (attention + MoE), prefill only."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        layer_idx: int,
        compute_kernel_config,
        cache_dir=None,
    ):
        super().__init__()
        self.compute_kernel_config = compute_kernel_config
        prefix = text_decoder_layer_state_dict_prefix(layer_idx)
        _cf = (lambda key: str(cache_dir / key)) if cache_dir is not None else (lambda _: None)

        # Norms
        self.input_norm_w = _load_norm_weight(
            state_dict,
            prefix + "input_layernorm.weight",
            HIDDEN_SIZE,
            mesh_device,
            cache_file_name=_cf(prefix + "input_layernorm.weight"),
        )
        self.post_attn_norm_w = _load_norm_weight(
            state_dict,
            prefix + "post_attention_layernorm.weight",
            HIDDEN_SIZE,
            mesh_device,
            cache_file_name=_cf(prefix + "post_attention_layernorm.weight"),
        )

        # Attention
        self.attn = TtMistral4Attention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            layer_prefix=prefix,
            compute_kernel_config=compute_kernel_config,
            cache_dir=cache_dir,
        )

        # bfloat4_b: 36 layers × 3 expert stacks × 64 MB = 6.9 GB; bf16 would be 27.6 GB (OOM).
        self.moe = TtMistral4MoELayer(
            mesh_device=mesh_device,
            state_dict=state_dict,
            layer_prefix=prefix,
            expert_dtype=ttnn.bfloat4_b,
            cache_dir=cache_dir,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            x:       [1, 1, seq, HIDDEN_SIZE]  replicated on all devices
            cos/sin: [1, 1, seq, QK_ROPE_HEAD_DIM]  replicated on all devices
        Returns:
            [1, 1, seq, HIDDEN_SIZE]
        """
        # ── Attention sub-layer ─────────────────────────────────────────
        # Residual adds and rms_norm outputs in L1 so consecutive ops avoid the
        # DRAM round-trip. Output `x` of each sub-layer remains [1,1,seq,H] which
        # fits L1 interleaved at seq up to ~2048 on a P150-class grid.
        _mem = ttnn.L1_MEMORY_CONFIG
        residual = x

        normed = _rms_norm(x, self.input_norm_w, self.compute_kernel_config, _mem)
        attn_out = self.attn.forward(normed, cos, sin)
        ttnn.deallocate(normed)

        x = ttnn.add(residual, attn_out, memory_config=_mem)
        ttnn.deallocate(attn_out)

        # ── MoE sub-layer ───────────────────────────────────────────────
        residual = x
        normed = _rms_norm(x, self.post_attn_norm_w, self.compute_kernel_config, _mem)
        moe_out = self.moe.forward(normed)
        ttnn.deallocate(normed)

        x = ttnn.add(residual, moe_out, memory_config=_mem)
        ttnn.deallocate(moe_out)

        return x

    def forward_with_cache(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        kv_cache: tuple,
        chunk_start_idx: int = 0,
    ) -> ttnn.Tensor:
        """Prefill forward (one chunk) that also fills the attention KV cache in-place."""
        _mem = ttnn.L1_MEMORY_CONFIG
        residual = x
        normed = _rms_norm_sharded(x, self.input_norm_w, self.compute_kernel_config)
        attn_out = self.attn.forward(normed, cos, sin, kv_cache=kv_cache, chunk_start_idx=chunk_start_idx)
        ttnn.deallocate(normed)
        x = ttnn.add(residual, attn_out, memory_config=_mem)
        ttnn.deallocate(attn_out)

        residual = x
        normed = _rms_norm_sharded(x, self.post_attn_norm_w, self.compute_kernel_config)
        moe_out = self.moe.forward(normed)
        ttnn.deallocate(normed)
        x = ttnn.add(residual, moe_out, memory_config=_mem)
        ttnn.deallocate(moe_out)
        return x

    def forward_decode(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        kv_cache: tuple,
        current_pos: int,
        cur_pos_tensor: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """Decode one token — all activations in L1; only all_reduce crosses DRAM."""
        _mem = ttnn.L1_MEMORY_CONFIG
        residual = x
        normed = _rms_norm_sharded_decode(x, self.input_norm_w, self.compute_kernel_config)
        attn_out = self.attn.forward_decode(normed, cos, sin, kv_cache, current_pos, cur_pos_tensor)
        ttnn.deallocate(normed)
        x = ttnn.add(residual, attn_out, memory_config=_mem)
        ttnn.deallocate(attn_out)

        residual = x
        normed = _rms_norm_sharded_decode(x, self.post_attn_norm_w, self.compute_kernel_config)
        moe_out = self.moe.forward_decode(normed)
        ttnn.deallocate(normed)
        x = ttnn.add(residual, moe_out, memory_config=_mem)
        ttnn.deallocate(moe_out)
        return x


# ── Full prefill model ─────────────────────────────────────────────────────


class TtMistral4TextPrefillLogits:
    """
    Mistral-Small-4 text prefill model: embed → N decoder layers → norm → lm_head.

    All neural computation uses TTNN ops (no PyTorch fallback).
    The only host-side operations are:
      - Loading / converting weights (one-time, at construction)
      - Final ``ttnn.to_torch()`` to return logits as a torch tensor

    Args:
        mesh_device:        TTNN MeshDevice (P150x8 → [1, 8] mesh)
        state_dict:         HF checkpoint dict (filtered to required prefixes)
        text_config:        HF ``text_config`` (e.g. from AutoConfig.text_config)
        num_decoder_layers: Number of decoder layers to instantiate (1..36)
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        text_config,
        num_decoder_layers: int,
    ):
        self.mesh_device = mesh_device
        self.num_decoder_layers = num_decoder_layers

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # ── Embedding table ─────────────────────────────────────────────
        embed_w = state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY].to(torch.bfloat16)
        # [vocab_size, HIDDEN_SIZE]
        self.embed_weight = ttnn.as_tensor(
            embed_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # ── Decoder layers ──────────────────────────────────────────────
        self.decoder_layers = []
        for i in range(num_decoder_layers):
            self.decoder_layers.append(
                TtMistral4DecoderLayer(
                    mesh_device=mesh_device,
                    state_dict=state_dict,
                    layer_idx=i,
                    compute_kernel_config=self.compute_kernel_config,
                )
            )

        # ── Final norm ──────────────────────────────────────────────────
        self.final_norm_w = _load_norm_weight(state_dict, "language_model.model.norm.weight", HIDDEN_SIZE, mesh_device)

        # ── LM head ─────────────────────────────────────────────────────
        lm_head_w = state_dict["language_model.lm_head.weight"].to(torch.bfloat16)
        # HF shape: [vocab_size, HIDDEN_SIZE] → transpose for matmul: [HIDDEN_SIZE, vocab_size]
        lm_head_w_t = lm_head_w.T.contiguous()
        self.lm_head_weight = ttnn.as_tensor(
            lm_head_w_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )  # [HIDDEN_SIZE, vocab_size]

    # ── RoPE helper ────────────────────────────────────────────────────────

    def _prepare_rope(
        self,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        seq_len: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Convert HF (cos, sin) torch tensors to TTNN device tensors.

        HF returns (cos, sin) either as [1, seq, D] or [1, 1, seq, D].
        We normalise to [1, 1, seq_len, QK_ROPE_HEAD_DIM] on device.
        If the last dim > QK_ROPE_HEAD_DIM, only the first QK_ROPE_HEAD_DIM
        elements are used.
        """
        cos_t, sin_t = position_embeddings

        def _prep(t: torch.Tensor) -> ttnn.Tensor:
            t = t.to(torch.bfloat16)
            # Ensure 4D
            while t.dim() < 4:
                t = t.unsqueeze(0)
            # Trim rope dim if necessary
            if t.shape[-1] > QK_ROPE_HEAD_DIM:
                t = t[..., :QK_ROPE_HEAD_DIM].contiguous()
            # Shape: [1, 1, seq_len, QK_ROPE_HEAD_DIM]
            t = t[:, :, :seq_len, :].contiguous()
            return ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return _prep(cos_t), _prep(sin_t)

    # ── Forward ────────────────────────────────────────────────────────────

    def __call__(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mode: str = "prefill",
    ) -> torch.Tensor:
        """
        Args:
            input_ids:            [1, seq_len] long tensor on CPU
            position_ids:         [1, seq_len] long tensor on CPU (unused in TTNN path)
            position_embeddings:  (cos, sin) from HF Mistral4RotaryEmbedding
            mode:                 must be "prefill"

        Returns:
            logits: [1, seq_len, vocab_size] bfloat16 torch tensor on CPU
        """
        assert mode == "prefill", f"TtMistral4TextPrefillLogits only supports prefill, got {mode!r}"
        assert position_embeddings is not None, "position_embeddings (cos, sin) required"

        seq_len = input_ids.shape[1]

        # ── Prepare RoPE tensors ───────────────────────────────────────────
        cos_tt, sin_tt = self._prepare_rope(position_embeddings, seq_len)

        # ── Embedding ──────────────────────────────────────────────────────
        input_ids_tt = ttnn.as_tensor(
            input_ids.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )  # [1, seq_len]

        x = ttnn.embedding(
            input_ids_tt,
            self.embed_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, seq_len, HIDDEN_SIZE]
        ttnn.deallocate(input_ids_tt)

        # Reshape to [1, 1, seq_len, HIDDEN_SIZE] for decoder layers
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # ── Decoder layers ─────────────────────────────────────────────────
        for layer in self.decoder_layers:
            x = layer.forward(x, cos_tt, sin_tt)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        # ── Final norm ─────────────────────────────────────────────────────
        x = _rms_norm(x, self.final_norm_w, self.compute_kernel_config)
        # [1, 1, seq_len, HIDDEN_SIZE]

        # ── LM head ────────────────────────────────────────────────────────
        logits_tt = ttnn.linear(
            x,
            self.lm_head_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, seq_len, vocab_size]
        ttnn.deallocate(x)

        # ── Gather to host ─────────────────────────────────────────────────
        # With replicated weights both devices produce the same logits.
        # We use device 0's output (ConcatMeshToTensor concatenates along
        # dim=0; slice [0:1] selects device 0).
        logits_host = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
        )  # [num_devices, 1, seq_len, vocab_size]
        ttnn.deallocate(logits_tt)

        # Take device-0 slice and squeeze the leading 1-dim
        logits_host = logits_host[0]  # [1, seq_len, vocab_size]

        return logits_host.to(torch.bfloat16)
