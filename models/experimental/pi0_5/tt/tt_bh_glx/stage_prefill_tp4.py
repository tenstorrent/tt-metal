# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VLM prefill: TP=4 on a 4-chip mesh.

All 18 Gemma-2B blocks run sequentially on the same 4-chip submesh.
Only the MLP is tensor-parallel; attention and norms run replicated:

  gate_proj, up_proj  column-parallel (sharded along mlp_dim output axis)
  down_proj           row-parallel    (sharded along mlp_dim input axis)
  AllReduce after down_proj to sum partial outputs across chips

Attention, norms, and residuals are identical on all 4 chips (replicated weights,
replicated activations).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig, Pi0_5ModelConfig
from models.experimental.pi0_5.tt.ttnn_common import get_ln_weight_memory_config
from models.experimental.pi0_5.tt.ttnn_gemma import (
    GemmaAttentionTTNN,
    build_matmul_pcfg,
    build_sharded_norm_pcfg,
    precompute_freqs_cis_meta_format,
    rms_norm_ttnn,
)

from . import stages

_TP = 4  # tensor-parallel degree


# ─────────────────────────── Weight loading ───────────────────────────────────


def _load_block_weights_tp4(
    weights: Dict[str, torch.Tensor],
    layer_idx: int,
    mesh_device,
) -> Dict[str, ttnn.Tensor]:
    """Load one VLM block's weights onto a TP=4 mesh.

    Attention / norms  → ReplicateTensorToMesh (same on every chip).
    gate_proj, up_proj → ShardTensorToMesh(dim=-1): column-parallel on mlp_dim.
    down_proj          → ShardTensorToMesh(dim=0): row-parallel on mlp_dim.
    """
    prefix = f"model.layers.{layer_idx}."
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    ln_mc = get_ln_weight_memory_config()
    block_w: Dict[str, ttnn.Tensor] = {}

    # Fused QKV — replicated across all chips
    q_key = f"{prefix}self_attn.q_proj.weight"
    k_key = f"{prefix}self_attn.k_proj.weight"
    v_key = f"{prefix}self_attn.v_proj.weight"
    if q_key in weights:
        # Fuse Q/K/V on the host (torch.cat) and upload once, instead of 3 uploads
        # + an on-device ttnn.concat. q/k/v output dims (2048/256/256) are all
        # tile-aligned, so bf8 per-tile quantization is identical pre/post concat.
        wqkv_torch = torch.cat([weights[q_key].T, weights[k_key].T, weights[v_key].T], dim=-1).contiguous()
        block_w["self_attn.wqkv"] = ttnn.from_torch(
            wqkv_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicate,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    for key, val in weights.items():
        if not key.startswith(prefix):
            continue
        new_key = key[len(prefix) :]
        # Individual Q/K/V handled above via fused path
        if new_key in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"):
            continue

        is_norm = "layernorm" in new_key or new_key.endswith("norm.weight")

        if new_key == "mlp.gate_proj.weight":
            # Column-parallel: weight [mlp_dim, hidden] → .T → [hidden, mlp_dim]; shard last dim
            block_w[new_key] = ttnn.from_torch(
                val.T.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )
        elif new_key == "mlp.up_proj.weight":
            block_w[new_key] = ttnn.from_torch(
                val.T.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )
        elif new_key == "mlp.down_proj.weight":
            # Row-parallel: weight [hidden, mlp_dim] → .T → [mlp_dim, hidden]; shard first dim
            block_w[new_key] = ttnn.from_torch(
                val.T.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )
        elif is_norm:
            # Gemma +1 offset pre-applied; TILE_LAYOUT matches tensor_1d_to_2d_ttnn
            block_w[new_key] = ttnn.from_torch(
                (val + 1.0).reshape(1, -1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=replicate,
                memory_config=ln_mc,
            )
        else:
            # o_proj and any remaining weights — replicated
            v = val.T.contiguous() if "weight" in new_key else val
            block_w[new_key] = ttnn.from_torch(
                v,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=replicate,
            )

    return block_w


# ─────────────────────────── TP=4 MLP ─────────────────────────────────────────


class GemmaMLPTP4:
    """GeGLU MLP with TP=4 column+row parallelism and AllReduce.

    Each chip holds mlp_dim/_TP columns of gate/up and mlp_dim/_TP rows of down.
    AllReduce after down_proj sums the partial hidden-size outputs across chips.
    Input and output activations are replicated on all chips.
    """

    def __init__(self, config: GemmaConfig, weights: Dict[str, ttnn.Tensor], mesh_device):
        self.config = config
        self.mesh_device = mesh_device
        self.hidden_size = config.width
        self.intermediate_size = config.mlp_dim
        # Tensor-parallel degree = mesh device count (4 for 1x4, 8 for 1x8, ...).
        self.tp = mesh_device.get_num_devices()

        # Weights already sharded/replicated during _load_block_weights_tp4
        self.gate_proj = weights["mlp.gate_proj.weight"]  # [hidden, mlp_dim/_TP] per chip
        self.up_proj = weights["mlp.up_proj.weight"]  # [hidden, mlp_dim/_TP] per chip
        self.down_proj = weights["mlp.down_proj.weight"]  # [mlp_dim/_TP, hidden] per chip

        # Grid from the mesh device (uniform across chips)
        g = mesh_device.compute_with_storage_grid_size()
        self.grid_size = (g.x, g.y)
        self.core_grid = ttnn.CoreGrid(y=g.y, x=g.x)
        num_cores = g.x * g.y

        _user = os.environ.get("PI0_VLM_CHUNK_SIZE", "").strip()
        if _user and _user.isdigit():
            self.chunk_size = int(_user)
        elif num_cores >= 100:
            # 1024 = single chunk for the production seq=1024 prefix. The TP=4 MLP
            # intermediate is 1/_TP-width (4096, not 16384), so it fits L1 in one
            # chunk — denser matmuls + one all_reduce/layer (vs the 768 default
            # inherited from the full-width single-device MLP). ~17.9→15.1 ms/chip.
            self.chunk_size = 1024
        else:
            self.chunk_size = 256

        if self.grid_size[0] >= 12 and self.grid_size[1] >= 10:
            self._pcfg_grid = (12, 10)
        elif self.grid_size[0] >= 12:
            self._pcfg_grid = (12, 8)
        else:
            self._pcfg_grid = (8, 8)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x is replicated on all chips, shape [B, seq, hidden] or [B, 1, seq, hidden]."""
        batch = x.shape[0]
        was_3d = len(x.shape) == 3
        if was_3d:
            x = ttnn.reshape(x, [batch, 1, x.shape[1], x.shape[2]])

        seq = x.shape[2]
        hid = x.shape[3]
        local_mlp = self.intermediate_size // self.tp  # mlp_dim per chip (4096 @TP=4, 2048 @TP=8)

        num_chunks = (seq + self.chunk_size - 1) // self.chunk_size
        out_chunks: List[ttnn.Tensor] = []

        for ci in range(num_chunks):
            start = ci * self.chunk_size
            end = min(start + self.chunk_size, seq)
            actual = end - start
            padded = ((actual + 31) // 32) * 32
            needs_pad = padded > actual

            xc = ttnn.slice(x, [0, 0, start, 0], [batch, 1, end, hid])
            if needs_pad:
                xc = ttnn.to_memory_config(xc, ttnn.DRAM_MEMORY_CONFIG)
                xc = ttnn.pad(xc, padding=((0, 0), (0, 0), (0, padded - actual), (0, 0)), value=0.0)

            m = padded // 32
            k_in = self.hidden_size // 32  # 2048/32 = 64
            n_local = local_mlp // 32  # 4096/32 = 128  (was 512 single-chip)
            k_local = local_mlp // 32  # 4096/32 = 128  (was 512 single-chip)
            n_out = self.hidden_size // 32  # 2048/32 = 64

            gate_pcfg = build_matmul_pcfg(m, k_in, n_local, *self._pcfg_grid, activation=(ttnn.UnaryOpType.GELU, True))
            up_pcfg = build_matmul_pcfg(m, k_in, n_local, *self._pcfg_grid)
            down_pcfg = build_matmul_pcfg(m, k_local, n_out, *self._pcfg_grid)

            common = dict(dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

            if gate_pcfg is not None:
                gate = ttnn.linear(xc, self.gate_proj, program_config=gate_pcfg, **common)
            else:
                gate = ttnn.linear(xc, self.gate_proj, core_grid=self.core_grid, activation="gelu", **common)

            if up_pcfg is not None:
                up = ttnn.linear(xc, self.up_proj, program_config=up_pcfg, **common)
            else:
                up = ttnn.linear(xc, self.up_proj, core_grid=self.core_grid, **common)
            ttnn.deallocate(xc)

            # Local GeGLU: [B, 1, padded, local_mlp] per chip
            hidden = ttnn.multiply(gate, up, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)

            # Row-parallel down → partial sum [B, 1, padded, hidden] per chip
            if down_pcfg is not None:
                partial = ttnn.linear(hidden, self.down_proj, program_config=down_pcfg, **common)
            else:
                partial = ttnn.linear(hidden, self.down_proj, core_grid=self.core_grid, **common)
            ttnn.deallocate(hidden)

            # AllReduce: sum partial outputs across all 4 chips → replicated [B, 1, padded, hidden].
            # all_reduce already lowers to reduce_scatter + all_gather internally on the 1xTP line;
            # num_links=2 (the hardware max — 2 eth channels per hop) ~1.67x's the collective.
            oc = ttnn.all_reduce(partial, num_links=2, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(partial)

            if needs_pad:
                oc = ttnn.slice(oc, [0, 0, 0, 0], [batch, 1, actual, hid])
            out_chunks.append(oc)

        if len(out_chunks) == 1:
            out = out_chunks[0]
        else:
            out = out_chunks[0]
            for i in range(1, len(out_chunks)):
                out = ttnn.concat([out, out_chunks[i]], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(out_chunks[i])

        if was_3d:
            out = ttnn.reshape(out, [batch, seq, hid])
        return out


# ──────────────────── TP=4 Gemma block ────────────────────────────────────────


class _GemmaBlockTP4:
    """Gemma-2B transformer block with TP=4 MLP on a 4-chip mesh.

    Attention and norms: identical to TP=1 (replicated on all chips).
    MLP: GemmaMLPTP4 (column/row parallel + AllReduce).
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
        mesh_device,
        cos_meta: Optional[ttnn.Tensor] = None,
        sin_meta: Optional[ttnn.Tensor] = None,
    ):
        self.config = config
        self.device = mesh_device
        self.layer_idx = layer_idx

        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]

        self.attention = GemmaAttentionTTNN(config, weights, layer_idx, mesh_device, cos_meta, sin_meta)
        self.mlp = GemmaMLPTP4(config, weights, mesh_device)

        self._rms_norm_sharded_pcfg = None
        self._rms_norm_sharded_memcfg = None
        self._rms_norm_sharded_m_padded = 0

    def _get_sharded_norm(self, m_padded: int):
        if self._rms_norm_sharded_pcfg is None or self._rms_norm_sharded_m_padded != m_padded:
            m_tiles = m_padded // 32
            hidden_tiles = self.config.width // 32
            disable_small_m = (
                os.environ.get("PI0_LN_INTERLEAVED_SMALL_M", "").lower() in ("1", "true", "yes", "on") and m_tiles == 1
            )
            if not disable_small_m:
                norm_cfg = build_sharded_norm_pcfg(
                    m_tiles, hidden_tiles, max_grid_x=8, max_grid_y=min(8, max(1, m_tiles))
                )
                if norm_cfg is not None:
                    pc, memcfg_factory, _ = norm_cfg
                    self._rms_norm_sharded_pcfg = pc
                    self._rms_norm_sharded_memcfg = memcfg_factory(1, m_padded, m_padded, self.config.width)
                    self._rms_norm_sharded_m_padded = m_padded
        return self._rms_norm_sharded_pcfg, self._rms_norm_sharded_memcfg

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        cos_override: Optional[ttnn.Tensor] = None,
        sin_override: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Tuple[ttnn.Tensor, ttnn.Tensor]]:
        m_padded = hidden_states.shape[1] if len(hidden_states.shape) == 3 else hidden_states.shape[2]
        sh_pc, sh_mc = self._get_sharded_norm(m_padded)

        normed = rms_norm_ttnn(hidden_states, self.input_layernorm_weight, self.config.rms_norm_eps, sh_pc, sh_mc)
        if sh_pc is not None:
            normed = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)

        attn_out, new_cache = self.attention.forward(
            normed, cos_override, sin_override, attention_mask, position_ids, None, use_cache=True
        )
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        normed = rms_norm_ttnn(
            hidden_states, self.post_attention_layernorm_weight, self.config.rms_norm_eps, sh_pc, sh_mc
        )
        if sh_pc is not None:
            normed = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)

        mlp_out = self.mlp.forward(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, mlp_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(mlp_out)

        return hidden_states, new_cache


# ────────────────────────── Prefill stage ─────────────────────────────────────


class StagePrefillTP4:
    """TP=4 VLM prefill: 18 Gemma-2B blocks on a 4-chip mesh.

    Usage:
        with open_galaxy_mesh(...) as h:
            # carve a 1x4 prefill submesh however fits the hardware
            prefill_mesh = h.parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
            stage = StagePrefillTP4(config, weights, prefill_mesh)

        prefix_on_mesh = ttnn.from_torch(
            prefix_embs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=prefill_mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(prefill_mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden, per_layer_kv = stage.run(prefix_on_mesh)
    """

    def __init__(self, config: Pi0_5ModelConfig, weights: Dict[str, dict], mesh_device):
        if config.vlm_config.depth != stages.VLM_TOTAL_LAYERS:
            raise RuntimeError(f"VLM depth {config.vlm_config.depth} != expected {stages.VLM_TOTAL_LAYERS}")

        self.config = config
        self.mesh_device = mesh_device

        vw = weights["vlm_language"]

        cos_meta, sin_meta = precompute_freqs_cis_meta_format(
            config.vlm_config.head_dim, config.max_seq_len, mesh_device
        )
        # Keep handles so run() can pre-slice the RoPE tables ONCE to seq_len and
        # share across all 18 blocks (instead of each block re-slicing cos/sin).
        self.cos_meta, self.sin_meta = cos_meta, sin_meta
        self.head_dim = config.vlm_config.head_dim

        self.blocks: List[_GemmaBlockTP4] = []
        for i in range(stages.VLM_TOTAL_LAYERS):
            bw = _load_block_weights_tp4(vw, i, mesh_device)
            self.blocks.append(_GemmaBlockTP4(config.vlm_config, bw, i, mesh_device, cos_meta, sin_meta))

        # Final VLM RMS norm — replicated
        self.vlm_norm = ttnn.from_torch(
            (vw["model.norm.weight"] + 1.0).reshape(1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=get_ln_weight_memory_config(),
        )
        self.vlm_norm_eps = config.vlm_config.rms_norm_eps

    def run(
        self,
        prefix_embs: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        per_chip_attn_mask: Optional[List[ttnn.Tensor]] = None,
        per_chip_cos: Optional[List[ttnn.Tensor]] = None,
        per_chip_sin: Optional[List[ttnn.Tensor]] = None,
    ) -> Tuple[ttnn.Tensor, List[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """prefix_embs must be a ttnn.Tensor replicated on the 4-chip mesh.

        Returns (final_hidden_on_mesh, [(K_i, V_i)]_i=0..17).
        All output tensors are replicated across the 4 chips.
        """
        hidden = prefix_embs
        per_layer_kv: List[Tuple[ttnn.Tensor, ttnn.Tensor]] = []

        # Pre-slice the RoPE cos/sin tables to seq_len ONCE and reuse across all
        # blocks — avoids the per-block slice(cos_meta)/slice(sin_meta) (2 ops ×
        # 18 layers). Only when the caller didn't already supply per-chip tables.
        cos_ov = sin_ov = None
        if per_chip_cos is None:
            seq = prefix_embs.shape[-2]
            cos_ov = ttnn.slice(self.cos_meta, [0, 0, 0, 0], [1, 1, seq, self.head_dim])
            sin_ov = ttnn.slice(self.sin_meta, [0, 0, 0, 0], [1, 1, seq, self.head_dim])

        for i, block in enumerate(self.blocks):
            m_i = per_chip_attn_mask[i] if per_chip_attn_mask is not None else attention_mask
            c_i = per_chip_cos[i] if per_chip_cos is not None else cos_ov
            s_i = per_chip_sin[i] if per_chip_sin is not None else sin_ov
            hidden, new_kv = block.forward(hidden, m_i, position_ids, c_i, s_i)
            per_layer_kv.append(new_kv)

        if cos_ov is not None:
            ttnn.deallocate(cos_ov)
            ttnn.deallocate(sin_ov)

        hidden = rms_norm_ttnn(hidden, self.vlm_norm, self.vlm_norm_eps)
        return hidden, per_layer_kv
