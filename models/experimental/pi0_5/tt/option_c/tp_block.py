# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TP=2 tensor-parallel Gemma transformer block for a (2,1) Blackhole sub-mesh.

Adapted from `option_b/tp_block.py` (TP=8 on a 4x2 sub-mesh). The
sharding plan is structurally identical — only the TP factor differs.

Per-chip arithmetic at TP=2 (VLM, single layer, bf8):
    weight load = 110 MB / 2 = **55 MB / chip** = 0.46 MB / L1 bank
    < 0.70 MB CB threshold → clears OPEN_ISSUE_MLP_CB_CLASH.

`in0_block_w` defaults are inherited from Option B's TP=8 path (8/4) —
the OPTION_C_TP_WITHIN_STAGE_PLAN.md flags retuning as TBD because per-chip
widths are 4× larger here. Left as-is for the first cut; revisit if a
forward fails on a shape-mismatch error rather than the CB clash.

Sharding plan (mirrors option_b TP=8, with `tp_size` parametric):

| Tensor                 | Pattern      | Per-chip shape         |
|------------------------|--------------|------------------------|
| q_proj.weight [W,H*D]  | col-parallel | [W, H*D/tp_size]       |
| kv_proj  [W,2*KV*D]    | replicated   | [W, 2*KV*D]            |
| o_proj.weight [H*D,W]  | row-parallel | [H*D/tp_size, W]       |
| gate_proj [W,M]        | col-parallel | [W, M/tp_size]         |
| up_proj   [W,M]        | col-parallel | [W, M/tp_size]         |
| down_proj [M,W]        | row-parallel | [M/tp_size, W]         |

After o_proj and down_proj, run `ttnn.all_reduce` to sum partial outputs
across the sub-mesh. K/V are replicated because num_kv_heads=1 doesn't
split into 2 either (and Q at 16 heads sharded TP=2 → 8 heads/chip).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.tt.ttnn_gemma import (
    build_matmul_pcfg,
    build_sharded_norm_pcfg,
)


# ---------------------------------------------------------------------- #
# Weight upload helpers                                                  #
# ---------------------------------------------------------------------- #


def _shard_along(
    t: torch.Tensor,
    submesh,
    dim: int,
    dtype,
    layout=ttnn.TILE_LAYOUT,
    memory_config=None,
) -> "ttnn.Tensor":
    """Shard `t` along tensor `dim` across the 2 chips of the (2,1) sub-mesh.

    Default `memory_config=DRAM`. The L1 migration helper
    (`tt/option_c/_l1_migration.py`) walks every uploaded tensor after init
    and migrates them to L1 — see OPTION_C_TP_WITHIN_STAGE_PLAN.md.
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    mapper = ttnn.shard_tensor_to_mesh_mapper(submesh, dim)
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=submesh,
        mesh_mapper=mapper,
        memory_config=memory_config,
    )


def _replicate(
    t: torch.Tensor,
    submesh,
    dtype,
    layout=ttnn.TILE_LAYOUT,
    memory_config=None,
) -> "ttnn.Tensor":
    """Replicate `t` to every chip in the sub-mesh."""
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    mapper = ttnn.replicate_tensor_to_mesh_mapper(submesh)
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=submesh,
        mesh_mapper=mapper,
        memory_config=memory_config,
    )


# ---------------------------------------------------------------------- #
# Block                                                                  #
# ---------------------------------------------------------------------- #


class Pi0_5OptionCSubmeshTPGemmaBlock:
    """One TP=N Gemma transformer block on an (N,1) sub-mesh (Option C prefill).

    Default tp_size=2 for the (2,1) col-pair carving of Option C's (6,3)
    prefill submesh — see `mesh_setup.create_tp_submeshes_2x1`.
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        submesh,
        cos_meta: "ttnn.Tensor",
        sin_meta: "ttnn.Tensor",
        tp_size: int = 2,
    ):
        if config.num_heads % tp_size != 0:
            raise ValueError(f"num_heads={config.num_heads} not divisible by tp_size={tp_size}")
        if submesh.get_num_devices() != tp_size:
            raise ValueError(f"submesh has {submesh.get_num_devices()} devices but tp_size={tp_size}")
        self.config = config
        self.layer_idx = layer_idx
        self.submesh = submesh
        self.cos_meta = cos_meta
        self.sin_meta = sin_meta
        self.tp_size = tp_size

        self.num_heads_per_chip = config.num_heads // tp_size
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.W = config.width

        # Per-chip storage-grid for matmul pcfg lookups.
        gsz = submesh.compute_with_storage_grid_size()
        self.grid_x = gsz.x
        self.grid_y = gsz.y

        # --- Linears -----------------------------------------------------
        prefix = f"model.layers.{layer_idx}."
        self.q_proj = _shard_along(
            weights[f"{prefix}self_attn.q_proj.weight"].T.contiguous(),
            submesh,
            dim=-1,
            dtype=ttnn.bfloat8_b,
        )
        kv_w = torch.cat(
            [
                weights[f"{prefix}self_attn.k_proj.weight"].T.contiguous(),
                weights[f"{prefix}self_attn.v_proj.weight"].T.contiguous(),
            ],
            dim=-1,
        ).contiguous()
        self.kv_proj = _replicate(kv_w, submesh, dtype=ttnn.bfloat8_b)

        self.o_proj = _shard_along(
            weights[f"{prefix}self_attn.o_proj.weight"].T.contiguous(),
            submesh,
            dim=0,
            dtype=ttnn.bfloat8_b,
        )
        self.gate_proj = _shard_along(
            weights[f"{prefix}mlp.gate_proj.weight"].T.contiguous(),
            submesh,
            dim=-1,
            dtype=ttnn.bfloat8_b,
        )
        self.up_proj = _shard_along(
            weights[f"{prefix}mlp.up_proj.weight"].T.contiguous(),
            submesh,
            dim=-1,
            dtype=ttnn.bfloat8_b,
        )
        self.down_proj = _shard_along(
            weights[f"{prefix}mlp.down_proj.weight"].T.contiguous(),
            submesh,
            dim=0,
            dtype=ttnn.bfloat8_b,
        )

        self.input_layernorm = _replicate(
            (weights[f"{prefix}input_layernorm.weight"] + 1.0).view(1, -1).contiguous(),
            submesh,
            dtype=ttnn.bfloat16,
        )
        self.post_attention_layernorm = _replicate(
            (weights[f"{prefix}post_attention_layernorm.weight"] + 1.0).view(1, -1).contiguous(),
            submesh,
            dtype=ttnn.bfloat16,
        )

        # Cached per-chip matmul pcfgs (keyed by shape).
        self._matmul_pcfgs: Dict[Tuple[int, int, int, int], object] = {}
        # Cached sharded RMSNorm pcfg + memcfg (keyed by m_padded).
        self._norm_pcfg = None
        self._norm_memcfg = None
        self._norm_m_padded = 0

        # all_reduce output → DRAM, same dodge Option B uses for the
        # all_reduce kernel's static CB region.
        self._allreduce_kwargs = dict(
            num_links=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------------ #
    # Helpers: pcfg cache + sharded norm                                  #
    # ------------------------------------------------------------------ #

    def _matmul_pcfg(self, m_tiles: int, k_tiles: int, n_tiles: int, in0_block_w=None):
        key = (m_tiles, k_tiles, n_tiles, in0_block_w or 0)
        cached = self._matmul_pcfgs.get(key)
        if cached is not None:
            return cached
        pcfg = build_matmul_pcfg(m_tiles, k_tiles, n_tiles, self.grid_x, self.grid_y, in0_block_w=in0_block_w)
        self._matmul_pcfgs[key] = pcfg
        return pcfg

    def _ensure_norm_pcfg(self, m_padded: int):
        if self._norm_m_padded == m_padded:
            return
        m_tiles = m_padded // 32
        hidden_tiles = self.W // 32
        cfg = build_sharded_norm_pcfg(
            m_tiles, hidden_tiles, max_grid_x=self.grid_x, max_grid_y=min(self.grid_y, max(1, m_tiles))
        )
        if cfg is not None:
            pc, memcfg_factory, _grid = cfg
            self._norm_pcfg = pc
            self._norm_memcfg = memcfg_factory(1, m_padded, m_padded, self.W)
        else:
            self._norm_pcfg = None
            self._norm_memcfg = None
        self._norm_m_padded = m_padded

    def _rms_norm(self, x: "ttnn.Tensor", weight: "ttnn.Tensor") -> "ttnn.Tensor":
        m_padded = x.shape[1] if len(x.shape) == 3 else x.shape[2]
        self._ensure_norm_pcfg(m_padded)
        if self._norm_pcfg is not None and self._norm_memcfg is not None:
            x_sh = ttnn.to_memory_config(x, self._norm_memcfg)
            out_sh = ttnn.rms_norm(
                x_sh,
                weight=weight,
                epsilon=self.config.rms_norm_eps,
                program_config=self._norm_pcfg,
                memory_config=self._norm_memcfg,
            )
            if x_sh is not x:
                ttnn.deallocate(x_sh)
            out = ttnn.to_memory_config(out_sh, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(out_sh)
            return out
        return ttnn.rms_norm(
            x,
            weight=weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _all_reduce(self, t: "ttnn.Tensor") -> "ttnn.Tensor":
        return ttnn.all_reduce(t, cluster_axis=None, **self._allreduce_kwargs)

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        use_cache: bool = False,
    ) -> Tuple["ttnn.Tensor", Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """One transformer block in TP=tp_size."""
        B = hidden_states.shape[0]
        S = hidden_states.shape[1] if len(hidden_states.shape) == 3 else hidden_states.shape[2]
        nq = self.num_heads_per_chip
        nkv = self.config.num_kv_heads
        D = self.head_dim
        m_tiles = (S + 31) // 32
        k_tiles_attn = self.W // 32

        x = self._rms_norm(hidden_states, self.input_layernorm)

        # `in0_block_w` retained from Option B TP=8 (8/4). At TP=2 per-chip
        # widths are larger so this may need retuning — flagged as TBD.
        # 2026-06-03: experimentally verified that dropping attn block_w 8→4 does
        # NOT shrink the CB region (some other kernel — probably rms_norm or
        # all_reduce — reserves the same ~473 KB / bank). Tuning matmul block_w
        # alone won't unlock the Q+O+MLP full-L1 path; mlp_only stays the limit.
        q_pcfg = self._matmul_pcfg(m_tiles, k_tiles_attn, (nq * D) // 32, in0_block_w=8)
        kv_pcfg = self._matmul_pcfg(m_tiles, k_tiles_attn, (2 * nkv * D) // 32, in0_block_w=8)
        q = ttnn.linear(
            x,
            self.q_proj,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=q_pcfg,
        )
        kv = ttnn.linear(
            x,
            self.kv_proj,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=kv_pcfg,
        )
        ttnn.deallocate(x)
        kv_dim = nkv * D
        k = kv[:, :, 0:kv_dim]
        v = kv[:, :, kv_dim : 2 * kv_dim]
        ttnn.deallocate(kv)

        q4 = ttnn.reshape(q, (B, S, nq, D))
        k4 = ttnn.reshape(k, (B, S, nkv, D))
        v4 = ttnn.reshape(v, (B, S, nkv, D))
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        q4 = ttnn.permute(q4, (0, 2, 1, 3))
        k4 = ttnn.permute(k4, (0, 2, 1, 3))
        v4 = ttnn.permute(v4, (0, 2, 1, 3))

        q4 = ttnn.experimental.rotary_embedding(q4, self.cos_meta, self.sin_meta)
        k4 = ttnn.experimental.rotary_embedding(k4, self.cos_meta, self.sin_meta)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q4,
            k4,
            v4,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scale,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        kv_cache = (k4, v4) if use_cache else None
        ttnn.deallocate(q4)
        if not use_cache:
            ttnn.deallocate(k4)
            ttnn.deallocate(v4)

        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))
        attn_out = ttnn.reshape(attn_out, (B, S, nq * D))

        o_pcfg = self._matmul_pcfg(m_tiles, (nq * D) // 32, self.W // 32, in0_block_w=8)
        o_partial = ttnn.linear(
            attn_out,
            self.o_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=o_pcfg,
        )
        ttnn.deallocate(attn_out)
        o_full = self._all_reduce(o_partial)
        ttnn.deallocate(o_partial)

        h_post_attn = ttnn.add(hidden_states, o_full)
        ttnn.deallocate(o_full)

        n = self._rms_norm(h_post_attn, self.post_attention_layernorm)

        mlp_n_tiles = (self.config.mlp_dim // self.tp_size) // 32
        gate_pcfg = self._matmul_pcfg(m_tiles, k_tiles_attn, mlp_n_tiles, in0_block_w=4)
        up_pcfg = self._matmul_pcfg(m_tiles, k_tiles_attn, mlp_n_tiles, in0_block_w=4)
        gate = ttnn.linear(
            n,
            self.gate_proj,
            activation="gelu",
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=gate_pcfg,
        )
        up = ttnn.linear(
            n,
            self.up_proj,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=up_pcfg,
        )
        ttnn.deallocate(n)
        hidden_mlp = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        down_pcfg = self._matmul_pcfg(m_tiles, mlp_n_tiles, self.W // 32, in0_block_w=4)
        mlp_partial = ttnn.linear(
            hidden_mlp,
            self.down_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=down_pcfg,
        )
        ttnn.deallocate(hidden_mlp)
        mlp_full = self._all_reduce(mlp_partial)
        ttnn.deallocate(mlp_partial)

        out = ttnn.add(h_post_attn, mlp_full)
        ttnn.deallocate(h_post_attn)
        ttnn.deallocate(mlp_full)
        return out, kv_cache
