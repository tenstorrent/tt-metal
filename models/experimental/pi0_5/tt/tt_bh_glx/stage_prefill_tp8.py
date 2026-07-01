# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VLM prefill: TP=8 on an 8-chip mesh.

All 18 Gemma-2B blocks run sequentially on the same 8-chip submesh.
Only the MLP is tensor-parallel; attention and norms run replicated:

  gate_proj, up_proj  column-parallel (sharded along mlp_dim output axis)
  down_proj           row-parallel    (sharded along mlp_dim input axis)
  AllReduce after down_proj to sum partial outputs across chips

Attention, norms, and residuals are identical on all 8 chips (replicated weights,
replicated activations).
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch
import ttnn

# Rows of the compute grid reserved below the down-proj matmul for fused RS workers.
_RS_GRID_ROWS = 2

from models.experimental.pi0_5.common.configs import GemmaConfig, Pi0_5ModelConfig
from models.experimental.pi0_5.tt.ttnn_common import get_ln_weight_memory_config
from models.experimental.pi0_5.tt.ttnn_gemma import (
    GemmaAttentionTTNN,
    build_bs_matmul_pcfg,
    build_matmul_pcfg,
    build_sharded_norm_pcfg,
    bs_matmul_divisible,
    make_bs_memcfg,
    precompute_freqs_cis_meta_format,
    rms_norm_ttnn,
)

_TP = 4  # symbol used only in the shape comments below; the real TP degree is
#          read at runtime from mesh_device.get_num_devices() (8 in production)
VLM_TOTAL_LAYERS = 18  # VLM Gemma-2B depth (all 18 blocks run on the TP mesh)


def _all_reduce_scatter_dim(shape: tuple, tp: int) -> int:
    """Mirror all_reduce_async finding_scatter_dim for TILE tensors."""
    dims = list(shape)
    if len(dims) >= 2:
        dims[-1] //= 32
        dims[-2] //= 32
    for d in range(len(dims) - 1, -1, -1):
        if dims[d] % tp == 0:
            return d
    raise RuntimeError(f"no all_reduce scatter dim for shape={shape} tp={tp}")


def _ccl_int_env(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v.isdigit() else default


def _ccl_base(tp: int) -> dict:
    kw = {
        "num_links": _ccl_int_env("PI0_CCL_NUM_LINKS", 2),
        "memory_config": ttnn.L1_MEMORY_CONFIG,
        "num_buffers_per_channel": 2,
    }
    if tp == 8:
        kw["topology"] = ttnn.Topology.Ring
    return kw


def _rs_kwargs(tp: int) -> dict:
    """Reduce-scatter: 2 links × 2 workers/link → 12 CCL cores (2 dir × (2 workers + 1 mux))."""
    kw = _ccl_base(tp)
    kw["num_workers_per_link"] = _ccl_int_env("PI0_CCL_RS_WORKERS", 2)
    return kw


def _ag_kwargs(tp: int) -> dict:
    """All-gather: 2 links × 4 workers/link → 20 CCL cores on TP=8 ring."""
    kw = _ccl_base(tp)
    default_workers = 4 if tp >= 8 else 2
    kw["num_workers_per_link"] = _ccl_int_env("PI0_CCL_AG_WORKERS", default_workers)
    return kw


def _ccl_kwargs(tp: int) -> dict:
    """Shared kwargs (reduce_scatter path); all_gather uses _ag_kwargs."""
    return _rs_kwargs(tp)


def _tp_all_gather(x: ttnn.Tensor, dim: int, tp: int) -> ttnn.Tensor:
    """Gather scattered shards across the 1×tp mesh (second half of all_reduce)."""
    return ttnn.all_gather(x, dim, **_ag_kwargs(tp))


def _tp_all_reduce(x: ttnn.Tensor, tp: int) -> ttnn.Tensor:
    """Sum-replicate partials across the 1×tp mesh (RS + AG)."""
    dim = _all_reduce_scatter_dim(tuple(x.shape), tp)
    scattered = ttnn.reduce_scatter(x, dim, **_rs_kwargs(tp))
    out = _tp_all_gather(scattered, dim, tp)
    ttnn.deallocate(scattered)
    return out


def _mlp_fused_rs_enabled() -> bool:
    """PI0_MLP_FUSED_RS=0: unfused down_proj linear + separate reduce_scatter."""
    return os.environ.get("PI0_MLP_FUSED_RS", "1").lower() not in ("0", "false", "no", "off")


def _mlp_bs_enabled() -> bool:
    """PI0_MLP_BS=0: s2i after post-attn LN → interleaved gate/up/down matmuls."""
    return os.environ.get("PI0_MLP_BS", "1").lower() not in ("0", "false", "no", "off")


def _is_2d_mcast_pcfg(pcfg) -> bool:
    return pcfg is not None and isinstance(pcfg, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig)


# Tuned (m_tiles, k_tiles, n_tiles) → (grid_x, mm_grid_y, in0_block_w) for fused down+RS.
# mm_grid_y is the matmul row count; RS workers occupy the next _RS_GRID_ROWS rows.
# in0_block_w=16 cuts K-loop iterations vs the default 8 on these 1/TP-width shapes.
_FUSED_DOWN_PCFG: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {
    (32, 64, 64): (12, 8, 16),  # TP=8, seq chunk 1024
    (32, 128, 64): (12, 8, 16),  # TP=4, seq chunk 1024
}


def _build_fused_down_pcfg(
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    default_grid: Tuple[int, int],
) -> Optional[ttnn.MatmulMultiCoreReuseMultiCastProgramConfig]:
    tuned = _FUSED_DOWN_PCFG.get((m_tiles, k_tiles, n_tiles))
    if tuned is not None:
        grid_x, mm_grid_y, in0_block_w = tuned
    else:
        grid_x, grid_y = default_grid
        mm_grid_y = grid_y - _RS_GRID_ROWS
        in0_block_w = 16 if k_tiles >= 64 else None
    pcfg = build_matmul_pcfg(
        m_tiles,
        k_tiles,
        n_tiles,
        grid_x,
        mm_grid_y,
        dst_budget=8,
        in0_block_w=in0_block_w,
    )
    return pcfg if _is_2d_mcast_pcfg(pcfg) else None


class _ChunkPcfgs:
    __slots__ = ("gate", "up", "down", "use_fused_down", "bs_gate", "bs_up", "bs_down")

    def __init__(self, gate, up, down, use_fused_down: bool, bs_gate=None, bs_up=None, bs_down=None):
        self.gate = gate
        self.up = up
        self.down = down
        self.use_fused_down = use_fused_down
        self.bs_gate = bs_gate
        self.bs_up = bs_up
        self.bs_down = bs_down


# ─────────────────────────── Weight loading ───────────────────────────────────


def _attn_headpar_enabled() -> bool:
    """PI0_TP8_ATTN_HEADPAR=1: shard attention Q-heads across chips (K/V replicated,
    row-parallel o_proj + all_reduce) instead of replicating attention on every chip."""
    return os.environ.get("PI0_TP8_ATTN_HEADPAR", "").lower() in ("1", "true", "yes", "on")


def _load_block_weights_tp8(
    weights: Dict[str, torch.Tensor],
    layer_idx: int,
    mesh_device,
    config: Optional[GemmaConfig] = None,
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
        qT, kT, vT = weights[q_key].T, weights[k_key].T, weights[v_key].T  # each [hidden, *]
        if _attn_headpar_enabled() and config is not None:
            # Head-parallel: shard Q-heads across chips, replicate K/V. Per-chip slab =
            # [this chip's hpc Q-heads | K | V]; concat slabs then ShardTensorToMesh(dim=-1)
            # so chip i gets its slab. K/V duplicated per slab (tiny). Matches the fused
            # [Q|K|V] layout nlp_create_qkv_heads expects, with num_q=hpc, num_kv=1.
            tp = mesh_device.get_num_devices()
            nq, hd = config.num_heads, config.head_dim
            assert (
                tp in (4, 8) and nq % tp == 0
            ), f"head-parallel attn needs tp in (4,8) & num_q%tp==0 (nq={nq}, tp={tp})"
            hpc = nq // tp
            slabs = [torch.cat([qT[:, i * hpc * hd : (i + 1) * hpc * hd], kT, vT], dim=-1) for i in range(tp)]
            wqkv_torch = torch.cat(slabs, dim=-1).contiguous()
            block_w["self_attn.wqkv"] = ttnn.from_torch(
                wqkv_torch,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            # Replicated: fuse Q/K/V on host and upload once. q/k/v dims (2048/256/256)
            # tile-aligned → bf8 per-tile quantization identical pre/post concat.
            wqkv_torch = torch.cat([qT, kT, vT], dim=-1).contiguous()
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
        elif new_key == "self_attn.o_proj.weight" and _attn_headpar_enabled() and config is not None:
            # Row-parallel: o_proj [hidden, heads*hd] → .T [heads*hd, hidden]; shard the in-dim
            # (dim=0) by head-group so chip i holds the rows for its hpc Q-heads. Matches the
            # wqkv Q-head split → matmul(concat_i, o_proj_i) = per-chip partial, summed by all_reduce.
            block_w[new_key] = ttnn.from_torch(
                val.T.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )
        else:
            # o_proj (replicated path) and any remaining weights — replicated
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


class GemmaMLPTP8:
    """GeGLU MLP with TP=4 column+row parallelism and AllReduce.

    Each chip holds mlp_dim/_TP columns of gate/up and mlp_dim/_TP rows of down.
    down_proj matmul is fused with reduce_scatter (matmul_reduce_scatter_async);
    all_gather completes the AllReduce. Gate/up stay unfused column-parallel matmuls.
    Input and output activations are replicated on all chips.
    """

    def __init__(self, config: GemmaConfig, weights: Dict[str, ttnn.Tensor], mesh_device):
        self.config = config
        self.mesh_device = mesh_device
        self.hidden_size = config.width
        self.intermediate_size = config.mlp_dim
        # Tensor-parallel degree = mesh device count (4 for 1x4, 8 for 1x8, ...).
        self.tp = mesh_device.get_num_devices()

        # Weights already sharded/replicated during _load_block_weights_tp8
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

        self._fused_rs = self.tp > 1 and _mlp_fused_rs_enabled()
        self._rs_buffer_cache: Dict[Tuple[int, int], Tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        self._pcfg_by_m: Dict[Tuple[int, Optional[Tuple[int, int]]], _ChunkPcfgs] = {}
        self._local_mlp = self.intermediate_size // self.tp
        self._bs_gate_up_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._rs_semaphores: Optional[List[ttnn.Tensor]] = None
        self._rs_barrier_semaphore = None
        self._rs_core_offset: Optional[ttnn.CoreCoord] = None
        self._compute_kernel_config = None
        self._k_in_tiles = self.hidden_size // 32
        self._k_local_tiles = self.intermediate_size // self.tp // 32
        self._n_out_tiles = self.hidden_size // 32
        self._max_padded = ((self.chunk_size + 31) // 32) * 32
        import os as _os_bf4

        self._down_dtype = (
            ttnn.bfloat4_b
            if _os_bf4.environ.get("PI0_DOWN_BF4", "").lower() in ("1", "true", "yes", "on")
            else ttnn.bfloat8_b
        )
        if self._fused_rs:
            self._init_fused_rs()
            self._get_fused_rs_buffers(1, self._max_padded, self._down_dtype)
        self._get_chunk_pcfs(self._max_padded // 32, bs_grid=None)

    def _get_chunk_pcfs(self, m_tiles: int, bs_grid: Optional[Tuple[int, int]] = None) -> _ChunkPcfgs:
        key = (m_tiles, bs_grid)
        cached = self._pcfg_by_m.get(key)
        if cached is not None:
            return cached
        gate = build_matmul_pcfg(
            m_tiles, self._k_in_tiles, self._k_local_tiles, *self._pcfg_grid, activation=(ttnn.UnaryOpType.GELU, True)
        )
        up = build_matmul_pcfg(m_tiles, self._k_in_tiles, self._k_local_tiles, *self._pcfg_grid)
        bs_gate = bs_up = bs_down = None
        if bs_grid is not None:
            gx, gy = bs_grid
            if bs_matmul_divisible(m_tiles, self._k_in_tiles, self._k_local_tiles, gx, gy):
                bs_gate = build_bs_matmul_pcfg(
                    m_tiles,
                    self._k_in_tiles,
                    self._k_local_tiles,
                    gx,
                    gy,
                    activation=(ttnn.UnaryOpType.GELU, True),
                )
                bs_up = build_bs_matmul_pcfg(m_tiles, self._k_in_tiles, self._k_local_tiles, gx, gy)
            if bs_matmul_divisible(m_tiles, self._k_local_tiles, self._n_out_tiles, gx, gy):
                bs_down = build_bs_matmul_pcfg(m_tiles, self._k_local_tiles, self._n_out_tiles, gx, gy)
        use_fused_down = False
        down = build_matmul_pcfg(m_tiles, self._k_local_tiles, self._n_out_tiles, *self._pcfg_grid)
        if self._fused_rs and bs_down is None:
            fused_down = _build_fused_down_pcfg(m_tiles, self._k_local_tiles, self._n_out_tiles, self._pcfg_grid)
            if fused_down is not None:
                down = fused_down
                use_fused_down = True
        cached = _ChunkPcfgs(gate, up, down, use_fused_down, bs_gate, bs_up, bs_down)
        self._pcfg_by_m[key] = cached
        return cached

    def _init_fused_rs(self) -> None:
        """Semaphores + core offset for matmul_reduce_scatter_async on down_proj."""
        g = self.mesh_device.compute_with_storage_grid_size()
        ccl_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        self._rs_semaphores = [ttnn.create_global_semaphore(self.mesh_device, ccl_cores, 0) for _ in range(3)]
        self._rs_barrier_semaphore = ttnn.create_global_semaphore(self.mesh_device, ccl_cores, 0)
        tuned = _FUSED_DOWN_PCFG.get((self._max_padded // 32, self._k_local_tiles, self._n_out_tiles))
        mm_grid_y = tuned[1] if tuned is not None else self._pcfg_grid[1] - _RS_GRID_ROWS
        self._rs_core_offset = ttnn.CoreCoord(0, mm_grid_y)
        # Match unfused ttnn.linear defaults (bf8 matmul); fp32 dest would halve subblock budget.
        self._compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _get_fused_rs_buffers(self, batch: int, padded: int, dtype) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        key = (batch, padded)
        cached = self._rs_buffer_cache.get(key)
        if cached is not None:
            return cached
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        mem = ttnn.L1_MEMORY_CONFIG
        hid = self.hidden_size
        inter = ttnn.from_torch(
            torch.zeros(batch, 1, padded, hid),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=replicate,
            memory_config=mem,
        )
        out = ttnn.from_torch(
            torch.zeros(batch, 1, padded, hid // self.tp),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=replicate,
            memory_config=mem,
        )
        self._rs_buffer_cache[key] = (inter, out)
        return inter, out

    def _down_proj_fused_rs(
        self,
        hidden: ttnn.Tensor,
        down_pcfg,
        down_dtype,
        batch: int,
        padded: int,
    ) -> ttnn.Tensor:
        """Row-parallel down_proj + reduce_scatter in one fused op, then all_gather."""
        ccl = _ccl_kwargs(self.tp)
        topology = ccl.get("topology", ttnn.Topology.Ring)
        persistent_intermediate, persistent_output = self._get_fused_rs_buffers(batch, padded, down_dtype)
        mm_out, scattered = ttnn.experimental.matmul_reduce_scatter_async(
            hidden,
            self.down_proj,
            persistent_intermediate,
            persistent_output,
            3,
            self._rs_semaphores,
            self._rs_core_offset,
            barrier_semaphore=self._rs_barrier_semaphore,
            num_links=ccl["num_links"],
            memory_config_rs=ccl["memory_config"],
            intermediate_memory_config_rs=ccl["memory_config"],
            memory_config_mm=ttnn.L1_MEMORY_CONFIG,
            topology=topology,
            dtype=down_dtype,
            program_config=down_pcfg,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(hidden)
        ttnn.deallocate(mm_out)
        oc = _tp_all_gather(scattered, 3, self.tp)
        ttnn.deallocate(scattered)
        return oc

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        bs_norm_factory=None,
        bs_grid: Optional[Tuple[int, int]] = None,
    ) -> ttnn.Tensor:
        """x is replicated on all chips, shape [B, seq, hidden] or [B, 1, seq, hidden].

        When bs_norm_factory + bs_grid are set, gate/up/down matmuls and GeGLU multiply
        stay block-sharded on the LN grid; sharded_to_interleaved runs only before all_reduce.
        """
        use_bs = bs_norm_factory is not None and bs_grid is not None
        batch = x.shape[0]
        was_3d = len(x.shape) == 3
        if was_3d:
            x = ttnn.reshape(x, [batch, 1, x.shape[1], x.shape[2]])

        seq = x.shape[2]
        hid = x.shape[3]
        gx = gy = None
        if use_bs:
            gx, gy = bs_grid

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
                if use_bs:
                    bs_mc_hidden = bs_norm_factory(batch, actual, padded, hid)
                    xc = ttnn.pad(xc, padding=((0, 0), (0, 0), (0, padded - actual), (0, 0)), value=0.0)
                    xc = ttnn.to_memory_config(xc, bs_mc_hidden)
                else:
                    xc = ttnn.to_memory_config(xc, ttnn.DRAM_MEMORY_CONFIG)
                    xc = ttnn.pad(xc, padding=((0, 0), (0, 0), (0, padded - actual), (0, 0)), value=0.0)
            elif use_bs:
                bs_mc_hidden = bs_norm_factory(batch, actual, padded, hid)
                xc = ttnn.to_memory_config(xc, bs_mc_hidden)

            m = padded // 32
            m_tiles = (batch * padded) // 32
            pcfs = self._get_chunk_pcfs(m_tiles, bs_grid if use_bs else None)
            gate_pcfg, up_pcfg, down_pcfg, use_fused_down = pcfs.gate, pcfs.up, pcfs.down, pcfs.use_fused_down
            use_bs_chunk = use_bs and pcfs.bs_gate is not None and pcfs.bs_up is not None and pcfs.bs_down is not None
            if use_bs and not use_bs_chunk:
                xc = ttnn.sharded_to_interleaved(xc, memory_config=ttnn.L1_MEMORY_CONFIG)

            common = dict(dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
            if use_bs_chunk:
                bs_mc_inter = make_bs_memcfg(batch, padded, self._local_mlp, gx, gy)
                bs_common = dict(
                    dtype=ttnn.bfloat8_b,
                    memory_config=bs_mc_inter,
                    compute_kernel_config=self._bs_gate_up_compute_config,
                )
                gate = ttnn.linear(xc, self.gate_proj, program_config=pcfs.bs_gate, **bs_common)
                up = ttnn.linear(xc, self.up_proj, program_config=pcfs.bs_up, **bs_common)
            else:
                if gate_pcfg is not None:
                    gate = ttnn.linear(xc, self.gate_proj, program_config=gate_pcfg, **common)
                else:
                    gate = ttnn.linear(xc, self.gate_proj, core_grid=self.core_grid, activation="gelu", **common)

                if up_pcfg is not None:
                    up = ttnn.linear(xc, self.up_proj, program_config=up_pcfg, **common)
                else:
                    up = ttnn.linear(xc, self.up_proj, core_grid=self.core_grid, **common)
            ttnn.deallocate(xc)

            if use_bs_chunk:
                hidden = ttnn.multiply(gate, up, memory_config=bs_mc_inter)
            else:
                hidden = ttnn.multiply(gate, up, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)

            if use_bs_chunk:
                down_common = dict(
                    dtype=self._down_dtype,
                    memory_config=bs_mc_hidden,
                    compute_kernel_config=self._bs_gate_up_compute_config,
                )
                partial = ttnn.linear(hidden, self.down_proj, program_config=pcfs.bs_down, **down_common)
                ttnn.deallocate(hidden)
                partial = ttnn.sharded_to_interleaved(partial, memory_config=ttnn.L1_MEMORY_CONFIG)
                oc = _tp_all_reduce(partial, self.tp)
                ttnn.deallocate(partial)
            elif use_fused_down:
                oc = self._down_proj_fused_rs(hidden, down_pcfg, self._down_dtype, batch, padded)
            else:
                down_common = dict(dtype=self._down_dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
                if down_pcfg is not None:
                    partial = ttnn.linear(hidden, self.down_proj, program_config=down_pcfg, **down_common)
                else:
                    partial = ttnn.linear(hidden, self.down_proj, core_grid=self.core_grid, **down_common)
                ttnn.deallocate(hidden)
                oc = _tp_all_reduce(partial, self.tp)
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


class _GemmaAttentionTP8(GemmaAttentionTTNN):
    """Head-parallel attention: each chip owns num_q_heads//tp Q-heads (K/V replicated).

    Thin reuse of GemmaAttentionTTNN: build it with num_heads = num_q_heads//tp so the
    inherited forward (fused-QKV → create_heads → RoPE → SDPA → concat_heads → o_proj)
    runs per-chip on this chip's heads. Expects head-sharded wqkv + row-parallel o_proj in
    `weights`. forward() therefore returns a per-chip PARTIAL [.,seq,hidden]; the caller
    (_GemmaBlockTP8) all_reduces it. head_dim/scale are unchanged (per-head).
    """

    def __init__(self, config, weights, layer_idx, mesh_device, cos_meta=None, sin_meta=None):
        tp = mesh_device.get_num_devices()
        hpc = config.num_heads // tp
        super().__init__(replace(config, num_heads=hpc), weights, layer_idx, mesh_device, cos_meta, sin_meta)


class _GemmaBlockTP8:
    """Gemma-2B transformer block with TP=4 MLP on a 4-chip mesh.

    MLP: GemmaMLPTP8 (column/row parallel + AllReduce).
    Attention: replicated by default; head-parallel + AllReduce when PI0_TP8_ATTN_HEADPAR=1.
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

        self._attn_headpar = _attn_headpar_enabled()
        attn_cls = _GemmaAttentionTP8 if self._attn_headpar else GemmaAttentionTTNN
        self.attention = attn_cls(config, weights, layer_idx, mesh_device, cos_meta, sin_meta)
        self.mlp = GemmaMLPTP8(config, weights, mesh_device)

        self._rms_norm_sharded_pcfg = None
        self._rms_norm_sharded_memcfg = None
        self._rms_norm_sharded_m_padded = 0
        self._norm_memcfg_factory = None
        self._norm_bs_grid: Optional[Tuple[int, int]] = None

    def _get_sharded_norm(self, m_padded: int):
        if self._rms_norm_sharded_pcfg is None or self._rms_norm_sharded_m_padded != m_padded:
            m_tiles = m_padded // 32
            hidden_tiles = self.config.width // 32
            disable_small_m = (
                os.environ.get("PI0_LN_INTERLEAVED_SMALL_M", "").lower() in ("1", "true", "yes", "on") and m_tiles == 1
            )
            self._norm_memcfg_factory = None
            self._norm_bs_grid = None
            if not disable_small_m:
                norm_cfg = build_sharded_norm_pcfg(
                    m_tiles, hidden_tiles, max_grid_x=8, max_grid_y=min(8, max(1, m_tiles))
                )
                if norm_cfg is not None:
                    pc, memcfg_factory, grid = norm_cfg
                    self._rms_norm_sharded_pcfg = pc
                    self._rms_norm_sharded_memcfg = memcfg_factory(1, m_padded, m_padded, self.config.width)
                    self._rms_norm_sharded_m_padded = m_padded
                    self._norm_memcfg_factory = memcfg_factory
                    self._norm_bs_grid = (grid.x, grid.y)
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
        # Keep block-sharded for fused QKV matmul (same grid as sharded LN).
        # if sh_pc is not None:
        #     normed = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)

        attn_out, new_cache = self.attention.forward(
            normed,
            cos_override,
            sin_override,
            attention_mask,
            position_ids,
            None,
            use_cache=True,
            bs_norm_factory=self._norm_memcfg_factory if sh_pc is not None else None,
            bs_grid=self._norm_bs_grid if sh_pc is not None else None,
        )
        ttnn.deallocate(normed)
        if self._attn_headpar:
            # Head-parallel: attn_out is a per-chip partial (this chip's Q-heads' contribution
            # to o_proj). Sum across chips → replicated, same as the MLP all_reduce.
            attn_out = _tp_all_reduce(attn_out, self.device.get_num_devices())
        # EXPERIMENT (PI0_RESID_BF8): carry the residual stream in bf8 — halves the
        # residual-add bytes, but accumulates quantization over 18 layers (PCC risk).
        import os as _os_resid

        _resid_bf8 = _os_resid.environ.get("PI0_RESID_BF8", "").lower() in ("1", "true", "yes", "on")
        _add_kw = dict(memory_config=ttnn.L1_MEMORY_CONFIG)
        if _resid_bf8:
            _add_kw["dtype"] = ttnn.bfloat8_b
        hidden_states = ttnn.add(hidden_states, attn_out, **_add_kw)
        ttnn.deallocate(attn_out)

        normed = rms_norm_ttnn(
            hidden_states, self.post_attention_layernorm_weight, self.config.rms_norm_eps, sh_pc, sh_mc
        )
        use_mlp_bs = sh_pc is not None and _mlp_bs_enabled()
        if sh_pc is not None and not use_mlp_bs:
            normed = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)

        mlp_out = self.mlp.forward(
            normed,
            bs_norm_factory=self._norm_memcfg_factory if use_mlp_bs else None,
            bs_grid=self._norm_bs_grid if use_mlp_bs else None,
        )
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, mlp_out, **_add_kw)
        ttnn.deallocate(mlp_out)

        return hidden_states, new_cache


# ────────────────────────── Prefill stage ─────────────────────────────────────


class StagePrefillTP8:
    """TP=4 VLM prefill: 18 Gemma-2B blocks on a 4-chip mesh.

    Usage:
        with open_prefill_tp8_mesh(tp=8, l1_small_size=24576) as prefill_mesh:
            stage = StagePrefillTP8(config, weights, prefill_mesh)

        prefix_on_mesh = ttnn.from_torch(
            prefix_embs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=prefill_mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(prefill_mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden, per_layer_kv = stage.run(prefix_on_mesh)
    """

    def __init__(self, config: Pi0_5ModelConfig, weights: Dict[str, dict], mesh_device):
        if config.vlm_config.depth != VLM_TOTAL_LAYERS:
            raise RuntimeError(f"VLM depth {config.vlm_config.depth} != expected {VLM_TOTAL_LAYERS}")

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

        self.blocks: List[_GemmaBlockTP8] = []
        for i in range(VLM_TOTAL_LAYERS):
            bw = _load_block_weights_tp8(vw, i, mesh_device, config.vlm_config)
            self.blocks.append(_GemmaBlockTP8(config.vlm_config, bw, i, mesh_device, cos_meta, sin_meta))

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
