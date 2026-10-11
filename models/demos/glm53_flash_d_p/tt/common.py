# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared device helpers for GLM-5.3-Flash: compute config, replicated upload / readback (harness boundary only)."""

import os

import torch

import ttnn


def hifi4_config(fp32_acc: bool = True, fidelity=None, l1_acc: bool = False):
    """HiFi4 + fp32 dest by default; ``fidelity`` overrides the math fidelity (see attn_fidelity). l1_acc: packer L1
    accumulation, for matmuls (mm_config)."""
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fidelity or ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=l1_acc,
    )


def mm_config(cfg):
    """A matmul's copy of ``cfg`` with packer L1 accumulation (GLM_MM_L1ACC, default on): same results (fp32 reference,
    tests/test_matmul_tune.py GLM_MMT_TRUTH=1), ttnn.linear 1.2..2.4x faster (MLA o_proj 640x16384x4096 1.86 -> 0.76 ms).
    """
    import os

    if os.environ.get("GLM_MM_L1ACC", "1") == "0":
        return cfg
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=cfg.math_fidelity,
        math_approx_mode=cfg.math_approx_mode,
        fp32_dest_acc_en=cfg.fp32_dest_acc_en,
        packer_l1_acc=True,
    )


def env_fidelity(name: str, default: str = "HiFi4"):
    """A ttnn.MathFidelity from the environment (LoFi, HiFi2, HiFi3, HiFi4)."""
    import os

    return getattr(ttnn.MathFidelity, os.environ.get(name, default))


def attn_fidelity():
    """Math fidelity of the attention matmuls (MLA, sparse SDPA, indexer, q_a, KDA): GLM_ATTN_FIDELITY, default HiFi4."""
    return env_fidelity("GLM_ATTN_FIDELITY")


def replicate(mesh, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    """Host tensor -> replicated device tensor (load time or the harness boundary, never inside a forward)."""
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def replicated_to_host(t: ttnn.Tensor) -> torch.Tensor:
    """Replicated device tensor -> host (chip 0's copy; harness boundary only)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


# ---- residual layout (GLM_RESIDUAL_LAYOUT): "split" (default) or "replicated"
# split: chip d = 2 r + c (row-major over the 2x2 mesh) holds rows [d S/4, (d + 1) S/4) of every per-token tensor
# between the attention / FFN modules (the indexer / MLA query split); replicated: every chip holds all S rows.
LAYOUTS = ("split", "replicated")
MC = ttnn.DRAM_MEMORY_CONFIG


def residual_layout() -> str:
    import os

    mode = os.environ.get("GLM_RESIDUAL_LAYOUT", "split")
    assert mode in LAYOUTS, f"GLM_RESIDUAL_LAYOUT={mode!r}, want one of {LAYOUTS}"
    return mode


def local_rows(t: ttnn.Tensor, dim: int = -2) -> ttnn.Tensor:
    """Replicated rows -> this chip's split quarter (no CCL): mesh_partition on axis 0, then axis 1."""
    a = ttnn.mesh_partition(t, dim=dim, cluster_axis=0, memory_config=MC)
    b = ttnn.mesh_partition(a, dim=dim, cluster_axis=1, memory_config=MC)
    ttnn.deallocate(a)
    return b


# row gathers on MiMo's fabric_all_gather (GLM_FABRIC_GATHER=1, default): bit-identical to ttnn.all_gather and faster
# at every model shape (tests/test_all_gather_bench.py: 640 x 4096 axis 1 281 -> 208 us, 640 x 1536 132 -> 88,
# 2560 x 4096 axis 0 340 -> 289); a fresh output per call, the op's own cached semaphores. Rows that are not a whole
# number of tiles keep ttnn.all_gather. A module attribute so tests/test_ab_layers.py can flip it.
FABRIC_GATHER = os.environ.get("GLM_FABRIC_GATHER", "1") == "1"
GATHER_LINKS = int(os.environ.get("GLM_MOE_LINKS", "2"))


_TOPO2D: dict = {}


def _ensure_2d_topology(t: ttnn.Tensor) -> None:
    """fabric_all_gather validates a 2D declared distribution; split-layout tensors made by mesh_partition / the
    collectives carry a 1D one. Re-declare rows (dim 2) split over both mesh axes - what the bytes are (as MiMo's
    moe_ag.gather_full and deepseek_v3_d_p's chunked prefill do)."""
    if t.tensor_topology().distribution_shape().dims() >= 2:
        return
    mesh = t.device()
    key = tuple(mesh.shape)
    if key not in _TOPO2D:
        dist = ttnn.MeshShape(*key)
        coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist)]
        _TOPO2D[key] = ttnn.TensorTopology(dist, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
    t.update_tensor_topology(_TOPO2D[key])


def gather_axis(t: ttnn.Tensor, axis: int) -> ttnn.Tensor:
    """all_gather of the rows (dim -2) over mesh axis ``axis``."""
    shape = list(t.shape)
    if not FABRIC_GATHER or t.layout != ttnn.TILE_LAYOUT or len(shape) != 4 or shape[-2] % ttnn.TILE_SIZE:
        return ttnn.all_gather(t, dim=-2, cluster_axis=axis, memory_config=MC)
    _ensure_2d_topology(t)
    shape[-2] *= tuple(t.device().shape)[axis]
    out = ttnn.empty(shape, dtype=t.dtype, layout=ttnn.TILE_LAYOUT, device=t.device(), memory_config=MC)
    ttnn.bringup.fabric_all_gather(t, dim=2, output_tensor=out, cluster_axis=axis, num_links=GATHER_LINKS)
    return out


def gather_half(t: ttnn.Tensor) -> ttnn.Tensor:
    """Split quarter -> mesh row r's half [r S/2, (r + 1) S/2) on both of its chips (all_gather on axis 1)."""
    return gather_axis(t, 1)


def gather_rows(t: ttnn.Tensor) -> ttnn.Tensor:
    """Split quarter -> all S rows on every chip (all_gather on axis 1, then axis 0)."""
    h = gather_half(t)
    out = gather_axis(h, 0)
    ttnn.deallocate(h)
    return out


_RING_OK = {}


def _ring_scatter_ok(mesh) -> bool:
    """Whether the whole mesh closes into a snake ring of direct links (fabric_reduce_scatter plans it, or refuses)."""
    hit = _RING_OK.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        from ttnn.bringup.fabric_all_gather_ttnn.fabric_all_gather_py import plan

        try:
            links = int(__import__("os").environ.get("GLM_MOE_LINKS", "2"))
            plan(mesh, cluster_axis=None, topology=ttnn.Topology.Ring, num_links=links)
            ok = True
        except ValueError:
            ok = False
        hit = _RING_OK[id(mesh)] = (mesh, ok)
    return hit[1]


def scatter_rows(t: ttnn.Tensor) -> ttnn.Tensor:
    """Per-chip partial sums over all S rows -> the split quarter of the 4-chip sum (reduce_scatter on axis 0, then
    axis 1): half the bytes of an all_reduce, and no slice afterwards.
    GLM_SCATTER_OP=fabric_ring (default): the partials typecast to bf16, then one ttnn.bringup.fabric_reduce_scatter
    over the whole mesh as a snake ring (GLM_MOE_LINKS links; meshes whose snake does not close fall back to
    fabric_bf16); fabric_bf16: the same op on axis 0 and then axis 1; "ttnn": fp32 ttnn.reduce_scatter.
    56k prefill 9.34 -> 8.95 s; KV PCC unchanged (s4096 kv_latent 0.96653 / 0.98409), 56k top1 0.8767 -> 0.8804."""
    import os

    op = os.environ.get("GLM_SCATTER_OP", "fabric_ring")
    if op == "fabric_ring" and _ring_scatter_ok(t.device()):
        # one reduce-scatter over the whole mesh, a ring along the snake (row-major blocks: the same rows as axis 0
        # then axis 1); LoudBox 2x4, [5120, 4096] bf16 per chip, 2 links: 427.7 vs 548.6 us for the two calls
        links = int(os.environ.get("GLM_MOE_LINKS", "2"))
        tb = t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16, memory_config=MC)
        out = ttnn.bringup.fabric_reduce_scatter(tb, cluster_axis=None, topology=ttnn.Topology.Ring, num_links=links)
        if tb is not t:
            ttnn.deallocate(tb)
        return out
    if op in ("fabric_bf16", "fabric_ring"):
        links = int(os.environ.get("GLM_MOE_LINKS", "2"))
        tb = t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16, memory_config=MC)
        a = ttnn.bringup.fabric_reduce_scatter(tb, cluster_axis=0, num_links=links)
        if tb is not t:
            ttnn.deallocate(tb)
        b = ttnn.bringup.fabric_reduce_scatter(a, cluster_axis=1, num_links=links)
        ttnn.deallocate(a)
        return b
    a = ttnn.reduce_scatter(t, dim=-2, cluster_axis=0, memory_config=MC)
    b = ttnn.reduce_scatter(a, dim=-2, cluster_axis=1, memory_config=MC)
    ttnn.deallocate(a)
    return b


def split_to_host(t: ttnn.Tensor) -> torch.Tensor:
    """Split device tensor -> host rows in order (chip d's quarter is rows d S/4 ..; harness boundary only)."""
    return torch.cat([ttnn.to_torch(p) for p in ttnn.get_device_tensors(t)], dim=-2)


def split_from_host(mesh, t: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Host [1, 1, S, W] -> the split layout (harness boundary only)."""
    s, w = t.shape[-2], t.shape[-1]
    rows = tuple(mesh.shape)
    return ttnn.from_torch(
        t.reshape(rows[0], rows[1], s // (rows[0] * rows[1]), w).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=MC,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=rows),
    )
