# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the from-scratch dots.ocr TP4 prefill rebuild.

Design: standard Megatron tensor-parallelism with a *replicated* hidden
stream. The hidden state is the full ``hidden_size`` on every chip between
sublayers, so RMSNorm is a local (per-chip) exact op and the only collectives
are the two all-reduces after ``o_proj`` and ``down_proj``.
"""

import math
import os
from dataclasses import dataclass

import torch
import ttnn


def tp4_lossy_matmul_dtype() -> "ttnn.DataType":
    """Weight dtype for the matmuls that use BFP4 in the production recipe.

    Default is the production ``bfloat4_b`` (fastest). Opt-in
    ``DOTS_OCR_TP4_HI_PRECISION`` raises them to recover accuracy on low-confidence
    tokens (e.g. page running-headers that drift flips out of the greedy decode)
    at a matmul-bandwidth / speed cost. Affects gate/up (early layers), down, and
    o_proj; qkv is already BFP8. Values:
        unset / "0"        -> bfloat4_b (production, fastest)
        "1" / "bf8"        -> bfloat8_b
        "bf16"             -> bfloat16  (max accuracy diagnostic, slowest)
    """
    v = os.environ.get("DOTS_OCR_TP4_HI_PRECISION", "0").strip().lower()
    if v in {"bf16", "bfloat16"}:
        return ttnn.bfloat16
    if v in {"1", "true", "yes", "on", "bf8", "bfloat8", "bfloat8_b"}:
        return ttnn.bfloat8_b
    return ttnn.bfloat4_b


@dataclass
class DotsOCRConfig:
    """Subset of the dots.ocr text-decoder config needed for prefill."""

    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_bias: bool = True
    max_position_embeddings: int = 131072

    @classmethod
    def from_hf(cls, hf_config):
        head_dim = getattr(hf_config, "head_dim", None) or (hf_config.hidden_size // hf_config.num_attention_heads)
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is None:
            rp = getattr(hf_config, "rope_parameters", {}) or {}
            rope_theta = rp.get("rope_theta", 1000000.0)
        return cls(
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            rope_theta=float(rope_theta),
            attention_bias=getattr(hf_config, "attention_bias", True),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 131072),
        )

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


def mesh_num_devices(mesh_device) -> int:
    if mesh_device is None or not hasattr(mesh_device, "get_num_devices"):
        return 1
    return int(mesh_device.get_num_devices())


def tp_cluster_axis(mesh_device) -> int:
    """Cluster axis along which the TP shards live (the >1 dim of a 1xN/Nx1 mesh)."""
    shape = [int(x) for x in mesh_device.shape]
    # For a (1, N) mesh the device dim is axis 1; for (N, 1) it is axis 0.
    return 0 if (len(shape) == 2 and shape[0] > 1 and shape[1] == 1) else 1


def tp_degree(mesh_device) -> int:
    """Devices along the TP axis (1 for DP-only meshes like ``(2, 1)``)."""
    if not hasattr(mesh_device, "shape"):
        return 1
    shape = [int(x) for x in mesh_device.shape]
    if len(shape) != 2:
        return 1
    return int(shape[1]) if int(shape[1]) > 1 else 1


def all_reduce(t: ttnn.Tensor, mesh_device, num_links: int = 1, output_memory_config=None) -> ttnn.Tensor:
    """All-reduce a [.., N] full-width partial-sum tensor across the TP axis.

    Implemented as reduce_scatter + all_gather (the proven 1x4-ring pattern on
    this host). ``output_memory_config`` selects where the gathered result lands
    (default DRAM; pass L1 for decode so the replicated hidden stays resident).
    Returns a replicated full-N tensor.

    Note: a single fused ``ttnn.all_reduce`` (Ring) was tried for the decode path
    and was speed-neutral (decode is not CCL-bandwidth-bound at M=1), so the
    bandwidth-efficient RS+AG is kept for both prefill and decode.
    """
    if tp_degree(mesh_device) <= 1:
        return t
    out_mc = output_memory_config or ttnn.DRAM_MEMORY_CONFIG
    orig_shape = list(t.shape)
    work = t
    if len(work.shape) < 4:
        work = ttnn.reshape(work, [1] * (4 - len(work.shape)) + orig_shape)
    axis = tp_cluster_axis(mesh_device)
    scattered = ttnn.reduce_scatter(
        work,
        dim=3,
        num_links=num_links,
        cluster_axis=axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )
    gathered = ttnn.all_gather(
        scattered,
        dim=3,
        num_links=num_links,
        cluster_axis=axis,
        memory_config=out_mc,
        topology=ttnn.Topology.Linear,
    )
    ttnn.deallocate(scattered)
    if len(orig_shape) < 4:
        gathered = ttnn.reshape(gathered, orig_shape[:-1] + [int(gathered.shape[-1])])
    return gathered


def matmul_m_dim(x) -> int:
    """Folded M dimension (product of all dims but the last) of a tensor."""
    s = x.shape
    m = 1
    for i in range(len(s) - 1):
        m *= int(s[i])
    return m


def _largest_divisor_leq(n: int, cap: int) -> int:
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def prefill_matmul_2d_config(mesh_device, m: int, k: int, n: int, fp32_dest: bool = False):
    """Tuned 2D-mcast matmul program config for a per-chip prefill matmul
    [M,K] x [K,N]. Returns None for shapes that aren't tile-aligned or for
    decode (M==32/1) so the caller falls back to the auto heuristic.

    With an interleaved (mcast) in0 the 2D kernel only needs gy | M_tiles,
    gx | N_tiles and in0_block_w | K_tiles — so it works for the awkward
    TP-sharded N/K that the auto heuristic degrades to in0_block_w=1 on.
    """
    if m % 32 or k % 32 or n % 32:
        return None
    mt, kt, nt = m // 32, k // 32, n // 32
    if mt <= 1:
        return None
    grid = mesh_device.compute_with_storage_grid_size()
    gy = _largest_divisor_leq(mt, grid.y)
    gx = _largest_divisor_leq(nt, grid.x)
    per_core_m = mt // gy
    per_core_n = nt // gx
    # in0_block_w: largest divisor of K-tiles in [2, 8] for good reuse.
    in0_block_w = 1
    for d in (8, 7, 6, 5, 4, 3, 2):
        if kt % d == 0:
            in0_block_w = d
            break
    if in0_block_w == 1:
        in0_block_w = kt  # K-tiles prime; do the whole K in one block
    # L1 budget guard: when M_tiles has no good divisor <= grid_y (e.g. a prime
    # tile count like 89 -> gy=1, per_core_M=89), the per-core circular buffers
    # blow past L1. Estimate the CB footprint and fall back to the auto heuristic
    # (correct, just not tuned) rather than crash.
    TILE_BYTES = 32 * 32 * 2  # bf16 tile; conservative upper bound
    in0_cb = per_core_m * in0_block_w * TILE_BYTES * 2  # double-buffered
    in1_cb = in0_block_w * per_core_n * TILE_BYTES * 2
    out_cb = per_core_m * per_core_n * TILE_BYTES
    if in0_cb + in1_cb + out_cb > 1_300_000:  # headroom under the 1.5MB L1
        return None

    dst = 4 if fp32_dest else 8
    out_subblock_w = _largest_divisor_leq(per_core_n, min(4, dst))
    out_subblock_h = 1
    while out_subblock_h * 2 * out_subblock_w <= dst and per_core_m % (out_subblock_h * 2) == 0:
        out_subblock_h *= 2
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def qkv_prefill_matmul_config(mesh_device, m: int, k: int, n: int):
    """2D-mcast program config for the TP4 prefill QKV matmul [M,K] x [K,N=640].

    ``prefill_matmul_2d_config`` bails for this op: M-tiles=89 (seq 2848) is prime,
    so its largest-divisor grid.y collapses to 1 -> per_core_M=89 -> the L1 guard
    trips -> None -> ttnn's auto heuristic (8x7=56 cores, in0_block_w=6, ~206 us).

    This instead spreads M across the full grid.y with ceil-padding (per_core_M=12
    for 89 tiles) and uses the sweep optimum from
    ``test_qkv_prefill_matmul_sweep`` (in0_block_w=4, out_subblock 2x3). Paired
    with an L1-interleaved in0 (the activation read is the bottleneck) this runs
    the M=2848/K=1536/N=640 op at ~148 us vs ~206 us (~1.4x). Numerics are
    unchanged -- only the tiling/scheduling differs.

    Returns None only for decode (M<=1 tile) so the caller keeps the auto path.
    NOTE: M (seq) need NOT be tile-aligned -- the real prefill seq is 2828 (not a
    multiple of 32), and the matmul pads M up to per_core_M*grid.y anyway. Requiring
    m%32==0 here (as prefill_matmul_2d_config does) returns None for the real shape,
    which -- now that in0 is L1 -- silently drops to ttnn's MatmulMultiCoreProgramConfig
    (HiFi4, ~1515us, ignores compute_kernel_config). See [[matmul-l1-in0-forces-hifi4]].
    So we ceil M to tiles and must ALWAYS return a config for prefill.
    """
    if k % 32 or n % 32:
        return None
    mt, kt, nt = math.ceil(m / 32), k // 32, n // 32
    if mt <= 1:
        return None
    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    per_core_m = math.ceil(mt / gy)
    per_core_n = math.ceil(nt / gx)
    in0_block_w = 4 if kt % 4 == 0 else _largest_divisor_leq(kt, 8)

    # Same conservative L1-CB guard as prefill_matmul_2d_config: bail (-> auto) if
    # the per-core circular buffers would not fit, rather than risk an OOM.
    TILE_BYTES = 32 * 32 * 2
    if (
        per_core_m * in0_block_w + in0_block_w * per_core_n
    ) * TILE_BYTES * 2 + per_core_m * per_core_n * TILE_BYTES > 1_300_000:
        return None

    out_subblock_w = _largest_divisor_leq(per_core_n, 4)
    out_subblock_h = 1
    while out_subblock_h * 2 * out_subblock_w <= 8 and per_core_m % (out_subblock_h * 2) == 0:
        out_subblock_h *= 2
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def to_l1(t: ttnn.Tensor) -> ttnn.Tensor:
    """Place a tensor in L1 interleaved (matmul in0 reads faster from L1)."""
    if t.memory_config().buffer_type == ttnn.BufferType.L1:
        return t
    return ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)


def all_gather_last_dim(t: ttnn.Tensor, mesh_device, num_links: int = 1) -> ttnn.Tensor:
    """All-gather a [.., N/ndev] column-sharded tensor along its last dim across
    the TP axis -> full replicated [.., N]. Used by the column-parallel LM head."""
    if tp_degree(mesh_device) <= 1:
        return t
    orig_shape = list(t.shape)
    work = t
    if len(work.shape) < 4:
        work = ttnn.reshape(work, [1] * (4 - len(work.shape)) + orig_shape)
    axis = tp_cluster_axis(mesh_device)
    gathered = ttnn.all_gather(
        work,
        dim=3,
        num_links=num_links,
        cluster_axis=axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )
    if len(orig_shape) < 4:
        gathered = ttnn.reshape(gathered, orig_shape[:-1] + [int(gathered.shape[-1])])
    return gathered


def to_replicated(torch_tensor: torch.Tensor, mesh_device, dtype, layout=ttnn.TILE_LAYOUT):
    """Send a torch tensor to device replicated on every chip."""
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_num_devices(mesh_device) > 1 else None
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def tp_mesh_mapper(mesh_device, dim: int):
    """Shard along ``dim`` on the TP axis; replicate across DP when mesh is 2D."""
    if tp_degree(mesh_device) <= 1:
        return ttnn.ReplicateTensorToMesh(mesh_device) if mesh_num_devices(mesh_device) > 1 else None
    shape = list(mesh_device.shape) if hasattr(mesh_device, "shape") else []
    if len(shape) == 2 and (shape[0] > 1 or shape[-1] == 1):
        return ttnn.ShardTensor2dMesh(mesh_device, dims=(None, dim), mesh_shape=shape)
    return ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=dim)


def shard_to_mesh(torch_tensor: torch.Tensor, mesh_device, dim: int, dtype, layout=ttnn.TILE_LAYOUT):
    """Send a torch tensor to device sharded along ``dim`` across the TP axis."""
    if mesh_num_devices(mesh_device) <= 1:
        return ttnn.from_torch(
            torch_tensor, dtype=dtype, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=tp_mesh_mapper(mesh_device, dim),
    )


def from_replicated_to_torch(t: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Read back a replicated device tensor as a single torch tensor (chip 0)."""
    if mesh_num_devices(mesh_device) <= 1:
        return ttnn.to_torch(t)
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # Replicated: every chip identical. Take the first replica slice.
    return full[: int(t.shape[0])]
