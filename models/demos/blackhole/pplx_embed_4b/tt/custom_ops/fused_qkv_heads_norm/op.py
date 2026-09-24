# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Head-split + per-head Q/K RMSNorm in one model-local ``ttnn.generic_op``.

Replaces three ops per layer (``nlp_create_qkv_heads`` + ``q_norm`` + ``k_norm``)
with one pass over the fused QKV activation. At bs=32 those three are 41 + 24 +
6 ms of kernel time per iteration, each a DRAM-bound pass over the same tensors.
Reader/writer reuse the head-split layout; a compute kernel normalises each Q
and K head over head_dim and applies gamma; V is copied through. Constants
(row-replicated gamma tiles, 1/head_dim scaler, eps) come from
``constants.make_norm_constants``.
"""
import os

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads.op import (
    _TILE_BYTES,
    _core_ranges,
    _Plan,
    _split_work_to_cores,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
READER_KERNEL = os.path.join(_HERE, "kernels", "reader_qkv_heads_norm.cpp")
COMPUTE_KERNEL = os.path.join(_HERE, "kernels", "compute_qkv_heads_norm.cpp")
COMPUTE_KERNEL_V2 = os.path.join(_HERE, "kernels", "compute_qkv_heads_norm_v2.cpp")  # dest-reuse, 6 passes/head
COMPUTE_KERNEL_V3 = os.path.join(_HERE, "kernels", "compute_qkv_heads_norm_v3.cpp")  # phases batched across heads
WRITER_KERNEL = os.path.join(_HERE, "kernels", "writer_qkv_heads_norm.cpp")
_BF16_TILE = _TILE_BYTES[ttnn.bfloat16]
TILE = 32

# Resident constants: per-core L1 shards holding the layer-independent inputs a core needs (its seq tile's cos and
# sin tiles, the rotation tile, the 1/head_dim scaler and eps), built once per RoPE table and core plan and shared
# by every layer. CBs 9 / 10 / 11 / 3 / 4 alias the shard, so the reader issues no reads for them and the first
# unit reaches compute sooner. Keyed by the RoPE table buffers: prefill always starts at position 0 here, so a
# table buffer's content is fixed by its shape.
_RESIDENT_CACHE = {}


def _resident_layout(head_dim_tiles):
    """Tile offsets inside a resident shard: cos | sin | rotation | scaler | eps."""
    wt = head_dim_tiles
    return {"cos": 0, "sin": wt, "trans": 2 * wt, "scaler": 2 * wt + 1, "eps": 2 * wt + 2, "tiles": 2 * wt + 3}


def _core_seq_tiles(per_core, head_groups, q_split, seq_tiles):
    """Seq tile of each core's units, or None when some core's units span more than one seq tile."""
    out, cursor = [], 0
    for _, _, n in per_core:
        tiles = {((u // q_split) // head_groups) % seq_tiles for u in range(cursor, cursor + n)}
        if len(tiles) != 1:
            return None
        out.append(tiles.pop())
        cursor += n
    return out


def _build_resident(device, used_cores, per_core, core_seq, rot_cos, rot_sin, trans_mat, head_dim, eps):
    import torch

    # The shard of core i must land on the core that runs work list entry i: shard order follows the column-major
    # enumeration of the core set, which is how _split_work_to_cores numbers cores.
    order = [(int(c.x), int(c.y)) for c in ttnn.corerange_to_cores(used_cores, row_wise=False)]
    if order != [(cx, cy) for cx, cy, _ in per_core]:
        return None
    lay = _resident_layout(head_dim // TILE)
    cos, sin = ttnn.to_torch(rot_cos).float(), ttnn.to_torch(rot_sin).float()
    trans = ttnn.to_torch(trans_mat).float().reshape(-1, TILE)[:TILE]
    scaler = torch.full((TILE, TILE), 1.0 / head_dim)
    eps_t = torch.full((TILE, TILE), float(eps))
    rows = []
    for s in core_seq:
        r = slice(s * TILE, (s + 1) * TILE)
        rows.append(torch.cat([cos[0, 0, r, :], sin[0, 0, r, :], trans, scaler, eps_t], dim=1))
    host = torch.cat(rows, dim=0).reshape(1, 1, len(core_seq) * TILE, lay["tiles"] * TILE)
    spec = ttnn.ShardSpec(used_cores, [TILE, lay["tiles"] * TILE], ttnn.ShardOrientation.COL_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)


def nlp_create_qkv_heads_norm_headsplit(
    qkv_fused: ttnn.Tensor,
    gamma_q_tiles: ttnn.Tensor,
    gamma_k_tiles: ttnn.Tensor,
    scaler_tile: ttnn.Tensor,
    eps_tile: ttnn.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_groups: int | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    rot_cos: ttnn.Tensor | None = None,
    rot_sin: ttnn.Tensor | None = None,
    trans_mat: ttnn.Tensor | None = None,
    q_dtype: ttnn.DataType | None = None,
    kv_dtype: ttnn.DataType | None = None,
    resident: bool = False,
    norm_eps: float | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """``qkv_fused``: ``[B, 1, S, (num_heads + 2*num_kv_heads) * head_dim]`` TILE bf16/bfp8.

    Returns ``(q, k, v)`` as ``[B, H, S, head_dim]`` with Q and K RMS-normalised per head
    (and rotated when ``rot_cos``/``rot_sin``/``trans_mat`` are given). ``q_dtype`` /
    ``kv_dtype`` (default: input dtype) pick the output dtypes; the packer converts, so
    e.g. ``q_dtype=bfloat8_b`` hands SDPA its Q operand without a Typecast op.
    ``resident`` (with rotary and ``norm_eps``): take cos/sin, the rotation tile, the scaler and eps from a cached
    per-core L1 shard (see ``_RESIDENT_CACHE``) instead of reading them per call; falls back when a core's units
    span more than one seq tile. The first cache miss copies the RoPE tables to the host, so it must not happen
    inside a trace capture.
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    device = qkv_fused.device()
    fuse_rotary = rot_cos is not None
    use_v2 = os.getenv("QWEN_FUSED_COMPUTE_V2", "0") == "1"
    # v3: v1's math with every phase run over a chunk of heads (a unit's Q heads, then its K heads), so each phase's
    # reconfig / init / CB handshakes are paid once per chunk; intermediate CBs hold a chunk. Bit-identical to v1.
    use_v3 = os.getenv("QWEN_FUSED_COMPUTE_V3", "0") == "1" and not use_v2
    # cos/sin tiles depend only on the seq tile; consecutive units of a core share it across the
    # head groups, so with the v2 compute they are read once per seq tile instead of once per unit.
    cache_rot = fuse_rotary and use_v2
    if fuse_rotary and (rot_sin is None or trans_mat is None):
        raise ValueError("rot_cos, rot_sin and trans_mat must be given together")
    # Placeholder bindings when rotary is off: the kernels never touch them (compile-time flag).
    cos_t, sin_t, trans_t = (rot_cos, rot_sin, trans_mat) if fuse_rotary else (qkv_fused, qkv_fused, qkv_fused)
    plan = _Plan.from_input(qkv_fused, num_heads, num_kv_heads)
    if head_groups is None:
        _env = os.getenv("QWEN_HEADSPLIT_GROUPS_QKV")
        head_groups = int(_env) if _env else plan.num_kv_heads
    if plan.num_kv_heads % head_groups != 0:
        raise ValueError(f"num_kv_heads ({plan.num_kv_heads}) must be divisible by head_groups ({head_groups})")
    heads_per_group = plan.num_kv_heads // head_groups
    # Q split: 2 halves the unit (half the group's Q heads plus K *or* V) so small shapes fill more
    # cores. bs1: 128 units of 24 tiles -> 256 units of 12 tiles. 1 = one unit per (block, group).
    q_split = int(os.getenv("QWEN_HEADSPLIT_Q_SPLIT", "1"))
    if q_split not in (1, 2) or (heads_per_group * plan.q_heads_per_kv) % q_split != 0:
        raise ValueError(f"QWEN_HEADSPLIT_Q_SPLIT={q_split} must be 1 or 2 and divide the group's Q heads")
    if q_split != 1 and use_v2:
        raise ValueError("QWEN_FUSED_COMPUTE_V2 does not support QWEN_HEADSPLIT_Q_SPLIT != 1")

    out_dtype = qkv_fused.dtype
    q_dtype = q_dtype or out_dtype
    kv_dtype = kv_dtype or out_dtype
    if q_dtype not in _TILE_BYTES or kv_dtype not in _TILE_BYTES:
        raise ValueError(f"unsupported output dtype q={q_dtype} kv={kv_dtype}")
    separate_q = q_dtype != kv_dtype  # Q gets its own output CB (17) with its own tile size
    q_shape = (plan.batch, plan.num_q_heads, plan.seq_len, plan.head_dim)
    kv_shape = (plan.batch, plan.num_kv_heads, plan.seq_len, plan.head_dim)
    q_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(q_shape), q_dtype, ttnn.TILE_LAYOUT, device, memory_config)
    k_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(kv_shape), kv_dtype, ttnn.TILE_LAYOUT, device, memory_config)
    v_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(kv_shape), kv_dtype, ttnn.TILE_LAYOUT, device, memory_config)

    grid = device.compute_with_storage_grid_size()
    num_cores, per_core = _split_work_to_cores(plan.num_blocks_total * head_groups * q_split, int(grid.x), int(grid.y))
    if num_cores == 0:
        raise RuntimeError("nlp_create_qkv_heads_norm_headsplit: nothing to do")
    used_cores = _core_ranges(per_core)

    Wt = plan.head_dim_tiles
    res_tensor = None
    if resident and fuse_rotary and norm_eps is not None and not use_v2:
        core_seq = _core_seq_tiles(per_core, head_groups, q_split, plan.seq_tiles)
        if core_seq is not None:
            key = (
                id(device),
                rot_cos.buffer_address(),
                rot_sin.buffer_address(),
                trans_mat.buffer_address(),
                tuple(rot_cos.shape),
                plan.head_dim,
                float(norm_eps),
                tuple(per_core),
            )
            if key not in _RESIDENT_CACHE:
                _RESIDENT_CACHE[key] = _build_resident(
                    device, used_cores, per_core, core_seq, rot_cos, rot_sin, trans_mat, plan.head_dim, norm_eps
                )
            res_tensor = _RESIDENT_CACHE[key]
    res_lay = _resident_layout(Wt)
    group_q_tiles = heads_per_group * plan.q_heads_per_kv * Wt
    group_kv_tiles = heads_per_group * Wt
    sub_q_tiles = group_q_tiles // q_split
    kv_parts = 2 if q_split == 1 else 1  # K and V in one unit, or K xor V per half
    unit_tiles = sub_q_tiles + kv_parts * group_kv_tiles
    tile_size = _TILE_BYTES[out_dtype]
    out_tiles = kv_parts * group_kv_tiles if separate_q else unit_tiles
    # intermediate CBs hold one head (v1) or all of a unit's Q and K heads (v3)
    nh = (sub_q_tiles + group_kv_tiles) // Wt if use_v3 else 1

    def cb(index, tiles, dtype, tsize):
        return ttnn.CBDescriptor(
            total_size=tiles * tsize,
            core_ranges=used_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tsize)],
        )

    cbs = [
        cb(0, unit_tiles * 2, out_dtype, tile_size),  # fused QKV in (double-buffered unit)
        cb(16, out_tiles * 2, kv_dtype, _TILE_BYTES[kv_dtype]),  # normalised Q|K|V out (K|V when separate_q)
        cb(1, Wt, ttnn.bfloat16, _BF16_TILE),  # gamma_q tiles (resident)
        cb(2, Wt, ttnn.bfloat16, _BF16_TILE),  # gamma_k tiles (resident)
        cb(5, nh * Wt, ttnn.bfloat16, _BF16_TILE),  # x^2
        cb(6, nh, ttnn.bfloat16, _BF16_TILE),  # mean-square (row values in col 0)
        cb(7, nh, ttnn.bfloat16, _BF16_TILE),  # rsqrt
        cb(8, nh * Wt, ttnn.bfloat16, _BF16_TILE),  # x * inv
    ]
    if separate_q:
        cbs.append(cb(17, sub_q_tiles * 2, q_dtype, _TILE_BYTES[q_dtype]))  # Q out in its own dtype

    def aliased(index, name, tiles):
        return ttnn.cb_descriptor_from_sharded_tensor(
            index, res_tensor, address_offset=res_lay[name] * _BF16_TILE, total_size=tiles * _BF16_TILE
        )

    if res_tensor is not None:
        cbs += [aliased(3, "scaler", 1), aliased(4, "eps", 1)]
    else:
        cbs += [
            cb(3, 1, ttnn.bfloat16, _BF16_TILE),  # 1/head_dim scaler (resident)
            cb(4, 1, ttnn.bfloat16, _BF16_TILE),  # eps (resident)
        ]
    if fuse_rotary:
        if res_tensor is not None:
            cbs += [aliased(9, "cos", Wt), aliased(10, "sin", Wt), aliased(11, "trans", 1)]
        else:
            cbs += [
                cb(9, Wt * (2 if cache_rot else 1), ttnn.bfloat16, _BF16_TILE),  # cos tiles for the unit's seq tile
                cb(10, Wt * (2 if cache_rot else 1), ttnn.bfloat16, _BF16_TILE),  # sin tiles
                cb(11, 1, ttnn.bfloat16, _BF16_TILE),  # 32x32 rotation tile (resident)
            ]
        cbs += [
            cb(12, nh * Wt, ttnn.bfloat16, _BF16_TILE),  # x @ T
            cb(13, nh * Wt, ttnn.bfloat16, _BF16_TILE),  # (x @ T) * sin
            cb(14, nh * Wt, ttnn.bfloat16, _BF16_TILE),  # x * cos
            cb(15, nh * Wt, ttnn.bfloat16, _BF16_TILE),  # normalised head(s) awaiting rotary
        ]

    reader_ct = [
        plan.q_heads_per_kv,
        plan.num_kv_heads,
        Wt,
        plan.in_w_tiles,
        plan.seq_tiles,
        head_groups,
        heads_per_group,
        int(fuse_rotary),
        int(cache_rot),
        q_split,
        int(res_tensor is not None),
    ]
    for t in (qkv_fused, gamma_q_tiles, gamma_k_tiles, scaler_tile, eps_tile, cos_t, sin_t, trans_t):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    compute_ct = [
        plan.q_heads_per_kv,
        heads_per_group,
        Wt,
        int(fuse_rotary),
        int(separate_q),
        int(cache_rot),
        head_groups,
        plan.seq_tiles,
        q_split,
    ]
    writer_ct = [
        plan.seq_tiles,
        Wt,
        plan.q_out_HtWt,
        plan.num_q_heads,
        plan.num_kv_heads,
        plan.q_heads_per_kv,
        head_groups,
        heads_per_group,
        plan.seq_tiles,
        int(separate_q),
        q_split,
    ]
    for t in (q_tensor, k_tensor, v_tensor):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_rt, compute_rt, writer_rt = [], [], []
    cursor = 0
    for cx, cy, n_units in per_core:
        core = (cx, cy)
        reader_rt.append(
            (
                core,
                [
                    qkv_fused.buffer_address(),
                    gamma_q_tiles.buffer_address(),
                    gamma_k_tiles.buffer_address(),
                    scaler_tile.buffer_address(),
                    eps_tile.buffer_address(),
                    cos_t.buffer_address(),
                    sin_t.buffer_address(),
                    trans_t.buffer_address(),
                    n_units,
                    cursor,
                ],
            )
        )
        compute_rt.append((core, [n_units, cursor]))
        writer_rt.append(
            (core, [q_tensor.buffer_address(), k_tensor.buffer_address(), v_tensor.buffer_address(), n_units, cursor])
        )
        cursor += n_units

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=READER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE_KERNEL_V2 if use_v2 else (COMPUTE_KERNEL_V3 if use_v3 else COMPUTE_KERNEL),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=False,
                    # Only matters when an output is bfp8; the stock Typecast packs precise.
                    bfp8_pack_precise=os.getenv("QWEN_FUSED_BFP8_PRECISE", "1") == "1",
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WRITER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        cbs=cbs,
    )
    io = [qkv_fused, gamma_q_tiles, gamma_k_tiles, scaler_tile, eps_tile]
    if fuse_rotary:
        io += [cos_t, sin_t, trans_t]
    if res_tensor is not None:
        io.append(res_tensor)
    ttnn.generic_op(io + [q_tensor, k_tensor, v_tensor], program_descriptor)
    return q_tensor, k_tensor, v_tensor
