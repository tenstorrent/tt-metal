# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The pure-python planner of ``sdpa_decode_lse``: the stock paged flash-decode program factory
(``ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp``, tt-metal 298c06426a8)
re-expressed for the one configuration Gemma 4's global decode layers use, so the core assignment, the tree
reduction, the circular buffers and every compile-time / runtime argument the copied kernels parse can be checked
on the host. No ttnn import.

The configuration (everything else is refused by ``SdpaDecodeLse.unsupported_reason``):

* paged, causal, positions from a ``cur_pos`` tensor (-1 = skip the row), one KV head per chip (MQA: the stock
  factory then takes ``num_heads_per_core = 1`` and ``num_cores_per_batch = num_cores_per_head``);
* Q ``[1, B, Hq <= 32, D]`` bf16 TILE interleaved (``PNHt = 1``), so no Q sharding, no row-major Q, no Q-head
  parallelism, no K multicast and no column-major group layout; output interleaved;
* no attention mask, no attention sink, no sliding window, no MLA, no speculative multi-position mode, an
  effective block size that is a tile multiple (no sub-tile block padding mask), a fixed K chunk (> 0).

Core assignment (the factory's linear branch, Q in DRAM): ``cores_per_head = max(1, min(grid,
max_cores_per_head_batch * B) // B)``; core ``i`` of the ``cores_per_head * B`` active cores is logical
``(i mod gx, i div gx)``, works on batch row ``i div cores_per_head`` as member ``i mod cores_per_head`` of that
row's reduction group; member 0 is the reducer, the output core and the tree root. A member's K chunks come from
``get_workload_for_core`` (the reversed split: member 0 takes the LAST chunks, so it always has data), and the
group merges its partial (m, l, O) in ``ceil(log2(cores_per_head))`` rounds of the binary tree of
``get_tree_reduction_params`` (ported below, with the orphan rule for a non-power-of-2 group).

Deviations from the factory, none of which changes a computed value:

* only the active cores carry the kernels, CBs and semaphores (the stock launches idle cores that return at
  once);
* the tree-reduction staging CB (``c_19``) holds ``rounds`` blocks, not ``cores_per_head - 1``: a block is
  addressed by the round it arrives in (``round * (out_tiles + 2 * PNHt)``), so only ``rounds`` are ever used --
  the factory's own comment calls the larger sizing legacy over-provisioning (it keeps it off the spec path);
* the writer takes one extra runtime arg (the stats address, at index 1) and one extra TensorAccessorArgs block
  (the stats tensor, after the output's).

Pair split (``plan(..., pair_split=True)``, the op's ``pair_split`` mode): every kernel is built with the
``SDPA_LSE_PAIR_SPLIT`` define and one more compile-time arg, the device's half ``h`` = its mesh column's parity
(reader / writer: after the last TensorAccessorArgs block; compute: index ``COMPUTE_CT_ARGS``). The row's ``n`` K
chunks split as half 0 = ``[0, ceil(n / 2))``, half 1 = ``[ceil(n / 2), n)``; within its half a device's
``cores_per_head`` cores take the stock reversed split (``pair_workload_for_core``, the mirror of
``kernels/pair_split.hpp``), so the core plan, the tree and every runtime arg are the same on both halves. Without
``pair_split`` the plan, the arguments and the preprocessed kernels are the stock ones above.
"""

import dataclasses
import math
import struct

TILE = 32
HALF_TILE = (16, 32)
FULL_TILE = (32, 32)
MAX_TREE_REDUCTION_ROUNDS = 6
MAX_CORES_PER_HEAD = 1 << MAX_TREE_REDUCTION_ROUNDS
MAX_Q_HEADS = 32  # PNHt == 1
HALF_TILE_MAX_Q_HEADS = 16  # the factory's use_half_tile: causal, bf16 Q, num_q_heads <= 16
NONE = 0xFFFFFFFF  # UINT32_MAX: "no child / no parent", and the "positions come from the cur_pos tensor" flag
STATS_COLS = 64  # the stats tensor's last dim: column 0 = m, column 32 = l
SEM_REDUCER, SEM_OUTPUT, SEM_K_MCAST = 0, 1, 2

# pair split: the kernels' define, and the finite statistics an empty half writes (O = 0, m = -30000, l = 1; bf16
# stores m as -29952). kernels/pair_split.hpp holds the same two packed bf16 patterns.
PAIR_SPLIT_DEFINE = ("SDPA_LSE_PAIR_SPLIT", "1")
PAIR_EMPTY_M = -30000.0
PAIR_EMPTY_L = 1.0
PAIR_EMPTY_M_PACKED = 0xC6EAC6EA  # bf16(-30000) = 0xC6EA, twice
PAIR_EMPTY_L_PACKED = 0x3F803F80  # bf16(1.0) = 0x3F80, twice

# CB indices of the stock kernels (sdpa_flash_decode.cpp / reader_decode_all.cpp / writer_decode_all.cpp)
CB_Q, CB_K, CB_V, CB_MASK, CB_SCALE, CB_M_IN, CB_L_IN = 0, 1, 2, 3, 5, 6, 7
CB_POS_WRITER, CB_PAGE_TABLE, CB_Q_RM, CB_COL_IDENTITY, CB_ZERO, CB_POS_COMPUTE = 8, 9, 10, 11, 12, 15
CB_OUT_O, CB_OUT_M, CB_OUT_L, CB_INTERMED, CB_OUT_FINAL = 16, 17, 18, 19, 20
CB_PREV_SUM_2, CB_EXP_MAX_DIFF_2, CB_OUT_ACC_2 = 21, 22, 23
CB_QK, CB_OUT_IM, CB_OUT_ACC, CB_MAX_1, CB_MAX_2, CB_SUM_1, CB_SUM_2, CB_EXP_MAX_DIFF = 24, 25, 26, 27, 28, 29, 30, 31

READER_CT_SCALARS = 36  # then 7 TensorAccessorArgs blocks: q, k, v, mask, cur_pos, page_table, sink
WRITER_CT_SCALARS = 23  # then 2 blocks: out, stats
COMPUTE_CT_ARGS = 28
READER_RT_FIXED = 18  # then output core x[B], y[B]
WRITER_RT_FIXED = 7 + 3 + MAX_TREE_REDUCTION_ROUNDS  # then group x / y, reducer x / y, output x / y
COMPUTE_RT_ARGS = 4 + 2 + MAX_TREE_REDUCTION_ROUNDS


def tree_rounds(n):
    """``ceil(log2(n))``: the rounds of a group of ``n`` cores (0 for one core)."""
    return 0 if int(n) <= 1 else (int(n) - 1).bit_length()


def _trailing_ones(v):
    t = 0
    while v & 1:
        v >>= 1
        t += 1
    return t


@dataclasses.dataclass(frozen=True)
class TreeParams:
    """``TreeReductionParams`` of one group member (``sdpa_decode_device_operation.hpp``)."""

    is_root: bool
    parent: int  # member index, NONE for the root
    send_at_round: int  # NONE for the root
    children: tuple  # MAX_TREE_REDUCTION_ROUNDS member indices, NONE where no child


def tree_reduction_params(core_id, n):
    """Port of ``get_tree_reduction_params(core_id, N)``: the binary tree on ``vid = (N-1) - core_id`` (core 0 is
    the root); at round r a member receives from ``vid - 2^r`` when the low r+1 bits of its vid are all ones, and
    sends once, at round ``trailing_ones(vid)``, to ``vid + 2^round`` -- or to the root when that is past N (the
    orphan rule of a non-power-of-2 group), the root taking orphans in the rounds it has free."""
    core_id, n = int(core_id), int(n)
    children = [NONE] * MAX_TREE_REDUCTION_ROUNDS
    if n <= 1:
        return TreeParams(True, NONE, NONE, tuple(children))
    vid = (n - 1) - core_id
    rounds = tree_rounds(n)
    is_root = core_id == 0
    for r in range(rounds):
        mask = (2 << r) - 1
        if (vid & mask) == mask:
            child_vid = vid - (1 << r)
            if child_vid < n:
                children[r] = (n - 1) - child_vid
    parent = send = NONE
    if not is_root:
        t = _trailing_ones(vid)
        parent_vid = vid + (1 << t)
        parent = (n - 1) - parent_vid if parent_vid < n else 0
        send = t
    else:
        for c in range(1, n):
            cv = (n - 1) - c
            t = _trailing_ones(cv)
            if cv + (1 << t) >= n and children[t] == NONE:
                children[t] = c
    return TreeParams(is_root, parent, send, tuple(children))


def workload_for_core(cur_pos, core_num, cores, k_chunk_size):
    """Port of ``get_workload_for_core`` without a window: ``(PSt, num_chunks, chunk_start, chunk_end)`` of member
    ``core_num`` of a ``cores``-member group for a row at ``cur_pos`` (keys 0..cur_pos) and ``k_chunk_size``-token
    chunks. Fewer chunks than cores: member c < num_chunks takes chunk ``num_chunks - c - 1``; else the reversed
    even split (member 0 takes the last ``chunks_per_core`` (+1 residual) chunks)."""
    cur_pos, core_num, cores, k = int(cur_pos), int(core_num), int(cores), int(k_chunk_size)
    valid = ((cur_pos + 1 + k - 1) // k) * k
    num_chunks = valid // k
    if cores > num_chunks:
        per = 1 if core_num < num_chunks else 0
        start = (num_chunks - core_num - 1) * per
        end = (num_chunks - core_num) * per
    else:
        per = num_chunks // cores
        residuals = num_chunks % cores
        rev = cores - core_num - 1
        start = rev * per + min(residuals, rev)
        end = start + per + (1 if rev < residuals else 0)
    return valid // TILE, num_chunks, start, end


# ── pair split ───────────────────────────────────────────────────────────────────────────────────────────────────
def pair_half_of(col):
    """The pair half of the device in mesh column ``col``: its parity (the pair is columns ``(c, c ^ 1)``)."""
    return int(col) % 2


def pair_partner(coord):
    """The mesh coordinate ``(r, c ^ 1)`` holding the other half of ``coord = (r, c)``'s rows."""
    r, c = (int(v) for v in coord)
    return r, c ^ 1


def pair_split_reason(mesh_shape):
    """None when a mesh of ``mesh_shape`` (``(rows, cols)``; None for a device that is not a mesh) can run the pair
    split, else why not."""
    if mesh_shape is None:
        return "pair_split needs a mesh device (a single device has no pair partner)"
    dims = tuple(int(d) for d in mesh_shape)
    if len(dims) != 2:
        return f"pair_split needs a 2-D (rows, cols) mesh, got shape {dims}"
    rows, cols = dims
    if rows * cols <= 1:
        return f"pair_split on a single device {dims}: no pair partner (the pair is mesh columns (c, c ^ 1))"
    if cols % 2:
        return f"pair_split pairs mesh columns (c, c ^ 1): the mesh {dims} has an odd number of columns"
    return None


def pair_half_chunks(num_chunks, half):
    """``(first chunk, chunk count)`` of half ``half`` of a row's ``num_chunks`` K chunks: half 0 = ``[0, ceil(n /
    2))``, half 1 = ``[ceil(n / 2), n)`` (empty for a one-chunk row)."""
    n, first = int(num_chunks), (int(num_chunks) + 1) // 2
    return (0, first) if int(half) == 0 else (first, n - first)


def pair_workload_for_core(cur_pos, core_num, cores, k_chunk_size, half):
    """Mirror of ``kernels/pair_split.hpp`` ``get_pair_workload_for_core``: ``(PSt, num_chunks, chunk_start,
    chunk_end, half_chunks)`` of member ``core_num`` of a ``cores``-member group on the device holding half ``half``
    of a row at ``cur_pos``. PSt and num_chunks are the whole row's; within the half the stock reversed split (member
    0 takes the half's last chunks, member c has data iff ``c < half_chunks``)."""
    cur_pos, core_num, cores, k = int(cur_pos), int(core_num), int(cores), int(k_chunk_size)
    valid = ((cur_pos + 1 + k - 1) // k) * k
    num_chunks = valid // k
    base, cnt = pair_half_chunks(num_chunks, half)
    if cores > cnt:
        start = end = base
        if core_num < cnt:
            start = base + cnt - core_num - 1
            end = start + 1
    else:
        per, residuals = divmod(cnt, cores)
        rev = cores - core_num - 1
        start = base + rev * per + min(residuals, rev)
        end = start + per + (1 if rev < residuals else 0)
    return valid // TILE, num_chunks, start, end, cnt


def pair_key_range(cur_pos, k_chunk_size, half):
    """The keys ``[lo, hi)`` half ``half`` of a row at ``cur_pos`` attends (``(0, 0)`` for a skipped row, ``lo == hi``
    for an empty half): its chunks clipped to ``cur_pos + 1``."""
    if int(cur_pos) < 0:
        return 0, 0
    k = int(k_chunk_size)
    n = -(-(int(cur_pos) + 1) // k)
    base, cnt = pair_half_chunks(n, half)
    end = min((base + cnt) * k, int(cur_pos) + 1)
    return min(base * k, end), end


@dataclasses.dataclass(frozen=True)
class CoreWork:
    """One active core: its index ``i`` in the factory's core order, the logical core, its batch row, its member
    index in the row's reduction group and the group tree."""

    index: int
    core: tuple  # logical (x, y)
    batch: int
    member: int
    tree: TreeParams

    @property
    def do_reduce(self):
        return self.member == 0

    @property
    def do_output(self):
        return self.member == 0


@dataclasses.dataclass(frozen=True)
class CorePlan:
    """The factory's core allocation for one KV head per chip: ``batch`` groups of ``cores_per_head`` cores."""

    grid: tuple
    batch: int
    cores_per_head: int
    work: tuple

    @property
    def active_cores(self):
        return len(self.work)

    @property
    def rounds(self):
        return tree_rounds(self.cores_per_head)

    def group(self, b):
        """The cores of row ``b``'s reduction group, member order."""
        return self.work[b * self.cores_per_head : (b + 1) * self.cores_per_head]

    @property
    def roots(self):
        """Row b's reducer = output core = tree root (member 0), b ascending."""
        return [self.work[b * self.cores_per_head] for b in range(self.batch)]


def plan_cores(grid, batch, max_cores_per_head_batch):
    """The factory's core allocation (``num_kv_heads == 1``, linear group order). Raises ValueError when the
    batch does not fit the grid or a group would need more than ``MAX_TREE_REDUCTION_ROUNDS`` rounds."""
    gx, gy = int(grid[0]), int(grid[1])
    batch, cap = int(batch), int(max_cores_per_head_batch)
    available = gx * gy
    if batch < 1:
        raise ValueError(f"batch {batch} < 1")
    if available < batch:
        raise ValueError(f"cores available ({available}) must be >= batch size ({batch})")
    if cap < 1:
        raise ValueError(f"max_cores_per_head_batch must be > 0, got {cap}")
    uncapped = min(available, cap * batch) // batch
    cph = max(1, uncapped)
    if tree_rounds(cph) > MAX_TREE_REDUCTION_ROUNDS:
        raise ValueError(
            f"tree reduction max {MAX_TREE_REDUCTION_ROUNDS} rounds ({MAX_CORES_PER_HEAD} cores/head), got "
            f"{cph} cores/head: cap max_cores_per_head_batch"
        )
    params = [tree_reduction_params(m, cph) for m in range(cph)]
    work = tuple(
        CoreWork(index=i, core=(i % gx, i // gx), batch=i // cph, member=i % cph, tree=params[i % cph])
        for i in range(cph * batch)
    )
    return CorePlan(grid=(gx, gy), batch=batch, cores_per_head=cph, work=work)


def tile_bytes(dtype, tile=FULL_TILE):
    """Bytes of one tile of ``dtype`` (``"bfloat16"``, ``"float32"``, ``"bfloat8_b"``) and shape ``tile``."""
    h, w = tile
    if dtype == "bfloat16":
        return h * w * 2
    if dtype == "float32":
        return h * w * 4
    if dtype == "bfloat8_b":
        return h * w + h * w // 16  # one shared exponent byte per 16 datums
    raise ValueError(f"no tile size for {dtype}")


def _largest_divisor_at_most(n, cap):
    h = min(n, cap)
    while h > 1 and n % h:
        h -= 1
    return max(h, 1)


def _pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


IDENTITY_SCALAR_PACKED = 0x3F803F80  # pack_two_bfloat16_into_uint32({1.0, 1.0})


@dataclasses.dataclass(frozen=True)
class Geometry:
    """The call's shapes and knobs as the factory derives them (``num_kv_heads == 1``; K and V share the view)."""

    batch: int  # B: the page table's rows
    num_q_heads: int  # Hq, the logical Q heads
    head_dim: int  # D: Q's last dim (also K's and V's under the view)
    max_blocks: int  # the page table's width
    block_size: int  # the effective block size (the view), a tile multiple
    num_cache_blocks: int  # k_cache.shape[0] (the factory's Bkv)
    kv_dtype: str  # "bfloat16" / "bfloat8_b"
    k_chunk_size: int  # tokens, a power-of-2 multiple of 32
    scale: float
    cur_pos_stick: int  # the cur_pos buffer's aligned page size
    page_table_stick: int  # the page table buffer's aligned page size
    cache_position_modulo: int = 0  # 0: unbounded
    fp32_dest_acc_en: bool = False
    exp_approx_mode: bool = True

    @property
    def pnht(self):
        return -(-self.num_q_heads // TILE)

    @property
    def pnh(self):
        return self.pnht * TILE

    @property
    def dht(self):
        return self.head_dim // TILE

    @property
    def vdht(self):
        return self.dht

    @property
    def st(self):
        return self.max_blocks * self.block_size // TILE

    @property
    def sk_t(self):
        return self.k_chunk_size // TILE

    @property
    def block_t(self):
        return self.block_size // TILE

    @property
    def capacity_t(self):
        return int(self.cache_position_modulo or 0) // TILE

    @property
    def use_half_tile(self):
        return self.num_q_heads <= HALF_TILE_MAX_Q_HEADS

    @property
    def dst_size(self):
        return 4 if self.fp32_dest_acc_en else 8

    @property
    def small_tile(self):
        """The tile of the Q-shaped CBs (q, mask, scalars, statistics, intermediates): 16x32 for <= 16 heads."""
        return HALF_TILE if self.use_half_tile else FULL_TILE

    def check(self):
        """ValueError when the geometry leaves the supported configuration."""
        if not 1 <= self.num_q_heads <= MAX_Q_HEADS:
            raise ValueError(f"num_q_heads {self.num_q_heads} not in 1..{MAX_Q_HEADS}")
        if self.head_dim % TILE:
            raise ValueError(f"head_dim {self.head_dim} is not a multiple of {TILE}")
        if self.block_size % TILE or self.block_size <= 0:
            raise ValueError(f"block_size {self.block_size} is not a positive multiple of {TILE}")
        if self.k_chunk_size <= 0 or self.k_chunk_size % TILE or not _pow2(self.k_chunk_size):
            raise ValueError(
                f"k_chunk_size {self.k_chunk_size} must be a power-of-2 multiple of {TILE} "
                "(the dynamic chunk mode, 0, is not ported)"
            )
        if self.cache_position_modulo and self.cache_position_modulo % self.block_size:
            raise ValueError(
                f"cache_position_modulo {self.cache_position_modulo} is not a multiple of the block {self.block_size}"
            )
        if self.kv_dtype not in ("bfloat16", "bfloat8_b"):
            raise ValueError(f"K / V dtype {self.kv_dtype} is not bfloat16 or bfloat8_b")


def matmul_config(g):
    """The factory's QK and PV matmul blockings and the compute defines."""
    dst = g.dst_size
    pnht, sk, dht, vdht = g.pnht, g.sk_t, g.dht, g.vdht
    qk_w = min(sk, dst)
    qk_h = _largest_divisor_at_most(pnht, dst // qk_w) if qk_w == sk else 1
    out_w = min(vdht, dst)
    out_h = _largest_divisor_at_most(pnht, dst // out_w) if out_w == vdht else 1
    dht_gran = min(dht, dst)
    if not _pow2(dht_gran):
        dht_gran = 1
    sub_exp = min(sk, dst)
    mul_bcast = min(pnht * sk, dst)
    if not (_pow2(sub_exp) and _pow2(mul_bcast)):
        raise ValueError(f"granularities {sub_exp} / {mul_bcast} must be powers of 2")
    if sk % qk_w or vdht % out_w:
        raise ValueError(f"subblock widths {qk_w} / {out_w} do not divide {sk} / {vdht}")
    return dict(
        qk_in0_block_w=dht,
        qk_out_subblock_w=qk_w,
        qk_out_subblock_h=qk_h,
        qk_in0_num_subblocks=pnht // qk_h,
        qk_in1_num_subblocks=sk // qk_w,
        out_in0_block_w=sk,
        out_out_subblock_w=out_w,
        out_out_subblock_h=out_h,
        out_in0_num_subblocks=pnht // out_h,
        out_in1_num_subblocks=vdht // out_w,
        defines=[
            ("EXP_APPROX_MODE", str(int(bool(g.exp_approx_mode)))),
            ("DHT_GRANULARITY", str(dht_gran)),
            ("SUB_EXP_GRANULARITY", str(sub_exp)),
            ("STATS_GRANULARITY", str(sub_exp)),
        ],
    )


def reader_ct(g, plan):
    """The reader's 36 scalar compile-time args (the TensorAccessorArgs blocks follow, host side)."""
    q_tile = tile_bytes("bfloat16", g.small_tile)
    args = [
        g.batch,
        g.pnht,
        g.st,
        g.dht,
        g.vdht,
        g.sk_t,
        plan.active_cores,
        0,  # is_q_sharded
        g.cur_pos_stick,
        1,  # is_paged_attention
        1,  # num_kv_heads
        g.block_t,
        g.num_cache_blocks,  # Bkv
        1,  # q_heads_parallel_factor
        plan.cores_per_head,
        1,  # num_heads_per_core
        g.batch,  # num_output_cores
        1,  # is_causal
        0,  # use_attention_mask
        0,  # use_attention_sink
        g.dst_size,  # max_dynamic_chunk_size
        0,  # tilize_q
        0,  # reuse_k (V is its own tensor)
        int(g.use_half_tile),
        g.pnht * g.dht * q_tile,  # q_chunk_size_bytes (sharded Q only)
        0,  # is_cur_pos_tensor_sharded
        0,  # is_page_table_sharded
        tile_bytes("bfloat16"),  # the Q tensor's page (full tile)
        0,  # sliding_window_size
        g.block_size,  # original_block_size
        SEM_K_MCAST,
        0,  # q_locally_available
        0,  # use_k_mcast (column-major group indexing)
        g.num_cache_blocks,  # Bmask
        g.capacity_t,
        0,  # spec_multi_pos_tiles
    ]
    assert len(args) == READER_CT_SCALARS
    return args


def writer_ct(g, plan):
    """The writer's 23 scalar compile-time args (the out and stats TensorAccessorArgs follow, host side)."""
    args = [
        g.pnht,
        g.st,
        g.vdht,
        g.sk_t,
        IDENTITY_SCALAR_PACKED,
        plan.cores_per_head,  # num_cores_per_batch
        plan.active_cores,
        SEM_REDUCER,
        SEM_OUTPUT,
        0,  # is_out_sharded
        g.num_q_heads,
        1,  # num_kv_heads
        plan.cores_per_head,
        1,  # num_heads_per_core
        g.batch,  # num_reducer_cores
        g.batch,  # num_output_cores
        2,  # output element size (bf16)
        1,  # is_causal
        g.dst_size,  # max_dynamic_chunk_size
        1,  # q_heads_parallel_factor
        0,  # sliding_window_size
        g.block_size,  # original_block_size
        0,  # spec_multi_pos_tiles
    ]
    assert len(args) == WRITER_CT_SCALARS
    return args


def compute_ct(g, plan):
    """The compute kernel's 28 compile-time args."""
    mm = matmul_config(g)
    args = [
        g.st,
        g.dht,
        g.vdht,
        g.pnht,  # Sq_chunk_t
        g.sk_t,
        mm["qk_in0_block_w"],
        mm["qk_out_subblock_w"],
        mm["qk_out_subblock_h"],
        mm["qk_in0_num_subblocks"],
        mm["qk_in1_num_subblocks"],
        mm["out_in0_block_w"],
        mm["out_out_subblock_w"],
        mm["out_out_subblock_h"],
        mm["out_in0_num_subblocks"],
        mm["out_in1_num_subblocks"],
        plan.cores_per_head,
        1,  # num_heads_per_core
        1,  # is_causal
        0,  # use_attention_mask
        0,  # use_attention_sink
        g.dst_size,  # max_dynamic_chunk_size
        0,  # tilize_q
        1,  # q_heads_parallel_factor
        int(g.use_half_tile),
        f32_bits(g.scale),
        0,  # sliding_window_size
        g.block_size,  # original_block_size
        0,  # spec_multi_pos_tiles
    ]
    assert len(args) == COMPUTE_CT_ARGS
    return args


@dataclasses.dataclass(frozen=True)
class CbSpec:
    index: int
    pages: int
    page_size: int
    dtype: str  # "bfloat16" / "bfloat8_b" / "int32"
    tile: tuple = None  # (h, w) for a tile-shaped CB, None for the raw sticks

    @property
    def total_size(self):
        return self.pages * self.page_size


def cb_specs(g, plan):
    """The factory's CB list for this configuration (q format bf16, mask / scalars / statistics /
    intermediates bf16 in the Q-shaped tile, K / V in the cache format, the output bf16 full tiles)."""
    small = g.small_tile
    s_bytes = tile_bytes("bfloat16", small)
    full = tile_bytes("bfloat16", FULL_TILE)
    kv = tile_bytes(g.kv_dtype, FULL_TILE)
    q_tiles = g.pnht * g.dht
    out_tiles = g.pnht * g.vdht
    qk_tiles = g.pnht * g.sk_t
    stats = g.pnht
    blocks = min(plan.cores_per_head - 1, plan.rounds)
    specs = [
        CbSpec(CB_Q, q_tiles, s_bytes, "bfloat16", small),
        CbSpec(CB_K, g.sk_t * g.dht * 2, kv, g.kv_dtype),
        CbSpec(CB_V, g.sk_t * g.vdht * 2, kv, g.kv_dtype),
        CbSpec(CB_MASK, qk_tiles, s_bytes, "bfloat16", small),
        CbSpec(CB_SCALE, 1, s_bytes, "bfloat16", small),
        CbSpec(CB_M_IN, stats, s_bytes, "bfloat16", small),
        CbSpec(CB_L_IN, stats, s_bytes, "bfloat16", small),
        CbSpec(CB_POS_WRITER, 1, g.cur_pos_stick, "int32"),
        CbSpec(CB_POS_COMPUTE, 1, g.cur_pos_stick, "int32"),
        CbSpec(CB_PAGE_TABLE, 1, g.page_table_stick, "int32"),
        CbSpec(CB_Q_RM, q_tiles, s_bytes, "bfloat16", small),
        CbSpec(CB_COL_IDENTITY, 1, full, "bfloat16", FULL_TILE),
        CbSpec(CB_ZERO, 1, s_bytes, "bfloat16", small),
        CbSpec(CB_QK, qk_tiles, s_bytes, "bfloat16", small),
        CbSpec(CB_OUT_IM, out_tiles, s_bytes, "bfloat16", small),
        CbSpec(CB_OUT_ACC, out_tiles, s_bytes, "bfloat16", small),
    ]
    for idx in (CB_MAX_1, CB_MAX_2, CB_SUM_1, CB_SUM_2, CB_EXP_MAX_DIFF, CB_PREV_SUM_2, CB_EXP_MAX_DIFF_2):
        specs.append(CbSpec(idx, stats, s_bytes, "bfloat16", small))
    specs += [
        CbSpec(CB_OUT_ACC_2, out_tiles, s_bytes, "bfloat16", small),
        CbSpec(CB_OUT_O, out_tiles, s_bytes, "bfloat16", small),
        CbSpec(CB_OUT_M, stats, s_bytes, "bfloat16", small),
        CbSpec(CB_OUT_L, stats, s_bytes, "bfloat16", small),
    ]
    if blocks > 0:
        specs.append(CbSpec(CB_INTERMED, (out_tiles + 2 * stats) * blocks, s_bytes, "bfloat16", small))
    specs.append(CbSpec(CB_OUT_FINAL, out_tiles, full, "bfloat16", FULL_TILE))
    return specs


def cb_bytes(specs):
    return sum(s.total_size for s in specs)


def runtime_args(g, plan, phys, addrs):
    """Per-core runtime args ``{(x, y): (reader, writer, compute)}`` in the stock layout (the writer's stats
    address at index 1). ``phys((x, y)) -> (noc_x, noc_y)`` maps a logical core to its NoC coordinates;
    ``addrs`` holds the buffer addresses ``q, k, v, cur_pos, page_table, out, stats``."""
    cph = plan.cores_per_head
    noc = {w.core: tuple(int(c) for c in phys(w.core)) for w in plan.work}
    root_x = [noc[w.core][0] for w in plan.roots]
    root_y = [noc[w.core][1] for w in plan.roots]
    out = {}
    for w in plan.work:
        group = plan.group(w.batch)
        reader = [
            addrs["q"],
            addrs["k"],
            addrs["v"],
            addrs["cur_pos"],
            addrs["page_table"],
            0,  # attn_mask
            0,  # attention_sink
            g.page_table_stick,
            int(w.do_output),
            0,  # cur_head (the KV head group)
            w.batch,
            w.member,
            NONE,  # cur_pos: read from the cur_pos tensor
            0,  # do_k_mcast
            0,  # mcast_x
            0,  # mcast_y0
            0,  # mcast_y1
            0,  # num_dests
        ]
        assert len(reader) == READER_RT_FIXED
        reader += root_x + root_y  # output cores
        writer = [
            addrs["out"],
            addrs["stats"],
            int(w.do_output),
            0,  # cur_head
            w.batch,
            w.member,
            NONE,  # cur_pos
            int(w.tree.is_root),
            w.tree.parent,
            w.tree.send_at_round,
            *w.tree.children,
        ]
        assert len(writer) == WRITER_RT_FIXED
        writer += [noc[m.core][0] for m in group] + [noc[m.core][1] for m in group]
        writer += root_x + root_y  # reducer cores (one per row)
        writer += root_x + root_y  # output cores (the same)
        compute = [int(w.do_reduce), w.batch, w.member, NONE, int(w.tree.is_root), w.tree.parent, *w.tree.children]
        assert len(compute) == COMPUTE_RT_ARGS
        out[w.core] = (reader, writer, compute)
    assert all(len(v[1]) == WRITER_RT_FIXED + 2 * cph + 4 * plan.batch for v in out.values())
    return out


@dataclasses.dataclass(frozen=True)
class LsePlan:
    """Everything the op builds a program from, minus the ttnn objects."""

    geometry: Geometry
    cores: CorePlan
    cbs: tuple
    reader_ct: tuple
    writer_ct: tuple
    compute_ct: tuple
    defines: tuple  # the compute kernel's (with PAIR_SPLIT_DEFINE under pair_split)
    pair_split: bool = False

    @property
    def dataflow_defines(self):
        """The reader's and the writer's defines: PAIR_SPLIT_DEFINE under pair_split, else none (the stock)."""
        return (PAIR_SPLIT_DEFINE,) if self.pair_split else ()

    def pair_ct(self, half):
        """The compile-time args every kernel appends on the device holding pair half ``half`` (none without
        pair_split): the reader / writer after their last TensorAccessorArgs block, the compute at index
        ``COMPUTE_CT_ARGS``."""
        if not self.pair_split:
            return ()
        if int(half) not in (0, 1):
            raise ValueError(f"pair half {half} is not 0 or 1")
        return (int(half),)

    @property
    def cb_total_bytes(self):
        return cb_bytes(self.cbs)

    def describe(self):
        g, c = self.geometry, self.cores
        return (
            f"B={g.batch} Hq={g.num_q_heads} D={g.head_dim} S={g.st * TILE} block={g.block_size} "
            f"k_chunk={g.k_chunk_size} kv={g.kv_dtype} grid={c.grid[0]}x{c.grid[1]} cores/head={c.cores_per_head} "
            f"active={c.active_cores} rounds={c.rounds} half_tile={g.use_half_tile} dst={g.dst_size} "
            f"CB={self.cb_total_bytes} B" + (" pair_split" if self.pair_split else "")
        )


def plan(g, grid, max_cores_per_head_batch, pair_split=False):
    """The full host plan of one call (``pair_split``: the per-device plan of the pair split, see the module
    docstring). Raises ValueError out of contract."""
    g.check()
    cores = plan_cores(grid, g.batch, max_cores_per_head_batch)
    mm = matmul_config(g)
    return LsePlan(
        geometry=g,
        cores=cores,
        cbs=tuple(cb_specs(g, cores)),
        reader_ct=tuple(reader_ct(g, cores)),
        writer_ct=tuple(writer_ct(g, cores)),
        compute_ct=tuple(compute_ct(g, cores)),
        defines=tuple(mm["defines"]) + ((PAIR_SPLIT_DEFINE,) if pair_split else ()),
        pair_split=bool(pair_split),
    )


def lse_from_stats_values(m, l, scale):
    """The row's log-sum-exp of the scaled scores from the exported statistics (floats or tensors with
    ``math``-like semantics are the caller's business; this is the scalar form for the host tests)."""
    return float(scale) * float(m) + math.log(float(l))
