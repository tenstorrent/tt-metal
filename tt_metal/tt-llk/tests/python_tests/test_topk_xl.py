# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SFPU topk_xl test (Blackhole).

Input:  flat bf16 elements (elt a is at L1[2*a]).
Output: per row, slot0's value tiles, then slot0's index tiles, as uint32.
- value word: bf16 in the high 16 bits, low 16 zero
- index word: the flat uint32 index
- padding lane: high 16 bits of value are 0xFF80 (bf16 -inf)
"""

# For the sake of exactness, SrcA stimuli are built by hand with numpy
# and compared against torch.topk.
import numpy as np
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TopKXLGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TOPK_XL
from helpers.utils import passed_test

ELEMENTS_PER_TILE = 1024
BF16_NEG_INF_HI16 = 0xFF80  # high 16 bits of bf16 -inf, padding
FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.UInt32)

# Generic separate_indices params. 0x2A9 is hard to alias with itself, so a
# wrong shift or mask width alters it.
GROUP_ID = 0x2A9
GROUP_SHIFT = 16
# TOPK_XL_INDEX_OP codes in sources/topk_xl_test.cpp.
INDEX_OP_CODE = {"separate": 1, "remove_msb": 2}


def _tiles_per_sequence(K: int) -> int:
    return (K + ELEMENTS_PER_TILE - 1) // ELEMENTS_PER_TILE


def _decode_row_major(raw: int, K: int) -> int:
    """
    Translate an add_lsb tile coordinate to its row-major position within a chunk.
    Mirrors ckernel::sfpu::_topk_xl_decode_row_major_index_.
    """
    raw &= 0xFFFF
    d = (raw >> 6) & 0xF
    d |= (raw & 0xF) << 4
    d |= ((raw >> 10) & 0x1) << 8
    d |= ((raw >> 4) & 0x1) << 9
    if K == 2048:
        d |= ((raw >> 5) & 0x1) << 10
    return d


def _distinct_bf16_from_hi16(hi16: np.ndarray) -> np.ndarray:
    """Turn uint16 into exactly-representable bf16 values as float32."""
    return (hi16.astype(np.uint32) << np.uint32(16)).view(np.float32).copy()


def _make_row(search_len: int, seed: int, mode: str) -> np.ndarray:
    """One row of `search_len` values as float32.

    mode="positive": distinct, exactly-representable, all >= 1.0 (unambiguous top-k).
    mode="signed":   distinct, exactly-representable, spanning negatives and
                     positives (bf16 sign handling in the sort).
    mode="random":   random bf16 (ties likely; not distinct).
    """
    rng = np.random.default_rng(seed)
    if mode == "positive":
        # hi16 from 0x3F80 up: consecutive exactly-representable bf16 >= 1.0.
        hi16 = 0x3F80 + rng.permutation(search_len)
        return _distinct_bf16_from_hi16(hi16)
    if mode == "signed":
        n_neg = search_len // 2
        pos = 0x3F80 + np.arange(search_len - n_neg)  # >= +1.0
        neg = 0xBF80 + np.arange(n_neg)  # <= -1.0 (sign bit set)
        hi16 = rng.permutation(np.concatenate([pos, neg]))
        return _distinct_bf16_from_hi16(hi16)
    if mode == "random":
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(search_len, generator=gen).to(torch.bfloat16).float().numpy()
    if mode == "all_equal":
        # Degenerate input: all identical.
        return np.full(search_len, 3.0, dtype=np.float32)
    if mode == "zeros_win":
        # Everything else is negative, so the +0 lanes are the largest values in the
        # row and win the top-K. That is the only arrangement in which a zero's
        # index reaches the output at all.
        #
        # Zeros go in chunk 0 only, so their coordinates form an exact set the
        # checker can compare against. Position 0 is excluded: it encodes to
        # coordinate 0, so its fused word would be 0x00000000, which is
        # indistinguishable from a flushed one.
        assert search_len % 2 == 0, "zeros_win expects num_chunks=2 with a full tail"
        chunk_len = search_len // 2
        vals = _distinct_bf16_from_hi16(0xBF80 + rng.permutation(search_len))
        zero_pos = 1 + rng.choice(chunk_len - 1, chunk_len // 2, replace=False)
        vals[zero_pos] = 0.0
        return vals
    raise ValueError(f"unknown mode {mode}")


def _build_input(K, num_chunks, tail_elements, num_rows, mode, as_float32=False):
    """
    Build the flat input buffer and the per-row value tensor for the golden.
    Returns (src_A, rows_fp32).

    `as_float32` keeps the buffer in float32 for the 32-bit unpack_to_dest path.
    Every mode generates exactly-representable bf16 values, so the float32 and
    bf16 buffers hold the same numbers and share one golden."""
    tiles_per_seq = _tiles_per_sequence(K)
    search_len = (num_chunks - 1) * K + tail_elements
    total_input_tiles = num_rows * num_chunks * tiles_per_seq

    src = np.zeros(total_input_tiles * ELEMENTS_PER_TILE, dtype=np.float32)
    rows = np.empty((num_rows, search_len), dtype=np.float32)

    for r in range(num_rows):
        rows[r] = _make_row(search_len, r, mode)
        for c in range(num_chunks):
            active = K if c < num_chunks - 1 else tail_elements
            unit_start = ((r * num_chunks + c) * tiles_per_seq) * ELEMENTS_PER_TILE
            src[unit_start : unit_start + active] = rows[r, c * K : c * K + active]
            # Remaining slots stay 0; the copy path clears inactive lanes to -inf
            # regardless, so their L1 contents are never read.

    src_t = torch.from_numpy(src)
    return (src_t if as_float32 else src_t.to(torch.bfloat16)), torch.from_numpy(rows)


def _config(
    K,
    num_chunks,
    tail_elements,
    num_rows,
    src_A,
    index_op=0,
    group_id=0,
    group_shift=16,
    core_id=0,
    ascending=False,
    fused_reduce=False,
    chunk_base_mode=0,
    chunk_base=0,
    dest_sync=DestSync.Full,
    formats=FORMATS,
):
    """Build the TestConfig for one variant.

    index_op: 0 row-major, 1 separate, 2 remove_msb (2 packs one region, else two).
    The fused-reduce path splits indices at the end, so it packs two regions too."""
    tiles_per_seq = _tiles_per_sequence(K)
    result_tiles = (1 if index_op == 2 else 2) * tiles_per_seq
    # 32-bit input does unpack_to_dest (is_32bit_input), while bf16 goes
    # through SrcA then the MATH A2D datacopy.
    is_32bit = formats.input_format in (DataFormat.Float32, DataFormat.Int32)
    # Unused dummy tile, but it is declared with the input format, so its dtype has
    # to follow it or the stimuli packer is handed a mislabeled tensor.
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=format_dict[formats.input_format])

    return TestConfig(
        test_name="sources/topk_xl_test.cpp",
        formats=formats,
        templates=[
            DEST_SYNC(dest_sync),  # K=2048 uses Dest tiles 0-7 (fp32 with SyncFull).
            TOPK_XL(
                k=K,
                num_chunks=num_chunks,
                tail_elements=tail_elements,
                num_rows=num_rows,
                index_op=index_op,
                group_id=group_id,
                group_shift=group_shift,
                core_id=core_id,
                ascending=ascending,
                fused_reduce=fused_reduce,
                chunk_base_mode=chunk_base_mode,
                chunk_base=chunk_base,
            ),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=num_rows * num_chunks * tiles_per_seq,
            tile_count_B=1,
            tile_count_res=num_rows * result_tiles,
        ),
        dest_acc=DestAccumulation.Yes,  # 32-bit fused value|index words in Dest.
        unpack_to_dest=is_32bit,
    )


def _as_uint32(result):
    """Read the flat L1 uint32 result into a numpy uint32 array."""
    return np.asarray(
        torch.tensor(result, dtype=format_dict[FORMATS.output_format]), dtype=np.uint32
    )


def _extract_hw_topk(result, K, num_rows):
    """Per row, pair value[j] with index[j] and drop bf16 -inf padding lanes.
    Returns a list of (indices_uint32, values_float32): the surviving top-K."""
    tiles_per_seq = _tiles_per_sequence(K)
    res = _as_uint32(result)
    per_row = 2 * tiles_per_seq * ELEMENTS_PER_TILE
    region = tiles_per_seq * ELEMENTS_PER_TILE
    assert (
        res.size == num_rows * per_row
    ), f"result size {res.size} != expected {num_rows * per_row}"

    out = []
    for r in range(num_rows):
        block = res[r * per_row : (r + 1) * per_row]
        value_words = block[:region]
        index_words = block[region:]
        hi16 = (value_words >> np.uint32(16)) & np.uint32(0xFFFF)
        finite = hi16 != BF16_NEG_INF_HI16
        out.append((index_words[finite], value_words[finite].view(np.float32)))
    return out


def _check(
    result,
    K,
    num_rows,
    rows,
    gold_indices,
    compare_index_set,
    index_offset=0,
    value_format=FORMATS.input_format,
):
    """Validate the packed result against the golden.

    Always: exactly K finite lanes per row, K distinct indices, the top-K value
    multiset matches, and each returned index points at the value returned
    alongside it. `compare_index_set` additionally requires the exact top-K index
    set; it is off for tie-prone inputs, where the tie-break makes the chosen
    indices ambiguous.

    `index_offset` is the starting chunk_base, which the LLK ORs into every
    reported index, so the golden positions shift by it."""
    gold_values = rows.gather(1, gold_indices)  # [num_rows, K] top-K values per row

    for r, (hw_idx, hw_val) in enumerate(_extract_hw_topk(result, K, num_rows)):
        assert (
            hw_idx.size == K
        ), f"row {r}: expected {K} finite top-K lanes, got {hw_idx.size}"

        hw_idx_list = [int(x) - index_offset for x in hw_idx.tolist()]
        assert len(set(hw_idx_list)) == K, f"row {r}: returned indices are not distinct"
        assert all(0 <= i < rows.shape[1] for i in hw_idx_list), (
            f"row {r}: indices outside the row after subtracting chunk_base "
            f"{index_offset}; got {sorted(hw_idx_list)[:4]} .. {sorted(hw_idx_list)[-4:]}"
        )

        if compare_index_set:
            hw_set = set(hw_idx_list)
            gold_set = set(int(x) for x in gold_indices[r].tolist())
            assert hw_set == gold_set, (
                f"row {r}: top-K index set mismatch\n"
                f"  missing (in golden, not hw): {sorted(gold_set - hw_set)[:16]}\n"
                f"  extra   (in hw, not golden): {sorted(hw_set - gold_set)[:16]}"
            )

        # Sort both: Dest tile order is internal. Correct even under ties.
        hw_vals_sorted = torch.sort(torch.from_numpy(hw_val.astype(np.float32))).values
        gold_vals_sorted = torch.sort(gold_values[r]).values
        assert passed_test(
            gold_vals_sorted, hw_vals_sorted, value_format
        ), f"row {r}: top-K value mismatch"

        # Index <-> value pairing. Exact, because every stimulus here is exactly
        # representable in bf16 and the value word is [bf16 << 16 | 0].
        row = rows[r]
        for idx, val in zip(hw_idx_list, hw_val.tolist()):
            assert float(row[idx]) == float(
                val
            ), f"row {r}: index {idx} value {val} != input {float(row[idx])}"


# num_chunks == 1 is the single-chunk path (copy -> local_sort -> separate ->
# rebuild), where search_len == K so the top-K is the whole row. num_chunks >= 2
# adds the merge tree, where the top-K is a strict subset. partial_tail pads the
# last chunk to exercise the -inf padding, and is a function of num_chunks so that
# num_chunks == 1 never asks for search_len < K. num_rows=2 covers the row loop and
# the chunk_base reset between rows.
@parametrize(
    K=[512, 1024, 2048],
    num_chunks=[1, 2, 4],
    partial_tail=lambda num_chunks: [False] if num_chunks == 1 else [False, True],
)
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl(K, num_chunks, partial_tail):
    num_rows = 2
    tail_elements = (K // 2) if partial_tail else K

    src_A, rows = _build_input(K, num_chunks, tail_elements, num_rows, "positive")
    gold_indices = get_golden_generator(TopKXLGolden)(rows, K)
    result = _config(K, num_chunks, tail_elements, num_rows, src_A).run().result
    _check(result, K, num_rows, rows, gold_indices, compare_index_set=True)


# Distinct values spanning negatives and positives: the top-K are the largest
# (positive) values, so negatives must sort below them. Exercises bf16 sign
# handling under the INT32 SFPSWAP compare.
@parametrize(
    K=[512, 1024, 2048],
    num_chunks=[2],
)
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_signed(K, num_chunks):
    num_rows = 1
    tail_elements = K  # full chunks, no padding

    src_A, rows = _build_input(K, num_chunks, tail_elements, num_rows, "signed")
    gold_indices = get_golden_generator(TopKXLGolden)(rows, K)
    result = _config(K, num_chunks, tail_elements, num_rows, src_A).run().result
    _check(result, K, num_rows, rows, gold_indices, compare_index_set=True)


# Random bf16, so ties are likely. The tie-break makes the chosen indices
# ambiguous, so validate K distinct indices + the top-K value multiset only.
@parametrize(
    K=[512, 1024, 2048],
    num_chunks=[2],
)
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_ties(K, num_chunks):
    num_rows = 1
    tail_elements = K  # full chunks, no padding: all lanes finite

    src_A, rows = _build_input(K, num_chunks, tail_elements, num_rows, "random")
    gold_indices = get_golden_generator(TopKXLGolden)(rows, K)
    result = _config(K, num_chunks, tail_elements, num_rows, src_A).run().result
    _check(result, K, num_rows, rows, gold_indices, compare_index_set=False)


@parametrize(K=[512, 1024, 2048])
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_all_equal(K):
    """Every key is identical, so every compare-exchange in local_sort
    and rebuild takes the tie branch and resolves on the index bits."""
    (K,) = K

    src_A, rows = _build_input(K, 1, K, 1, "all_equal")  # single chunk
    result = _config(K, 1, K, 1, src_A).run().result
    hw_idx, hw_val = _extract_hw_topk(result, K, 1)[0]

    idx = sorted(int(i) for i in hw_idx.tolist())
    assert idx == list(range(K)), "all-equal row did not return 0..K-1"
    assert np.all(hw_val == rows[0][0].item()), "all-equal row returned a changed value"


def _check_separate(
    result, K, num_rows, rows, group_id=GROUP_ID, group_shift=GROUP_SHIFT, core_id=0
):
    """
    separate_indices: value region [value|0], index region
    [group_id<<shift | core_id<<11 | tile coordinate]. Checks the group_id and
    core_id fields, that the coordinate identifies each element's position, and
    the values.
    """
    res = _as_uint32(result)
    region = _tiles_per_sequence(K) * ELEMENTS_PER_TILE
    per_row = 2 * region
    assert res.size == num_rows * per_row

    for r in range(num_rows):
        block = res[r * per_row : (r + 1) * per_row]
        value_words = block[:region]
        index_words = block[region:]
        finite = (
            (value_words >> np.uint32(16)) & np.uint32(0xFFFF)
        ) != BF16_NEG_INF_HI16

        hw_val = value_words[finite].view(np.float32)
        hw_iw = index_words[finite]
        assert hw_iw.size == K, f"row {r}: expected {K} lanes, got {hw_iw.size}"

        group = hw_iw >> np.uint32(group_shift)
        raw = hw_iw & np.uint32((1 << group_shift) - 1)
        assert np.all(group == group_id), (
            f"row {r}: group_id bits wrong at shift {group_shift}; got "
            f"{sorted(set(int(g) for g in group.tolist()))[:4]}, want {group_id}"
        )

        # core_id sits in bits [15:11]. A group_shift of 11 or less puts the group
        # id over that field, leaving no separable core bits to read back.
        if group_shift > 11:
            hw_core = (raw >> np.uint32(11)) & np.uint32(0x1F)
            assert np.all(hw_core == core_id), (
                f"row {r}: core_id bits [15:11] wrong; got "
                f"{sorted(set(int(c) for c in hw_core.tolist()))[:4]}, want {core_id}"
            )

        pos = [_decode_row_major(int(x), K) for x in raw.tolist()]
        assert set(pos) == set(range(K)), f"row {r}: decoded positions != 0..K-1"

        # The coordinate names a position: input[pos] must be the value paired with
        # it. Exact, since `positive` is consecutive bf16 patterns, so any tolerance
        # would admit a coordinate several positions off.
        row = rows[r]
        for p, v in zip(pos, hw_val.tolist()):
            assert float(row[p]) == float(
                v
            ), f"row {r}: pos {p} value {v} != input {float(row[p])}"
        # value multiset matches the input (single chunk => all values).
        assert passed_test(
            torch.sort(rows[r]).values,
            torch.sort(torch.from_numpy(hw_val.astype(np.float32))).values,
            FORMATS.input_format,
        )


def _check_remove_msb(result, K, num_rows):
    """
    remove_msb_values: fused region -> [0 | raw]. Check the value half is
    zeroed and the decoded positions form the full 0..K-1 set.
    """
    res = _as_uint32(result)
    region = _tiles_per_sequence(K) * ELEMENTS_PER_TILE
    assert res.size == num_rows * region

    for r in range(num_rows):
        block = res[r * region : (r + 1) * region]
        hi16 = (block >> np.uint32(16)) & np.uint32(0xFFFF)
        finite = hi16 != BF16_NEG_INF_HI16  # padding lanes are still -inf
        real = block[finite]
        assert real.size == K, f"row {r}: expected {K} lanes, got {real.size}"

        untouched = int(np.count_nonzero(real >> np.uint32(16)))
        assert untouched == 0, f"row {r}: value half not zeroed in {untouched} lanes"

        raw = real & np.uint32(0xFFFF)
        pos = [_decode_row_major(int(x), K) for x in raw.tolist()]
        assert set(pos) == set(
            range(K)
        ), f"row {r}: decoded positions are not the full 0..K-1 set"


@parametrize(
    K=[512, 1024, 2048],
    index_op=["separate", "remove_msb"],
)
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_index_ops(K, index_op):
    num_rows = 1

    src_A, rows = _build_input(K, 1, K, num_rows, "positive")  # single chunk
    configuration = _config(
        K,
        1,
        K,
        num_rows,
        src_A,
        index_op=INDEX_OP_CODE[index_op],
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
    )
    result = configuration.run().result

    if index_op == "separate":
        _check_separate(result, K, num_rows, rows)
    else:
        _check_remove_msb(result, K, num_rows)


def _positional_rank(values, descending):
    """
    rank[lane] = position of that lane's value in the sorted order.
    Requires distinct values.
    """
    order = np.argsort(-values if descending else values, kind="stable")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    return rank


@parametrize(K=[512, 1024, 2048], core_id=[1, 31])
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_core_id(K, core_id):
    """add_lsb_indices<K, core_id> puts core_id in index bits [15:11] (5 bits, up
    to 32 cores). The positions must come out unchanged."""
    num_rows = 1

    src_A, rows = _build_input(K, 1, K, num_rows, "positive")
    configuration = _config(
        K,
        1,
        K,
        num_rows,
        src_A,
        index_op=INDEX_OP_CODE["separate"],
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
        core_id=core_id,
    )
    result = configuration.run().result
    _check_separate(result, K, num_rows, rows, core_id=core_id)


# group_id_bit_shift is a runtime value saved in LREG12, so the group id can
# sit anywhere above the coordinate. 11 puts it immediately above the widest
# coordinate (and over the core_id field, so core_id must be 0), and at 20
# it clears the field entirely.
@parametrize(K=[512, 1024, 2048], group_shift=[11, 20])
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_group_shift(K, group_shift):
    src_A, rows = _build_input(K, 1, K, 1, "positive")
    configuration = _config(
        K,
        1,
        K,
        1,
        src_A,
        index_op=INDEX_OP_CODE["separate"],
        group_id=GROUP_ID,
        group_shift=group_shift,
    )
    result = configuration.run().result
    _check_separate(result, K, 1, rows, group_shift=group_shift)


# chunk_base is saved by one of three inits, based on a template argument:
# - init_static<hi, lo> takes both
# - init_upper<hi>(lo) only the high half
# - init(base) takes neither
# All three send an (upper16, lower16) pair to _sfpu_load_config32_, but the split varies.
# 0x1F800 is nonzero in both halves, and it's a multiple of every valid K.
@parametrize(K=[512, 1024, 2048], chunk_base_mode=[0, 1, 2])
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_chunk_base(K, chunk_base_mode):
    """Assert that the reported index is chunk_base + position."""
    num_rows, num_chunks = 2, 2
    chunk_base = 0x1F800

    src_A, rows = _build_input(K, num_chunks, K, num_rows, "positive")
    gold_indices = get_golden_generator(TopKXLGolden)(rows, K)
    configuration = _config(
        K,
        num_chunks,
        K,
        num_rows,
        src_A,
        chunk_base_mode=chunk_base_mode,
        chunk_base=chunk_base,
    )
    result = configuration.run().result
    _check(
        result,
        K,
        num_rows,
        rows,
        gold_indices,
        compare_index_set=True,
        index_offset=chunk_base,
    )


@parametrize(K=[512, 1024, 2048])
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_rebuild_ascending(K):
    """
    rebuild(..., ascending=true). `_topk_xl_merge_` has no direction argument and
    always keeps the max half, so the direction changes the order the survivors are
    rebuilt into, not which of them survive. Both runs must return the same top-K,
    and each lane must hold the same rank counted from the opposite end.

    The two directions are compared against each other rather than each checked
    alone, because rebuild does not leave Dest sorted in lane order."""
    (K,) = K
    num_rows, num_chunks = 1, 2

    src_A, rows = _build_input(K, num_chunks, K, num_rows, "positive")
    gold_indices = get_golden_generator(TopKXLGolden)(rows, K)

    desc_cfg = _config(K, num_chunks, K, num_rows, src_A, ascending=False)
    asc_cfg = _config(K, num_chunks, K, num_rows, src_A, ascending=True)
    # Build both before running either: `prepare()` is the build half of `run()`,
    # and under --compile-producer `run()` skips as soon as the first variant is
    # built, so the second would otherwise never emit its ELF.
    desc_cfg.prepare()
    asc_cfg.prepare()
    desc, asc = desc_cfg.run().result, asc_cfg.run().result

    _check(desc, K, num_rows, rows, gold_indices, compare_index_set=True)
    _check(asc, K, num_rows, rows, gold_indices, compare_index_set=True)

    for r, ((_, desc_val), (_, asc_val)) in enumerate(
        zip(
            _extract_hw_topk(desc, K, num_rows),
            _extract_hw_topk(asc, K, num_rows),
        )
    ):
        rank_desc = _positional_rank(desc_val, descending=True)
        rank_asc = _positional_rank(asc_val, descending=False)
        mismatch = int(np.count_nonzero(rank_desc != rank_asc))
        assert mismatch == 0, (
            f"row {r}: ascending rebuild is not the mirror of the descending one: "
            f"{mismatch}/{K} lanes hold a different rank; first at lane "
            f"{int(np.argmax(rank_desc != rank_asc))}"
        )


def _check_fused_reduce(result, K, num_rows, num_chunks, rows):
    """
    Checks the group_id bits, the top-K values, and each returned coordinate
    against the value paired with it.
    """
    res = _as_uint32(result)
    region = _tiles_per_sequence(K) * ELEMENTS_PER_TILE
    per_row = 2 * region
    assert res.size == num_rows * per_row

    for r in range(num_rows):
        block = res[r * per_row : (r + 1) * per_row]
        value_words = block[:region]
        index_words = block[region:]
        finite = (
            (value_words >> np.uint32(16)) & np.uint32(0xFFFF)
        ) != BF16_NEG_INF_HI16

        hw_val = value_words[finite].view(np.float32)
        hw_iw = index_words[finite]
        assert hw_iw.size == K, f"row {r}: expected {K} lanes, got {hw_iw.size}"
        assert np.all(
            hw_iw >> np.uint32(GROUP_SHIFT) == GROUP_ID
        ), f"row {r}: group_id bits wrong"

        raw = hw_iw & np.uint32((1 << GROUP_SHIFT) - 1)
        pos = [_decode_row_major(int(x), K) for x in raw.tolist()]
        assert all(0 <= p < K for p in pos), f"row {r}: coordinate outside 0..K-1"

        # A coordinate is per-chunk, so it must match its value in one of the
        # chunks. Exact, for the same reason as in _check.
        row = rows[r]
        for p, v in zip(pos, hw_val.tolist()):
            candidates = [float(row[c * K + p]) for c in range(num_chunks)]
            assert (
                float(v) in candidates
            ), f"row {r}: coordinate {p} value {v} matches no chunk ({candidates})"

        # The surviving values are the top-K of the whole row.
        gold = torch.topk(rows[r], K).values
        assert passed_test(
            torch.sort(gold).values,
            torch.sort(torch.from_numpy(hw_val.astype(np.float32))).values,
            FORMATS.input_format,
        ), f"row {r}: fused-reduce top-K value mismatch"


def _check_denormal_fused_word(result, K, rows):
    """The +0 lanes must come back with their own coordinates intact.

    A +0 value zeroes bits 30:23 of the fused word, so the 32-bit word is an fp32
    denormal. An FP32-mode move would flush it and take the index with it."""
    res = _as_uint32(result)
    region = _tiles_per_sequence(K) * ELEMENTS_PER_TILE
    block = res[: 2 * region]
    value_words = block[:region]
    index_words = block[region:]
    finite = ((value_words >> np.uint32(16)) & np.uint32(0xFFFF)) != BF16_NEG_INF_HI16

    hw_val = value_words[finite].view(np.float32)
    hw_iw = index_words[finite]
    assert hw_iw.size == K, f"expected {K} lanes, got {hw_iw.size}"

    row = rows[0].numpy()
    expected = {p for p in range(K) if row[p] == 0.0}  # zeros live in chunk 0
    assert expected, "stimulus produced no zeros"

    zero_lane = hw_val == 0.0
    assert int(np.count_nonzero(zero_lane)) == len(expected), (
        f"expected {len(expected)} zero-valued lanes in the top-K, got "
        f"{int(np.count_nonzero(zero_lane))}; zeros should outrank every negative"
    )

    raw = hw_iw[zero_lane] & np.uint32((1 << GROUP_SHIFT) - 1)
    got = {_decode_row_major(int(x), K) for x in raw.tolist()}
    assert got == expected, (
        "index bits of the denormal-shaped fused words did not survive: "
        f"{len(got)} distinct coordinates for {len(expected)} zero lanes; "
        f"missing {sorted(expected - got)[:8]}, unexpected {sorted(got - expected)[:8]}"
        + ("  (all zero: the index was flushed)" if got == {0} else "")
    )


@parametrize(K=[512, 1024, 2048])
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_denormal_fused_word(K):
    """
    Assert a legal input containing +0 returns the correct set and indices.

    Runs fused, because that is where the value and the index share one word. On
    the unfused path they sit in separate regions, so a value word is plain
    0x00000000 and flushing it is a no-op."""
    (K,) = K

    src_A, rows = _build_input(K, 2, K, 1, "zeros_win")
    configuration = _config(
        K,
        2,
        K,
        1,
        src_A,
        fused_reduce=True,
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
    )
    result = configuration.run().result
    _check_denormal_fused_word(result, K, rows)


@parametrize(K=[512, 1024, 2048], num_chunks=[2, 4])
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_fused_reduce(K, num_chunks):
    """
    merge/rebuild with fused=true: half the operand distance (there's no index
    region between the slots), 16 instead of 18 instructions in the MOP, and
    half the iteration count.
    """

    src_A, rows = _build_input(K, num_chunks, K, 1, "positive")
    configuration = _config(
        K,
        num_chunks,
        K,
        1,
        src_A,
        fused_reduce=True,
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
    )
    result = configuration.run().result
    _check_fused_reduce(result, K, 1, num_chunks, rows)


@parametrize(K=[512, 1024, 2048], partial_tail=[False, True])
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_input_float32(K, partial_tail):
    """
    fp32 input does unpack_to_dest instead of A2D datacopy.

    `partial_tail` makes it so UNPACK and MATH have to agree on how to do the copy:
      - K=1024: the copy clears SrcA with -inf instead of doing ZEROACC
      - K=2048: the tail's second tile is empty, so both threads return early
    """
    tail_elements = (K // 2) if partial_tail else K
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.UInt32)

    src_A, rows = _build_input(K, 2, tail_elements, 1, "positive", as_float32=True)
    gold_indices = get_golden_generator(TopKXLGolden)(rows, K)
    configuration = _config(K, 2, tail_elements, 1, src_A, formats=formats)
    result = configuration.run().result
    _check(
        result,
        K,
        1,
        rows,
        gold_indices,
        compare_index_set=True,
        value_format=formats.input_format,
    )


# DestSync.Half halves the Dest budget (4 fp32 tiles), which the unfused merge tree
# fits only for K <= 1024: K=2048 needs two 2-tile value regions plus their index
# regions, so 8 tiles under fp32 SyncFull.
@parametrize(K=[512, 1024])
@skip_for_wormhole
@skip_for_quasar
def test_topk_xl_dest_sync_half(K):
    (K,) = K

    src_A, rows = _build_input(K, 2, K, 2, "positive")
    gold_indices = get_golden_generator(TopKXLGolden)(rows, K)
    configuration = _config(K, 2, K, 2, src_A, dest_sync=DestSync.Half)
    result = configuration.run().result
    _check(result, K, 2, rows, gold_indices, compare_index_set=True)
