# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
CORRECTNESS HARNESS FOR THE *FULL* SFPLOADMACRO REBUILD (swap AND store).

A clone of ``test_topk_rebuild_macro.py`` pointed at
``sources/topk_rebuild_full_macro_test.cpp``, which is in turn a clone of
``topk_rebuild_macro_test.cpp`` with two changes:

  * the macro Sequence's Store byte is turned ON (selector 3, delay 2) and the
    Misc word puts the Store sub-unit on WaitForElapsedInstructions;
  * ``canonical_big_block_with_replay``'s sub-block A (rsf >= 4, K = 2048) and
    sub-block B (rsf >= 2, K >= 1024) run ``macro_ce_body``, a 14-instruction
    body in which the SFPSWAP rides the macro's Simple slot AND the matching
    SFPSTORE rides its Store slot.

WHAT IS UNDER TEST, AND WHY IT IS NEW. Every previous macro rebuild could only
hide the FIRST level of a lattice, because a macro-scheduled Simple must have
``VD == macroVD`` -- the register that macro's own load just wrote. Sub-blocks A
and B are the one place in the rebuild where the lattice IS one level
(``bitonic_sort_len_k``, 4 SFPSWAP) and where the store addresses equal the load
addresses, so the merge's FULL trick applies: 20 instructions / 24 cycles become
14 / 14. Both sub-blocks are ``if constexpr``'d away at rsf == 1, so K = 512 is
unchanged and acts as a control that the clone broke nothing.

Measured in ``perf_topk_rebuild_xl.py`` (Blackhole, MATH_ISOLATE, controls at
1.000 / 1.999):

    K       shipping   first-level macro   full macro
    512      374            350               350      (A and B compiled out)
    1024     822            774               734      = 1.120x
    2048    1810           1714              1554      = 1.165x

WHY THE PERF NUMBER NEEDS THIS. A misconfigured -- or simply all-zero, i.e.
"schedule nothing" -- ``LoadMacroConfig.Sequence`` degenerates an
``SFPLOADMACRO`` into a plain ``SFPLOAD``, which issues at the SAME rate as a
correctly configured macro because the scheduled ``SFPSWAP`` and ``SFPSTORE``
ride otherwise idle sub-units either way. Losing a whole bitonic level would
make the arm look FASTER, not slower. Only a golden comparison tells the two
apart.

AND THE ONE FAILURE MODE TIMING CANNOT SEE. A rebuild only PERMUTES its K
survivors, so a SINGLE merge+rebuild pair returns the same SET however broken
the sort is. ``num_chunks=4`` -- three CHAINED merge+rebuild pairs, where a
mis-ordered rebuild output becomes the next merge's supposedly-sorted operand --
is what turns a broken macro schedule red. It is exercised by
``test_topk_xl``, ``test_topk_xl_fused_reduce`` and
``test_topk_xl_rebuild_macro_mutation`` below.

CONTROLS BUILT IN.
  * Every test whose ``fused_reduce`` is False runs the SHIPPING unfused
    ``_topk_xl_rebuild_`` through the cloned kernel, so a failure there would
    indicate the clone broke something structural rather than the macro being
    wrong.
  * ``test_topk_xl_fused_reduce``, ``test_topk_xl_denormal_fused_word`` and
    ``test_topk_xl_rebuild_ascending`` exercise the macro rebuild at
    K = 512, 1024 and 2048 in BOTH sort directions. The directions are not a
    free ride on each other: they need different InstructionTemplate sets and a
    different load-order permutation, and in sub-blocks A and B they also swap
    which half of the eight registers the macro stores.
  * ``test_topk_xl_rebuild_macro_mutation`` is the tripwire: it clears the
    sequence's 0x80 bit so every scheduled ``SFPSWAP`` compares the loaded
    register against itself, and asserts the result STOPS matching the golden.
"""

# For the sake of exactness, SrcA stimuli are built by hand.
# helpers.stimuli_generator is very awkward for these tests.
from dataclasses import dataclass

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TopKXLGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    TopKSortDirection,
    TopKXLChunkBaseMode,
    TopKXLIndexOp,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TOPK_XL, TemplateParameter
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]


@dataclass
class REBUILD_MUTATE(TemplateParameter):
    """Mutation control for the SFPLOADMACRO rebuild.

    Emits ``#define REBUILD_MUTATE <0|1>``. With 1 the Simple sequence byte's
    0x80 bit is cleared, so the macro assigns ``Insn.VC = macroVD`` instead of
    ``Insn.VB = macroVD`` (SFPLOADMACRO.md:100) and the scheduled ``SFPSWAP``
    compares the freshly loaded register against ITSELF -- a no-op
    compare-exchange that silently drops one bitonic level. It MUST turn a green
    run red; if it does not, the golden is blind to the level this change
    touches and the passing run proves nothing.

    Defined here rather than in helpers/test_variant_parameters.py because it is
    consumed by exactly one kernel -- same rationale as the parameter classes in
    perf_topk_merge_macro.py.
    """

    rebuild_mutate: int = 0

    def convert_to_cpp(self) -> str:
        return f"#define REBUILD_MUTATE {self.rebuild_mutate}"


ELEMENTS_PER_TILE = 1024
FACE_DIM = 16  # a tile is a 2x2 grid of 16x16 faces
ELEMENTS_PER_FACE = FACE_DIM * FACE_DIM
BF16_NEG_INF_HI16 = 0xFF80  # high 16 bits of bf16 -inf, padding
FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.UInt32)

# Generic separate_indices params. 0x2A9 is hard to alias with itself, so a
# wrong shift or mask width alters it.
GROUP_ID = 0x2A9
GROUP_SHIFT = 16
# add_lsb_indices writes core_id into bits [15:11].
CORE_ID_SHIFT = 11
CORE_ID_MASK = 0x1F


def _tiles_per_sequence(K: int) -> int:
    return (K + ELEMENTS_PER_TILE - 1) // ELEMENTS_PER_TILE


def _decode_row_major(raw: int, K: int) -> int:
    """
    Translate an add_lsb tile coordinate into the element's offset in the chunk.
    Mirrors ckernel::sfpu::_topk_xl_decode_row_major_index_.

    add_lsb_indices stamps each lane with its own (row, column) in the 32x32 tile,
    plus, for K=2048, which of the sequence's two tiles it sits in:

        bits [4:0] row, bit [5] tile in sequence, bits [10:6] column

    The chunk is laid out as tiles, and a tile is four 16x16 faces in raster order,
    each face row-major. So the offset is: pick the tile, pick the face the (row,
    column) falls in, then index row-major inside that face.
    """
    raw &= 0xFFFF
    row = raw & 0x1F
    tile = ((raw >> 5) & 0x1) if K == 2048 else 0
    column = (raw >> 6) & 0x1F

    face = (row // FACE_DIM) * 2 + (column // FACE_DIM)
    offset_in_face = (row % FACE_DIM) * FACE_DIM + (column % FACE_DIM)
    return tile * ELEMENTS_PER_TILE + face * ELEMENTS_PER_FACE + offset_in_face


def _bitcast_float32(words: torch.Tensor) -> torch.Tensor:
    """Reinterpret integer bit patterns as float32."""
    return words.to(torch.int32).view(torch.float32)


def _distinct_bf16_from_hi16(hi16: torch.Tensor) -> torch.Tensor:
    """Turn uint16 into exactly-representable bf16 values as float32."""
    return _bitcast_float32(hi16.to(torch.int64) << 16)


def _make_row(search_len: int, seed: int, mode: str) -> torch.Tensor:
    """
    One row of `search_len` values as float32.
    Seeded by row index so that rows within a variant differ.

    mode="positive": distinct, exactly-representable, all >= 1.0 (unambiguous top-k).
    mode="signed":   distinct, exactly-representable, spanning negatives and
                     positives (bf16 sign handling in the sort).
    mode="random":   random bf16 (ties likely; not distinct).
    """
    gen = torch.Generator().manual_seed(seed)
    if mode == "positive":
        # hi16 from 0x3F80 up: consecutive exactly-representable bf16 >= 1.0.
        hi16 = 0x3F80 + torch.randperm(search_len, generator=gen)
        return _distinct_bf16_from_hi16(hi16)
    if mode == "signed":
        n_neg = search_len // 2
        pos = 0x3F80 + torch.arange(search_len - n_neg)  # >= +1.0
        neg = 0xBF80 + torch.arange(n_neg)  # <= -1.0 (sign bit set)
        hi16 = torch.cat([pos, neg])[torch.randperm(search_len, generator=gen)]
        return _distinct_bf16_from_hi16(hi16)
    if mode == "random":
        return torch.randn(search_len, generator=gen).to(torch.bfloat16).float()
    if mode == "all_equal":
        # Degenerate input: all identical.
        return torch.full((search_len,), 3.0, dtype=torch.float32)
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
        vals = _distinct_bf16_from_hi16(
            0xBF80 + torch.randperm(search_len, generator=gen)
        )
        zero_pos = 1 + torch.randperm(chunk_len - 1, generator=gen)[: chunk_len // 2]
        vals[zero_pos] = 0.0
        return vals
    raise ValueError(f"unknown mode {mode}")


def _build_input(K, num_chunks, tail_elements, num_rows, mode, as_float32):
    """
    Build the flat input buffer and the per-row value tensor for the golden.
    Returns (src_A, rows_fp32).

    `as_float32` keeps the buffer in float32 for the 32-bit unpack_to_dest path.
    Every mode generates exactly-representable bf16 values, so the float32 and
    bf16 buffers hold the same numbers and share one golden.
    """
    tiles_per_seq = _tiles_per_sequence(K)
    search_len = (num_chunks - 1) * K + tail_elements
    total_input_tiles = num_rows * num_chunks * tiles_per_seq

    src = torch.zeros(total_input_tiles * ELEMENTS_PER_TILE, dtype=torch.float32)
    rows = torch.empty((num_rows, search_len), dtype=torch.float32)

    for r in range(num_rows):
        rows[r] = _make_row(search_len, r, mode)
        for c in range(num_chunks):
            active = K if c < num_chunks - 1 else tail_elements
            unit_start = ((r * num_chunks + c) * tiles_per_seq) * ELEMENTS_PER_TILE
            src[unit_start : unit_start + active] = rows[r, c * K : c * K + active]
            # Remaining slots stay 0; the copy path clears inactive lanes to -inf
            # regardless, so their L1 contents are never read.

    return (src if as_float32 else src.to(torch.bfloat16)), rows


def _variant(
    K,
    num_chunks=1,
    tail_elements=None,
    num_rows=1,
    mode="positive",
    index_op=TopKXLIndexOp.RowMajor,
    group_id=0,
    group_shift=GROUP_SHIFT,
    core_id=0,
    sort_direction=TopKSortDirection.Descending,
    fused_reduce=False,
    chunk_base_mode=TopKXLChunkBaseMode.Static,
    chunk_base=0,
    dest_sync=DestSync.Full,
    formats=FORMATS,
    rebuild_mutate=0,
):
    """
    Build the stimulus and the TestConfig for one variant. Returns (config, rows).

    `tail_elements` defaults to K, i.e. full chunks with no -inf padding.
    RemoveMsb packs one region in place, every other index op packs two.
    The fused path splits indices at the end, so it packs two regions as well.
    """
    tail_elements = K if tail_elements is None else tail_elements
    tiles_per_seq = _tiles_per_sequence(K)
    result_tiles = (1 if index_op == TopKXLIndexOp.RemoveMsb else 2) * tiles_per_seq
    # 32-bit input does unpack_to_dest (is_32bit_input), while bf16 goes
    # through SrcA then the MATH A2D datacopy.
    is_32bit = formats.input_format in (DataFormat.Float32, DataFormat.Int32)
    # Unused dummy tile, but it is declared with the input format, so its dtype has
    # to follow it or the stimuli packer is handed a mislabeled tensor.
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=format_dict[formats.input_format])

    src_A, rows = _build_input(K, num_chunks, tail_elements, num_rows, mode, is_32bit)

    config = TestConfig(
        test_name="sources/topk_rebuild_full_macro_test.cpp",
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
                sort_direction=sort_direction,
                fused_reduce=fused_reduce,
                chunk_base_mode=chunk_base_mode,
                chunk_base=chunk_base,
            ),
            REBUILD_MUTATE(rebuild_mutate),
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
    return config, rows


def _run(K, **kwargs):
    """Build and run one variant. Returns (result, rows)."""
    config, rows = _variant(K, **kwargs)
    return config.run().result, rows


def _run_test_topk(K, compare_index_set=True, index_offset=0, **kwargs):
    """
    The whole flow for the variants that report a flat index: build the stimulus,
    run it, and check the result against the golden. `kwargs` go to `_variant`.
    """
    result, rows = _run(K, **kwargs)
    _check(
        result,
        K,
        rows,
        compare_index_set,
        index_offset=index_offset,
        value_format=kwargs.get("formats", FORMATS).input_format,
    )


def _extract_hw_topk(result, K, num_rows, num_regions=2):
    """
    Per row, pair value[j] with index[j] and drop bf16 -inf padding lanes.
    Returns a list of (index_words, values_float32): the surviving top-K.

    `num_regions` is 1 for the index ops that pack the fused word in place; the
    fused word doubles as the index word there.
    """
    res = torch.tensor(result, dtype=format_dict[FORMATS.output_format])
    region = _tiles_per_sequence(K) * ELEMENTS_PER_TILE
    per_row = num_regions * region
    assert (
        res.numel() == num_rows * per_row
    ), f"result size {res.numel()} != expected {num_rows * per_row}"

    out = []
    for r in range(num_rows):
        block = res[r * per_row : (r + 1) * per_row]
        value_words = block[:region]
        index_words = block[region:] if num_regions == 2 else value_words
        # Lane mask that drops the bf16 -inf padding the copy path writes.
        finite = ((value_words >> 16) & 0xFFFF) != BF16_NEG_INF_HI16
        out.append((index_words[finite], _bitcast_float32(value_words[finite])))
    return out


def _check_values(r, hw_val, gold_val, value_format=FORMATS.input_format):
    """The returned values are the top-K, as a multiset: Dest lane order is internal."""
    assert passed_test(
        torch.sort(gold_val).values, torch.sort(hw_val).values, value_format
    ), f"row {r}: top-K value mismatch"


def _check(
    result,
    K,
    rows,
    compare_index_set,
    index_offset=0,
    value_format=FORMATS.input_format,
):
    """
    Validate the packed result against the golden.

    Always: exactly K finite lanes per row, K distinct indices, the top-K value
    multiset matches, and each returned index points at the value returned
    alongside it. `compare_index_set` additionally requires the exact top-K index
    set; it is off for tie-prone inputs, where the tie-break makes the chosen
    indices ambiguous.

    `index_offset` is the starting chunk_base, which the LLK ORs into every
    reported index, so the golden positions shift by it.
    """
    num_rows = rows.shape[0]
    gold_indices = get_golden_generator(TopKXLGolden)(rows, K)
    gold_values = rows.gather(1, gold_indices)  # [num_rows, K] top-K values per row

    for r, (hw_idx, hw_val) in enumerate(_extract_hw_topk(result, K, num_rows)):
        assert (
            hw_idx.numel() == K
        ), f"row {r}: expected {K} finite top-K lanes, got {hw_idx.numel()}"

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

        _check_values(r, hw_val, gold_values[r], value_format)

        # Index <-> value pairing. Exact, because every stimulus here is exactly
        # representable in bf16 and the value word is [bf16 << 16 | 0].
        row = rows[r]
        for idx, val in zip(hw_idx_list, hw_val.tolist()):
            assert float(row[idx]) == float(
                val
            ), f"row {r}: index {idx} value {val} != input {float(row[idx])}"


def _check_coordinates(
    result,
    K,
    rows,
    num_chunks=1,
    group_id=0,
    group_shift=GROUP_SHIFT,
    core_id=None,
):
    """
    Validate the index ops that report a tile coordinate instead of a flat index:
    index word is [group_id<<shift | core_id<<11 | tile coordinate].

    Checks the group_id and core_id fields, that every coordinate identifies its
    element's position, and that the values are the top-K. With num_chunks == 1
    the coordinates must additionally be the full 0..K-1 set, since the top-K is
    then the whole row.

    `core_id=None` skips the core_id check: a group_shift of 11 or less puts the
    group id over bits [15:11], leaving no separable core bits to read back.
    """
    num_rows = rows.shape[0]
    for r, (hw_iw, hw_val) in enumerate(_extract_hw_topk(result, K, num_rows)):
        assert hw_iw.numel() == K, f"row {r}: expected {K} lanes, got {hw_iw.numel()}"

        group = hw_iw >> group_shift
        raw = hw_iw & ((1 << group_shift) - 1)
        assert (group == group_id).all(), (
            f"row {r}: group_id bits wrong at shift {group_shift}; got "
            f"{sorted(set(int(g) for g in group.tolist()))[:4]}, want {group_id}"
        )

        if core_id is not None:
            hw_core = (raw >> CORE_ID_SHIFT) & CORE_ID_MASK
            assert (hw_core == core_id).all(), (
                f"row {r}: core_id bits [15:11] wrong; got "
                f"{sorted(set(int(c) for c in hw_core.tolist()))[:4]}, want {core_id}"
            )

        pos = [_decode_row_major(int(x), K) for x in raw.tolist()]
        if num_chunks == 1:
            assert set(pos) == set(range(K)), f"row {r}: decoded positions != 0..K-1"
        else:
            assert all(0 <= p < K for p in pos), f"row {r}: coordinate outside 0..K-1"

        # A coordinate is per-chunk, so it must match its value in one of the
        # chunks. Exact, since `positive` is consecutive bf16 patterns, so any
        # tolerance would admit a coordinate several positions off.
        row = rows[r]
        for p, v in zip(pos, hw_val.tolist()):
            candidates = [float(row[c * K + p]) for c in range(num_chunks)]
            assert (
                float(v) in candidates
            ), f"row {r}: coordinate {p} value {v} matches no chunk ({candidates})"

        _check_values(r, hw_val, torch.topk(row, K).values)


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
def test_topk_xl(K, num_chunks, partial_tail):
    _run_test_topk(
        K,
        num_chunks=num_chunks,
        tail_elements=(K // 2) if partial_tail else K,
        num_rows=2,
    )


# "signed": distinct values spanning negatives and positives, so the top-K are the
# largest (positive) ones and negatives must sort below them. Exercises bf16 sign
# handling under the INT32 SFPSWAP compare; the index set is unambiguous.
# "random": raw bf16, so ties are likely and the tie-break makes the chosen indices
# ambiguous. Validate K distinct indices + the top-K value multiset only.
@parametrize(K=[512, 1024, 2048], mode=["signed", "random"])
def test_topk_xl_input_modes(K, mode):
    _run_test_topk(K, num_chunks=2, mode=mode, compare_index_set=(mode == "signed"))


@parametrize(K=[512, 1024, 2048])
def test_topk_xl_all_equal(K):
    """
    Every key is identical, so every compare-exchange in local_sort
    and rebuild takes the tie branch and resolves on the index bits.
    """
    (K,) = K

    result, rows = _run(K, mode="all_equal")  # single chunk
    hw_idx, hw_val = _extract_hw_topk(result, K, 1)[0]

    idx = sorted(int(i) for i in hw_idx.tolist())
    assert idx == list(range(K)), "all-equal row did not return 0..K-1"
    assert (hw_val == rows[0][0].item()).all(), "all-equal row returned a changed value"


# separate_indices splits the fused word into a value region and an index region.
# core_id lands in index bits [15:11] (5 bits, up to 32 cores) and the positions must
# come out unchanged. group_id_bit_shift is a runtime value saved in LREG12, so the
# group id can sit anywhere above the coordinate: 11 puts it immediately above the
# widest coordinate (and over the core_id field, so core_id must be 0), and 20 clears
# that field entirely.
@parametrize(
    K=[512, 1024, 2048],
    core_id=[0, 1, 31],
    group_shift=lambda core_id: [16, 11, 20] if core_id == 0 else [16],
)
def test_topk_xl_separate_indices(K, core_id, group_shift):
    result, rows = _run(
        K,
        index_op=TopKXLIndexOp.Separate,
        group_id=GROUP_ID,
        group_shift=group_shift,
        core_id=core_id,
    )
    _check_coordinates(
        result,
        K,
        rows,
        group_id=GROUP_ID,
        group_shift=group_shift,
        # Below bit 11 the group id covers the core_id field: nothing to read back.
        core_id=core_id if group_shift > CORE_ID_SHIFT else None,
    )


@parametrize(K=[512, 1024, 2048])
def test_topk_xl_remove_msb(K):
    """
    remove_msb_values: fused region -> [0 | raw], packed in place as one region.
    Check the value half is zeroed and the decoded positions form the full 0..K-1 set.
    """
    (K,) = K

    result, rows = _run(
        K, index_op=TopKXLIndexOp.RemoveMsb, group_id=GROUP_ID, group_shift=GROUP_SHIFT
    )
    for r, (fused, _) in enumerate(_extract_hw_topk(result, K, rows.shape[0], 1)):
        assert fused.numel() == K, f"row {r}: expected {K} lanes, got {fused.numel()}"
        untouched = int(torch.count_nonzero(fused >> 16))
        assert untouched == 0, f"row {r}: value half not zeroed in {untouched} lanes"

        pos = [_decode_row_major(int(x), K) for x in (fused & 0xFFFF).tolist()]
        assert set(pos) == set(
            range(K)
        ), f"row {r}: decoded positions are not the full 0..K-1 set"


# chunk_base is saved by one of three inits, based on a template argument:
# - init_static<hi, lo> takes both
# - init_upper<hi>(lo) only the high half
# - init(base) takes neither
# All three send an (upper16, lower16) pair to _sfpu_load_config32_, but the split varies.
# 0x1F800 is nonzero in both halves, and it's a multiple of every valid K.
@parametrize(K=[512, 1024, 2048], chunk_base_mode=list(TopKXLChunkBaseMode))
def test_topk_xl_chunk_base(K, chunk_base_mode):
    """Assert that the reported index is chunk_base + position."""
    chunk_base = 0x1F800

    _run_test_topk(
        K,
        num_chunks=2,
        num_rows=2,
        chunk_base_mode=chunk_base_mode,
        chunk_base=chunk_base,
        index_offset=chunk_base,
    )


def _positional_rank(values, descending):
    """
    rank[lane] = position of that lane's value in the sorted order.
    Requires distinct values.
    """
    order = torch.argsort(-values if descending else values, stable=True)
    rank = torch.empty(order.numel(), dtype=torch.int64)
    rank[order] = torch.arange(order.numel())
    return rank


@parametrize(K=[512, 1024, 2048])
def test_topk_xl_rebuild_ascending(K):
    """
    rebuild(..., ascending=true). `_topk_xl_merge_` has no direction argument and
    always keeps the max half, so the direction changes the order the survivors are
    rebuilt into, not which of them survive. Both runs must return the same top-K,
    and each lane must hold the same rank counted from the opposite end.

    The two directions are compared against each other rather than each checked
    alone, because rebuild does not leave Dest sorted in lane order.
    """
    (K,) = K
    num_rows = 1

    desc_cfg, rows = _variant(
        K, num_chunks=2, sort_direction=TopKSortDirection.Descending
    )
    asc_cfg, _ = _variant(K, num_chunks=2, sort_direction=TopKSortDirection.Ascending)
    # Build both before running either: `prepare()` is the build half of `run()`,
    # and under --compile-producer `run()` skips as soon as the first variant is
    # built, so the second would otherwise never emit its ELF.
    desc_cfg.prepare()
    asc_cfg.prepare()
    desc, asc = desc_cfg.run().result, asc_cfg.run().result

    _check(desc, K, rows, compare_index_set=True)
    _check(asc, K, rows, compare_index_set=True)

    for r, ((_, desc_val), (_, asc_val)) in enumerate(
        zip(
            _extract_hw_topk(desc, K, num_rows),
            _extract_hw_topk(asc, K, num_rows),
        )
    ):
        rank_desc = _positional_rank(desc_val, descending=True)
        rank_asc = _positional_rank(asc_val, descending=False)
        differs = rank_desc != rank_asc
        mismatch = int(torch.count_nonzero(differs))
        assert mismatch == 0, (
            f"row {r}: ascending rebuild is not the mirror of the descending one: "
            f"{mismatch}/{K} lanes hold a different rank; first at lane "
            f"{int(differs.nonzero()[0])}"
        )


@parametrize(K=[512, 1024, 2048])
def test_topk_xl_denormal_fused_word(K):
    """
    Assert a legal input containing +0 returns the correct set and indices.

    The +0 lanes must come back with their own coordinates intact. A +0 value
    zeroes bits 30:23 of the fused word, so the 32-bit word is an fp32 denormal
    and an FP32-mode move would flush it, taking the index with it.

    Runs fused, because that is where the value and the index share one word. On
    the unfused path they sit in separate regions, so a value word is plain
    0x00000000 and flushing it is a no-op.
    """
    (K,) = K

    result, rows = _run(
        K,
        num_chunks=2,
        mode="zeros_win",
        fused_reduce=True,
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
    )
    hw_iw, hw_val = _extract_hw_topk(result, K, 1)[0]
    assert hw_iw.numel() == K, f"expected {K} lanes, got {hw_iw.numel()}"

    row = rows[0]
    expected = {p for p in range(K) if float(row[p]) == 0.0}  # zeros live in chunk 0
    assert expected, "stimulus produced no zeros"

    zero_lane = hw_val == 0.0
    n_zero = int(torch.count_nonzero(zero_lane))
    assert n_zero == len(expected), (
        f"expected {len(expected)} zero-valued lanes in the top-K, got "
        f"{n_zero}; zeros should outrank every negative"
    )

    raw = hw_iw[zero_lane] & ((1 << GROUP_SHIFT) - 1)
    got = {_decode_row_major(int(x), K) for x in raw.tolist()}
    assert got == expected, (
        "index bits of the denormal-shaped fused words did not survive: "
        f"{len(got)} distinct coordinates for {len(expected)} zero lanes; "
        f"missing {sorted(expected - got)[:8]}, unexpected {sorted(got - expected)[:8]}"
        + ("  (all zero: the index was flushed)" if got == {0} else "")
    )


@parametrize(K=[512, 1024, 2048], num_chunks=[2, 4])
def test_topk_xl_fused_reduce(K, num_chunks):
    """
    merge/rebuild with fused=true: half the operand distance (there's no index
    region between the slots), 16 instead of 18 instructions in the MOP, and
    half the iteration count.
    """
    result, rows = _run(
        K,
        num_chunks=num_chunks,
        fused_reduce=True,
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
    )
    _check_coordinates(
        result, K, rows, num_chunks=num_chunks, group_id=GROUP_ID, core_id=None
    )


@parametrize(K=[512, 1024, 2048], partial_tail=[False, True])
def test_topk_xl_input_float32(K, partial_tail):
    """
    fp32 input does unpack_to_dest instead of A2D datacopy.

    `partial_tail` makes it so UNPACK and MATH have to agree on how to do the copy:
      - K=1024: the copy clears SrcA with -inf instead of doing ZEROACC
      - K=2048: the tail's second tile is empty, so both threads return early
    """
    _run_test_topk(
        K,
        num_chunks=2,
        tail_elements=(K // 2) if partial_tail else K,
        formats=InputOutputFormat(DataFormat.Float32, DataFormat.UInt32),
    )


# DestSync.Half halves the Dest budget (4 fp32 tiles), which the unfused merge tree
# fits only for K <= 1024: K=2048 needs two 2-tile value regions plus their index
# regions, so 8 tiles under fp32 SyncFull.
@parametrize(K=[512, 1024])
def test_topk_xl_dest_sync_half(K):
    (K,) = K

    _run_test_topk(K, num_chunks=2, num_rows=2, dest_sync=DestSync.Half)


# mutate=1 attacks the Simple slot (the scheduled SFPSWAP), mutate=2 attacks the
# Store slot (the scheduled SFPSTORE, which is what is NEW in this kernel over
# test_topk_rebuild_macro.py). mutate=2 is a no-op at K=512, where sub-blocks A
# and B are `if constexpr`'d away and no macro schedules a store that matters --
# so it is only asserted at K >= 1024.
@parametrize(K=[512, 1024, 2048], rebuild_mutate=lambda K: [1] if K == 512 else [1, 2])
def test_topk_xl_rebuild_macro_mutation(K, rebuild_mutate):
    """
    MUTATION CONTROL. Not a feature test -- a test of whether the tests above can
    see the thing they claim to prove.

    Two independent mutations, one per half of the trick.

    ``REBUILD_MUTATE=1`` clears bit 0x80 of the Simple sequence byte, so the
    macro assigns ``Insn.VC = macroVD`` instead of ``Insn.VB = macroVD``
    (SFPLOADMACRO.md:100) and the scheduled ``SFPSWAP`` becomes
    ``SFPSWAP(VC=macroVD, VD=macroVD)`` -- a compare of the freshly loaded value
    against itself, which writes nothing new. One full bitonic level of the
    rebuild silently disappears, in every lattice phase.

    ``REBUILD_MUTATE=2`` zeroes the Store sequence byte instead ("schedule
    nothing"), leaving the scheduled ``SFPSWAP`` intact. That is the control for
    what is NEW in this kernel: in sub-blocks A and B the macroVD half of every
    compare-exchange is written back to Dst BY THE MACRO and by nothing else, so
    without it half of each of those bodies' output stays stale. It is only
    asserted at K >= 1024, because at K = 512 sub-blocks A and B are
    ``if constexpr``'d away and every remaining macro store is a redundant write
    that the body's own closing ``store16_rows_x2`` overwrites anyway.

    WHAT THE ASSERTION HAS TO LOOK AT, AND WHY THE OBVIOUS CHOICE IS BLIND.
    The rebuild is a PERMUTATION of the K survivors -- it sorts them, it never
    drops one. So comparing the sorted SET of returned values across a mutated
    and an unmutated run of a SINGLE merge+rebuild is guaranteed to find nothing,
    no matter how broken the rebuild is. (Measured: it finds nothing.) Two things
    make the mutation visible, and this test uses both:

      * ``num_chunks=4`` puts three merge+rebuild pairs in series. A mis-ordered
        rebuild output is the next merge's sorted operand, and
        ``_topk_xl_merge_``'s elementwise max only yields the true top-K if that
        operand really is sorted -- so the error escapes into the returned SET.
      * The lane-order comparison below is sensitive to the permutation directly,
        which is what the mutation actually damages.

    If this test were to PASS -- i.e. if the mutated kernel still agreed with the
    unmutated one -- then nothing above establishes that the macro-scheduled
    compare-exchange does any work, and the perf number would be measuring a
    degenerate SFPLOADMACRO.
    """
    good_cfg, rows = _variant(
        K,
        num_chunks=4,
        fused_reduce=True,
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
    )
    bad_cfg, _ = _variant(
        K,
        num_chunks=4,
        fused_reduce=True,
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
        rebuild_mutate=rebuild_mutate,
    )
    # Build both before running either: `prepare()` is the build half of `run()`,
    # and under --compile-producer `run()` skips as soon as the first variant is
    # built, so the mutated one would otherwise never emit its ELF. Same reason
    # `test_topk_xl_rebuild_ascending` does it.
    good_cfg.prepare()
    bad_cfg.prepare()
    good, bad = good_cfg.run().result, bad_cfg.run().result

    # Sanity: the unmutated build of this exact variant must agree with the torch
    # golden, otherwise a red mutation arm would be evidence of nothing.
    _check_coordinates(good, K, rows, num_chunks=4, group_id=GROUP_ID, core_id=None)

    _, good_val = _extract_hw_topk(good, K, 1)[0]
    _, bad_val = _extract_hw_topk(bad, K, 1)[0]

    set_differs = int(
        torch.count_nonzero(torch.sort(good_val)[0] != torch.sort(bad_val)[0])
    )
    order_differs = int(torch.count_nonzero(good_val != bad_val))

    assert order_differs > 0 or set_differs > 0, (
        f"MUTATION {rebuild_mutate} NOT DETECTED: disabling the SFPLOADMACRO "
        f"sequence's {'Simple' if rebuild_mutate == 1 else 'Store'} slot drops "
        "a whole bitonic level (mutate=1) or half of sub-blocks A and B's "
        "writeback (mutate=2), yet the mutated run returns the same K values in "
        "the same lane order across three chained merge+rebuild pairs. The "
        "harness cannot see what this change touches, so the passing runs above "
        "do not establish that the macro is doing any work."
    )
