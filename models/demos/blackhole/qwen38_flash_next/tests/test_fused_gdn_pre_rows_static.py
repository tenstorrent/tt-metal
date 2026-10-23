# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_pre_rows`` without a device: the CB table against all six kernels, the kernel argument contracts against
``run``'s source, the LLK text pins of the two compute kernels (each phase's exact call from the chain-op mirror
table, the ``mac`` init/call adjacency, the two explicit ``APPROX = true`` calls the norm needs while the rest of the
kernel is ``APPROX = false``), the selection tiles against ``gdn_rows_reference.fir_taps``, the page maps against
``to_prim_qk`` / ``to_prim_vec`` and the prep reader's flat page map, the work split (coverage, consecutive runs and
the cost balance group A's cut is cut for), and the rows mask."""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_pre_rows as gp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_reference as ref
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

SOURCES = {role: (fp.REPO_ROOT / path).read_text() for role, path in gp.KERNELS.items()}
COMPUTE_QKV = SOURCES["compute_qkv"]
COMPUTE_GATES = SOURCES["compute_gates"]
RUN_SOURCE = re.sub(r"\s+", "", inspect.getsource(gp.run))


def _without_comments(source: str) -> str:
    """The kernel's code with its ``//`` comments dropped: the pins that say a call is ABSENT must not trip over a
    comment that cites the API form it replaces."""

    return "\n".join(re.sub(r"//.*$", "", line) for line in source.splitlines())


def _cb_constants(source: str) -> dict[str, int]:
    """The ``CB_<NAME> = <index>`` constants a kernel declares, lowercased to the Python table's names."""

    return {name.lower(): int(value) for name, value in re.findall(r"\bCB_([A-Z0-9_]+)\s*=\s*(\d+)", source)}


# ------------------------------------------------------------------------------------------------- the CB table


def test_cb_table_is_consistent_and_fits_the_hardware():
    indices = [index for _n, index, _d, _p, _g in gp.CBS]
    assert len(set(indices)) == len(indices)
    assert max(indices) < 32  # NUM_CIRCULAR_BUFFERS on every architecture the model runs on
    assert set(gp.CB_INDEX) == {name for name, *_rest in gp.CBS}
    groups = {group for *_rest, group in gp.CBS}
    assert groups == {"A", "B"}
    # the two groups' indices are disjoint, so the six kernels never collide on a buffer index
    a = {i for _n, i, _d, _p, g in gp.CBS if g == "A"}
    b = {i for _n, i, _d, _p, g in gp.CBS if g == "B"}
    assert not (a & b)
    assert gp.FP32_COPY_CBS == (gp.CB_INDEX["const"],)


def test_per_core_l1_stays_far_under_the_budget():
    assert gp.l1_bytes("A") == 190464 and gp.l1_bytes("B") == 61440
    assert gp.l1_bytes("A") + gp.l1_bytes("B") < 1024 * 1024  # neither group is near a Tensix's L1


@pytest.mark.parametrize("role", sorted(gp.KERNELS))
def test_kernel_cb_constants_match_the_python_table(role):
    declared = _cb_constants(SOURCES[role])
    assert declared, role
    for name, index in declared.items():
        assert name in gp.CB_INDEX, f"{role} declares CB_{name.upper()} which the Python table does not name"
        assert gp.CB_INDEX[name] == index, f"{role} CB_{name.upper()} = {index}, table says {gp.CB_INDEX[name]}"
    group = "A" if role.endswith("_qkv") else "B"
    for name in declared:
        assert (
            gp.CB_GROUP[name] == group
        ), f"{role} (group {group}) uses {name}, which belongs to group {gp.CB_GROUP[name]}"


# --------------------------------------------------------------------------------------- the argument contracts


@pytest.mark.parametrize(
    "role, args",
    [
        ("reader_qkv", gp.READER_QKV_ARGS),
        ("compute_qkv", gp.COMPUTE_QKV_ARGS),
        ("writer_qkv", gp.WRITER_QKV_ARGS),
        ("reader_gates", gp.READER_GATES_ARGS),
        ("compute_gates", gp.COMPUTE_GATES_ARGS),
        ("writer_gates", gp.WRITER_GATES_ARGS),
    ],
)
def test_fixed_runtime_args_are_read_in_order(role, args):
    literal = [int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", SOURCES[role])]
    assert literal == list(range(len(args))), role


def test_the_dataflow_kernels_take_the_chunk_count_and_chain_their_accessors():
    for role in ("writer_qkv", "writer_gates"):
        assert "get_compile_time_arg_val(0)" in SOURCES[role], role  # chunks (NC): the head-major page stride
    for role in ("reader_qkv", "reader_gates", "compute_qkv", "compute_gates"):
        # a projection page is c * 130 + column tile and the compute reads no page map at all
        assert "get_compile_time_arg_val" not in SOURCES[role], role
    for role, accessors, first in (
        ("reader_qkv", 8, 0),
        ("writer_qkv", 3, 1),
        ("reader_gates", 2, 0),
        ("writer_gates", 3, 1),
    ):
        source = SOURCES[role]
        assert f"TensorAccessorArgs<{first}>()" in source, role
        assert source.count("next_compile_time_args_offset()") == accessors - 1, role
    for role in ("compute_qkv", "compute_gates"):
        assert "TensorAccessorArgs" not in SOURCES[role], role


def test_run_builds_the_arguments_the_kernels_read():
    assert "fortensorin(projected,history,*taps,selects,scalars):" in RUN_SOURCE
    assert "reader_a_cta=[]" in RUN_SOURCE and "reader_b_cta=[*fp.accessor_args(projected)," in RUN_SOURCE
    assert (
        "reader_a_addrs=[projected.buffer_address(),history.buffer_address(),"
        "*(tap.buffer_address()fortapintaps),selects.buffer_address(),scalars.buffer_address(),int(rows),]"
        in RUN_SOURCE
    )
    assert "writer_a_cta=[chunks,*fp.accessor_args(q_c),*fp.accessor_args(k_c),*fp.accessor_args(v)]" in RUN_SOURCE
    assert "writer_a_addrs=[q_c.buffer_address(),k_c.buffer_address(),v.buffer_address(),int(rows)]" in RUN_SOURCE
    assert "reader_b_addrs=[projected.buffer_address(),constants.buffer_address()]" in RUN_SOURCE
    assert "writer_b_cta=[chunks,*fp.accessor_args(beta_c),*fp.accessor_args(g_c),*fp.accessor_args(sig)]" in RUN_SOURCE
    assert "writer_b_addrs=[beta_c.buffer_address(),g_c.buffer_address(),sig.buffer_address(),int(rows)]" in RUN_SOURCE
    # both compute kernels take (units, then the per-unit pairs) and no compile-time args
    assert "[(w.core,[w.count,*pairs(split.a_units,w)])forwinsplit.a_work]" in RUN_SOURCE
    assert "[(w.core,[w.count,*pairs(split.b_units,w)])forwinsplit.b_work]" in RUN_SOURCE
    assert RUN_SOURCE.count('KERNELS["compute_qkv"],split.a_cores,[],') == 1
    assert RUN_SOURCE.count('KERNELS["compute_gates"],split.b_cores,[],') == 1


def test_the_two_compute_kernels_take_the_dest_widths_of_the_ops_they_mirror():
    assert "fp32_dest=False" in RUN_SOURCE and "fp32_dest=True" in RUN_SOURCE
    assert "approx=False" in RUN_SOURCE and RUN_SOURCE.count("approx=False") == 2
    assert 'defines=[("INP_FLOAT32","1")]' in RUN_SOURCE
    assert "unpack_to_dest_fp32=FP32_COPY_CBS" in RUN_SOURCE
    # the fp32-dest options belong to group B only: group A's compute never asks for them
    a_block = RUN_SOURCE[RUN_SOURCE.index('KERNELS["compute_qkv"]') : RUN_SOURCE.index('KERNELS["writer_qkv"]')]
    assert "fp32_dest=False" in a_block and "unpack_to_dest_fp32" not in a_block and "INP_FLOAT32" not in a_block


# ------------------------------------------------------------------------------------------------ the LLK pins


def test_the_input_window_is_pushed_and_popped_whole():
    """cb_api.h: the pushes and pops of one CB cycle must sum to the CB size, and a multi-tile read is only
    contiguous while the window does not straddle the buffer's end.  CB_IN carries an 8-tile window on 16 pages and
    moves it as one block -- a 4-tile slide on 12 pages read four tiles past the end every third unit of a run."""

    reader, compute = SOURCES["reader_qkv"], COMPUTE_QKV
    assert (
        "constexpr uint32_t WINDOW = 2 * HEAD_TILES;" in reader
        and "constexpr uint32_t WINDOW = 2 * HEAD_TILES;" in compute
    )
    assert "cb_reserve_back(CB_IN, WINDOW);" in reader and "cb_push_back(CB_IN, WINDOW);" in reader
    assert reader.count("cb_reserve_back(CB_IN") == 1 and reader.count("cb_push_back(CB_IN") == 1
    assert "cb_wait_front(CB_IN, WINDOW);" in compute and "cb_pop_front(CB_IN, WINDOW);" in compute
    assert compute.count("cb_pop_front(CB_IN") == 1
    pages = dict((name, pages) for name, _i, _d, pages, _g in gp.CBS)["in"]
    assert pages % (2 * gp.HEAD_TILES) == 0 and pages >= 2 * (2 * gp.HEAD_TILES)


def test_conv_mirror_is_the_chains_multiply_mac_and_silu():
    # ttnn.multiply(tap0, w0): the tap row materialised by unary_bcast<ROW>, then the binary_ng SFPU multiply
    assert "unary_bcast_init<BroadcastType::ROW>(CB_TAP);" in COMPUTE_QKV
    assert "unary_bcast<BroadcastType::ROW>(CB_TAP, t, dst0);" in COMPUTE_QKV
    assert "mul_binary_tile_init();" in COMPUTE_QKV and "mul_binary_tile(dst0, dst1, dst0);" in COMPUTE_QKV
    # ttnn.mac: the ternary SFPU kernel, and mac_tile only ever immediately after its init
    init, call = "mac_tile_init<DataFormat::Float16_b>();", "mac_tile<DataFormat::Float16_b>(0, 1, 2, 0);"
    assert COMPUTE_QKV.count(init) == 1 and COMPUTE_QKV.count(call) == 1
    between = COMPUTE_QKV[COMPUTE_QKV.index(init) + len(init) : COMPUTE_QKV.index(call)]
    assert between.strip() == "", f"something sits between mac_tile_init and mac_tile: {between!r}"
    # ttnn.silu: the unary SFPU kernel's own init and call
    assert "silu_tile_init();" in COMPUTE_QKV and "silu_tile(dst0);" in COMPUTE_QKV


def test_fir_selection_is_two_matmuls_into_one_dest_tile():
    assert "reconfig_data_format<SrcOrder::Reverse>(CB_SEL, CB_IN);" in COMPUTE_QKV
    assert "matmul_init(CB_SEL, CB_IN);" in COMPUTE_QKV
    assert "matmul_tiles(CB_SEL, CB_IN, before, d, dst0);" in COMPUTE_QKV
    assert "matmul_tiles(CB_SEL, CB_IN, select_index(SELECT_CUR, shift), HEAD_TILES + d, dst0);" in COMPUTE_QKV
    assert "compute_kernel_hw_startup<SrcOrder::Reverse>(CB_SEL, CB_IN, CB_SHIFT);" in COMPUTE_QKV


def test_norm_mirror_is_the_layernorm_rmsnorm_sequence():
    assert "mul_init(CB_X, CB_X);" in COMPUTE_QKV and "mul_tiles(CB_X, CB_X, i, i, i);" in COMPUTE_QKV
    assert "reconfig_data_format(CB_SCALER, CB_XMM2);" in COMPUTE_QKV  # the scaler in SrcA, the data in SrcB
    assert "reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XMM2, CB_SCALER, CB_EX2);" in COMPUTE_QKV
    assert "reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XMM2, CB_SCALER, j, 0, dst0);" in COMPUTE_QKV
    assert "add_init(CB_EX2, CB_EPS);" in COMPUTE_QKV and "add_tiles(CB_EX2, CB_EPS, 0, 0, dst0);" in COMPUTE_QKV
    assert "mul_bcast_cols_init(CB_X, CB_EX2PE);" in COMPUTE_QKV
    assert "mul_tiles_bcast_cols(CB_X, CB_EX2PE, i, 0, i);" in COMPUTE_QKV
    # numeric.h's real order: reduce_uninit, then the reconfig, then the 1/W scale
    order = [COMPUTE_QKV.index(text) for text in ("reduce_uninit();", "binop_with_scalar_tile_init();")]
    assert order == sorted(order)
    assert COMPUTE_QKV.index("reduce_uninit();") < COMPUTE_QKV.index("calculate_binop_with_scalar")


def test_the_two_norm_calls_are_written_with_the_ops_approx_true():
    """The layernorm op is compiled ``math_approx_mode = true`` and this kernel ``APPROX = false``, so its 1/W scale
    and its rsqrt are the only two calls written out through the SFPU macros with ``true`` substituted for APPROX."""

    scale = re.search(
        r"MATH\(SFPU_UNARY_CALL\(\s*DST_SYNC_MODE,\s*false /\* is_fp32_dest_acc_en \*/,\s*"
        r"calculate_binop_with_scalar,\s*\(true /\* APPROX \*/, MUL_UNARY, 8 /\* ITERATIONS \*/, "
        r"false /\* is_fp32_dest_acc_en \*/\),\s*dst0,\s*VectorMode::RC,\s*RECIP_W\)\);",
        COMPUTE_QKV,
    )
    assert scale, "the 1/W scale is not the APPROX = true binop_with_scalar call"
    assert "MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::rsqrt_init, (true /* APPROX */)));" in (COMPUTE_QKV)
    rsqrt = re.search(
        r"MATH\(SFPU_UNARY_CALL\(\s*DST_SYNC_MODE,\s*false /\* is_fp32_dest_acc_en \*/,\s*calculate_rsqrt,\s*"
        r"\(true /\* APPROX \*/,\s*8 /\* ITERATIONS \*/,\s*false /\* is_fp32_dest_acc_en \*/,\s*"
        r"false /\* FAST_APPROX \*/\),\s*dst0,\s*VectorMode::RC\)\);",
        COMPUTE_QKV,
    )
    assert rsqrt, "the rsqrt is not the APPROX = true calculate_rsqrt call"
    assert "constexpr uint32_t RECIP_W = 0x3C000000;" in COMPUTE_QKV  # 1 / 128, the norm's W
    # no other call smuggles an APPROX override in, and the plain API forms are not used for these two
    assert COMPUTE_QKV.count("true /* APPROX */") == 3  # the scale, the rsqrt init, the rsqrt call
    code = _without_comments(COMPUTE_QKV)
    assert "rsqrt_tile" not in code and "mul_unary_tile" not in code


def test_v_runs_the_chains_own_row_mask_multiply():
    """``ttnn.multiply(v_slice, row_mask_bf16_col)`` is binary_ng's column-broadcast SFPU kernel.  It is what masks
    v's rows past ``rows`` and what canonicalises a ``-0.0`` out of the SiLU, so the mirror runs it on every v tile
    rather than writing the SiLU output as it is."""

    assert "unary_bcast_init<BroadcastType::COL>(CB_MASK);" in COMPUTE_QKV
    assert "unary_bcast<BroadcastType::COL>(CB_MASK, 0, dst0);" in COMPUTE_QKV
    assert "pack_tile(dst0, CB_MASKFULL);" in COMPUTE_QKV
    assert "copy_tile(CB_MASKFULL, 0, dst1);" in COMPUTE_QKV
    assert COMPUTE_QKV.count("v_row_mask();") == 1  # every v unit, and only the v units
    body = COMPUTE_QKV[COMPUTE_QKV.index("ALWI void v_row_mask()") :]
    assert body.index("mul_binary_tile(dst0, dst1, dst0);") < body.index("cb_push_back(CB_OUT, HEAD_TILES);")
    # the reader builds the mask column for the unit's tile row and takes `rows` to do it
    reader = SOURCES["reader_qkv"]
    assert "void build_row_mask(uint32_t cb, uint32_t keep)" in reader
    assert "elements[face_element(row, 0)] = BF16_ONE;" in reader
    assert "constexpr uint16_t BF16_ONE = 0x3F80;" in reader
    assert "build_row_mask(CB_MASK, keep);" in reader
    # q and k keep the writer's zeroing (their chain mask is a whole-tile scalar multiply); v does not
    writer = SOURCES["writer_qkv"]
    assert "if (group < QK_GROUPS) {\n            if (chunk > last_chunk) {" in writer
    assert writer.count("zero_rows_bf16(") == 3  # the definition plus the two q/k arms


def test_gate_mirror_is_the_fp32_chain():
    assert "sigmoid_tile_init<false>();" in COMPUTE_GATES
    assert "sigmoid_tile<VectorMode::RC, false>(dst0);" in COMPUTE_GATES
    assert "add_binary_tile_init();" in COMPUTE_GATES
    assert "add_binary_tile<ckernel::DstRoundingMode::NearestEven>(dst0, dst1, dst0);" in COMPUTE_GATES
    assert "softplus_tile_init();" in COMPUTE_GATES
    assert "softplus_tile(dst0, F_ONE, F_ONE, F_TWENTY);" in COMPUTE_GATES
    assert "constexpr uint32_t F_ONE = 0x3F800000u, F_TWENTY = 0x41A00000u;" in COMPUTE_GATES
    assert "mul_binary_tile_init();" in COMPUTE_GATES and "mul_binary_tile(dst0, dst1, dst0);" in COMPUTE_GATES
    assert (
        "typecast_tile_init<DF_FP32, DF_BF16>();" in COMPUTE_GATES
        and "typecast_tile<DF_FP32, DF_BF16>(dst0);" in COMPUTE_GATES
    )
    assert "constexpr uint32_t DF_FP32 = (uint32_t)DataFormat::Float32, DF_BF16 = (uint32_t)DataFormat::Float16_b;" in (
        COMPUTE_GATES
    )
    # the a/b unit's typecast to fp32 is a copy_tile: the packer does bf16 -> fp32, no SFPU op
    assert "copy_tile(CB_AB, B_INDEX, dst0);" in COMPUTE_GATES and "copy_tile(CB_AB, A_INDEX, dst0);" in COMPUTE_GATES


def test_no_dest_width_dependent_call_lands_in_the_wrong_kernel():
    for wrong in ("softplus_tile", "typecast_tile", "sigmoid_tile", "add_binary_tile"):
        assert wrong not in COMPUTE_QKV, f"{wrong} is a 32-bit-DEST op and belongs in compute_gates"
    for wrong in ("mac_tile", "silu_tile", "reduce_tile", "matmul_tiles", "rsqrt", "mul_bcast_cols_init"):
        assert wrong not in COMPUTE_GATES, f"{wrong} is a 16-bit-DEST op and belongs in compute_qkv"


# ----------------------------------------------------------------------------------- the host-built constants


def _rows(chunks: int, seed: int = 5):
    generator = torch.Generator().manual_seed(seed)
    total = chunks * gp.TILE
    qkv = (torch.randn(total, gp.QKV_WIDTH, generator=generator) * 0.6).to(torch.bfloat16)
    history = torch.zeros(gp.TILE, gp.QKV_WIDTH)
    history[: gp.HISTORY_ROWS] = torch.randn(gp.HISTORY_ROWS, gp.QKV_WIDTH, generator=generator) * 0.6
    return qkv, history.to(torch.bfloat16)


@pytest.mark.parametrize("chunks", [1, 2, 5])
def test_selection_tiles_reproduce_the_references_fir_taps(chunks):
    """``Sel_prev_or_hist(s) @ previous + Sel_cur(s) @ current`` per tile row is the reference's row shift, tile row 0
    included (where the previous tile is the history tile)."""

    qkv, history = _rows(chunks)
    selects = gp.selection_tiles().float()
    taps = ref.fir_taps(qkv, history)  # tap 0 .. tap 3, each [T, 2560]
    for chunk in range(chunks):
        current = qkv[chunk * gp.TILE : (chunk + 1) * gp.TILE].float()
        previous = history.float() if chunk == 0 else qkv[(chunk - 1) * gp.TILE : chunk * gp.TILE].float()
        before_kind = gp.SELECT_HIST if chunk == 0 else gp.SELECT_PREV
        for tap in range(gp.HISTORY_ROWS):
            shift = gp.HISTORY_ROWS - tap  # tap 0 is the shift by 3, tap 2 the shift by 1
            built = (
                selects[gp.select_index(before_kind, shift)] @ previous
                + selects[gp.select_index(gp.SELECT_CUR, shift)] @ current
            )
            expected = taps[tap][chunk * gp.TILE : (chunk + 1) * gp.TILE].float()
            assert torch.equal(built, expected), f"tap {tap} of tile row {chunk}"
        # tap 3 is the rows themselves: no selection at all
        assert torch.equal(taps[3][chunk * gp.TILE : (chunk + 1) * gp.TILE].float(), current)


def test_selection_tiles_are_exact_zero_one_matrices_with_one_term_per_row():
    tiles = gp.selection_tiles()
    assert tiles.shape == (gp.SELECT_TILES, gp.TILE, gp.TILE) and tiles.dtype == torch.bfloat16
    values = tiles.float()
    assert set(values.unique().tolist()) <= {0.0, 1.0}
    for shift in range(1, gp.HISTORY_ROWS + 1):
        cur = values[gp.select_index(gp.SELECT_CUR, shift)]
        prev = values[gp.select_index(gp.SELECT_PREV, shift)]
        hist = values[gp.select_index(gp.SELECT_HIST, shift)]
        # exactly one 1.0 per output row, split between the "before" tile and the current one
        assert torch.equal(cur.sum(dim=1) + prev.sum(dim=1), torch.ones(gp.TILE))
        assert torch.equal(cur.sum(dim=1) + hist.sum(dim=1), torch.ones(gp.TILE))
        assert float(prev[shift:].sum()) == 0.0 and float(cur[:shift].sum()) == 0.0


def test_scalar_tiles_reproduce_the_dataflow_helpers_fills():
    tiles = gp.scalar_tiles()
    assert tiles.shape == (gp.SCALAR_TILES, gp.TILE, gp.TILE) and tiles.dtype == torch.bfloat16
    words = tiles.view(torch.int16).to(torch.int64) & 0xFFFF

    # calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>: row 0 of each of the four faces = 1.0, the rest zero
    reduce = words[gp.SCALAR_REDUCE]
    expected = torch.zeros(gp.TILE, gp.TILE, dtype=torch.int64)
    expected[0, :] = 0x3F80
    expected[gp.TILE // 2, :] = 0x3F80
    assert torch.equal(reduce, expected)
    assert float(tiles[gp.SCALAR_REDUCE].float().sum()) == 2 * gp.TILE  # 64 ones, nothing else

    # generate_bcast_col_scalar: bits >> 16 (TRUNCATED, not rounded) in column 0 of all 32 rows
    eps = words[gp.SCALAR_EPS]
    assert torch.equal(eps[:, 0], torch.full((gp.TILE,), gp.EPS_BITS >> 16, dtype=torch.int64))
    assert float(eps[:, 1:].sum()) == 0.0
    truncated = ref.bf16_trunc(torch.tensor([gp.NORM_EPS], dtype=torch.float32))
    assert torch.equal(tiles[gp.SCALAR_EPS, :, 0], truncated.expand(gp.TILE))
    assert gp.EPS_BITS == 0x320637BD  # 1e-6 / 128 as fp32; the tile keeps its top 16 bits, 0x3206

    # fill_with_val_bfloat16: bfloat16(128 ** -0.5) = 0x3DB5 in every element, NOT the fp32 0x3DB504F3
    assert gp.SCALE_BF16_BITS == 0x3DB5
    assert torch.equal(words[gp.SCALAR_SCALE], torch.full((gp.TILE, gp.TILE), 0x3DB5, dtype=torch.int64))
    assert torch.equal(tiles[gp.SCALAR_SCALE], torch.tensor(gp.QK_SCALE).to(torch.bfloat16).expand(gp.TILE, gp.TILE))


def test_constant_row_tiles_repeat_the_row_in_every_row():
    dt_bias = torch.arange(gp.HEADS).float() * 0.25 - 1.0
    neg_exp_A = -torch.exp(torch.arange(gp.HEADS).float() * 0.1)
    tiles = gp.constant_row_tiles(dt_bias, neg_exp_A)
    assert tiles.shape == (gp.CONST_TILES, gp.TILE, gp.TILE) and tiles.dtype == torch.float32
    for index, row in ((gp.CONST_DT_BIAS, dt_bias), (gp.CONST_NEG_EXP_A, neg_exp_A)):
        assert torch.equal(tiles[index, :, : gp.HEADS], row.reshape(1, -1).expand(gp.TILE, gp.HEADS))
        assert float(tiles[index, :, gp.HEADS :].abs().sum()) == 0.0  # the 20 unused columns stay zero


# ------------------------------------------------------------------------------------------------ the page maps


@pytest.mark.parametrize("chunks", [1, 2, 64])
def test_qk_and_vec_page_maps_are_the_references_prim_layouts(chunks):
    """Reading the pages the writer names out of a ``[12, NC, 32, 128]`` / ``[12, NC, 32, 1]`` tile grid must give
    ``to_prim_qk`` / ``to_prim_vec`` of a token-major index tensor."""

    total = chunks * gp.TILE
    index = torch.arange(total * gp.HEADS * gp.HEAD_DIM).reshape(total, gp.HEADS, gp.HEAD_DIM).float()
    prim = ref.to_prim_qk(index)  # [12, NC, 32, 128]
    for head in (0, 5, gp.HEADS - 1):
        for chunk in (0, chunks // 2, chunks - 1):
            for column in range(gp.HEAD_TILES):
                page = gp.qk_page(head, chunk, column, chunks)
                # the tile grid of [12, NC, 32, 128] is 12 * NC row tiles of 4 column tiles each
                assert page == (head * chunks + chunk) * gp.HEAD_TILES + column
                block = prim[head, chunk, :, column * gp.TILE : (column + 1) * gp.TILE]
                rows = torch.arange(chunk * gp.TILE, (chunk + 1) * gp.TILE)
                columns = torch.arange(column * gp.TILE, (column + 1) * gp.TILE)
                assert torch.equal(block, index[rows][:, head][:, columns])
    pages = [gp.qk_page(h, c, d, chunks) for h in range(gp.HEADS) for c in range(chunks) for d in range(gp.HEAD_TILES)]
    assert sorted(pages) == list(range(gp.HEADS * chunks * gp.HEAD_TILES))

    vector = torch.arange(total * gp.HEADS).reshape(total, gp.HEADS).float()
    prim_vec = ref.to_prim_vec(vector)  # [12, NC, 32, 1]
    for head in range(gp.HEADS):
        for chunk in range(chunks):
            assert gp.vec_page(head, chunk, chunks) == head * chunks + chunk
            rows = torch.arange(chunk * gp.TILE, (chunk + 1) * gp.TILE)
            assert torch.equal(prim_vec[head, chunk, :, 0], vector[rows, head])
    vec_pages = [gp.vec_page(h, c, chunks) for h in range(gp.HEADS) for c in range(chunks)]
    assert sorted(vec_pages) == list(range(gp.HEADS * chunks))


@pytest.mark.parametrize("chunks", [1, 2, 64])
def test_flat_page_map_is_the_prep_readers_token_major_address(chunks):
    """``v`` and ``sig`` keep the chain's token-major ``[1, 1, T, 1536]``: tile (tile row c, column 4 hv + d).  That is
    the prep reader's flat-v page ``(c * Ct + rt) * HV * Vt + hv * Vt + ct`` at ``Ct = 1``, ``HV = 12``, ``Vt = 4``."""

    pages = []
    for chunk in range(chunks):
        for head in range(gp.HEADS):
            for column in range(gp.HEAD_TILES):
                page = gp.flat_page(chunk, head, column)
                assert page == chunk * (gp.HEADS * gp.HEAD_TILES) + head * gp.HEAD_TILES + column
                pages.append(page)
    assert sorted(pages) == list(range(chunks * gp.VALUE_TILES))


def test_group_column_tiles_cover_the_projections_qkv_columns_once():
    seen = []
    for group in range(gp.GROUPS):
        first = gp.group_first_tile(group)
        seen.extend(range(first, first + gp.HEAD_TILES))
        kind = gp.group_kind(group)
        heads = gp.group_value_heads(group)
        if kind == "v":
            assert first >= gp.V_TILE0 and len(heads) == 1
        else:
            assert len(heads) == gp.HEADS // gp.QK_HEADS  # the GQA expand: one key head feeds three value heads
            assert all(head // (gp.HEADS // gp.QK_HEADS) == heads[0] // (gp.HEADS // gp.QK_HEADS) for head in heads)
    assert sorted(seen) == list(range(gp.QKV_TILES))  # every q|k|v column tile of a tile row, exactly once
    assert [gp.group_kind(g) for g in range(gp.GROUPS)] == ["q"] * 4 + ["k"] * 4 + ["v"] * 12
    # a value head reads the key head the reference's gqa_expand gives it
    expanded = ref.gqa_expand(torch.arange(gp.QK_WIDTH).float().reshape(1, gp.QK_WIDTH))
    for group in range(gp.QK_HEADS):
        for head in gp.group_value_heads(group):
            assert int(expanded[0, head, 0]) == group * gp.HEAD_DIM


# ----------------------------------------------------------------------------------------------- the work split


def _mesh(x: int = 13, y: int = 10):
    return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=x, y=y))


@pytest.mark.parametrize("chunks", [1, 2, 64])
def test_the_split_covers_every_unit_exactly_once(chunks):
    mesh = _mesh()
    split = gp.split_units(mesh, chunks)
    assert split.a_units == gp.group_a_units(chunks) and split.b_units == gp.group_b_units(chunks)
    assert len(split.a_units) == gp.GROUPS * chunks and len(split.b_units) == gp.GATE_UNITS * chunks
    for work, units in ((split.a_work, split.a_units), (split.b_work, split.b_units)):
        covered = [u for w in work for u in units[w.start : w.start + w.count]]
        assert covered == units  # contiguous runs, in order, every unit once
        assert sum(w.count for w in work) == len(units)
        assert all(w.count >= 1 for w in work)
        assert len({(w.core.x, w.core.y) for w in work}) == len(work)  # one run per core


@pytest.mark.parametrize("chunks", [1, 2, 64])
def test_group_a_runs_are_consecutive_tile_rows_of_one_column_group(chunks):
    """Column-major order is what lets a core keep the previous tile row in L1: within a run the column group only
    ever changes at a tile row 0, and the tile row otherwise increments by one."""

    split = gp.split_units(_mesh(), chunks)
    for w in split.a_work:
        run = split.a_units[w.start : w.start + w.count]
        for (group, chunk), (next_group, next_chunk) in zip(run, run[1:]):
            if next_group == group:
                assert next_chunk == chunk + 1
            else:
                assert next_group == group + 1 and next_chunk == 0
    assert split.a_units[: chunks + 1] == [(0, c) for c in range(chunks)] + [(1, 0)]


def test_the_two_groups_sit_on_disjoint_core_rectangles():
    mesh = _mesh()
    split = gp.split_units(mesh, 64)
    a = {(w.core.x, w.core.y) for w in split.a_work}
    b = {(w.core.x, w.core.y) for w in split.b_work}
    assert not (a & b)
    assert max(x for x, _y in a) < min(x for x, _y in b)  # group B takes the trailing grid columns
    assert len(split.a_cores.ranges()) <= 2 and len(split.b_cores.ranges()) <= 2  # one kernel group per range
    assert len(b) == gp.GROUP_B_COLUMNS * 10
    tight = gp.split_units(mesh, 64, group_b_cores=20)
    assert len({(w.core.x, w.core.y) for w in tight.b_work}) == 20
    assert max(x for x, _y in {(w.core.x, w.core.y) for w in tight.a_work}) < 11


def test_the_split_refuses_to_starve_group_a(expect_error):
    with expect_error(ValueError):
        gp.split_units(_mesh(x=1, y=4), 8)
    with expect_error(ValueError):
        gp.split_units(_mesh(), 0)


# ------------------------------------------------------------------------------- group A's cost-weighted runs


def _loads(weights, work) -> list[int]:
    """The weighted load of every core's run."""

    return [sum(weights[w.start : w.start + w.count]) for w in work]


def _even_work(mesh, units: int, cores: int):
    """The count-even cut group A had before the weights: the same core order, the same consecutive runs."""

    return gp._column_work(units, mesh.compute_with_storage_grid_size(), 0, cores)


@pytest.mark.parametrize("chunks", [1, 7, 64])
def test_the_unit_weights_line_up_with_the_unit_list(chunks):
    weights = gp.group_a_weights(chunks)
    units = gp.group_a_units(chunks)
    assert len(weights) == len(units)
    assert weights == [gp.UNIT_COST[gp.group_kind(group)] for group, _chunk in units]
    blocks = gp.group_a_weight_blocks(chunks)
    assert [weight for weight, count in blocks for _ in range(count)] == weights
    # column-major order puts the kinds in runs, so group A is three blocks at every NC
    assert [count for _weight, count in blocks] == [gp.QK_HEADS * chunks, gp.QK_HEADS * chunks, gp.HEADS * chunks]
    assert gp.UNIT_COST["q"] > gp.UNIT_COST["k"] > gp.UNIT_COST["v"] > 0  # the measured order


@pytest.mark.parametrize("grid_x", [11, 13])
@pytest.mark.parametrize("chunks", [1, 2, 8, 16, 64])
def test_group_a_is_cut_by_cost_as_low_as_a_consecutive_cut_can_be(chunks, grid_x):
    """Every unit is placed once (the coverage test above), no core is loaded past the mean run plus one unit, and
    the heaviest run is the lightest a consecutive cut can make it: one hundredth of a microsecond less needs more
    runs than the split has cores."""

    mesh = _mesh(x=grid_x)
    split = gp.split_units(mesh, chunks)
    weights = gp.group_a_weights(chunks)
    loads = _loads(weights, split.a_work)
    cores = len(split.a_work)
    assert sum(loads) == sum(weights) and len(loads) == cores
    assert max(loads) <= -(-sum(weights) // cores) + max(gp.UNIT_COST.values())
    assert gp._runs_needed(gp.group_a_weight_blocks(chunks), max(loads) - 1) > cores


@pytest.mark.parametrize("grid_x", [11, 13])
@pytest.mark.parametrize("chunks", [1, 2, 8, 16, 64])
def test_the_cost_cut_never_loses_to_the_count_cut(chunks, grid_x):
    mesh = _mesh(x=grid_x)
    split = gp.split_units(mesh, chunks)
    weights = gp.group_a_weights(chunks)
    even = _even_work(mesh, len(weights), len(split.a_work))
    assert max(_loads(weights, split.a_work)) <= max(_loads(weights, even))


def test_one_tile_row_is_the_count_cut_core_for_core():
    """At NC = 1 -- the MTP verify tile -- group A is one unit per column group on as many cores, so the cost cut
    and the count cut are the same split and the verify path's program is untouched."""

    for mesh in (_mesh(), _mesh(x=11)):
        split = gp.split_units(mesh, 1)
        even = _even_work(mesh, len(split.a_units), len(split.a_work))
        assert len(split.a_work) == gp.GROUPS
        assert [(w.core.x, w.core.y, w.start, w.count) for w in split.a_work] == [
            (w.core.x, w.core.y, w.start, w.count) for w in even
        ]


def test_the_slab_cut_is_the_one_that_was_measured():
    """T = 2048 on the 90 group A cores of a 11x10 grid: the count cut waits for a 15-unit q run (393.15 hundredths
    of a microsecond of modelled core time), the cost cut for a 16-unit v run (336.48)."""

    mesh = _mesh(x=11)
    split = gp.split_units(mesh, 64)
    weights = gp.group_a_weights(64)
    assert len(split.a_work) == 90 and len(split.b_work) == 20
    assert max(_loads(weights, _even_work(mesh, len(weights), 90))) == 39315
    assert max(_loads(weights, split.a_work)) == 33648


@pytest.mark.parametrize("chunks", [8, 64])
def test_no_core_rebuilds_the_taps_more_often_than_a_consecutive_cut_must(chunks):
    """A tap re-read is a core starting a column group (``reader_qkv`` / ``compute_qkv`` do it whenever the group
    changes, not only on the first unit).  A consecutive cut pays one per core plus one per internal group boundary
    it does not land on, which is the least any cut into this many runs can pay."""

    split = gp.split_units(_mesh(x=11), chunks)
    events = 0
    for w in split.a_work:
        previous = None
        for group, _chunk in split.a_units[w.start : w.start + w.count]:
            events += group != previous
            previous = group
    assert events <= len(split.a_work) + gp.GROUPS - 1


def test_the_run_cut_handles_its_edges():
    assert gp._run_counts([(5, 4)], 1) == [4]
    assert gp._run_counts([(5, 4)], 4) == [1, 1, 1, 1]
    assert gp._run_counts([(5, 3)], 7) == [1, 1, 1]  # more cores than units: one each, no empty run
    assert sum(gp._run_counts([(7, 5), (1, 20)], 6)) == 25  # every unit placed
    assert gp._runs_needed([(7, 5), (1, 20)], 6) > 25  # a limit under the heaviest unit is impossible, not a hang
    assert gp._runs_needed([(7, 5), (1, 20)], 7) == 8  # five runs of one 7, then the twenty 1s in three


# ------------------------------------------------------------------------------------------------ the rows mask


def _case(chunks: int, seed: int = 11):
    generator = torch.Generator().manual_seed(seed)
    total = chunks * gp.TILE
    projected = torch.zeros(total, gp.PROJECTION_WIDTH)
    projected[:, : gp.A_COLUMN] = torch.randn(total, gp.A_COLUMN, generator=generator) * 0.6
    projected[:, gp.A_COLUMN : gp.A_COLUMN + gp.HEADS] = torch.randn(total, gp.HEADS, generator=generator) * 1.5 - 1.0
    projected[:, gp.B_COLUMN : gp.B_COLUMN + gp.HEADS] = torch.randn(total, gp.HEADS, generator=generator) * 1.5
    history = torch.zeros(gp.TILE, gp.QKV_WIDTH)
    history[: gp.HISTORY_ROWS] = torch.randn(gp.HISTORY_ROWS, gp.QKV_WIDTH, generator=generator) * 0.6
    return (
        projected.to(torch.bfloat16),
        history.to(torch.bfloat16),
        (torch.randn(gp.CONV_KERNEL, gp.QKV_WIDTH, generator=generator) * 0.5).to(torch.bfloat16),
        torch.randn(gp.HEADS, generator=generator) * 0.5,
        -torch.exp(torch.rand(gp.HEADS, generator=generator) * 3.0),
    )


def test_reference_returns_the_six_outputs_in_the_device_layouts():
    chunks = 2
    out = gp.reference(*_case(chunks))
    assert set(out) == {"q_c", "k_c", "v", "beta_c", "g_c", "sig"}
    assert out["q_c"].shape == (gp.HEADS, chunks, gp.TILE, gp.HEAD_DIM) and out["q_c"].dtype == torch.bfloat16
    assert out["k_c"].shape == out["q_c"].shape and out["k_c"].dtype == torch.bfloat16
    assert out["beta_c"].shape == (gp.HEADS, chunks, gp.TILE, 1) and out["beta_c"].dtype == torch.float32
    assert out["g_c"].shape == out["beta_c"].shape and out["g_c"].dtype == torch.float32
    assert out["v"].shape == (chunks * gp.TILE, gp.VALUE_WIDTH) and out["v"].dtype == torch.bfloat16
    assert out["sig"].shape == out["v"].shape and out["sig"].dtype == torch.bfloat16
    # q carries the composite's second scale and k does not: q is k's unit vector scaled once more
    assert not torch.equal(out["q_c"], out["k_c"])


@pytest.mark.parametrize("rows", [1, 5, 32, 33, 63])
def test_the_rows_mask_zeroes_exactly_the_rows_at_or_past_rows(rows):
    """What the writers do to the outputs is what the chain's ``x * row_mask`` multiplies do: rows below ``rows``
    unchanged, rows at or past it exact zeros."""

    chunks = 2
    case = _case(chunks)
    full = gp.reference(*case)
    masked = gp.reference(*case, rows=rows)
    assert masked is not full
    assert gp.mask_rows(full, rows, chunks * gp.TILE).keys() == masked.keys()
    for name in ("q_c", "k_c", "beta_c", "g_c"):
        kept = torch.arange(chunks * gp.TILE).reshape(chunks, gp.TILE) < rows
        assert torch.equal(masked[name][:, kept], full[name][:, kept])
        assert float(masked[name][:, ~kept].abs().sum()) == 0.0
    assert torch.equal(masked["v"][:rows], full["v"][:rows])
    assert float(masked["v"][rows:].abs().sum()) == 0.0
    # sig is never masked: the chain's z sigmoid carries no row mask
    assert torch.equal(masked["sig"], full["sig"])
    every = gp.reference(*case, rows=chunks * gp.TILE)  # all rows valid: the mask is a no-op
    assert all(torch.equal(every[name], full[name]) for name in full)


def test_the_rows_mask_refuses_an_out_of_range_row_count(expect_error):
    outputs = gp.reference(*_case(1))
    for rows in (0, -1, 33):
        with expect_error(ValueError):
            gp.mask_rows(outputs, rows, gp.TILE)


def test_chain_on_device_is_the_chains_calls_in_the_chains_order():
    """The oracle the device test compares against is the chain itself; this pins that the transcription keeps the
    chain's ops, argument order and broadcast forms (``ttnn/gdn.py`` 2544-2684 and the composite's relayout)."""

    source = inspect.getsource(gp.chain_on_device)
    flat = re.sub(r"\s+", "", source)
    calls = re.findall(r"ttnn\.([a-z_0-9]+)\(", source)
    for expected in (
        "to_layout",
        "slice",
        "concat",
        "multiply",
        "mac",
        "silu",
        "matmul",
        "reshape",
        "rms_norm",
        "typecast",
        "sigmoid",
        "add",
        "softplus",
        "permute",
    ):
        assert expected in calls, expected
    assert calls.index("mac") < calls.index("silu") < calls.index("matmul")
    assert "ttnn.mac(pieces[index],taps[index],conv)" in flat  # the chain's operand order
    assert "ttnn.rms_norm(heads_tensor,epsilon=NORM_EPS)" in flat  # no weight, no compute config
    assert "ttnn.multiply(normed,QK_SCALE,memory_config=l1)" in flat
    assert "ttnn.multiply(v_slice,mask_bf16_col,memory_config=dram)" in flat
    assert "ttnn.add(a_fp32,dt_bias,memory_config=l1)" in flat
    assert "ttnn.softplus(shifted,beta=1.0,threshold=20.0,memory_config=l1)" in flat
    assert "ttnn.multiply(neg_exp_A,softplus,memory_config=" in flat  # the constant is the LEFT operand
    assert "ttnn.reshape(ttnn.permute(tensor,(0,2,1,3)),(HEADS,rows_total,HEAD_DIM))" in flat  # head_split_tile
    assert "ttnn.reshape(head_major,(HEADS,chunks,TILE,HEAD_DIM))" in flat  # to_chunks_tile
    assert "ttnn.reshape(ttnn.permute(rows_view,(0,2,1)),(HEADS,rows_total))" in flat  # headvec_split_tile
    assert "head_major=ttnn.multiply(head_major,QK_SCALE)" in flat  # the composite's own q scale, q only
    assert "ttnn.typecast(ttnn.sigmoid(z_fp32,memory_config=l1),ttnn.bfloat16,memory_config=dram)" in flat
    assert "math_approx_mode=False" in flat and "fp32_dest_acc_en=True" in flat  # the chain's select compute config
