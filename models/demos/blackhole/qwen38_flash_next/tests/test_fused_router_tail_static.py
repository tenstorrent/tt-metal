# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the fused router_tail family (base, fast, rowtrim, sort-core) without a device."""

from __future__ import annotations

import inspect
import re
import types
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail as rt
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail_sc
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail_sc as sc

# ============================================================================
# was test_fused_router_tail_static.py
# The fused router tail without a device: its registry entry and switch, the CB / argument contract between the
# Python side and the three kernels, the exactness pins in the compute kernel (the replaced ops' instruction sequences),
# the writer's tile-face addressing against a torch model of the tile layout, and the model's resolve site.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


SOURCES = {name: (fp.REPO_ROOT / path).read_text() for name, path in rt.KERNELS.items()}


def test_registered_bitwise_with_a_gate():
    entry = fused.kernel("router_tail")
    assert entry.tolerance == fused.BITWISE
    assert entry.fused is rt.router_tail and entry.composed is rt.router_tail_composed
    assert entry.gate is not None and entry.gate.layers == tuple(range(48)) and entry.gate.topk is None
    assert entry.default_on and fused.resolve("router_tail", {}) is rt.router_tail  # the foundation test pins the set
    assert fused.resolve("router_tail", {fused.OFF_ENV: "router_tail"}) is rt.router_tail_composed
    assert fused.resolve("router_tail", {fused.ENV: "router_tail", fused.OFF_ENV: "all"}) is rt.router_tail_composed
    assert (
        inspect.signature(rt.router_tail).parameters.keys()
        == inspect.signature(rt.router_tail_composed).parameters.keys()
    )


def test_cb_table_is_consistent():
    indices = [index for _name, index, _dtype, _pages in rt.CBS]
    assert len(set(indices)) == len(indices) and max(indices) < 32
    assert set(rt.UNPACK_TO_DEST_FP32) <= set(rt.CB_INDEX)
    for name in rt.UNPACK_TO_DEST_FP32:
        assert dict((n, d) for n, _i, d, _p in rt.CBS)[name] == ttnn.float32
    assert rt.WIDTH_TILES == 16 and rt.EXPERTS == 512 and rt.TOP_K == 10


@pytest.mark.parametrize("kernel", ["reader", "compute", "writer"])
def test_named_compile_time_args_exist_on_the_python_side(kernel):
    names = set(re.findall(r'get_named_compile_time_arg_val\("([a-z_0-9]+)"\)', SOURCES[kernel]))
    assert names, kernel
    assert names <= set(rt.CB_INDEX) | {"Wt", "top_k", "stage_pages"} | {n for n, _i, _d in rt.PAR_CBS}, names - set(
        rt.CB_INDEX
    )


def test_runtime_arg_layout_matches_the_python_side():
    reader, writer = SOURCES["reader"], SOURCES["writer"]
    assert [int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d)\)", reader)] == list(range(len(rt.READER_ARGS)))
    assert [int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d)\)", writer)] == list(range(len(rt.WRITER_ARGS)))
    assert "TensorAccessorArgs<0, 0>()" in reader and "next_compile_time_args_offset()" in reader
    assert "TensorAccessorArgs<0, 0>()" in writer and "next_compile_time_args_offset()" in writer
    compute = SOURCES["compute"]
    assert [int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d)\)", compute)] == list(range(len(rt.COMPUTE_ARGS)))
    source = inspect.getsource(rt.program_parts)  # the descriptors' builder (router_tail_program wraps it)
    assert "[logits.buffer_address(), index_template.buffer_address(), t, r, m]" in source
    assert "[scores.buffer_address(), indices.buffer_address(), t, r, m]" in source
    assert "per_core(lambda _t, _r, p, m: [p, m])" in source  # pass_mask, token_mask
    assert "token_mask >> row" in SOURCES["reader"] and "token_mask >> row" in SOURCES["writer"]


def test_compute_kernel_pins_the_replaced_ops_instruction_sequences():
    compute = SOURCES["compute"]
    # softmax.cpp numeric stable: MAX reduce, sub bcast cols + precise exp, SUM reduce with recip, mul bcast cols
    assert "PoolType::MAX,\n            ReduceDim::REDUCE_ROW" in compute
    assert "exp_tile_init<false>()" in compute and "exp_tile<false>(wt8)" in compute
    assert (
        "sub_tiles_bcast_cols(cb_in0, cb_max" in compute
        and "mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_recip" in compute
    )
    assert "qwen38_recip_tile_init();\n                qwen38_recip_tile(0);" in compute
    # topk.cpp single core: unstable network, largest, end phase 5, values DST 0/1, indices DST 2/3
    assert "topk_local_sort<false>(0, 0 /* largest */, 5 /* end_phase */)" in compute
    assert "transpose_tile(cb_probs, w, slot)" in compute and "copy_tile(cb_index, w, slot + 2)" in compute
    # reduce.cpp: ttnn.sum on fp32 takes the accurate SFPU path (input to dest); binary_ng SFPU div; typecast fp32 -> bf16
    assert (
        "ReduceFp32Mode::Accurate" in compute
        and "cb_vals" in rt.UNPACK_TO_DEST_FP32
        and "cb_vals,\n            cb_norm_scaler,\n            cb_sums" in compute
    )
    assert "div_binary_tile(0, 1, 0)" in compute
    assert (
        "typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0)"
        in compute
    )
    assert "stable_sort" not in compute.replace("stable_sort false", "")
    # the lane form: the same network, masked to the live tokens' passes; the LLK call stays for pass_mask 0
    assert "topk_local_sort_lanes<false>(0, 0 /* largest */, 5 /* end_phase */, pass_mask)" in compute
    assert compute.index("if (pass_mask == 0) {") < compute.index("topk_local_sort_lanes<false>")
    assert '#include "topk_lanes.h"' in compute and '#include "exp_live.h"' in compute
    # the live-vector exp serves by default; the A/B switch keeps the full exp_tile in its own branch
    assert compute.index("#if FRT_EXP_LIVE") < compute.index("exp_tile_live(wt8, live_pairs)") < compute.index("#else")
    assert "const uint32_t live_pairs = exp_live_pairs(token_mask);" in compute


def _llk_sort_with_pass_guard() -> str:
    """The LLK's _bitonic_topk_phases_steps with `if (pass_mask & (1u << (face * 2 + col)))` around each pass's phase
    loop -- what kernels/topk_lanes.h must contain, regenerated from the LLK source of this tree."""

    llk = (
        (fp.REPO_ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_topk.h")
        .read_text()
        .splitlines()
    )
    declaration = next(i for i, l in enumerate(llk) if "inline void _bitonic_topk_phases_steps(" in l)
    start = max(i for i in range(declaration) if llk[i].startswith("template <"))
    end = next(i for i in range(start, len(llk)) if llk[i].startswith("}") and "topk_replay_init = -1;" in llk[i - 1])
    body = llk[start : end + 1]
    signature = declaration - start
    body[signature] = (
        body[signature]
        .replace("_bitonic_topk_phases_steps(", "_bitonic_topk_phases_steps_lanes(")
        .replace("const int i_start_step)", "const int i_start_step, const std::uint32_t pass_mask)")
    )
    col_open = next(i for i, l in enumerate(body) if l == "        for (int col = 0; col < 2; col++)") + 1
    close = next(i for i, l in enumerate(body) if l == "            dst_addr_offset += 2;")
    inner = ["    " + l if l.strip() else l for l in body[col_open + 1 : close]]
    guard = ["            if (pass_mask & (1u << (face * 2 + col)))", "            {"]
    return "\n".join(body[: col_open + 1] + guard + inner + ["            }"] + body[close:])


def _code(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


def test_masked_sort_is_the_llk_sort_with_a_pass_guard():
    header = (fp.REPO_ROOT / rt.LANES_HEADER).read_text()
    expected = _llk_sort_with_pass_guard()
    assert _code(expected) in _code(
        header
    ), "kernels/topk_lanes.h no longer matches the LLK's _bitonic_topk_phases_steps + guard"
    assert expected.count("if (pass_mask & (1u << (face * 2 + col)))") == 1  # one guard, around each pass's phase loop
    assert (
        _code(header).index(_code("#ifdef TRISC_MATH"))
        < _code(header).index(_code(expected))
        < _code(header).rindex("#endif")
    )
    assert "topk_replay_init = -1;" in header  # the replay-buffer bookkeeping is the LLK's
    assert _code(
        "uint32_t idst, int idir, int i_end_phase, uint32_t pass_mask, int i_start_phase = 0, int i_end_step = 0, int i_start_step = 0)"
    ) in _code(
        header
    )  # the LLK's phase / step window passes through, defaulted as topk_local_sort defaults it
    assert "        i_start_phase,\n        i_end_step,\n        i_start_step,\n        pass_mask));" in header
    assert "VectorMode::RC_custom" in header and "calculate_bitonic_topk_phases_steps_lanes" in header


def test_full_exp_control_preserves_the_precise_path(monkeypatch):
    monkeypatch.delenv(rt.EXP_LIVE_ENV, raising=False)
    assert rt._compute_defines() == []
    monkeypatch.setenv(rt.EXP_LIVE_ENV, "0")
    assert rt._compute_defines() == [("FRT_EXP_LIVE", "0")]
    assert "exp_tile<false>(wt8)" in SOURCES["compute"]


def test_single_core_form_runs_the_full_exp(monkeypatch):
    """QWEN38_ROUTER_TAIL_LANES=0 gives every core token_mask ALL_TOKENS: every pair live, the LLK loop's own per-vector
    sequence (no saving, no change)."""

    monkeypatch.setenv(rt.LANES_ENV, "0")
    plan, lanes = rt._core_plan(1)
    assert not lanes and plan == [(0, 0, rt.ALL_TOKENS)]
    assert rt.ALL_TOKENS == (1 << 32) - 1 and _exp_live_pairs(rt.ALL_TOKENS) == 0xFF


def test_lanes_plan_one_core_per_live_pass(expect_error):
    fake = tuple(
        frozenset(range(8 * p, 8 * p + 8)) for p in range(4)
    )  # a stand-in map; the pinned one comes from the probe
    assert rt.lanes_plan(1, fake) == [(1, 0b1)]
    assert rt.lanes_plan(8, fake) == [(1, 0xFF)]
    assert rt.lanes_plan(9, fake) == [(1, 0xFF), (2, 0x100)]
    assert rt.lanes_plan(32, fake) == [(1, 0xFF), (2, 0xFF00), (4, 0xFF0000), (8, 0xFF000000)]
    with expect_error(ValueError):
        rt.lanes_plan(33, fake)
    # the pinned map (device probe 2026-09-18): pass = (row half, row parity)
    assert rt.PASS_TOKENS == (
        frozenset(range(0, 16, 2)),
        frozenset(range(1, 16, 2)),
        frozenset(range(16, 32, 2)),
        frozenset(range(17, 32, 2)),
    )
    assert frozenset().union(*rt.PASS_TOKENS) == frozenset(range(32))
    assert rt.lanes_plan(1) == [(1, 0b1)]
    assert rt.lanes_plan(2) == [(1, 0b01), (2, 0b10)]
    assert rt.lanes_plan(5) == [(1, 0b10101), (2, 0b01010)]
    assert rt.lanes_plan(16) == [(1, 0x5555), (2, 0xAAAA)]
    assert rt.lanes_plan(17) == [(1, 0x5555), (2, 0xAAAA), (4, 0x10000)]
    masks = [m for _p, m in rt.lanes_plan(32)]
    assert sum(masks) == rt.ALL_TOKENS and len(masks) == rt.PASSES
    assert rt.lanes_enabled({}) and rt.lanes_enabled({rt.LANES_ENV: "1"})  # the lane form serves by default
    assert not rt.lanes_enabled({rt.LANES_ENV: "0"})  # the single-core form is the off switch


# The LLK's fp32-dest exp loop as it was before 69bd2d415cb ("GDN prep: fp32 exp as a recorded replay body"), which made the
# default fp32 path _sfpu_exp_fp32_accurate_tti_ and left this sfpi loop under #ifdef DISABLE_SFPLOADMACRO as the plain
# reference stream.  kernels/exp_live.h keeps the loop (verified on device), so it is compared with this frozen text.
_FROZEN_LLK_EXP_LOOP = [
    "for (int d = 0; d < ITERATIONS; d++) {",
    "sfpi::vFloat val = sfpi::dst_reg[0];",
    "sfpi::dst_reg[0] =",
    "_ckernel_sfpu_exp_accurate_<SCALE_EN, is_fp32_dest_acc_en>(val, exp_base_scale_factor);",
    "sfpi::dst_reg++;",
    "}",
]


def _llk_exp_precise_loop_with_vector_guard() -> str:
    """The frozen precise fp32-dest loop of the LLK's calculate_exponential with `if ((pair_mask >> (d >> 1)) & 1u)` around
    the load / exp / store and the `dst_reg++` kept outside it -- what kernels/exp_live.h must contain."""

    llk = [
        l.strip()
        for l in (fp.REPO_ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h")
        .read_text()
        .splitlines()
    ]
    start = next(i for i, l in enumerate(llk) if l.startswith("void calculate_exponential("))
    # the frozen loop is still the LLK's reference stream (the DISABLE_SFPLOADMACRO form of the fp32 branch)
    assert any(
        llk[i : i + 6] == _FROZEN_LLK_EXP_LOOP for i in range(start, len(llk) - 5)
    ), "the LLK's sfpi exp loop changed"
    # the replayed form that runs by default is tested bit-exact against that sfpi loop
    replay_tests = (fp.REPO_ROOT / "tests/tt_metal/tt_metal/llk/test_sfpu_exp_fp32_replay.cpp").read_text()
    assert "TEST_F(LLKBlackholeSingleCardFixture, TensixSfpuExpFp32ReplayBitExact)" in replay_tests
    return "\n".join(
        [
            "    for (int d = 0; d < end; d++) {",  # end = 2 * (last live pair + 1): the face step is CR-based (see the header)
            "        if ((pair_mask >> (d >> 1)) & 1u) {",
            "            sfpi::vFloat val = sfpi::dst_reg[0];",
            "            sfpi::dst_reg[0] = _ckernel_sfpu_exp_accurate_<SCALE_EN, is_fp32_dest_acc_en>(val, exp_base_scale_factor);",
            "        }",
            "        sfpi::dst_reg++;",
            "    }",
        ]
    )


def test_live_exp_is_the_llk_precise_loop_with_a_vector_guard():
    header = (fp.REPO_ROOT / rt.EXP_LIVE_HEADER).read_text()
    expected = _llk_exp_precise_loop_with_vector_guard()
    assert expected in header, "kernels/exp_live.h no longer matches the LLK's precise exp loop + guard"
    assert "            end = 2 * (p + 1);" in header and "for (int p = 0; p < ITERATIONS / 2; p++) {" in header
    # the face step the trailing skip relies on: a carriage-return SETRWC (D = CR + 8, twice), not an increment of D
    common = (fp.REPO_ROOT / "tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h").read_text()
    assert "TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, num_rows, 0, 0, p_setrwc::SET_D);" in common
    assert (
        _code(header).index(_code("#ifdef TRISC_MATH"))
        < _code(header).index(_code(expected))
        < _code(header).rindex("#endif")
    )
    # the fp32-dest path is the loop, not the bf16 replay body: pinned on the LLK source and on the copy
    llk = (fp.REPO_ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h").read_text()
    assert (
        "_sfpu_exp_21f_bf16_tti_<SCALE_EN, is_fp32_dest_acc_en, CLAMP_NEGATIVE, ITERATIONS>" in llk
    )  # the other branch
    assert _code('static_assert(is_fp32_dest_acc_en, "the live-vector exp is the precise fp32-dest path') in _code(
        header
    )
    # the face loop is the LLK's VectorMode::RC dispatch: start, four faces each followed by the face increment, done
    common = (fp.REPO_ROOT / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h").read_text()
    rc = common[
        common.index("if (vector_mode == VectorMode::RC)") : common.index("else if (vector_mode == VectorMode::R)")
    ]
    assert (
        "for (int face = 0; face < 4; face++)" in rc and rc.count("_llk_math_eltwise_sfpu_inc_dst_face_addr_();") == 1
    )
    params = (
        fp.REPO_ROOT / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h"
    ).read_text()
    assert (
        params.index("_llk_math_eltwise_sfpu_start_(dst_index);")
        < params.index("_llk_math_eltwise_sfpu_apply_vector_mode_(")
        < params.index("_llk_math_eltwise_sfpu_done_();")
    )
    math = header[header.index("inline void exp_tile_live_math") : header.index("}  // namespace sfpu")]
    assert (
        math.index("_llk_math_eltwise_sfpu_start_(dst_index);")
        < math.index("for (int face = 0; face < 4; face++)")
        < math.index("calculate_exponential_live<is_fp32_dest_acc_en>((pair_mask >> (4 * (face >> 1))) & 0xFu);")
        < math.index("_llk_math_eltwise_sfpu_inc_dst_face_addr_();")
        < math.index("_llk_math_eltwise_sfpu_done_();")
    )
    # the face increment is two 8-row steps = 16 DST rows = one face of the fp32 tile; a vector is two rows
    assert "math::inc_dst_addr<8>();\n    math::inc_dst_addr<8>();" in common
    assert "_sfpu_check_<DST_SYNC_MODE>(idst, VectorMode::RC)" in header
    exp_api = (fp.REPO_ROOT / "tt_metal/hw/inc/api/compute/eltwise_unary/exp.h").read_text()
    assert (
        "(approx, is_fp32_dest_acc_en, scale_en, iterations, (input_clamping == InputClamping::ClampToNegative))"
        in exp_api
    )
    assert "int iterations = 8" in exp_api  # 8 vectors per face


def _exp_live_pairs(token_mask: int) -> int:
    """The kernel's exp_live_pairs in Python."""

    pairs = 0
    for p in range(4):
        if (token_mask >> (4 * p)) & 0xF:
            pairs |= 1 << p
        if (token_mask >> (16 + 4 * p)) & 0xF:
            pairs |= 1 << (4 + p)
    return pairs


def test_live_pair_mask_covers_every_live_token_row_of_the_fp32_tile():
    header = (fp.REPO_ROOT / rt.EXP_LIVE_HEADER).read_text()
    assert "if ((token_mask >> (4 * p)) & 0xFu) {\n            pairs |= 1u << p;" in header
    assert "if ((token_mask >> (16 + 4 * p)) & 0xFu) {\n            pairs |= 1u << (4 + p);" in header
    # the fp32 tile in DST: face f = 16 rows of 16 datums (token rows (f >> 1) * 16 + i, experts (f & 1) * 16 + j); one SFPU
    # vector = 4 rows x 8 columns of a face (tt-llk ckernel_sfpu_binary_bcast.h: "One SFPLOAD/SFPSTORE moves 4 dest rows x
    # 8 cols (32 lanes)", address +2 = the other column half, +4 = the next 4 rows), so vectors 2p, 2p + 1 = rows
    # 4p..4p + 3 at all 16 columns.  Every live token row must sit in a pair whose bit is set, in both faces that hold it;
    # the guard's `(pair_mask >> (d >> 1)) & 1u` runs both vectors of a set pair.
    bcast = (
        fp.REPO_ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_bcast.h"
    ).read_text()
    assert "One SFPLOAD/SFPSTORE moves 4 dest rows x 8 cols (32 lanes)" in bcast
    assert "addr +2 -> rows 0-3,  cols  8-15" in bcast and "addr +4 -> rows 4-7,  cols  0-7" in bcast
    assert "addr +16 -> start of next face" in bcast
    for token_mask in (0b1, 0b10, 0b11, 0b10101, 0b1010, 0x5555, 0xAAAA, 0x10000, 0x80000000, rt.ALL_TOKENS, 0):
        pairs = _exp_live_pairs(token_mask)
        for row in range(32):
            face_pair, vector = row // 16, (row % 16) // 2
            bit = (pairs >> (4 * face_pair + vector // 2)) & 1
            if (token_mask >> row) & 1:
                assert bit == 1, (token_mask, row)
    assert _exp_live_pairs(0b1) == 0b1 and _exp_live_pairs(0b1111) == 0b1 and _exp_live_pairs(0b10000) == 0b10
    assert (
        _exp_live_pairs(0x5555) == 0xF and _exp_live_pairs(0xAAAA) == 0xF
    )  # a lane core's 8 tokens: all 4 pairs of faces 0/1
    assert _exp_live_pairs(rt.ALL_TOKENS) == 0xFF and _exp_live_pairs(0) == 0
    # the decode body (rows 1): one pair = 4 of the 32 vectors run (vectors 0, 1 of faces 0 and 1)
    assert bin(_exp_live_pairs(rt.lanes_plan(1)[0][1])).count("1") == 1
    # the MTP verify rows (5, two lane cores): tokens {0, 2, 4} -> pairs 0, 1; tokens {1, 3} -> pair 0
    assert [_exp_live_pairs(m) for _p, m in rt.lanes_plan(5)] == [0b11, 0b1]


def _tile_faces(matrix: torch.Tensor) -> torch.Tensor:
    """A 32x32 matrix in tile memory order: faces 0..3 (rows 0-15 / 16-31 x cols 0-15 / 16-31), row-major inside."""

    faces = [matrix[r : r + 16, c : c + 16].reshape(-1) for r in (0, 16) for c in (0, 16)]
    return torch.cat(faces)


def test_writer_face_addressing_matches_the_tile_layout():
    writer = SOURCES["writer"]
    scores_tile = torch.arange(32 * 32).reshape(32, 32)  # [token, k]
    index_tile = torch.arange(32 * 32).reshape(32, 32) * 7  # [k, token]
    sc, idx = _tile_faces(scores_tile), _tile_faces(index_tile)
    assert "sc[(face_row * 2) * 256 + in_face * 16 + k]" in writer
    assert "idx[face_row * 256 + k * 16 + in_face]" in writer
    for row in range(32):
        face_row, in_face = row >> 4, row & 15
        for k in range(rt.TOP_K):
            assert sc[(face_row * 2) * 256 + in_face * 16 + k] == scores_tile[row, k]
            assert idx[face_row * 256 + k * 16 + in_face] == index_tile[k, row]


def test_reader_zero_fill_and_column_broadcast_match_the_tile_layout():
    reader = SOURCES["reader"]
    assert "tile[256 + i] = 0u" in reader and "tile[768 + i] = 0u" in reader  # faces 1 and 3
    assert "const uint32_t first_zero = (row < rows_in_tile && ((token_mask >> row) & 1u)) ? top_k : 0u;" in reader
    assert "const uint32_t base = (row >> 4) * 2 * 256 + (row & 15) * 16;" in reader
    tile = torch.arange(1, 32 * 32 + 1).reshape(32, 32)
    flat = _tile_faces(tile)
    for face in range(4):
        row0, col0 = (face >> 1) * 16, (face & 1) * 16
        for i in range(256):
            row, col = row0 + (i >> 4), col0 + (i & 15)
            assert flat[face * 256 + i] == tile[row, col]
    for row in range(32):
        left_face = (row >> 4) * 2
        assert flat[left_face * 256 + (row & 15) * 16] == tile[row, 0]
    assert torch.equal(flat[256:512].reshape(16, 16), tile[:16, 16:]) and torch.equal(
        flat[768:].reshape(16, 16), tile[16:, 16:]
    )


def test_index_template_is_the_transposed_topk_index_tile():
    k = torch.arange(32).reshape(32, 1)
    template = (torch.arange(512) // 32 * 32).reshape(1, 512) + k  # what router_tail_prepare uploads
    for w in range(16):
        generated = torch.arange(512)[w * 32 : (w + 1) * 32].repeat(32, 1)  # generate_index_tile: [r, c] = w*32 + c
        assert torch.equal(template[:, w * 32 : (w + 1) * 32], generated.T)


def test_model_resolves_once_and_uses_it_for_one_tile_rows():
    source = (fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/moe.py").read_text()
    assert source.count('fused.resolve("router_tail") if self.row_contract.row_tiles == 1 else None') == 1
    route = source[source.index("    def _route(") : source.index("    def _routed_partial(")]
    assert "self._route_tail(logits, top_k=TOP_K, compute_kernel_config=self.compute_config)" in route
    assert route.index("if self._route_tail is not None:") < route.index("probabilities = ttnn.softmax(")


def test_rawsort_flag_adds_the_define_and_default_is_unchanged(monkeypatch):
    monkeypatch.delenv(rt.EXP_LIVE_ENV, raising=False)
    monkeypatch.delenv(rt.RAWSORT_ENV, raising=False)
    assert rt._compute_defines() == []
    monkeypatch.setenv(rt.RAWSORT_ENV, "0")
    assert rt._compute_defines() == []
    monkeypatch.setenv(rt.RAWSORT_ENV, "1")
    assert rt._compute_defines() == [("FRT_RAWSORT", "1")]
    monkeypatch.setenv(rt.EXP_LIVE_ENV, "0")
    assert rt._compute_defines() == [("FRT_EXP_LIVE", "0"), ("FRT_RAWSORT", "1")]
    compute = SOURCES["compute"]
    assert (
        "#define FRT_RAWSORT 0" in compute
        and compute.count("#if FRT_RAWSORT") == 1
        and compute.count("#if !FRT_RAWSORT") == 1
    )


# ============================================================================
# was test_fused_router_tail_fast_static.py
# The shortened sort-core tail (QWEN38_ROUTER_TAIL_FAST=1) without a device: off by default, option plumbing, the
# kernels keep the sort-core form's softmax / sum / div / sort text, and the fp32-index decode.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


K = lambda name: (fp.REPO_ROOT / sc.FAST_KERNELS[name]).read_text()  # noqa: E731
BASE = lambda name: (fp.REPO_ROOT / sc.KERNELS[name]).read_text()  # noqa: E731


def _fast_code(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


def test_on_by_default_and_option_plumbing():  # default on since x42; QWEN38_ROUTER_TAIL_FAST=0 restores the sort-core tail
    assert sc.fast_opts({}) == sc.FAST_DEFAULT_OPTS and sc.fast_opts({sc.FAST_ENV: "0"}) == ()
    assert sc.fast_opts({sc.FAST_ENV: "1"}) == sc.FAST_DEFAULT_OPTS
    assert sc.fast_opts({sc.FAST_ENV: "1", sc.FAST_OPTS_ENV: "row0,split"}) == ("row0", "split")
    with pytest.raises(ValueError):  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        sc.fast_opts({sc.FAST_ENV: "1", sc.FAST_OPTS_ENV: "bogus"})
    assert set(sc.FAST_DEFAULT_OPTS) <= set(sc.FAST_OPTS) and "incr" not in sc.FAST_DEFAULT_OPTS


class _fast_Logits:
    dtype = ttnn.float32

    def device(self):
        return "mesh"

    def buffer_address(self):
        return 0x1000


@pytest.fixture
def fast_stubs(monkeypatch):
    monkeypatch.setattr(fp, "accessor_args", lambda tensor: [])
    monkeypatch.setattr(
        sc.gr_read, "noc_map", lambda mesh: {(x, y): (x + 18, y + 18) for x in range(8) for y in range(8)}
    )
    monkeypatch.setattr(sc, "float_index_template", lambda mesh: _fast_Logits())
    return ttnn.CoreRange(ttnn.CoreCoord(2, 3), ttnn.CoreCoord(5, 7))


def test_fast_program_has_the_split_index_kernel_and_defines(monkeypatch, fast_stubs):
    monkeypatch.setenv(sc.FAST_ENV, "1")
    monkeypatch.setenv(sc.PAR_ENV, "0")  # the old program: exactly these six kernels
    kernels, cbs, semaphores = sc.two_core_parts(
        _fast_Logits(), _fast_Logits(), _fast_Logits(), _fast_Logits(), rows=1, top_k=10, rectangle=fast_stubs
    )
    names = sorted(k.kernel_source.rsplit("/", 1)[-1] for k in kernels)
    assert names == sorted(
        [
            "reader_rtf_main.cpp",
            "writer_rtf.cpp",
            "compute_rtf_main.cpp",
            "reader_rtf_sort.cpp",
            "compute_rtf_sort.cpp",
            "writer_rtf_index.cpp",
        ]
    )
    defines = {k.kernel_source.rsplit("/", 1)[-1]: dict(k.defines) for k in kernels}
    assert defines["compute_rtf_sort.cpp"].keys() >= {"RTF_DIRECT", "RTF_INLINE", "RTF_IDXF"}
    assert defines["reader_rtf_sort.cpp"].get("RTF_SPLIT") == "1" and defines["writer_rtf.cpp"].get("RTF_IDXF") == "1"
    assert defines["reader_rtf_main.cpp"].keys() >= {"RTF_ROW0", "RTF_NOTOKEN", "RTF_IDXF"}


def test_fast_off_builds_the_sort_core_sources(monkeypatch, fast_stubs):
    monkeypatch.setenv(sc.FAST_ENV, "0")  # the fast tail is the default; "0" builds the sort-core sources
    kernels, _cbs, _sem = sc.two_core_parts(
        _fast_Logits(), _fast_Logits(), _fast_Logits(), _fast_Logits(), rows=1, top_k=10, rectangle=fast_stubs
    )
    assert sorted(k.kernel_source.rsplit("/", 1)[-1] for k in kernels) == sorted(
        [
            "reader_rt_main.cpp",
            "writer_router_tail.cpp",
            "compute_rt_main.cpp",
            "reader_rt_sort.cpp",
            "compute_rt_sort.cpp",
        ]
    )
    assert all(not k.defines or all(not n.startswith("RTF_") for n, _v in k.defines) for k in kernels)


def test_fast_compute_main_is_the_sort_core_one_plus_switches():
    fast, base = K("compute_main"), BASE("compute_main")
    # the same code with the two switches' #if lines removed
    stripped = re.sub(r"#if !RTF_(NOSTART|NOTOKEN)\n(.*?)#endif\n", r"\2", fast, flags=re.S)
    stripped = re.sub(r"#ifndef RTF_\w+\n#define RTF_\w+ 0[^\n]*\n#endif\n", "", stripped)
    # QWEN38_RTF_ROWTRIM: the helper block goes, the call site keeps its #else branch (the wrappers)
    stripped = re.sub(r"#ifdef QWEN38_RTF_ROWTRIM_ITERS\n(?:(?!#else).)*?#endif\n", "", stripped, flags=re.S)
    stripped = re.sub(r"#ifdef QWEN38_RTF_ROWTRIM_ITERS\n.*?#else\n(.*?)#endif\n", r"\1", stripped, flags=re.S)
    assert _fast_code(base.split("#include", 1)[1]) == _fast_code(stripped.split("#include", 1)[1])


def test_fast_sort_keeps_the_network_and_the_shift():
    fast = K("compute_sort")
    assert (
        "frt_sort64(pass_mask)" in fast
        and "topk_local_sort_lanes<false>(0, 0 /* largest */, 5 /* end_phase */, pass_mask)" in fast
    )
    assert "SORT_SHIFT_BITS = 0x43000000" in fast and "VectorMode::C" in fast  # faces 0 and 2 hold token 0
    assert "(w == 0) ? 0 : 1" in fast and "frt_load_tile(cb_src, cb_index, w" in fast


def test_fp32_index_decode_matches_python():
    import struct

    for e in range(512):
        bits = struct.unpack("<I", struct.pack("<f", float(e)))[0]
        assert (0 if bits == 0 else (((bits & 0x7FFFFF) | 0x800000) >> (150 - (bits >> 23)))) == e
    for name in ("reader_main", "writer"):
        assert "(((bits & 0x7FFFFFu) | 0x800000u) >> (150u - (bits >> 23)))" in K(name)


def test_par_program_swaps_the_sort_kernels_for_the_fold_and_merge_kernels(monkeypatch, fast_stubs):
    monkeypatch.setenv(sc.FAST_ENV, "1")
    kernels, _cbs, _sems = sc.two_core_parts(
        _fast_Logits(), _fast_Logits(), _fast_Logits(), _fast_Logits(), rows=1, top_k=10, rectangle=fast_stubs
    )
    names = sorted(k.kernel_source.rsplit("/", 1)[-1] for k in kernels)
    assert sc.par_enabled() and names == sorted(
        [
            "reader_rtf_main.cpp",
            "writer_rtf.cpp",
            "compute_rtf_main.cpp",
            "reader_rtf_fold.cpp",
            "writer_rtf_fold.cpp",
            "compute_rtf_fold.cpp",
            "reader_rtf_merge.cpp",
            "compute_rtf_merge.cpp",
        ]
    )


# ============================================================================
# was test_fused_router_tail_rowtrim_static.py
# No-device contract of QWEN38_RTF_ROWTRIM: default on, 2 iterations per face in VectorMode::R when on, the wrappers
# untouched when off.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


KERNEL = Path(router_tail_sc.__file__).parent / "kernels" / "compute_rtf_main.cpp"
DEFINE = "QWEN38_RTF_ROWTRIM_ITERS"


def test_default_on_and_off_switch():
    assert router_tail_sc.ROWTRIM_ENV == "QWEN38_RTF_ROWTRIM"
    assert router_tail_sc.rowtrim_defines({}) == [(DEFINE, "2")]
    assert router_tail_sc.rowtrim_defines({"QWEN38_RTF_ROWTRIM": "0"}) == []
    assert router_tail_sc.rowtrim_defines({"QWEN38_RTF_ROWTRIM": "1"}) == [(DEFINE, "2")]


def test_define_reaches_core_a_compute_only():
    src = Path(router_tail_sc.__file__).read_text()
    assert src.count("*rowtrim_defines()") == 1
    line = next(l for l in src.splitlines() if "*rowtrim_defines()" in l)
    assert "rt._compute_defines()" in line  # the compute_main descriptor


def test_kernel_wrappers_kept_when_off_and_expansions_when_on():
    text = KERNEL.read_text()
    off = text[text.index(f"#ifdef {DEFINE}\n        div_binary_tile_rows") :]
    assert "#else\n        div_binary_tile(0, 1, 0);" in off
    assert (
        "typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);"
        in off
    )
    helpers = text[text.index(f"#ifdef {DEFINE}\n// div_binary_tile<true>") : text.index("void kernel_main()")]
    assert helpers.count(f"{DEFINE} /* ITERATIONS */") == 2
    assert helpers.count("VectorMode::R)))") == 2 and "VectorMode::RC" not in helpers
    assert "calculate_sfpu_binary_div" in helpers and "calculate_typecast_fp32_to_fp16b" in helpers


# ============================================================================
# was test_fused_router_tail_sc_static.py
# The sort-core router top-k (``router_tail_sc``, QWEN38_ROUTER_SORTCORE=1) without a device: flag off -> today's
# ``router_tail.program_parts`` unchanged; flag on (rows == 1) -> the two-core form; its kernels keep today's sort, softmax,
# sum and division verbatim; the CB / argument / semaphore contract between the Python side and the kernels.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


SC_SOURCES = {name: (fp.REPO_ROOT / path).read_text() for name, path in sc.KERNELS.items()}
TODAY = (fp.REPO_ROOT / rt.KERNELS["compute"]).read_text()


def _sc_code(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


class _sc_Logits:
    dtype = ttnn.float32

    def device(self):
        return "mesh"

    def buffer_address(self):
        return 0x1000


@pytest.fixture
def sc_stubs(monkeypatch):
    monkeypatch.setattr(fp, "accessor_args", lambda tensor: [])
    monkeypatch.setattr(
        sc.gr_read, "noc_map", lambda mesh: {(x, y): (x + 18, y + 18) for x in range(8) for y in range(8)}
    )
    monkeypatch.setattr(sc.fp, "tensor_bytes", lambda tensor: 1)
    return types.SimpleNamespace(rect=ttnn.CoreRange(ttnn.CoreCoord(2, 3), ttnn.CoreCoord(5, 7)))


def test_flag_off_is_todays_program_parts_unchanged(monkeypatch):
    seen = {}

    def today(*args, **kwargs):
        seen["call"] = (args, kwargs)
        return "today"

    monkeypatch.setattr(rt, "program_parts", today)
    assert sc.enabled(1, {})  # on by default
    for env in ({sc.SORTCORE_ENV: "0"}, {sc.SORTCORE_ENV: ""}):
        assert not sc.enabled(1, env)
    monkeypatch.setenv(sc.SORTCORE_ENV, "0")
    got = sc.program_parts("l", "i", "s", "x", rows=1, top_k=10, rectangle="r", sem_base=3)
    assert got == "today"
    assert seen["call"] == ((("l", "i", "s", "x")), dict(rows=1, top_k=10, rectangle="r", sem_base=3))


@pytest.mark.parametrize("rows", [2, 5, 8, 32, 128])
def test_flag_on_other_rows_fall_back(monkeypatch, rows):
    monkeypatch.setenv(sc.SORTCORE_ENV, "1")
    monkeypatch.setenv(sc.LANES_ENV, "0")  # QWEN38_ROUTER_SC_LANES now defaults on and serves rows 2..32
    monkeypatch.setattr(rt, "program_parts", lambda *a, **k: "today")
    assert not sc.enabled(rows)
    assert sc.program_parts("l", "i", "s", "x", rows=rows, top_k=10) == "today"


def test_flag_on_needs_the_lane_form(monkeypatch):
    monkeypatch.setenv(sc.SORTCORE_ENV, "1")
    monkeypatch.setenv(rt.LANES_ENV, "0")
    assert not sc.enabled(1)
    monkeypatch.delenv(rt.LANES_ENV)
    assert sc.enabled(1)


def test_flag_on_rows_1_is_the_two_core_form(monkeypatch, sc_stubs):
    monkeypatch.setenv(sc.SORTCORE_ENV, "1")
    monkeypatch.setenv(sc.FAST_ENV, "0")  # the base two-core form (the fast tail is the default since x42)
    monkeypatch.setattr(rt, "program_parts", lambda *a, **k: pytest.fail("fell back"))
    kernels, cbs, semaphores = sc.program_parts(
        _sc_Logits(), _sc_Logits(), _sc_Logits(), _sc_Logits(), rows=1, top_k=10, rectangle=sc_stubs.rect, sem_base=2
    )
    assert len(kernels) == 5 and len(semaphores) == 1
    assert semaphores[0].id == 2  # sem_base
    by_source = {k.kernel_source.rsplit("/", 1)[-1]: k for k in kernels}
    assert set(by_source) == {
        "reader_rt_main.cpp",
        "writer_router_tail.cpp",
        "compute_rt_main.cpp",
        "reader_rt_sort.cpp",
        "compute_rt_sort.cpp",
    }
    cores = {}
    for name, kernel in by_source.items():
        ranges = list(kernel.core_ranges.ranges())
        assert len(ranges) == 1 and ranges[0].start == ranges[0].end, name
        cores[name] = (ranges[0].start.x, ranges[0].start.y)
    a, b = (2, 3), (3, 3)  # the first two cores of the rectangle's first row
    assert {
        cores[n] for n in cores if n in ("reader_rt_main.cpp", "writer_router_tail.cpp", "compute_rt_main.cpp")
    } == {a}
    assert {cores[n] for n in cores if n in ("reader_rt_sort.cpp", "compute_rt_sort.cpp")} == {b}
    assert [(r.start.x, r.start.y) for r in semaphores[0].core_ranges.ranges()] == [a]
    # core B's reader gets core A's NoC coordinates (the stub map adds 18); both compute kernels get pass 1 / token 1
    sort_reader = by_source["reader_rt_sort.cpp"]
    assert list(sort_reader.runtime_args[3][3]) == [0x1000, 0x1000, 0, 2 + 18, 3 + 18]
    named = dict(by_source["reader_rt_main.cpp"].named_compile_time_args)
    assert named["sem_id"] == 2 and named["top_k"] == 10 and named["Wt"] == 16
    # today's CB table on both cores
    assert [(c.total_size) for c in cbs] == [
        (rt.TOKEN_PAGE_BYTES if n in rt.TOKEN_CBS else fp.TILE_BYTES[ttnn.float32 if n == "cb_in0" else d]) * p
        for n, _i, d, p in rt.CBS
    ]
    assert all(len(list(c.core_ranges.ranges())) == 1 for c in cbs)


def test_python_side_matches_the_kernels():
    names = {}
    for key, text in SC_SOURCES.items():
        names[key] = set(re.findall(r'get_named_compile_time_arg_val\("([A-Za-z_0-9]+)"\)', text))
        assert names[key], key
        assert names[key] <= set(rt.CB_INDEX) | {"Wt", "top_k", "stage_pages", "sem_id"}, key
    args = lambda key: [int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d)\)", SC_SOURCES[key])]  # noqa: E731
    assert args("reader_main") == [0, 1]  # logits_addr, tile_row
    assert args("reader_sort") == [0, 1, 2, 3, 4]  # logits, index, tile_row, a_x, a_y
    assert args("writer") == list(range(len(rt.WRITER_ARGS)))
    assert 1 in args("compute_main") and 0 in args("compute_sort")  # token_mask / pass_mask
    source = inspect.getsource(sc.two_core_parts)
    assert "[(core_a, [logits_addr, 0, 1])]" in source
    assert "[(core_b, [logits_addr, index_addr, 0, a_x, a_y])]" in source
    assert "[(core_a, [SORT_PASS_MASK, 1])]" in source and "[(core_b, [SORT_PASS_MASK, 1])]" in source
    assert sc.SORT_PASS_MASK == 1 and 0 in rt.PASS_TOKENS[0]  # pass 0 holds token 0
    assert rt.lanes_plan(1) == [(sc.SORT_PASS_MASK, 1)]  # today's one-core lane plan for one row


def test_sort_core_keeps_todays_sort_verbatim():
    sort = SC_SOURCES["compute_sort"]
    for piece in (
        "FORCE_INLINE void frt_load_tile(uint32_t cb_probs, uint32_t cb_index, uint32_t w, uint32_t slot) {\n"
        "    reconfig_data_format_srca(cb_probs);\n    transpose_init(cb_probs);\n    transpose_tile(cb_probs, w, slot);\n"
        "    reconfig_data_format_srca(cb_index);\n    copy_init(cb_index);\n    copy_tile(cb_index, w, slot + 2);\n}",
        "FORCE_INLINE void frt_sort64(uint32_t pass_mask) {\n    if (pass_mask == 0) {\n"
        "        ckernel::topk_local_sort<false>(0, 0 /* largest */, 5 /* end_phase */);\n    } else {\n"
        "        topk_local_sort_lanes<false>(0, 0 /* largest */, 5 /* end_phase */, pass_mask);\n    }\n}",
        "compute_kernel_hw_startup(cb_probs, cb_index, cb_vals_t);\n    ckernel::topk_tile_init();",
        "tile_regs_acquire();\n        for (uint32_t w = 0; w < Wt; ++w) {\n"
        "            frt_load_tile(cb_probs, cb_index, w, (w == 0) ? 0 : 1);\n            if (w != 0) {\n"
        "                frt_sort64(pass_mask);\n            }\n        }\n        tile_regs_commit();",
        "pack_tile(2, cb_idx_t);",
    ):
        assert _sc_code(piece) in _sc_code(sort), piece
    # x18's raw-logit copy pass is the same block as today's FRT_RAWSORT one
    copy_pass = "tile_regs_acquire();\n            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {\n                copy_tile(cb_in0, wt + wt8, wt8);"
    assert _sc_code(copy_pass) in _sc_code(sort) and _sc_code(copy_pass) in _sc_code(TODAY)
    assert '#include "../../router_tail/kernels/topk_lanes.h"' in sort
    assert "softmax" not in _sc_code(sort).replace("fz_rt_c_softmax", "")  # no softmax on the sort core
    assert "wait_front(1)" not in sort and "cb_max_scaler" not in sort
    # the sort keys are logit + 128 (positive like the probabilities: the SFPSWAP breaks negative ties differently)
    assert "SORT_SHIFT_BITS = 0x43000000" in sort and "add_unary_tile(wt8, SORT_SHIFT_BITS)" in sort


def test_main_core_keeps_todays_softmax_sum_and_div_verbatim():
    main = SC_SOURCES["compute_main"]
    softmax_begin = TODAY.index('FUSED_ZONE("fz_rt_c_softmax");')
    softmax_end = TODAY.index("#else", TODAY.index("probs.push_back(ndst);"))
    softmax = TODAY[softmax_begin:softmax_end]
    softmax = softmax[: softmax.rindex("    }")]
    assert _sc_code(softmax) in _sc_code(main)
    tail = TODAY[TODAY.index('FUSED_ZONE("fz_rt_c_sum");') :]
    assert _sc_code(tail) in _sc_code(main)
    assert "topk" not in _sc_code(main) and "frt_sort64" not in main
    assert "transpose" not in _sc_code(main)


def test_reader_gathers_by_the_tile_face_formula():
    reader = SC_SOURCES["reader_main"]
    assert "idx[k * 16]" in reader  # [k, token] tile, token 0 (row k of face 0)
    assert "p[(e >> 5) * 1024 + ((e >> 4) & 1u) * 256 + (e & 15u)]" in reader
    assert "noc_semaphore_wait_min(sem, 1)" in reader and "noc_semaphore_set(sem, 0)" in reader
    sender = SC_SOURCES["reader_sort"]
    assert (
        sender.index("noc_async_write(")
        < sender.index("noc_async_write_barrier()")
        < sender.index("noc_semaphore_inc(")
    )
    assert "get_noc_addr(a_x, a_y, src)" in sender  # the same L1 address on core A


def _simulate_gather(probs_row, idx_k):
    """Torch-free model of the reader's gather over the tile layout (16 tiles x 1024 words, 4 faces of 16 x 16)."""

    tiles = [[0.0] * 1024 for _ in range(16)]
    for e, value in enumerate(probs_row):
        face = (e >> 4) & 1
        tiles[e >> 5][face * 256 + (e & 15)] = value  # row 0 of the tile
    flat = [w for t in tiles for w in t]
    return [flat[(e >> 5) * 1024 + ((e >> 4) & 1) * 256 + (e & 15)] for e in idx_k]


def test_gather_model_picks_probability_of_each_index():
    probs = [float(i) + 0.5 for i in range(512)]
    idx = [511, 0, 17, 33, 256, 300, 31, 32, 64, 480]
    assert _simulate_gather(probs, idx) == [probs[i] for i in idx]


def test_not_a_registered_kernel_and_off_by_default():
    assert "router_tail_sc" not in {e.name for e in fused.REGISTRY.values()} if hasattr(fused, "REGISTRY") else True
    assert sc.SORTCORE_ENV == "QWEN38_ROUTER_SORTCORE"
    moe = (fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/fused/moe_dense/__init__.py").read_text()
    assert "rt_sc.program_parts(" in moe and "rt.program_parts(" not in moe
