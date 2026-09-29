# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused router tail without a device: its registry entry and switch, the CB / argument contract between the
Python side and the three kernels, the exactness pins in the compute kernel (the replaced ops' instruction sequences),
the writer's tile-face addressing against a torch model of the tile layout, and the model's resolve site."""

from __future__ import annotations

import inspect
import re

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail as rt

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
    assert names <= set(rt.CB_INDEX) | {"Wt", "top_k", "stage_pages"}, names - set(rt.CB_INDEX)


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
        (fp.REPO_ROOT / "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h").read_text().splitlines()
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


def _llk_exp_precise_loop_with_vector_guard() -> str:
    """The precise fp32-dest loop of the LLK's calculate_exponential with `if ((pair_mask >> (d >> 1)) & 1u)` around the
    load / exp / store and the `dst_reg++` kept outside it -- what kernels/exp_live.h must contain."""

    llk = (
        (fp.REPO_ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h")
        .read_text()
        .splitlines()
    )
    start = next(i for i, l in enumerate(llk) if l.startswith("void calculate_exponential("))
    branch = next(
        i for i in range(start, len(llk)) if llk[i].strip() == "} else {"
    )  # the fp32-dest (non-replay) branch
    loop = next(i for i in range(branch, len(llk)) if "for (int d = 0; d < ITERATIONS; d++) {" in llk[i])
    assert loop == branch + 1, "the LLK's fp32-dest branch gained a statement before its loop: re-derive exp_live.h"
    close = next(i for i in range(loop, len(llk)) if llk[i].strip() == "}" and "dst_reg++" in llk[i - 1])
    assert (
        llk[close + 1].strip() == "}"
    ), "the LLK's fp32-dest branch gained a statement after its loop: re-derive exp_live.h"
    body = [l.strip() for l in llk[loop : close + 1]]
    assert body[1:4] == [
        "sfpi::vFloat val = sfpi::dst_reg[0];",
        "sfpi::dst_reg[0] =",
        "_ckernel_sfpu_exp_accurate_<SCALE_EN, is_fp32_dest_acc_en>(val, exp_base_scale_factor);",
    ], body
    assert body[4] == "sfpi::dst_reg++;" and body[5] == "}"
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
    bcast = (fp.REPO_ROOT / "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary_bcast.h").read_text()
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
