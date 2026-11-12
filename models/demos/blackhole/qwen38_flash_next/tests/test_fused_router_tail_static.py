# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The router tail's two LLK copies without a device: ``kernels/topk_lanes.h`` and ``kernels/exp_live.h`` are LLK
functions copied verbatim plus a mask, so each is regenerated here from the LLK source of this tree and compared, and
the LLK facts the masks rely on are checked on the same sources; an LLK change fails here instead of diverging the
served router silently."""

from __future__ import annotations

import re

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail as rt


def _code(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


def test_masked_sort_is_the_llk_sort_with_a_pass_guard():
    # kernels/topk_lanes.h must contain the LLK's _bitonic_topk_phases_steps with
    # `if (pass_mask & (1u << (face * 2 + col)))` around each pass's phase loop, regenerated from this tree's LLK.
    header = (fp.REPO_ROOT / rt.LANES_HEADER).read_text()
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
    expected = "\n".join(body[: col_open + 1] + guard + inner + ["            }"] + body[close:])
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


def test_live_exp_is_the_llk_precise_loop_with_a_vector_guard():
    # kernels/exp_live.h must contain the precise fp32-dest loop of the LLK's calculate_exponential with
    # `if ((pair_mask >> (d >> 1)) & 1u)` around the load / exp / store and the `dst_reg++` kept outside it.
    header = (fp.REPO_ROOT / rt.EXP_LIVE_HEADER).read_text()
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
    expected = "\n".join(
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
