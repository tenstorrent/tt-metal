# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of QWEN38_RTF_ROWTRIM: default on, 2 iterations per face in VectorMode::R when on, the wrappers
untouched when off."""

from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail_sc

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
