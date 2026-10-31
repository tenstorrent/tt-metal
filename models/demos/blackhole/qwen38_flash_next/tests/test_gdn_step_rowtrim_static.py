# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of QWEN38_GDN_ROWTRIM (gdn_step split 4): default level 2 (1 and 0 off), level 1 / 2 defines, kernels default to today's ops."""

from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import conv_split_v4 as v4

KERNELS = Path(v4.__file__).parent / "kernels"
LOCAL, MASKED = "QWEN38_GDN_ROWTRIM_ITERS", "QWEN38_GDN_ROWTRIM_MASKED_ITERS"


def test_default_is_level_two_and_zero_is_off(monkeypatch):
    monkeypatch.delenv(v4.ROWTRIM_ENV, raising=False)
    assert v4.rowtrim_defines() == v4.rowtrim_defines("2")
    monkeypatch.setenv(v4.ROWTRIM_ENV, "0")
    assert v4.rowtrim_defines() == ([], [])
    assert v4.rowtrim_defines("0") == ([], [])


def test_level_one_trims_the_row_local_ops_of_the_head_cores_only():
    assert v4.rowtrim_defines("1") == ([(LOCAL, "2")], [])


def test_level_two_adds_the_masked_ops_and_the_norm_cores():
    assert v4.rowtrim_defines("2") == ([(LOCAL, "2"), (MASKED, "2")], [(MASKED, "2")])


def test_environment_selects_the_level(monkeypatch):
    monkeypatch.setenv(v4.ROWTRIM_ENV, "1")
    assert v4.rowtrim_defines() == ([(LOCAL, "2")], [])
    monkeypatch.setenv(v4.ROWTRIM_ENV, "2")
    assert v4.rowtrim_defines() == ([(LOCAL, "2"), (MASKED, "2")], [(MASKED, "2")])


def test_unknown_level_is_refused(expect_error):
    with expect_error(ValueError, match="QWEN38_GDN_ROWTRIM"):
        v4.rowtrim_defines("3")


def test_kernels_default_to_the_stock_ops_and_use_the_wrappers():
    header = (KERNELS / "rowtrim4.h").read_text()
    for name in (LOCAL, MASKED):
        assert f"#ifndef {name}\n#define {name} 0\n#endif" in header
    assert header.count("VectorMode::R)") + header.count("VectorMode::R,") == 7  # every trimmed call is R, none RC
    assert header.count("ITERS /* ITERATIONS */") == 6  # recip takes ITERS as its runtime argument
    head = (KERNELS / "compute_vb4.cpp").read_text()
    norm = (KERNELS / "compute_norm4.cpp").read_text()
    for text in (head, norm):
        assert '#include "rowtrim4.h"' in text
        for stock in (
            "add_unary_tile(",
            "sqrt_tile(",
            "qwen38_recip_tile(",
            "sigmoid_tile<",
            "mul_unary_tile(",
            "sub_binary_tile(",
        ):
            assert stock not in text.replace("sub_binary_tile_init", "")
    assert head.count("rowtrim::LOCAL") == 6 and head.count("rowtrim::MASKED") == 2
    assert norm.count("rowtrim::MASKED") == 3 and "rowtrim::LOCAL" not in norm
    # decay_state (every state row live) and the mask product (it zeroes the dead delta_b rows) stay stock
    assert head.count("mul_binary_tile(") == 3  # decay x2, the mask
    # the gate cores broadcast scalars to full tiles: untouched
    assert "rowtrim" not in (KERNELS / "compute_gate4.cpp").read_text()


def test_split_four_program_passes_the_defines_to_head_and_norm_compute_only():
    text = Path(v4.__file__).read_text()
    assert "defines=[*defines, *head_trim]" in text and "defines=[*defines, *norm_trim]" in text
    assert text.count("_trim]") == 2


def test_level_two_builds_only_for_a_pinned_projection(monkeypatch):
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

    plain, pinned = object(), object()
    fp.mark_pinned(pinned)
    monkeypatch.delenv(v4.ROWTRIM_ENV, raising=False)
    monkeypatch.delenv(fp.ZERO_DEAD_ROWS_ENV, raising=False)
    assert v4.effective_level(pinned) == "2"
    assert v4.effective_level(plain) == "1"  # unpinned: level 1, the define the program cache sees
    assert v4.rowtrim_defines(v4.effective_level(plain)) == v4.rowtrim_defines("1")
    monkeypatch.setenv(v4.ROWTRIM_ENV, "2")
    assert v4.effective_level(plain) == "1"
    monkeypatch.setenv(v4.ROWTRIM_ENV, "1")
    assert v4.effective_level(pinned) == "1"
    monkeypatch.setenv(v4.ROWTRIM_ENV, "0")
    assert v4.effective_level(plain) == "0"
    monkeypatch.delenv(v4.ROWTRIM_ENV)
    monkeypatch.setenv(fp.ZERO_DEAD_ROWS_ENV, "0")
    assert v4.effective_level(pinned) == "1"  # zeroing off: pinned or not, no level 2
