# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x46 static pins of QWEN38_MOE_FUSE=1: the index-ready semaphore raise exists only behind the define MOE_FUSE_SEM, which only
the moe_experts fused form (``index_signal``) sets; without it router_tail_sc's descriptors and kernels are unchanged."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "ttnn" / "fused"


def test_writer_raise_is_behind_the_define():
    text = (ROOT / "router_tail_sc" / "kernels" / "writer_rtf.cpp").read_text()
    lines = text.splitlines()
    depth, inside = 0, False
    for line in lines:
        if line.strip().startswith("#ifdef MOE_FUSE_SEM"):
            inside = True
            continue
        if inside and line.strip() == "#endif":
            inside = False
            continue
        if not inside:
            assert "MOE_FUSE_SEM" not in line and "fuse_" not in line, line
    assert text.count("#ifdef MOE_FUSE_SEM") == 1


def test_only_the_signalled_form_sets_the_define():
    text = (ROOT / "router_tail_sc" / "__init__.py").read_text()
    assert text.count('("MOE_FUSE_SEM"') == 1
    i = text.index('("MOE_FUSE_SEM"')
    assert "if index_signal is not None:" in text[:i][-600:]
    assert "index_signal=None" in text  # default: no signal
    for other in ("router_tail", "gr_read", "gr_fold", "qsa_block"):
        assert "MOE_FUSE_SEM" not in (ROOT / other / "__init__.py").read_text()


def test_moe_dense_unfused_keeps_its_groups():
    text = (ROOT / "moe_dense" / "__init__.py").read_text()
    assert "if expert_parts is None:\n        kernels += [untilize_reader, untilize_writer]" in text
    assert "experts=None" in text


def test_experts_reader_waits_only_with_a_semaphore():
    text = (ROOT / "moe_experts" / "kernels" / "reader_moe_experts.cpp").read_text()
    assert "if constexpr (wait_sem != 0xFFFF)" in text
    init = (ROOT / "moe_experts" / "__init__.py").read_text()
    assert "0xFFFF if wait_sem is None else wait_sem" in init
