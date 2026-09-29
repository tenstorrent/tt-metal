# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The two phase prims of ``chunk_gated_delta_rule`` are bound under the ``ttnn.prim.`` namespace, the runtime file
that binds them is an allowlisted runtime patch, and the prefill note says so.  No device."""

from __future__ import annotations

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_DIR.parents[3]
NANOBIND = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule_nanobind.cpp"
PRIMS = ("chunk_gdn_prep", "chunk_gdn_scan")


def _nanobind_source() -> str:
    return (REPO_ROOT / NANOBIND).read_text(encoding="utf-8")


def test_both_prims_are_bound_under_the_prim_namespace():
    source = _nanobind_source()
    for name in PRIMS:
        assert f'ttnn::bind_function<"{name}", "ttnn.prim.">' in source, f"{name} is not bound under ttnn.prim."
        # the forwarder hands the prim every argument; the docstring tells a caller what the composite passes
        assert f"ttnn::prim::{name}(" in source
        assert f"ttnn.prim.{name}" in source


def test_the_composite_keeps_its_own_binding():
    source = _nanobind_source()
    assert 'ttnn::bind_function<"chunk_gated_delta_rule", "ttnn.transformer.">' in source
    assert "&ttnn::transformer::chunk_gated_delta_rule," in source


def test_the_forwarders_take_the_composite_defaults():
    """memory_config=None is DRAM and compute_kernel_config=None is the composite's own kernel config, so a caller
    that leaves both unset reproduces the composite's call."""

    source = _nanobind_source()
    assert source.count("memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG)") == len(PRIMS)
    assert source.count("phase_kernel_config(") == len(PRIMS) + 1  # the two call sites and the helper itself
    for default in ("MathFidelity::HiFi4", "/*default_approx_mode=*/false", "/*default_fp32_acc=*/true"):
        assert default in source


def test_the_prefill_note_names_the_two_prims():
    text = (MODEL_DIR / "docs/PREFILL.md").read_text(encoding="utf-8")
    for name in PRIMS:
        assert f"ttnn.prim.{name}" in text
