# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The served Gumbel default must stay IID (#48291).

This exists because the default was flipped to the on-device permuted-vocab RNG for throughput and
that silently corrupted generated text: on a matched 4-seed A/B with one variable, `host` answered
correctly 4/4 while `device` corrupted 2/4 — token-level duplication across neighbouring canvas
positions, because `ttnn.rand` is not IID along the axis the permuted draw puts the 256 canvas
positions on. See doc/decision_fidelity/degenerate_output_fix.md.

The perf temptation has not gone away (device mode removes ~313 ms/step of host RNG and a
~256 MiB/step replicated PCIe copy), so the default is pinned here rather than left to a comment.
Anyone flipping it back must delete this test, which is the point.

CPU-only: no device, no weights, no checkpoint.
"""

import pytest

GV = pytest.importorskip("models.experimental.diffusion_gemma.tt.generator_vllm")

IID_MODES = {"host"}


def test_served_default_is_an_iid_sampler():
    assert GV.DEFAULT_VLLM_GUMBEL_MODE in IID_MODES, (
        f"served Gumbel default is {GV.DEFAULT_VLLM_GUMBEL_MODE!r}; only {sorted(IID_MODES)} draw "
        "IID noise across canvas positions on this hardware, and a non-IID default corrupts text"
    )


def test_default_is_not_the_reverted_device_mode():
    assert GV.DEFAULT_VLLM_GUMBEL_MODE != "device", (
        "device mode was the default from 2026-07-24 and was reverted on 2026-07-25 after it "
        "corrupted generated text on 2 of 4 matched seeds; it stays selectable via "
        "DG_VLLM_GUMBEL_MODE but must not be the default"
    )


def test_launcher_default_matches_the_module_default():
    """The GPQA launcher exports its own default; the two must not drift apart."""
    from pathlib import Path

    script = Path(GV.__file__).resolve().parent.parent / "doc" / "optimize_perf" / "run_upfront_gpqa.sh"
    if not script.is_file():
        pytest.skip("run_upfront_gpqa.sh not present in this checkout")
    text = script.read_text()
    expected = f'DG_VLLM_GUMBEL_MODE="${{DG_VLLM_GUMBEL_MODE:-{GV.DEFAULT_VLLM_GUMBEL_MODE}}}"'
    assert expected in text, f"launcher default disagrees with DEFAULT_VLLM_GUMBEL_MODE; expected {expected!r}"
