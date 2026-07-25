# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The served Gumbel default, and the kernel fix it depends on (#48291).

This default has moved twice, so it is pinned rather than left to a comment:

* 2026-07-24 flipped to ``device`` (the on-device permuted-vocab RNG) for the ~313 ms/step host RNG
  and ~256 MiB/step replicated PCIe copy it removes;
* 2026-07-25 reverted to ``host`` after a matched 4-seed A/B showed ``device`` corrupting generated
  text on 2 of 4 seeds;
* restored to ``device`` once the CAUSE was fixed. The cause was never in the sampler: the Blackhole
  SFPU PRNG is a sliding window over one stream, so element ``(read t, lane i)`` carried
  ``stream[t + i]``, and the production noise shape put the 256 canvas positions on that axis --
  64 of 256 positions held a byte-identical copy of another position's noise. With the kernel
  advancing the window per element, the same A/B answers correctly 4/4 on both arms and the
  degeneracy guard never fires, at ~53.6 vs ~36.3 tokens/block/s.

So the invariant worth pinning is no longer "the default is host" but "the default is only allowed
to be a device RNG while the kernel-level independence gate holds". These tests pin the default and
point at that gate; ``tests/ttnn/nightly/unit_tests/operations/rand/test_rand_independence.py``
pins the kernel property itself.

CPU-only: no device, no weights, no checkpoint.
"""

import pytest

GV = pytest.importorskip("models.experimental.diffusion_gemma.tt.generator_vllm")

SUPPORTED_UPFRONT_MODES = {"host", "device"}


def test_served_default_is_a_materialized_mode():
    """Up-front capture needs a materialized full-vocabulary source; chunked/argmax are rejected."""
    assert GV.DEFAULT_VLLM_GUMBEL_MODE in SUPPORTED_UPFRONT_MODES, (
        f"served Gumbel default is {GV.DEFAULT_VLLM_GUMBEL_MODE!r}; up-front capture supports only "
        f"{sorted(SUPPORTED_UPFRONT_MODES)}"
    )


def test_served_default_is_device_for_throughput():
    """`device` is the default because `host` does not meet the throughput bar (~1.48x slower).

    If this has been changed back to `host`, the reason should be a NEW correctness finding, and
    the kernel gate in test_rand_independence.py is where to look first -- a regression there is
    what would make a device RNG unusable again.
    """
    assert GV.DEFAULT_VLLM_GUMBEL_MODE == "device", (
        f"served Gumbel default is {GV.DEFAULT_VLLM_GUMBEL_MODE!r}, not 'device'. That is a "
        "throughput regression (~53.6 -> ~36.3 tokens/block/s) unless a correctness finding "
        "justifies it; see doc/decision_fidelity/degenerate_output_fix.md"
    )


def test_the_kernel_gate_this_default_depends_on_exists():
    """The device default is only defensible while the ttnn.rand independence gate is present."""
    from pathlib import Path

    repo_root = Path(GV.__file__).resolve().parents[4]
    gate = repo_root / "tests" / "ttnn" / "nightly" / "unit_tests" / "operations" / "rand" / "test_rand_independence.py"
    assert gate.is_file(), (
        f"missing {gate}: the served device Gumbel default depends on the Blackhole ttnn.rand "
        "fix, and that gate is what keeps the fix from silently regressing"
    )
    text = gate.read_text()
    assert "test_rand_columns_are_distinct" in text, "the duplicate-column gate is gone"


def test_launcher_default_matches_the_module_default():
    """The GPQA launcher exports its own default; the two must not drift apart."""
    from pathlib import Path

    script = Path(GV.__file__).resolve().parent.parent / "doc" / "optimize_perf" / "run_upfront_gpqa.sh"
    if not script.is_file():
        pytest.skip("run_upfront_gpqa.sh not present in this checkout")
    expected = f'DG_VLLM_GUMBEL_MODE="${{DG_VLLM_GUMBEL_MODE:-{GV.DEFAULT_VLLM_GUMBEL_MODE}}}"'
    assert (
        expected in script.read_text()
    ), f"launcher default disagrees with DEFAULT_VLLM_GUMBEL_MODE; expected {expected!r}"
