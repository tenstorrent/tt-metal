# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Stage-A CPU tests (skeleton, written during the D1 disk block; bodies land with the dataflow).

CONTRACT v1.1 §3 "host front end and back end" (rtol 1e-5, index logic exact) and §6 A (the dataflow within 1e-5
relative of the reference in fp32 on the test goldens; route S emulated in fp32 and float64). Each check gets a
positive control that must fail (D0 item 4).
"""

from pathlib import Path

import pytest

GOLDENS = Path("/home/andy/bringup/goldens/flowstate")
TEST_CASES = ("single", "model16", "single_long", "nan_mix", "mask0", "daily", "monthly", "fivemin", "cpm_edge")

pytestmark = pytest.mark.skipif(not (GOLDENS / "single.pt").exists(), reason="test goldens not generated yet")

TODO = "stage A: pending the reference venv (D1 disk block)"


@pytest.mark.parametrize("case", TEST_CASES)
def test_bookkeeping(case):
    """fl / mask_n / max_context / min_context / L exact against the golden's recorded values."""
    pytest.skip(TODO)


@pytest.mark.parametrize("case", TEST_CASES)
def test_causal_revin(case):
    """NaN mask, appended [0, 1] steps, running mean / stdev and normalized tokens, rtol 1e-5."""
    pytest.skip(TODO)


def test_constants():
    """Lbar, Bbar per layer and Legendre W from the reference expressions; control: float64 constants must fail."""
    pytest.skip(TODO)


@pytest.mark.parametrize("case", ("single", "nan_mix", "mask0", "monthly", "cpm_edge"))
def test_dataflow_fft(case):
    """The device decomposition (FFT at pow-2 N) within 1e-5 relative; control: circular conv at N = L must fail."""
    pytest.skip(TODO)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_scan_emulation(dtype):
    """Route S chunked scan error per layer and channel against the reference and the float64 truth (B0 goldens)."""
    pytest.skip(TODO)


@pytest.mark.parametrize("case", TEST_CASES)
def test_back_end(case):
    """CPM combine, de-normalization, torch.quantile, weighted mean; cpm_edge returns H - fl + 1 steps."""
    pytest.skip(TODO)
