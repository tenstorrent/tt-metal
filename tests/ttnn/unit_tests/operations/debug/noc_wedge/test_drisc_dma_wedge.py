# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Blackhole hang: both DRISC-usable DRAM tiles of one bank stream GDDR DMA reads on the SAME TX stream with ~7 reads
in flight each, next to Tensix NOC1 DRAM reads of that bank (its NOC1 endpoint is one of the two tiles). Traffic
only; the device code is the gtest DriscNocWedge/DriscNocWedgeFixture.DISABLED_Run in
tests/tt_metal/tt_metal/api/test_drisc_noc_wedge.cpp (DRISC kernels need the C++ API: DramConfig), this wrapper runs
one case of it in a subprocess so it goes through scripts/run_safe_pytest.sh (device lock, dispatch timeout, reset).

Needs: Blackhole firmware >= 19.12 (programmable DRAM cores), the unit_tests_api binary (cmake -DTT_METAL_BUILD_TESTS=ON,
ninja unit_tests_api).

What a hang looked like (flat routed expert DRISC senders, bh-lb-09, 2026-10-14): the Tensix NOC1 DRAM reads never
returned (tt-triage: the readers' NOC1 reads outstanding), 3 of 3 runs; tile 1 alone, tile 2 alone, no DMA, split
streams or <= 2 reads in flight all passed. The flat op worked around it with split streams / a 2-read cap.

Cases (gtest parameter names): same_stream_depth7 (expect hang), same_stream_depth7_noforward (open: does it need
the DRISC's NoC writes?), and the controls split_streams_depth7, same_stream_depth2, noc1_endpoint_tile_only_depth7,
free_tile_only_depth7, same_stream_depth7_no_noc_reads (expect clean).
"""

import os
import subprocess

import pytest

# the DMA data check (DriscDmaCheck/DriscDmaCheckFixture): one tile, two jobs on streams 0 and 1; "wrong" = the op saw
# wrong weights (2026-11-02, GLM-Flash), "ok" = bit-exact there, "open" = not tried there
CHECK_CASES = {
    "free_tile_2jobs_depth3": "ok",
    "free_tile_2jobs_depth4": "wrong",
    "free_tile_2jobs_depth5": "wrong",
    "free_tile_2jobs_depth4_noforward": "open",
    "free_tile_1job_depth5": "open",
    "endpoint_tile_2jobs_depth4": "open",
    "free_tile_2jobs_depth4_noc1_reads": "open",
}
CASES = {
    "same_stream_depth7": "hang",
    "same_stream_depth7_noforward": "open",
    "split_streams_depth7": "clean",
    "same_stream_depth2": "clean",
    "noc1_endpoint_tile_only_depth7": "clean",
    "free_tile_only_depth7": "clean",
    "same_stream_depth7_no_noc_reads": "clean",
}


def _binary():
    home = os.environ.get("TT_METAL_HOME", os.getcwd())
    for p in ("build/test/tt_metal/unit_tests_api", "build_Release/test/tt_metal/unit_tests_api"):
        if os.path.exists(os.path.join(home, p)):
            return os.path.join(home, p)
    return None


def _run_gtest(flt, expect):
    binary = _binary()
    if binary is None:
        pytest.skip("unit_tests_api not built (cmake -DTT_METAL_BUILD_TESTS=ON; ninja unit_tests_api)")
    out = subprocess.run(
        [binary, "--gtest_also_run_disabled_tests", f"--gtest_filter={flt}"],
        env={**os.environ, "TT_METAL_SLOW_DISPATCH_MODE": "1"},  # (the DRAM-kernel fixture is slow dispatch only)
        cwd=os.environ.get("TT_METAL_HOME", os.getcwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(out.stdout)
    assert out.returncode == 0, f"{flt} (expect {expect}) exited {out.returncode}"
    if "[  SKIPPED ]" in out.stdout or "[  PASSED  ] 1 test" not in out.stdout:
        pytest.skip(f"{flt} did not run (skipped by the fixture: no programmable DRAM cores?)")


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("case", list(CASES))
def test_drisc_dma_wedge(case):
    _run_gtest(f"DriscNocWedge/DriscNocWedgeFixture.DISABLED_Run/{case}", CASES[case])


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("case", list(CHECK_CASES))
def test_drisc_dma_check(case):
    """Fails (gtest EXPECT) if any landed block differs from the pattern."""
    _run_gtest(f"DriscDmaCheck/DriscDmaCheckFixture.DISABLED_Run/{case}", CHECK_CASES[case])
