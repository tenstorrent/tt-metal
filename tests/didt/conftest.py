# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from tt_py_test_utils_common.op_test_base import OpTestBase

# --enable-vdd-offsets: VDD over the firmware's VCORE at each sweep clock, in mV (0 = the firmware's own voltage)
VDD_OFFSETS_MV = [0, 25, 50]


def pytest_collection_modifyitems(config, items):
    """A VF sweep runs every clock in one test, far past pytest.ini's 300 s timeout. Lift it unless --timeout is given."""
    if not config.getoption("--didt-vf-sweep"):
        return
    if any(arg == "--timeout" or arg.startswith("--timeout=") for arg in config.invocation_params.args):
        return
    for item in items:
        item.add_marker(pytest.mark.timeout(0))


@pytest.fixture(autouse=True)
def didt_vf_sweep(request):
    """Hand --didt-vf-sweep / --sweep-iterations to OpTestBase, so every didt test runs the sweep unchanged."""
    freqs = request.config.getoption("--didt-vf-sweep")
    iterations = request.config.getoption("--sweep-iterations")
    offsets = None
    if freqs:
        freqs = [int(f) for f in freqs.replace(" ", "").split(",") if f]
        iterations = 1000 if iterations is None else iterations
        offsets = VDD_OFFSETS_MV if request.config.getoption("--enable-vdd-offsets") else [0]
    saved = dict(OpTestBase.vf_sweep_defaults)
    OpTestBase.vf_sweep_defaults = {"freqs": freqs or None, "sweep_iterations": iterations, "vdd_offsets": offsets}
    yield
    OpTestBase.vf_sweep_defaults = saved


@pytest.fixture(autouse=True)
def didt_grid_origin_supported(request):
    """--grid-origin moves the grid only in minimal_matmul (MinimalMatmulConfig.compute_with_storage_origin); the
    other didt ops always start at core (0, 0), so refuse it there instead of ignoring it."""
    origin = request.config.getoption("--grid-origin")
    if origin and origin.replace(" ", "") != "0x0" and "grid_origin" not in request.fixturenames:
        pytest.fail(
            f"--grid-origin {origin}: this test cannot move its grid; test_minimal_matmul, "
            "test_deepseek_v3_128k_matmul and test_mm_mcast_write_ack_failure can"
        )
