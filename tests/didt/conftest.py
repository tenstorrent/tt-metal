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
    """Hand the VF sweep options to OpTestBase, so every didt test runs the sweep unchanged."""
    freqs = request.config.getoption("--didt-vf-sweep")
    iterations = request.config.getoption("--sweep-iterations")
    offsets = None
    if freqs:
        freqs = [int(f) for f in freqs.replace(" ", "").split(",") if f]
        iterations = 1000 if iterations is None else iterations
        explicit = request.config.getoption("--vdd-offsets")
        if explicit:
            offsets = [int(o) for o in explicit.replace(" ", "").split(",") if o]
        else:
            offsets = VDD_OFFSETS_MV if request.config.getoption("--enable-vdd-offsets") else [0]
    saved = dict(OpTestBase.vf_sweep_defaults)
    OpTestBase.vf_sweep_defaults = {
        "freqs": freqs or None,
        "sweep_iterations": iterations,
        "vdd_offsets": offsets,
        "start_step": request.config.getoption("--vf-sweep-start-step") if freqs else 0,
        "results_path": request.config.getoption("--vf-sweep-results") if freqs else None,
    }
    yield
    OpTestBase.vf_sweep_defaults = saved
