# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from tt_py_test_utils_common.op_test_base import OpTestBase


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
    if freqs:
        freqs = [int(f) for f in freqs.replace(" ", "").split(",") if f]
        iterations = 1000 if iterations is None else iterations
    saved = dict(OpTestBase.vf_sweep_defaults)
    OpTestBase.vf_sweep_defaults = {"freqs": freqs or None, "sweep_iterations": iterations}
    yield
    OpTestBase.vf_sweep_defaults = saved
