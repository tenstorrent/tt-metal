# SPDX-License-Identifier: Apache-2.0
"""The Scaling view's parallelism facts come from the run itself -- tp_degree in the ledger + the
board topology -- not a mesh string passed on the command line."""
from tt_hw_planner.optimize_dashboard import _parallelism


def test_parallelism_is_read_from_the_runs_own_ledger_and_topology():
    ledger = {"tp_degree": [{"kind": "tp_degree", "value": 8}]}
    topology = {str(i): [i] for i in range(32)}  # 32 devices
    par = _parallelism(ledger, topology, 32)
    assert par["tp"] == 8 and par["devices"] == 32 and par["batch"] == 32
    assert par["dp"] == 4, "data-parallel degree = devices / tensor-parallel degree"
    assert _parallelism({}, None, None) is None, "nothing declared -> no panel"
