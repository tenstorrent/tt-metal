# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU only: the torch mirror of the TT decomposition (structure_flow_host.functional_forward) against the fp32
golden calls. Any gap here is a decomposition or host-table bug, not device numerics, so the bound is tight.

    python models/experimental/trellis2/tests/test_structure_flow_reference.py [call ...]
"""

from __future__ import annotations

import sys

import pytest
import torch

from models.experimental.trellis2.reference import structure_flow_host as host
from models.experimental.trellis2.tests import goldens

CALLS = [0, 1, 20, 21]


@pytest.mark.skipif(goldens.missing_reason() is not None, reason=goldens.missing_reason() or "")
@pytest.mark.parametrize("call", CALLS)
def test_functional_matches_golden(call):
    state = host.load_state_dict(f"{goldens.weights_dir()}/{goldens.SS_CKPT}")
    x, t, cond, gold, _ = goldens.call_inputs(call)
    with torch.no_grad():
        out = host.functional_forward(state, x, t, cond)
    assert goldens.pcc(out, gold) > 0.99999
    assert (out - gold).abs().max() / gold.abs().max() < 1e-4


if __name__ == "__main__":
    torch.set_num_threads(16)
    state = host.load_state_dict(f"{goldens.weights_dir()}/{goldens.SS_CKPT}")
    for call in [int(a) for a in sys.argv[1:]] or CALLS:
        x, t, cond, gold, branch = goldens.call_inputs(call)
        with torch.no_grad():
            out = host.functional_forward(state, x, t, cond)
        err = ((out - gold).abs().max() / gold.abs().max()).item()
        print(
            f"call {call:2d} ({branch}) t={float(t[0]):7.2f} pcc={goldens.pcc(out, gold):.8f} max_rel_err={err:.2e}",
            flush=True,
        )
