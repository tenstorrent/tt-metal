# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F56 device check: the built-in component checks (checks="auto") on the device, one Hy4 component per output kind
plus the precision edge cases (bfp8 experts, attention with its second inputs). Must pass: no false alarms.

    BRINGUP_SPEC=models/demos/hy4_preview_d_p/bringup/spec.yaml scripts/run_safe_pytest.sh --run-all \
        models/demos/common/bringup/dev/f56_device_check.py
"""
import pytest

from models.demos.common.bringup.dev.f49_mutation_proof import install_caches
from models.demos.common.bringup.testing.component import run_component_test
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
install_caches(S)  # one reference per layer; the golden source top-k for shared layers (Hy4 has no swap_context hook)
CASES = [
    ("attn_norm", 0),  # float, a norm: eps via the small input
    ("attn_residual", 0),  # float, iHC post mix
    ("indexer", 0),  # index (top-k positions, unsorted device output)
    ("attention", 1),  # float, stateful: chunk 0, the x2 input
    ("router", 1),  # selection
    ("experts", 1),  # float, bfp8 weights (low precision kind)
]


@pytest.mark.parametrize("step,layer", CASES, ids=[f"{s}_L{l}" for s, l in CASES])
@mesh_parametrize
def test_f56_device(mesh_device, step, layer):
    assert run_component_test(S, step, layer, mesh_device, None, None, checks="auto")
