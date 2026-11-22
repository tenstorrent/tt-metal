# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Model-level determinism (FIX.25): the spec's device model runs the same prompts again in one process (cold first
sequence after the load, warm ones after another prompt); every chunk's final hidden and every layer's served state
must be bit-identical (testing/model_determinism.py). Every model-level gate runs it (core/gate.py with_determinism);
it may share a batch process with the gate's other runs (core/batch.py SHARED_TESTS), always first in it."""

from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.model_determinism import check

S = spec()
pytestmark = device_timeout(S)


@mesh_parametrize
def test_determinism(mesh_device):
    layers = S.layers()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    err, _ = check(S, model, layers)
    assert err is None, err
