# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GDN decode on the 1x4 mesh against the torch reference: one checkpoint layer, positions 0..3 (one FIR ring of the
kernel-4 causal conv), the recurrent and conv state carried on both sides, PCC per position.

test_gdn_component.py proves ``Qwen38GDN`` against the pinned Transformers module on the CPU; this is the device side.

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_gdn_tp.py -v -s
Env:  MODEL_WEIGHTS_DIR, MESH_DEVICE="(1, 4)"; QWEN38_CACHE_ROOT optional (component cache reuse)
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, HIDDEN_SIZE, pcc
from models.demos.blackhole.qwen38_flash_next.tt.gdn import Qwen38GDN, Qwen38GDNWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNGDN, Qwen38TTNNGDNWeights

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1",
    reason="a device test: set QWEN38_FUSED_DEVICE_TEST=1 on a held four-die line (the no-device sets are masked)",
)

POSITIONS = 4


@run_for_blackhole()
@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_index", [0, 46])  # the first and the last GDN layer (layers 3 mod 4 are QSA)
def test_gdn_tp(tp_harness, layer_index, pcc_thresholds):
    harness = tp_harness
    weights = Qwen38TTNNGDNWeights.from_checkpoint(
        harness.checkpoint,
        harness.mesh_device,
        harness.contract,
        harness.component_cache_root,
        layer_index=layer_index,
        tt_metal_sha=harness.tt_metal_sha,
    )
    module = Qwen38TTNNGDN(harness.mesh_device, harness.contract, weights, collective_topology=harness.topology)
    reference = Qwen38GDN(Qwen38GDNWeights.from_checkpoint(harness.checkpoint, layer_index))
    generator = torch.Generator().manual_seed(1000 + layer_index)
    hidden = torch.randn((1, POSITIONS, HIDDEN_SIZE), generator=generator).to(torch.bfloat16)

    state = module.allocate_state()
    reference_state = None
    try:
        for position in range(POSITIONS):
            token = hidden[:, position : position + 1]
            device_token = harness.upload_sharded(token.reshape(1, 1, 1, HIDDEN_SIZE))
            result = module.forward_decode(device_token, state)
            assert result.state is state, "GDN replaced its fixed-address state"
            got = harness.download_sharded(result.hidden_sharded).reshape(1, 1, HIDDEN_SIZE)
            ttnn.deallocate(device_token)
            ttnn.deallocate(result.hidden_sharded)

            expected, reference_state = reference.forward(token, reference_state)
            score = pcc(got, expected)
            logger.info(f"layer {layer_index} position {position}: PCC {score:.6f}")
            assert score >= pcc_thresholds["test_gdn_tp"], f"position {position}: PCC {score:.6f}"
        assert state.recurrent.dtype == ttnn.float32
    finally:
        state.deallocate()
        weights.deallocate()
