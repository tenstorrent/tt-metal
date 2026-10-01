# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MoE (router, BF4 routed experts, shared expert) on the 1x4 mesh against the torch reference: layer 0's block on
three independent tokens (the block is stateless), the top-10 expert set within one boundary swap, PCC on the output.
The routed experts are the BF4 cache's, so the PCC floor is below the BF16 blocks'.

test_moe_component.py proves ``Qwen38MoE`` against the pinned Transformers module on the CPU; this is the device side.

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_moe_tp.py -v -s
Env:  MODEL_WEIGHTS_DIR, MESH_DEVICE="(1, 4)", QWEN38_CACHE_ROOT (skipped without it); QWEN38_BF4_CORPUS with
      QWEN38_BF4_CORPUS_VERIFICATION when the experts come from the CPU-staged corpus
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, HIDDEN_SIZE, pcc
from models.demos.blackhole.qwen38_flash_next.tt.moe import Qwen38MoE, Qwen38MoEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1",
    reason="a device test: set QWEN38_FUSED_DEVICE_TEST=1 on a held four-die line (the no-device sets are masked)",
)

LAYER_INDEX = 0
ROWS = 3


@run_for_blackhole()
@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_moe_tp(tp_harness, pcc_thresholds):
    harness = tp_harness
    builder, layer = harness.build_layer(LAYER_INDEX)
    reference = Qwen38MoE(Qwen38MoEWeights(harness.checkpoint, harness.placement, layer_index=LAYER_INDEX))
    generator = torch.Generator().manual_seed(3000)
    hidden = torch.randn((1, ROWS, HIDDEN_SIZE), generator=generator).to(torch.bfloat16)

    with builder.expert_streamer.layer(LAYER_INDEX) as (packed_w0_w1, packed_w2):
        for row in range(ROWS):
            token = hidden[:, row : row + 1]
            device_token = harness.upload_sharded(token.reshape(1, 1, 1, HIDDEN_SIZE))
            result = layer.mlp.forward(device_token, packed_w0_w1, packed_w2, return_routing=True)
            got = harness.download_sharded(result.hidden_sharded).reshape(1, 1, HIDDEN_SIZE)
            indices = harness.download_replicated(result.routing.indices).to(torch.int64).flatten()
            scores = harness.download_replicated(result.routing.scores).to(torch.float32).flatten()
            for tensor in (device_token, result.hidden_sharded, result.routing.scores, result.routing.indices):
                ttnn.deallocate(tensor)

            expected, routing = reference(token)
            # Router logits are bf16: a near-tie at the top-10 boundary may swap one expert on random inputs.
            reference_experts = routing.indices.flatten().tolist()
            assert (
                len(set(indices.tolist()) & set(reference_experts)) >= len(reference_experts) - 1
            ), f"row {row}: experts {indices.tolist()} vs reference {reference_experts}"
            if indices.tolist() == reference_experts:
                torch.testing.assert_close(scores, routing.scores.flatten().to(torch.float32), rtol=0.02, atol=0.01)
            score = pcc(got, expected)
            logger.info(f"layer {LAYER_INDEX} row {row}: PCC {score:.6f}, experts {indices.tolist()}")
            assert score >= pcc_thresholds["test_moe_tp"], f"row {row}: PCC {score:.6f}"
