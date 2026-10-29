# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x46: QWEN38_MOE_FUSE=1 (the routed experts as kernel groups of the moe_dense program) against the default decode MoE
(separate moe_dense and moe_experts programs), BIT FOR BIT on the TP=8 mesh, real layers, many random hidden rows; plus
traced timing of both forms (the whole MoE block of one layer: gather, router, shared, top-k, experts, post, reduce).
Run with QWEN38_MOE_FUSE=1 in the environment (the layer reads it at build time; the test flips ``moe_fuse_on``)."""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import traced_wall_us
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, HIDDEN_SIZE
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import Qwen38TTNNMoESyncPolicy

LAYERS = [int(v) for v in os.environ.get("X46_LAYERS", "0,11,30").split(",")]
TOKENS = int(os.environ.get("X46_TOKENS", "12"))
PARAMS = {**DEVICE_PARAMS, "trace_region_size": 64 * 1024 * 1024}


def per_die(tensor):
    return [ttnn.to_torch(t).view(torch.int16) for t in ttnn.get_device_tensors(tensor)]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
def test_fuse_bitwise_and_timing(tp_harness):
    harness = tp_harness
    results = Path(os.environ.get("X46_RESULTS", "/tmp/x46.jsonl"))
    ok = True
    for layer_index in LAYERS:
        builder, layer = harness.build_layer(layer_index)
        mlp = layer.mlp
        assert mlp.moe_fuse_on, dict(
            experts=mlp.moe_experts_on,
            dense=mlp.moe_dense_fused,
            rows=mlp.rows,
            post=mlp.moe_post_fused,
            l1=mlp.routing_in_l1,
            env=os.environ.get("QWEN38_MOE_FUSE"),
        )
        gen = torch.Generator().manual_seed(1000 + layer_index)
        with builder.expert_streamer.layer(layer_index) as (w01, w2):

            def run(token_dev, fused):
                mlp.moe_fuse_on = fused
                result = mlp.forward(token_dev, w01, w2, return_routing=True)
                out = per_die(result.hidden_sharded), per_die(result.routing.indices)
                for t in (result.hidden_sharded, result.routing.scores, result.routing.indices):
                    ttnn.deallocate(t)
                return out

            bad = 0
            for t in range(TOKENS):
                hidden = (torch.randn(1, 1, 1, HIDDEN_SIZE, generator=gen) * (0.5 + t * 0.3)).to(torch.bfloat16)
                token = harness.upload_sharded(hidden)
                base = run(token, False)
                fused = run(token, True)
                for a, b in zip(base[0] + base[1], fused[0] + fused[1]):
                    bad += int((a != b).sum())
                ttnn.deallocate(token)
            row = dict(layer=layer_index, tokens=TOKENS, differing_words=bad)
            if os.environ.get("X46_TIME", "1") == "1":
                token = harness.upload_sharded(torch.randn(1, 1, 1, HIDDEN_SIZE, generator=gen).to(torch.bfloat16))
                mlp.synchronization_policy = Qwen38TTNNMoESyncPolicy.RESIDENT_ASYNC  # no host drains under a capture
                for name, flag in (("default_us", False), ("fused_us", True)):

                    def call():
                        mlp.moe_fuse_on = flag
                        r = mlp.forward(token, w01, w2, return_routing=True)
                        for x in (r.hidden_sharded, r.routing.scores, r.routing.indices):
                            ttnn.deallocate(x)

                    row[name] = round(traced_wall_us(harness.mesh_device, call, replays=30)[0], 2)
                ttnn.deallocate(token)
            logger.info(json.dumps(row))
            with results.open("a") as f:
                f.write(json.dumps(row) + "\n")
            ok &= bad == 0
    assert ok
