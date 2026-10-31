# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_S2I_FOLD=1 (the streamed router linear writes the fp32 DRAM tile itself: no ShardedToInterleaved) against the
default decode MoE, bit for bit on the TP=8 mesh, real layers.  Needs QWEN38_FUSED_DEVICE_TEST=1."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, HIDDEN_SIZE
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="needs the TP=8 mesh")
LAYERS = [int(v) for v in os.environ.get("X46_LAYERS", "0,11,30").split(",")]
TOKENS = int(os.environ.get("X46_TOKENS", "12"))


def per_die(tensor):
    return [ttnn.to_torch(t).view(torch.int16) for t in ttnn.get_device_tensors(tensor)]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_s2i_fold_bitwise(tp_harness, monkeypatch):
    harness = tp_harness
    bad_total = 0
    for layer_index in LAYERS:
        builder, layer = harness.build_layer(layer_index)
        mlp = layer.mlp
        gen = torch.Generator().manual_seed(2000 + layer_index)
        with builder.expert_streamer.layer(layer_index) as (w01, w2):

            def run(token_dev, fold):
                monkeypatch.setenv("QWEN38_S2I_FOLD", "1" if fold else "0")
                result = mlp.forward(token_dev, w01, w2, return_routing=True)
                out = per_die(result.hidden_sharded), per_die(result.routing.indices), per_die(result.routing.scores)
                for t in (result.hidden_sharded, result.routing.scores, result.routing.indices):
                    ttnn.deallocate(t)
                return out

            for t in range(TOKENS + 2):  # the first two calls are the eager warmups of the streamed linear
                hidden = (torch.randn(1, 1, 1, HIDDEN_SIZE, generator=gen) * (0.5 + t * 0.3)).to(torch.bfloat16)
                token = harness.upload_sharded(hidden)
                base = run(token, False)
                fold = run(token, True)
                if t >= 2:
                    for a, b in zip(sum(base, []), sum(fold, [])):
                        bad_total += int((a != b).sum())
                ttnn.deallocate(token)
        print(f"layer {layer_index}: differing words so far {bad_total}")
    assert bad_total == 0
