# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x52: the MoE block (router + shared gate|up|scalar linears streamed) with QWEN38_MM_STREAM on against off, bitwise, TP=8 mesh."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, HIDDEN_SIZE
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 0}], indirect=True)
def test_moe_block_stream_vs_stock(tp_harness):
    harness = tp_harness
    builder, layer = harness.build_layer(0)
    mlp = layer.mlp
    gen = torch.Generator().manual_seed(77)
    bad_total = 0
    with builder.expert_streamer.layer(0) as (w01, w2):

        def run(token, on):
            os.environ["QWEN38_MM_STREAM"] = "1" if on else "0"
            r = mlp.forward(token, w01, w2, return_routing=True)
            out = [ttnn.to_torch(t).view(torch.int16) for t in ttnn.get_device_tensors(r.hidden_sharded)]
            idx = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(r.routing.indices)]
            sc = [ttnn.to_torch(t).view(torch.int16) for t in ttnn.get_device_tensors(r.routing.scores)]
            for t in (r.hidden_sharded, r.routing.scores, r.routing.indices):
                ttnn.deallocate(t)
            return out, idx, sc

        for t in range(6):
            hidden = (torch.randn(1, 1, 1, HIDDEN_SIZE, generator=gen) * (0.5 + 0.4 * t)).to(torch.bfloat16)
            token = harness.upload_sharded(hidden)
            run(token, True)  # the shape's first call is the stock op (it learns the output layout)
            a = run(token, False)
            b = run(token, True)
            bad = [sum(int((x != y).sum()) for x, y in zip(a[i], b[i])) for i in range(3)]
            logger.info(f"token {t}: differing words hidden/indices/scores {bad}")
            bad_total += sum(bad)
            ttnn.deallocate(token)
    assert bad_total == 0
