"""x46 probe: marginal traced cost of moe_post after moe_experts (launch gap included) on one die."""
import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import load_layer_shard, traced_wall_us
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_moe_experts_device import (
    BF16,
    LAYER,
    SHARD,
    Setup,
    make_routing,
    random_hidden,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_experts as me
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_post


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_probe(mesh_device):
    mesh = mesh_device
    weights = load_layer_shard(mesh, LAYER, SHARD)
    for n in (1, 3):
        setup = Setup(mesh, 1, weights=weights)
        gen = torch.Generator().manual_seed(5)
        idx, sc = make_routing(gen, 1, SHARD, (n,), 0)
        hidden = random_hidden(gen, 1)
        h = setup.hidden_l1(hidden)
        i_l1, s_l1 = setup.routing_tensors(idx, sc)
        shared = setup.upload(torch.randn(1, 1, 1, 2560).to(torch.bfloat16), BF16, ttnn.TILE_LAYOUT)
        sig = setup.upload(torch.rand(1, 1, 32, 32).to(torch.bfloat16), BF16, ttnn.TILE_LAYOUT)
        excl = me.model_exclude(mesh)

        def a():
            me.moe_experts(h, i_l1, setup.owner_row, setup.w01, setup.w2, setup.combine, exclude=excl)

        def b():
            a()
            ttnn.deallocate(moe_post.moe_post(setup.combine, s_l1, i_l1, setup.owner_row, shared, sigmoid=sig))

        def c():
            ttnn.deallocate(moe_post.moe_post(setup.combine, s_l1, i_l1, setup.owner_row, shared, sigmoid=sig))

        r = {
            k: round(traced_wall_us(mesh, f, replays=100)[0], 2)
            for k, f in (("experts", a), ("experts_then_post", b), ("post", c))
        }
        r["n"] = n
        Path(os.environ.get("X37_RESULTS", "/tmp/probe.jsonl")).open("a").write(json.dumps(r) + "\n")
        print("PROBE", r)
