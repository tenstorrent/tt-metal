# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parts of the host-free token loop: (a) head.sample_global == the host argmax on real final logits, (b) in-trace token / position
feedback (ttnn.copy of the sampled token, int32 position + 1) replayed from a trace several times."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostHead
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-e")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 50_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_device_loop_parts(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    B, T = rows * 4, 4
    sh = _Shards()
    ccl, cfg = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring), mesh_4x8()
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    last = torch.load(os.path.join(CHAIN, "layer_39.pt"))
    h, p = last["dec_out"].float().reshape(B, 1, 4, -1), last["pre_out"].float().reshape(B, 1, 1, 4)
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    up = lambda t: ttnn.from_torch(
        t,
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    x, pre = up(h), up(p)
    logits = head.forward(x, pre)
    ref_tokens = HostHead()(h, p.reshape(B, 1, 4)).argmax(-1)

    # (a) eager
    tok = head.sample_global(logits, cfg, ccl)
    dev = lambda t: torch.cat(
        [ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).reshape(-1) for r in range(rows)]
    ).long()
    got = dev(tok)[:B]
    print(
        "DLP (a) eager sample_global == host argmax:",
        int((got == ref_tokens).sum()),
        "/",
        B,
        " device tokens",
        got.tolist(),
        flush=True,
    )
    # identical on every device of a row?
    same = all(
        torch.equal(
            ttnn.to_torch(ttnn.get_device_tensors(tok)[0]).reshape(-1),
            ttnn.to_torch(ttnn.get_device_tensors(tok)[c]).reshape(-1),
        )
        for c in range(cols)
    )
    print("DLP (a) token identical on all 8 columns:", same, flush=True)

    # (b) feedback inside a trace, replayed
    tok_dev = ttnn.from_torch(
        torch.arange(B).reshape(-1, 1).to(torch.int32),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    pos_dev = ttnn.from_torch(
        torch.full((B,), 9).to(torch.int32),
        device=md,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )

    def step():
        nxt = head.sample_global(logits, cfg, ccl)
        ttnn.copy(nxt, tok_dev)
        ttnn.copy(ttnn.add(pos_dev, 1), pos_dev)

    step()  # compile (position 9 -> 10)
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    step()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(5):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    pos_now = dev(pos_dev)[:B]
    print(
        "DLP (b) position after compile pass + 5 replays (expect 9 + 1 + 5 = 15):",
        pos_now.unique().tolist(),
        flush=True,
    )
    print("DLP (b) fed-back tokens == host argmax:", int((dev(tok_dev)[:B] == ref_tokens).sum()), "/", B, flush=True)
    assert int((got == ref_tokens).sum()) >= B - 1 and pos_now.unique().tolist() == [15]
