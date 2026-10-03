# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-built step state (tt/step_state.py) vs the host-built ``step_inputs`` of the attention classes, several positions."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41Attention, DSV41CompressedAttention
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.step_state import DSV41StepState
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@torch.no_grad()
def test_step_state(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    B = rows * 4
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    cfg = mesh_4x8()
    for lid in (0, 2, 20):  # window, ratio 2, ratio 1
        w = load_layer(lid, with_moe=False)
        meta = w["meta"]
        if meta["ratio"] == 0:
            attn = DSV41Attention(md, cfg, ccl, w["attn"], w["freqs_cis"], max_seq=256)
        else:
            attn = DSV41CompressedAttention(
                md, cfg, ccl, w["attn"], w["freqs_cis"], meta["ratio"], w["compressor"], max_comp=128
            )
        ss = DSV41StepState(attn)
        shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
        worst = 0.0
        for p in (0, 3, 9, 10, 127, 128, 129, 200, 255):
            if meta["ratio"] and (p + 1) // meta["ratio"] > 128:
                continue  # compressed length beyond max_comp is not supported
            positions = torch.full((B,), p)
            host_st = attn.step_inputs(positions)
            pos_dev = ttnn.from_torch(positions.to(torch.int32), device=md, dtype=ttnn.int32, mesh_mapper=shard)
            dev_st = ss.build(pos_dev)
            for k, hv in host_st.items():
                if k in ("slot", "complete"):
                    continue
                a = ttnn.to_torch(ttnn.get_device_tensors(hv)[0]).float()
                b = ttnn.to_torch(ttnn.get_device_tensors(dev_st[k])[0]).float()
                if a.shape != b.shape:
                    print(
                        f"STP layer {lid} pos {p} key {k}: SHAPE host {tuple(a.shape)} dev {tuple(b.shape)}", flush=True
                    )
                    worst = 1e9
                    continue
                d = (a - b).abs().max().item()
                if d > 0:
                    print(f"STP layer {lid} pos {p} key {k}: max|diff| {d}", flush=True)
                worst = max(worst, d)
        print(f"STP layer {lid} (ratio {meta['ratio']}): worst max|diff| over positions/keys = {worst}", flush=True)
        assert worst < 1e-2
