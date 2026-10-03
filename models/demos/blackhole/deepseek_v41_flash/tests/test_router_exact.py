# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Router fidelity on real data: expert-set agreement with the checkpoint's own gate for the kernel (bf16) router and the
exact (fp32 topk) router, on the real FFN inputs of several layers (tests/dump_ffn_inputs.py)."""

import os

import pytest
import torch

import ttnn
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig
from models.demos.blackhole.deepseek_v41_flash.tt import router as router_mod
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import CONFIG_PATH
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-e")
LAYERS = [int(x) for x in os.environ.get("DSV41_ROUTER_LAYERS", "1,2,8,14,20").split(",")]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@torch.no_grad()
def test_router_fidelity(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    sh = _Shards()
    cfg = TTMoEGateConfig.from_yaml(CONFIG_PATH.read_text()).model_copy(update={"batch_per_device": 4})
    shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=(rows, cols))
    for L in LAYERS:
        d = torch.load(os.path.join(CHAIN, f"ffn_inputs_{L}.pt"))
        x, ref_idx, ref_wt = d["x"], d["idx"], d["wt"]
        W = sh.get(f"layers.{L}.ffn.gate.weight").float().T.contiguous()  # [hidden, 384]
        b = sh.get(f"layers.{L}.ffn.gate.bias").float()
        rank = torch.nn.functional.softplus(x.float() @ W).sqrt() + b
        K = float(rank.topk(7, -1).values[:, 5].mean())
        gate = router_mod.DSV41Gate(mesh_device, cfg, W, b, bias_shift=K)
        tt_x = ttnn.from_torch(
            x.reshape(1, 1, 16, 5120),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        for mode in ("kernel", "exact"):
            os.environ["DSV41_ROUTER"] = mode
            tt_w, tt_i = gate.forward(tt_x)
            ttnn.synchronize_device(mesh_device)
            devs_i, devs_w = ttnn.get_device_tensors(tt_i), ttnn.get_device_tensors(tt_w)
            idx = torch.cat([ttnn.to_torch(devs_i[r * cols]).reshape(-1, 6) for r in range(rows)]).long()[:16]
            wt = torch.cat([ttnn.to_torch(devs_w[r * cols]).reshape(-1, 6) for r in range(rows)]).float()[:16]
            same = sum(set(a.tolist()) == set(c.tolist()) for a, c in zip(idx, ref_idx))
            # weights compared per expert id
            errs = []
            for t in range(16):
                dm = {int(e): float(v) for e, v in zip(idx[t], wt[t])}
                errs += [
                    abs(dm[int(e)] - float(v)) / max(abs(float(v)), 1e-6)
                    for e, v in zip(ref_idx[t], ref_wt[t])
                    if int(e) in dm
                ]
            print(
                f"ROUTER layer {L:2d} {mode:6s}: same expert set {same}/16 tokens, weight rel-err mean {sum(errs) / max(len(errs), 1):.4f} max {max(errs or [0]):.4f}"
            )
        os.environ.pop("DSV41_ROUTER", None)
