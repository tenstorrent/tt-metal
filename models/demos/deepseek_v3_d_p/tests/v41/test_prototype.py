# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 §4 minimal prefill prototype vs the CPU oracle (real dims, synthetic weights).

Prototype layers 0..4 stand for V4.1 layers 2, 3, 20, 21, 24 (``prototype_oracle``). Each block runs on
device with the oracle's inputs (teacher-forced streams and pre_mix); the compressed-KV path runs on the
host fallback, which consumes the device's activations and carries the sharing state from each source to
its consumers in execution order. Per layer it records PCC of the attention (oracle attention input),
the MoE (oracle FFN input), the block output streams and the next pre_mix, and checks bit-identical
repeats. The acceptance bars are the epic's block bars; this test reports and asserts only finiteness
and determinism, plus a loose functional floor — prototype dispositions are recorded, not gated.
"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41 import prototype_oracle as po
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.host_fallback import HostCompressedAttention
from tests.ttnn.utils_for_testing import comp_pcc

FUNCTIONAL_FLOOR = 0.9  # prototype sanity floor, not the acceptance bar (block >= 0.99 synthetic)
REPORT = os.environ.get("TT_V41_PROTOTYPE_REPORT")


def _pack_streams(x, tp):
    """[S, n, D] -> [1, 1, S, n*D], each TP chip's hidden slice of every stream contiguous."""
    s, n, d = x.shape
    return x.reshape(s, n, tp, d // tp).permute(0, 2, 1, 3).reshape(1, 1, s, n * d)


def _unpack_streams(t, n, tp):
    s, width = t.shape[-2], t.shape[-1]
    d = width // n
    return t.reshape(s, tp, n, d // tp).permute(0, 2, 1, 3).reshape(s, n, d)


def _pcc(expected, actual):
    return comp_pcc(expected.float(), actual.float(), 0.0)[1]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        )
    ],
    indirect=True,
)
def test_v41_prototype(mesh_device, device_params):
    torch.manual_seed(0)
    cfg = po.PrototypeScheduleConfig
    args = po.model_args()
    reference = po.build_reference(args)
    rec = po.oracle(reference, args)
    host = HostCompressedAttention(reference)
    topology = per_axis_topology(device_params["fabric_config"])[1]
    shape = tuple(mesh_device.shape)
    sp, tp = shape
    n = cfg.HC_MULT

    def up(t, dims, dtype=ttnn.float32):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=dims),
        )

    def down(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))

    report = {}
    for layer in range(cfg.NUM_LAYERS):
        v41_layer = po.V41_LAYER[layer]
        logger.info(f"prototype layer {layer} (V4.1 layer {v41_layer}, {cfg.block_type(layer).value})")
        block = TtV41Block(
            mesh_device, cfg, layer, po.device_weights(reference, layer), po.SEQ, host, topology=topology
        )

        # Isolated sublayers on the oracle's inputs first: the attention's host path rewrites the shared
        # compressed-KV state, and the block runs below must leave the device-driven state for consumers.
        attn_in = up(rec[f"block{layer}.attn_in"][None, None], (2, 3), ttnn.bfloat16)
        attn_out = down(block.attn(attn_in))[0, 0]
        ffn_in = up(rec[f"block{layer}.ffn_in"][None, None], (2, 3), ttnn.bfloat16)
        ffn_out = down(block._moe(ffn_in))[0, 0]

        x = up(_pack_streams(rec[f"block{layer}.x_in"].float(), tp), (2, 3))
        pre = up(rec[f"block{layer}.pre_in"].float()[None, None], (2, None))

        outs = []
        for _ in range(2):
            x_out, pre_out = block(x, pre)
            outs.append((_unpack_streams(down(x_out)[0, 0], n, tp), down(pre_out)[0, 0, :, :n]))
        (x0, p0), (x1, p1) = outs
        assert torch.isfinite(x0).all() and torch.isfinite(p0).all(), f"layer {layer}: non-finite block output"
        deterministic = torch.equal(x0, x1) and torch.equal(p0, p1)

        entry = {
            "v41_layer": v41_layer,
            "block_type": cfg.block_type(layer).value,
            "attention_pcc": _pcc(rec[f"block{layer}.attn_out"], attn_out),
            "moe_pcc": _pcc(rec[f"block{layer}.ffn_out"], ffn_out),
            "block_out_pcc": _pcc(rec[f"block{layer}.x_out"], x0),
            "pre_mix_pcc": _pcc(rec[f"block{layer}.pre_out"], p0),
            "deterministic": deterministic,
        }
        logger.info(f"layer {layer}: {entry}")
        report[layer] = entry
        del block

    if REPORT:
        with open(REPORT, "w") as f:
            json.dump(report, f, indent=2)
    for layer, entry in report.items():
        assert entry["deterministic"], f"layer {layer}: repeated block runs differ"
        assert entry["block_out_pcc"] >= FUNCTIONAL_FLOOR, f"layer {layer}: {entry}"
