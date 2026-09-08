# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Talker MLP decode PCC on a 2-chip mesh (TP=2), for both gate/up grid arms.

``test_qwen3_tts_pcc.py`` opens ``ttnn.open_device(device_id=0)``, so every test in it
runs ``tp_size=1`` and the TP=2 path — column-parallel gate/up, row-parallel down, the
all_reduce that sums the partial products — has never had a PCC gate at all. That gap is
what made the N300 arm of ``_DECODE_GATE_UP_CORES`` look unverifiable.

This closes it: same shapes and dtype the demo runs, both grid arms in one process,
each scored against the same full-precision torch reference.

    TT_VISIBLE_DEVICES=1 MESH_DEVICE=N300 \\
      TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \\
      python -m pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_mlp_tp2_pcc.py
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn

HIDDEN, INTERMEDIATE = 2048, 6144
# Floor, not a target: TP=1 mlp_decode gates at 0.9947 for the same shapes, and TP=2
# adds one cross-chip reduction on top of the same arithmetic.
PCC_MIN = 0.9947


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), l1_small_size=32768)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_mlp_decode_tp2_grid_arms(mesh):
    from models.demos.qwen3_tts.reference.functional import swiglu_mlp as torch_swiglu_mlp
    from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size
    from models.demos.qwen3_tts.tt.mlp import MLP

    tp = get_tp_size(mesh)
    assert tp == 2, f"this test needs a 2-chip mesh; got tp_size={tp}"

    torch.manual_seed(0)
    x_torch = torch.randn(1, 1, 1, HIDDEN, dtype=torch.bfloat16)
    g = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
    u = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
    d = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)
    ref = torch_swiglu_mlp(x_torch.squeeze(1), g, u, d)

    sd = {
        "test_layer.mlp.gate_proj.weight": g,
        "test_layer.mlp.up_proj.weight": u,
        "test_layer.mlp.down_proj.weight": d,
    }

    prev = os.environ.get("QWEN3_TTS_DECODE_GATE_UP_CORES")
    results = {}
    try:
        for label, cores in (("find_grid_k_n (32 cores, in0_block_w=2)", "0"), ("swept (8 cores, in0_block_w=8)", "8")):
            os.environ["QWEN3_TTS_DECODE_GATE_UP_CORES"] = cores
            # bfloat8_b: what talker.py builds by default since QWEN3_TTS_BF8_WEIGHTS
            # became default ON.
            mlp = MLP(mesh, HIDDEN, INTERMEDIATE, sd, "test_layer", weight_dtype=ttnn.bfloat8_b)
            got = mlp._decode_gate_up_swept_cores
            x_tt = ttnn.from_torch(
                x_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            y_tt = mlp(x_tt, mode="decode")
            # all_reduce leaves every chip holding the same sum; score chip 0.
            y = ttnn.to_torch(y_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0:1].squeeze(1)
            pcc = _pcc(ref, y)
            rel_rms = float(
                (y.to(torch.float32) - ref.to(torch.float32)).pow(2).mean().sqrt()
                / ref.to(torch.float32).pow(2).mean().sqrt()
            )
            results[label] = (pcc, rel_rms, got, y.to(torch.float32).clone())
            print(f"  [{label}] swept_cores={got}  PCC = {pcc:.8f}  rel_rms = {rel_rms*100:.3f} %")
            ttnn.deallocate(x_tt)
            ttnn.deallocate(y_tt)
    finally:
        if prev is None:
            os.environ.pop("QWEN3_TTS_DECODE_GATE_UP_CORES", None)
        else:
            os.environ["QWEN3_TTS_DECODE_GATE_UP_CORES"] = prev

    keys = list(results)
    base, new = results[keys[0]], results[keys[1]]
    arm_pcc = _pcc(base[3], new[3])
    print(f"\n  baseline vs swept, arm-to-arm PCC = {arm_pcc:.8f}")
    print(f"  PCC vs torch: {base[0]:.8f} -> {new[0]:.8f}  (floor {PCC_MIN})")

    assert base[2] is None, "cores=0 should keep find_grid_k_n's grid"
    assert new[2] == 8, f"cores=8 should have been applied; got {new[2]}"
    for label, (pcc, _r, _c, _y) in results.items():
        assert pcc > PCC_MIN, f"[{label}] PCC {pcc:.6f} <= {PCC_MIN}"
