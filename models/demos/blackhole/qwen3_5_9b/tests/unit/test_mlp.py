# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TP SwiGLU MLP vs the HF Qwen3_5MLP golden.

Mirrors the gemma4 unit-test structure: one PCC test parametrized over the
production activation shapes — decode (32 rows = max batch) and the prefill
buckets. The sibling tests/test_mlp_tp.py only covers 32 rows with a
hand-written torch reference; this adds the HF module as golden and the long
prefill shapes where the matmul program selection differs.

Run:
    HF_MODEL=/home/ttuser/models/Qwen3.5-27B-FP8 \
      pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_mlp.py -v
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tests.test_mlp_tp import _load_one_layer_mlp
from models.demos.blackhole.qwen3_5_9b.tests.unit.reference import hf_mlp, model_path, text_config
from models.demos.blackhole.qwen3_5_9b.tt.mlp import Qwen35MLP
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs

# 32 = decode batch shape [1,1,B,dim]; the rest are prefill buckets [1,1,S,dim].
N_ROWS = [32, 128, 512, 2048]


@torch.no_grad()
# Pair each mesh shape with the device_params it needs (a cartesian product would
# force FABRIC_1D onto the unit mesh too, where the single fabric router has no
# ethernet partner and the open times out). The single-device forward returns
# before tt_all_reduce, so it needs no fabric; only the (1,4) TP reduce-scatter does.
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        ((1, 1), {}),
        ((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=True,
)
@pytest.mark.parametrize("n_rows", N_ROWS, ids=lambda n: f"rows{n}")
def test_mlp_pcc(mesh_device, n_rows, reset_seeds, ensure_gc):
    mp = model_path()
    os.environ.setdefault("HF_MODEL", mp)  # Qwen35ModelArgs resolves the checkpoint from HF_MODEL
    args = Qwen35ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()

    # One layer's gate/up/down only — the full-model dequant is a host-OOM hazard.
    mlp_state = _load_one_layer_mlp(mp, 0)

    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    mlp = Qwen35MLP(mesh_device, mlp_state, None, args=args, tt_ccl=tt_ccl)

    x = torch.randn(1, 1, n_rows, args.dim, dtype=torch.float32)
    ref = hf_mlp(text_config(mp), mlp_state)(x[0, 0])  # [n_rows, dim]

    x_tt = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = mlp.forward(x_tt)
    # Output is hidden-dim-fractured on TP (reduce-scatter), full on one device.
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0)
    out_torch = ttnn.to_torch(out, mesh_composer=composer)[0, 0].float()

    passing, pcc = comp_pcc(ref, out_torch, 0.99)
    logger.info(f"MLP PCC (rows={n_rows}) = {pcc}")
    assert passing, f"MLP PCC too low at rows={n_rows}: {pcc}"
