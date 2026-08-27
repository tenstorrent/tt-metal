# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE: the dense SwiGLU MLP block vs the torch reference.

Block contract (shared with attention — see tests/test_factory.py):

    in :  [1, 1, s, 12288]   full emb, replicated across the TP cols  (a post-norm activation)
    out:  [1, 1, s,  3072]   emb/tp, reduce-scattered across TP       (the sharded residual layout)

``gate_proj`` / ``up_proj`` are column-parallel and stored FUSED as one weight;
``down_proj`` is row-parallel and its partial sum is closed by a reduce-scatter, not an all-reduce.

Target: SP=8 x TP=4 Galaxy

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_dense_mlp_vs_ref.py -k 1x1
      pytest models/demos/mistral_medium_d_p/tests/unit/test_dense_mlp_vs_ref.py -k 8x4   # 32 chips (Galaxy)
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_d_p.reference.torch_reference import swiglu_mlp
from models.demos.mistral_medium_d_p.tt.mlp import MLP

from ..test_factory import gather_tp_shards, mesh_setup, parametrize_mesh_with_fabric, replicate
from .shapes import FFN, HIDDEN, HFConfigStub


def _random_mlp_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "gate_proj.weight": torch.randn(FFN, HIDDEN, generator=g) * 0.02,
        "up_proj.weight": torch.randn(FFN, HIDDEN, generator=g) * 0.02,
        "down_proj.weight": torch.randn(HIDDEN, FFN, generator=g) * 0.02,
    }


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (8, 4)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [512], ids=["s512"])
def test_dense_mlp_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full block vs ``swiglu_mlp``, with the reduce-scattered output reassembled on the host."""
    torch.manual_seed(0)
    mesh_config, ccl = mesh_setup(mesh_device, linear_fabric=True)
    tp = mesh_config.tp
    w = _random_mlp_weights()

    x = torch.randn(1, seq_len, HIDDEN) * 0.1
    ref = swiglu_mlp(x.float(), w["gate_proj.weight"], w["up_proj.weight"], w["down_proj.weight"])

    mlp = MLP(mesh_device, HFConfigStub(), w, ccl, mesh_config=mesh_config, weight_dtype=ttnn.bfloat16)
    x_tt = replicate(x.reshape(1, 1, seq_len, HIDDEN), mesh_device)

    # Signposts bound the perf-measured region (tests/perf/test_mlp_perf.py runs this test under
    # tracy and sums device kernel time between them): the single forward, excluding weight
    # load/tilize at construction and the host->device input write.
    signpost(header="MLP_START")
    out_tt = mlp(x_tt)
    signpost(header="MLP_END")

    # Sharded-residual contract: each chip returns hidden/tp, scattered across the TP cols.
    assert out_tt.shape[-1] == HIDDEN // tp, f"expected emb/tp={HIDDEN // tp} per chip, got {out_tt.shape[-1]}"
    out = gather_tp_shards(out_tt, mesh_device).reshape(1, seq_len, HIDDEN)

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"dense SwiGLU MLP vs ref (TP={tp}, s={seq_len}): {pcc}")
    assert passing, f"MLP PCC fail at TP={tp}: {pcc}"
