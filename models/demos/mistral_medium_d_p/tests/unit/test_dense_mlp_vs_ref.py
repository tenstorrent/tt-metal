# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE: the dense SwiGLU MLP block vs the torch reference.

Block contract (shared with attention — see tests/test_factory.py):

    in :  [1, 1, s_local, 12288]   full emb, replicated across the TP cols (a post-norm
                                   activation); the GLOBAL sequence is SP-sharded across the
                                   mesh rows, s_local = seq_len / sp
    out:  [1, 1, s_local,  3072]   emb/tp, reduce-scattered across TP (the sharded residual layout)

``gate_proj`` / ``up_proj`` are column-parallel and stored FUSED as one weight;
``down_proj`` is row-parallel and its partial sum is closed by a reduce-scatter, not an all-reduce.

The MLP is pointwise in the sequence, so SP costs it no collective — but the input here IS
SP-sharded (every row gets a DIFFERENT token chunk, unlike ``replicate``), so the test pins the
row<->chunk mapping end to end: output on row r / col c must equal ref[token chunk r][emb shard c].

``seq_len`` is the GLOBAL sequence; each row computes s_local = seq_len / sp of it (5k/8 = 640
tokens per chip on the Galaxy target).

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

from ..test_factory import mesh_setup, parametrize_mesh_with_fabric
from .shapes import FFN, HIDDEN, HFConfigStub


def _random_mlp_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "gate_proj.weight": torch.randn(FFN, HIDDEN, generator=g) * 0.02,
        "up_proj.weight": torch.randn(FFN, HIDDEN, generator=g) * 0.02,
        "down_proj.weight": torch.randn(HIDDEN, FFN, generator=g) * 0.02,
    }


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (8, 4)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [5 * 1024], ids=["s5k"])  # GLOBAL; each row computes seq_len / sp
def test_dense_mlp_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full block vs ``swiglu_mlp``, with the SP x TP sharded output reassembled on the host."""
    torch.manual_seed(0)
    mesh_config, ccl = mesh_setup(mesh_device, linear_fabric=True)
    tp, sp = mesh_config.tp, mesh_config.sp
    rows, cols = mesh_config.mesh_shape
    w = _random_mlp_weights()

    s_local = seq_len // sp
    assert seq_len % (sp * 32) == 0, f"seq_len={seq_len} must split into tile-aligned SP={sp} shards"

    x = torch.randn(1, seq_len, HIDDEN) * 0.1
    ref = swiglu_mlp(x.float(), w["gate_proj.weight"], w["up_proj.weight"], w["down_proj.weight"])

    mlp = MLP(mesh_device, HFConfigStub(), w, ccl, mesh_config=mesh_config, weight_dtype=ttnn.bfloat8_b)
    # The real prefill layout: sequence (dim 2) SP-sharded across the mesh rows, replicated across
    # the TP cols — each row sees its own s_local token chunk.
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, HIDDEN),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.shard_mapper(mesh_device, mesh_dims=(2, None)),
    )

    # Signposts bound the perf-measured region (tests/perf/test_mlp_perf.py runs this test under
    # tracy and sums device kernel time between them): the single forward, excluding weight
    # load/tilize at construction and the host->device input write.
    signpost(header="MLP_START")
    out_tt = mlp(x_tt)
    signpost(header="MLP_END")

    # Sharded-residual contract: each chip returns [1, 1, s_local, hidden/tp].
    assert out_tt.shape[-2] == s_local, f"expected s_local={s_local} per chip, got {out_tt.shape[-2]}"
    assert out_tt.shape[-1] == HIDDEN // tp, f"expected emb/tp={HIDDEN // tp} per chip, got {out_tt.shape[-1]}"

    # Reassemble the 2D-sharded output: rows concat on the sequence dim (SP), cols on emb (TP).
    out = ttnn.to_torch(
        out_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(2, 3)),
    ).reshape(1, seq_len, HIDDEN)

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"dense SwiGLU MLP vs ref (SP={sp}, TP={tp}, s_local={s_local}, S={seq_len}): {pcc}")
    assert passing, f"MLP PCC fail at SP={sp}/TP={tp}: {pcc}"
