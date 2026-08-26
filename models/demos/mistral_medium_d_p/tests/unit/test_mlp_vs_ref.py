# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE: the dense SwiGLU MLP block vs the torch reference.

Block contract (shared with attention — see tests/test_factory.py):

    in :  [1, 1, s, 12288]   full emb, replicated across the TP cols  (a post-norm activation)
    out:  [1, 1, s,  3072]   emb/tp, reduce-scattered across TP       (the sharded residual layout)

``gate_proj`` / ``up_proj`` are column-parallel (28672/4 = 7168 per chip) and stored FUSED as one
``[12288, 2*7168]`` weight; ``down_proj`` is row-parallel (contracts 7168 per chip) and its partial
sum is closed by a reduce-scatter, not an all-reduce.

**TP=4 is the case that matters here.** The fused gate|up weight must be built so device *i* holds
``[gate_i | up_i]`` contiguously; a naive ``cat([gate, up], -1)`` would hand device 0
``[gate_0 | gate_1]`` instead. At TP=1 there is only one shard, so that bug is invisible — which is
exactly why `1x4` is in the parametrize list and not just `1x1`.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_mlp_vs_ref.py -k 1x1
      pytest models/demos/mistral_medium_d_p/tests/unit/test_mlp_vs_ref.py -k 1x4   # 4 chips
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_d_p.config import MeshConfig
from models.demos.mistral_medium_d_p.reference.torch_reference import swiglu_mlp
from models.demos.mistral_medium_d_p.tt.mlp import MLP

from ..test_factory import gather_tp_shards, mesh_setup, parametrize_mesh_with_fabric, replicate
from .shapes import FFN, HIDDEN, HFConfigStub, per_chip


def _random_mlp_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "gate_proj.weight": torch.randn(FFN, HIDDEN, generator=g) * 0.02,
        "up_proj.weight": torch.randn(FFN, HIDDEN, generator=g) * 0.02,
        "down_proj.weight": torch.randn(HIDDEN, FFN, generator=g) * 0.02,
    }


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
@pytest.mark.parametrize("seq_len", [512], ids=["s512"])
def test_mlp_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full block vs ``swiglu_mlp``, with the reduce-scattered output reassembled on the host."""
    torch.manual_seed(0)
    mesh_config, ccl = mesh_setup(mesh_device)
    tp = mesh_config.tp
    w = _random_mlp_weights()

    x = torch.randn(1, seq_len, HIDDEN) * 0.1
    ref = swiglu_mlp(x.float(), w["gate_proj.weight"], w["up_proj.weight"], w["down_proj.weight"])

    mlp = MLP(mesh_device, HFConfigStub(), w, ccl, mesh_config=mesh_config, weight_dtype=ttnn.bfloat16)
    out_tt = mlp(replicate(x.reshape(1, 1, seq_len, HIDDEN), mesh_device))

    # Sharded-residual contract: each chip returns hidden/tp, scattered across the TP cols.
    assert out_tt.shape[-1] == HIDDEN // tp, f"expected emb/tp={HIDDEN // tp} per chip, got {out_tt.shape[-1]}"
    out = gather_tp_shards(out_tt, mesh_device).reshape(1, seq_len, HIDDEN)

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"dense SwiGLU MLP vs ref (TP={tp}, s={seq_len}): {pcc}")
    assert passing, f"MLP PCC fail at TP={tp}: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_fused_gate_up_shard_is_per_device_interleaved(mesh_device, device_params, reset_seeds):
    """Device *i* must hold ``[gate_i | up_i]``, not a slice of ``cat([gate, up])``.

    Checked directly on the built weight rather than only through PCC, so a failure names the cause.
    Multiply the whole fused weight by a one-hot input to read back its rows.
    """
    mesh_config, ccl = mesh_setup(mesh_device)
    tp = mesh_config.tp
    i_local = per_chip(tp)["ffn"]
    w = _random_mlp_weights(seed=7)
    mlp = MLP(mesh_device, HFConfigStub(), w, ccl, mesh_config=mesh_config, weight_dtype=ttnn.bfloat16)

    per_dev = ttnn.get_device_tensors(mlp.w13)
    gate_t, up_t = w["gate_proj.weight"].t(), w["up_proj.weight"].t()  # [H, FFN]
    for dev_idx in range(tp):
        got = ttnn.to_torch(per_dev[dev_idx]).reshape(HIDDEN, 2 * i_local)
        want = torch.cat(
            [
                gate_t[:, dev_idx * i_local : (dev_idx + 1) * i_local],
                up_t[:, dev_idx * i_local : (dev_idx + 1) * i_local],
            ],
            dim=-1,
        )
        passing, pcc = comp_pcc(want, got, 0.999)
        assert passing, (
            f"device {dev_idx}'s fused gate|up shard is wrong (pcc {pcc}): it must be "
            f"[gate[:, {dev_idx * i_local}:{(dev_idx + 1) * i_local}] | up[:, same]], not a slice of cat([gate, up])"
        )
    logger.info(f"fused gate|up interleave correct on all {tp} devices")


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_mlp_rejects_clamped_swiglu(mesh_device, device_params, reset_seeds, expect_error):
    """gpt-oss / M3 ship 'swigluoai'; running it through plain silu would be silently wrong."""
    with expect_error(NotImplementedError, "swiglu"):
        MLP(
            mesh_device,
            HFConfigStub(hidden_act="swigluoai"),
            {},
            None,
            mesh_config=MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1]),
        )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_mlp_rejects_bias(mesh_device, device_params, reset_seeds, expect_error):
    """MistralMLP is bias-free; a stray bias must fail loud rather than be dropped."""
    w = _random_mlp_weights()
    w["down_proj.bias"] = torch.zeros(HIDDEN)
    with expect_error(AssertionError, "bias-free"):
        MLP(
            mesh_device,
            HFConfigStub(),
            w,
            None,
            mesh_config=MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1]),
        )
