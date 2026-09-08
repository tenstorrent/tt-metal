# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M4: the context encoder ``hidden_norm(fc(target_hidden))``, real weights + real taps.

The input is the captured fixture's ``target_hidden`` -- the actual Qwen3.6-27B residual
stream at layers [1, 16, 31, 46, 61] -- so this grades the encoder on the exact activation
distribution it will see in production, not on noise.

This is also the first on-device check of ``fc``'s input permutation. ``test_weights.py``
proves the index algebra exactly on CPU; this proves the loader and the sharded matmul agree
with it.

Run:
    MESH_DEVICE=T3K pytest models/demos/blackhole/qwen36/tests/dflash/test_encoder_tp.py -v -s
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.dflash.conftest import load_fixture
from models.demos.blackhole.qwen36.tests.test_factory import get_pcc_threshold, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash.ccl import ccl_topology
from models.demos.blackhole.qwen36.tt.dflash.encoder import DFlashContextEncoder
from models.demos.blackhole.qwen36.tt.dflash.weights import (
    DFlashWeights,
    load_fc,
    permute_fc_input_activation,
    read_state_dict,
)
from models.tt_transformers.tt.ccl import TT_CCL


@torch.no_grad()
@parametrize_mesh_tp()
def test_encoder_tp(mesh_device, reset_seeds, ensure_gc, request, drafter_cfg):
    cfg = drafter_cfg
    tp = mesh_device.get_num_devices()
    fx = load_fixture(512)

    target_hidden = fx["target_hidden"].float()  # [1, ctx, 25600], real target taps
    ctx_len = target_hidden.shape[1]

    sd = read_state_dict(keys=["fc.weight", "hidden_norm.weight"])

    # ---- oracle: the reference's own two ops, with the real weights -------------------
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    ref_norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    with torch.no_grad():
        ref_norm.weight.copy_(sd["hidden_norm.weight"])
    expected = ref_norm(target_hidden @ sd["fc.weight"].float().T)

    # ---- device ----------------------------------------------------------------------
    tt_ccl = TT_CCL(mesh_device)
    weights = DFlashWeights(
        fc=load_fc(mesh_device, cfg, sd),
        hidden_norm=ttnn.from_torch(
            sd["hidden_norm.weight"].reshape(1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        norm=None,
        layers=(),
    )
    encoder = DFlashContextEncoder(mesh_device, cfg, weights, tt_ccl, topology=ccl_topology(mesh_device))

    # Permute the fixture activation into per-chip tap order, then shard contiguously. On
    # device the tap layout already produces this order, so nothing is permuted at inference.
    permuted = permute_fc_input_activation(target_hidden, tp, cfg.hidden_size, len(cfg.target_layer_ids))
    tt_in = ttnn.from_torch(
        permuted.reshape(1, 1, ctx_len, cfg.target_feature_size).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    tt_out = encoder.forward(tt_in)

    # Output is replicated: every device must hold the same full [1,1,ctx,5120].
    stacked = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert stacked.shape[0] == tp, stacked.shape
    actual = stacked[:1].reshape(1, ctx_len, cfg.hidden_size).float()

    max_dev = max((stacked[d : d + 1].float() - stacked[:1].float()).abs().max().item() for d in range(1, tp))
    logger.info(f"replication spread across {tp} devices: {max_dev:.3e}")

    passing, pcc = comp_pcc(expected, actual, get_pcc_threshold(request))
    logger.info(f"encoder (fc + hidden_norm) PCC {pcc}")
    assert passing, f"encoder PCC {pcc}"
