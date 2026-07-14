# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone PCC bring-up test for the DFlash drafter context-KV prefill module (issue #49586, Phase 1).

Drives ``TtDFlashDrafterKV`` in isolation (no full verifier prefill) with the SAME synthetic
context feature the torch golden uses, and PCCs the device K/V drafter cache against
``torch_dflash_golden``. This is the Phase-1a milestone: prove the drafter KV *math + sharding*
before wiring the FC taps into the real ``TtPrefillTransformer`` layer loop (Phase-1b).

Run (on a Blackhole galaxy; set the drafter weights path):

    DFLASH_DRAFTER_WEIGHTS=/path/to/Kimi-K2.6-DFlash/model.safetensors \
    MESH_DEVICE=<e.g. 2x4 or 8x4> \
    pytest models/demos/deepseek_v3_d_p/tests/dflash/test_dflash_prefill.py -svv

NOTE (bring-up): not yet run on hardware. Expect to iterate on the head-split (`_split_heads`),
matmul program configs, and the readback mesh composer. See TODO markers in
``tt/dflash/tt_dflash_drafter_kv.py``.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tests.dflash import torch_dflash_golden as golden
from models.demos.deepseek_v3_d_p.tt.dflash.dflash_drafter_config import DFlashDrafterConfig
from models.demos.deepseek_v3_d_p.tt.dflash.tt_dflash_drafter_kv import TtDFlashDrafterKV
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from tests.ttnn.utils_for_testing import comp_pcc

WEIGHTS_ENV = "DFLASH_DRAFTER_WEIGHTS"
# bf16 device path vs fp32 golden through 2 matmuls (7168 contraction) + norms + rope: start loose,
# tighten once bring-up is clean.
PCC_THRESHOLD = 0.98


# Mesh + fabric params mirror test_prefill_block.py so this launches the same way on the galaxy.
# FABRIC_1D is needed because _tp_all_reduce does an all_gather across the TP axis.
_FABRIC_1D = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
}


@pytest.mark.parametrize("seq_len", [128, 512], ids=["seq128", "seq512"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 4),
            _FABRIC_1D,
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        ),
        pytest.param(
            (8, 4),
            _FABRIC_1D,
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dflash_drafter_kv_pcc(mesh_device, device_params, num_links, topology, seq_len):
    weights = os.environ.get(WEIGHTS_ENV)
    if not weights or not os.path.exists(weights):
        pytest.skip(f"set {WEIGHTS_ENV}=/path/to/Kimi-K2.6-DFlash/model.safetensors")

    cfg = DFlashDrafterConfig()
    mesh_shape = tuple(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    tp = mesh_shape[tp_axis]
    assert cfg.num_key_value_heads % tp == 0, f"num_kv_heads {cfg.num_key_value_heads} not divisible by tp {tp}"
    H = cfg.hidden_size

    # ---- golden (torch): random context feature -> reduced -> target_hidden -> per-layer K/V ----
    ref = golden.generate_reference(weights, seq_len)
    ctx = ref["context_feature"]  # [1, S, 6*H] fp32 — the SAME input we feed the device
    gk = ref["k"][:, 0]  # [num_layers, kv_heads, S, head_dim]
    gv = ref["v"][:, 0]

    # ---- device ----
    from safetensors.torch import load_file

    sd = load_file(weights)  # full drafter state dict; the module selects the 20-tensor subset
    drafter = TtDFlashDrafterKV(
        mesh_device,
        cfg,
        state_dict=sd,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        max_seq_len=cfg.context_len,
        num_links=num_links,
        topology=topology,
    )

    # Feed each of the 6 target-layer hiddens as an FC tap: [1,1,S,H] TP-sharded on hidden (dim 3),
    # SP-replicated (Phase-1 seq-contiguous, unsharded).
    hidden_shard = [None, None]
    hidden_shard[tp_axis] = 3
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=hidden_shard)

    drafter.reset()
    for j, tid in enumerate(cfg.target_layer_ids):
        h_j = ctx[:, :, j * H : (j + 1) * H].to(torch.bfloat16).reshape(1, 1, seq_len, H)
        h_tt = ttnn.from_torch(
            h_j,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        drafter.tap(h_tt, tid)
    drafter.write_kv_cache()
    ttnn.synchronize_device(mesh_device)

    # Readback: TP shards (kv-head) concat on dim 1; SP replicas concat on dim 0 (layers) -> take replica 0.
    def _read(cache):
        host = ttnn.to_torch(
            cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_shape)
        )
        host = host[: cfg.num_hidden_layers]  # first SP replica
        return host[:, :, :seq_len, :].to(torch.float32)

    dk = _read(drafter.k_cache)  # [num_layers, kv_heads, S, head_dim]
    dv = _read(drafter.v_cache)

    worst = 1.0
    for i in range(cfg.num_hidden_layers):
        ok_k, pcc_k = comp_pcc(gk[i], dk[i], PCC_THRESHOLD)
        ok_v, pcc_v = comp_pcc(gv[i], dv[i], PCC_THRESHOLD)
        logger.info(f"draft layer {i}: K pcc={pcc_k} (ok={ok_k})  V pcc={pcc_v} (ok={ok_v})")
        assert ok_k, f"K drafter layer {i} PCC {pcc_k} < {PCC_THRESHOLD}"
        assert ok_v, f"V drafter layer {i} PCC {pcc_v} < {PCC_THRESHOLD}"
