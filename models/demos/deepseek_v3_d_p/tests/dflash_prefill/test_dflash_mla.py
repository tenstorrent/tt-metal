# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: MLA (DFlash2) drafter context latent (``TtDFlashMlaDrafter``) vs the torch port of blaze's ``latent_kv``.

    MESH_DEVICE=8x4 pytest models/demos/deepseek_v3_d_p/tests/dflash_prefill/test_dflash_mla.py -svv

Random (seeded, checkpoint-convention) weights for the prefill subset; the K3 DFlash2 preset config. Covers a
single chunk at offset 0 (finds rope/layout bugs first), multi-chunk streaming, a mid-sequence resume
(multi-turn: the first chunk is NOT rewritten), a non-zero slot, and both the TP-replicated and TP-deduped
latent cache. c_kv (cols 0-511) is checked before k_pe (512-575): c_kv never touches RoPE, so
c_kv-passes-while-k_pe-fails points at the rope table or pairing, not at the FC / taps / norm.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.dflash_prefill.mla_latent import (
    context_latents,
    synthetic_prefill_state_dict,
)
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import kimi_k3_dflash2_mla_config
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_kv_validation import read_slot_dflash_latent
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.tt_dflash_mla_drafter import TtDFlashMlaDrafter
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_dflash_mla_kv_cache
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD = 0.999
# PCC is scale-invariant; a missing YaRN amplitude or a gamma off by a constant would still pass it.
SCALE_TOL = 0.01
CHUNK_GLOBAL = PREFILL_CHUNK_TOKENS


def _tap_chunk(mesh_device, drafter, cfg, taps, lo, chunk_global):
    """Feed one chunk's taps, TP-sharded on hidden and SP-sharded on seq (each chip taps its own slice)."""
    mesh_shape = tuple(mesh_device.shape)
    dims = [None, None]
    dims[1] = 3
    dims[0] = 2
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims)
    H = cfg.hidden_size
    drafter.reset()
    for tid, tap in zip(cfg.target_layer_ids, taps):
        h = tap[lo : lo + chunk_global].to(torch.bfloat16).reshape(1, 1, chunk_global, H)
        h_tt = ttnn.from_torch(
            h,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        drafter.tap(h_tt, tid)
        ttnn.deallocate(h_tt)


def _check(expected, actual, tag):
    r = expected.shape[-1] - 64
    for i in range(expected.shape[0]):
        for part, cols in (("c_kv", slice(0, r)), ("k_pe", slice(r, None))):
            e, a = expected[i, :, cols], actual[i, :, cols]
            ok, pcc = comp_pcc(e, a, PCC_THRESHOLD)
            scale = (a.norm() / e.norm()).item()
            logger.info(f"{tag} layer {i} {part}: pcc={pcc} scale={scale:.4f}")
            assert ok, f"{tag} layer {i} {part}: PCC {pcc} < {PCC_THRESHOLD}"
            assert abs(scale - 1.0) <= SCALE_TOL, f"{tag} layer {i} {part}: ||dev||/||ref|| = {scale:.4f}"


@pytest.mark.timeout(0)
@pytest.mark.parametrize("tp_shard_kv", [False, True], ids=["tp-replicated", "tp-dedup"])
@pytest.mark.parametrize(
    "n_chunks, first_chunk, slot_idx, num_users",
    [
        pytest.param(1, 0, 0, 1, id="1chunk-offset0"),
        pytest.param(2, 0, 0, 1, id="2chunk"),
        pytest.param(3, 1, 0, 1, id="resume-at-chunk1"),
        pytest.param(2, 0, 1, 2, id="slot1-of-2"),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dflash_mla_latent_pcc(
    mesh_device, device_params, num_links, tp_shard_kv, n_chunks, first_chunk, slot_idx, num_users
):
    topology = per_axis_topology(device_params["fabric_config"])[1]
    cfg = kimi_k3_dflash2_mla_config()
    sd = synthetic_prefill_state_dict(cfg)
    ctx_len = n_chunks * CHUNK_GLOBAL

    gen = torch.Generator().manual_seed(0)
    taps = [torch.randn(ctx_len, cfg.hidden_size, generator=gen) for _ in cfg.target_layer_ids]
    expected = context_latents(sd, taps, cfg)  # [layers, ctx_len, 576]

    drafter = TtDFlashMlaDrafter(
        mesh_device,
        cfg,
        state_dict=sd,
        sp_axis=0,
        tp_axis=1,
        max_seq_len=ctx_len,
        chunk_size=CHUNK_GLOBAL,
        num_links=num_links,
        topology=topology,
        tp_shard_kv=tp_shard_kv,
    )
    cache = allocate_dflash_mla_kv_cache(
        mesh_device, cfg, ctx_len, num_users=num_users, tp_shard_kv=tp_shard_kv, bank_aligned=False
    )
    for c in range(first_chunk, n_chunks):
        lo = c * CHUNK_GLOBAL
        _tap_chunk(mesh_device, drafter, cfg, taps, lo, CHUNK_GLOBAL)
        drafter.forward(cache, lo, slot_idx=slot_idx, actual_end=lo + CHUNK_GLOBAL)
    ttnn.synchronize_device(mesh_device)

    actual = read_slot_dflash_latent(
        mesh_device,
        cache,
        chunk_size_global=CHUNK_GLOBAL,
        num_layers=cfg.num_hidden_layers,
        slot_id=slot_idx,
        out_len=ctx_len,
    )
    lo = first_chunk * CHUNK_GLOBAL
    if lo:
        # A resumed slot must leave the prefix it did not rewrite untouched (here: still zero).
        assert actual[:, :lo].abs().max() == 0, "resume wrote below its kv_actual_global"
    _check(expected[:, lo:], actual[:, lo:], f"slot{slot_idx}")
