# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One MiMo decoder layer on the Galaxy as TP=8 x DP=4: four independent 1x8 replicas.

Each replica gets its own users (4 per replica, different prompt lengths, some past the 128-token
window). It runs paged prefill for every user, then a few batched decode steps, and every output
is compared with the fp32 torch reference. The four replicas are enqueued back to back and read
afterwards, so they run concurrently on their rows.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.reference import model as ref
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.common import (
    DP,
    GALAXY_DEVICE_PARAMS,
    block_page_table,
    cache_dir,
    dp_submeshes,
    load_config_and_ckpt,
    pcc,
)
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.common import MeshPlan
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.decoder import TtMiMoDecoderLayer
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.rope import MiMoRotarySetup

MAX_SEQ = 1024
MAX_BATCH = 32
BLOCK = 64
USERS_PER_REPLICA = 4
DECODE_STEPS = 3
HIDDEN_STD = 0.5
KV_DTYPE = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("MIMO_KV_DTYPE", "bfp8")]


def to_tt(x, plan):
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape[-2:]).bfloat16(),
        device=plan.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=plan.replicate(),
    )


def replicated(plan, t, dtype, layout):
    return ttnn.from_torch(t, device=plan.mesh_device, dtype=dtype, layout=layout, mesh_mapper=plan.replicate())


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 1, 5])
def test_decoder_layer_tp8_dp4(mesh_device, layer_idx):
    torch.manual_seed(1234 + layer_idx)
    cfg, ckpt = load_config_and_ckpt()
    weights = ref.load_layer_weights(ckpt, cfg, layer_idx)
    ref_layer = ref.ReferenceDecoderLayer(cfg, layer_idx, weights=weights)
    dims = cfg.attention_dims(layer_idx)
    logger.info(f"layer {layer_idx}: swa={dims.is_swa} moe={cfg.is_moe(layer_idx)} kv_heads={dims.num_kv_heads}")

    blocks_per_user = MAX_SEQ // BLOCK
    replicas = []
    for r, sm in enumerate(dp_submeshes(mesh_device)):
        plan = MeshPlan(sm)
        ropes = {
            "swa": MiMoRotarySetup(sm, MAX_BATCH, dims.head_dim, dims.rope_dim, MAX_SEQ, cfg.swa_rope_theta),
            "full": MiMoRotarySetup(sm, MAX_BATCH, dims.head_dim, dims.rope_dim, MAX_SEQ, cfg.rope_theta),
        }
        # Replica 0 converts the weights and fills the cache; the other rows load the cached tensors.
        layer = TtMiMoDecoderLayer(
            plan, cfg, layer_idx, weights if r == 0 else None, ropes, cache_dir=cache_dir("layer_bfp8")
        )
        kv = layer.attn.allocate_kv_cache(MAX_BATCH * blocks_per_user, BLOCK, KV_DTYPE)
        page_table = block_page_table(MAX_BATCH, blocks_per_user, seed=r)
        replicas.append(
            dict(
                plan=plan,
                ropes=ropes,
                layer=layer,
                kv=kv,
                page_table=replicated(plan, page_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            )
        )

    # Users: replica r, slot u; lengths 128/256/384 so SWA layers see the window and the sink.
    users = {}
    for r in range(DP):
        for u in range(USERS_PER_REPLICA):
            seq_len = 128 * (1 + (u + r) % 3)
            x = (torch.randn(seq_len, cfg.hidden_size) * HIDDEN_STD).bfloat16().float()
            hist = ref.KVHistory()
            with torch.no_grad():
                y = ref_layer(x, torch.arange(seq_len), hist)
            users[(r, u)] = dict(seq_len=seq_len, x=x, ref_prefill=y, hist=hist)

    # ---- prefill: enqueue every user on every replica, then read
    pending = {}
    for u in range(USERS_PER_REPLICA):
        for r, rep in enumerate(replicas):
            usr = users[(r, u)]
            x_tt = to_tt(usr["x"], rep["plan"])
            pending[(r, u)] = rep["layer"].forward_prefill(x_tt, u, rep["page_table"], rep["kv"])
    worst = {"prefill": 1.0, "prefill_delta": 1.0, "decode": 1.0, "decode_delta": 1.0}
    for (r, u), out in pending.items():
        usr = users[(r, u)]
        got = replicas[r]["plan"].to_torch_replicated(out).float()[0, 0, : usr["seq_len"]]
        p = pcc(got, usr["ref_prefill"])
        pd = pcc(got - usr["x"], usr["ref_prefill"] - usr["x"])
        worst["prefill"], worst["prefill_delta"] = min(worst["prefill"], p), min(worst["prefill_delta"], pd)
        logger.info(f"prefill replica {r} slot {u} len {usr['seq_len']}: pcc {p:.5f} block-delta pcc {pd:.5f}")

    decode_pool = {r: ([], []) for r in range(DP)}
    # ---- decode: all four users of a replica in one 32-slot batch; empty slots have position -1
    for step in range(DECODE_STEPS):
        pending = []
        for r, rep in enumerate(replicas):
            plan = rep["plan"]
            x = torch.zeros(MAX_BATCH, cfg.hidden_size)
            pos = torch.full((MAX_BATCH,), -1, dtype=torch.int32)
            for u in range(USERS_PER_REPLICA):
                usr = users[(r, u)]
                p = usr["seq_len"] + step
                xu = (torch.randn(cfg.hidden_size) * HIDDEN_STD).bfloat16().float()
                with torch.no_grad():
                    usr["ref_decode"] = ref_layer(xu[None], torch.tensor([p]), usr["hist"])[0]
                usr["x_decode"] = xu
                x[u], pos[u] = xu, p
            rot = {
                True: rep["ropes"]["swa"].get_rot_mats(pos.clamp(min=0)),
                False: rep["ropes"]["full"].get_rot_mats(pos.clamp(min=0)),
            }
            cur_pos = replicated(plan, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            mask = replicated(plan, (pos >= 0).bfloat16().reshape(1, 1, -1, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            out = rep["layer"].forward_decode(to_tt(x, plan), rot, cur_pos, rep["page_table"], rep["kv"], mask)
            pending.append(out)
        for r, out in enumerate(pending):
            got = replicas[r]["plan"].to_torch_replicated(out).float()[0, 0]
            for u in range(USERS_PER_REPLICA):
                usr = users[(r, u)]
                p = pcc(got[u], usr["ref_decode"])
                xd, rd = usr["x_decode"], usr["ref_decode"]
                pd = pcc(got[u] - xd, rd - xd)
                # The output is bf16 (|x| ~ 0.5): the best delta PCC any bf16 result can reach.
                floor = pcc(rd.bfloat16().float() - xd, rd - xd)
                rel = ((got[u] - rd).norm() / (rd - xd).norm()).item()
                worst["decode"] = min(worst["decode"], p)
                decode_pool[r][0].append(got[u] - xd)
                decode_pool[r][1].append(rd - xd)
                logger.info(
                    f"decode step {step} replica {r} slot {u} pos {usr['seq_len'] + step}: pcc {p:.5f} delta {pd:.5f} "
                    f"(bf16 floor {floor:.5f}, |delta| {(rd - xd).norm():.3f}, rel err {rel:.4f})"
                )

    for r, (got_d, ref_d) in decode_pool.items():
        pd = pcc(torch.stack(got_d), torch.stack(ref_d))
        worst["decode_delta"] = min(worst["decode_delta"], pd)
        logger.info(f"decode replica {r}: block-delta pcc over {len(got_d)} user-steps {pd:.5f}")
    logger.info(f"layer {layer_idx} worst PCC over 4 replicas x 4 users: {worst}")
    assert worst["prefill"] > 0.999 and worst["decode"] > 0.999, worst
    assert worst["prefill_delta"] > 0.98 and worst["decode_delta"] > 0.98, worst
