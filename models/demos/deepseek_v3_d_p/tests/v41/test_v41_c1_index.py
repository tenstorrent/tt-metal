# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The c1 capture's index selection for the live user alone (``V41_DECODE_C1_INDEX=1``):
the pieces that let the capture carry no slot, each bit for bit against the per-user path they replace.

No weights; four users of a 20,480-row pool with FP4-like keys and queries as ``test_indexer.py``'s range
test:

- ``range_scores(dyn=True)`` (the user and its newest read from the ``c1usr`` page) against ``range_scores``
with
  the user as an argument, for a live user at every slot, an idle slot and the no-live-row page;
- ``candidate_scores(base_page=...)`` (the slot's block page read from the page) against ``block_base``;
- the one-hot row pick (``C1Moe.onehots`` of the ``c1all`` row, a HiFi4 matmul) against the query's rows;
- ``c1_place`` (the selection into the live row, sentinels elsewhere) against torch, for every row and for
none;
- ``TtV41Indexer.select_split`` on the dyn score against the per-user score: the top-k and the published
blocks.

Then the dyn scorer under trace (``V41_CI_REPLAYS`` replays) for its device time against the four per-user
launches. Results: ``V41_CI_OUT``."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_indexer import _fp4_like, _replicated
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import _C1_HIFI4, c1_place
from models.demos.deepseek_v3_d_p.tt.v41.indexer import SENTINEL, TtV41Indexer
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import candidate_scores, range_scores
from models.demos.deepseek_v3_d_p.tt.v41.moe_c1 import _SENTINEL as ONEHOT_SENTINEL
from models.demos.deepseek_v3_d_p.tt.v41.moe_c1 import C1Moe

ROWS = 32


def _all_chips_equal(t, what: str) -> torch.Tensor:
    devs = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]
    for i, other in enumerate(devs[1:], 1):
        assert torch.equal(other, devs[0]), f"{what}: chip {i} differs from chip 0"
    return devs[0][0, 0]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=64 * 1024 * 1024), indirect=True)
def test_v41_c1_index(mesh_device, device_params):
    replays = int(os.environ.get("V41_CI_REPLAYS", 20))
    cfg = C
    cap = 20480
    width = cap + 32
    users = 4
    nb = width // cfg.CANDIDATE_BLOCK_SIZE  # block pages a user
    g = torch.Generator().manual_seed(86)
    newest = torch.full((ROWS,), -1, dtype=torch.int64)
    newest[:users] = torch.tensor([19999, 299, -1, cap - 1])  # a long prefix, a short one, idle, the full pool
    q = _fp4_like((ROWS, 32, 128), g)
    w = (torch.randn(ROWS, 32, generator=g) * 0.05).to(torch.bfloat16)
    k = _fp4_like((users * width, 128), g)
    k.view(users, width, 128)[:, cap:] = 0
    dev_q = _replicated(mesh_device, q.reshape(1, 1, ROWS, -1), ttnn.bfloat16)
    dev_w = _replicated(mesh_device, w.reshape(1, 1, ROWS, 32), ttnn.bfloat16)
    dev_k = ttnn.from_torch(
        k.reshape(users, 1, width, 128),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    dev_pin = _replicated(mesh_device, newest.view(1, 1, 1, ROWS).to(torch.int32), ttnn.uint32)
    blocks = _replicated(mesh_device, k.reshape(1, 1, users * nb, cfg.CANDIDATE_BLOCK_SIZE * 128), ttnn.bfloat16)
    table = C1Moe.identity_table(mesh_device)
    result = {"cap": cap, "users": users, "cases": []}

    def usr_page(live: int | None):
        """The ``c1usr`` row of ``_host_inputs``: [user, block page of its block 0, newest, sentinel...]."""
        page = torch.full((1, 1, 1, ROWS), -1, dtype=torch.int32)
        page[..., :2] = 0
        if live is not None:
            page[..., 0], page[..., 1], page[..., 2] = live, live * nb, int(newest[live])
        return _replicated(mesh_device, page, ttnn.uint32)

    def pin_page(live: int | None):
        """The ``c1pin`` row: the live user's newest in every word."""
        page = torch.full((1, 1, 1, ROWS), -1 if live is None else int(newest[live]), dtype=torch.int32)
        return _replicated(mesh_device, page, ttnn.uint32)

    # 1. the dyn scorer against the per-user scorer, bit for bit, every chip
    for live in [0, 1, 2, 3, None]:
        usr = usr_page(live)
        dyn = _all_chips_equal(range_scores(dev_q, dev_w, dev_k, usr, 0, width, dyn=True), f"dyn live {live}")
        ref = _all_chips_equal(range_scores(dev_q, dev_w, dev_k, dev_pin, live or 0, width), f"static {live}")
        if live is None:  # no live row: every score -inf, as an idle user's
            assert bool((dyn == float("-inf")).all()), "the no-live-row page scored something"
        else:
            assert torch.equal(_bits(dyn), _bits(ref)), f"range_scores dyn differs from static for user {live}"
        finite = int(torch.isfinite(dyn).sum())
        logger.info(f"range_scores dyn: live {live}: {finite} finite scores, bit-equal to the per-user launch")
        result["cases"].append({"range_scores": live, "finite": finite})

    # 2. the candidate scorer with the base page against block_base, on random blocks of each user (sentinel
    # tail)
    source = TtV41Indexer(
        mesh_device,
        cfg,
        cfg.CANDIDATE_SOURCE_LAYER,
        {
            "wq_b": torch.zeros(cfg.INDEX_N_HEADS * cfg.INDEX_HEAD_DIM, cfg.Q_LORA_RANK),
            "weights_proj": torch.zeros(cfg.INDEX_N_HEADS, cfg.EMB_SIZE),
        },
    )
    consumer = TtV41Indexer(
        mesh_device,
        cfg,
        24,
        {
            "wq_b": torch.zeros(cfg.INDEX_N_HEADS * cfg.INDEX_HEAD_DIM, cfg.Q_LORA_RANK),
            "weights_proj": torch.zeros(cfg.INDEX_N_HEADS, cfg.EMB_SIZE),
        },
    )
    kb = cfg.CANDIDATE_TOPK_BLOCKS
    topk = cfg.INDEX_TOPK
    for live in [0, 1, 3]:
        n_u = int(newest[live])
        ids = torch.randperm(max(1, (n_u + 1) // cfg.CANDIDATE_BLOCK_SIZE), generator=g)[:kb]
        ids = (
            torch.cat([ids, torch.full((kb - ids.numel(),), SENTINEL, dtype=torch.int64)]) if ids.numel() < kb else ids
        )
        ids_u = _replicated(mesh_device, ids.view(1, 1, 1, kb).to(torch.int32), ttnn.uint32)
        # the one-hot rows (c1all: the live user in every word) pick the query and weights into every row
        all_ = _replicated(mesh_device, torch.full((1, 1, 1, ROWS), live, dtype=torch.int32), ttnn.uint32)
        sel_all, _ = C1Moe.onehots(all_, all_, table)
        q_t = ttnn.to_layout(dev_q, ttnn.TILE_LAYOUT)
        w_t = ttnn.to_layout(dev_w, ttnn.TILE_LAYOUT)
        q_n = ttnn.to_layout(ttnn.matmul(sel_all, q_t, compute_kernel_config=_C1_HIFI4), ttnn.ROW_MAJOR_LAYOUT)
        w_n = ttnn.to_layout(ttnn.matmul(sel_all, w_t, compute_kernel_config=_C1_HIFI4), ttnn.ROW_MAJOR_LAYOUT)
        q_pick = _all_chips_equal(q_n, "q one-hot")
        w_pick = _all_chips_equal(w_n, "w one-hot")
        # as values: the one-hot matmul turns a -0.0 of the FP4-like query into +0.0 (1.0 x -0.0 + 0.0), which
        # no
        # score can tell apart (relu(q . k) of either is the same bits)
        assert torch.equal(q_pick.float(), q[live].reshape(1, -1).expand(ROWS, -1).float()), "q one-hot pick"
        assert torch.equal(w_pick.float(), w[live].reshape(1, -1).expand(ROWS, -1).float()), "w one-hot pick"
        pin_n = pin_page(live)
        usr = usr_page(live)
        dyn = consumer.select_live_in_blocks(q_n, w_n, blocks, pin_n, ids_u, usr, width, topk)
        ref = consumer.select_user_in_blocks(dev_q, dev_w, blocks, dev_pin, ids_u, live, width, live * nb, topk)
        dyn_t, ref_t = _all_chips_equal(dyn, "consumer dyn"), _all_chips_equal(ref, "consumer static")
        assert torch.equal(dyn_t, ref_t), f"consumer selection differs for user {live}"
        # and the scorer itself, bit for bit
        n = ROWS
        per = kb // n
        ids_n = ttnn.reshape(ids_u, [1, 1, n, per])
        s_dyn = candidate_scores(q_n, w_n, blocks, ids_n, pin_n, 0, width, block_base=0, base_page=usr)
        s_ref = candidate_scores(q_n, w_n, blocks, ids_n, pin_n, 0, width, block_base=live * nb)
        assert torch.equal(_bits(_all_chips_equal(s_dyn, "cs dyn")), _bits(_all_chips_equal(s_ref, "cs static")))
        picked = int((dyn_t != SENTINEL).sum())
        logger.info(f"consumer: live {live}: {picked} rows selected, the dyn path bit-equal to the per-user path")
        result["cases"].append({"consumer": live, "selected": picked})

        # 3. the candidate source's split selection on the dyn score: top-k and published blocks as the
        # per-user's
        pin_u = ttnn.slice(usr, [0, 0, 0, 2], [1, 1, 1, 3])
        pin_s = ttnn.slice(dev_pin, [0, 0, 0, live], [1, 1, 1, live + 1])
        sc_dyn = range_scores(dev_q, dev_w, dev_k, usr, 0, width, dyn=True)
        sc_ref = range_scores(dev_q, dev_w, dev_k, dev_pin, live, width)
        t_dyn, p_dyn = source.select_split(sc_dyn, pin_u, width, topk)
        t_ref, p_ref = source.select_split(sc_ref, pin_s, width, topk)
        assert torch.equal(_all_chips_equal(t_dyn, "source dyn"), _all_chips_equal(t_ref, "source static"))
        assert (p_dyn.ids is None) == (p_ref.ids is None)
        if p_dyn.ids is not None:
            assert torch.equal(_all_chips_equal(p_dyn.ids, "blocks dyn"), _all_chips_equal(p_ref.ids, "blocks static"))
        logger.info(f"source: live {live}: the split selection and the published blocks bit-equal")

    # 4. the placement: a random selection with a sentinel tail into every row, and into none
    sel = torch.randint(0, 1 << 20, (topk,), generator=g)
    sel[topk - 7 :] = SENTINEL
    sel_dev = _replicated(mesh_device, sel.view(1, 1, 1, topk).to(torch.int32), ttnn.uint32)
    for u in list(range(ROWS)) + [None]:
        put_row = torch.full((1, 1, 1, ROWS), ONEHOT_SENTINEL, dtype=torch.int32)
        if u is not None:
            put_row[..., u] = 0
        placed = _all_chips_equal(c1_place(sel_dev, _replicated(mesh_device, put_row, ttnn.uint32)), f"place {u}")
        expect = torch.full((ROWS, topk), SENTINEL, dtype=torch.int64)
        if u is not None:
            expect[u] = sel
        got = placed.to(torch.int64) & 0xFFFFFFFF
        assert torch.equal(got, expect), f"placement into row {u}: {(got != expect).sum()} words differ"
    logger.info("c1_place: every row and none, bit-exact (ids below 2^20, the sentinel tail kept)")

    # 5. device time under trace: the dyn scorer once against the four per-user launches
    def timed(fn, n):
        fn()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(4):
            fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        for _ in range(n):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        per = (time.time() - t0) / (n * 4)
        ttnn.release_trace(mesh_device, tid)
        return per

    usr0 = usr_page(0)
    t_dyn = timed(lambda: range_scores(dev_q, dev_w, dev_k, usr0, 0, width, dyn=True), replays)
    t_static = timed(lambda: [range_scores(dev_q, dev_w, dev_k, dev_pin, u, width) for u in range(users)], replays)
    result["trace_us"] = {"dyn_one_user": t_dyn * 1e6, "static_four_users": t_static * 1e6}
    logger.info(
        f"under trace: dyn scorer {t_dyn * 1e6:.1f} us a call, the four per-user launches {t_static * 1e6:.1f} us"
    )
    out = os.environ.get("V41_CI_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
