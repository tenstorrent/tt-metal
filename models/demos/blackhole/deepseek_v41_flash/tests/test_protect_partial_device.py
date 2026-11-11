# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""T24 device test of DSV41_PROTECT_PARTIAL / DSV41_RESUME_FROM_RECORD through the vLLM adapter (real model, paged APC, the B=16 geometry U=4 / Up=2 by default; DSV41_PP_U / DSV41_PP_UP).

Case A (mechanism of the T9 B=8 recomputes): a prefix hit whose first piece is shorter than a chunk (the step budget ran out). A donor shares the first 2048 tokens with X; X is planned as a hit (2048), its
first piece is [2048, 2333) (285 tokens), the second [2333, 3300). Old behaviour: the first piece is skipped, nothing is registered, the second piece finds no carried state and recomputes from 0.
Protected: the hit is restored and one chunk is computed; the second piece resumes one chunk ahead of vLLM's start. Checked: no recompute, last-token logits within the cold-vs-cold noise.

Case B (resume from a record): a cold prompt Y is prefilled in two pieces [0, 2048) / [2048, 3300); between them its carried state is dropped (slot taken by another prompt). Old: the second piece recomputes
from 0. DSV41_RESUME_FROM_RECORD=1: it resumes from the deepest record of its own prefix (2048) through the paged index. Checked: logits within the noise, tokens computed.

    DSV41_APC_PAGED=1 pytest models/demos/blackhole/deepseek_v41_flash/tests/test_protect_partial_device.py -s
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

U = int(os.environ.get("DSV41_PP_U", "4"))
UP = int(os.environ.get("DSV41_PP_UP", "2"))
CHUNK = 512
LAYERS = os.environ.get("DSV41_PH_LAYERS", "0-23")
MAXCTX = int(os.environ.get("DSV41_PH_MAXCTX", "8192"))
LEN = int(os.environ.get("DSV41_PP_LEN", "3300"))
HIT = int(os.environ.get("DSV41_PP_HIT", "2048"))
SHORT = int(os.environ.get("DSV41_PP_SHORT", "285"))


def set_env():
    os.environ.update(
        {
            "DSV41_APC_PAGED": "1",
            "DSV41_PF_PAGED": "read",
            "DSV41_PF_SINGLE": "1",
            "DSV41_PF_UMASK": "1",
            "DSV41_PREFILL_UP": str(UP),
            "DSV41_PREFILL_UP_BATCH": "1",
            "DSV41_POOL_DTYPE": "fp8",
            "DSV41_FP8_SERVING": "1",
            "DSV41_PF_PAGED_FP8": "1",
            "DSV41_MO_OVERLAP": "0",
            "DSV41_VLLM_INTERLEAVE": "1",
            "DSV41_VLLM_CHUNK": str(CHUNK),
            "DSV41_PIECE_LAG": "1",
            "DSV41_APC_PAGED_HALO_BUDGET": os.environ.get(
                "DSV41_PP_BUDGET", "3"
            ),  # (6 blocks per prompt -> a halo record every 2nd block: block 5 (2560) has none)
            "DSV41_PREFILL_ROW_AWARE": os.environ.get("DSV41_PP_ROW_AWARE", "1"),
        }
    )
    if UP < U:
        os.environ["DSV41_APC_PAGED_UP"] = "1"


def cmp(a, b):
    d = (a.float() - b.float()).abs()
    return dict(
        max=float(d.max()),
        argmax_equal=int(a.argmax()) == int(b.argmax()),
        bits=bool(torch.equal(a.float(), b.float())),
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_IL_TRACE_REGION", "1600000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(9000)
@torch.no_grad()
def test_protect_partial_through_the_adapter(mesh_device, tmp_path):
    set_env()
    sw_p, sw_r = tmp_path / "protect", tmp_path / "resume"
    sw_p.write_text("0")
    sw_r.write_text("0")
    os.environ["DSV41_PROTECT_PARTIAL_FILE"] = str(sw_p)
    os.environ["DSV41_RESUME_FROM_RECORD_FILE"] = str(sw_r)
    sw_s = tmp_path / "snap"
    sw_s.write_text("0")
    os.environ["DSV41_SNAPSHOT_ON_EVICT_FILE"] = str(sw_s)
    from models.demos.blackhole.deepseek_v41_flash.tt import apc_paged_model as APM
    from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
    from models.demos.blackhole.deepseek_v41_flash.tt.generator import Generator
    from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM
    from models.tt_transformers.tt.common import PagedAttentionConfig

    md = mesh_device
    log = lambda m: print(m, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    os.environ["DSV41_VLLM_S_PAD"] = str(-(-LEN // CHUNK) * CHUNK)
    pp = default_page_params(MAXCTX, U)
    pp["page_max_num_blocks_per_dp"] += APM.extra_pool_pages(U)
    t0 = time.perf_counter()
    args, m, _, _ = create_tt_model(
        md,
        B,
        MAXCTX,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    log(f"model built in {time.perf_counter() - t0:.0f} s (layers {LAYERS}), U={U} Up={m.Up}")
    ad = DeepseekV41ForCausalLM(Generator([m], [args], md), B, MAXCTX)
    assert ad.paged is not None, "paged APC refused (see the log)"
    pa = ad.paged
    events = []
    sink = logger.add(
        lambda msg: events.append(msg.record["message"])
        if any(
            k in msg.record["message"] for k in ("without carried state", "resume from record", "shorter than a chunk")
        )
        else None
    )
    orig_log = m.log

    def log_spy(msg):
        if "recomputing from" in str(msg):
            events.append(str(msg))
        return orig_log(msg)

    m.log = log_spy

    g = torch.Generator().manual_seed(2468)
    rnd = lambda n: torch.randint(1000, 100000, (n,), generator=g, dtype=torch.int64)

    def call(P, st, e, rid, slot, final, total):
        t1 = time.perf_counter()
        out = ad.prefill_forward(
            P.reshape(1, -1).to(torch.int32),
            prompt_lens=[e],
            start_pos=[st],
            empty_slots=[slot],
            sampling_params=None,
            prefill_final=[final],
            prefill_total=[total],
            request_ids=[rid],
        )
        return out, time.perf_counter() - t1

    def clean():
        for s in range(B):
            ad.release_request(s)
        for x in range(B):
            m.release_user(x)
        pa.reset()
        ad.inprog.users.clear()
        ad.inprog.reported.clear()
        ad.pf_done.clear()
        events.clear()

    def set_flags(protect, resume, snap=False):
        sw_p.write_text("1" if protect else "0")
        sw_r.write_text("1" if resume else "0")
        sw_s.write_text("1" if snap else "0")

    results = {}
    X, Y = rnd(LEN), rnd(LEN)
    D = torch.cat([X[:HIT], rnd(LEN - HIT)])

    # ---- cold references (two users of different rows: the cold-vs-cold noise) ----------------------------------------------------------------
    set_flags(False, False)
    clean()
    refs = {}
    for name, P in (("X", X), ("Y", Y)):
        for k, slot in enumerate((0, 4)):
            clean()
            out, dt = call(P, 0, LEN, f"cold{k}-{name}", slot, True, LEN)
            refs[(name, k)] = out[0, 0].float().clone()
        results[f"cold_vs_cold_{name}"] = cmp(refs[(name, 0)], refs[(name, 1)])
    log(json.dumps(results))

    def noise(name):
        return max(results[f"cold_vs_cold_{name}"]["max"], 1e-3)

    # ---- case A: hit piece shorter than a chunk ---------------------------------------------------------------------------------------------------
    for protect in (False, True):
        set_flags(protect, False)
        clean()
        out, _ = call(D, 0, LEN, "donor", 2, True, LEN)
        m.flush_key_exports(why="test")
        ad.release_request(2)  # (the donor request ends; its blocks stay cached)
        plan = pa.idx.plan_hit("hitX", X.numpy(), LEN)
        log(f"[A protect={protect}] plan: hit {plan.hit_tokens} row {plan.row} user {plan.user}")
        assert plan.hit_tokens == HIT, plan
        slot = plan.user  # (logical slot = model user: the slot map is the identity here)
        events.clear()
        call(X, HIT, HIT + SHORT, "hitX", slot, False, LEN)
        out, dt = call(X, HIT + SHORT, LEN, "hitX", slot, True, LEN)
        lg = out[0, 0].float()
        r = dict(
            case="A", protect=protect, events=list(events), vs_cold=cmp(lg, refs[("X", 0)]), second_piece_s=round(dt, 2)
        )
        r["ok"] = bool(r["vs_cold"]["max"] <= 3 * noise("X") and r["vs_cold"]["argmax_equal"])
        results[f"A_{protect}"] = r
        log(json.dumps(r))
        pa.idx.check()

    # ---- case B: lost carried state, resume from a record ------------------------------------------------------------------------------------------
    for resume in (False, True):
        set_flags(resume, resume)
        clean()
        slot = 1
        call(Y, 0, HIT, "Y", slot, False, LEN)
        m.flush_key_exports(why="test")  # (what the slot eviction does before the carried state is overwritten)
        user = ad.slots.phys[slot]
        assert m.pf_resume.get(user) == HIT, m.pf_resume
        # (the slot is taken by another prompt of the row: the carried state is gone, the committed blocks / records stay)
        m.pf_resume.pop(user, None)
        sl = m.__dict__.get("pf_slot_of", {}).pop(user, None)
        if sl is not None and m.__dict__.get("pf_slot_owner") is not None:
            m.__dict__["pf_slot_owner"][sl] = None
        events.clear()
        out, dt = call(Y, HIT, LEN, "Y", slot, True, LEN)
        lg = out[0, 0].float()
        r = dict(
            case="B", resume=resume, events=list(events), vs_cold=cmp(lg, refs[("Y", 0)]), second_piece_s=round(dt, 2)
        )
        r["ok"] = bool(r["vs_cold"]["max"] <= 3 * noise("Y") and r["vs_cold"]["argmax_equal"])
        results[f"B_{resume}"] = r
        log(json.dumps(r))
        pa.idx.check()
    # ---- case C: slot eviction of a prompt in progress at a block without a record, with / without the snapshot -----------------------------------------------------------------
    PIECE = 5 * CHUNK  # 2560: block 5 carries no record (every 2nd block)
    Q = rnd(LEN)
    S_pad = m.single_spad(CHUNK)
    for snap in (False, True):
        set_flags(False, True, snap)
        clean()
        call(Y, 0, PIECE, "Y", 0, False, LEN)  # (user 0 / slot 0 of row 0)
        m.flush_key_exports(why="test")
        u_y = ad.slots.phys[0]
        m.prefill_interleaved(  # a second prompt in progress in the same row: both prefill slots are taken
            [(1 if u_y != 1 else 2, Q, 0, PIECE)],
            CHUNK,
            s_pad_max=S_pad,
            totals={(1 if u_y != 1 else 2): LEN},
            final={(1 if u_y != 1 else 2): False},
            rids={(1 if u_y != 1 else 2): "Q"},
        )
        assert m.pf_resume.get(u_y) == PIECE, m.pf_resume
        # a third prompt of the row takes a slot: the owner of the lowest slot (Y) is evicted (the snapshot runs first when enabled)
        m._slot_alloc([(3, None, 0, CHUNK)], m.pf_resume)
        assert u_y not in m.pf_resume, "the eviction did not take Y's slot"
        events.clear()
        out, dt = call(Y, PIECE, LEN, "Y", 0, True, LEN)
        lg = out[0, 0].float()
        r = dict(
            case="C",
            snapshot=snap,
            events=list(events),
            vs_cold=cmp(lg, refs[("Y", 0)]),
            second_piece_s=round(dt, 2),
            snapshots=pa.stats.get("snapshots", 0),
        )
        r["ok"] = bool(r["vs_cold"]["max"] <= 3 * noise("Y") and r["vs_cold"]["argmax_equal"])
        results[f"C_{snap}"] = r
        log(json.dumps(r))
        pa.idx.check()
    set_flags(False, False)
    clean()
    pa.idx.check()
    out_f = os.environ.get("DSV41_PP_OUT")
    if out_f:
        json.dump(results, open(out_f, "w"), indent=1)
    logger.remove(sink)
    # old behaviour is reproduced with the flags off, the protected / resumed variants have no recompute and match the cold logits
    assert any(
        "without carried state" in e for e in results["A_False"]["events"]
    ), "control A: the old behaviour did not recompute"
    assert not any(
        "without carried state" in e or "recomputing from" in e for e in results["A_True"]["events"]
    ), results["A_True"]
    assert results["A_True"]["ok"], results["A_True"]
    assert any(
        "without carried state" in e for e in results["B_False"]["events"]
    ), "control B: the old behaviour did not recompute"
    assert any("resume from record" in e for e in results["B_True"]["events"]), results["B_True"]
    assert not any("without carried state" in e for e in results["B_True"]["events"]), results["B_True"]
    assert results["B_True"]["ok"], results["B_True"]

    def resumed_at(res):
        import re

        for e in res["events"]:
            x = re.search(r"resumes at (\d+)", e)
            if x:
                return int(x.group(1))
        return None

    assert resumed_at(results["C_False"]) == 4 * CHUNK, results["C_False"]  # no record at 2560: the deepest earlier one
    assert resumed_at(results["C_True"]) == PIECE, results["C_True"]  # the snapshot made one
    assert results["C_True"]["snapshots"] >= 1 and results["C_True"]["ok"] and results["C_False"]["ok"], results
    log(f"ALL OK; stats {ad.protect_stats}")
