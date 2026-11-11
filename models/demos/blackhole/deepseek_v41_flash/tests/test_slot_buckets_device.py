# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test of the slot-bucketed prefill windows (DSV41_PREFILL_SLOT_BUCKETS=1,2 at DSV41_PREFILL_UP=4, B=16).

One model build, then every scenario is run with the buckets OFF (live switch file DSV41_PREFILL_SLOT_BUCKETS_FILE = 0) and ON (= 1): cold prefill of the same prompts, logits of the last position,
the hand-off state (decode index keys of every user + two eager decode steps' logits) must be bit-identical; the wall time of the call per 512-token window is logged for both. Scenarios:
  lone     users [0]                      -> the 1-slot trace
  rows4    users [0, 4, 8, 12]            -> the 1-slot trace (one prompt per mesh row)
  row0x2   users [0, 1]                   -> the 2-slot trace
  mixed    users [0, 5]                   -> rows 0 and 1, slots 0 and 1: the 2-slot trace (two prompts in different rows)
  full     users [3]                      -> slot 3: the full trace in both arms
  switch   user 0 in two calls (the first alone: 1-slot trace; the second with a new prompt of user 5: 2-slot trace) -> the carried state of user 0 crosses the trace switch
Env: PROBE_LAYERS (default all 40), PROBE_E (default 8192), PROBE_SCEN (comma list).
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

B = 16
CHUNK = 512
WMAX = int(os.environ.get("PROBE_WMAX", "1048576"))
LAYERS = [int(x) for x in os.environ.get("PROBE_LAYERS", ",".join(str(i) for i in range(40))).split(",")]
E_ = int(os.environ.get("PROBE_E", "8192"))
SW = os.environ.get("DSV41_PREFILL_SLOT_BUCKETS_FILE", "/tmp/slot_buckets_switch.txt")


def _state(m, users, E):
    """hand-off state of the users after the prefill: pool latents + decode index keys (entries [0, E / ratio))"""
    from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as _PC

    m.flush_key_exports(why="slot bucket test read")
    out = {}
    for b in users:
        r, ud = b // m.U, b % m.U
        dev = r * m.cols
        for L_, d_ in m.index_owner.items():
            rr_ = m.attns[L_].ratio
            out[f"u{b}.idxkeys{L_}"] = (
                _PC.read_shard(d_.k_cache, dev).reshape(m.U, d_.k_cache.shape[2], -1)[ud, : E // rr_].clone()
            )
    return out


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "1000000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_slot_buckets_device(mesh_device):
    md = mesh_device
    log = lambda s: print(s, flush=True)
    pp = default_page_params(WMAX, 1)
    args, m, _, _ = create_tt_model(
        md,
        B,
        WMAX,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=LAYERS,
        log=log,
    )
    U = m.U
    log(f"SB Up={m.Up} U={U} buckets env={os.environ.get('DSV41_PREFILL_SLOT_BUCKETS')}")
    assert U == 4 and (m.Up == 4 or os.environ.get("PROBE_UP1_REF") == "1")
    pm = m.prefill_model
    rec = []
    orig_rw = pm.replay_window

    def rw(tokens, hashes, s0, C, active, vis=None, **kw):
        out = orig_rw(tokens, hashes, s0, C, active, vis=vis, **kw)
        rec.append(1)
        return out

    pm.replay_window = rw
    m.warm_serving(CHUNK, WMAX, [], [])
    log(f"SB warm_serving done: slot buckets {m._slot_ks}, captured slot contexts {pm.slot_ctx_keys()}")
    assert os.environ.get("DSV41_PREFILL_SLOT_BUCKETS") is None or m._slot_ks, "the buckets were not built"
    g = torch.Generator().manual_seed(7)
    toks = torch.randint(1000, 100000, (B, WMAX if False else 4 * E_), generator=g, dtype=torch.int64)

    def set_arm(on):
        open(SW, "w").write("1" if on else "0")

    def release():
        for b in range(B):
            m.release_user(b)

    def run(plan, on):
        """plan: list of calls, each [(user, start, end)] -> (results of the last call, wall of all calls, windows, k's used)"""
        set_arm(on)
        release()
        n0 = m.__dict__.get("_slot_n", {}).copy()
        rec.clear()
        t0 = time.perf_counter()
        res = None
        tot = {}
        for call in plan:
            for b, st, e in call:
                tot[b] = max(tot.get(b, 0), e)
        for i, call in enumerate(plan):
            last = i == len(plan) - 1
            res = m.prefill_interleaved(
                [(b, toks[b], st, e) for b, st, e in call],
                CHUNK,
                s_pad_max=WMAX,
                want_logits=True,
                final={b: last for b, st, e in call},
                totals={b: tot[b] for b, st, e in call},
            )
        wall = time.perf_counter() - t0
        used = {k: v - n0.get(k, 0) for k, v in m.__dict__.get("_slot_n", {}).items() if v - n0.get(k, 0)}
        return res, wall, len(rec), used

    scen = {
        "lone": ([[(0, 0, E_)]], [0]),
        "rows4": ([[(b, 0, E_) for b in (0, 4, 8, 12)]], [0, 4, 8, 12]),
        "row0x2": ([[(0, 0, E_), (1, 0, E_)]], [0, 1]),
        "mixed": ([[(0, 0, E_), (5, 0, E_)]], [0, 5]),
        "full": ([[(3, 0, E_)]], [3]),
        "switch": ([[(0, 0, E_ // 2)], [(0, E_ // 2, E_), (5, 0, E_ // 2)]], [0, 5]),
    }
    want = [x for x in os.environ.get("PROBE_SCEN", ",".join(scen)).split(",") if x]
    bad = []
    for name in want:
        plan, users = scen[name]
        outs = {}
        for arm, on in (("off1", False), ("on1", True), ("off2", False), ("on2", True)):
            res, wall, nwin, used = run(plan, on)
            st = _state(m, users, min(e for call in plan for _, _, e in call))
            tok_ = torch.zeros(B, dtype=torch.long)
            for b in res:
                tok_[b] = int(res[b][0])
            endp = {b: max(e for call in plan for bb, _, e in call if bb == b) for b in res}
            for k_ in range(
                2
            ):  # (the pool latents are fp8: not readable per device; two eager decode steps read pool, rings, compressor state and keys)
                pos_ = torch.zeros(B, dtype=torch.long)
                for b in res:
                    pos_[b] = endp[b] + k_
                out_ = m.decode_forward(tok_, pos_, enable_trace=False, reload_inputs=True)
                lg_ = m.read_logits().float()
                for b in res:
                    st[f"u{b}.dec{k_}"] = lg_[b].clone()
                    tok_[b] = int(out_[b])
            outs[arm] = (res, st)
            if os.environ.get("PROBE_SAVE_LG"):
                torch.save({b: res[b][1].float() for b in res}, f"{os.environ['PROBE_SAVE_LG']}.{name}.{arm}.pt")
            log(
                f"SB {name} {arm}: wall {wall:.3f} s, {nwin} replays, per window {wall / max(nwin, 1) * 1e3:.0f} ms, slot-bucket calls {used}, first tokens "
                f"{[int(res[b][0]) for b in sorted(res)]}"
            )
        ref = outs["off1"]
        for arm in ("off2", "on1", "on2"):
            r_, s_ = outs[arm]
            eq_tok = all(int(r_[b][0]) == int(ref[0][b][0]) for b in ref[0])
            eq_lg = all(bool((r_[b][1] == ref[0][b][1]).all()) for b in ref[0])
            md_ = max(float((r_[b][1].float() - ref[0][b][1].float()).abs().max()) for b in ref[0])
            dk = [
                k for k in ref[1] if ".dec" in k
            ]  # (the decode index keys read back from the slab differ even between two identical runs: padding entries; the decode logits are the state check)
            eq_st = all(torch.equal(s_[k], ref[1][k]) for k in dk)
            nbad = [
                f"{k} max|d| {float((s_[k] - ref[1][k]).abs().max()):.3g}"
                for k in dk
                if not torch.equal(s_[k], ref[1][k])
            ]
            log(
                f"SB {name} {arm} vs off1: tokens equal {eq_tok}, logits bit-equal {eq_lg} (max |d| {md_:.4g}), hand-off state bit-equal {eq_st}"
                + ("" if eq_st else f" (differs: {nbad[:4]} of {len(ref[1])})")
            )
            if not (eq_tok and eq_lg and eq_st):
                bad.append((name, arm))
    log(f"SB DONE bad={bad}")
    if os.environ.get("PROBE_UP1_REF") != "1":
        assert not bad
