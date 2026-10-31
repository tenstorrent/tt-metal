# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test of the delta host snapshots (DSV41_APC_DELTA) and of the decode-output snapshot (DSV41_APC_DECODE): the production B=8 geometry (4 rows x 2 users, single chunk trace, fp8 pool,
bounded replay), host tier (DSV41_APC_PAGED=0), layers 0-23 by default.

A (delta store): prompt P is run cold with snapshots (every snapshot is compared bit for bit with the whole-prefix snapshot taken at the same moment, DSV41_APC_DELTA_CHECK=1), then the longer
prompt P2 = P + random tokens runs with the cache (hit at the deepest snapshot) and with the cache off. hit-vs-cold is compared with cold-vs-cold (two users, two mesh rows) as the noise baseline.
B (decode output): P is prefilled, then DSV41_AD_STEPS (700) greedy decode steps on user 0 (the generator hook is replayed by hand: ``decode_poll`` after every step, ``decode_finish`` at the end);
the next turn Q = P + answer + random tokens must hit at the last chunk multiple of the decoded sequence; its last-token logits are compared with a cache-off run of Q (hit-vs-cold vs cold-vs-cold)
and the first decode tokens of both are compared.

    pytest models/demos/blackhole/deepseek_v41_flash/tests/test_apc_delta_device.py -s
"""

import json
import os
import time

import pytest
import torch

import ttnn

U = 2
CHUNK = 512
LAYERS = os.environ.get("DSV41_AD_LAYERS", "0-23")
MAXCTX = int(os.environ.get("DSV41_AD_MAXCTX", "16384"))
STEPS = int(os.environ.get("DSV41_AD_STEPS", "700"))
E0 = int(os.environ.get("DSV41_AD_E", "3000"))
EXT = int(os.environ.get("DSV41_AD_EXT", "1700"))


def set_env():
    os.environ.update(
        {
            "DSV41_APC_PAGED": "0",
            "DSV41_APC_DELTA": "1",
            "DSV41_APC_DELTA_CHECK": os.environ.get("DSV41_AD_CHECK", "1"),
            "DSV41_APC_DECODE": "1",
            "DSV41_APC_GRID": "512",
            "DSV41_PF_PAGED": "read",
            "DSV41_PF_SINGLE": "1",
            "DSV41_PF_UMASK": "1",
            "DSV41_PREFILL_UP": str(U),
            "DSV41_PREFILL_UP_BATCH": "1",
            "DSV41_POOL_DTYPE": "fp8",
            "DSV41_FP8_SERVING": "1",
            "DSV41_PF_PAGED_FP8": "1",
            "DSV41_MO_OVERLAP": "0",
        }
    )


def cmp(a, b):
    d = (a.float() - b.float()).abs()
    return dict(
        max=float(d.max()),
        mean=float(d.mean()),
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
def test_delta_and_decode_snapshots(mesh_device):
    set_env()
    from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
    from models.tt_transformers.tt.common import PagedAttentionConfig

    md = mesh_device
    log = lambda m: print(m, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    pp = default_page_params(MAXCTX, U)
    t0 = time.perf_counter()
    args, m, pool, _ = create_tt_model(
        md,
        B,
        MAXCTX,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    log(f"model built in {time.perf_counter() - t0:.0f} s (layers {LAYERS})")
    apc = m.enable_apc(int(40 * 2**30), "pow2")
    assert apc is not None and apc.delta and apc.decode, "delta / decode switches did not reach the prefix cache"
    S_pad = m.single_spad(CHUNK)
    g = torch.Generator().manual_seed(91)
    rnd = lambda n: torch.randint(1000, 100000, (n,), generator=g, dtype=torch.int64)

    def run(u, P, st=0):
        e = P.numel()
        res = m.prefill_interleaved(
            [(u, P, st, e)],
            CHUNK,
            s_pad_max=S_pad,
            want_logits=True,
            totals={u: e},
            final={u: True},
            rids={u: f"r{u}-{e}"},
        )
        apc.finish_pending()
        m.flush_key_exports(why="test")
        return int(res[u][0]), res[u][1].float().clone()

    def step(u, tok_, pos_):
        tok = torch.zeros(B, dtype=torch.long)
        pos = torch.zeros(B, dtype=torch.long)
        tok[u], pos[u] = tok_, pos_
        return int(m.decode_forward(tok, pos, enable_trace=True, reload_inputs=True).clone()[u])

    def clean():
        for x in range(B):
            m.release_user(x)
            apc.decode_drop(x)

    def cold(u, P):
        apc.enabled = False
        try:
            return run(u, P)
        finally:
            apc.enabled = True

    results = []

    # ---- A: delta snapshots, hit == cold --------------------------------------------------------------------------------------------
    P = rnd(E0)
    P2 = torch.cat([P, rnd(EXT)])
    clean()
    apc.clear()
    run(0, P)  # (compile / capture + snapshots with the bit-identity check on every one)
    clean()
    apc.clear()
    run(0, P)
    st = apc.store
    log(
        f"A: store after P: {len(st.snaps)} records, {len(st.blocks)} blocks, {st.bytes / 2**20:.1f} MiB; stats {apc.stats}"
    )
    assert apc.stats.get("delta_check_failures", 0) == 0 and apc.stats.get("delta_checks", 0) > 0
    clean()
    r0 = apc.stats["restored_tokens"]
    t1_, l_hit = run(4, P2)
    hit = apc.stats["restored_tokens"] - r0
    clean()
    t2_, l_c1 = cold(0, P2)
    clean()
    t3_, l_c2 = cold(6, P2)
    rA = dict(
        case="A",
        e=P2.numel(),
        hit=hit,
        hit_vs_cold=cmp(l_hit, l_c1),
        cold_vs_cold=cmp(l_c1, l_c2),
        toks=(t1_, t2_, t3_),
    )
    rA["ok"] = bool(
        hit > 0
        and rA["hit_vs_cold"]["max"] <= 3 * max(rA["cold_vs_cold"]["max"], 1e-3)
        and rA["hit_vs_cold"]["argmax_equal"]
    )
    log(json.dumps(rA))
    results.append(rA)
    assert apc.stats.get("delta_check_failures", 0) == 0

    # ---- B: decode output --------------------------------------------------------------------------------------------------------------
    clean()
    apc.clear()
    Pd = rnd(E0)
    tok_first, _ = run(0, Pd)
    m.flush_key_exports(why="test")
    gen = [tok_first]
    n_fed = Pd.numel()
    cur = tok_first
    t_dec = time.perf_counter()
    for k in range(STEPS):
        cur = step(0, cur, n_fed)  # feeds the token at position n_fed (KV written), returns the next one
        n_fed += 1
        gen.append(cur)
        apc.decode_poll(0, n_fed)
    log(f"B: {STEPS} decode steps in {time.perf_counter() - t_dec:.1f} s; captures {apc.stats.get('dec_captures', 0)}")
    seq = torch.cat([Pd, torch.tensor(gen[:STEPS], dtype=torch.long)])  # tokens whose KV the model holds
    assert seq.numel() == n_fed
    apc.decode_finish(0, seq)
    s_exp = seq.numel() // CHUNK * CHUNK
    log(
        f"B: decode snapshots {apc.stats.get('dec_snapshots', 0)}; store {len(apc.store.snaps)} records {len(apc.store.blocks)} blocks; expect a record at {s_exp}"
    )
    assert apc.stats.get("dec_snapshots", 0) == 1, "no decode snapshot was stored"
    Q = torch.cat([seq, rnd(300)])
    clean()
    r0 = apc.stats["restored_tokens"]
    q1_, lq_hit = run(4, Q)
    hit = apc.stats["restored_tokens"] - r0
    d_hit = [q1_]
    for k in range(8):
        d_hit.append(step(4, d_hit[-1], Q.numel() + k))
    clean()
    q2_, lq_c1 = cold(0, Q)
    d_c = [q2_]
    for k in range(8):
        d_c.append(step(0, d_c[-1], Q.numel() + k))
    clean()
    q3_, lq_c2 = cold(6, Q)
    rB = dict(
        case="B",
        e=Q.numel(),
        hit=hit,
        expect_hit=s_exp,
        hit_vs_cold=cmp(lq_hit, lq_c1),
        cold_vs_cold=cmp(lq_c1, lq_c2),
        decode_hit=d_hit,
        decode_cold=d_c,
        decode_equal=d_hit == d_c,
    )
    rB["ok"] = bool(hit == s_exp and rB["hit_vs_cold"]["argmax_equal"])
    log(json.dumps(rB))
    results.append(rB)
    out = os.environ.get("DSV41_AD_OUT")
    if out:
        json.dump(results, open(out, "w"), indent=1)
    clean()
    assert all(r["ok"] for r in results), results
    log(f"ALL OK; stats {apc.stats}")
