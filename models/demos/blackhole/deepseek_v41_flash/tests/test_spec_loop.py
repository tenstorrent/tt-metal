# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3: the full speculative loop on the device (draft -> verify -> accept/reject -> next draft, ONE traced step per round), 16 users = 8 GSM8K prompts x 2.
The prompt is FED THROUGH THE DEVICE STEP ITSELF (blocks of n prompt tokens, accept count forced), starting from empty caches: KV / compressor state, the drafter's
main_kv rings and the Engram hash state are all produced by the device (the CPU reference dump dsv4-chain-g is NOT used for state: its final argmax disagrees with
the CPU greedy stream, so seeding from it is inconsistent). Greedy acceptance: the output must equal plain greedy decoding (DSV41_K=0 run).

Env: DSV41_K (verified drafts k = 1..5; 0 = plain greedy baseline on the same code, writes greedy_dev_g.pt), DSV41_LAYERS (default 0-39), DSV41_GEN (tokens per
user, default 40; positions must stay < 128: ratio-1 layers hold compress_len <= 128), DSV41_CHAIN, DSV41_REPLAYS (device-only timing).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram, HostEngramRows
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.mtp import BLOCK, DSparkDrafter, load_mtp_stage
from models.demos.blackhole.deepseek_v41_flash.tt.spec_chain import SpecChain
from models.demos.blackhole.deepseek_v41_flash.tt.spec_decoder import SpecDecoder, SpecVerifier
from models.demos.blackhole.deepseek_v41_flash.tt.spec_state import SpecStepState

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-g")
MTP_REF = os.environ.get("DSV41_MTP_REF", "/mnt/tt-data/ssinghal/dsv4-spec-accept/mtp_ref.pt")
GREEDY = "/mnt/tt-data/ssinghal/dsv4-spec-accept/greedy_dev_g.pt"
MAXPOS = 127


def hash_block(h, tokens, base):
    """``NgramHashState.forward`` with a PER-USER start position: tokens [B,n], base [B] -> hash ids [B,n,layers,cols]."""
    B, n = tokens.shape
    comp = h.token_map[tokens]
    for b in range(B):
        h.cache[b, int(base[b]) : int(base[b]) + n] = comp[b]
    positions = base.reshape(B, 1) + torch.arange(n).reshape(1, n)
    toks, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
    for shift in range(h.layout.max_ngram_size):
        source = h.cache[:B].gather(1, (positions - shift).clamp_min(0))
        blocked = blocked | (positions < shift) | (source == h.DEAD)
        toks.append(torch.where(blocked, h.pad_id, source))
    toks = torch.stack(toks, dim=-1)
    products = toks.unsqueeze(2) * h.multipliers
    rolling, out = products[..., 0], []
    for i in range(1, h.layout.max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        out.append(rolling.unsqueeze(-1) % h.primes[:, i - 1])
    return torch.cat(out, dim=-1) + h.offsets


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 1_200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(21600)
@torch.no_grad()
def test_spec_loop(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    a_, _, b_ = os.environ.get("DSV41_LAYERS", "0-39").partition("-")
    layer_ids = list(range(int(a_), int(b_ or a_) + 1))
    k = int(os.environ.get("DSV41_K", "1"))
    n = 1 + k
    G = int(os.environ.get("DSV41_GEN", "40"))
    U = 4
    log = lambda m: print(m, flush=True)
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S = toks["prefill_tokens"].shape[1]
    B0 = toks["prefill_tokens"].shape[0]
    B = rows * U
    reps = B // B0
    prompt = toks["prefill_tokens"].repeat(reps, 1)
    res = torch.load("/mnt/tt-data/ssinghal/dsv4-spec-accept/results_snapshot.pt")
    ref_all = res["stream"][:B0].repeat(
        reps, 1
    )  # CPU greedy stream (prompt + 64 tokens): informational reference + tokens padding the last prompt block
    chain = SpecChain(md, users_per_row=U, n=n, log=log)
    Tn = B * n
    sh = _Shards()
    pool = ThreadPoolExecutor(max_workers=2)
    futs = {}
    submit = lambda L: futs.setdefault(L, pool.submit(load_layer, L)) if L in layer_ids else None
    for L in layer_ids[:2]:
        submit(L)
    built, groups, t0 = [], {}, time.time()
    for L in layer_ids:
        ref = torch.load(
            os.path.join(CHAIN, f"layer_{L}.pt")
        )  # only for gate_cutoff / state shapes: the state itself is EMPTY (S = 0)
        ref["S"] = 0
        st0 = {}
        for kk, v in ref["state"].items():
            z = torch.zeros_like(v).repeat(reps, *([1] * (v.dim() - 1)))
            st0[kk] = (
                torch.full_like(z, float(os.environ.get("DSV41_SCORE_INIT", "-1e9"))) if kk == "score_state" else z
            )
        ref["state"] = st0
        submit(L + 1), submit(L + 2)
        layer, attn = chain.build_layer(L, ref, futs.pop(L).result())
        key = getattr(attn, "ratio", 0)
        if key not in groups:
            groups[key] = SpecStepState(attn)
        built.append((L, layer, key))
    log(f"built {len(layer_ids)} layers in {time.time() - t0:.0f}s")
    engram_ids = [l for l in (1, 14) if l in layer_ids]
    host_rows = HostEngramRows(tuple(engram_ids), max_batch_size=B) if engram_ids else None
    dev_engram = {l: DSV41DeviceEngram(md, l, sh, mesh_config=chain.mesh_config, ccl=chain.ccl) for l in engram_ids}
    embedding = DSV41DeviceEmbedding(md, sh.get("embed.weight"), users_per_row=chain.T)
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    if k > 0:
        t0 = time.time()
        stage_w = [load_mtp_stage(i, sh) for i in range(3)]
        drafter = DSparkDrafter(
            md, chain.mesh_config, chain.ccl, stage_w, embedding.weight, head, users_per_row=U, n=n
        )  # rings start empty (zeros)
        log(f"drafter built {time.time() - t0:.0f}s")
        dec = SpecDecoder(md, built, embedding, head, drafter, dev_engram, step_states=groups, n=n)
    else:
        dec = SpecVerifier(md, built, embedding, head, dev_engram, step_states=groups)
    dec.enable_sampling(chain.mesh_config, chain.ccl)
    if host_rows is not None:
        t0 = time.time()
        host_rows.load_ram()
        log(f"Engram tables in RAM {time.time() - t0:.0f}s")
    hasher = host_rows.engram.hash if host_rows is not None else None
    if host_rows is not None:
        host_rows.hashes(prompt, 0)

    def feed(Xb, base_):
        rows_d = {}
        if host_rows is not None:
            hs = hash_block(hasher, Xb, base_)
            rows_d = {l: r.reshape(Tn, 1, -1) for l, r in host_rows.rows_all(hs, engram_ids).items()}
        pos = (base_.reshape(B, 1) + torch.arange(n).reshape(1, n)).reshape(-1)
        dec.set_packed_inputs(Xb.reshape(-1), rows_d, pos)

    def readback():
        if k > 0:
            v = torch.cat(
                [ttnn.to_torch(ttnn.get_device_tensors(dec.pack)[r * cols]).reshape(1, -1) for r in range(rows)]
            ).long()  # [rows, T_loc + U + 5U]
            T_loc = U * n
            a = v[:, :T_loc].reshape(B, n)
            m = v[:, T_loc : T_loc + U].reshape(B)
            d = v[:, T_loc + U :].reshape(rows, BLOCK, U).permute(0, 2, 1).reshape(B, BLOCK)  # [B, 5] d_1..d_5
            return a, m, d
        a = head.combine(dec.sampled)[:Tn].reshape(B, n)
        return a, torch.zeros(B, dtype=torch.long), None

    def run_round(Xb, base_):
        feed(Xb, base_)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return readback()

    # ---- compile pass + trace (inputs of the first prompt block; the compile pass writes the same positions the real first round rewrites) ----
    snaps = dec.snapshot_states()
    X = ref_all[:, :n].clone()
    base = torch.zeros(B, dtype=torch.long)
    feed(X, base)
    if k > 0:
        dec.set_force(n - 1)
    dec.forward()
    ttnn.synchronize_device(md)
    dec.restore_states(snaps)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    dec.forward()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    dec.restore_states(snaps)
    log("trace captured")

    # ---- prompt phase: feed the S prompt tokens through the device in blocks of n (m forced so the commit selects the right compressor state) ----
    nb = -(-S // n)
    t0 = time.time()
    for r in range(nb):
        base = torch.full((B,), r * n, dtype=torch.long)
        Xp = ref_all[
            :, r * n : r * n + n
        ].clone()  # tokens beyond S (last block only) come from the CPU stream; their cache entries are rewritten later
        if k > 0:
            dec.set_force((S - 1 - r * n) if r == nb - 1 else n - 1)
        a, m, d = run_round(Xp, base)
        if os.environ.get("DSV41_DEBUG") == "1":
            lg = head.gather_logits(dec.logits)[:Tn].float()
            log(
                f"DEBUG block {r}: logits finite {float(torch.isfinite(lg).float().mean()):.3f} absmax {float(torch.nan_to_num(lg).abs().max()):.1f} argmax tokens user0 {a[0].tolist()}"
            )
    log(f"prompt phase: {nb} blocks of {n} through the device in {time.time() - t0:.1f}s")
    last = (S - 1) - (nb - 1) * n  # block index of position S-1 in the last prompt block
    first = a[:, last].clone()  # device greedy t_S
    X = torch.zeros(B, n, dtype=torch.long)
    X[:, 0] = first
    if k > 0:
        X[:, 1:] = d[:, :k]
        dec.set_force(-1)
    log(
        f"device first token vs CPU reference stream t_S: match {(first == ref_all[:, S]).float().mean():.2f}; first-token distinct values {sorted(set(first.tolist()))[:6]}"
    )
    base = torch.full((B,), S, dtype=torch.long)
    if k > 0:
        snaps2 = dec.snapshot_states()
        # ---- diagnostics: teacher-forced blocks (CPU reference stream tokens, every block accepted): device argmax accuracy vs the reference and the drafter's
        # match with the reference stream, fed with the DEVICE hidden states ----
        dec.set_force(n - 1)
        nb_t = min(int(os.environ.get("DSV41_TEACHER_ROUNDS", "8")), (MAXPOS - S - n + 1) // n)
        acc_bb, acc_dr, cnt_dr = [], torch.zeros(BLOCK), torch.zeros(BLOCK)
        base_t = torch.full((B,), S, dtype=torch.long)
        for r in range(nb_t):
            Xt = torch.stack([ref_all[:, S + r * n + j] for j in range(n)], 1)
            a_t, _, d_t = run_round(Xt, base_t)
            want = torch.stack([ref_all[:, S + r * n + j + 1] for j in range(n)], 1)
            acc_bb.append((a_t == want).float().mean(0))
            ok = a_t[:, n - 1] == ref_all[:, S + r * n + n]
            for i in range(BLOCK):
                acc_dr[i] += float(((d_t[:, i] == ref_all[:, S + r * n + n + 1 + i]) & ok).sum())
                cnt_dr[i] += float(ok.sum())
            base_t += n
        bb = torch.stack(acc_bb).mean(0)
        log(
            f"TEACHER_SUMMARY backbone argmax match vs CPU ref per block index {[round(float(x), 3) for x in bb]}; drafter d1..d5 match vs ref stream "
            f"{[round(float(acc_dr[i] / max(cnt_dr[i], 1)), 3) for i in range(BLOCK)]} (n={[int(c) for c in cnt_dr]})"
        )
        dec.set_force(-1)
        dec.restore_states(snaps2)

    gen = [[int(first[b])] for b in range(B)]
    done = torch.zeros(B, dtype=torch.bool)
    rounds, walls, emitted, ms = 0, [], [], []
    per_pos = torch.zeros(BLOCK + 1)
    gap_l = [[] for _ in range(B)]  # gap_l[b][i] = top1-top2 logit gap of the argmax that produced generated token i+1
    while not bool(done.all()):
        t = time.perf_counter()
        a, m, d = run_round(X, base)
        walls.append((time.perf_counter() - t) * 1e3)
        if (
            os.environ.get("DSV41_GAP", "1") == "1"
        ):  # top-1/top-2 logit gap of every verified row (outside the timed region): near-tie evidence for exactness analysis
            lgv = head.gather_logits(dec.logits)[:Tn].reshape(B, n, -1).float()
            t2 = lgv.topk(2, dim=-1).values
            gaps_r = t2[..., 0] - t2[..., 1]  # [B, n]
        for b in range(B):
            if done[b]:
                continue
            mb = int(m[b])
            if os.environ.get("DSV41_GAP", "1") == "1":
                gap_l[b] += [float(x) for x in gaps_r[b, : mb + 1]]
            gen[b] += [int(x) for x in a[b, : mb + 1]]
            emitted.append(mb + 1)
            ms.append(mb)
            base[b] += mb + 1
            X[b, 0] = a[b, mb]
            if k > 0:
                X[b, 1:] = d[b, :k]
            if len(gen[b]) >= G + 1 or int(base[b]) + n - 1 > MAXPOS:
                done[b] = True
        rounds += 1
    wall = sum(walls) / len(walls)
    mean_tok = sum(emitted) / len(emitted)
    log(
        f"SPEC_LOOP k={k} n={n}: {rounds} rounds, mean round wall {wall:.1f} ms (steady: {sum(walls[1:]) / max(len(walls) - 1, 1):.1f}), mean emitted tokens/round/user {mean_tok:.3f} "
        f"-> {1e3 * mean_tok / wall:.1f} tok/s/user; accepted drafts/round {sum(ms) / len(ms):.3f}"
    )
    if k > 0:
        ms_t = torch.tensor(ms)
        log("P(m >= j): " + str([round(float((ms_t >= j).float().mean()), 3) for j in range(1, k + 1)]))
    lens = [min(len(g), G + 1) for g in gen]
    stream = torch.full((B, G + 1), -1, dtype=torch.long)
    for b in range(B):
        stream[b, : lens[b]] = torch.tensor(gen[b][: lens[b]])
    torch.save(
        {"stream": stream, "k": k, "S": S, "gap": gap_l}, GREEDY if k == 0 else GREEDY.replace(".pt", f"_k{k}.pt")
    )
    if k > 0 and os.path.exists(GREEDY):
        base_stream = torch.load(GREEDY)["stream"]
        ident, first_div = 0, []
        for b in range(B):
            L_ = min(lens[b], int((base_stream[b] >= 0).sum()))
            eq = stream[b, :L_] == base_stream[b, :L_]
            first_div.append(int((~eq).nonzero()[0]) if bool((~eq).any()) else -1)
            ident += int(bool(eq.all()))
        log(
            f"EXACTNESS vs device plain greedy: {ident}/{B} users identical over the compared length; first divergence index per user {first_div}"
        )
        base_gap = torch.load(GREEDY).get("gap")
        for b in range(B):
            i = first_div[b]
            if i > 0 and base_gap is not None and len(base_gap[b]) >= i and len(gap_l[b]) >= i:
                # stream[b, i] was produced by the argmax of the row that emitted generated token i (gap list index i-1)
                log(
                    f"DIVERGENCE user {b} token {i}: plain tok {int(base_stream[b, i])} (top1-top2 gap in plain {base_gap[b][i - 1]:.3f}), spec tok {int(stream[b, i])} (gap in spec {gap_l[b][i - 1]:.3f}), "
                    f"CPU ref tok {int(ref_all[b, S + i])}"
                )
    ref_stream = ref_all[:, S : S + G + 1]
    agree = [(stream[b, : lens[b]] == ref_stream[b, : lens[b]]).float().mean().item() for b in range(B)]
    log(f"agreement with the CPU reference greedy stream (informational): mean {sum(agree) / B:.3f}")
    log(
        "sample device stream user 0: "
        + str(stream[0, :20].tolist())
        + " | CPU ref: "
        + str(ref_stream[0, :20].tolist())
    )
    nrep = int(os.environ.get("DSV41_REPLAYS", "10"))
    t = time.perf_counter()
    for _ in range(nrep):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    log(f"DEVICE_ROUND k={k}: pure device replay {(time.perf_counter() - t) / nrep * 1e3:.1f} ms")
    if k > 0:  # time breakdown of one round: verify (40 layers + head + argmax) / + accept / full round
        res_t = {}
        for mode in ("verify", "accept", None):
            dec.stop_after = mode
            tid2 = ttnn.begin_trace_capture(md, cq_id=0)
            dec.forward()
            ttnn.end_trace_capture(md, tid2, cq_id=0)
            ttnn.synchronize_device(md)
            t = time.perf_counter()
            for _ in range(nrep):
                ttnn.execute_trace(md, tid2, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            res_t[mode] = (time.perf_counter() - t) / nrep * 1e3
            ttnn.release_trace(md, tid2)
        log(
            f"ROUND_BREAKDOWN k={k}: verify {res_t['verify']:.1f} ms, + accept {res_t['accept'] - res_t['verify']:.1f} ms, + commit/write_main/draft {res_t[None] - res_t['accept']:.1f} ms "
            f"= device round {res_t[None]:.1f} ms; host (wall - device) {wall - res_t[None]:.1f} ms"
        )
