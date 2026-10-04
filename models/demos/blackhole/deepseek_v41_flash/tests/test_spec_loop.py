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
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import DSV41PagedStepState
from models.demos.blackhole.deepseek_v41_flash.tt.spec_chain import SpecChain
from models.demos.blackhole.deepseek_v41_flash.tt.spec_decoder import SpecDecoder, SpecVerifier
from models.demos.blackhole.deepseek_v41_flash.tt.spec_paged import SpecPagedChain
from models.demos.blackhole.deepseek_v41_flash.tt.spec_state import SpecStepState

PROMPT_DIR = os.environ.get(
    "DSV41_PROMPT_DIR"
)  # e.g. /mnt/tt-data/ssinghal/dsv4-prefill-s2048b1f: ONE real long prompt (tiled over the users) + the CPU reference first token
CHAIN = PROMPT_DIR or os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-g")
MTP_REF = os.environ.get("DSV41_MTP_REF", "/mnt/tt-data/ssinghal/dsv4-spec-accept/mtp_ref.pt")
TAG = os.environ.get("DSV41_TAG", "")  # distinguishes partial-layer runs
GREEDY_NP = f"/mnt/tt-data/ssinghal/dsv4-spec-accept/greedy_dev_g{TAG}.pt"  # non-paged plain greedy (reference of the paged runs)
PAGED = (
    os.environ.get("DSV41_SPEC_PAGED", "0") == "1"
)  # KV in the paged pool (tt/spec_paged.py), positions up to DSV41_CTX
CTX = int(os.environ.get("DSV41_CTX", "384"))
GREEDY = GREEDY_NP.replace("greedy_dev_g", "greedy_paged_g") if PAGED else GREEDY_NP
MAXPOS = CTX - 1 if PAGED else 127


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
    if (
        PROMPT_DIR
    ):  # long-prompt mode: reference = prompt + the CPU reference first token (+ its decode tokens); padded so block slicing works
        fin = torch.load(os.path.join(PROMPT_DIR, "final.pt"))
        tail = torch.cat([fin["prefill_argmax"].reshape(B0, 1), toks["decode_tokens"].reshape(B0, -1)], dim=1)
        ref_all = torch.cat([toks["prefill_tokens"], tail, torch.zeros(B0, 16, dtype=torch.long)], dim=1).repeat(
            reps, 1
        )
    else:
        res = torch.load("/mnt/tt-data/ssinghal/dsv4-spec-accept/results_snapshot.pt")
        ref_all = res["stream"][:B0].repeat(
            reps, 1
        )  # CPU greedy stream (prompt + 64 tokens): informational reference + tokens padding the last prompt block
    chain = (
        SpecPagedChain(md, users_per_row=U, n=n, ctx=CTX, log=log)
        if PAGED
        else SpecChain(md, users_per_row=U, n=n, log=log)
    )
    Tn = B * n
    sh = _Shards()
    pool = ThreadPoolExecutor(max_workers=2)
    futs = {}
    submit = lambda L: (
        futs.setdefault(
            L, pool.submit(load_layer, L, max_seq_len=CTX + 64 if PAGED else 256, with_indexer=PAGED and CTX > 512)
        )
        if L in layer_ids
        else None
    )
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
            groups[key] = (
                DSV41PagedStepState(attn, max_pos=CTX + 64, with_indexer=CTX > 512, per_user_valid=True)
                if PAGED
                else SpecStepState(attn)
            )
        built.append((L, layer, key))
    log(f"built {len(layer_ids)} layers in {time.time() - t0:.0f}s")
    engram_ids = [l for l in (1, 14) if l in layer_ids]
    host_rows = (
        HostEngramRows(tuple(engram_ids), max_batch_size=B, max_seq_len=CTX + 64 if PAGED else 256)
        if engram_ids
        else None
    )
    dev_engram = {l: DSV41DeviceEngram(md, l, sh, mesh_config=chain.mesh_config, ccl=chain.ccl) for l in engram_ids}
    embedding = DSV41DeviceEmbedding(md, sh.get("embed.weight"), users_per_row=chain.T)
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    if k > 0:
        t0 = time.time()
        stage_w = [load_mtp_stage(i, sh) for i in range(3)]
        drafter = DSparkDrafter(
            md,
            chain.mesh_config,
            chain.ccl,
            stage_w,
            embedding.weight,
            head,
            users_per_row=U,
            n=n,
            max_pos=CTX + 64 if PAGED else 256,
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
    snaps2 = (
        dec.snapshot_states()
    )  # allocated BEFORE the trace capture: device buffers allocated while a trace exists can overlap the trace's intermediates (corruption / CCL hang)
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
    last_start = (nb - 1) * n  # first position of the last fed block
    if os.environ.get("DSV41_TAIL_CHECK") == "1":
        # ---- hand-off emulation: after a real PREFILL the pool / compressor state is complete but (a) the drafter rings are empty and (b) ``prev_cs`` is only valid for the
        # last prompt token. The drafter is seeded by replaying the LAST 128 prompt tokens through the verify step from an EVEN start position (even rows write no latent, so
        # ``prev_cs`` at the start is never read). Check: corrupt prev_cs, replay [p0, S), the first token and the 5 drafts must equal the full-replay ones. ----
        first0, d0 = a[:, (S - 1) - (nb - 1) * n].clone(), d.clone()
        for _, layer, _ in dec.layers:
            pc = getattr(layer.attention, "prev_cs", None)
            if pc is not None:
                ttnn.copy(ttnn.zeros_like(pc), pc)
        p0 = max(0, ((S - 128) // 2) * 2)
        nb2 = -(-(S - p0) // n)
        for r in range(nb2):
            b0 = p0 + r * n
            base = torch.full((B,), b0, dtype=torch.long)
            dec.set_force((S - 1 - b0) if r == nb2 - 1 else n - 1)
            a, m, d = run_round(ref_all[:, b0 : b0 + n].clone(), base)
        first1 = a[:, (S - 1) - (p0 + (nb2 - 1) * n)]
        log(
            f"TAIL_REPLAY start {p0} ({nb2} blocks): first token equal {bool((first1 == first0).all())}; drafts d1..d5 equal in {float((d[:, :BLOCK] == d0[:, :BLOCK]).float().mean()):.3f} of entries"
        )
        last_start = p0 + (nb2 - 1) * n
    last = (S - 1) - last_start  # block index of position S-1 in the last prompt block
    first = a[:, last].clone()  # device greedy t_S
    X = torch.zeros(B, n, dtype=torch.long)
    X[:, 0] = first
    if k > 0:
        X[:, 1 : 1 + min(k, BLOCK)] = d[:, : min(k, BLOCK)]  # rows beyond the 5 drafts (padded block) stay 0
        dec.set_force(-1)
    log(
        f"device first token vs CPU reference stream t_S: match {(first == ref_all[:, S]).float().mean():.2f}; first-token distinct values {sorted(set(first.tolist()))[:6]}"
    )
    base = torch.full((B,), S, dtype=torch.long)
    if k > 0 and os.environ.get("DSV41_TEACHER", "1") == "1" and not PROMPT_DIR:
        for a_, s_ in snaps2:  # refresh in place (no allocation after the trace capture)
            if s_ is not None:
                ttnn.copy(a_.prev_cs, s_)
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

    if k > 0 and os.environ.get("DSV41_TEACHER_FULL") == "1":
        # ---- teacher-forced replay of the PLAIN stream (same mode, k = 0 run): at every position the verify row sees exactly the plain history, so any argmax
        # difference is numerics of the 1+k-row path (or a bug), not trajectory drift. Reports mismatches with the top1-top2 gaps of both paths. ----
        pl = torch.load(GREEDY)
        ps, pgap = (
            pl["stream"],
            pl["gap"],
        )  # ps[b, i] = generated token i (position S + i); pgap[b][i - 1] = gap of the argmax that produced token i
        dec.set_force(n - 1)
        base_t = torch.full((B,), S, dtype=torch.long)
        mism, total = [], 0
        for r in range((ps.shape[1] - 1) // n):
            if int(base_t[0]) + n - 1 > MAXPOS:
                break
            Xt = ps[:, r * n : r * n + n].clone()
            if bool((Xt < 0).any()):
                break
            a_t, _, _ = run_round(Xt, base_t)
            v2 = torch.cat(
                [ttnn.to_torch(ttnn.get_device_tensors(dec.top2)[q * cols]).reshape(U * n, -1) for q in range(rows)]
            ).float()
            t2 = v2.topk(2, dim=-1).values.reshape(B, n, 2)
            for j in range(n):
                t = r * n + j + 1  # generated index predicted by row j
                if t >= ps.shape[1] or int((ps[:, t] < 0).sum()) == B:
                    continue
                for b in range(B0):  # the B0 distinct prompts (user b + B0 repeats b)
                    if int(ps[b, t]) < 0 or len(pgap[b]) < t:
                        continue
                    total += 1
                    if int(a_t[b, j]) != int(ps[b, t]):
                        mism.append((b, int(base_t[b]) + j, t, float(pgap[b][t - 1]), float(t2[b, j, 0] - t2[b, j, 1])))
            base_t += n
        log(
            f"TEACHER_FULL k={k}: {len(mism)} argmax mismatches of {total} rows vs the plain stream (distinct prompts 0..{B0 - 1})"
        )
        for b, pos_, t, gp, gs in mism:
            log(
                f"  TF_MISMATCH user {b} position {pos_} (gen token {t}, block row {(pos_ - S) % n}): plain gap {gp:.3f}, spec gap {gs:.3f}"
            )
        dec.set_force(-1)
        return

    gen = [[int(first[b])] for b in range(B)]
    done = torch.zeros(B, dtype=torch.bool)
    rounds, walls, emitted, ms = 0, [], [], []
    per_pos = torch.zeros(BLOCK + 1)
    gap_l = [[] for _ in range(B)]  # gap_l[b][i] = top1-top2 logit gap of the argmax that produced generated token i+1
    while not bool(done.all()):
        t = time.perf_counter()
        log(f"ROUND {rounds} start base {base[:4].tolist()}...")
        a, m, d = run_round(X, base)
        walls.append((time.perf_counter() - t) * 1e3)
        if (
            os.environ.get("DSV41_GAP", "1") == "1"
        ):  # top-1/top-2 logit gap of every verified row (outside the timed region): near-tie evidence for exactness analysis
            v2 = torch.cat(
                [ttnn.to_torch(ttnn.get_device_tensors(dec.top2)[r * cols]).reshape(U * n, -1) for r in range(rows)]
            ).float()  # [Tn, 2*cols]
            t2 = v2.topk(2, dim=-1).values.reshape(B, n, 2)
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
                X[b, 1 : 1 + min(k, BLOCK)] = d[b, : min(k, BLOCK)]
            if len(gen[b]) >= G + 1 or int(base[b]) + n - 1 > MAXPOS:
                done[b] = True
                base[b] = min(
                    int(base[b]), MAXPOS - n + 1
                )  # finished users keep running in the batch: keep their positions inside the cache (> 127 hangs the MoE dispatch)
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
        base_stream = torch.load(GREEDY)["stream"]  # plain greedy of the SAME mode (paged / non-paged)
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
                    f"CPU ref tok {int(ref_all[b, S + i]) if S + i < ref_all.shape[1] else -1}"
                )
    ref_stream = ref_all[:, S : S + G + 1]
    lr = ref_stream.shape[1]
    agree = [
        (stream[b, : min(lens[b], lr)] == ref_stream[b, : min(lens[b], lr)]).float().mean().item() for b in range(B)
    ]
    if PAGED:  # paged vs NON-paged stream of the same k (k = 0: plain) over the common length
        npf = GREEDY_NP if k == 0 else GREEDY_NP.replace(".pt", f"_k{k}.pt")
        if os.path.exists(npf):
            npst = torch.load(npf)["stream"]
            dv = []
            for b in range(B):
                L_ = min(lens[b], int((npst[b] >= 0).sum()))
                ne = (stream[b, :L_] != npst[b, :L_]).nonzero()
                dv.append(int(ne[0]) if len(ne) else -1)
            log(
                f"PAGED vs NON-PAGED (same k={k}) stream: first divergence per user {dv} (-1 = identical over the common length, {L_} tokens)"
            )
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
    if (
        os.environ.get("DSV41_PROFILE", "0") == "1"
    ):  # eager round with a device sync around every layer section (aggregated over the 40 verify layers + 3 draft stages)
        dec.stop_after = "verify"
        dec.forward()  # warm (programs compiled)
        ttnn.synchronize_device(md)
        dec.profile = {}
        dec.forward()
        ttnn.synchronize_device(md)
        prof = {kk: v * 1e3 for kk, v in dec.profile.items() if not kk.startswith("_")}
        log(
            f"PROFILE k={k} n={n} T={U * n} (verify layers only, ms summed over {len(dec.layers)} layers, eager+sync): "
            + ", ".join(f"{kk} {v:.1f}" for kk, v in sorted(prof.items(), key=lambda kv: -kv[1]))
            + f" | total {sum(prof.values()):.1f}"
        )
        dec.profile = None
