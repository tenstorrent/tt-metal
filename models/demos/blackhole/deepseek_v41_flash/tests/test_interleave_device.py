# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test of INTERLEAVED prefill (tt/dsv41_model.Model.prefill_interleaved): users 0..B-2 decode for N steps while user B-1 is prefilled in chunks
between the steps (random prompts, few layers by default, DSV41_LAYERS=0-3). Checks, with the SAME model object:

  (a) the decode STATE of the running users (pool pages + window rings, ratio-2 compressor state ``prev_cs``, decode index keys) is bit-identical before / after every
      interleaved prefill window, and the tokens AND logits of the running users are bit-identical to an uninterrupted run (teacher forced with the same tokens),
  (b) the first token / logits of the new user prefilled in chunks between decode steps are bit-identical to the same prompt prefilled in ONE call, and the joint decode
      steps after it joined are bit-identical to the uninterrupted run,
  (c) (DSV41_IL_LEGACY=1) the first-token logits of every user vs the whole-batch ``prefill_forward`` (the original traced path),
  (d) the time of an interleaved chunk step vs a decode step.

    DSV41_LAYERS=0-3 pytest models/demos/blackhole/deepseek_v41_flash/tests/test_interleave_device.py -s
Env: DSV41_IL_U (users per row, 4), DSV41_IL_ISL (1536), DSV41_IL_CHUNK (512), DSV41_IL_STEPS (8), DSV41_IL_NEWLEN (new user's prompt length, default ISL).
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

U = int(os.environ.get("DSV41_IL_U", "4"))
ISL = int(os.environ.get("DSV41_IL_ISL", "1536"))
CHUNK = int(os.environ.get("DSV41_IL_CHUNK", "512"))
STEPS = int(os.environ.get("DSV41_IL_STEPS", "8"))
POST = int(os.environ.get("DSV41_IL_POST", "4"))
NEWLEN = int(os.environ.get("DSV41_IL_NEWLEN", str(ISL)))
LAYERS = os.environ.get("DSV41_LAYERS", "0-3")
PCC_MIN = float(
    os.environ.get("DSV41_IL_PCC_MIN", "0.99999")
)  # (bit-identical logits have PCC 1.0; the host PCC of identical vectors prints as 1.00006)


def pcc(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


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
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_interleaved_prefill(mesh_device):
    os.environ.setdefault("DSV41_PF_UMASK", "1")
    md = mesh_device
    log = lambda m: print(m, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    new = B - 1
    max_ctx = max(ISL, NEWLEN) + STEPS + POST + 64
    S_pad = -(-max(ISL, NEWLEN) // CHUNK) * CHUNK
    pp = default_page_params(max_ctx, U)
    args, m, pool, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    log(f"model: B={B} U={U} layers {LAYERS} use_indexer={m.use_indexer} chunk {CHUNK} S_pad {S_pad} max_ctx {max_ctx}")
    g = torch.Generator().manual_seed(1234)
    prompts = torch.randint(1000, 100000, (B, max(ISL, NEWLEN)), generator=g)
    lens = {u: ISL - 37 * (u % 5) for u in range(new)}  # ragged, mostly unaligned
    lens[new] = NEWLEN
    run_users = list(range(new))

    def prefill(items):
        return m.prefill_interleaved(items, CHUNK, s_pad_max=S_pad, want_logits=True)

    def decode(tok, pos):
        out = m.decode_forward(tok, pos, enable_trace=True, reload_inputs=True)
        return out.clone(), m.read_logits().clone()

    def reset():
        for u in range(B):
            m.release_user(u)
        m.pf_resume = {}

    def new_chunks():
        n = -(-NEWLEN // CHUNK)
        return [(new, prompts[new], j * CHUNK, min((j + 1) * CHUNK, NEWLEN)) for j in range(n)]

    def feed_of(first, nxt_pos, parked=None):
        tok = torch.zeros(B, dtype=torch.long)
        pos = torch.zeros(B, dtype=torch.long)
        for u, f in first.items():
            tok[u], pos[u] = f, nxt_pos[u]
        for u, p in (parked or {}).items():
            pos[u] = p
        return tok, pos

    # ---- state snapshot / diff (bit exact) -----------------------------------------------------------------------------------------------
    def snap():
        out = {"pool": [ttnn.to_torch(ttnn.get_device_tensors(m.pool.pool)[r * m.cols]).clone() for r in range(m.rows)]}
        out["prev_cs"] = {
            L: [ttnn.to_torch(ttnn.get_device_tensors(a.prev_cs)[r * m.cols]).clone() for r in range(m.rows)]
            for L, a in m.attns.items()
            if getattr(a, "prev_cs", None) is not None
        }
        out["k"] = {
            L: [ttnn.to_torch(ttnn.get_device_tensors(d.k_cache)[r * m.cols]).clone() for r in range(m.rows)]
            for L, d in m.index_owner.items()
        }
        return out

    def new_user_rows(r):
        """pool rows of the new user (its pages and its ring rows) on mesh row r"""
        if r != new // U:
            return set()
        from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P

        rows_ = set()
        for pg in m.pool.allocs[r].pages.get(new, []):
            rows_.update(range(pg * P.PAGE_ROWS, (pg + 1) * P.PAGE_ROWS))
        for slot in range(m.pool.n_ring_layers):
            base = m.pool.ring_base(slot) + (new % U) * m.pool.ring_rows
            rows_.update(range(base, base + m.pool.ring_rows))
        return rows_

    def same(x, y):
        """bit-level equality that treats NaN == NaN (uninitialised cache entries can hold NaN patterns)"""
        x, y = x.float(), y.float()
        return bool(((x == y) | (x.isnan() & y.isnan())).all())

    def diff_state(tag, s0, s1):
        bad = []
        for r in range(m.rows):
            a_, b_ = s0["pool"][r].reshape(-1, 512), s1["pool"][r].reshape(-1, 512)
            ne = (
                (~((a_.float() == b_.float()) | (a_.float().isnan() & b_.float().isnan())))
                .any(dim=1)
                .nonzero()
                .reshape(-1)
                .tolist()
            )
            skip = new_user_rows(r)
            ne = [x for x in ne if x not in skip]
            if ne:
                bad.append(f"pool row {r}: {len(ne)} rows changed (first {ne[:6]})")
        for L, lst in s0["prev_cs"].items():
            for r in range(m.rows):
                a_, b_ = lst[r].reshape(-1, 1024), s1["prev_cs"][L][r].reshape(-1, 1024)
                du = [u for u in range(U) if not same(a_[u], b_[u]) and r * U + u != new]
                if du:
                    bad.append(f"prev_cs L{L} row {r}: users {du} changed")
        for L, lst in s0["k"].items():
            for r in range(m.rows):
                a_, b_ = lst[r], s1["k"][L][r]
                du = [u for u in range(U) if not same(a_[u], b_[u]) and r * U + u != new]
                if du:
                    bad.append(f"index keys L{L} row {r}: users {du} changed")
        log(f"STATE {tag}: " + ("decode state of the running users untouched" if not bad else "; ".join(bad)))
        return bad

    # ---- one run: prefill of the running users, STEPS decode steps (+ the new user), POST joint steps -----------------------------------------
    state_bad = []

    def run(interleave, forced=None):
        reset()
        t0 = time.perf_counter()
        res = prefill([(u, prompts[u], 0, lens[u]) for u in run_users])
        log(
            f"  [{'B' if interleave else 'A'}] prefill of {len(run_users)} users: {time.perf_counter() - t0:.2f} s, {m.timing}"
        )
        first = {u: int(res[u][0]) for u in run_users}
        pos = {u: lens[u] for u in run_users}
        fed, logs_run, step_t, chunk_t = [], [], [], []
        chunks = new_chunks()
        at = {2 * (j + 1) - 1: j for j in range(len(chunks))} if interleave else {}  # chunk j after decode step 2j+1
        done_end, f_new = 0, None
        for k in range(STEPS):
            if forced is not None:
                first = dict(forced["fed"][k])
            fed.append(dict(first))
            tok, p = feed_of(first, pos, {new: done_end} if interleave and done_end else None)
            t0 = time.perf_counter()
            out, lg = decode(tok, p)
            step_t.append(time.perf_counter() - t0)
            logs_run.append(lg[run_users].clone())
            for u in run_users:
                first[u], pos[u] = int(out[u]), pos[u] + 1
            if k in at:
                it = chunks[at[k]]
                s_before = snap()
                t0 = time.perf_counter()
                r = prefill([it])
                chunk_t.append((time.perf_counter() - t0, dict(m.timing)))
                state_bad.extend(diff_state(f"chunk {at[k]} (window [{it[2]}, {it[3]}))", s_before, snap()))
                done_end = it[3]
                if done_end == NEWLEN:
                    f_new = r[new]
        if interleave:
            assert f_new is not None, "raise DSV41_IL_STEPS: the new prompt must be complete before the join"
        else:
            t0 = time.perf_counter()
            f_new = prefill([(new, prompts[new], 0, NEWLEN)])[new]
            chunk_t.append((time.perf_counter() - t0, dict(m.timing)))
        joint, jfed = [], []
        first[new], pos[new] = int(f_new[0]), NEWLEN
        for k in range(POST):
            if forced is not None:
                first = dict(forced["jfed"][k])
            jfed.append(dict(first))
            tok, p = feed_of(first, pos)
            out, lg = decode(tok, p)
            joint.append(lg.clone())
            for u in first:
                first[u], pos[u] = int(out[u]), pos[u] + 1
        return dict(
            fed=fed, jfed=jfed, logs=logs_run, f_new=f_new, joint=joint, step_t=step_t, chunk_t=chunk_t, res=res
        )

    def agree(x, y):
        return float((x.argmax(-1) == y.argmax(-1)).float().mean())

    def cmp_runs(tag, X, Y):
        worst = 1.0
        for k in range(STEPS):
            pc = pcc(X["logs"][k], Y["logs"][k])
            per_user = [round(pcc(X["logs"][k][i], Y["logs"][k][i]), 4) for i in range(new)]
            log(
                f"{tag} step {k}: PCC {pc:.5f} argmax agreement {agree(X['logs'][k], Y['logs'][k]):.2f} bit-identical {torch.equal(X['logs'][k], Y['logs'][k])}; per user {per_user}"
            )
            worst = min(worst, pc)
        same = torch.equal(X["f_new"][1], Y["f_new"][1]) and int(X["f_new"][0]) == int(Y["f_new"][0])
        log(f"{tag} new user first token {int(X['f_new'][0])} vs {int(Y['f_new'][0])}: logits bit-identical {same}")
        for k in range(POST):
            pc = pcc(X["joint"][k], Y["joint"][k])
            log(f"{tag} joint step {k}: PCC {pc:.5f} argmax agreement {agree(X['joint'][k], Y['joint'][k]):.2f}")
            worst = min(worst, pc)
        return worst, same

    # reference runs: A0 (first run after the captures), A (uninterrupted reference); B interleaved, teacher-forced with A's tokens
    A0 = run(False)
    A = run(False, forced=A0)  # same feed as A0: run-to-run noise baseline
    log(
        f"A: decode step {1000 * sum(A['step_t'][1:]) / max(1, len(A['step_t']) - 1):.1f} ms, new-user prefill in one call {A['chunk_t'][0][0]:.2f} s ({A['chunk_t'][0][1]})"
    )
    Bres = run(True, forced=A0)
    ct = Bres["chunk_t"]
    log(
        f"B: decode step {1000 * sum(Bres['step_t'][1:]) / max(1, len(Bres['step_t']) - 1):.1f} ms; interleaved chunk steps (s): "
        + ", ".join(
            f"{t:.2f} ({d.get('replays')} replay {d.get('replay_s', 0):.2f} host {d.get('host_s', 0):.2f} export {d.get('export_s', 0):.2f} pre {d.get('pre_s', 0):.2f})"
            for t, d in ct
        )
    )
    noise, _ = cmp_runs("A0 vs A (run-to-run noise, identical feed, no interleave)", A0, A)
    worst, new_same = cmp_runs("A vs B (interleaved)", A, Bres)
    log(
        f"worst PCC interleaved vs uninterrupted {worst:.5f}, noise baseline {noise:.5f}; state diffs {len(state_bad)}; new-user first token bit-identical to the one-call prefill {new_same}"
    )

    # cross-process reference (e.g. the same prompts prefilled with DSV41_PREFILL_UP == users per row): first-token logits of every user
    first_logits = {u: A["res"][u][1].clone() for u in run_users}
    first_logits[new] = A["f_new"][1].clone()
    if os.environ.get("DSV41_IL_SAVE"):
        torch.save({"logits": first_logits, "Up": m.Up}, os.environ["DSV41_IL_SAVE"])
        log(f"saved first-token logits to {os.environ['DSV41_IL_SAVE']}")
    if os.environ.get("DSV41_IL_REF") and os.path.exists(os.environ["DSV41_IL_REF"]):
        ref = torch.load(os.environ["DSV41_IL_REF"])
        pcs = [pcc(ref["logits"][u], first_logits[u]) for u in sorted(first_logits)]
        am = sum(int(ref["logits"][u].argmax()) == int(first_logits[u].argmax()) for u in first_logits)
        log(
            f"REF (Up={ref['Up']}) vs this run (Up={m.Up}): first-token logits min PCC {min(pcs):.6f}, argmax equal {am}/{len(first_logits)}"
        )
        assert min(pcs) > 0.995, "first-token logits differ from the reference run"

    if os.environ.get("DSV41_IL_LEGACY", "0") == "1":
        reset()
        full_lens = torch.tensor([lens[u] for u in range(B)])
        m.prefill_model.active_mask = None
        _, lg = m.prefill_forward(prompts.clone(), full_lens, chunk=CHUNK, want_logits=True, s_pad_max=S_pad)
        r = prefill([(u, prompts[u], 0, lens[u]) for u in range(B)])
        pcs = [pcc(lg[u], r[u][1]) for u in range(B)]
        log(
            f"(c) first-token logits interleaved-path vs legacy whole-batch prefill: min PCC {min(pcs):.6f}, argmax equal {sum(int(lg[u].argmax()) == int(r[u][0]) for u in range(B))}/{B}"
        )
        assert min(pcs) > 0.999

    if os.environ.get("DSV41_IL_ADAPTER", "1") == "1":
        # the vLLM adapter on top of the same model: new requests, a prompt split in chunks over several prefill steps (the plugin re-picks the state slot of the
        # continuation every step), decode steps between the chunks (the prompt in progress is parked), then the joint decode
        from types import SimpleNamespace

        from models.demos.blackhole.deepseek_v41_flash.tt.generator import Generator
        from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM

        os.environ.update(
            {"DSV41_VLLM_INTERLEAVE": "1", "DSV41_VLLM_CHUNK": str(CHUNK), "DSV41_VLLM_S_PAD": str(S_pad)}
        )
        reset()
        ad = DeepseekV41ForCausalLM(Generator([m], [args], md), B, max_ctx)
        sp = SimpleNamespace(temperature=[0.0] * B, top_k=[1] * B, enable_log_probs=[False] * B)
        n_run = len(run_users)
        toks = torch.zeros(n_run, max(lens[u] for u in run_users), dtype=torch.int32)
        for u in run_users:
            toks[u, : lens[u]] = prompts[u, : lens[u]].to(torch.int32)
        t0 = time.perf_counter()
        lg = ad.prefill_forward(
            toks, prompt_lens=[lens[u] for u in run_users], empty_slots=run_users, sampling_params=None
        )
        log(f"ADAPTER prefill of {n_run} requests (host sampling): {time.perf_counter() - t0:.2f} s")
        first_ad = {u: int(lg[i].argmax()) for i, u in enumerate(run_users)}
        assert [first_ad[u] for u in run_users] == [
            int(A["res"][u][0]) for u in run_users
        ], "adapter first tokens differ from the model-level prefill"
        ad.release_request(
            14
        )  # a request finishes: its logical slot is free again (so that the new prompt can change its slot between the chunks)
        run_ad = [u for u in run_users if u != 14]
        pos_ad, cur = {u: lens[u] for u in run_ad}, {u: first_ad[u] for u in run_ad}
        chunks_ad, f_ad = new_chunks(), None
        free_slots = [15, 14, 15, 14, 15]
        t_chunk, t_dec = [], []
        for k in range(STEPS):
            tk = torch.zeros(B, 1, dtype=torch.int32)
            ps = torch.full((B,), -1, dtype=torch.long)
            for u in run_ad:
                tk[u, 0], ps[u] = cur[u], pos_ad[u]
            t0 = time.perf_counter()
            out = ad.decode_forward(tk, ps, sampling_params=sp)
            t_dec.append(time.perf_counter() - t0)
            for u in run_ad:
                cur[u], pos_ad[u] = int(out[u, 0]), pos_ad[u] + 1
            j = k // 2 if k % 2 == 1 else None
            if j is not None and j < len(chunks_ad):
                _, _, st_, e_ = chunks_ad[j]
                t0 = time.perf_counter()
                # a different free logical slot every step (15, 14, 15): the continuation is found by its token prefix
                lg = ad.prefill_forward(
                    prompts[new : new + 1, :e_].to(torch.int32),
                    prompt_lens=[e_],
                    start_pos=[st_],
                    empty_slots=[free_slots[j]],
                    sampling_params=None,
                )
                t_chunk.append(time.perf_counter() - t0)
                if e_ == NEWLEN:
                    f_ad = lg
        assert f_ad is not None
        log(
            f"ADAPTER: decode step {1000 * sum(t_dec[1:]) / max(1, len(t_dec) - 1):.1f} ms, chunk steps {[round(t, 2) for t in t_chunk]} s; new user first token {int(f_ad[0].argmax())} (model level {int(A['f_new'][0])})"
        )
        assert (
            int(f_ad[0].argmax()) == int(A["f_new"][0]) and pcc(f_ad[0, 0], A["f_new"][1]) > 0.9995
        ), "adapter chunked prefill differs from the model-level one"
        # joint decode through the adapter: the new request lives in the logical slot of its last chunk
        last_slot = free_slots[len(chunks_ad) - 1]
        tk = torch.zeros(B, 1, dtype=torch.int32)
        ps = torch.full((B,), -1, dtype=torch.long)
        for u in run_ad:
            tk[u, 0], ps[u] = cur[u], pos_ad[u]
        tk[last_slot, 0], ps[last_slot] = int(f_ad[0].argmax()), NEWLEN
        out = ad.decode_forward(tk, ps, sampling_params=sp)
        assert out.shape == (B, 1) and int(ad.inprog.parked().get(ad.slots.phys[last_slot], -1)) == -1
        log(f"ADAPTER joint decode step ok: tokens of the new request {int(out[last_slot, 0])}")
        os.environ["DSV41_VLLM_INTERLEAVE"] = "0"

    assert not state_bad, f"the decode state of running users changed during an interleaved prefill: {state_bad[:3]}"
    assert new_same, "chunked prefill of the new user differs from the one-call prefill"
    assert worst >= PCC_MIN, f"decode logits of the interleaved run deviate: worst PCC {worst:.6f} < {PCC_MIN}"
    assert all(
        torch.equal(A["logs"][k], Bres["logs"][k]) for k in range(STEPS)
    ), "decode logits of the running users are not bit-identical to the uninterrupted run"
    assert all(
        torch.equal(A["joint"][k], Bres["joint"][k]) for k in range(POST)
    ), "joint decode logits after the new user joined are not bit-identical to the uninterrupted run"
