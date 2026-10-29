# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2: go/no-go measurements for DFlash2 speculative decode on the resident engine (one user).

coexist: the resident decode (real Qwen3.8 weights, lm_head, host-written token states) and the ttnn DFlash2
         drafter (DFlash2DrafterTP, qwen36_resident/dflash.py) on the same mesh. The prompt + greedy decode
         runs (A) alone, (B) with eager drafter extend + draft calls queued after every resident step, (C) the
         same with the resident step, the extend and the draft all traced; B and C must reproduce A's tokens
         and logits bit for bit (L1 / fabric / trace coexistence). Then the traced resident step, extend and
         draft are timed in short bursts, and a mock DFlash2 iteration (resident step + argmax readback +
         extend + draft) end to end.
proxy:   verify-cost proxy: real weights, traced resident steps at batch 1, the 2-row verify (MTP), and 4 / 8
         users with the GDN heads of one value head on one core (RESIDENT_HEAD_GROUPS=1: every user's
         recurrence one after another, the cost of the sequential-row verify, GDN option A) or the default
         grouping, at several context lengths; device time per step from back-to-back replays, plus the
         streamers' per-layer timeline.

usage: python -m qwen36_resident.tests.dfl_e2 coexist --out coexist.json
       python -m qwen36_resident.tests.dfl_e2 proxy --out proxy.json [--configs B1,V2,B4G1,B8G1,B8G2]
"""
import argparse
import faulthandler
import gc
import json
import os
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import HIDDEN, Dims, ResidentModel, State, is_attn
from qwen36_resident.tests.long_ctx import Clock, close_mesh, log, open_mesh

REF = "/home/ttuser/experiments/qwen36_27b/profiles/resident/hf_ref_qwen38.pt"


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def ms(t0):
    return 1e3 * (time.perf_counter() - t0)


def target(mesh):
    ck = Q.Checkpoint(Q.checkpoint_dir())
    cfg = ck.config
    layers, interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
    d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=cfg["vocab_size"])
    n_attn = sum(is_attn(l, interval) for l in range(layers))
    return ck, d, layers, interval, n_attn


# ---------------------------------------------------------------------------------------------- coexist
def run_coexist(mesh, a):
    from qwen36_resident.dflash import build_drafter, tap_buffers

    clock = Clock(mesh, log)
    ck, d, layers, interval, n_attn = target(mesh)
    emb = Q.embedding(ck)
    vocab = ck.config["vocab_size"]
    st = State(d, layers - n_attn, n_attn, 0, max_pos=a.max_pos, zero=True, kv=False)
    w = Q.load(ck, d, layers, interval)
    model = ResidentModel(mesh, d, w, st, layers, interval, lm_head=True)
    del w
    clock("resident model")
    prompt = torch.load(REF)["seqs"][0]["ids"][: a.prompt].tolist()
    P, N = len(prompt), a.new

    def run(interleave=None):
        """the prompt token by token, then N greedy tokens; (generated tokens, logits of every generated step)"""
        model.reset()
        toks, rows = list(prompt), []
        for p in range(P + N - 1):
            model.step(model.token_state(emb[toks[p]].float(), p))
            if interleave is not None:
                interleave(p)
            if p >= P - 1:
                toks.append(model.argmax())
                rows.append(model.logits()[:, : d.vocab_chip].reshape(-1)[:vocab].clone())
        return toks[P:], torch.stack(rows)

    out = dict(prompt=P, new=N)
    ref_toks, ref_logits = run()
    clock("A: resident alone (eager)")
    log(f"A tokens {ref_toks}")

    drafter = build_drafter(mesh, ck, embed=emb)
    clock("drafter built")
    R = drafter.block
    g = torch.Generator().manual_seed(7)
    # stand-in taps (the resident model does not write taps yet): residual-like values, finite
    tap_vals = torch.randn(len(drafter.taps), R, HIDDEN, generator=g) * 2
    taps = tap_buffers(mesh, len(drafter.taps), R, tap_vals)
    nb, bs = a.draft_blocks, 64
    drafter.alloc(torch.arange(nb, dtype=torch.int32).reshape(1, nb), bs, taps)
    ctx0 = 128
    fill = tap_buffers(mesh, len(drafter.taps), ctx0, torch.randn(len(drafter.taps), ctx0, HIDDEN, generator=g) * 2)
    drafter.fill_context(fill, 0)
    for t in fill:
        ttnn.deallocate(t)
    # every drafter program once, eagerly, before any trace exists
    drafter.extend_context(ctx0, 1, traced=False)
    drafter.draft(prompt[-1], ctx0 + 1, traced=False)
    ttnn.synchronize_device(mesh)
    clock("drafter warm (eager)")
    ctx = [ctx0 + 1]

    def interleave_factory(traced):
        def f(p):
            # a mock DFlash2 iteration's drafter work: extend by 3 committed rows, then draft the next block
            drafter.extend_context(ctx[0], 3, traced=traced)
            ctx[0] += 3
            drafter.draft(int(prompt[p % P]), ctx[0], traced=traced)
            if ctx[0] > nb * bs - 2 * R - 256:
                ctx[0] = ctx0 + 1

        return f

    toks_b, logits_b = run(interleave_factory(False))
    clock("B: resident + eager drafter")
    out["B_tokens_equal"] = toks_b == ref_toks
    out["B_logits_equal"] = bool(torch.equal(logits_b, ref_logits))
    out["B_logits_maxabs"] = float((logits_b - ref_logits).abs().max())
    log(
        f"B: tokens equal {out['B_tokens_equal']}, logits bit-equal {out['B_logits_equal']} (max |d| {out['B_logits_maxabs']})"
    )

    model.capture_trace()
    drafter.arm_traces()
    clock("resident trace captured")
    toks_c, logits_c = run(interleave_factory(True))  # the first extend / draft capture their traces
    clock("C: traced resident + traced drafter")
    out["C_tokens_equal"] = toks_c == ref_toks
    out["C_logits_equal"] = bool(torch.equal(logits_c, ref_logits))
    out["C_logits_maxabs"] = float((logits_c - ref_logits).abs().max())
    out["drafter_stats"] = dict(drafter.stats)
    log(
        f"C: tokens equal {out['C_tokens_equal']}, logits bit-equal {out['C_logits_equal']} (max |d| {out['C_logits_maxabs']})"
    )
    # once more, so C ran with every trace already captured (the capture happened inside the first iteration)
    toks_c2, logits_c2 = run(interleave_factory(True))
    out["C2_tokens_equal"] = toks_c2 == ref_toks
    out["C2_logits_equal"] = bool(torch.equal(logits_c2, ref_logits))
    log(f"C2: tokens equal {out['C2_tokens_equal']}, logits bit-equal {out['C2_logits_equal']}")

    # ---- timing, short bursts with idle gaps (chip 1 throttles under sustained load)
    tm = {}
    tok_host = model.token_state(emb[prompt[-1]].float(), P - 1)

    def burst(name, fn, iters=a.iters):
        time.sleep(2)
        ttnn.synchronize_device(mesh)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append(ms(t0))
        tm[name] = dict(median_ms=med(ts), min_ms=min(ts), mean_ms=sum(ts) / len(ts))
        log(f"  {name}: median {tm[name]['median_ms']:.2f} ms, min {tm[name]['min_ms']:.2f} ms")

    def res_device_only():
        model.step(tok_host)
        ttnn.synchronize_device(mesh)

    def res_argmax():
        model.step(tok_host)
        model.argmax()

    def res_b2b():
        for _ in range(8):
            model.step(tok_host)
        ttnn.synchronize_device(mesh)

    def ext():
        drafter.extend_context(ctx0 + 1, 3, traced=True)
        ttnn.synchronize_device(mesh)

    def drf():
        drafter.draft(int(prompt[-1]), ctx0 + 4, traced=True)

    lay = drafter.layout()

    def drf_device():
        ttnn.execute_trace(mesh, lay.draft_tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)

    def drf_stage():
        drafter._stage_draft(lay, [int(prompt[-1])], [ctx0 + 4])
        ttnn.synchronize_device(mesh)

    def ext_stage():
        drafter._stage_extend(lay, ctx0 + 1, 3)
        ttnn.synchronize_device(mesh)

    def iteration():
        model.step(tok_host)
        model.argmax()
        drafter.extend_context(ctx0 + 1, 3, traced=True)
        drafter.draft(int(prompt[-1]), ctx0 + 4, traced=True)

    burst("resident step (write + replay + sync)", res_device_only)
    burst("resident step + argmax readback", res_argmax)
    burst("resident 8 replays back to back (per 8)", res_b2b, iters=8)
    burst("drafter extend traced (stage + replay + sync)", ext)
    burst("drafter extend staging only", ext_stage)
    burst("drafter draft traced (stage + replay + readback + select_path)", drf)
    burst("drafter draft replay only (+ sync)", drf_device)
    burst("drafter draft staging only", drf_stage)
    burst("mock iteration (resident 1 row + argmax + extend + draft)", iteration)
    # context dependence of the draft (sliding window 2048): draft at a long drafter context
    far = nb * bs - 2 * R - 64
    burst("drafter draft traced at C=%d" % far, lambda: drafter.draft(int(prompt[-1]), far, traced=True))
    out["timing"] = tm
    out["phases"] = clock.rows
    drafter.free()
    model.release_trace()
    return out


# ---------------------------------------------------------------------------------------------- proxy
def read_timeline(model, layers, interval):
    """streamer 0 of chip 0: per layer (duration in wall-clock ticks, reader stall ticks of each entry)"""
    ts = ttnn.to_torch(model.ts_t, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh, dim=0)).to(torch.int64)
    ts = ts & 0xFFFFFFFF
    S = model.S
    rows = []
    for chip in range(model.d.n):
        per_core = []
        for i in range(S):
            t = ts[chip * S + i].reshape(layers, 32)
            ends = t[:, 3]
            dur = (ends[1:] - ends[:-1]) % 2**32
            # compute's phase ends (streamer_compute.cpp marks 0..9): per layer the ticks of rmsnorm, projection, conv,
            # out-projection (incl. the wait for the mixer), the gap to the mlp half, rmsnorm, gate|up, silu, down
            c = t[:, 16:26]
            phases = ((c[1:, 1:] - c[1:, :-1]) % 2**32).tolist()
            per_core.append(
                dict(
                    dur=dur.tolist(), stall=t[1:, 4:8].tolist(), span=int((ends[-1] - ends[0]) % 2**32), phases=phases
                )
            )
        rows.append(per_core)
    return rows


def summarize_timeline(tl, layers, interval, step_ms):
    """per layer kind: mean duration (ms, scaled by the step's measured time over the timeline span) and mean
    reader stall per entry (projection, out, gate|up, down), over all streamers of all chips"""
    spans = [c["span"] for chip in tl for c in chip]
    span = med(spans)
    # the timeline covers layers 1..L-1 of the step; scale ticks to ms with the steady per-layer rate
    tick_ms = step_ms / span * (layers - 1) / layers if span else 0
    kinds = {"attn": [], "gdn": []}
    stalls = {"attn": [], "gdn": []}
    for chip in tl:
        for c in chip:
            for k, (dur, st) in enumerate(zip(c["dur"], c["stall"])):
                l = k + 1
                kind = "attn" if is_attn(l, interval) else "gdn"
                kinds[kind].append(dur)
                stalls[kind].append(st)
    phases = {"attn": [], "gdn": []}
    for chip in tl:
        for c in chip:
            for k, ph in enumerate(c["phases"]):
                phases["attn" if is_attn(k + 1, interval) else "gdn"].append(ph)
    out = {}
    for kind in kinds:
        if not kinds[kind]:
            continue
        m = sum(kinds[kind]) / len(kinds[kind])
        s = torch.tensor(stalls[kind], dtype=torch.float64).mean(0).tolist()
        ph = torch.tensor(phases[kind], dtype=torch.float64).median(0).values.tolist()
        out[kind] = dict(
            layer_ms=m * tick_ms,
            stall_ms=[x * tick_ms for x in s],
            layers=len(kinds[kind]) // len(spans),
            # rms1->proj, proj->conv, conv->out, out->mlp, mlp->rms2, rms2->gu, gu->silu, silu->down (median, us)
            phase_us=[round(x * tick_ms * 1e3, 1) for x in ph],
        )
    out["tick_ms"] = tick_ms
    return out


def run_proxy(mesh, a):
    clock = Clock(mesh, log)
    ck, d, layers, interval, n_attn = target(mesh)
    n_gdn = layers - n_attn
    w = Q.load(ck, d, layers, interval)
    positions = [int(p) for p in a.positions.split(",")]
    max_pos = max(positions) + 128
    out = dict(positions=positions, configs={})
    g = torch.Generator().manual_seed(3)
    for cfg in a.configs.split(","):
        verify = cfg.startswith("V")
        R = int(cfg[1:].split("G")[0]) if verify else 0
        B = int(cfg[1:].split("G")[0])
        G = int(cfg.split("G")[1]) if "G" in cfg else 0
        if G:
            os.environ["RESIDENT_HEAD_GROUPS"] = str(G)
        else:
            os.environ.pop("RESIDENT_HEAD_GROUPS", None)
        sts = State(d, n_gdn, n_attn, 0, max_pos, zero=True, kv=False)
        if not verify and B > 1:
            sts = [sts] + [State(d, n_gdn, n_attn, 0, max_pos, zero=True, kv=False) for _ in range(B - 1)]
        model = ResidentModel(
            mesh, d, w, sts, layers, interval, True, verify=R if verify else False, timeline=a.timeline
        )
        clock(f"{cfg}: built (G={model.head_groups}, ring {model.ring_bytes})")
        x0 = torch.randn(B, HIDDEN, generator=g) * 0.05

        def tok(p):
            if verify:
                return model.token_state(x0, list(range(p, p + R)), ring=p, slot=0)
            if B == 1:
                return model.token_state(x0[0], p)
            return model.token_state(x0, [p] * B, ring=p)

        model.step(tok(positions[0]))
        ttnn.synchronize_device(mesh)
        model.capture_trace()
        rows = {}
        for p in positions:
            th = tok(p)
            model.step(th)
            ttnn.synchronize_device(mesh)
            time.sleep(1.5)
            per, host = [], []
            for _ in range(a.reps):
                # device time: replays back to back on the token state written above
                t0 = time.perf_counter()
                for _ in range(a.iters):
                    model.step(None)
                ttnn.synchronize_device(mesh)
                per.append(ms(t0) / a.iters)
            for _ in range(a.reps):
                # with the host's per-step work: token state write, replay, argmax readback
                t0 = time.perf_counter()
                for _ in range(a.iters // 3):
                    model.step(th)
                    model.argmax()
                host.append(ms(t0) / (a.iters // 3))
            row = dict(step_ms=med(per), all_ms=per, host_loop_ms=med(host))
            if a.timeline:
                model.step(th)
                ttnn.synchronize_device(mesh)
                tl = read_timeline(model, layers, interval)
                row["timeline"] = summarize_timeline(tl, layers, interval, row["step_ms"])
                raw = ttnn.to_torch(model.ts_t, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh, dim=0))
                row["raw_s0"] = (raw[0].to(torch.int64) & 0xFFFFFFFF).reshape(layers, 32).tolist()  # chip 0 streamer 0
            rows[p] = row
            log(
                f"{cfg} pos {p}: {row['step_ms']:.2f} ms/step device, {row['host_loop_ms']:.2f} ms with host I/O "
                + (json.dumps(row.get("timeline")) if a.timeline else "")
            )
        out["configs"][cfg] = dict(G=model.head_groups, ring_bytes=model.ring_bytes, rows=rows)
        model.release_trace()
        del model
        gc.collect()
        ttnn.synchronize_device(mesh)
    out["phases"] = clock.rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["coexist", "proxy"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--new", type=int, default=12)
    ap.add_argument("--max-pos", type=int, default=4096)
    ap.add_argument("--draft-blocks", type=int, default=64)
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--positions", default="128,2048,8192,32000")
    ap.add_argument("--configs", default="B1,V2,B8G1,B8G2,B4G1")
    ap.add_argument("--timeline", type=int, default=1)
    ap.add_argument("--trace-region", type=int, default=268435456)
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    mesh = open_mesh(trace_region=a.trace_region)
    res = {}
    try:
        res = run_coexist(mesh, a) if a.mode == "coexist" else run_proxy(mesh, a)
    finally:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
        close_mesh(mesh)


if __name__ == "__main__":
    main()
