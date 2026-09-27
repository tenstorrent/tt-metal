"""Real-checkpoint verification + cost of the streaming (chunk-causal, bucketed) traced CFM solve.

Recreated 2026-09-25 (the 2026-09-23 original was lost with an instance, never committed), and extended per
review: the lost version only checked chunk-ALIGNED valid lengths (300 -> 384, 650 -> 768), which cannot tell
whether the padding term reaches the attention -- at an aligned length the chunk-causal term alone already hides
every padded key. Real streaming produces non-aligned lengths at every utterance's final `finalize=True` flow call
(all remaining tokens), so this checks both kinds, at all three real bucket sizes.

Real `flow.pt` weights throughout (encoder, encoder_proj, spk affine, estimator). Conditioning: `mu` is the real
encoder's streaming-mode output (+ encoder_proj) for a fixed random token sequence -- prefix-consistent across
lengths, as a growing prefix is; `spks` is the real affine layer on a random unit x-vector; `cond` carries a
200-frame "prompt" region (a copy of mu's first 200 frames, standing in for a real prompt mel -- no audio assets on
this instance). Bucket padding (positions >= valid) is filled with large random garbage, not zeros, so any leak
into padded keys shows.

Per (valid, bucket) case, 10-step solves, all streaming=True:
  torch_exact  -- `CausalConditionalCFMRef` in fp32 on CPU at the EXACT valid length: the ground truth.
  tt_exact     -- traced TT solve at the exact valid length (the lost script's "exact-length reference").
  tt_bucket    -- traced TT solve at the bucket size, padding masked by the real mask.
  tt_chunkonly -- (non-aligned cases) the same bucket with an all-ones mask: chunk-causal term only, padding
                  visible -- the negative control for a missing padding term.
Per bucket, the bucketed solves run back to back (aligned first, then non-aligned, then the chunk-only control),
so the non-aligned case and the control REUSE the aligned case's trace (same key) -- the per-solve padding
refresh on trace reuse, with real weights. (The first run of this script interleaved the exact-length solve
between them, which evicted the bucket trace at capacity 1, so reuse was not exercised; fixed.) The chunk-only
control is compared against the TT exact-length solve as well as torch: against TT exact the correct mask gives
max|diff| 0, so any leak shows directly instead of hiding under the bf16-vs-fp32 gap. A non-streaming TT-vs-torch
solve at each aligned length is the baseline for that gap.

Pass/fail (exit status 1 on any failure), all at the valid positions:
  - TT bucket vs TT exact: max|diff| <= BUCKET_EXACT_MAX_ABS. Same program family, same inputs, padding fully
    masked, so the two should agree to a few bf16 ulps. They have been bit-identical on every run so far, but
    bucketing changes shapes and chunking, so bit-identity is reported, not required.
  - The chunk-only control MUST FAIL that same check against TT exact. PCC alone cannot be the gate: on
    2026-09-27 the control scored PCC 0.9986-0.9997 against TT exact, above any 0.99 gate, with max|diff| 2-3.
    If the control passes, the check cannot see a missing padding term at that length.
  - Each non-aligned case and its control must actually have reused the aligned case's trace. Otherwise the
    per-solve padding refresh on reuse was not exercised.

Timing (host wall clock, device-synchronized, trace allocation tracker OFF): per bucket, the capture solve, a
recapture after release with warm kernels/weights, and steady-state reuse solves; plus a non-streaming traced
solve at the same T for the cost of the mask.

Run (device; python_env's .pth files put the repo, ttnn/ and tools/ on sys.path, so no PYTHONPATH):
  HF_HOME=/home/user/models /home/user/tt-metal/python_env/bin/python .../cfm_streaming_real_checkpoint_check.py
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file
from models.demos.audio.cosyvoice2.tt.flow.decoder import TtCausalConditionalCFM, TtCausalConditionalDecoder
from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef

N_STEPS = 10
GATE = 0.99
# TT bucketed vs TT exact-length solve, max|diff| at the valid positions. Outputs are O(1)-O(4) in magnitude, where
# a bf16 ulp is <= 1/64; the correct mask gave 0 and the chunk-only control 2.06-3.09 (2026-09-27, N150).
BUCKET_EXACT_MAX_ABS = 0.05
PROMPT_FRAMES = 200
# (valid mel frames, bucket mel frames). Buckets are the linear-step-64 token buckets x2 (tt/flow/encoder.py's
# bucket_length). Aligned = multiple of CHUNK_SIZE_UP (50).
CASES = [(300, 384), (310, 384), (650, 768), (660, 768), (1500, 1536), (1510, 1536)]
TIMING_REPS = 5


def build_conditioning(ref: CausalMaskedDiffWithXvecRef, max_frames: int):
    torch.manual_seed(0)
    tokens = torch.randint(0, ref.input_embedding.num_embeddings, (1, max_frames // 2))
    with torch.no_grad():
        h = ref.encoder(ref.input_embedding(tokens), streaming=True)
        mu = ref.encoder_proj(h)  # [1, max_frames, 80]
        xvec = torch.randn(1, 192)
        spks = ref.spk_embed_affine_layer(xvec / xvec.norm(dim=-1, keepdim=True))
    cond = torch.zeros_like(mu)
    cond[:, :PROMPT_FRAMES] = mu[:, :PROMPT_FRAMES]
    return mu, cond, spks


def padded(mu, cond, valid: int, bucket: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    mu_b = torch.randn(1, bucket, 80, generator=g) * 5.0
    cond_b = torch.randn(1, bucket, 80, generator=g) * 5.0
    mu_b[:, :valid] = mu[:, :valid]
    cond_b[:, :valid] = cond[:, :valid]
    mask = torch.zeros(1, bucket, 1)
    mask[:, :valid] = 1.0
    return mu_b, cond_b, mask


def timed_solve(tt_cfm, mu, mask, spks, cond, streaming):
    ttnn.synchronize_device(tt_cfm.device)
    t0 = time.perf_counter()
    out = tt_cfm.forward(mu, mask, N_STEPS, spks, cond, use_trace=True, streaming=streaming)
    ttnn.synchronize_device(tt_cfm.device)
    return out, (time.perf_counter() - t0) * 1e3


def stats(a: torch.Tensor, b: torch.Tensor):
    _, pcc = comp_pcc(a, b, GATE)
    return float(pcc), (a - b).abs().max().item()


def main():
    ref = CausalMaskedDiffWithXvecRef.from_checkpoint(load_checkpoint_file("flow.pt"))
    ref.eval()
    mu, cond, spks = build_conditioning(ref, max(v for v, _ in CASES))

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000)
    rows, timing = [], []
    try:
        tt_cfm = TtCausalConditionalCFM(
            device,
            TtCausalConditionalDecoder(device, ref.decoder.estimator),
            ref.decoder.rand_noise,
            ref.decoder,
            trace_cache_capacity=1,
        )
        for bucket in sorted({b for _, b in CASES}):
            group = sorted((v for v, b in CASES if b == bucket), key=lambda v: (v % 50 != 0, v))  # aligned first
            key = (bucket, 80, True)
            tt_cfm.release_cfm_trace()
            # 1. Bucketed solves back to back: the first captures this bucket's trace, every later one (the
            #    non-aligned valid length, then the chunk-only control) REUSES it -- the padding term must follow
            #    each solve's own mask, not the capturing solve's.
            bucket_out, reused, controls, first_slot = {}, {}, {}, None
            for i, valid in enumerate(group):
                mu_b, cond_b, mask_b = padded(mu, cond, valid, bucket, seed=100 + valid)
                bucket_out[valid] = timed_solve(tt_cfm, mu_b, mask_b, spks, cond_b, True)[0][:, :valid]
                first_slot = first_slot or tt_cfm._traces.get(key)
                reused[valid] = i > 0 and tt_cfm._traces.get(key) is first_slot
                if valid % 50:
                    co = timed_solve(tt_cfm, mu_b, torch.ones_like(mask_b), spks, cond_b, True)[0][:, :valid]
                    controls[valid] = (co, tt_cfm._traces.get(key) is first_slot)
            # 2. References at the exact valid length: TT traced (the lost script's reference) and torch fp32.
            for valid in group:
                mu_v, cond_v, ones_v = mu[:, :valid], cond[:, :valid], torch.ones(1, valid, 1)
                t0 = time.perf_counter()
                with torch.no_grad():
                    want = ref.decoder.forward(mu_v, ones_v, N_STEPS, spks=spks, cond=cond_v, streaming=True)
                t_torch = time.perf_counter() - t0
                tt_cfm.release_cfm_trace()
                tt_exact, _ = timed_solve(tt_cfm, mu_v, ones_v, spks, cond_v, True)
                row = {
                    "valid": valid,
                    "bucket": bucket,
                    "aligned": valid % 50 == 0,
                    "reused_trace": reused[valid],
                    "bucket_vs_torch": stats(want, bucket_out[valid]),
                    "exact_vs_torch": stats(want, tt_exact),
                    "bucket_vs_exact": stats(tt_exact, bucket_out[valid]),
                    "torch_s": t_torch,
                }
                if valid in controls:
                    co, co_reused = controls[valid]
                    row["chunkonly_vs_torch"] = stats(want, co)
                    row["chunkonly_vs_exact"] = stats(tt_exact, co)
                    row["chunkonly_reused_trace"] = co_reused
                if valid % 50 == 0:
                    # Baseline: the same length NON-streaming, TT vs torch -- how much of the TT-vs-torch gap is
                    # plain bf16-vs-fp32 over a 10-step solve, independent of streaming.
                    with torch.no_grad():
                        want_ns = ref.decoder.forward(mu_v, ones_v, N_STEPS, spks=spks, cond=cond_v)
                    tt_cfm.release_cfm_trace()
                    got_ns, _ = timed_solve(tt_cfm, mu_v, ones_v, spks, cond_v, False)
                    row["nonstreaming_tt_vs_torch"] = stats(want_ns, got_ns)
                rows.append(row)
                print(f"done valid={valid} bucket={bucket}: {row}", flush=True)
            tt_cfm.release_cfm_trace()

        # Timing, per bucket: capture (from a released state), recapture after release (kernels, conv weights
        # already warm), steady-state reuse; non-streaming traced at the same T for the mask's cost.
        for bucket in sorted({b for _, b in CASES}):
            valid = max(v for v, b in CASES if b == bucket and v % 50 == 0)
            mu_b, cond_b, mask_b = padded(mu, cond, valid, bucket, seed=7)
            tt_cfm.release_cfm_trace()
            _, ms_capture = timed_solve(tt_cfm, mu_b, mask_b, spks, cond_b, True)
            tt_cfm.release_cfm_trace()
            _, ms_recapture = timed_solve(tt_cfm, mu_b, mask_b, spks, cond_b, True)
            steady = [timed_solve(tt_cfm, mu_b, mask_b, spks, cond_b, True)[1] for _ in range(TIMING_REPS)]
            tt_cfm.release_cfm_trace()
            # Non-streaming needs an all-ones mask (fused SDPA runs unmasked); values are irrelevant for timing.
            ones_b = torch.ones(1, bucket, 1)
            timed_solve(tt_cfm, mu_b, ones_b, spks, cond_b, False)  # capture
            steady_ns = [timed_solve(tt_cfm, mu_b, ones_b, spks, cond_b, False)[1] for _ in range(TIMING_REPS)]
            tt_cfm.release_cfm_trace()
            timing.append((bucket, ms_capture, ms_recapture, steady, steady_ns))
            print(f"timed bucket={bucket}", flush=True)
        tt_cfm.release_cfm_trace()
    finally:
        ttnn.close_device(device)

    def f(x):
        return f"{x[0]:.6f} / {x[1]:.3g}"

    print("\n## PCC / max|diff| at the valid positions, 10-step streaming solve, real flow.pt\n")
    print(
        "| valid | bucket | chunk-aligned | reused aligned case's trace | TT bucket vs torch exact "
        "| TT exact vs torch exact | TT bucket vs TT exact | chunk-only control vs TT exact (reused trace) "
        "| chunk-only control vs torch exact | non-streaming TT vs torch (baseline) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        na = "n/a"
        co_e = f"{f(r['chunkonly_vs_exact'])} ({r['chunkonly_reused_trace']})" if "chunkonly_vs_exact" in r else na
        co_t = f(r["chunkonly_vs_torch"]) if "chunkonly_vs_torch" in r else na
        ns = f(r["nonstreaming_tt_vs_torch"]) if "nonstreaming_tt_vs_torch" in r else na
        print(
            f"| {r['valid']} | {r['bucket']} | {r['aligned']} | {r['reused_trace']} | {f(r['bucket_vs_torch'])} "
            f"| {f(r['exact_vs_torch'])} | {f(r['bucket_vs_exact'])} | {co_e} | {co_t} | {ns} |"
        )
    print("\n## Cost, 10-step traced solve (ms)\n")
    print(
        "| bucket (mel) | capture | recapture (warm) | streaming steady mean (min-max) | ms/Euler step "
        "| non-streaming steady mean (min-max) |"
    )
    print("|---|---|---|---|---|---|")
    for bucket, cap, recap, st, ns in timing:
        m, mns = sum(st) / len(st), sum(ns) / len(ns)
        print(
            f"| {bucket} | {cap:.1f} | {recap:.1f} | {m:.1f} ({min(st):.1f}-{max(st):.1f}) | {m / N_STEPS:.2f} "
            f"| {mns:.1f} ({min(ns):.1f}-{max(ns):.1f}) |"
        )

    failures = []
    for r in rows:
        tag = f"valid={r['valid']} bucket={r['bucket']}"
        d = r["bucket_vs_exact"][1]
        if d > BUCKET_EXACT_MAX_ABS:
            failures.append(f"{tag}: TT bucket vs TT exact max|diff| {d:.4g} > {BUCKET_EXACT_MAX_ABS}")
        if not r["aligned"] and not r["reused_trace"]:
            failures.append(f"{tag}: did not reuse the aligned case's trace (padding refresh on reuse not exercised)")
        if "chunkonly_vs_exact" in r:
            dc = r["chunkonly_vs_exact"][1]
            if dc <= BUCKET_EXACT_MAX_ABS:
                failures.append(
                    f"{tag}: chunk-only control PASSED the check (max|diff| {dc:.4g} <= {BUCKET_EXACT_MAX_ABS}), "
                    "so the check cannot detect a missing padding term here"
                )
            if not r["chunkonly_reused_trace"]:
                failures.append(f"{tag}: chunk-only control did not reuse the aligned case's trace")
    n_bitexact = sum(r["bucket_vs_exact"][1] == 0.0 for r in rows)
    print(f"\n## Verdict\n\nTT bucket vs TT exact bit-identical at {n_bitexact}/{len(rows)} (valid, bucket) pairs.")
    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        raise SystemExit(1)
    print(
        f"PASS: bucket vs exact max|diff| <= {BUCKET_EXACT_MAX_ABS} everywhere; every chunk-only control fails it; "
        "every non-aligned case and control reused the aligned case's trace."
    )


if __name__ == "__main__":
    main()
