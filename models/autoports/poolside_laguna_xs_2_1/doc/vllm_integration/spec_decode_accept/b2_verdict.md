# Stage B2 — on-device spec-decode verdict: correctness-proven, does NOT net-win (blocker identified)

On-device driver: `doc/vllm_integration/scripts/spec_decode_driver.py` (full 40-layer model, 1×4 mesh,
weight-cache off). Logs: `b2_device.log` (prefill-verify), `b2_decodeverify.log` (traced decode-verify).

## Results

| verify path | ISL | K | correctness | spec t/s/u | baseline t/s/u | ratio | note |
|---|---|---|---|---|---|---|---|
| prefill (working) | 4096 | 16 | **token-identical ✅** | — | — | — | mean_accept 3.53/16 |
| prefill (working) | 32768 | 16 | (same path) | 5.17 | 5.54 (greedy-via-verify) | **0.93×** | mean_accept 1.20/16 |
| **traced decode-verify** | 8192 | 8 | — | — | — | — | **CRASHES at `warmup_verify_decode` trace capture** |

## Why the B1 projection (~2.5×) does not materialize on device

B1's host replay measured **acceptance** (mean committed tokens per verify ≈ 2.5) and correctly projected a
~2.5× speedup **assuming verify ≈ one decode step**. On device that assumption only holds for the **traced
batched decode-verify** path — and that path **crashes at trace capture** (the CCL-trace-coexistence hazard:
the resident B=1 decode trace + a B=K+1 verify trace deadlock the mesh; the driver's `selfcheck` mode exists
because this path was already known-fragile).

The **working** verify path is suffix-**prefill** verify. It is KV-read-efficient (K+1 query rows share one
KV read), but a single prefill-verify over a long context is inherently far slower than a traced decode step:
at 32k a prefill-verify is ~180 ms vs a production traced-decode step ~77 ms (~2.4×). So even at the measured
accept (mean m+1 ≈ 2.2), spec-decode's effective per-token time (~184 ms) is ~2.4× **slower** than production
traced decode (~77 ms). Smaller K lowers verify cost but also lowers accept; the prefill-verify baseline is
already ~2–4× slower than traced decode, so no K makes prefill-verify beat traced decode.

## Verdict

**Do NOT ship spec-decode on the current stack.** It is correctness-proven (token-identical to greedy) and
has strong acceptance on real agent trajectories (B1), but nets ~break-even/slower because the only verify
path fast enough to win (traced batched decode-verify) crashes at trace capture. **Follow-up to unlock it:**
fix the traced decode-verify trace capture (resolve the CCL two-trace coexistence — capture the B=K+1 verify
trace as the *only* resident decode-bearing trace, as the driver already attempts) so verify ≈ one decode
step; then B1's ~2.5× accept converts to a real ~2× decode speedup. Config for that follow-up (from B1):
min_n=1, K=16, max_n∈{8,10}.

The shipped **k64 decode config** (Stage A, teacher top1 0.95 + fast) remains the durable per-user decode win
of this goal; spec-decode is a documented, correctness-proven follow-up, not a regression.
