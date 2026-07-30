# DiffusionGemma optimize-perf (#47465) — topology audit, roofline and device gates

Status: current for the weight roofline, the argmax result and the up-front device gates; **provenance-only** for the dense-128 traced step decomposition (that MoE is long gone).
Owns: the canonical weight-byte model and all-128 expert floor, the dense-128 traced step decomposition, the ROW_MAJOR argmax win, and the up-front-capture / GPQA-hang reproduction commands.
See also: [refuted list](../REFUTED.md) · [stage hub](README.md) · [campaign ledger](perf_progress.md)

Stage dg-08. Device: QB2 / `bh-qbge-06` / P150x4, mesh `(1,4)`, TP=4. Branch `diffusion-gemma-function`, baseline SHA `aff8f2105d3`, venv `/home/zni/venvs/tt-diffusion-gemma` (Python 3.12, ttnn + transformers 5.12.1), `build_Release`. Full environment recipe: [plan.md](../../plan.md).

> **MEASUREMENT TRAP / standing limitation.** `build_Release` has `ENABLE_TRACY=OFF`, so
> `TT_METAL_DEVICE_PROFILER=1` raises `TT_FATAL` and `tt-perf-report` op CSVs are unavailable. The
> approved substitutes are **synchronized per-op device timing** plus **Metal trace capture/replay**,
> which does work. Every op-level table here is a synchronized measurement, not a Tracy capture.

## Terminal decision path — the ROW_MAJOR argmax win

`ttnn.argmax` over the 262144 vocab runs **single-core on TILE input at 1239.7 ms** and **multi-core
on ROW_MAJOR input at 14.4 ms**, and the ROW_MAJOR result is **bit-identical** (exact match) to the
TILE result — an **86× per-op** win, wired into `sampling.argmax_last_dim`. It was called twice per
step, so it removed ~2480 ms/step at full depth: the full-depth traced step fell **~37%**
(6642 → 4175.7 ms) and the terminal sub-path **~58×** (eager ≈2494 ms → traced 43.06 ms).

| terminal variant (`[1,1,256,262144]`, traced, warmed) | ms/step |
|---|---|
| original (TILE argmax ×2 + per-call `ttnn.full`) | **untraceable**; eager ≈ 2494 |
| ROW_MAJOR argmax + preallocated constants | **43.06** |
| + `share_z` (reuse the temperature-scaled `z` across gumbel argmax, clean argmax and entropy) | **42.30** (kept) |

- **LANDED:** preallocated accept/renoise constants (`make_denoise_constants`) are both required for trace safety and **23% faster** (0.416 → 0.321 ms), because `ttnn.full` / `ttnn.zeros_like` inside a capture raise `TT_FATAL: Writes are not supported during trace capture`.
- The whole net-new sort/cumsum/scatter entropy-accept chain over the 256 axis costs **0.566 ms**, ~1.3% of the step — it is not and never was a bottleneck.
- Refuted alternatives (`topk` k=1/k=32, `chunked_entropy`, L1 placement of the accept entropy tensor): one line each in the [refuted list](../REFUTED.md). Why the terminal reduction cannot be pushed further (reduction-op limit, the 18-bit-index bf16 wall): [nonmoe_roofline/README.md](nonmoe_roofline/README.md).

## Canonical weight-byte roofline

Unlike autoregressive decode, each denoise step **re-reads the model weights and recomputes over the full 256-token canvas against the frozen prefix** — there is no incremental single-token KV read, so the per-step floor is set by **weight traffic**. From `config.json`: H=2816, 30 layers, qh16/kv8/hd256, shared intermediate 2112, 128 experts × moe_inter 704, vocab 262144 → **25.16B params → 50.3 GB bf16 → 12.58 GB/chip at TP=4** (consistent with the 13.24 GiB build).

| per-step weight traffic scenario | GB/chip | @256 GB/s | @512 | @1024 |
|---|---|---|---|---|
| all 128 experts activated | 12.58 | **49.1 ms** | 24.6 ms | 12.3 ms |
| hypothetical 8/128 experts (perfect reuse — not reachable) | 1.88 | 7.3 ms | 3.7 ms | 1.8 ms |
| dense only (no experts) | 1.16 | 4.5 ms | 2.3 ms | 1.1 ms |

Split: non-expert (embed + LM head + 30× attn/shared-MLP/router/norms) **2.32B params = 1.16 GB/chip**; experts (30 × 128 × gate/up/down) **22.84B = 11.42 GB/chip**.

**CANONICAL structural fact.** A 256-token canvas with top-8 routing makes 256×8 = 2048 (token, expert) pairs over 128 experts, so by coupon-collector `E[distinct] = 128·(1 − e⁻¹⁶) ≈ 128.0` — essentially **all 128 experts activate**, and the all-experts row is the real per-step weight floor. **Top-8 sparsity buys compute and data movement, never weight bytes, at this canvas width.** The all-128-resident build is ~13.1–13.27 GiB/chip, ~88.6–89% of weight DRAM. The practical per-chip bandwidth denominator (~235 GB/s achievable vs ~256 GB/s nominal) used for every bandwidth-efficiency claim lives in [nonmoe_roofline/README.md](nonmoe_roofline/README.md).

## Dense-128 traced step decomposition — PROVENANCE ONLY

Reduced-layer traced fit (L=1/2/4 → 30L; the method and the profiler op-buffer ceiling that forces
it are owned by [whole_gen_opprofile/README.md](whole_gen_opprofile/README.md)):

| metric (traced, QB2 `(1,4)` TP=4, dense-128 era) | value |
|---|---|
| per-layer denoise | **137.55 ms/layer** |
| fixed overhead (embed + LM head + terminal sampling + final norm) | **49.24 ms** |
| full 30-layer denoise step | **4175.7 ms** |
| commit (256 single-token decode-appends) | 1042 ms/layer + 206 ms fixed = **31.5 s/block** |
| per block (fixed 48 steps + commit) | **≈ 231.9 s**; 256 tokens/block |

Traced was **within ~3% of eager**, so that path was **op-cost bound, not dispatch-gap bound**: 98.8% of the step was the per-layer backbone, and the measured step sat ~85–170× the bandwidth roofline. The op-count causes were named at the time: `_chunked_norm_forward` (8 slices + 8 sharded norms + concat per canvas norm, ~90 ops/layer), `_apply_rope_chunked` (manual per-32-token per-head RoPE), and `_manual_gqa_attention` (the ttnn SDPA missed its L1 static CB by <1 tile). All three are closed: the full-canvas norm shipped ([stage hub](README.md)) and the GQA-fallback claim was refuted at 0/30 fallback layers ([refuted list](../REFUTED.md)).

## Correctness floor and gates

- `run_fixed_denoise_steps` replays with committed argmax **100.00% identical** to eager under device
  canvas feedback with no host readback (`TRACE_SAFE_OK`).
- `tests/test_device_entropy_accept.py` + `tests/test_tt_sampling.py` = **18 passed** with
  `DG_RUN_DEVICE=1` on QB2.
- Isolation gate: `git diff aff8f2105d3..HEAD -- models/demos/gemma4/` is empty — only
  `models/experimental/diffusion_gemma` changed.

## Reproduction — up-front capture device gates (2026-07-22)

env: see [plan.md](../../plan.md). Each device invocation below was originally run with three
additional MoE/terminal env exports that have since been deleted and are now inert — see
[flag triage](flag_triage_20260728.md).

```
# CPU
pytest -q tests/test_upfront_capture.py tests/test_serving_block_contract.py -k 'not test_device'
# -> 23 passed, 1 skipped, 4 deselected

# NOTE (2026-07-30): all three commands were repaired. Their old test names no longer exist -- the
# tests were renamed -- and DG_UPFRONT_STEPS / DG_UPFRONT_BASELINE_CONTROL have no reader anywhere in
# the repo, so they set nothing. The step count is fixed at 48 by UPFRONT_DENOISE_STEPS, not by env.
# The recorded timings below were measured against the OLD test bodies; treat them as historical
# magnitudes, not as expectations for these runs.

# device mechanics gate: one 48-trace capture reused across two different prompt lengths
DG_RUN_DEVICE=1 DG_TRACE_REGION_SIZE=1073741824 DG_DENOISE_REVEAL_PMAX=1024 \
DG_UPFRONT_NUM_LAYERS=full pytest --timeout=600 -q \
  tests/test_upfront_capture.py::test_device_startup_capture_reuses_one_48_trace_set
# -> historical: 1 passed in 29.48s

# stale-cross-request-state gate: two sequential requests must match eager with no recapture
DG_RUN_DEVICE=1 DG_TRACE_REGION_SIZE=10737418240 DG_DENOISE_REVEAL_PMAX=1024 \
DG_UPFRONT_NUM_LAYERS=full pytest --timeout=900 -q \
  tests/test_upfront_capture.py::test_device_two_sequential_requests_match_eager_without_recapture
# -> historical: 1 passed in 154.02s; 48 traces captured once, four blocks replayed, 192 trace
#    executions, no recapture

# eager baseline control: up-front must match eager on tokens, realized K and halt
DG_RUN_DEVICE=1 DG_TRACE_REGION_SIZE=10737418240 \
DG_DENOISE_REVEAL_PMAX=1024 DG_UPFRONT_NUM_LAYERS=full pytest --timeout=900 -q \
  tests/test_upfront_capture.py::test_device_upfront_matches_eager_tokens_realized_k_and_halt
# -> historical: 1 passed in 90.68s
```

Digests to compare against: three-way committed SHA256
`924ae03b6111734d8ab1d2d4c88ec6a7da5ba6612c50b2f0e3c27d0511980e0f` (up-front / per-request / eager)
and prompt-B baseline `82dac3229b72134447b6ad8f1571a6520215c9e0642b07c8a5a715d3706075b4`. Artifacts:
`upfront_reuse_across_prompts.json`, `upfront_bit_exactness.json`, `upfront_multi_request_smoke.json`,
`triage/upfront_control_hang_{tt-triage,summary}.txt`.

A full-depth K=48 run without MoE tuning hit pytest's 300 s timeout while synchronizing the first
48-trace replay after a **successful** capture — a workload-timeout control, not a trace-correctness
failure.

## GPQA second-request hang (2026-07-22)

**ROOT CAUSE.** vLLM's compile-only phase deferred without compiling real prefill shapes, so the
first real 160-token prompt compiled and allocated a new prefill program while 48 denoise traces were
active; that violated trace address stability and corrupted CCL state, and the next prefill stalled in
`AllBroadcast` with all four devices in the causal-prefill broadcast writer waiting on its semaphore.
Triage: `triage/upfront_earlyhalt_gpqa_hang_tt-triage.txt`. The fix and the three refuted hypotheses
are in [stage hub](README.md) and the [refuted list](../REFUTED.md).

**VALIDATION.** `DG_UPFRONT_PREFILL_WARMUP_LENS=160,192,256,384,480` ran eight sequential real
GPQA-Diamond requests with traced early halt: all passed and released, realized K 10–43, TTFT
5.38–18.63 s, `capture_events` stayed 48. Artifact: `upfront_earlyhalt_gpqa_20260722.{json,md}`.

- **MEASUREMENT TRAP:** that eight-request lm-eval exact-match score was 0 **only because the
  lifecycle run capped generation at one 256-token block**, truncating reasoning before answer
  extraction. It is not a quality result.
- **LIFECYCLE TRAP:** `release_persistent_capture()` is terminal shutdown immediately before mesh
  close — a test-only same-process release-then-recapture stalled in `AllBroadcastDeviceOperation`.
- **TRIAGE TRAP:** a compact tt-triage summary reports script execution status, not an idle-device
  integrity verdict, and rows sampled while a broadcast is running are not post-recovery checks.
  Reset plus a `(1,4)` mesh open/close smoke is the health evidence.
