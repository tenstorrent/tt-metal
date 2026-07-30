# DiffusionGemma — optimize-perf stage hub (#47465)

Status: current — the serving/model-path contract for dg-08.
Owns: shipped MoE + norm paths, up-front capture, trace hazards, `DG_VLLM_GUMBEL_MODE`, the @48 headline, the #51080 lever table, optimize-perf traps.
See also: [refuted list](../REFUTED.md) · [environment recipe](../../plan.md) · [flag triage](flag_triage_20260728.md) · [campaign ledger](perf_progress.md)

Scope: the optimization unit is the **denoise step over the 256-token canvas** (≤48 steps/block) plus the commit, on QB2 `bh-qbge-06` P150x4 mesh `(1,4)` TP=4 — *not* per-token autoregressive decode. Preserved policy: BF16 weights/activations/KV, the BF16 ordered online-chunk self-conditioning reduction, and the diffusion decisions (temperature 0.8→0.4, Gumbel-max, entropy-budget accept, random-token renoise, commit = clean argmax). No `models/demos/gemma4/` edits ([no-shared-edits rule](../../AGENTS.md)).

## Shipped defaults, and what selects them

- **Up-front model-lifetime trace capture is the ONLY supported Metal denoise trace path.** `DG_UPFRONT_CAPTURE` defaults to `1`; `DG_UPFRONT_CAPTURE=0` is the eager opt-out and is required when per-step trajectory records are needed — replayed traces do not produce them.
- `DG_UPFRONT_PREFILL_WARMUP_LENS` must list every admitted aligned prefill length and `DG_TRACE_REGION_SIZE` must be an explicit positive reservation. Both stay **fail-loud** and are deliberately not defaulted: Metal takes the trace region as an open-time constructor argument with no getter, so a default would silence the guard without reserving anything, and a trace-region overflow poisons the device (`tt-smi -r`). The 48-step capture is enforced by the `UPFRONT_DENOISE_STEPS` constant in `tt/generator_vllm.py`, not by an env flag.
- `DG_DENOISE_REVEAL_PMAX` is **optional**: unset, it is derived as the tile-rounded served `max_model_len` and logged at startup; an explicit value must be positive and tile-aligned.
- `DG_VLLM_GUMBEL_MODE` defaults to **`device`** — ~53.6 vs ~36.3 tokens/block/s against `host` (~1.48×). History: device default 2026-07-24, reverted to `host` 2026-07-25 for corrupted text (matched 4-seed A/B: `host` 4/4 correct, `device` 2/4 corrupted), restored after the `ttnn.rand` kernel fix, and `host` **deleted 2026-07-28** — there is no IID reference arm any more. The default DEPENDS on that kernel fix (the Blackhole SFPU PRNG is a sliding window over one stream, so 64 of 256 canvas positions received a byte-identical copy of another position's noise); its residual is pinned by `tests/ttnn/.../test_rand_independence.py`. See [device Gumbel restored](../decision_fidelity/device_gumbel_restored.md). `chunked` and `argmax` are not materialized full-tensor sources and are rejected under up-front capture.
- **The denoise MoE is `tt/concat_moe.py` concat-experts and it is the ONLY one**; it also owns the expert precision policy. `tt/sparse_moe.py` is now prefill-only (ragged zero-drop top-8) and nothing in it is on the denoise path. `DG_MOE_CONCAT`, `DG_SPARSE_MOE` and `DG_ALLOW_DENSE_MOE` no longer exist — **these names do nothing**; see [flag triage](flag_triage_20260728.md). Why the token-gather path was deleted: [winter borrow](winter_borrow_20260727.md) and the [refuted list](../REFUTED.md). The OPT-004 "3.47× / ~13×" geometry numbers were C=32 measurements of that deleted path — provenance only ([opt004_matmul_geometry.md](opt004_matmul_geometry.md)).
- **The denoise RMSNorm runs the whole 256-row canvas in ONE width-sharded op** (shipped as the only path 2026-07-30). With fp32 partial accumulation the 256-row and 8×32-row shapes are bit-identical (0 of 69,206,016 elements over 96 device slices, vs 13.0% differing at bf16), the norm is 2.8× more accurate against an fp64 reference, and it costs −2.3%. Root cause of the historical delta: ttnn's `rmsnorm` defaults to bf16 partial accumulation (`fp32_acc=false`) and no DG or gemma4 norm had ever passed a `compute_kernel_config`. Mechanism and the void flip gate: [l1_residency.md](l1_residency.md).
- `DG_PREFILL_RAGGED_LONG` defaults on: every multi-token prefill uses the ragged top-8 path and sequences above 4096 are processed in 4096-token slices. Setting it to `0` reproduces the historical dense all-128-expert fallback and the 4K→16K cliff.
- Prefill MoE tuned geometry is default-on **only** for the measured Blackhole QB2 `(1,4)` TP=4, 11×10-grid, BF16, 128-expert shape (gate/up `6x1` grid, K-block 44; down `11x4`, K-block 3, two N tiles/core). `DG_PREFILL_MOE_TUNED=0` selects the stock fallback and a context-local dispatcher stops concurrent Gemma4 calls inheriting it. Result: dense 256-token layer-0 MoE 135.51 → 21.16 ms (6.40×) elementwise exact; warmed full 30-layer 1024-token forward 16.3414 → 2.6158 s (6.247×); final logits and every KV shard exact (`max_abs=0`), including a non-aligned 1001-token prompt.
- The batched commit collapses 256 single-token decode-appends into ONE causal masked prefill-append over the 256-token canvas (`tt/commit_batched.py` + `reference/attention_mask.py` `causal=True`); it is the default and its flag is gone. Correctness argument and the invalid-gate ruling: [commit_batching.md](commit_batching.md).
- Reveal masking, non-lazy startup capture and window-1 early halt are **intrinsic** to the up-front path. Do not add legacy selector flags for them.

## The @48 model-faithful headline and the decision-fidelity baseline

Model-faithful throughput is measured at K=48. Traced serving denoise **17.92 tokens/block/s** (14.289 s steady block) = **2.72×** over eager 6.58, with coherent K=48 text byte-identical to the eager adaptive path; the 2026-07-10 selected default reproduces **18.844 t/s / 13.5849 s @48**. The canonical identity gate every candidate must preserve is the three-block clean-commit digest `a9f0d18709b07d1e` @48 (`24393ba7aad6077c` @12) under the canonical prompt, seed and fixed step budget. The era's verdict, "precision-neutral levers exhausted at 17.8–18.2 t/s", has been superseded twice (see the [refuted list](../REFUTED.md)).

**These absolute numbers were measured before the concat MoE became the only denoise MoE and are provenance, not current results.** Ledger: [perf_progress.md](perf_progress.md).

## The denoise per-step cost

> **OPEN CONTRADICTION (unexplained):** the denoise per-step cost is stated as ~233 ms traced (2026-07-08, [perf_progress.md](perf_progress.md)), 257.575 ms warmed traced (2026-07-10, [selfcond_prechunk.md](selfcond_prechunk.md)), ~447 ms traced / ~484 ms eager projection (2026-07-15, [perf_progress.md](perf_progress.md)), ~465–540 ms ([context_speed_sweep_20260722.md](context_speed_sweep_20260722.md)), ~0.9 s with MoE 89% ([official_sampler_earlyhalt_20260722.md](official_sampler_earlyhalt_20260722.md)), ~428–496 ms with MoE 56.9% ([upfront_gumbel_overlap_devicemode_20260724.md](upfront_gumbel_overlap_devicemode_20260724.md)), and ~4–5.6 s ([context_speed_sweep_20260722.md](context_speed_sweep_20260722.md), absorbed from the deleted `ttft_ts_sweep.md`). Each was measured on a different, now-superseded MoE path. Not explained.

No per-step figure has been measured on the concat MoE in these docs. The nearest current-path datum is the 2026-07-28 device arm in [flag triage](flag_triage_20260728.md): concat `7b29837d637ec26b` at 9.449 s/block.

## Trace hazards and the trace-lifetime rule

- **Allocate every persistent cross-replay buffer (canvas / committed / signal / rope / noise / init) BEFORE `begin_trace_capture`.** A buffer allocated into post-capture-freed memory overlaps trace scratch and is clobbered on every replay: replay 1 is right, replays 2+ diverge. The historical 60.5% whole-loop-trace divergence was **this probe bug, not a self-conditioning race**.
- `ttnn.full` / `ttnn.zeros_like` / a cold `ttnn.copy` inside a capture raise `TT_FATAL: Writes are not supported during trace capture`. Warm the copy eagerly first, and clone the ACTUAL first-step outputs so layouts match (argmax is ROW_MAJOR uint32, the canvas buffer is TILE).
- **A trace-capture FATAL poisons the device** — the next `open_mesh_device` hangs at 0% CPU (observed twice) and needs `tt-smi -r`. A clean non-fatal exit leaves the device healthy.
- Compiling a new prefill program while traces are active can corrupt trace/CCL state, so every admitted prefill shape must compile before capture and unseen runtime shapes fail loudly.
- **A borrowed tensor needs a BUFFER audit, not an object audit**: `ttnn.to_memory_config` returns a fresh Tensor object that can alias its input's buffer, so an `is not` check deallocated the model KV cache and the next op died with `Input Tensor is not allocated`. See `_is_distinct_buffer` in `tt/diffusion_attention.py` and [per_layer_prefix_spans.md](per_layer_prefix_spans.md).

## Up-front denoise capture contract

Capture happens once, during `warmup_model_prefill`, and the adapter/controller is retained for the model lifetime. Each request prefills the model-owned KV cache, rebinds the fixed-span adapter in place, replays the startup trace, and **detaches without releasing it**; the persistent adapter is not released at request teardown, and the wrapper destructor invokes the idempotent persistent-release path before inherited model/mesh teardown. `capture_events` stays at its startup value — there is no recapture. The reveal span is fixed at `p_max`, costs O(`p_max`) per step, and is also enforced as the served `prompt + generated` cap. Prefill-shape warmup is mandatory.

**Second-request prefill-hang root cause (2026-07-22).** vLLM's compile-only phase deferred without compiling real prefill shapes, so the first real 160-token prompt compiled and allocated a new prefill program while 48 denoise traces were active; that violated trace address stability and corrupted CCL state, and the next prefill stalled in `AllBroadcast` with all four devices in the causal-prefill broadcast writer waiting on its semaphore. The fix honours vLLM's two-phase warmup (`enable_trace=False` compiles configured prefill lengths, `enable_trace=True` captures denoise), makes decode warmup a no-op, requires `DG_UPFRONT_PREFILL_WARMUP_LENS` in vLLM mode, and rejects unseen aligned runtime prefill lengths before device execution. Validation and commands: [work_log.md](work_log.md).

Evidence artifacts to keep — all direct wrapper/session runs on 4× Blackhole p300c, no Tracy or live-server profiling: `upfront_reuse_across_prompts.json` (32→320→32 spans, `capture_events` stays 1, the 320-token prefill overwrites the mock commit span `[32:288]`), `upfront_bit_exactness.json` (chunked-Gumbel / per-request / eager SHA256 identity), `upfront_multi_request_smoke.json` (one capture / 192 trace executions, exact A roundtrip), `upfront_earlyhalt_gpqa_20260722.{json,md}` (eight sequential GPQA requests, K=10–43, zero recapture).

## Measurement traps

- **Three metrics, never compared.** Pure prefill, serving prefill, and DiffusionGemma "TTFT" (prefill + the first whole 256-token block) are different metrics, and none is autoregressive TTFT. The historical 18.844 t/s row is warmed same-shape argmax replay with prompt-only prefix visibility — not first-request TTFT and not multi-request throughput.
- **Never rank on a full-generation total, and never time block 0.** Block 0 carries first-block trace capture and program compilation (≈5.5 s vs ≈1.7 s steady in the q-chunk sweep). It produced both a bogus "+50% regression" and a bogus split-candidate win before the harness switched to `mean(per_block_latency_s[1:])`.
- GPQA needs three denominators and a `prefill_block0`-vs-question-count check — see [decision fidelity](../decision_fidelity/README.md).
- The #48291 doc-0 garbage (`níní…1111…`) was root-caused 2026-07-24 as a malformed thinking-template contract in the eval invocation (a manual `<|think|>` system message with server `enable_thinking` not applied, so the checkpoint template emits an empty-closed thought suffix) — NOT a sampler/precision/trace defect. The server-side `enable_thinking=true` contract fixed it and doc-0 was device-confirmed at `exact_match=1` with `\boxed{C}`.

> **OPEN CONTRADICTION (unexplained):** the full-canvas RMSNorm delta was published as "~2e-6/norm, PCC 0.999998" ([l1_residency.md](l1_residency.md)) and then retracted for **5.73 bf16 ULP** after the originating bench was found to report PCC > 1.0 elsewhere (floor ~5e-5) — four orders of magnitude apart, never reconciled. The related "answers get 27% shorter" objection was a 10-question artifact: −10% at 71 questions and gone at 198, where the full-canvas run was LONGER (11,069 chars) and scored **71.21% vs 66.67%** for the previous full 198-question run. Not explained.

> **OPEN CONTRADICTION (unexplained):** early halt is recorded as never firing under #48291 and as firing at `[9,17,2]`/48 and K=10–43 under the concat MoE. See [early_halt.md](early_halt.md). Not explained.

## Prefix-cost levers off #51080 (2026-07-24)

| item | outcome | evidence |
|---|---|---|
| SDPA `q_chunk` 32 → 64/128 | **refuted** — bit-exact but ≤1%, inside noise; default stays 32 | [refuted list](../REFUTED.md), `sweep_denoise_qchunk.sh` |
| Borrow the KV cache instead of cloning it per step | **landed, default ON** (`DG_PREFIX_BORROW=0` opts out); bit-identical `committed_sha256`, ~5.5% off the steady block | `verify_prefix_borrow.sh` |
| HF sliding-layer key retention on the 25 sliding layers | **landed, default ON** (`DG_DENOISE_SLIDING_WINDOW=0` opts out); unbound blocks bit-identical, enforcement proven live on device | [per_layer_prefix_spans.md](per_layer_prefix_spans.md), `verify_denoise_sliding_window.sh` |
| Bounded 1024-row sliding read | **landed unconditional; flag deleted 2026-07-29.** SDPA key rows/step 130560 → 53760 (2.43×) at `p_max` 4096 and 499200 → 115200 (4.33×) at 16384; **−18.9% s/block**, completions byte-identical 10/10. Grep a log for `bounded sliding read:` to confirm engagement | [per_layer_prefix_spans.md](per_layer_prefix_spans.md) |
| Canvas-tail workspace + `fill_cache` | **refuted** — bit-identical but ~1.4% *slower*: the bounded span already removed the concat it targets | [per_layer_prefix_spans.md](per_layer_prefix_spans.md) |

## Prefill authority

Current pure full-depth 64K-build artifact: `context_window_prefill_only_chunkedlong_20260713_msl65536.{json,md}` — 1K 0.78 s, 4K 1.37 s, 8K 4.15 s, 16K 5.55 s, 32K 10.84 s, 64K 35.58 s (synchronized first executions per shape, including shape-specific first-use compilation). The 64K-build sweep at `233b88276ab` measured **2950 tok/s at 16K, 3022 tok/s at 32K, 1842 tok/s at 64K**; the 64K decline is long-context attention chunking, not a return to dense MoE. **The similarly named artifact WITHOUT `chunkedlong` is the superseded dense-fallback control.** Root cause and bit-identity: [chunked_ragged_prefill.md](chunked_ragged_prefill.md).

## Reproduction

env: see [plan.md](../../plan.md).

```
demo/serving_smoke.py --upfront                  # 48-trace startup capture + replay, same fail-loud contract as vLLM
demo/serving_smoke.py --upfront --num-layers 2   # turns a 15-minute reproduction into ~2 minutes
```

Device-gate commands and their artifact paths: [work_log.md](work_log.md).

## Artifact index

- [work_log.md](work_log.md) topology audit + weight roofline + up-front device gates · [perf_progress.md](perf_progress.md) campaign ledger · [path_to_100tps.md](path_to_100tps.md) roadmap arithmetic only · [flag_triage_20260728.md](flag_triage_20260728.md) the only list of dead flag names.
- [early_halt.md](early_halt.md) · [commit_batching.md](commit_batching.md) · [l1_residency.md](l1_residency.md) · [per_layer_prefix_spans.md](per_layer_prefix_spans.md) · [chunked_ragged_prefill.md](chunked_ragged_prefill.md) · [winter_borrow_20260727.md](winter_borrow_20260727.md).
- Rooflines / op profiles: [nonmoe_roofline/README.md](nonmoe_roofline/README.md) · [whole_gen_opprofile/README.md](whole_gen_opprofile/README.md) · [opt004_matmul_geometry.md](opt004_matmul_geometry.md).
- Self-conditioning defaults: [selfcond_prechunk.md](selfcond_prechunk.md), `selfcond_prechunk_e2e.json`, `selfcond_logits_l1_e2e.json`.
- Precision / fidelity: [datatype_sweep](../datatype_sweep/README.md) · [decision fidelity](../decision_fidelity/README.md); serving: [vllm_integration](../vllm_integration/README.md).
- Raw logs `artifacts/*.log`; `perf_summary.json` is a dense-era snapshot.
