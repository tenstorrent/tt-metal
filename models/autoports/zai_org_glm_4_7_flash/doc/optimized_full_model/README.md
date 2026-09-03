# GLM-4.7-Flash optimized full model — stage report

Target: `zai-org/GLM-4.7-Flash` (`Glm4MoeLiteForCausalLM`, 30.6B total / ~3.6B
active params, 47 decoder layers = 1 dense + 46 MoE, top-4-of-64 routing +
1 shared expert, vocab 154880, advertised context 202752), **one Blackhole
p150-class chip**, device 0, 1x1 mesh, 11x10 compute grid, 8 DRAM banks.
Branch `ttmodelmanager/glm47-flash-probe`, from full-model commit
`70b509b361a`. **This stage's diff against `70b509b361a` is zero lines in
`tt/*.py`** — see "What was investigated" for why, and note this claim is
narrower than "nothing was found": one real, unshipped optimization
opportunity was found (item 5) and left as a documented candidate rather
than a shared-code change, for reasons given there.

This report was rewritten after an independent `$stage-review` round found
five P1 defects in the first draft: a mandatory DRAM-sharded LM-head trial
that was never run, a wrong conclusion from an uninstrumented microbenchmark
(item 5, corrected below), missing autoregressive/degenerate-output evidence,
a circular decode-closure computation, and this stage's own evidence runs
overwriting the *previous* stage's committed JSON artifacts in place. All
five are fixed in this version; see `work_log.md` OFM-010 for the full
review response.

## Headline (before = full-model FM-023 commit `70b509b361a`, after = this stage, same harness)

| | before (full-model) | after (this stage) | delta |
|---|---|---|---|
| TTFT, prompt 128, warmed prefill+first token | 334.2 ms | 334.0 ms | noise |
| TTFT, prompt 154, request-boundary inclusive (AIME ref) | 590.8 ms | 590.6 ms | noise |
| Traced decode, batch 1, model trace only (no sampling) | 21.758 ms/tok (45.96 t/s/u) | 21.760 ms/tok (45.96 t/s/u) | noise |
| Traced decode, batch 1, token-out (sampling + 1-word readback) | 22.994 ms/tok (43.49 t/s/u) | 23.013 ms/tok (43.45 t/s/u) | noise |
| End to end, prompt 128 / generate 128 | 3.241 s (39.49 tok/s) | 3.241 s (39.49 tok/s) | unchanged |
| prefill top-1 / top-5 / top-100 (AIME24, 100 positions) | 0.880 / 1.000 / 1.000 | 0.880 / 1.000 / 1.000 | unchanged |
| teacher-forced top-1 / top-5 / top-100 | 0.850 / 1.000 / 1.000 | 0.850 / 1.000 / 1.000 | unchanged |
| autoregressive (256 tok, chat prompt) | adjacent_duplication 0.0, trigram_loop 0.0246 | adjacent_duplication 0.0, trigram_loop 0.0246 | unchanged |
| batch 32 @ 8192 tok/user | 10 passed | 10 passed | unchanged |
| watcher (3 runs) | 0 faults | 0 faults | unchanged |

Bar is top-5 >= 0.98, top-100 = 1.00. Both clear it, matching full-model
exactly (this stage changed no source). `perf_summary.json` has the
skill-mandated roofline/device/e2e reconciliation.

## Decoder-layer-stack lower bound (Full-Model Decode Closure)

The first draft of this section computed `gap = token_out_measured -
(layer_stack + terminal)` where `terminal` was itself defined as
`token_out_measured - model_only_measured` — an identity that always
evaluates to `model_only - layer_stack` no matter what the real numbers are,
so it could not have detected a closure gap in either direction. Corrected
version, with `terminal` measured independently (device time for the
sampling ops, not derived by subtraction) via the reduced 2-layer profile's
signposted windows:

```
layer_stack_ms       = 46 * 0.491 (moe, optimized_decoder ctx-1024 traced wall)
                      + 1 * 0.447 (dense)                       = 23.033 ms/token
measured_model_only   (traced replay, real 47-layer model, wall = device time;
                        see perf_summary.json's device_vs_e2e_note)          = 21.760 ms/token
terminal_ms           (sampling device time from the reduced-profile window,
                        perf_report_summary.json decode_tokenout minus
                        decode_model device_us_per_step: 2752.7 - 1708.8)   = 1.044 ms
                       (+ one host token readback, measured separately)      = 0.133 ms
token_out_estimate    = measured_model_only + terminal_ms + readback         = 22.937 ms
token_out_measured    (this stage, fresh run)                                = 23.013 ms
gap                   = token_out_measured - token_out_estimate              = +0.076 ms (+0.33%)
```

Independently: `measured_model_only (21.760) - layer_stack_ms (23.033) =
-1.273 ms`, i.e. the real 47-layer traced model is *faster* than the naive
sum of isolated single-layer wall-clock probes — isolated single-layer tests
carry more per-call trace-boundary overhead proportionally than the same ops
embedded in the real 47-layer trace, so `layer_stack_ms` is a conservative
(pessimistic) upper bound, not a floor the full model must reach. Both of
these are legitimate, independent observations now (not the same number
printed twice): the model-vs-naive-sum comparison shows no isolated-to-full
regression, and the terminal-work reconciliation (+0.33%) shows the terminal
path costs almost exactly what the reduced-profile device-time breakdown
predicts. **Neither computation shows a >10-15% gap; nothing to split.**
`perf_summary.json` has the same reconciliation in the skill-mandated shape,
plus a first-order DRAM roofline estimate (~3.9 ms/token, ~17.9% of e2e) —
see that file's `named_limitations` for why it is a lower-confidence,
first-order model rather than a `tt-perf-report`-derived figure.

## What was investigated (real hardware, real weights/shapes, this stage)

### 1. LM head — the largest single op (51.1% of model-only device time) — DRAM-sharded matmul trial (mandatory per `$optimize`: "If tt-perf-report says a matmul is DRAM-bound and it is not DRAM-sharded, trying DRAM-sharded matmul is mandatory")

The shipped LM head (`tt/model.py:204-245`, `_lm_head_1d_pc`) uses a wide-1D
mcast config, interleaved DRAM weight, `HiFi2` fidelity, swept over core
count and `in0_block_w` in `doc/full_model/head_probe.json` — but **never
against a `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` candidate**,
which is what `tt-perf-report` recommends for this exact row (`Bound=DRAM`,
70.7% DRAM utilization, `Advice: "Try a DRAM-sharded program config ..."`).
This was a real gap in the first draft, not addressed by the `SLOW`-bound
audit below (the LM head's `Bound` is `DRAM`, a different classification).

Built and traced a DRAM-sharded candidate
(`probe_scripts/lm_head_dram_test.py`): weight width-sharded across the 8
DRAM banks (`154880 / (32*8) = 605` tiles/bank exactly, no padding needed),
activation width-sharded on the matching 8-core raster (mirroring the
`wqkv_a`/`wo` pattern in `tt/optimized_decoder.py`), output width-sharded to
match (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` requires a
sharded output). Swept every legal `in0_block_w` (divisors of the activation
shard's 8-tile K-width: 1, 2, 4, 8):

| candidate | result |
|---|---|
| shipped (1D-mcast, 110 cores, `in0_block_w=4`, HiFi2) | 883.5-883.8 us/call |
| DRAM-sharded, `in0_block_w` in {1, 2, 4, 8} | **all four fail**: `TT_THROW @ program.cpp:1875: Statically allocated circular buffers on core range [0-0 - 7-9] grow to 17,187,328 B which is beyond max L1 size of 1,572,864 B` |

The failure is independent of `in0_block_w` (all four legal values hit the
identical CB-size wall), which localizes the blocker to the fixed
`per_core_N = 605` tiles/core the op's 8-core (bank-count-fixed) grid forces
at this vocab width — DRAM-sharded matmul's core count equals the DRAM bank
count and cannot be increased the way a 1D-mcast grid can, so the usual
"increase core count" escape (`$optimize`'s L1-OOM advice) is not available
for this op class at this N. **This is an exact, quantified (17.2 MB needed
vs 1.5 MB available, 10.9x over) op-contract blocker, not an untried
candidate** — the mandatory trial was run and failed for a specific,
recorded reason.

Also tested the fidelity axis on the shipped (legal) 1D-mcast topology,
since `tt-perf-report` separately flags `HiFi2` with "HiFi2 discards the
lowest bit of activations" as an accuracy note and `$optimize` requires
LoFi/HiFi2 to be compared per dominant projection:

| fidelity (shipped 1D-mcast topology) | us/call |
|---|---|
| HiFi2 (shipped) | 883.5-883.8 |
| LoFi | 878.5 |

0.6% device-time difference (~5 us), i.e. **~0.02% of the 23.0 ms token-out
step** — below this harness's own run-to-run spread on every other
measurement in this stage. Not adopted: the LM head is the final projection
before sampling, `doc/full_model/README.md`'s FM-021 finding already
established that LM-head program-config changes are measurable *accuracy*
changes on real weights (0.05 top-1 drop from an `in0_block_w` change with a
similar sub-1% speed motivation), so a real-weight accuracy re-check would be
needed before shipping a LoFi LM head — not worth running for a ~0.02% e2e
gain. Commands: `probe_scripts/lm_head_dram_test.py`, log
`logs/lm_head_dram_sweep.log`.

### 2. SLOW-bound (as opposed to DRAM-bound) matmul rows — confirmed already at the swept-optimal geometry

Separately from the LM head (which is `Bound=DRAM`), `tt-perf-report` flags
7 other matmul roles `Bound=SLOW` (neither compute- nor DRAM-bandwidth-bound):
wqkv_a, wq_b, w_uk, w_uv, dense gate/up, router, shared-expert down. Pulled
`Inner Dim Block Size` / `Math Fidelity` / dtype from the fresh Tracy CSV for
every one and cross-referenced the exact `tt/optimized_decoder.py`
constructor call that produces that shape — all seven match the
already-shipped `doc/optimized_decoder/README.md` policy exactly (see
`work_log.md` OFM-002 for the row-by-row table). `tt-perf-report`'s advice on
these seven is `in0_block_w=N looks good 🤷` (no change) plus the same
generic HiFi2/HiFi4-for-accuracy note addressed for the router below.
**OPT-013 ("policy reaches the measured ops") holds for the full-model
measured path**, not only the standalone decoder tests.

Router (`32x2048x64`, `HiFi4 BF16 x FP32`) carries its own fidelity advice
("HiFi2 may also work ... 2x the throughput of HiFi4"). Tested directly
(`probe_scripts/router_fidelity_test.py`): HiFi4 12.06 us/call vs HiFi2 11.14
us/call (7.6% device-time difference on the isolated op, **~0.19% of the
47-layer model-only step** after scaling by 46 MoE-layer occurrences). Not
adopted: the router's `fp32`/`HiFi4` policy is explicitly documented
elsewhere in this codebase (`optimized_decoder.py:152-154`,
`doc/optimized_decoder/README.md`'s precision-policy table) as required for
MoE routing *selection semantics* — near-tied expert scores decide which
expert receives a token, and this repo already tracks router sub-ulp ties as
a correctness-sensitive area (`dev_optimize.py --check-ties`). Verifying
HiFi2 doesn't flip any real routing decision needs a real-weight tie-position
check, which was not run for a ~0.19% gain that directly targets a
documented correctness-sensitive tensor group.

### 3. `attn_lat` (flash-decode output feeding `w_uv`) placed in DRAM, not L1 (`tt-perf-report` advice: "place input 0 in L1")

Tested with `probe_scripts/attn_lat_l1_test.py` (patches
`paged_flash_multi_latent_attention_decode`'s `memory_config=` on an isolated
`OptimizedDecoder` instance, moe layer, ctx 1024, batch 1, 32-iteration timed
trace replay; logs `logs/attn_lat_dram_b1.log` / `attn_lat_l1_b1.log`):

| arm | decode ms/token |
|---|---|
| DRAM (shipped) | 0.4928 |
| L1 | 0.4916 |

0.24% difference — inside this harness's spread (`dev_optimize.py`'s
independent baseline for the same config: 0.4921). The isolated op is 11.78
us out of ~445 us/layer average. Not adopted: no measured win, and the
neighboring `kvpe` DRAM placement is explicitly commented as an L1-headroom
decision at batch 32 (`optimized_decoder.py:521`); shipping an L1 candidate
for `attn_lat` would need the same batch-32 CB-budget verification for a
result that doesn't clear noise at batch 1.

### 4. Runtime fallback audit and the disclosed `ttnn.split` capture-time downgrade

`tt/model.py:427-441` places sampler-ready decode logits in L1 and documents
that `TTSampling`'s internal `ttnn.split` logs `L1 budget exceeded ... DRAM
downgrade` and migrates to DRAM before slicing — **once per fresh sampling-
trace capture, never during replay** — and explicitly asks stage 07 to
re-run the L1-vs-DRAM A/B if this stage adds resident L1 pressure. It does
not (item 3's candidate was rejected). Re-ran `probe/logits_memory_ab_probe.py`
fresh (`logits_memory_ab.json`, `logs/logits_memory_ab.log`): L1 arm 2.916 ms
token-out vs DRAM arm 2.940 ms on the reduced 2-layer probe, tokens
bit-identical between arms — L1 remains faster including the capture-time
fallback cost, matching the full-model conclusion within noise.

Grepped every log this stage produced for
`fallback|downgrade|L1 budget exceeded|unsafe`. Two distinct signatures
appear (the first draft incorrectly said there was only one, and incorrectly
called the message "byte-identical" when the `have N B` figure varies with
L1 fragmentation across runs — corrected here):

1. The disclosed `ttnn.split` L1->DRAM downgrade above — always at
   sampling-trace-capture time, never during replay.
2. `Allocating device buffers is potentially unsafe due to the existence of
   an active trace ... Use the trace allocation tracker to verify.` — a
   generic warning `ttnn` prints whenever a buffer is allocated while any
   trace is live, regardless of whether that specific allocation is actually
   unsafe. This is exactly what `TT_METAL_TRACE_ALLOC_TRACKING=1` exists to
   adjudicate. Re-ran `probe/trace_alloc_probe.py` fresh with that env var
   set (`trace_alloc.json`, `logs/trace_alloc_probe.log`): `tracking_enabled:
   true`, `verdict: clean`, and **both shipped-path arms
   (`shipped_path_warmed_single_chunk`, `shipped_path_first_use_multi_chunk`)
   report `unsafe_total: 0`**; only the deliberately-adversarial
   `hook_bypassed_first_use_multi_chunk` control (which disables the safety
   hook on purpose, to prove the tracker can detect the hazard at all) shows
   `unsafe_total: 32`. **Resolution: controlled** — the warning text alone
   does not distinguish "corrupted" from "allocated while a trace exists but
   proven safe," and the tracker's own instrumented verdict is the thing
   that resolves it, which this stage re-verified rather than assumed.

`test_no_host_fallback_during_traced_decode` (in the 47-test full-model
suite) passed. **Runtime fallback audit is clean for the measured (replay)
path**, with both fallback signatures now classified rather than one of them
going unmentioned.

### 5. Sampler TopK vocab chunking — corrected finding: a real, quantified, un-shipped candidate exists in shared code

**This item's first-draft conclusion was wrong and is corrected here.** The
first-draft microbenchmark (`probe_scripts/topk_width_test.py`,
`topk_width_test2.py`) called `ttnn.topk` at each candidate chunk width with
**no warmup calls before the timed loop**, so the first of the 32 timed
iterations at a never-before-compiled shape paid full JIT/program-cache
compile cost, which dominated the average for the wider (novel) shapes. That
produced the false result "chunks=2/1 are 19-26x slower," which the first
draft reported as "confirmed already optimal, not a stale conservative
limit." It is not correct.

Re-measured properly (`probe_scripts/topk_width_clean.py`: 8 uncounted
warmup calls per shape, then 32 timed iterations in a fresh device per
candidate to avoid any cross-run cache contamination, plus a correctness
check against `torch.topk` on the same input — all three candidates return
values matching `torch.topk`'s top-32, within `atol=2e-2`):

| chunks | width/chunk | ms/call (warmed, correct) |
|---|---|---|
| 4 (shipped) | 38720 | 0.8265 |
| 2 | 77440 | 0.7676 |
| **1** | **154880** | **0.7333** |

Once properly warmed, **fewer/wider chunks are faster, not slower** — a
single call over the full padded vocab is ~11.3% faster than the shipped
4-way split, and produces correct top-32 values. This is consistent with the
routing-predicate analysis in `models/common/sampling/_utils.py`
(`topk_would_route_to_large_indices`): at `k=32<=64` (the small-k arm),
`padded_width=154880` is not a power of two, so it is `structurally
_ineligible` regardless of width, which routes it to the Blackhole
`topk_large_indices` composite (legal up to `2^19=524288`) rather than the
slow single-core linear fallback my corrupted first measurement implicitly
assumed it fell into. `models/common/sampling/tt_sampling.py`'s module-level
`TOPK_MAX_WIDTH = 64*1024` (which forces the 4-way split via
`num_single_device_vocab_splits`) predates this composite and is, at this
model's vocab width on this chip, measurably more conservative than the op
actually requires.

**Not adopted this stage.** `TOPK_MAX_WIDTH` and `num_single_device_vocab_
splits` live in `models/common/sampling/tt_sampling.py`, shared across every
model that uses this sampling module, across architectures (the Blackhole
routing predicate is architecture-gated; Wormhole and others take a
different path this stage has not characterized) and across vocab sizes
(`tt_sampling.py`'s own comment cites GPT-OSS's ~200k and Gemma-2's 256000
vocabs sharing this constant). Changing it safely needs verification this
single-model, single-chip stage cannot responsibly claim to have done, and a
narrower per-model override (subclassing `TTSampling`/overriding
`num_single_device_vocab_splits`) would need its own accuracy re-verification
and a re-capture of the sampling trace, which was not done given the time
budget for a review-response pass whose primary job was fixing the P1
correctness findings below.

Note also (found while investigating this, and worth citing precisely rather
than re-testing): `tt_sampling.py:919-922` already documents a *different*,
previously-run upstream A/B (PR #53167) that tested padding vocab chunks to
a power of two specifically to steer `ttnn.topk` toward the stock multi-core
factory instead of the `topk_large_indices` composite, and found "no
end-to-end decode benefit." That is not the same question as this item
(chunk *count* at a fixed non-power-of-two width, staying on the
`topk_large_indices` composite throughout) — this stage's finding is new
information, not something PR #53167 already answered — but it is worth
recording as the existing evidence base this candidate builds on.

**Materiality:** the entire sampling step (all TopK/route ops plus
penalties/gather glue) costs ~1.12 ms of the real 47-layer model's 23.013 ms
token-out step (4.9%); this candidate's ~11% isolated-op improvement would
be roughly a 90 us / ~0.4% reduction in the full token-out step if shipped
safely. It does not dominate token-out decode (the LM head and 47 decoder
layers are the other 95.1%), so the goal's "if a sampler op dominates
token-out decode, fix it" gate does not apply, and this is recorded as a
quantified, reproducible, *not-yet-shipped* common-infra improvement
candidate rather than either "already optimal" (wrong, per above) or a
blocking gap.

### 6. Per-op device-time comparison across the isolated decoder and full-model captures (raised by review, investigated and refuted)

The review compared naive per-op-name *averages* between
`doc/optimized_decoder/tracy/moe/decode_opt_perf_report.csv` (isolated
decoder) and this stage's `decode_model_perf_report.csv` (full model) and
found `UntilizeCodegenDeviceOperation` apparently regressing 6.21 -> 15.29
us/call and `ReshapeViewDeviceOperation` 2.86 -> 8.38 us/call, raising the
possibility that embedding the decoder in the full model costs materially
more per op than the isolated decoder test suggested. Investigated directly
by histogramming the *raw per-call* device times (not averages) for both ops
in both CSVs:

```
isolated UntilizeCodegenDeviceOperation: two clusters, ~4.6 us and ~15.3 us (alternating)
full     UntilizeCodegenDeviceOperation: one cluster, ~15.3 us
isolated ReshapeViewDeviceOperation: three clusters, ~0 us, ~6.4 us, ~10.4 us
full     ReshapeViewDeviceOperation: two clusters, ~6.4 us, ~10.4 us
```

The full model's per-call values are **identical, to the microsecond, to a
subset of the isolated decoder's own per-call values** — the isolated
decoder simply has additional, cheaper call sites (a ~4.6 us untilize, a
~0 us reshape) that the full model's real embedding-to-layer-to-LM-head
plumbing does not exercise in the same proportion, most likely because the
standalone `OptimizedDecoder` test harness feeds synthetic activations
through an extra shape-adaptation step the real generator's embedding output
does not need. Averaging across these different call-site populations (as a
naive `total_time / total_calls` does) mixes them and produces an apparent
regression that isn't in the per-call data at all. **Resolution: refuted,
no regression.** This also means `doc/optimized_decoder/perf_summary.json`'s
`decode_ms_per_token_device: 0.3337` (derived from the isolated capture) is
plausibly an undercount from dropped device-profiler markers in that
specific capture rather than the full model being slower — a pre-existing
measurement-quality note about the *prior* stage's artifact, left as-is
there (not retroactively edited) but recorded here since this stage is what
surfaced it. It does not change any number in this stage's own report,
which uses wall-clock (not device-time-only) figures throughout.

## Correctness evidence (fresh, this stage, `zai-org/GLM-4.7-Flash` AIME24 chat-template reference)

| check | result | log |
|---|---|---|
| `run_prefill_check` | top1=0.880 top5=**1.000** top100=**1.000** | `logs/run_prefill_check.log` |
| `run_teacher_forcing` | top1=0.850 top5=**1.000** top100=**1.000**, TTFT 590.6ms, decode 43.98 t/s/u | `logs/run_teacher_forcing.log` |
| `run_autoregressive` (256 tok, chat prompt) + `check_degenerate_output.py` | adjacent_duplication 0.0, trigram_loop_fraction 0.0246, hf/tt token agreement 34/256 (informational, not gated), **no degenerate output detected** | `logs/run_autoregressive.log`, `logs/check_degenerate_output.log`, `readiness_autoregressive/`, `degenerate_check.json` |
| `test_full_model.py` (full suite) | 47 passed | `logs/pytest_full_model_only.log` |
| `test_full_model_batch.py` (batch 32 @ 8192 tok/user) | 10 passed | `logs/pytest_full_model_batch32.log` |
| `test_full_model_perf.py` | perf.json refreshed | `logs/pytest_full_model_perf.log` |
| context contract (`check_context_contract.py`) | OK, supported=202752 (full HF context, no reduction) | `logs/check_context_contract.log` |
| watcher (`TT_METAL_WATCHER=2`, matching `run_evidence_sweep.sh`'s own convention — stricter/shorter poll interval than the skill's `=10` example, not a correctness gap), 3 runs: reduced smoke / reduced trace / full capacity | 0 / 0 / 0 faults | `logs/watcher_*.log` |
| trace-allocation tracker (`TT_METAL_TRACE_ALLOC_TRACKING=1`) | `verdict: clean`, shipped-path arms `unsafe_total: 0` | `trace_alloc.json`, `logs/trace_alloc_probe.log` |
| `$qualitative-check` shared suite (6 chat-template prompts, greedy, HF control reused via `--skip-hf`, TT side regenerated fresh this stage) | coherent, correctly-formatted reasoning-trace completions on both HF and TT; greedy prefix agreement vs HF 8/128, 16/128, 45/128, 14/128, 32/128, 15/128 tokens (content-level divergence from free-running greedy compounding bf16/bf4 rounding, not a defect — bounded by teacher-forced top-5=1.000 over 100 positions); no repetition/gibberish/wrong-language/truncation in any of the six | `qualitative/` (refreshed, current source hash `8e51b6af129a6dad`), `logs/run_qualitative_suite.log` |

Bar is top-5 >= 0.98 and top-100 = 1.00. Both clear it, matching the
full-model numbers exactly (this stage changed no source).

## Capability contract

Unchanged from `doc/full_model/README.md` and `doc/context_contract.json`:
**supported context 202752 = full HF-advertised context, no reduction**;
paged latent KV cache **612 B/token/layer = 28,764 B/token across 47
layers** (the first draft of this report mis-stated this as "28,764
B/token/layer"; corrected here; `capacity.json`'s
`kv_cache_bytes_per_token_per_layer: 612` is the source of truth); batch 1 is
the primary latency target with batch 32 @ 8192 tok/user tested (10 passed,
this stage); non-aligned prompt lengths (1, 17, 63, 65, 129, 154, 1057, 2049,
2600, ...) remain supported through the public generator, unchanged code
path. `check_context_contract.py` confirms OK for 202752 (`logs/`).

## Preserved contracts (unchanged, verified against measured full-model ops, not re-swept)

Per-tensor-group dtype/fidelity policy, KV-cache dtype (bf8 deployment / bf16
supported), residual/norm sharded layout, DRAM-sharded decode matmul
geometries, sparse routed-expert path (`ttnn.sparse_matmul`, indexed batch-1
/ union batch>1), and the SDPA/flash-decode config are all exactly
`doc/optimized_decoder/README.md`'s shipped policy — item 2 above
independently re-verified (via the fresh Tracy CSV, not by re-reading the
old doc) that every dominant matmul in the *full model's* measured decode
path carries that policy's dtype and `in0_block_w`, closing the "policy
reached the measured ops" (OPT-013) requirement for the full-model context
specifically. `$datatype-sweep` (next stage) owns the broad Pareto frontier;
no broad sweep was run here (`head_probe.json`'s bf4-LM-head candidate at 624
us vs bf8's 878 us is a real, larger, correctly-deferred candidate that
belongs to that stage, not this one).

The generator's split-sampling contract (`models/common/sampling`,
semantically-greedy top-k=32 gather then `k=1,p=0,temp=1`, `tt_out_tok`
device-side feedback, device-derived RoPE index, changed-only page-table
refresh, nonblocking `ttnn.execute_trace(..., blocking=False)` decode replay,
zero eager decode/sampling steps in the measured run, on-demand sampling-mode
recapture off-trace) is exactly the full-model FM-023 contract — verified via
the fresh `perf.json`'s `host_work_counters_over_the_measured_generate`
(`eager_decode_steps: 0`), not re-implemented.

## tt-perf-report evidence (reduced 2-layer profile: HF layers 0 dense + 1 moe, real embedding/norm/LM-head/sampler, real paged-cache shapes)

Fresh capture this stage, same methodology as `doc/full_model` (full-stack
profiling is intentionally avoided per `$optimize`; ~3200 device ops/step at
47 layers would overflow Tracy's buffers). Commands:

```
python -m tracy -r -p -v -m pytest tests/test_full_model_profile.py -q -s
tt-perf-report <ops_csv> --arch p150 --start-signpost PERF_FM_DECODE_MODEL --end-signpost PERF_FM_DECODE_MODEL_END ...
tt-perf-report <ops_csv> --arch p150 --start-signpost PERF_FM_DECODE_TOKENOUT --end-signpost PERF_FM_DECODE_TOKENOUT_END ...
tt-perf-report <ops_csv> --arch p150 --start-signpost PERF_FM_PREFILL --end-signpost PERF_FM_PREFILL_END ...
python tests/summarize_perf_report.py --tracy-dir doc/optimized_full_model/tracy --out doc/optimized_full_model/perf_report_summary.json
```

| window | device us/step | top cost | DRAM roofline (modeled, tool-computed) |
|---|---|---|---|
| decode, model only | 1708.8 | LM head `32x2048x154880` 51.1% | 47.0% (240 GB/s achieved -> ~511 GB/s implied peak) |
| decode, token-out | 2752.7 | LM head 31.7% + TopK/route ops 30.2% combined | 29.1% (149 GB/s) |
| prefill (128 tok) | 10435.8 | routed sparse gate/up 29.4%, routed sparse down 18.9% | (sparse rows unscored without `--active-experts K`, see Known Limitations) |

Full tables: `tracy/{decode_model,decode_tokenout,prefill}_perf_report.txt`
(advice-enabled) and matching `.csv`/stacked `.csv`/`.png`.
`perf_report_summary.json` has the full per-window op breakdown.
`perf_summary.json` has the skill-mandated roofline/device/e2e shape.

Operation-topology audit (measured path, not isolated ops): the terminal
path per decode step is `[47 decoder layers] -> final RMSNorm (sharded,
carries the residual layout from the decoder, ~5.8 us) -> LM head (interleaved-
DRAM-weight 1D-mcast matmul, bf8 weight, HiFi2, `32x2048x154880`, ~883 us,
the single largest op in the model at ~4.1% of the real 47-layer token-out
step; DRAM-sharded alternative tried and blocked, item 1) -> split-sampling
(local top-k=32 over 4 vocab chunks, gather, greedy select, ~1.12 ms; a
faster 1-chunk alternative exists but was not shipped, item 5) -> one
`tt_out_tok` device buffer -> one host readback (~0.13 ms)`. No full-vocab
all-gather (single chip, nothing to gather across), no
`ArgMaxDeviceOperation`, no host argmax in the measured path
(`host_argmax_calls` stays 0 across the traced run). Embedding lookup is a
one-time-per-prefill cost, not part of the decode critical path measured
here.

## Known limitations (freshness / provenance / style / deferred candidates; no open correctness bug)

- **Item 5's TopK chunk-count candidate (~0.4% of token-out decode) is a
  real, quantified, reproducible improvement that was not shipped**, because
  it lives in `models/common/sampling/tt_sampling.py` (shared across models
  and architectures) and this single-model stage cannot responsibly verify
  it is safe everywhere that constant is used. `probe_scripts/topk_width_
  clean.py` is the reproducible repro for whoever picks this up.
- `doc/optimized_decoder/perf_summary.json`'s `decode_ms_per_token_device:
  0.3337` (from the *prior* stage) is plausibly an undercount from dropped
  device-profiler markers in that capture (item 6); not retroactively
  corrected there, since this stage does not own that artifact, but this
  stage's own numbers do not rely on it (they use wall-clock).
- `perf_summary.json`'s `roofline_ms_per_token_estimate` (~3.9 ms) is a
  first-order params x dtype-size model, not a `tt-perf-report`-derived
  figure; see that file's `named_limitations` for the gap between it and
  `tt-perf-report`'s own per-window modeled-roofline percentages (which
  cannot be safely scaled from a 2-layer capture to 47 layers — attempted
  once during this review-response pass via naive per-op-name call-count
  classification and abandoned as unreliable, the same failure mode the
  review caught in its own per-op comparison, item 6).
- `SparseMatmulDeviceOperation` rows in every window are DRAM/FLOP-unscored
  because `--active-experts K` was not passed to `tt-perf-report`; the
  routed-expert rows are excluded from all roofline percentages as a result
  (the tool's own warning, reproduced in every console log).
- Several `doc/optimized_full_model/` artifacts (`perf.json`, `capacity.json`,
  `logits_memory_ab.json`, `trace_alloc.json`, `qualitative/`) are copies
  produced by re-running full-model's own test modules, whose `DOC_DIR`/
  default output paths are hard-coded to `doc/full_model/`. This stage
  copies each fresh result into `doc/optimized_full_model/` immediately and
  then `git checkout`s `doc/full_model/` before moving to the next step, so
  the prior stage's committed artifacts are never left modified — verified
  before every commit in this stage (`git status --porcelain -- doc/
  full_model/` is empty at commit time). The autoregressive and degenerate-
  output checks were pointed at `doc/optimized_full_model/readiness_
  autoregressive/` and `doc/optimized_full_model/degenerate_check.json` via
  their own `--output-dir`/`--json`/positional-path arguments instead, so
  they never touched `doc/full_model/` or the model-root `readiness_
  autoregressive/` at all.
- This stage made zero source changes (see header), so there is no
  tuning-driven "before/after" delta in the headline table — it is a
  same-harness reproduction proving nothing regressed, plus the investigation
  record above.

## Artifacts

- `perf.json`, `capacity.json`, `logits_memory_ab.json`, `trace_alloc.json`,
  `perf_report_summary.json`, `perf_summary.json` — fresh this stage.
- `tracy/` — fresh Tracy capture + `tt-perf-report` tables/CSVs for
  decode-model, decode-token-out, and prefill windows.
- `qualitative/`, `readiness_autoregressive/`, `degenerate_check.json` —
  fresh this stage, current source hash.
- `probe_scripts/` — every microbenchmark used above, including the
  corrected TopK script (`topk_width_clean.py`) and the two flawed ones kept
  for record (`topk_width_test.py`, `topk_width_test2.py`), the LM-head
  DRAM-sharded trial, the router-fidelity trial, and the attn_lat L1 trial.
- `logs/` — every command's stdout/stderr from this stage's evidence run.
- `work_log.md` — the investigation narrative (OFM-001 through OFM-011),
  including the review-response record.
