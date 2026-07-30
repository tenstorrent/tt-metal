# Self-conditioning: embedding prechunk and the logits L1 chain (dg-08)

Status: current — both levers are the shipped defaults. `DG_SELFCOND_PRECHUNK_EMBED=0` and
`DG_SELFCOND_LOGITS_L1=off` are the diagnostic opt-outs (the selector accepts exactly
`{off, chain}`); both live in `tt/self_conditioning.py`.
Owns: the selected self-conditioning defaults and their numbers, the 256K full-vocab-entropy-plus-
trace capacity limit, and the three rejected adjacent candidates absorbed from
`selfcond_logits_l1.md`, `selfcond_logits_split_rejection.md`, `selfcond_vocab_chunk_rejection.md`.
See also: [refuted list](../REFUTED.md), [optimize_perf hub](README.md).

Over the 100-line cap: four refutations, one open contradiction and three measurement traps.

## What shipped

**Embedding prechunk.** The 256K-vocabulary self-conditioning signal is evaluated as 32 ordered 8K
chunks. The old path stored one `[262144, 2816]` BF16 table and copied an 8K-row device slice before
every chunk matmul on every denoise step; the selected path builds the same table as **32 persistent
8K-row tensors** handed directly to the existing matmul. It changes no value, dtype, matmul shape,
chunk boundary or accumulation order — it removes **32 embedding-table `ttnn.slice` ops per step** —
and the chunks are built before trace capture, so their addresses are persistent. Payload is exactly
`262144 × 2816 × 2 = 1,476,395,008` bytes/chip (1408 MiB) before and after: storage goes from one
1408 MiB allocation to 32 × 44 MiB, so persistent bytes change by **zero** and allocation count by
**+31**.

**Logits L1 chain.** Each 8K logits slice, its immediate `subtract → exp`, the denominator reduction
and the ordered denominator accumulator stay in interleaved L1; the 32 chunk matmuls, the ordered
numerator accumulator and the final divide stay in DRAM. Operation order is unchanged; no dtype,
value, chunk boundary, matmul shape or diffusion decision changes.

## Selected numbers

Component gate (`prof_step_breakdown.py`, real checkpoint, 2 real MoE layers, 5 warmed iterations,
synchronized once per timed batch):

| probe | arm | full two-layer step | soft embedding |
|---|---|---:|---:|
| prechunk | monolithic table, DRAM | 82.334 ms | 25.863 ms |
| prechunk | prechunked, DRAM | 73.575 ms (**−10.6%**) | 18.210 ms (**−29.6%**) |
| L1 chain | DRAM control | 73.524 ms | 18.213 ms |
| L1 chain | only the dynamic slice in L1 | 72.891 ms | 17.721 ms |
| L1 chain | selected logits + denominator chain | **71.359 ms** | **16.038 ms** (**−11.94%**) |

The two probes are separate sessions; their DRAM controls agree to 0.07% (73.575 vs 73.524, 18.210
vs 18.213). Controls stayed flat across the prechunk A/B (hidden 48.223 → 47.576, LM head
4.361 → 4.362, terminal 28.906 → 28.899).

Traced full-model, the **final reviewed unset-default** rows — the selected headline, not the faster
explicit-candidate samples:

| lever | @48 steady block | @48 t/s | full generation | derived warmed step |
|---|---:|---:|---:|---:|
| prechunk selected default | 13.6817 s | 18.711 | 153.3410 s | 258.631 ms |
| + L1 chain (current) | **13.5849 s** | **18.844** | 153.9791 s | **257.575 ms** |

Prechunk alone: +3.03% @48 and +1.76% @12 (4.3710 s = 58.568 t/s); warmed traced step
268.033 → 258.631 ms (9.403 ms; 3.51% latency, 3.64% step rate). Its full serving session — prefill
1.1120 s, TTFT (prefill + block 0) 125.977 s, blocks 124.8640 / 13.7052 / 13.6582 s, full prefill +
3-block generation 153.3410 s — improved complete generation by 1.36%, with explicit and default
rows agreeing within 0.13% at steady state.

The L1 chain delta is **−0.71% block, +0.71% throughput, −0.41% step, and a +0.42% full-generation
REGRESSION**; no full-generation win is claimed. At @12 it was **not** a win: one explicit chain
sample regressed 4.2752 → 4.2981 s, an earlier unset-default process measured 4.2647 s, and the
required fresh paired final measured 4.3122 s / 59.366 t/s. Run spread, reported rather than
interpreted: three fresh independent controls at 13.6284 / 13.6161 / 13.6051 s; two explicit chain
processes at 13.4969 / 13.5253 s (medians 13.6161 → 13.5111 s, −0.77%; 18.801 → 18.9475 t/s,
+0.78%); a later 8K/default control at 13.6321 s; a first post-removal fresh process at 13.5120 s;
the final review-followup at 13.5849 s, deliberately taken as the conservative headline.

Superseded, one line: the selected prechunk @48 row replaces a preliminary 18.819 t/s row collected
before trace-region provenance was emitted; the 0.57% difference is normal run variance and did not
change selection.

> **OPEN CONTRADICTION (unexplained):** one same-model **sequential** A/B moved the WRONG way,
> 13.6456 → 13.7841 s in the second session, contradicting the independent-process result that
> selected the L1 chain. It is retained as a real limitation. The only counter-argument on record is
> that production constructs one fresh session, and that two independent candidate processes plus
> the final unset-default process all reproduced the @48 improvement. **Not explained.**

All arms retained the established committed-token digest `a9f0d18709b07d1e` (@48) /
`24393ba7aad6077c` (@12), so both levers cleared an output-identity gate and needed no #48291 or
HF-fidelity waiver.

## Rejected adjacent candidates

Four refuted candidates and their one-clause reasons are in [refuted list](../REFUTED.md): the
`ttnn.split` over the dynamic logits, the 32K vocab chunk, and the two explicit-L1-placement
variants. What must not be re-derived:

* **Component sweep behind the vocab-chunk rejection** (soft embedding / full two-layer step): 8192
  control 16.038 / 71.359 ms; 16384 15.554 / 71.810; 32768 15.322 / 70.994; 65536 15.223 / 72.447.
  The component optimum is **non-monotonic** — 65536 has the fastest soft embedding and the slowest
  full step. Larger chunks cut slice/exp/matmul/reduction/ordered-add launch count without changing
  dtype or persistent bytes, but they change each matmul/reduction grouping, which is the mechanism
  by which the commit digest moves.
* **POLICY RULE that must survive:** a full per-step acceptance campaign cannot make a final-commit
  mismatch eligible, so it is not spent. A digest change under identical prompt, seed, step count,
  precision policy and traced workload is **decisive on its own**.
* Artifacts: `selfcond_logits_split_rejection.json`, `selfcond_vocab_chunk_rejection.json`.

> **MEASUREMENT TRAP.** The `ttnn.split` candidate's LOWER full-generation total (151.4487 vs
> 152.1866 s) came entirely from variable FIRST-BLOCK TRACE CAPTURE (123.0523 vs 123.8355 s), not
> warmed execution. Never rank candidates on a full-generation total — see [hub](README.md).

> **MEASUREMENT TRAP.** A run contaminated by a concurrent external device user must be **discarded
> outright**, not reported with a caveat — that is what happened to the traced arm of the
> extended-L1 candidate.

> **HARNESS TRAP.** Both `denoise_hidden_forward` and `_apply_lm_head` CONSUME/DEALLOCATE their
> inputs, so a component harness must hand each timed invocation a clone while keeping the
> persistent benchmark source allocated. The harness was repaired before the prechunk measurement
> was accepted.

## Correctness and capability gates

The six-field exact diffusion-decision probe is owned by
[decision fidelity](../decision_fidelity/README.md). Its DG-specific caveat, which must not be lost:
**no KV commit occurs during intermediate denoise steps, so those rows are CANDIDATES**;
DiffusionGemma commits the LAST candidate, and the artifact asserts that the last per-step candidate
hash equals the trajectory's actual commit hash in both processes.

Results for these levers (`verify_selfcond_prechunk_decisions.py`): **RUN-first argmax** 48/48 steps
exact for all six fields including entropy means and accept counts — trajectory SHA
`b2e74f4edd6a2e3562b81b04b2f94bdb9881011b225f6f353cb8668449d2ab51`, final commit SHA
`e3b1344d8f795aa0c40a8d96c58e7d94bdb3c234ac9c67bf3d21faed687eafdc` in both runs. **Production
sampler** with 48 deterministic `ChunkedGumbelNoise` descriptors (seeds 2–49, chunk size 1024, FP32
noise, descriptor hash `fb8f0108c5516e18a00ae30ad67f81990e3e1886973497a13f3941e67e1d7aa3`) 48/48
exact — trajectory SHA `55260b7946ce281c85449030f5f177fc320725adcd7d6757ec7265f103a9c0cf`; the L1
chain reproduced the same six fields plus final commit exactly for all 48 steps. **Qualitative:**
three chat-template-rendered prompts at 16 fixed steps in fresh control and default processes,
argmax then chunked-Gumbel — every committed block SHA and complete decoded text matched exactly
within both A/Bs, while the recorded classifications (Chinese prefix then punctuation/digit
degeneration; coherent answer then repetition; correct answer then emoji/replacement-glyph
degeneration) are **pre-existing #48291 defects the artifacts explicitly do NOT dismiss**. Unit
gates: 41 proportional unit tests for the L1 chain; a watcher run matching zero error signatures.

> **TRAP from the qualitative gate:** none of the stored decoded outputs contains the literal word
> `user` — it appears only in the rendered chat-template prompts, so grepping decoded text for it is
> a false signal.

**256K capability, verified:** a full 30-layer `max_seq_len=262144` 256-canvas smoke with a
non-aligned 24-token logical prompt and the production chunked-Gumbel sampler passed the FULL 48-step
budget — post-build 29.704 GiB/chip, post-prefill-plus-one-committed-block 31.134 GiB/chip, 0.733
GiB/chip free, `DG_VLLM_SERVING_SMOKE_SUCCESS`, clean teardown.

**OPEN CAPACITY LIMIT.** The full-depth 256K allocation cannot be combined with Metal trace capture:
with zero reservation the model ops fit but `end_trace_capture` detects the trace buffer overlapping
the DRAM high-water mark, and reservations of 512, 256, 192 and 176 MiB avoid or approach that
overlap but leave **no contiguous 128 MiB buffer** for `token_entropy`'s `exp_shifted` /
`expected_terms`. That is an exact full-vocab-entropy-plus-trace capacity limit at max context,
recorded in `doc/context_contract.json`. The finding stands; the traced path it was measured against
has since been replaced by up-front capture.

Watcher smoke evidence: `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`, 2 real layers, traced
4-step one-block, non-aligned logical prompt length 24 → `DG_VLLM_SERVING_SMOKE_SUCCESS`, 256 tokens,
all four devices attached/detached, no error/assert/NoC violation; log SHA-256
`25a5a17fd87d7e84bf7f2ac2f12f9a2e70dbd427c520b578e8f947e23ae1a051`.

## Provenance discipline and reproduction

The e2e harness fails before mesh open unless `DG_TRACE_REGION_SIZE` is **exactly** 10737418240 (10
GiB) and the workload is the canonical P150x4 / TP=4 / 256-canvas / seed-0 / three-block @48-and-@12
configuration; `selfcond_prechunk_e2e.json` embeds checkpoint and tokenizer hashes, source HEAD,
dirty-worktree source hashes, build cache hash, mesh/TP, prompt, seed, allocation and
`ENABLE_TRACY=OFF` provenance. Every hardware command was preceded by a process-ownership check
(`ps -eo pid,ppid,pgid,tty,stat,etime,args` filtered for cursor-agent / pytest / serving_smoke /
bench / verify / tt-smi / diffusion_gemma processes).

> **INFRASTRUCTURE TRAP.** An exploratory run accidentally inherited `TT_METAL_WATCHER=10`, so its
> 114.8 s watcher-instrumented blocks had to be discarded; stopping that contaminated run left one
> ERISC heartbeat stale and the next mesh open failed before model code. Recovery was bounded
> four-device list → `tt-smi -r` → bounded four-device list → `(1,4)` open/close `MESH_SMOKE_OK`,
> and all accepted measurements were collected afterwards with watcher/profiler variables unset.

env: see [plan](../../plan.md). Run each A/B once with the selector at `0`, once at `1`.

```bash
# component
python -u models/experimental/diffusion_gemma/doc/optimize_perf/prof_step_breakdown.py \
  --num-layers 2 --iters 5
# six-field decisions (omit --gumbel-mode chunked for the RUN-first argmax artifact), then compare
python -u models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py \
  --steps 48 --gumbel-mode chunked --out decisions-control.json
python -u models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py \
  --compare decisions-control.json decisions-default.json --out decisions-comparison.json
# 256K capability
python -u models/experimental/diffusion_gemma/demo/serving_smoke.py --num-blocks 1 \
  --canvas-length 256 --max-denoising-steps 48 --max-seq-len 262144 --gumbel-mode chunked \
  --disable-eos-stop --local-files-only --metrics-json selfcond_prechunk_256k_chunked.json
# device-free gate
pytest -q models/experimental/diffusion_gemma/tests/test_tt_self_conditioning.py \
  models/experimental/diffusion_gemma/tests/test_denoise_forward.py
```

The traced-throughput rows above came from `bench_lever_e2e.py`, which no longer exists; those
numbers are provenance. Artifacts, all present in this directory:
`selfcond_prechunk_{e2e,decisions,gumbel_decisions,qualitative,gumbel_qualitative,256k_chunked,watcher_summary,summary}.json`
and `selfcond_logits_l1_{e2e,decisions,gumbel_decisions,gumbel_qualitative,256k_chunked,watcher,watcher_summary}.json`.

## Limits

Removes embedding-table slices only — the 32 dynamic logits slices and the online softmax/matmul
operations remain. The production vocabulary is exactly divisible by 8192; the builder and unit gate
retain a final short chunk for non-divisible vocabularies. Prompt length, canvas length and KV state
are untouched; no extra runtime copy or host fallback. The noise-regime prerequisite these docs
originally stated is **inverted** — see
[early halt](early_halt.md#noise-regime--the-claim-is-inverted).
