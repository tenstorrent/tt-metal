# Post-fix eval rerun: the mandatory rows, measured

Run 2026-08-17 on `qb2-120-p04t03` (4×P300C, 1×4 mesh) after the sliding-cache
read-wrap fix (`AUTOFIX_SLIDING_CACHE_READ_WRAP.md`). Every number below was
measured on this machine; the recorded baseline column is from
`tti_eval_gpqa_cot.json` / `tti_eval_ifeval.json`.

## Environment, and why it is comparable

- `tenstorrent/vllm` at **upstream `dev` @ 7c99bd3b8, unmodified**, plus the TT
  plugin, with `EXTRA_MODELS_DIR` pointing at the committed
  `experiments/extra_models_dir` bundle. Verified empirically that
  `TTGemma4ForConditionalGeneration` resolves to
  `models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm:Gemma4ForCausalLM`
  and not to `models/demos/gemma4` — the checkpoint's unprefixed arch does point
  at the demo model, but `check_and_update_config` prefixes the arch before
  resolution, so the bundle wins.
- `tt-inference-server` at `c8509ac2` with all four committed patches applied by
  `git am` (clean), including the `EXPERIMENTAL` eval-enforcement fix.
- lm-eval from the TT fork `tstescoTT/lm-evaluation-harness@evals-common`,
  reporting `0.4.10.dev0` — the version the baseline recorded — in a separate venv
  so it cannot perturb tt-metal's.
- Server launched with the exact command in `RUN_NOTES.md` (`--mesh-device P300x2
  --max-num-seqs 32 --max-model-len 262144 --sampling-profile full`,
  `trace_region_size 220000000`, `FABRIC_1D_RING`).
- Only `pydantic` moved in tt-metal's env (2.9.2 → 2.13.4, required by vLLM
  itself). `torch` 2.11.0+cpu and `transformers` 5.10.2 unchanged; `ttnn` imports.

The environment is validated by reproduction, not assertion: with the serving
execution width restored (below), `meta_ifeval` reproduces the recorded row to
four decimals — 0.7857 / 0.8372 / 0.8214 / 0.8605.

## A second defect, found by this rerun: serving execution width

The wrap fix alone did **not** restore the recorded IFEval row. Measured, same
server command, same eval invocation, `num_concurrent=32`:

| decoder / serving state | prompt strict | inst strict | verdict |
|---|---:|---:|---|
| pre-fix decoders (551c3fdbc6d) | 0.0714 | 0.0465 | collapse |
| wrap fix only | 0.0357 | 0.0233 | collapse |
| wrap fix + padded lane width | **0.7857** | **0.8372** | reproduces baseline exactly |

So the collapse is **not** caused by the wrap fix — the pre-fix decoders collapse
the same way. It is caused by `eb459b3bf9e`, a commit titled "Document Gemma 4
Stage 11 GPQA blocker" that also changed `tt/generator_vllm.py`:

```python
-        execution_batch = 1 if logical_batch == 1 else self.max_batch_size
+        execution_batch = logical_batch
```

`RUN_NOTES.md` records the server checkout for the passing IFEval row as
`4b17e185dea9` — before that commit. `AUTOFIX.md` describes the change as a win
("improved the first-five finite probe from 2/5 to 4/5 and reduced wall time from
384 s to 86 s"), but it was measured on a five-document probe at a time when every
long generation was still corrupted by the read wrap, so the regression it
introduced was invisible.

What active-row execution actually does, measured here with identical prompts so
any difference is a serving defect rather than model behaviour:

| concurrency | clean rows | corrupted rows |
|---:|---|---|
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 4 | 4 | 0 |
| 8 | 8 | 0 |
| 9 | 0–7 | **row 8** — garbage from its first token, runs to the cap |
| 10 | 0–7 | **rows 8–9** |

Eight 158-token prompts fill one 2048-token prefill batch
(`max_num_batched_tokens=2048`, chunked prefill on), so the 9th request's prefill
necessarily runs while rows 0–7 are already decoding — and that row is the one
that comes out corrupted. With 18-token prompts the boundary moves (clean at 10,
corrupted at 12), which rules out a fixed row count and points at the same
trigger: a prefill executed against a decode batch already in flight. That is
hypothesis 2 of `AUTODEBUG_GPQA_DIVERGENCE.md` ("active-row lifecycle"), which was
never tested.

A second, milder effect is visible in the same sweep and is expected rather than a
defect: identical inputs produce 785 / 675 / 688 / 710 generated tokens at
concurrency 1 / 2 / 4 / 10, with all copies inside a batch agreeing. Decode batch
size changes the arithmetic, and at a ~1.5 %/token near-tie flip rate any change
of rounding re-rolls the trajectory.

The padded width is restored in `tt/generator_vllm.py` because it is the correct
one of the two, but it is not free: aggregate decode throughput at 10 concurrent
requests is ~14.7 tok/s padded versus ~53 tok/s active-row. **Re-deriving a
correct active-row path is worth doing** — the fix is in whatever prefill/decode
state a late prefill disturbs, not in the execution width itself.

## Mandatory accuracy rows

`gpqa_diamond_cot_zeroshot`, 10-document CI subset, `max_gen_toks=32768`, seed 42,
chat template applied, `--log_samples` retained this time:

| run | concurrency | flexible-extract | vs threshold 9 | wall time |
|---|---:|---:|---|---:|
| recorded baseline (blocked) | 32 | **4/10** | fail | 12,061 s |
| wrap fix, active-row width | 32 | 8/10 | fail | 2,048 s |
| wrap fix, serial | **1** | **9/10** | **meets** | 342 s |
| wrap fix + padded width | 32 | not run to completion | — | stopped |

The last row was stopped deliberately: padded-lane decode at 10 concurrent requests
runs at ~14.7 tok/s aggregate, and the release flow serves this model at
`max_concurrency: 1` (below), so a concurrency-32 GPQA number is not the figure the
gate will use.

`ifeval`, 28-document CI subset, task budget `max_gen_toks=1280`:

| run | concurrency | prompt strict | inst strict | TTI scalar |
|---|---:|---:|---:|---:|
| recorded baseline | 32 | 0.7857 | 0.8372 | 82.62 |
| wrap fix + padded width | 32 | 0.7857 | 0.8372 | 82.62 |
| HF control | — | 0.8571 | 0.8837 | 87.04 |

IFEval is unchanged by the wrap fix, which is consistent: its answers are a few
hundred tokens and rarely reach absolute position 1024.

## Per-document audit — what the aggregate hides

From `--log_samples` on the concurrency-32 wrap-fix run (8/10). The blocked run
deleted its samples, which is why its four cap-exhausted requests were never
classified:

| doc | generated tokens | boxed | target | correct | note |
|---:|---:|---|---|---|---|
| 0 | 32,761 (cap) | – | C | ✗ | degenerate: `_0_0_0_0…` from the first token |
| 1–7 | 669–1,114 | A,B,D,C,B,D,B | all match | ✓ ×7 | clean, terminated on EOS |
| 8 | 2,080 tok / 32,816 chars | – | D | ✗ | degenerate: `______…` from the first token |
| 9 | 32,768 (cap) | – | B | ✓ | coherent but never terminates |

Re-run **standalone**, the two failures are not model errors at all: doc 0 gives
785 tokens and `\boxed{C}` (correct), doc 9 gives 1,056 tokens and `\boxed{B}`
(correct), doc 8 gives 1,237 coherent tokens and `\boxed{A}` against target D — a
genuine wrong answer. All three stop on EOS. That is what pinned the second defect
to the serving path rather than the model.

## Where Stage 11 stands

- The **model path** clears the mandatory GPQA gate: 9/10 against threshold 9,
  serially, with the wrap fix. Against the model's published GPQA Diamond 82.3 %,
  9/10 is exactly in line.
- `meta_ifeval` reproduces its recorded passing row exactly.
- The **serving path** has an open defect at concurrency (above). Until it is
  fixed, a concurrency-32 release run cannot be trusted to measure model quality,
  and that is the configuration the release workflow uses.
- The formal Stage 11 verdict still requires the TTI release workflow, not just
  `lm_eval`. That has not been run here.

Worth keeping in mind for whoever sets the gate: 9/10 versus the control's 10/10
is not a statistically meaningful difference at n=10 (Fisher exact p ≈ 1.0), and
the threshold `floor(10 × 1.0 × 0.95) = 9` is derived from the control's own point
estimate. An implementation matching this model's published accuracy fails that
bar roughly half the time. The gate needs a sample-size-aware acceptance rule.

## The configuration this model will actually be judged by

`tt-inference-server` branch `vvukoman/add-8-models-to-release-flow`
(commit `60f80c4b`) adds this model to the Shield release flow. Everything above
was measured on the *old* Stage 11 recipe; the release recipe differs on almost
every axis that matters, so the numbers above do not transfer.

| axis | Stage 11 as graded | release flow (`60f80c4b`) |
|---|---|---|
| task | `gpqa_diamond_cot_zeroshot` | **`r1_gpqa_diamond`**, scored `exact_match,none` |
| thinking | off (template default) | **on**, `--default-chat-template-kwargs '{"enable_thinking": true}'` |
| decoding | greedy | **sampled**: temp 1.0, top_k 20, top_p 0.95 |
| subset | 5 % (10 docs) | **20 %** CI-nightly (~40 docs) |
| context | `max_model_len 262144` | `max_context 49152`, client `max_length 131072` |
| mesh | `MESH_DEVICE=P300x2` | **`MESH_DEVICE=P150x4`** — the spec notes the `p300_x2` descriptor "corrupted gemma4 decode logits" |
| fabric | `FABRIC_1D_RING` | `FABRIC_1D` |
| sampling mode | `sample_on_device_mode: all` | `decode_only` — needed so token ids ≥ 65536 are reachable |
| block size | harness default | `--block_size 64` (`GEMMA4_PAGE_BLOCK_SIZE=64`) |
| concurrency | `--max-num-seqs 32` | **`max_concurrency: 1`** |
| extra tasks | — | `terminal_bench_2`, `swe_bench_verified` (agentic, need Docker) |

Two consequences worth acting on:

1. The mesh descriptor. The release spec deliberately drives QB2's four chips as
   `P150x4` because `p300_x2` "corrupted gemma4 decode logits". Every measurement on
   this branch — including the blocked Stage 11 run — used `P300x2`. That should be
   re-checked on the autoport: if the descriptor matters here too, some of this
   port's recorded evidence was taken on a configuration the release flow rejects.
2. `max_concurrency: 1` means the serving-concurrency defect above would not be
   exercised by the release gate. It still needs fixing — a model that corrupts
   rows under concurrency is not servable — but it is not what blocks the gate.

### Reference: the canonical implementation already does this correctly

`models/demos/gemma4` — the implementation the plugin's builtin map resolves
`Gemma4ForCausalLM` to — passes the modulo to **all three** paged ops, including the
SDPA read (`tt/attention/decode.py:268`, `**paged_modulo_kwargs`), and its
`generator_vllm.py:406` comment states the invariant explicitly: pass
`cache_position_modulo=sliding_window` "to the three paged ops so they correctly
address the bounded physical pool". The autoport passed it to two of three. That
implementation is the quality bar for this port, and it is worth diffing against for
the remaining gaps (starting with the serving execution width).

### Autoport under the release configuration

Server launched exactly as the release spec describes, but with
`EXTRA_MODELS_DIR` set so the **autoport** is served rather than the demo model:

```bash
MESH_DEVICE=P150x4 HF_MODEL=google/gemma-4-26B-A4B-it \
EXTRA_MODELS_DIR=.../experiments/extra_models_dir \
GEMMA4_PAGE_BLOCK_SIZE=64 GEMMA4_MAX_TOKENS_ALL_USERS=49152 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-26B-A4B-it --block_size 64 --max_num_seqs 1 \
  --max_model_len 49152 --port 8000 \
  --additional-config '{"tt": {"sample_on_device_mode": "decode_only", "fabric_config": "FABRIC_1D", "trace_region_size": 200000000}}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking": true}' --reasoning-parser gemma4
```

The autoport comes up healthy on this configuration (`/health` 200, arch resolves to
`models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm:Gemma4ForCausalLM`) and
generates in thinking mode at ~25–30 tok/s single-user.

### Driving it through the TTI harness rather than by hand

Driving `lm_eval` directly is not the same test: the harness builds the command,
applies `limit_samples_map`, scores the declared `result_keys`, and emits the
verdict. Run locally against the already-serving autoport with:

```bash
cd tt-inference-server   # c8509ac2 + the 4 committed patches + cherry-pick 60f80c4b
MODEL_SPECS_ENV=dev ONLY_BENCHMARK_TARGETS=1 CACHE_ROOT=<cache> python run.py \
  --workflow evals \
  --runtime-model-spec-json <this dir>/autoport_releaseflow_spec.json \
  --tt-device p300x2 --service-port 8000 --server-url http://127.0.0.1 --no-auth \
  --skip-system-sw-validation --disable-trace-capture \
  --limit-samples-mode smoke-test
```

Three integration gaps had to be closed to get that far, none of them documented
on either branch:

1. **`MODEL_SPECS_ENV=dev` is mandatory.** `60f80c4b` registers these eight models
   in `workflows/model_specs/dev/llm.yaml`, and `EVAL_CONFIGS` is built only from
   the catalog that is actually loaded (`model_spec.py:1206`, default `prod`).
   Without it: `AssertionError: Model:=… not found in EVAL_CONFIGS`.
2. **The autoport spec's `model_name` must match the catalog key exactly.** The
   committed `autoport_release_spec.json` says `Gemma-4-26B-A4B-it`; the release-flow
   catalog key is `gemma-4-26B-A4B-it` (lower-case `g`). `EVAL_CONFIGS` is keyed by
   `model_spec.model_name` for specs present in `MODEL_SPECS`
   (`eval_config.py:5638`), so a runtime autoport spec is unreachable until the two
   agree. `autoport_releaseflow_spec.json` uses the catalog spelling.
3. **`--limit-samples-mode smoke` is invalid**; the enum accepts `smoke-test`
   (`workflow_types.py:354`, `SMOKE_TEST/CI_COMMIT/CI_NIGHTLY/CI_LONG`).

Running it this way also revealed a difference from the hand-rolled command: TTI
passes the spec's `until: []`, whereas a manual invocation inherits the r1 task's
`until: ['<|end_of_text|>','<|endoftext|>','<|im_end|>']`. Those markers belong to
other model families and are inert for Gemma, but it is the kind of divergence that
makes the harness run the authoritative one.

### `r1_gpqa_diamond` on the autoport, through TTI

| run | limit | docs | `exact_match,none` | workflow rc | wall time |
|---|---|---:|---:|---|---:|
| `--workflow evals` | `smoke-test` | 3 | **1.0 (3/3)** | 0 | 921 s |

Three documents is a plumbing validation, not a graded score — treat it as
"the release recipe runs end to end on the autoport and the model answers
correctly in thinking mode", nothing stronger. Generation is long: ~5 min per
document at ~29 tok/s single-user, because thinking mode plus `until: []` plus a
32,768-token budget lets the model reason at length. A CI-nightly run (0.2 →
~40 documents) is therefore a multi-hour job on one host.

### What of CI is reproducible on one host, and what is not

Reproduced locally:

- the eval itself, through TTI's own harness rather than a hand-written `lm_eval`
  command, so the spec's `gen_kwargs`/`model_kwargs`/`limit_samples_map` and the
  declared `result_keys` scoring all take effect;
- the release workflow's venv provisioning, including `EVALS_AGENTIC` (harbor
  0.6.5 clones and builds without Docker — Docker is only needed to *run* the
  agentic tasks);
- the whole serving path with the release spec's server flags.

Not reproducible here, and not claimed:

- **TTI-managed server startup.** TTI has no launcher for `models/autoports/*`, so
  the autoport can only be tested through the external-server path with a
  hand-launched server. The startup/warmup/trace-capture path CI exercises is
  therefore untested for this port by construction, not by omission.
- **`terminal_bench_2` and `swe_bench_verified`** — the container lacks Docker.
- **A CI-nightly graded number**, for the runtime reason above.

`--workflow release` additionally requires `HF_TOKEN` in the environment
(`run.py:841`), which `--workflow evals` does not.

### The local release verdict

`--workflow release --limit-samples-mode smoke-test` ran to a verdict against the
external autoport server. Full report copied to
`experiments/post_fix_evals/local_release_report.md`:

| section | result |
|---|---|
| **Acceptance** | ❌ FAIL, 2 blockers |
| Benchmarks | ✅ PASS 1/1 |
| Evals | ❌ FAIL — 0/3 passed, 2 failed, **1 NA** |
| Spec tests | 🟨 NA (no blocks — no spec-test mapping for this model) |

Both blockers are the Docker-dependent agentic rows, `terminal_bench_2` and
`swe_bench_verified`. Neither can run in this container, so this FAIL is an
environment limit, not a model result.

Benchmarks pass at the **strictest tier** on the release recipe — ISL 128 / OSL 128
at concurrency 1: TTFT 261.2 ms against a 300 ms target (ratio 0.87), 26.7 decode
TPS against 26 (1.03), throughput/user ratio 1.08. Functional and complete tiers
pass with margin.

The important finding is the eval row. `r1_gpqa_diamond` **scored 100 and is still
graded `NA`**, because the release-flow config carries no reference to compare
against: `published_score=None` and `gpu_reference_score_ref="TBD"`
(`eval_config.py`, this model's `EvalTaskScore`). So the accuracy gate cannot be
passed by any score at all in its current form — which is the same class of blocker
`ACCURACY_BLOCKER.md` recorded for the old recipe, now reappearing on the new one.
Somebody has to supply a published or GPU reference for `r1_gpqa_diamond` before
this row can grade, and that is a release-config task rather than a model task.

One provenance defect worth fixing in the spec: the report's metadata prints
`tt_metal_commit 4b17e185dea9` and `vllm_commit 938c45ed`, which are copied
verbatim from the spec JSON and describe the *old* Stage 11 run — not the code that
actually produced this report. A runtime spec that hard-codes commit fields will
mislabel every future run.

### How much of this is "as CI would run it" — and two rows CI cannot grade yet

The benchmark PASS above was graded against the **autoport spec's own** tiers
(functional 1000 ms/10 TPS, complete 500/20, target 300/26), which the Stage 11
author wrote. The release branch declares something different for this model in
`reference_config/benchmarking/benchmark_targets/model_performance_reference.json`:

```
gemma-4-26B-A4B-it / p300x2, isl 128 / osl 128 / concurrency 1:
  targets: { theoretical: { ttft_ms: 76.0, tput_user: 43.0, tput: 43.0 } }
  _comment: "ASSUMED, NOT VALIDATED … inferred from gemma-3-27b-it t3k …
             LIKELY PESSIMISTIC BY A LARGE MARGIN"
```

Re-running the benchmark workflow with exactly those targets substituted produces:

> Note: No perf targets are configured for these sweep points, so these rows are
> reported for information only and are not graded.

because the grader only recognises `functional` / `complete` / `target`
(`workflows/acceptance_criteria.py:14`, `llm_module/target_checks.py:30`), and
`theoretical` is consumed by a different reader (`workflows/utils_report.py:45`).
No `dev/llm.yaml` entry carries tiered targets at all, and the gemma-4-26B entry
carries no perf block. So as drafted:

- the **accuracy** row grades `NA` — no published or GPU reference;
- the **benchmark** row is **ungraded** — only a `theoretical` tier exists;
- the **agentic** rows need Docker.

That is consistent with the commit calling itself an "initial draft". For the
autoport specifically, the measured 261.2 ms TTFT / 26.7 TPS would *miss* the
assumed 76 ms / 43 TPS by 3.4× and 0.62× if those numbers were ever graded — worth
knowing, but the comment in that file predicts real MoE throughput should exceed
43, so the assumption itself is the thing to validate first.

Finally, `workflows/model_spec.py` in the same commit shows the release flow can
carry an autoport directly: `muse_glimmer_impl` uses
`code_path="models/autoports/meta_models_muse_glimmer_30b"`. Giving this port its
own `ImplSpec` would let TTI launch and grade it the way CI will, instead of
requiring the hand-launched external-server path used here.

### Not tested here, explicitly

1. **CI-nightly limits.** Everything above used `smoke-test` (3 documents for
   `r1_gpqa_diamond`). The graded limit is 0.2 (~40 documents) — roughly 3–4 hours
   at ~5 min/document in thinking mode.
2. **TTI-launched server**, for the reason above.
3. **`terminal_bench_2` / `swe_bench_verified`** — no Docker in this container.
4. **`spec_tests`** — no mapping for this model; the step is a no-op.
5. **The catalog path** (`impl: tt_transformers` → `models/demos/gemma4`), which is
   what CI would serve for the entry as drafted, rather than this autoport.
6. **The decoder unit suites** after the cache-view refactor — only the new
   read-wrap test was run; the device has been occupied by these evals.
7. **`MESH_DEVICE=P150x4` vs `P300x2` decode-logit corruption**, which the release
   spec asserts and no measurement on this branch has checked.

## Would CI pass or fail today?

**Upstream `main` plus the release-flow branch: it would report PASS, vacuously.**
The catalog entry sets `status: EXPERIMENTAL`, and upstream that status waives
everything this port would be judged on:

- `ModelStatusTypes.EXPERIMENTAL.required_target_tiers == []`
  (`workflows/workflow_types.py:322`) — no benchmark tier is enforced. The
  docstring is explicit: "an EXPERIMENTAL model (forge, new bring-up) can fail
  every performance benchmark and still be released."
- `evals_enforced` is `bool(required_target_tiers)` (`:336`), so it is `False` for
  EXPERIMENTAL, and upstream `_evals_enforced()` returns that value.

So an ungradable accuracy row, a Docker-failed agentic row and any perf miss would
all pass acceptance. The PASS would certify nothing.

**The local run reported FAIL only because this checkout carries the branch's own
patch** `b07735ae`, which hard-codes `_evals_enforced() -> True`:

> Treating an EXPERIMENTAL model as an implicit eval waiver can turn failed or
> missing accuracy rows into a release PASS without a declared issue.

`git diff origin/main -- report_module/acceptance_criteria.py` confirms that fix is
not upstream.

| enforcement | verdict | driver |
|---|---|---|
| upstream `main` today | **PASS** | EXPERIMENTAL waives evals and all benchmark tiers |
| + the branch's eval-enforcement patch | **FAIL** | 2 blockers (agentic, no Docker); accuracy row `NA` regardless |

Neither verdict is a statement about the port. Under strict enforcement the
blockers are a missing `r1_gpqa_diamond` reference and a missing Docker daemon;
under upstream enforcement nothing is checked at all. Three things have to land
before a release verdict means anything for this model: a published or GPU
reference for `r1_gpqa_diamond`, graded benchmark tiers instead of a lone
`theoretical` block, and the eval-enforcement fix upstream.
