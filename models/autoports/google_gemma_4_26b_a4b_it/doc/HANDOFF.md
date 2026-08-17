# Handoff: debugging the GPQA failure from another machine

Written 2026-08-17 so this can be picked up independently, while the Qwen3.6-27B
port is worked in parallel. Everything needed is on this branch; nothing required
is left on the originating host.

## The failure, in one line

`meta_gpqa_cot` / `gpqa_diamond_cot_zeroshot` scores **4/10** against a
same-harness HF control's **10/10** (threshold 9), while `meta_ifeval` passes at
exactly its threshold (TT 82.62 % vs HF 87.04 %). Full detail:
`doc/tti_release/ACCURACY_BLOCKER.md`, `HF_REFERENCE_NOTES.md`, and the two
`hf_reference_*.json` control artifacts.

## Do not reason from PCC

This port has the strongest correctness evidence of the three in this fleet, and
it is against **real HF weights**: prefill PCC 0.9986/0.9971, decode
0.99965/0.99979, replayed decode bit-exact, advertised context 262,144 held. Layer
PCC is healthy and **does not predict this failure**. See
`doc/READINESS_ANALYSIS.md`.

## The gap that most likely explains it

Longest **correctness-measured generation** anywhere on this branch is **100
tokens** (`--gen-len 100`); prefill PCC was taken at **S=33**. The eval that
passes generates **1,280** tokens; the one that fails generates **32,768**.
`position 262143` decode and `S=262143` prefill are *capacity* probes — one decode
step at a high position, and a prefill that fits — not sustained generation.

Ordering: passes at 100 (measured), passes at 1,280, **fails at 32,768**.

Two ranked mechanisms, from `doc/tti_release/AUTODEBUG_GPQA_DIVERGENCE.md`:

1. **A single early greedy-token flip from the numerical policy.** That doc's own
   framing is better than "accumulated drift": "one flip changes the entire
   remaining reasoning trajectory and final choice". Note this model is unusually
   precision-sensitive — stage 08 found **BFP8 activations collapse it to
   0 %/1 % top-1/top-5**, where both dense ports in this fleet ship BFP8
   activations happily.
2. **Long-generation cache/position/active-row lifecycle.**
   `sliding_window=1024`, so a 32,768-token generation wraps the sliding cache
   about **32 times**; stage 04 tested exactly one boundary (1024 vs 1025), and
   **25 of 30 layers are sliding**.

The doc's focused experiment — replay the ten requests at concurrency 1 recording
the first HF/TT divergence index — was **specified but never run**. No artifact
exists. An early divergence implicates base numerics; one that grows with index
implicates accumulation.

## The next experiment, ready to run

`doc/tti_release/experiments/` on this branch contains:

- `gemma_gpqa_ab.sh` — starts the server and runs the identical ten-document
  eval, changing **only** `GEMMA4_PRECISION_CONFIG`. Baseline is the known 4/10;
  HF's 10/10 is the control.
- `precision_A_bfp8_decode_mlp.json` — single variable: decode-only packed MLP
  **BFP4_B → BFP8_B**, the safest change the AutoDebug doc ranks first.
- `precision_B_bfp8_hifi2.json` — variant A plus `dense_mlp` fidelity
  **LOFI → HIFI2**, the second step in that ranked plan.

Reads directly: **recovers toward 10/10 → numerics** (mechanism 1, and the fix is
a precision decision); **stays ~4/10 → lifecycle** (mechanism 2, a more serious
defect).

`build_generator` / `Gemma4FullModel` load
`doc/datatype_sweep/selected_precision_config.json` by default;
`GEMMA4_PRECISION_CONFIG` or the `precision_config_path` constructor argument
overrides it, so no code change is needed to A/B a policy.

## Traps that cost time here — read before running

- **This is an instruction-tuned model: use `local-chat-completions`.** An
  operator attempt with lm-eval's raw `--model local-completions` against
  `/v1/completions` returned **HTTP 400** for every request. The committed script
  already uses the chat backend.
- **After killing a server mid-run, reset the devices before the next attempt.**
  A `pkill` of the vLLM process followed by an immediate relaunch died in
  firmware init with
  `RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset`. Sequence:
  `tt-smi -r`, then a 1×4 mesh open/close + `create_global_semaphore` smoke,
  *then* relaunch. Skipping this makes the next run fail differently and reads as
  worsening hardware.
- **Set `max_gen_toks` explicitly.** lm-eval's API backend defaults to **256**,
  which truncates chain-of-thought. This model's canonical GPQA allowance is
  32,768 and its IFEval allowance 1,280. (The sibling Qwen port's low scores were
  produced under that 256 default — its `gen_kwargs` recorded only
  `{'stream': False, 'seed': 42}`.)
- **A per-model `.gitignore` does not survive a branch switch.** It is tracked on
  this branch, so switching away leaves the previous model's ignored artifacts
  untracked *and unignored* — 1,293 MiB of `.tracy` captures in one observed case.
  Move the other model's `models/autoports/<model>/` out of the tree after
  switching, and confirm `git status --porcelain` reads clean.

## Environment facts needed on a fresh machine

- Weights: `google/gemma-4-26B-A4B-it` is **not gated**; 48.1 GiB, **2**
  safetensors shards. Preflight passed here: tokenizer loads fast (vocab 262,144),
  all shards open, config gives 30 layers = 25 sliding + 5 full attention, 16 Q /
  8 KV heads, `sliding_window=1024`, `max_position_embeddings=262144`, 128 experts
  top-8, `moe_intermediate_size=704`.
- Target mesh is **P300X2** (1×4, four Blackhole p300c). The readiness harness
  rejects `--mesh-device P300X2` without the mesh label in
  `models/common/readiness_check/mesh_device.py`, which this branch carries.
- Agent deps must **not** go into tt-metal's `python_env` — `openai-codex`
  silently upgrades `pydantic` past the pinned 2.9.2. Use a separate runner venv.

## Two things about the eval set worth knowing before treating 4/10 as official

1. **TTI declares this model's GPQA task as `r1_gpqa_diamond`**, not the
   `gpqa_diamond_cot_zeroshot` the stage graded itself on. I am not spinning that
   as good news — "r1" implies R1-style long reasoning, so the declared task
   likely generates *longer* chains and would fail harder — but the recorded
   number is on a variant TTI does not ask for.
2. **This model also requires `terminal_bench_2` and `swe_bench_verified`**, both
   agentic. Those need Docker, which the model container lacks. They are
   satisfiable from the host: serve from the container, run the Harbor client
   outside it (host, or a sibling container with
   `-v /var/run/docker.sock:/var/run/docker.sock` so Terminal-Bench can spawn its
   task containers), and point `api_base` at the container's bridge address
   (verified reachable here at `http://172.17.0.2:8000`, HTTP 200). TTI's
   `EVALS_AGENTIC` venv clones and editable-installs SWE-agent and Harbor v0.6.5;
   it was never created on the originating host.

---

## CORRECTION: this branch alone is NOT sufficient — two other repos are needed

The opening of this document says "everything needed is on this branch; nothing
required is left on the originating host". **That is wrong.** Audited
2026-08-17. A second machine needs four things, and two of them are not published
anywhere yet.

| repo | ref needed | state |
|---|---|---|
| `tenstorrent/tt-metal` | `mvasiljevic/fmf/google-gemma-4-26b-a4b-it` | pushed |
| `tenstorrent/vllm` | **4 local commits on `dev`** | **local only** |
| `tenstorrent/tt-inference-server` | **4 local commits on `main`** | **local only** |
| `tenstorrent/agentic-research` | `mvasiljevic/forge-lane-pipeline-findings` (context) | pushed |

### tenstorrent/vllm — mandatory, and unpublished

Cloned at `dev` = `7c99bd3b8`, then four commits were made locally:

- **`938c45ed7 Register Gemma 4 TT vLLM adapter`** — **without this the plugin does
  not know this architecture and the model cannot be served at all.**
- `c5f35e550 Fix async host sampling RNG isolation`
- `ed7a409b9 Register Qwen3.6 autoport vLLM adapter`
- `bc1dbf107 Add Falcon3 TT vLLM integration support`

The TT plugin registers models in
`plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models()`;
upstream `dev` carries no Gemma-4 entry.

### tenstorrent/tt-inference-server — needed for the release workflow

Cloned at `main` = `c8509ac2`, then four commits locally:

- `ca152fe2 Support autoport external-server release specs` — also carries the
  **`EXPERIMENTAL` eval-enforcement fix**, without which a failed or missing
  accuracy row still reports a release PASS.
- `bd15f1cd Add Falcon3 Base nightly eval references`
- `e26e723b Propagate model context to API evals`
- `b9a18e8f Qwen3.6-27B eval config, terminal-bench token budget, external-chat
  meta evals` (operator-preserved; the stage blocked without committing it)

### What to do

Ask the operator to push branches for those two repos before starting, or
reproduce them: the tt-metal branch is self-contained for the *model*, but the
serving path (vllm) and the release/eval path (tt-inference-server) are not.

The precision A/B in `doc/tti_release/experiments/` needs only the tt-metal branch
plus the vllm Gemma registration — it does not need tt-inference-server, since it
drives `lm_eval` directly against the server rather than through the release
workflow.

---

## RESOLVED: this branch is now self-sufficient. No other repo push is required.

Supersedes the correction above. Everything a second machine needs is on this
branch. Tested 2026-08-17.

### tenstorrent/vllm is no longer a dependency — use the bundle

Upstream `dev` **already registers all six Gemma-4 arch names**; the local commit
was only a one-line retarget of `_gemma4_target` from
`models.demos.gemma4...` to the autoport. The plugin has a designed hook for
supplying a model without editing it — `_register_models_from_extra_dir()`, whose
own docstring says it "runs first so a distributed bundle can supply a model
without touching this file", and the builtin map that follows uses
`_register_model_if_missing`, so the bundle wins.

So: run **upstream `tenstorrent/vllm` at `dev`, unmodified**, and set

```bash
export EXTRA_MODELS_DIR=<repo>/models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/experiments/extra_models_dir
```

`extra_models_dir/gemma4_autoport/vllm_metadata.json` is committed here. The hook
TT-prefixes the arch, yielding `TTGemma4ForConditionalGeneration`.

### tt-inference-server is not needed to debug this

The precision A/B drives `lm_eval` **directly against the vLLM server**, not
through the release workflow, so tt-inference-server is required only to reproduce
the *release verdict* — not to localise the defect. Note if you do run the release
workflow from upstream: without `ca152fe2` its `EXPERIMENTAL` status silently
disables eval enforcement, so a failed or missing accuracy row still reports
PASS.

### And nothing is stranded anyway — the patches are committed

`experiments/patches/` carries both sets of local commits as `git am`-able
patches, so they can be reproduced exactly without access to the originating host:

- `vllm-dev-4-commits.patch` (22 KiB) — Gemma-4 retarget, async host-sampling RNG
  isolation fix, Qwen3.6 and Falcon3 registrations. Base: `dev` @ `7c99bd3b8`.
- `tt-inference-server-main-4-commits.patch` (114 KiB) — autoport external-server
  release specs **including the EXPERIMENTAL eval-enforcement fix**, Falcon3 Base
  nightly eval references, model-context propagation to API evals, and the
  Qwen3.6 eval config / Terminal-Bench token budget. Base: `main` @ `c8509ac2`.

Apply with `git am < …patch` on the stated base, or cherry-pick what you need.

### The lowest-dependency way to attack the failure

The precision A/B needs a vLLM server. **The hypothesis can be tested with no
vLLM at all**: `models.common.readiness_check.run_teacher_forcing` and
`run_autoregressive` drive `tt/generator.py` directly, and that is how stages 06
and 07 produced their AIME24 top-k numbers. Running either at a generation length
of a few thousand tokens — rather than the 100 used so far — exercises exactly the
regime where GPQA fails, needs only this tt-metal branch, and produces a
first-divergence index. That is the experiment
`doc/tti_release/AUTODEBUG_GPQA_DIVERGENCE.md` specified and never ran.

---

## QUESTION THE PIPELINE BEFORE THE MODEL — four defects in the grading path

Added 2026-08-17. Across three ports in this fleet the **pipeline has been wrong
more often than the models**: lm-eval's implicit 256-token cap truncating
chain-of-thought, TTI's `EXPERIMENTAL` status silently disabling eval
enforcement, Falcon's IFEval graded against the wrong metric variant of a model
card (apparent 0.544 "quality failure" that was parity all along),
`--plugin-config` being stale against current vLLM, and a CCL shape bug invisible
to a near-square test suite. **Treat the harness as a suspect here too.** Four
concrete issues in the GPQA grading path, all verifiable from committed
artifacts:

### 1. `until: ['</s>']` is a stop string from the wrong model family

Both arms ran with `generation_kwargs.until = ['</s>']`. That is the
Llama/Mistral EOS. **Gemma-4's stop tokens are `<eos>` and ids `[1, 106, 50]`**
(106 = `<end_of_turn>`), so `</s>` can never appear in its output and that stop
can never fire. Present in both arms, so probably not the differential — both
paths likely fall back on the model's own EOS — but it is a real
task-configuration defect, and it means generation length is governed by
`max_gen_toks=32768` and EOS handling rather than by the configured stop.

### 2. `strict-match` scores **0.0 in both arms** — the filter never matches

TT strict 0.0 / flexible 0.4; HF strict 0.0 / flexible 1.0. A filter that
returns zero for a *known-good* reference is not validating anything: the strict
answer-format extractor does not match this model's output shape at all. The
entire verdict therefore rests on `flexible-extract` alone. Before concluding
anything about the model, establish that the extractor is fit for its output
format.

### 3. No per-sample outputs were captured — the failure cannot be inspected

`log_samples` is `None` in **both** arms, so no generated text was retained.
There is currently no way to tell whether the six failing documents are *wrong
answers* or *right answers the extractor missed*. Those are completely different
defects — one is the model, the other is the pipeline.

### 4. The pass threshold is hostage to a ten-sample control

Acceptance used `floor(10 × 1.0 × 0.95) = 9`, i.e. 9 of 10 required, derived from
an HF control that happened to score a perfect 10/10 on ten documents. Had the
control scored 9/10 the bar would have been 8. A single fortunate reference
document tightens the bar on the port. `flexible-extract` stderr is 0.163.

## Therefore: change the first experiment

**Do not start with the precision A/B.** Start by re-running the *same* eval with
`--log_samples` and reading the six failures. It is the cheapest step and it
discriminates pipeline from model:

- answers correct but unextracted → **pipeline defect** (extractor/filter), and
  the precision A/B would have chased a ghost;
- answers genuinely wrong → **model defect**, and the ranked hypotheses in
  `AUTODEBUG_GPQA_DIVERGENCE.md` apply; proceed to the precision A/B;
- generations truncated at 32,768 with no answer emitted → **stop/EOS handling**,
  i.e. issue 1 above, and the fix is task configuration.

`experiments/gemma_gpqa_ab.sh` already passes `--log_samples`, so running it once
against the **unmodified** selected policy gives you the baseline samples the
original run never kept. Do that before changing any precision.

Also worth re-checking rather than assuming: whether the server-side chat
template and the HF control's locally-applied template produce the *same* prompt.
The autoport ships a passthrough compatibility template; if the two arms prompt
differently, they are not measuring the same task and the 4/10 vs 10/10 gap is
not attributable to the model at all.

### Outcome of the four points above

All four were independently confirmed on the second machine, and the advice to
read the samples before touching precision was correct — it is what led to the
root cause. Specifically:

1. Confirmed. The same wrong-family stop string appears in the new
   `r1_gpqa_diamond` recipe as `['<|end_of_text|>','<|endoftext|>','<|im_end|>']`,
   equally inert; the release spec overrides it to `[]`.
2. Confirmed. `strict-match` is 0.0 on both arms, so the verdict rests on
   `flexible-extract` alone.
3. Acted on. Re-running with `--log_samples` and classifying every document is
   what exposed the real defect: two failures were **degenerate from their very
   first token** — neither a wrong answer nor a missed extraction — and that led
   to the sliding-cache read wrap below.
4. Confirmed, and it survives the fix: at 9/10 the port now sits one document
   from failing a bar derived from a single ten-sample control.

The precision A/B was also run, and the prediction that it would chase a ghost
was right: HiFi2 everywhere plus BFP8 decode MLP plus FP32 logits moves the
per-token disagreement from 8/512 to 9/512 at +28 % decode time.

## CLOSED: the blocker was a sliding-cache read-side wrap bug

Found and fixed 2026-08-17 on the second machine. Full write-up and evidence:
**`doc/tti_release/AUTOFIX_SLIDING_CACHE_READ_WRAP.md`**.

`paged_update_cache` received `cache_position_modulo=1024`; the matching
`paged_scaled_dot_product_attention_decode` did not, in all three decode paths
(`optimized_decoder.py`, `multichip_decoder.py`, `functional_decoder.py`). Per
that op's own documentation, without it "positions past the bounded capacity
collapse onto physical block 0 and silently corrupt the cache". 25 of 30 layers
are sliding with a cache sized at exactly 1024 tokens, so every generation
crossing **absolute position 1024** was corrupted. Passing the write side's
`update_kwargs` to the read call fixes it, at no throughput cost
(28.33 vs 28.84 t/s/u).

Same prompt, same 2,048-token budget, shipped precision policy:

| | tokens generated | EOS | output |
|---|---:|---|---|
| before | 2048 (cap) | no | coherent to ~890, then token soup |
| after | 884 | yes | complete derivation, `Final Answer: \boxed{A}` (correct) |

Corrections to this document's earlier sections, all measured rather than argued:

1. **Ranked hypothesis 1 (precision) is refuted.** HiFi2 everywhere + BFP8 decode
   MLP + FP32 logits changes the pre-wrap TT/HF greedy disagreement from 8/512 to
   9/512 — no improvement, +28 % decode time.
2. **Ranked hypothesis 2 (cache lifecycle) was right**, but the mechanism is a
   missing kwarg on the read path, not "accumulated" anything, and it is not
   subtle: it is total corruption at a fixed absolute position.
3. **"The first-divergence experiment was never run" is wrong.** `RUN_NOTES.md:99`
   and `AUTOFIX.md:50-52` record it: divergence at generated index 15 with
   re-prefill recovery. That divergence is unrelated near-tie noise — 7 of 8
   pre-wrap disagreements pick HF's rank-2 token at an HF top1−top2 margin whose
   median is 1.25, against 15.19 at agreements.
4. **"Passes at 100, passes at 1,280, fails at 32,768" is not a length ladder.**
   Both evals crossed the same boundary, they just crossed it at different rates.
   `ifeval.yaml` sets `max_gen_toks: 1280` in the *task*, and task
   `generation_kwargs` beat the API backend's 256 default
   (`lm_eval/models/api_models.py:963`), so both TT and HF IFEval ran at 1,280 —
   the `{'stream': False, 'seed': 42}` in `tti_eval_ifeval.json` records the CLI
   override only, not the effective budget. IFEval mostly passed because typical
   instruction-following answers are a few hundred tokens and never reach absolute
   position 1024; the documents whose answers did run long were corrupted, which is
   the most likely source of the 82.62 % vs 87.04 % gap (≈1 prompt of 28). GPQA's
   task YAML genuinely sets no budget, which is why it needed an explicit
   `max_gen_toks=32768` and why nearly every GPQA request crossed the boundary.
5. **The greedy sampler is exonerated.** The shipped sharded chunked-topk path
   matched an exact argmax on identical logits in 1,024 step-level A/Bs.
6. **Reproducing the exact failing eval needs one thing not mentioned here:** the
   GPQA documents come from the gated `Idavidrein/gpqa` dataset, so a second
   machine needs an HF token with those terms accepted.
