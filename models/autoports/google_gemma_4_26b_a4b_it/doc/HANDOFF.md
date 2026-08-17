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
