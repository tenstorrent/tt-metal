# Dispatching Qwen3.8-27B through agentic bring-up CI

What it cost to take a port that ran clean locally and make it run in CI, and
which of those costs were avoidable. Written against runs 33388601937,
33396877687 and 33405666767 on `tenstorrent/tt-agentic-bringup-qb2`, all
pinning tt-metal `16cfafb73024f236cc648feacd0d2c09babf2e71`.

Companion to `google_gemma_4_31b/doc/agentic_bringup_ci_dispatch.md`, which
covers the same chain for Gemma. Read that one first; this one records only what
differs for Qwen and what that branch could not have told us.

## The chain

`tt-agentic-bringup-qb2` (workflow dispatch) -> `tt-shield`
(`determine_server_type.py`, picks the server) -> `tt-inference-server`
(`run.py`, builds the image and drives the sweep). Each hop can reject the
dispatch, and the build sits between the second and third: **every mistake past
`determine-server-type` costs a ~60 min image build before you learn about
it.**

## Cost asymmetry, which is the whole point

| Check | Cost | Catches |
| --- | --- | --- |
| Reproduce spec resolution locally | seconds | catalog and impl-name errors |
| Grep the port for host-shaped path assumptions | ~1 min | both failures below |
| Verify the **committed** tree, not the working tree | seconds | a spec that never got pushed |
| Dispatch and find out | ~60 min build, plus queue | everything else |

Both CI failures we hit were findable by the second row. Neither was found that
way.

## What actually failed, in order

### Run 33388601937 -- `OSError: Repo id must be in the form ...`

`model.py:254`. The port resolved weights by constructing
`$HF_HOME/hub/models--<id>/snapshots/<revision>` and handing the result to a
HuggingFace API that will accept a path only if it exists. On this host
`HF_HOME` happens to be the directory that layout is true for, so the
construction succeeded locally and never exercised the failure.

Fixed by asking the library instead of predicting it:
`snapshot_download(MODEL_ID, revision=..., local_files_only=True)`.

### Run 33396877687 -- `FileNotFoundError: model.safetensors.index.json`

`model.py:67`. The fix above returned a cache directory that existed and
passed `is_dir()` but held only metadata -- no shard index. The guard tested
for a directory when what it needed was a usable checkpoint.

Fixed by gating on the artefact rather than the directory: walk an ordered
candidate list (`MODEL_WEIGHTS_DIR`, then `snapshot_download`, then the
constructed path) and accept the first whose `model.safetensors.index.json` is
a file; if none is usable, raise with every candidate listed.

```python
SNAPSHOT_INDEX_FILE = "model.safetensors.index.json"

def snapshot_is_usable(candidate) -> bool:
    try:
        return (Path(candidate) / SNAPSHOT_INDEX_FILE).is_file()
    except (OSError, TypeError, ValueError):
        return False
```

The lesson is narrower than "handle the cache": **an existence check should
test the thing you are about to read, not the container it lives in.** The first
fix failed because it moved the prediction rather than removing it.

## Why the Gemma branch could not have warned us

Worth stating, because the obvious retrospective is "you had a branch that did
this already". It did not cover this. Gemma takes an explicit directory:

```python
DEFAULT_MODEL_DIR = Path("models/autoports/google_gemma_4_31b")
raw = Path(os.environ.get(MODEL_DIR_ENV, DEFAULT_MODEL_DIR))
```

and its prerequisites doc records a verified checkpoint path. Qwen *constructs*
a cache layout. Neither Gemma doc mentions `HF_HOME`, `snapshot`, weights
directories or the HF cache at all -- checked by grep. A port that is handed a
path cannot hit either failure above, so no amount of reading that branch would
have surfaced them.

Both were still preventable here, just not from there: the path
`MODEL_WEIGHTS_DIR` eventually resolved to was printed in run 33388601937's own
log, and was rejected during the first fix as "a tt-inference-server concept the
model should not depend on". That judgement cost the second build.

## The health clock is what actually failed the run

Run 33405666767 got past both weights fixes -- the engine reached
`Initializing a V1 LLM engine` -- and then died on the clock:

```
16:02:15  Waiting for inference server at .../health (timeout 1200s)
16:22:15  Inference server did not become healthy within 1200s
16:22:15  Inference server not healthy; aborting sweep.
```

Where those 1200 s went:

| From | To | Spent on |
| --- | --- | --- |
| 16:02:18 | 16:09:10 | staging weights (`Fetching 32 files`), ~6 min 52 s |
| 16:09:32 | 16:22:15 | engine load, killed at 12 min 43 s, incomplete |

Measured model load on this host is 881-950 s (~15 min), so ~13 min was never
going to be enough. **The run failed for a reason unrelated to the model.**

### Why it binds, and why raising it is not a magic number

`llm_module/server_control.py:25` sets
`DEFAULT_WAIT_HEALTHY_TIMEOUT_S = 3600.0`, and `wait_for_healthy(timeout=None)`
exists precisely to fall through to it. `llm_module/runner.py:64` declared
`wait_healthy_timeout_s: float = 1200.0` and always passed it, so the
fallthrough could never happen; no call site and no environment variable could
raise it. `capture_trace_timeout_s` had the same shape.

Fixed by defaulting both to `None` (`09599f09` on the
`mvasiljevic/qwen38-autoport-qb2` tt-inference-server branch), which restores
the intended fallthrough to 3600 s. Callers wanting a tighter bound can still
pass one. For calibration, Gemma's real CI bring-up was 1401 s.

### Staging is inside the window, and does not amortise

Attempt 2 staged weights in 6 min 50 s; attempt 3 in 6 min 52 s. The
`cache_root/weights` volume does **not** carry over between runs, so every
dispatch pays that ~7 min inside the health window. A plain re-dispatch would
have failed identically -- which is why the timeout, not the retry, was the fix.

### The local run passed the same gate, for a reason that does not transfer

Locally the window was 950.6 s of 1200 s (79%), because the weights were
already on the host and no staging happened inside it. Local bring-up is
therefore **not** evidence that CI bring-up fits: the two differ by the whole
staging step.

The 3600 s and 2400 s windows observed in Gemma run 33176296979 are the
`release` workflow (evals child first), a different controller path. They do
not apply to `benchmarks`.

## Pre-dispatch checklist

1. **Reproduce spec resolution** the way `tt-shield` does, from the committed
   ref. `determine_server_type.py:128` raises `No default impl for <model> on
   <device>` whenever the match count is not exactly one -- **including zero**,
   so the message does not distinguish "absent from the catalog" from "present
   but no default". Both Gemma failures were the first; the message says the
   second.
2. **Grep the port for host-shaped assumptions** before every first dispatch:
   `grep -rnE "HF_HOME|snapshot|/huggingface|MODEL_WEIGHTS|Path\\(\"/" tt/`.
   Anything that builds an absolute path from an environment variable is a
   candidate failure.
3. **Verify the committed tree, not the working tree.**
   `git diff --stat <base> HEAD`, `git show HEAD:<path>`, and for a pushed
   branch `gh api repos/<o>/<r>/contents/<path>?ref=<branch>`.
4. **Keep the dispatched ref stable.** Docs and unrelated commits on the CI
   branch change the SHA and force a fresh ~60 min build. This document is on
   `mvasiljevic/qwen38-autoport-qb2-docs` for that reason; the CI branch tip
   stays at the pinned SHA so re-dispatches reuse the cached image.
5. **Confirm before dispatching to shared CI**, and re-read any in-flight
   instruction first.

## Reading a failed run

Do not diagnose from the job log -- it is an extract ("Extracted N/M lines").
Download the artifact:

```bash
gh api repos/<owner>/<repo>/actions/runs/<id>/artifacts \
  --jq '.artifacts[] | "\(.name)  \(.size_in_bytes)"'
gh run download <id> --repo <owner>/<repo> -n workflow_logs_<...> -D ./art
```

It holds what the job log does not: `docker_server/vllm_*.log` (the engine's
own output), `run_logs/*.log` (harness side, with the health-wait timestamps
used in the table above), and `runtime_model_specs/*.json` (the config actually
applied). Both failures above were first diagnosed from the truncated job log;
the health-window numbers only became available once the artifact was opened.

## Mistakes made while doing this

| Mistake | Consequence | Correction |
| --- | --- | --- |
| Fixed the first weights failure by replacing one path prediction with another | Second 60 min build failed on a metadata-only cache directory | Gate on the file you are about to read (`model.safetensors.index.json`), not on the directory |
| Rejected `MODEL_WEIGHTS_DIR` as "a tt-inference-server concept the model should not depend on" | Discarded the correct answer, which run 1 had already printed in its own log | The harness telling you where it staged the weights is evidence, not coupling |
| Skipped the offered pre-dispatch grep for host-shaped paths | Both failures were in its output | Run the cheap static check; it costs a minute against a 60 min build |
| Measured 3600 s in a Gemma **release** run and concluded the CI health clock was not a risk | Generalised across workflows: `benchmarks` uses the 1200 s path, and the run in flight while I wrote that died on it | Measure the path you are about to run, not an adjacent one; check which controller the workflow uses |
| Treated the passing local bring-up as evidence CI would fit | Local skips the ~7 min weight staging that CI pays inside the same window | Compare like windows: add staging to any local bring-up figure before predicting CI |
| Diagnosed both failures from the truncated job log | Delayed finding the health numbers, which were in the artifact all along | Download `workflow_logs` first |

## Run history from this host

| Run | Ref | Outcome |
| --- | --- | --- |
| 33388601937 | `16cfafb7302` (pre-fix) | failed, `OSError` on HF repo id at `model.py:254` |
| 33396877687 | fix 1 | failed, `FileNotFoundError` on shard index at `model.py:67` |
| 33405666767 | fix 2 | build ok, weights ok; **failed on the 1200 s health clock** mid-load |
| next | + `09599f09` (timeout fallthrough) | not yet dispatched |

## Unrelated ordering note, recorded because it cost eval time

`tt-inference-server/workflow_module/workflows.py:497-500` pins
`llm_children = ("evals", "benchmarks", "spec_tests")`. The release workflow
therefore runs evals before the benchmark sweep, so a defect a 12 s sweep point
would expose is instead met during a multi-hour eval. The ordering is a
`ClassVar`, not configurable per model or per run.
