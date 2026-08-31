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

## Local is not the weaker test -- on one axis it is the stricter one

Measured, not assumed:

| Path | Health timeout | Observed bring-up | Used |
| --- | --- | --- | --- |
| CI, first bring-up (run 33176296979, Gemma) | 3600 s | 1401 s | 39% |
| CI, benchmarks child (same run) | 2400 s | server already up | -- |
| Local `--local-server` (Qwen3.8, this host) | 1200 s | 950.6 s | 79% |

`llm_module/server_control.py:25` defaults to `DEFAULT_WAIT_HEALTHY_TIMEOUT_S
= 3600.0`. `llm_module/runner.py:64` declares
`wait_healthy_timeout_s: float = 1200.0`, and that value is the one the
`--local-server` path uses. No call site passes it and no environment variable
reaches it, so **the local budget is a third of the CI budget and is not
configurable without editing the harness.** A cold-cache local run of a model
this size can fail bring-up while the same model passes in CI.

The 2400 s value observed on the benchmarks child is not present in this
checkout; CI runs a different tt-inference-server commit, so treat the two
columns as measurements of two builds, not one.

This corrects, for this repo, the Gemma branch's fourth finding that
`wait_healthy_timeout_s = 1200` is marginal in CI. On the CI path it is 3600.
The 1200 is real, but it binds locally.

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
| Predicted a CI health-timeout risk from the 1200 s default in `runner.py` | Wrong by 3x -- CI uses 3600 s; the exposure is local, not CI | Measure the window from a successful run's artifact before predicting |
| Diagnosed both failures from the truncated job log | Delayed finding the health numbers, which were in the artifact all along | Download `workflow_logs` first |

## Run history from this host

| Run | Ref | Outcome |
| --- | --- | --- |
| 33388601937 | `16cfafb7302` (pre-fix) | failed, `OSError` on HF repo id at `model.py:254` |
| 33396877687 | fix 1 | failed, `FileNotFoundError` on shard index at `model.py:67` |
| 33405666767 | fix 2 | dispatched 2026-08-31T14:57:39Z |

## Unrelated ordering note, recorded because it cost eval time

`tt-inference-server/workflow_module/workflows.py:497-500` pins
`llm_children = ("evals", "benchmarks", "spec_tests")`. The release workflow
therefore runs evals before the benchmark sweep, so a defect a 12 s sweep point
would expose is instead met during a multi-hour eval. The ordering is a
`ClassVar`, not configurable per model or per run.
