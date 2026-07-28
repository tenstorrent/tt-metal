# skillexp run plan — 2 machines × 4 models × 4 arms

Hand this file to either machine. It is self-contained: roles, exact commits, exact commands,
branch names, and the handoff protocol. Read `.agents/EXPERIMENT.md` for *why* the factors are cut
the way they are.

**Goal:** measure how much of the optimize stage's speedup comes from `$graph-fusing` and how much
from `$shard-advise`, over four models, with the functional decoder held identical per model.

---

## 0. Roles

| | **MACHINE A** | **MACHINE B** |
|---|---|---|
| host | `qb2-120-p05t03` | the second QB2 |
| `$shard-advise` | **always on** | **always off** |
| phase 2 arm | `mvasiljevic/qb2/skillexp/nofuse-advise` | `mvasiljevic/qb2/skillexp/nofuse-noadvise` |
| phase 3 arm | `mvasiljevic/qb2/skillexp/fuse-advise` | `mvasiljevic/qb2/skillexp/fuse-noadvise` |
| needs tt-mlir advisor build | **yes** | **no** |
| owns functional decoder for | phi35-mini, qwen36-27b | north-mini-code-1-0, gemma4-26b-a4b-it |

Each machine ends up producing **both of its arms for all four models**. The functional decoder for
a model is produced once, by its owner, and consumed by both machines.

> **Known confound, stated on purpose.** The shard-advise factor is perfectly correlated with machine
> identity. Two QB2s should be equivalent, but silicon/thermal/build drift is not zero. Mitigations
> below (§1.4) — pin the same tt-metal commit, record `tt-smi`, and at the end cross-run **one**
> model's `nofuse-noadvise` arm on machine A as a machine-effect probe. If that reproduces machine
> B's number within noise, the confound is bounded.

---

## 1. Prerequisites (both machines, once)

### 1.1 tt-metal checkout and build

All five skillexp branches have **identical trees outside `.agents/`**, so build once and never
rebuild between arms.

```bash
cd ~/tt-metal
git fetch origin
git checkout mvasiljevic/qb2/skillexp/base     # commit below
# build per your normal QB2 recipe, then:
git rev-parse HEAD          # record in the run log
```

**Pin the exact commits before you start.** All five are pushed to
`github.com/tenstorrent/tt-metal`; resolve and record them, then never move off them:

```bash
for b in base fuse-advise fuse-noadvise nofuse-advise nofuse-noadvise; do
  printf '%-48s %s\n' "mvasiljevic/qb2/skillexp/$b" \
    "$(git rev-parse --short=11 origin/mvasiljevic/qb2/skillexp/$b)"
done | tee -a ~/skillexp-logs/ENV.md
```

The pinned values are in the experiment record's `README.md` (§Compiler, device, pinned commits) in
agentic-research `skill-contribution-experiments/skillexp-fusing-advisor/` on branch
`mvasiljevic/skillexp-fusing-advisor` — retrievable from git, not from someone's home directory. If
your resolved SHAs differ from those, **stop and reconcile**: somebody moved a branch, and every
completed run is suspect.

Per-machine setup is not in git — it is machine-local infra. Get the container launch script from
machine A (`~/run_tt_xla_container.sh`), pass `HF_TOKEN` through the environment rather than copying a
hardcoded one, and note that **Codex auth is per-machine**: `~/.codex` does not travel, so a second
machine needs its own `codex login` plus an `AUTH_OK` probe before it runs anything.

`base` is `mvasiljevic/gpt-oss-pipeline-progress` (`aab03552379`) plus skills/prompt-only commits.
Nothing in `ttnn/`, `tt_metal/`, or `models/` changed, so an existing build of `aab03552379` is
valid. Deliberately **not** used as base: `mvasiljevic/gpt-oss-trace-tracker`, whose skills are one
commit fresher but which also merges a trace-allocation-tracker runtime change (30 files in
`ttnn/`). Changing the runtime under a perf comparison is not worth one doc paragraph, and its skill
delta only points at a checker that build would not have.

### 1.2 Codex runner

Runs **in the container**, not on the host:

```bash
docker exec -u mvasiljevic -w /home/mvasiljevic/tt-metal mvasiljevic-ttxla bash -lc '
  source ~/tt-metal/python_env/bin/activate
  export TT_METAL_HOME=/home/mvasiljevic/tt-metal
  python -m pip install -r .agents/requirements.txt
  ~/.local/bin/codex --version
  python -c "import ttnn; print(\"ttnn OK\")"
'
```

Git: the **host** is not gh-authed. Commit on the host, **push from the container**. Keep
`git commit` and `git push` as separate commands (a compound one gets blocked). Every repo needs
`git config --local user.name mvasiljevic` and `user.email mvasiljevic@tenstorrent.com` before
committing from the host — the host global identity is someone else's.

### 1.3 tt-mlir shard advisor — MACHINE A ONLY

Machine B never runs the advisor and needs none of this.

```bash
git clone -b mvasiljevic/shard-advisor-dram-sharding https://github.com/tenstorrent/tt-mlir.git
cd tt-mlir && git checkout 618cd4e75d          # pin exactly
cmake -G Ninja -B build -DTTMLIR_ENABLE_OPMODEL=ON -DTTMLIR_ENABLE_TTNN_JIT=ON \
      -DTTMLIR_ENABLE_RUNTIME=ON -DTTMLIR_ENABLE_STABLEHLO=ON
cmake --build build
export TTMLIR_ADVISOR_HOME=$PWD
```

Then `.agents/skills/shard-advise/SETUP.md` Part A.2 to verify. Two sharp edges from prior runs:

- tt-mlir symlinks your tt-metal as its metal source, so **whichever branch tt-metal sits on decides
  whether tt-mlir compiles**. If the build fails on `bh_ring_size` at the MoE weight packers, export
  the build's tt-metal commit to a sibling path and bind-mount it at the *same* path in a second
  container (`~/run_ttxla_dram_pin_container.sh` on machine A) so ccache and the incremental build
  stay valid.
- Every `cmake --build` overwrites `$TT_METAL_HOME/ttnn/ttnn/_ttnn.so` with an incompatible copy.
  Re-copy `build_Release/lib/_ttnn.so` over it after each build or `import ttnn` dies on
  `get_root_mesh_buffer`.

### 1.4 Record once, per machine

Into `~/skillexp-logs/ENV.md`, committed with the first push:

```bash
tt-smi -ls --local            # board revs, ARC/fw versions
hostname; nproc; free -g
git -C ~/tt-metal rev-parse HEAD
git -C ~/tt-mlir rev-parse HEAD   # machine A only
```

---

## 2. Models

| short | HF id | `<model_dir>` | owner |
|---|---|---|---|
| phi35-mini | `microsoft/Phi-3.5-mini-instruct` | `microsoft_phi_3_5_mini_instruct` | **A** |
| qwen36-27b | `Qwen/Qwen3.6-27B` | `qwen_qwen3_6_27b` | **A** |
| north-mini-code-1-0 | `CohereLabs/North-Mini-Code-1.0` | `coherelabs_north_mini_code_1_0` | **B** |
| gemma4-26b-a4b-it | `google/gemma-4-26B-A4B-it` | `google_gemma_4_26b_a4b_it` | **B** |

`<model_dir>` is the HF id lowercased with every non-alphanumeric replaced by `_`. Downstream gates
resolve the autoport directory from it — do not add hardware or experiment qualifiers.

`DECODE_BATCH=32` for every model and every stage; batch 1 is measured alongside it and is the
primary optimization target. If a model cannot run batch 32, the stage records
the byte calculation / failed capacity probe and the largest feasible batch, and **that number is
then used for every arm of that model** — write it into `~/skillexp-logs/ENV.md` and tell the other
machine before phase 2.

---

## 3. Branch and tag naming

**Eight branches total, not twenty.** The arm is the unit of analysis, so one result branch per
arm accumulates all four models; completion is marked per model by a tag, not by a branch.

| what | name | count |
|---|---|---|
| functional decoder, per model | `mvasiljevic/qb2/skillexp/fd/<model_dir>` | 4 |
| results, per arm (all 4 models) | `mvasiljevic/qb2/skillexp/run/<arm>` | 4 |
| this machine's live status | `mvasiljevic/qb2/skillexp/status/<machine>` | 1 each |
| "FD is done, you may pull it" | tag `skillexp/fd-ready/<model_dir>` | 4 |
| "this arm+model is done" | tag `skillexp/done/<arm>/<model_dir>` | 16 |

`<arm>` is one of `nofuse-advise`, `nofuse-noadvise`, `fuse-advise`, `fuse-noadvise`.
`<machine>` is `a` or `b`.

Model directories are disjoint, so four models coexist on one arm branch without interacting, and
redoing one model is a new commit touching only its directory — never a history rewrite. Arm-vs-arm
comparison is then a single `git diff` between two branches instead of a cross-product of sixteen.

**Only tags mean "complete".** A branch may be pushed mid-run; a tag is pushed only after the stage
passed its gate. Wait on tags, never on branches:

```bash
# block until the other machine's FD for a model is ready
until git ls-remote --tags origin "refs/tags/skillexp/fd-ready/<model_dir>" | grep -q .; do
  sleep 300; git fetch origin --tags --quiet
done
```

---

## 4. The run

Set these once per shell (in the container):

```bash
cd /home/mvasiljevic/tt-metal
source ~/tt-metal/python_env/bin/activate
export TT_METAL_HOME=/home/mvasiljevic/tt-metal
export TTMLIR_ADVISOR_HOME=~/tt-mlir      # MACHINE A only
```

The one command shape used everywhere:

```bash
run_stage() {   # run_stage <branch> <hf_id> <model_dir> <logtag> <start-index> <prompt...>
  local br=$1 hf=$2 md=$3 tag=$4 idx=$5; shift 5
  git checkout "$br"
  python .agents/scripts/multigoal \
    --repo /home/mvasiljevic/tt-metal \
    --codex-bin ~/.local/bin/codex --codex-home ~/.codex \
    --sandbox danger-full-access --approval-policy never \
    --replace HF_MODEL="$hf" \
    --replace MODEL_DIR="models/autoports/$md" \
    --replace DECODE_BATCH=32 \
    --start-index "$idx" \
    --log-dir ~/skillexp-logs/"$tag" \
    "$@"
}
```

Runner exit codes: `0` all green · `3`/`5` a goal ended blocked/failed · `6` a stage failed its gate
critically · `7` the gate harness itself is broken. On `usageLimited`/`budgetLimited`, resume the
stopped thread rather than restarting the stage:
`--resume-stage N --log-dir <same dir>`. `--start-index N` starts a *fresh* thread and loses context.

### Phase 1 — functional decoder (your two models only)

From `base`, never from an arm branch. All four arms must optimize the *same* functional decoder.

```bash
# MACHINE A: phi35-mini, qwen36-27b     MACHINE B: north-mini-code-1-0, gemma4-26b-a4b-it
run_stage mvasiljevic/qb2/skillexp/base "$HF" "$MD" "p1-fd-$MD" 1 \
  .agents/prompts/model_bringup_multigoal/01-functional-decoder.txt
```

Then, per model — commit on host, push from container, tag last:

```bash
git checkout -B mvasiljevic/qb2/skillexp/fd/$MD
git add models/autoports/$MD ~/skillexp-logs/ENV.md
git commit -m "skillexp FD: $MD functional decoder (base $SHA_BASE, DECODE_BATCH=32)"
git push origin mvasiljevic/qb2/skillexp/fd/$MD
git tag -a skillexp/fd-ready/$MD -m "FD complete: PCC <x>, traced decode b1 <y> / b32 <z>"
git push origin skillexp/fd-ready/$MD
```

The tag message carries the numbers the other machine needs to sanity-check its pull. Do **not** tag
a stage that did not pass its gate — report it instead (§6).

### Phase 2 — optimize WITHOUT fusing, all four models

Arm: A → `nofuse-advise`, B → `nofuse-noadvise`. Own models first, then the other machine's once its
`fd-ready` tags appear.

```bash
git fetch origin --tags
ARM=nofuse-advise                       # MACHINE B: nofuse-noadvise
git checkout -B skillexp-work mvasiljevic/qb2/skillexp/$ARM
git merge --no-edit mvasiljevic/qb2/skillexp/fd/$MD      # brings in the shared FD

run_stage skillexp-work "$HF" "$MD" "p2-$ARM-$MD" 2 \
  .agents/prompts/model_bringup_multigoal/02-optimized-decoder.txt

# first model of this arm creates the branch; later models add commits to it
git fetch origin && git checkout -B mvasiljevic/qb2/skillexp/run/$ARM \
  $(git rev-parse --verify --quiet origin/mvasiljevic/qb2/skillexp/run/$ARM || echo skillexp-work)
git merge --no-edit skillexp-work            # no-op when this is the first model
git add models/autoports/$MD
git commit -m "skillexp $ARM: $MD optimized decoder (FD from skillexp/fd-ready/$MD)"
git push origin mvasiljevic/qb2/skillexp/run/$ARM
git tag -a skillexp/done/$ARM/$MD -m "traced decode b1 <y> -> <y'>, b32 <z> -> <z'>, PCC <p>"
git push origin skillexp/done/$ARM/$MD
```

The `git merge` of the FD branch is conflict-free: the arm branch touches only `.agents/`, the FD
branch only `models/autoports/<model_dir>/`.

Machine A's stage-02 gate requires a real `ttnn-advise capture` under
`models/autoports/$MD/doc/optimized_decoder/shard_advise/` (`report.json` + `final_ir.mlir`). If
`dram_sharded_considered == 0` in `report.json`, the **capture** is wrong, not the model — fix the
capture before accepting the stage (see SETUP.md "Capture preconditions"). Machine B has no such
gate; its `02-optimized-decoder.check.sh` does not exist.

### Phase 3 — optimize WITH fusing, all four models

Arm: A → `fuse-advise`, B → `fuse-noadvise`. Same FD baseline as phase 2 — start from the FD branch
again, **not** from the phase-2 result.

```bash
ARM=fuse-advise                         # MACHINE B: fuse-noadvise
git checkout -B skillexp-work mvasiljevic/qb2/skillexp/$ARM
git merge --no-edit mvasiljevic/qb2/skillexp/fd/$MD

run_stage skillexp-work "$HF" "$MD" "p3-$ARM-$MD" 2 \
  .agents/prompts/model_bringup_multigoal/01b-fused-decoder.txt \
  .agents/prompts/model_bringup_multigoal/02-optimized-decoder.txt
```

Then push and tag exactly as in phase 2 with the new `$ARM`.

### Phase 4 — machine-effect probe (machine A, one model, optional but cheap)

```bash
ARM=nofuse-noadvise; MD=microsoft_phi_3_5_mini_instruct
# same as phase 2, on machine A, using machine B's arm branch
```

Push as `mvasiljevic/qb2/skillexp/run/nofuse-noadvise-onA`. If it lands within noise of machine
B's number for that model, the machine/advisor confound is bounded.

---

## 5. What each run must leave behind

Under `models/autoports/<model_dir>/doc/`:

- `functional_decoder/` — README + work_log, prefill/decode PCC per layer kind, traced decode at
  batch 1 **and** 32, `tt-perf-report` tables + CSVs, watcher-clean run, determinism evidence.
- `context_contract.json` — HF-advertised context, current supported context, batches the decode
  contract covers. Must pass `.agents/scripts/check_context_contract.py --model-dir models/autoports/<model_dir>`
  with the **strict** gate: below-advertised context without DRAM evidence is a critical failure.
- `fused_decoder/` — fuse arms only: before/after PCC and latency at both batches.
- `optimized_decoder/` — before/after PCC and latency at both batches, chosen and rejected configs
  with evidence, `tt-perf-report` tables. Advise arms also `shard_advise/report.json` +
  `final_ir.mlir`, and a work_log line per advisor recommendation: applied, or rejected with
  before/after numbers.

The numbers the comparison turns on: **warmed traced decode latency at batch 1 and at
`DECODE_BATCH`, before and after the stage, from the same harness.** Batch 1 is the primary target
(per `$optimize`); `DECODE_BATCH` must not regress. Report both for every arm — they are not
substitutable, because shard params differ between a sub-tile and a full-tile activation.

---

## 6. Coordination and failure

- **Comms are tags plus your own status branch.** Run the monitor in tmux beside the run; it
  publishes to `mvasiljevic/qb2/skillexp/status/<machine>`, which only your machine writes, so there
  is never a merge. See the experiment record in agentic-research
  (`skill-contribution-experiments/skillexp-fusing-advisor/`, branch
  `mvasiljevic/skillexp-fusing-advisor`):

  ```bash
  MACHINE=a ./scripts/skillexp_monitor.sh   # per machine, in tmux, once
  ./scripts/skillexp_board.sh               # the 4x4 board, from anywhere with a tt-metal clone
  ```
- **A blocked stage is a result, not a hole.** Push whatever the stage produced to its
  `run/<arm>` branch, do **not** tag it done, and let the monitor record the exit code and failing
  gate. Do not silently substitute a weaker measurement.
- **Do not touch the five skillexp skill branches.** If a skill genuinely needs a fix mid-experiment,
  it must land on `base` and be cherry-picked to all four arms, and every completed run before it is
  invalidated. Say so in `STATUS.md` before doing it.
- **Device recovery** is `$tt-device-usage`: `tt-smi -ls --local`, `tt-smi -r`, `sleep 20`, re-list.
  The `sleep 20` matters — without it a spurious recheck forces an extra reset.
- **One hardware-facing command at a time.** No parallel model runs on one machine: they share the
  devices and would corrupt each other's perf numbers, which are the entire point.
