# TT-Metal Testing Notes

This repo is often tested on a remote reserved machine after connecting with `ird`.

To check all reservations:

```bash
ird list
```

If you are on a lab box and need the remote container first:

```bash
ird connect-to 1
```

All commands below assume you are running from the repo root:

```bash
/localdev/kevinmi/tt-metal
```

## Cache Setup

Your user cache should live under `/localdev/kevinmi/.cache` because home directory space is limited.

If `~/.cache` is still a real directory, migrate it:

```bash
mkdir -p /localdev/kevinmi
mv ~/.cache /localdev/kevinmi
ln -s /localdev/kevinmi/.cache ~/.cache
```

If `~/.cache` is already a symlink to `/localdev/kevinmi/.cache`, leave it as-is.

`/localdev` is per machine. On a machine where the directory was not created by you, the ownership may be wrong. Fix it from a Docker image or VM where you have `sudo` access:

```bash
sudo chown $USER:group_1211400513 /localdev/kevinmi
sudo chown $USER:group_1211400513 /localdev/kevinmi/.cache
```

If the workspace looks stale or corrupted, use one of the repo cleanup workflows below. If device-kernel or build-artifact issues persist, prefer `./build_metal.sh --clean`, `ccache -C`, and removing `.cpmcache/` before rebuilding.

## Repo Update And Build Workflows

Use one of the workflows below depending on how broken the repo state looks.

### Update only and build

Use this when the checkout is healthy and you just want the latest `main`, submodules, and a rebuild before switching back to the branch under test.

```bash
git checkout main
git fetch origin
git pull --rebase --prune
git submodule sync
git submodule update --init --recursive
git checkout <previous branch>
./build_metal.sh
```

### Clear and update everything, then build

Use this when the repo has drifted into a bad state and you want to clean the checkout without fully clearing caches.

```bash
git fetch origin
git checkout main
git submodule sync
git pull --rebase --prune
# NOTE: clean -fdx will delete all untracked files including files in .gitignore.
# Back up anything you want to keep first.
git clean -fdx
git submodule update --init --recursive
./create_venv.sh
./build_metal.sh
```

### Clear everything including cache

Use this before rebuilding if you hit strange device kernel errors or other failures that do not look related to your code change.

```bash
./build_metal.sh --clean
ccache -C
rm -rf .cpmcache/

git fetch origin
git checkout main
git submodule sync
git pull --rebase --prune
# NOTE: clean -fdx will delete all untracked files including files in .gitignore.
# Back up anything you want to keep first.
git clean -fdx
git submodule foreach --recursive git clean -xfd
git submodule update --init --recursive
./create_venv.sh
./build_metal.sh
```

If this still fails, the next step is usually a fresh repository checkout in a new directory.

## Python Environment

Create and activate the Python environment before running model tests.

```bash
./create_venv.sh
source python_env/bin/activate
export PYTHONPATH=$(pwd)
export TT_DIT_CACHE_DIR=/localdev/kevinmi/.cache
```

- You do **not** need to remove the Python environment every time you rebuild. The existing env can be reused across rebuilds.
- After installing the Python environment, run `uv pip install imageio-ffmpeg`.

## Building

Run `./build_metal.sh` from the repo root after modifying any kernel code (C++ device kernels, data movement, compute kernels, etc.). Python-only changes do not require a rebuild.

## Machine Check

Check device inventory:

```bash
tt-smi -ls
```

Use the output with the chart below to identify the machine type and choose the right test expectations.

If the box is unhealthy or you see hugepage, NOC address, or segmentation fault errors during device init, reset all devices before blaming the model change:

```text
NOC address of a hugepage does not match the expected address
Fatal Python error: Segmentation fault
```

```bash
tt-smi -r 0,1,2,3,4,5,6,7
```

| Product Type | Architecture | Model Value |
| --- | --- | --- |
| Single N150 - 1 chip | wormhole_b0 | x1 |
| Single N300 - 2 chips | wormhole_b0 | x2 |
| Loud Box N300 - 4 card mesh (8 chips) | wormhole_b0 | lb |
| N150 Galaxy 6U - 32 card mesh | wormhole_b0 | glx6u |
| N150 Galaxy - 32 card mesh (deprecated) | wormhole_b0 | glx4u |
| Single P100a - 1 chip, 120 Tensix core | blackhole | p100 |
| Single P150a - 1 chip, 140 Tensix core | blackhole | p150 |
| Single P150b - 1 chip, 140 Tensix core | blackhole | p150 |
| Single P150c - 1 chip, 140 Tensix core | blackhole | p150 |
| Single P300a - 2 chips | blackhole | p300 |
| Single P300c - 2 chips | blackhole | p300 |
| Desk Box P150a - 2 card mesh (2 chips) | blackhole | db |
| Quiet Box P150a - 4 card mesh (4 chips) | blackhole | qb |
| Quiet Box Global Edition P300c - 2 card mesh (4 chips) | blackhole | qbge |
| Loud Box P150b - 8 card mesh (8 chips) | blackhole | lb |
| Galaxy6u - 32 cards (32 chips) | blackhole | glx6u |

## Testing

- You **cannot** run two tests on device at the same time. Only one test process can use the device at a time. **Always check `ps aux | grep python` before starting any device test.** If a background task is running a device sweep, wait for it to finish. Running a second device process will hang or corrupt firmware state, requiring `tt-smi -r` reset.

## Task Tracking (Context Anchoring)

Any task that spans multiple steps, involves kernel debugging, numerical investigation, model bringup, or performance optimization **must** have a tracking document. The document is external memory that survives across sessions — it captures decisions, reasoning, rejected alternatives, and current state so a new session can resume in seconds instead of re-discovering everything.

### When and where

Create a tracking doc for any multi-step task: kernel work, model bringup, numerical debugging, or perf optimization. Skip it for single-command tasks or quick questions.

Place it at the repo root as `UPPER_SNAKE_CASE.md` (e.g. `CONV3D_FP32_REDUCTION.md`, `LTX2_VAE_BRINGUP.md`).

### Required sections

Every tracking doc must have these sections. Keep it under 100 lines — this is a working record, not documentation.

```markdown
# <Task Name>

## Goal
One sentence: what are we trying to achieve and on what hardware.

## Decisions
| Decision | Reason | Rejected Alternative |
|----------|--------|----------------------|
| ... | ... | ... |

## Constraints & Workarounds
- Hardware target (WH/BH, chip count, mesh shape)
- Dtype / math fidelity requirements
- L1 / DRAM budget limits
- Upstream dependencies or blocked-on items
- Ops falling back to CPU torch (e.g. LayerNorm PCC too low on BH)
- Ops decomposed into other ttnn ops (e.g. Conv3D L1 OOM → depthwise + pointwise)
- For each workaround note what the permanent fix would be

## Surprises & Discoveries
- Unexpected hardware behavior, numerical quirks, or edge cases found during the work
- e.g. "untilize leaves unpacker in Float32 — next tilize reads bf16 data as garbage"

## Open Questions
- [ ] Unanswered questions that affect next steps

## State
- [x] Completed steps
- [ ] Next steps

## Key Measurements
Numerical accuracy (PCC, MAE, max error) and/or device perf (kernel duration, throughput) as they are collected. Include test commands to reproduce.
```

### Rules for maintaining the doc

1. **Create at task start.** Fill in Goal, Constraints, and initial State before writing any code.
2. **Update at decision points.** When a design choice is made (blocking strategy, data format, reduction method), add it to Decisions with the reasoning and what was rejected.
3. **Record measurements immediately.** When a test produces PCC or perf numbers, add them to Key Measurements with the exact command used.
4. **Resolve open questions.** When an open question is answered, check it off and move the conclusion to Decisions or Constraints.
5. **Mark completed steps.** Check off State items as they finish; add new items as scope evolves.
6. **Log workarounds.** When a ttnn op falls back to CPU torch or is decomposed into other ops, add it to Constraints & Workarounds with what the permanent fix would be. Remove only when the fix lands and is verified.
7. **Never delete history.** If a decision is reversed, keep the original entry and add a new row explaining the reversal. The reasoning behind the old decision is still valuable.

### Session startup

At the start of any session that continues prior work, check for a tracking doc at the repo root matching the task. If one exists, read it first — it is the authoritative record of where things stand, not the previous conversation.

### Autonomous execution plans

When the user asks for autonomous long-running work (e.g. "port this model end-to-end", "optimize all conv layers"), use an ExecPlan as described in `PLANS.md`. Place it at the repo root as `PLAN_<TASK_NAME>.md`.

## Debug Tools

- [Kernel Debug Print](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/kernel_debug_print.html) — print tiles, scalars, and strings from device kernels to host
- [Watcher](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/watcher.html) — background thread that monitors device status and catches hangs/asserts
- [Tracy Profiler](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy.html) — profile device-side RISC-V and host-side Python/C++ code
- [Device Program Profiler](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html) — duration counts on marked sections of device programs
- [All tools](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/index.html) — full list including kernel asserts, inspector, tt-triage

## Troubleshooting

### Blackhole demo tests skipped — no matching system type

When dispatching `(Blackhole) Demo tests`, you **must** set `system-type` to match the SKU the model is configured for. The default is `P150 (1xP150)`. If the model only has entries for a different SKU (e.g. `bh_loudbox`, `bh_quietbox_2`), the test matrix loads but every test is skipped with no error.

```bash
# Wrong — default system-type is P150, wan22 only runs on LoudBox/QuietBox 2
gh workflow run "(Blackhole) Demo tests" --ref <branch> -f model=wan2.2-t2v-a14b

# Right — specify the system type
gh workflow run "(Blackhole) Demo tests" --ref <branch> -f model=wan2.2-t2v-a14b -f system-type="LoudBox (8xP150)"
```

Check `tests/pipeline_reorg/blackhole_demo_tests.yaml` for which `skus:` a model uses, then match to the `system-type` dispatch option.

### Firmware cache `File exists` error on device init

If tests fail immediately with:

```text
filesystem error: cannot create directories: File exists [/home/kevinmi/.cache/tt-metal-cache/.../firmware/brisc/]
```

The error means `~/.cache` is a symlink whose target does not exist (e.g. `/localdev/kevinmi/.cache` was never created or was wiped when the machine was reimaged). The symlink itself counts as "exists", so `create_directories` fails. Follow the steps in **Cache Setup** above to recreate the target and fix ownership.
