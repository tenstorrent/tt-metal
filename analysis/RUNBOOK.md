# SDPA and TopK measurement, test and model runbook (Blackhole p100a)

Scope: every command used to measure, test and model SDPA and TopK in the September 2026 revamp
campaign, written for a Tenstorrent engineer with a Blackhole p100a card who has not run any of this
before. Every command below was read out of a script, a harness, a log or a source file in this
workspace; none was invented. Where a command needs a file that is not in a pushed git branch, the
exact location is given together with the way to recreate it.

Conventions follow `handoff/revamp/README.md`: plain factual prose, no em dashes, no first person,
every number and every path traceable to a file on disk. Every path is written against the variables
of `handoff/revamp/PORTABLE_CONTRACT.md` (`WORK`, `TTM`, `TTM_FRESH`, `POLARIS`, `HANDOFF`, `DD`), so
that any command can be pasted after the single export block of section 1.1 and nothing points at
another user's directory. The scripts resolve the same variables themselves, from their own location,
so a copied workspace runs unedited.


## 0. Cheat sheet

One screen. Details and verification in the sections that follow. Everything below assumes the
variable block of section 1.1 has been pasted first, and that the clones of section 1.1 exist.

```bash
# the one variable block every later command depends on  (section 1.1)
export WORK=/proj_sw/user_dev/$USER/SDPA
export TTM=$WORK/tt-metal TTM_FRESH=$WORK/tt-metal-fresh POLARIS=$WORK/polaris
export HANDOFF=$WORK/handoff/revamp DD=$WORK/data/bh_zones
export TOPKOUT=$WORK/data/topk    # a directory of your own when writing new TopK cells
```

```bash
# calibration checkout (every SDPA number in the campaign was measured on this tree)
export TT_METAL_HOME=$TTM ARCH_NAME=blackhole TT_METAL_FORCE_JIT_COMPILE=1
cd $TT_METAL_HOME && source python_env/bin/activate
# fresh main-tip checkout (TopK, drift checks): no venv of its own, PYTHONPATH wins over the editable install
export TT_METAL_HOME=$TTM_FRESH ARCH_NAME=blackhole
export PYTHONPATH=$TT_METAL_HOME:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools
source $TTM/python_env/bin/activate && cd $TT_METAL_HOME
```

```bash
# one SDPA prefill point, device profiler on, reduced to per-invocation walls  (section 3.1)
cd $TTM/analysis/campaigns
SDPA_CAUSAL=1 SDPA_SEQ=4096 SDPA_QCHUNK=128 SDPA_KCHUNK=128 ./run_zone.sh myrun_zoff
# zones on, then off again; ZONES READER_STUB MASK_OFF EXP_STUB BARRIER_THR  (section 4.3)
./set_zone_config.sh 1 0 0 0 0
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 ./run_zone.sh myrun_zon
./set_zone_config.sh 0 0 0 0 0
# all 155 perf counters, five replays, merged into one device log  (section 5)
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 ./run_zone.sh myrun_zoff_mp mp
# one regime point through any harness, with an ops report  (section 3.4)
R1_MODE=cross R1_SQ=1024 R1_SK=8192 R1_ITERS=3 ./run_regime.sh myrun_zoff plain "analysis/r1_regimes.py::test_r1_regime"
# paged decode sweep  (section 3.10)
DEC_B=32 DEC_POS=1024,4096 DEC_GRID=8x8 ./run_decode.sh myrun_decode
```

```bash
# read a capture back by hand  (sections 4.5 and 5.4)
cd $TTM
tail -n +2 /path/to/<tag>.csv > /tmp/zr.csv
python analysis/zone_reduce.py /tmp/zr.csv --out /tmp/mytag
python analysis/post_process.py generated/profiler/.logs/profile_log_device.csv --seq-lens 4096 --out /tmp/counters.csv
# ops report: per-op DEVICE KERNEL DURATION, needed for decode and the model level runs  (section 6)
python -m tracy -r -m pytest analysis/decode_sweep.py::test_decode_sweep -s
ls generated/profiler/reports/*/ops_perf_results_*.csv
```

```bash
# TopK: enumerate, run one class group, join the ops report to the cells  (section 7)
python3 $TTM/analysis/campaigns/topk_campaign.py --list --classes ALL
cd $TTM_FRESH
python -m tracy -r -p -v $TTM/analysis/campaigns/topk_campaign.py \
    --classes L1,L2,L3,L4,L5 --out $TOPKOUT
python3 $TTM/analysis/campaigns/topk_campaign.py --postprocess \
    generated/profiler/reports/<date>/ops_perf_results_<date>.csv \
    $TOPKOUT/cells_L1_L2_L3_L4_L5.csv
```

```bash
# the model, its tests and its static check; no card needed  (section 9)
cd $POLARIS
.venv/bin/python -c "from ttsim.perf.roofline_sdpa import SdpaConfig, predict; r = predict(SdpaConfig(S=4096, q_chunk=128, k_chunk=128, num_heads=32, num_kv_heads=8, num_cores=110, is_causal=True)); print(r.wall_clock_cycles, r.components)"
.venv/bin/python -m pytest tests/test_perf -q
.venv/bin/python -m mypy ./
# validate the model against a tracy ops CSV (decode adds --positions <sidecar>)  (section 9.5)
.venv/bin/python tools/sdpa_validate.py \
    $HANDOFF/data/model_level/llama8b_attn_prefill_S4096_ops_perf_results.csv \
    -o /tmp/val --model llama8b_prefill_S4096
```

Four facts that cost the most time when forgotten:

- `TT_METAL_FORCE_JIT_COMPILE=1` on every run whose kernels depend on a header toggle. Without it the
  cached kernel binaries are reused and the toggle silently has no effect (section 11.1).
- `--perf-counter-multipass` produces no ops report. Read the merged device log instead (section 11.4).
- The device wall is `max(KERNEL zone end) - min(KERNEL zone start)` over all cores and RISCs, not the
  median per-core span, and a loop harness is 3.5 to 4 percent slower than a single-op one
  (section 11.3).
- A profiler build on Blackhole main tip overflows the BRISC firmware region. The fix is the
  `dev_mem_map.h` constant that `mvlahovic/analyze_sdpa_fresh` carries as a commit (section 11.2).


## 1. Machines and prerequisites

### 1.1 First: clone the code into your own directory

Nothing in this runbook runs until the repositories are cloned under the reader's own user directory.
Everything needed to rerun the campaign is on pushed branches, so there is nothing to copy out of
anyone else's workspace: two clones of tt-metal (or one clone and one worktree) and one clone of
polaris.

The layout every later section assumes, which is the shape of `PORTABLE_CONTRACT.md`:

```
$WORK/                   = /proj_sw/user_dev/$USER/SDPA
  tt-metal/              clone, branch mvlahovic/sdpa_topk_harness: the SDPA zone instrumentation,
                         every measurement harness under analysis/, and the runners, campaign
                         scripts and reducers under analysis/campaigns/ (the calibration vehicle)
  tt-metal-fresh/        second checkout or worktree, branch mvlahovic/analyze_sdpa_fresh: main tip
                         at stock firmware (the TopK campaign and the kernel checks)
  polaris/               clone, branch mvlahovic/roofline_model_sdpa or mvlahovic/roofline_model_topk
```

The one variable block. Every command in every later section is pasteable after this block and after
nothing else:

```bash
export WORK=/proj_sw/user_dev/$USER/SDPA
export TTM=$WORK/tt-metal
export TTM_FRESH=$WORK/tt-metal-fresh
export POLARIS=$WORK/polaris
export HANDOFF=$WORK/handoff/revamp
export DD=$WORK/data/bh_zones
export TOPKOUT=$WORK/data/topk
mkdir -p $WORK $DD $TOPKOUT
```

`WORK`, `TTM`, `TTM_FRESH` and `POLARIS` are the reader's own clones, and every command in this
runbook resolves its code through them. `HANDOFF`, `DD` and `TOPKOUT` are the one thing that is not
in git and therefore not cloned. `HANDOFF` names an analysis workspace: the reports, the offline
scripts and the measured data that no branch carries. The raw profiler dumps of the campaign are
2.4 GB (1.9 GB of per-tag device logs plus 526 MB of merged multipass logs), so they are in no
repository; they stay in the analysis workspace on the shared filesystem, together with the reports
that cite them. Point `HANDOFF` at that workspace when a command below reads one of its files.

Setting them is an environment override, which is what `PORTABLE_CONTRACT.md` asks for: every script
resolves the same names from the environment first and falls back to its own position, so no script is
edited either way. `DD` and `TOPKOUT` must name directories the reader can write, which the block
above creates; a runner fails on its first write otherwise. Point them elsewhere to keep two
campaigns apart:

```bash
export DD=$WORK/data/bh_zones_r2 TOPKOUT=$WORK/data/topk_r2 && mkdir -p $DD $TOPKOUT
```

Polaris, the model. Both model branches are pushed, so this is a plain clone and checkout. Check out
one of the two, not both in turn:

```bash
git clone https://github.com/tenstorrent/polaris.git $POLARIS
cd $POLARIS
git checkout mvlahovic/roofline_model_sdpa      # option 1: the SDPA model, PR 493
git checkout mvlahovic/roofline_model_topk      # option 2: SDPA plus TopK, PR 530; the branch of record
git log --oneline -3
git branch --show-current
```

`mvlahovic/roofline_model_topk` is the branch every section 9 command was run against, and
`mvlahovic/roofline_model_sdpa` is its SDPA-only predecessor, kept for the PR 493 review state.
Confirm both are on the remote before cloning if there is any doubt:

```bash
git ls-remote --heads https://github.com/tenstorrent/polaris.git | grep roofline_model
```

The polaris environment. Two paths work. The repository README documents the Miniforge path, which is
what CI follows:

```bash
cd $POLARIS
conda env create --file environment.yaml     # the runtime environment, named polaris
conda env create --file envdev.yaml          # the development environment, named polarisdev
conda activate polarisdev
pre-commit install
```

The path this workspace used instead is a uv virtual environment inside the checkout, which is what
every `$POLARIS/.venv/bin/python` command in sections 7, 9 and 10 refers to. `uv` already exists in the
ird image at `/opt/venv/bin/uv`, so nothing has to be installed to build it:

```bash
cd $POLARIS
/opt/venv/bin/uv venv --python 3.12 .venv
/opt/venv/bin/uv pip install --python .venv/bin/python \
    deepdiff einops gitpython hydra-core loguru lxml markdown-it-py matplotlib networkx numpy \
    onnx openpyxl pillow pydantic pyelftools pyyaml scipy simpy sphinx sphinx-rtd-theme \
    pandas tabulate mypy pytest pytest-mock pytest-cov
/opt/venv/bin/uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/python --version
.venv/bin/python -m pytest tests/test_perf -q
```

That is the `environment.yaml` dependency set with the conda pins dropped, plus `pytest`,
`pytest-mock`, `pytest-cov` and a CPU-only `torch`, plus `pandas` and `tabulate` because the reducers,
the figure scripts and the offline model scripts import them. The versions that installs are newer
than the `environment.yaml` pins (`environment.yaml` pins python 3.13.2, matplotlib 3.10.0, numpy
2.2.3; the workspace `.venv` is python 3.12.13 with matplotlib 3.11.1, numpy 2.5.1, pandas 3.0.5,
torch 2.13.0+cpu) and the suites pass on it: 363 collected under `tests/test_perf`, and the whole
repository suite green (section 9.2). The full development pin set, needed only for a faithful
reproduction of the CI static-analysis step, is in section 9.1.

tt-metal, the calibration checkout. One branch carries the whole measurement side. Confirm both
tt-metal branches are on the remote first, because a clone cannot reach what has not been pushed:

```bash
git ls-remote --heads https://github.com/tenstorrent/tt-metal.git \
    | grep -E 'sdpa_topk_harness|analyze_sdpa_fresh'
git clone https://github.com/tenstorrent/tt-metal.git $TTM
cd $TTM
git checkout mvlahovic/sdpa_topk_harness
git submodule update --init --recursive
```

That branch is the tree that reproduces the calibration measurements. It carries three things, and
listing them is worth the two commands because every later section assumes all three are present:

```bash
cd $TTM
git ls-tree -r --name-only HEAD analysis/
git ls-tree -r --name-only HEAD ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/ | grep zone
```

- The SDPA zone instrumentation, as committed kernels rather than as a patch to apply: the two new
  headers `sdpa_zones.hpp` (the `SDPA_ZACC` accumulate zones and the `SDPA_ZFLUSH` emit) and
  `sdpa_zone_config.hpp` (the five compile-time toggles), plus the instrumented
  `compute/compute_streaming.hpp`, `compute/compute_common.hpp`, `compute/sdpa.cpp`,
  `dataflow/dataflow_common.hpp`, `dataflow/reader_interleaved.cpp` and
  `dataflow/writer_interleaved.cpp`. Section 4 is what they do.
- Every measurement harness of section 3 under `analysis/`, 28 tracked entries including the three
  `analysis/kernels/zone_tax_*` sources: `attn_decode_perf.py`, `attn_prefill_perf.py`,
  `chunked_sweep.py`, `decode_latency_sweep.py`, `decode_sweep.py`, `gap_sweep.py`,
  `indexer_probe.py`, `joint_sweep.py`, `make_charts.py`, `membound_test.py`,
  `mla_decode_latency_sweep.py`, `mla_perf_sweep.py`, `p2_sweep.py`, `post_process.py`,
  `prefill_latency_sweep.py`, `r1_decode.py`, `r1_mla_decode.py`, `r1_regimes.py`, `sdpa_sweep.py`,
  `sparse_sweep.py`, `topk_sweep.py`, `zone_reduce.py`, `zone_sweep.py`, `zone_tax.py` and
  `README.md`. None of these is a working-tree-only file any more.
- The runners, the campaign block scripts, the reducers and `topk_campaign.py` under
  `analysis/campaigns/`. Sections 4, 7 and 8 drive them; they are run from that directory and write
  their captures and derived tables into `$DD`.

One deliberate absence. `analysis/roofline.py`, the exploratory first roofline written during
TEN-4716, is internal only and is not on the branch by design, and neither is
`analysis/validate_roofline.py`, which fits that roofline's constants against internal validation
data. Every command in this runbook that uses either file is marked INTERNAL at the command, and the
model of record for the campaign is the polaris copy `ttsim/perf/roofline_sdpa.py` of section 9, which
is on a pushed branch and is what every published number was checked against.

tt-metal-fresh, the main tip checkout. The TopK campaign of section 7 and the kernel state checks of
blocks T2.7 and R1e run there, not on the calibration tree. A worktree of the same clone is the
cheapest way to get it, and a second clone works identically:

```bash
cd $TTM && git worktree add $TTM_FRESH mvlahovic/analyze_sdpa_fresh
```

```bash
git clone --reference $TTM https://github.com/tenstorrent/tt-metal.git $TTM_FRESH
cd $TTM_FRESH && git checkout mvlahovic/analyze_sdpa_fresh
git submodule update --init --recursive
```

`mvlahovic/analyze_sdpa_fresh` is `origin/main` with two things added: the BRISC firmware size bump of
section 1.6, committed rather than left as a local edit, and the harnesses that run on main tip,
including `analysis/sparse_sweep_fresh.py`, which exists on no other branch and differs from
`sparse_sweep.py` only in the `kv_format` argument main tip requires. Verify both before the first
build:

```bash
cd $TTM_FRESH
git log --oneline -3
grep -n "define MEM_BRISC_FIRMWARE_SIZE" tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h
ls analysis/sparse_sweep_fresh.py
```

The firmware constant must read `(6 * 1024 + 2560)`, the same as `origin/main`. An earlier revision of
this branch carried `(6 * 1024 + 3072)`; that bump is no longer needed and breaks device init on a
profiler build, so if a workspace still shows the larger value, revert it (section 11.2).

Finally, confirm the three checkouts before anything touches the card:

```bash
for d in $TTM $TTM_FRESH $POLARIS; do echo "$d $(git -C $d branch --show-current) $(git -C $d rev-parse --short HEAD)"; done
```

### 1.2 The card and the host

| Item | Value | Source |
|---|---|---|
| Card | one Blackhole p100a, 11x10 = 110 worker cores | `bh/card_log.md` header |
| Clock | 1350 MHz (profiler sync reported 1.349959 GHz) | `bh/zone_decomposition.md` section 0 |
| Firmware bundle | 19.9.0 on every run of this campaign | printed in every run log, captured into each PROVENANCE line |
| Driver | TT-KMD 2.4.1 | `bh/fresh_build.md` header |
| Host | `$(hostname)`, your own reserved host; the reference campaign ran on one reserved host | `bh/fresh_build.md`, `bh/card_log.md` |

Reserve a Blackhole machine through ird. Every host string below is per reservation, so read your
own out of the environment rather than copying one:

```bash
hostname
hostname -f
```

One process per card. The campaign ran exactly one device process at a time; every runner script
deletes `generated/profiler/.logs/profile_log_device.csv` before starting and fails the run if the file
is missing afterwards, so two concurrent processes would overwrite each other's capture. The workspace
convention in `handoff/revamp/README.md` states that the single p100a is used only by the agent that
owns it and everyone else works read only.

Check no other process is using the card before a campaign. The campaign scripts do not run a device
health check of their own, so this is the only pre-flight step:

```bash
ps -ef | grep -E "pytest|tracy" | grep -v grep
```

### 1.3 The two tt-metal checkouts

Both live under `$WORK` and, when the second one is a worktree, share one `.git` object store. Both
branches are pushed.

| Checkout | Branch | HEAD | Role |
|---|---|---|---|
| `tt-metal` | `mvlahovic/sdpa_topk_harness` | `842f66d8cae` | calibration vehicle: kernels equal `aae92de5471` (the July calibration state, also reachable as `f803ede49f5`), plus the zone instrumentation, every measurement harness and the campaign scripts |
| `tt-metal-fresh` | `mvlahovic/analyze_sdpa_fresh` | `d546b027717` | main tip vehicle: `origin/main` `2dbd14bf632` merged in, plus the BRISC firmware size commit and the harnesses that run on main tip |

The calibration branch history, which shows what was added on top of the July calibration state. The
four commits whose subjects still read "never push" are the instrumentation and harness work as it was
developed; they are ancestors of the pushed branch, so the wording in their subjects is historical and
no longer applies:

```bash
cd $TTM
git log --oneline -6
git branch --show-current
```

```
842f66d8cae add the remaining sdpa measurement harnesses
396362c91af scratch: R1 re-measurement harnesses (regimes, non-paged decode, paged MLA decode, batch knob) (never push)
fa65028d281 scratch: paged decode calibration harness (never push)
c912ff95cdb scratch: non-streaming zone port, production harness knobs, model-level attention perf harness (never push)
b78b6aa6d12 scratch: SDPA zone instrumentation, ablation toggles, zone-tax and zone-sweep harness (revamp T0/T2; never push)
aae92de5471 Pin k_chunk explicitly in the SDPA sweep; add topk and indexer harnesses
```

The main tip branch history, which shows the two additions over `origin/main`:

```bash
cd $TTM_FRESH
git log --oneline -5
```

```
d546b027717 add the sdpa and topk measurement harnesses
b2c7f9de8d2 bump the blackhole brisc firmware size so a profiler build fits
622ff9e5d59 Merge origin/main into mvlahovic/analyze_sdpa_fresh
2dbd14bf632 [Bug fix] Fixing llk_pack_dummy (#56145)
78608b96a50 [Cleanup] Metal 2.0 port: experimental/paged_cache (#56022)
```

The instrumentation reached the calibration branch as commits, so there is nothing to apply on a fresh
clone. The recorded diff `bh/zone_patch.diff` is still in the handoff workspace and is still the way to
port the instrumentation onto a different base, for example a newer main tip; section 4.2 is that
route. It is `git diff 72620d5` of the instrumentation commits and touches exactly these files:

```
analysis/attn_decode_perf.py
analysis/attn_prefill_perf.py
analysis/decode_sweep.py
analysis/kernels/zone_tax_body.hpp
analysis/kernels/zone_tax_compute.cpp
analysis/kernels/zone_tax_dm.cpp
analysis/r1_decode.py
analysis/r1_mla_decode.py
analysis/r1_regimes.py
analysis/zone_reduce.py
analysis/zone_sweep.py
analysis/zone_tax.py
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zone_config.hpp
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zones.hpp
```

### 1.4 Where each harness lives

Every harness of section 3 is tracked, and the split between the two branches is the only thing to
know. Confirm it with:

```bash
cd $TTM
git ls-tree -r --name-only HEAD analysis/
git status --porcelain analysis/ | grep '^??'
```

Tracked on `mvlahovic/sdpa_topk_harness`, 28 entries under `analysis/`: `README.md`,
`attn_decode_perf.py`, `attn_prefill_perf.py`, `chunked_sweep.py`, `decode_latency_sweep.py`,
`decode_sweep.py`, `gap_sweep.py`, `indexer_probe.py`, `joint_sweep.py`,
`kernels/zone_tax_body.hpp`, `kernels/zone_tax_compute.cpp`, `kernels/zone_tax_dm.cpp`,
`make_charts.py`, `membound_test.py`, `mla_decode_latency_sweep.py`, `mla_perf_sweep.py`,
`p2_sweep.py`, `post_process.py`, `prefill_latency_sweep.py`, `r1_decode.py`, `r1_mla_decode.py`,
`r1_regimes.py`, `sdpa_sweep.py`, `sparse_sweep.py`, `topk_sweep.py`, `zone_reduce.py`,
`zone_sweep.py`, `zone_tax.py`. The runners, the campaign block scripts, the reducers and
`topk_campaign.py` are tracked beside them under `analysis/campaigns/` (sections 7 and 8).

Tracked on `mvlahovic/analyze_sdpa_fresh`, 19 entries: the subset that runs on main tip, which is the
same set minus the zone-dependent harnesses (`zone_tax.py` and its three kernels,
`attn_*_perf.py`, `decode_sweep.py`, `r1_*.py`), plus one file that exists on no other branch,
`analysis/sparse_sweep_fresh.py`. It is `sparse_sweep.py` with a single change, the `kv_format`
argument that main tip requires:

```
out = ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=scale,
```

Untracked on both branches, and untracked by design: `analysis/roofline.py` and
`analysis/validate_roofline.py`, the exploratory first roofline of TEN-4716 and its constant fit.
Both are internal only (section 1.1), so `git status --porcelain analysis/` reports them as `??` in a
working tree that has them and they are simply absent in a fresh clone. Commands that need them are
marked INTERNAL where they appear (sections 3.20 and 4.6).

The reports and the measured data are not in git at all; they are in the analysis workspace
`$HANDOFF` on the shared filesystem, for the reason section 1.1 gives: 2.4 GB of raw profiler dumps.
The figure and the offline model scripts are tracked under `analysis/campaigns/figs/` and
`analysis/campaigns/model/`, except two figure scripts that are internal and absent from this
branch.

### 1.5 The polaris checkout

```bash
cd $POLARIS
git branch --show-current
git log --oneline -3
```

The model work sits on `mvlahovic/roofline_model_topk` (current HEAD `b5afe33`, "annotate the topk
roofline for mypy", PR 530) with `mvlahovic/roofline_model_sdpa` as the SDPA-only predecessor
(`7aaecf22`, PR 493). Both are pushed, so `git ls-remote --heads origin` shows both at those shas.
Other campaign branches that exist in the development clone and are not needed to reproduce anything
here: `mvlahovic/sdpa_revamp`, `mvlahovic/sdpa_revamp_topk`, `mvlahovic/sdpa_revamp_topk2`,
`mvlahovic/sdpa_wall_recast`, `mvlahovic/topk_roofline`.

### 1.6 The profiler firmware region: leave it alone

On Blackhole main tip the profiler-enabled `brisc.elf` used to come out at 0x2204 bytes against a
0x2200 firmware region, so a profiler build failed to link BRISC firmware on first device run, and the
response was to bump one constant in `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h`. That
bump is gone from both measurement branches and must not come back:

```bash
cd $TTM        # and $TTM_FRESH
grep -n "define MEM_BRISC_FIRMWARE_SIZE" tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h
```

It must read `(6 * 1024 + 2560)`, the stock value. Nothing has to be patched here, and nothing should
be: the host HAL under `tt_metal/llrt/hal/tt-1xx/blackhole/` derives the NCRISC and TRISC bases from
this constant, so a header edit without a matching host rebuild leaves the two disagreeing and every
worker core times out in firmware init on a profiler run (section 11.2). An earlier revision of these
branches carried `(6 * 1024 + 3072)` and that is what the symptom looks like when it survives a
workspace copy. If you see the larger value:

```bash
sed -i 's/(6 \* 1024 + 3072)/(6 * 1024 + 2560)/' tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h
```

Whether the revert needs a rebuild depends on when the tree's host library was built. A library built
before the bump already expects the stock bases and nothing has to be relinked; a library built while the
bump was in place has the shifted bases compiled in, and reverting the header alone leaves the same
mismatch in the other direction, which looks like a device that stops answering the profiler sync and
then `Read 0xffffffff over PCIe: the board should be reset`. Compare the header's commit date with
`build/lib/libtt_metal.so`; if the library is older than the revert, run `cmake --build build` and then
`cmake --build build --target install` (the install step is what refreshes `build/lib/`), and reset the
card once with `tt-smi -r` if a run already hung on it. The July note recorded in `bh/fresh_build.md` is that the bump
does not change compute-kernel cycle counts. The calibration checkout `tt-metal` does not need the
patch: its kernels and firmware predate the firmware growth on main.

### 1.7 Python environments

There is exactly one tt-metal virtual environment, in the calibration checkout, and the fresh tree
reuses it through `PYTHONPATH`:

```bash
ls -d $TTM/python_env
$TTM/python_env/bin/python --version
```

That reports Python 3.10.19. `tt-metal-fresh` has no `python_env` of its own. Calibration checkout
activation:

```bash
export TT_METAL_HOME=$TTM
export ARCH_NAME=blackhole
export TT_METAL_FORCE_JIT_COMPILE=1
cd $TT_METAL_HOME && source python_env/bin/activate
```

Fresh tree activation. The baseline `python_env` is an editable install whose finder points at the
baseline paths, but `PYTHONPATH` entries win, so pointing `PYTHONPATH` at the fresh tree is enough:

```bash
export TT_METAL_HOME=$TTM_FRESH
export ARCH_NAME=blackhole
export PYTHONPATH=$TT_METAL_HOME:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools
source $TTM/python_env/bin/activate
cd $TT_METAL_HOME
```

Verify the fresh tree is the one loaded before touching the card:

```bash
python -c "import ttnn, tracy; print(ttnn.__file__); print(tracy.__file__)"
python -m tracy --help | grep -E "perf-counter|capture-perf-counters"
```

If the venv ever has to be rebuilt for a tree:

```bash
cd $TT_METAL_HOME && ./create_venv.sh --env-dir $TT_METAL_HOME/python_env
```

That is uv based and costs about 3.3 GB.


## 2. Building tt-metal with the profiler

The Tracy profiler is on by default in `build_metal.sh` (the help text reads
"--disable-profiler Disable Tracy profiler (enabled by default)"), so the campaign build is the plain
default build:

```bash
cd $TTM_FRESH
git submodule update --init --recursive
./build_metal.sh
```

Effective options, verified in `build_Release/CMakeCache.txt` and identical between the two trees:
`CMAKE_BUILD_TYPE=Release`, `ENABLE_TRACY=ON`, `TRACY_DEBUG_CATEGORY=off`, `TT_UNITY_BUILDS=ON`,
`TT_ENABLE_LIGHT_METAL_TRACE=ON`, `ENABLE_DISTRIBUTED=ON`, `WITH_PYTHON_BINDINGS=ON`,
`ENABLE_CCACHE=OFF`, tests off, toolchain
`cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake` (clang-20), Ninja generator, install prefix
`build_Release` with a `build -> build_Release` symlink.

Confirm the two knobs that matter:

```bash
grep -E "^ENABLE_TRACY|^CMAKE_BUILD_TYPE" $TTM_FRESH/build_Release/CMakeCache.txt
```

Expected artifacts after a clean build (1434 ninja steps in `bh/fresh_build.log`):

```bash
ls -l $TTM_FRESH/build_Release/lib/libtt_metal.so
ls -l $TTM_FRESH/build_Release/lib/_ttnn.so
ls -l $TTM_FRESH/ttnn/ttnn/_ttnn.so
ls $TTM_FRESH/build_Release/tools/profiler/bin/
```

The last directory must hold `tracy-capture`, `tracy-capture-daemon` and `tracy-csvexport`.

Apply the BRISC patch of section 1.6 before the first device run, then rebuild. Kernel binaries and
device firmware are JIT built on first device use; `TT_METAL_FORCE_JIT_COMPILE=1` forces kernel
rebuilds on every process (section 11.1).

Disk cost recorded for the fresh tree: `build_Release` 2.3 GB, `.cpmcache` 3.6 GB, sources and
submodules 1.1 GB, about 7.0 GB total. The calibration tree adds `python_env` 3.3 GB and a shared
`.git` of 1.8 GB.

Smallest smoke configs, from `bh/fresh_build.md`, to catch the firmware overflow and API drift before
a grid:

```bash
cd $TTM_FRESH
SDPA_SEQ_LENS=1024 SDPA_NH=4 SDPA_NKV=1 SDPA_SKIP_REF=1 python -m tracy -p -- pytest analysis/sdpa_sweep.py::test_sweep -s
TK_NS=256 TK_KS=16 TK_ITERS=1 python -m tracy -p -- pytest analysis/topk_sweep.py -s
```


## 3. The SDPA sweep harnesses

Every harness is a pytest module under `analysis/` in a tt-metal checkout. All take their
configuration from environment variables so that one process runs one point and the device profiler
CSV carries one run host ID per op invocation. None takes command line arguments; the pytest node id
selects the test and, for the parametrized harnesses, `-k` selects a parameter value.

Common discipline across the campaign, from `bh/campaign_r1.md` and `bh/zone_decomposition.md`:
three op invocations in one process on the same tensors, invocation 0 discarded, mean of invocations 1
and 2 reported. Cycles are at 1.35 GHz.

The generic wrappers `run_zone.sh`, `run_regime.sh`, `run_decode.sh`, `run_fresh.sh` and
`run_regime_fresh.sh` (section 8.1) set `TT_METAL_HOME`, `ARCH_NAME`, `TT_METAL_FORCE_JIT_COMPILE=1`,
activate the venv, run the harness under tracy, write a PROVENANCE line, and call the reducer. The
worked examples below show both the wrapper form and the bare tracy form as it appears in
`bh/card_log.md`.

### 3.1 analysis/zone_sweep.py (prefill anchor and hold-outs)

Test: `test_zone_sweep`. Mirrors `analysis/sdpa_sweep.py::_run_sdpa_op_only` exactly, with `k_chunk`
pinned explicitly. Calls `ttnn.transformer.scaled_dot_product_attention` on DRAM interleaved
`fa_rand` tensors from `tests.ttnn.unit_tests.operations.sdpa.test_sdpa_prefill`.

| Env var | Default | Meaning |
|---|---|---|
| `SDPA_SEQ` | `4096` | query and KV sequence length S |
| `SDPA_NH` | `32` | query heads |
| `SDPA_NKV` | `8` | KV heads (GQA) |
| `SDPA_HEAD_DIM` | `128` | head dim |
| `SDPA_QCHUNK` | `128` | `SDPAProgramConfig.q_chunk_size` |
| `SDPA_KCHUNK` | value of `SDPA_QCHUNK` | `SDPAProgramConfig.k_chunk_size` |
| `SDPA_CAUSAL` | `1` | `1` sets `is_causal=True` |
| `SDPA_ITERS` | `3` | op invocations in the process |
| `SDPA_DTYPE` | `bfp8_b` | Q dtype and output dtype; accepts `bfp8_b` or `bfloat16` |
| `SDPA_KV_DTYPE` | value of `SDPA_DTYPE` | K and V dtype override (used for the DRAM law runs) |
| `SDPA_FIDELITY` | `HiFi2` | `LoFi`, `HiFi2`, `HiFi3` or `HiFi4` |
| `SDPA_EXP_APPROX` | `1` | `SDPAProgramConfig.exp_approx_mode` |
| `SDPA_GRID` | unset (device grid, 11x10 = 110 cores) | `GXxGY`, for example `8x8` for the 64-core production grid |
| `SDPA_FP32_ACC` | `0` | `fp32_dest_acc_en`; `1` selects the non-streaming `compute_common.hpp` path |
| `SDPA_PACKER_L1_ACC` | `0` | `packer_l1_acc` in the compute config (the SDPA program factory does not read it, see section 11) |
| `SDPA_MATH_APPROX` | `1` | `math_approx_mode` |
| `SDPA_BATCH` | `1` | batch dimension of Q, K and V (hold-out axis added for R1c) |

Worked example, the causal anchor:

```bash
cd $TTM/analysis/campaigns
SDPA_CAUSAL=1 SDPA_SEQ=4096 SDPA_QCHUNK=128 SDPA_KCHUNK=128 ./run_zone.sh t21_causal_q128k128_zoff
```

The bare form the wrapper runs:

```bash
cd $TTM
export TT_METAL_HOME=$PWD ARCH_NAME=blackhole TT_METAL_FORCE_JIT_COMPILE=1
source python_env/bin/activate
SDPA_CAUSAL=1 SDPA_SEQ=4096 SDPA_QCHUNK=128 SDPA_KCHUNK=128 \
    python -m tracy -m pytest analysis/zone_sweep.py::test_zone_sweep -s
```

The production Llama configuration (T2.4), which selects the non-streaming kernel:

```bash
cd $TTM/analysis/campaigns
SDPA_CAUSAL=1 SDPA_FIDELITY=HiFi4 SDPA_EXP_APPROX=0 SDPA_MATH_APPROX=0 SDPA_FP32_ACC=1 \
    SDPA_PACKER_L1_ACC=1 SDPA_DTYPE=bfloat16 SDPA_KV_DTYPE=bfloat8_b SDPA_GRID=8x8 \
    SDPA_SEQ=4096 SDPA_QCHUNK=256 SDPA_KCHUNK=256 ./run_zone.sh t24_prod_causal_S4096_q256k256_g64_zoff
```

### 3.2 analysis/zone_tax.py (cost of a zone boundary)

Test: `test_zone_tax`. Launches `analysis/kernels/zone_tax_dm.cpp` (reader slot NCRISC and writer slot
BRISC) and `analysis/kernels/zone_tax_compute.cpp` (TRISC0, TRISC1, TRISC2 all run the same loop)
through `ttnn.generic_op` on a small core grid. One program per `(mode, n, loop)` triple. The KERNEL
zone duration per RISC divided by `n`, minus the mode-0 baseline at the same `n`, is the per-zone cost.

| Env var | Default | Meaning |
|---|---|---|
| `ZT_GRID` | `0:2000:16,3:2000:16,2:2000:16,0:100:16,1:100:16` | comma list of `mode:n:loop`. mode 0 baseline, 1 raw `DeviceZoneScopedN`, 2 native `DeviceZoneScopedSumN1`, 3 custom `SDPA_ZACC`. `n` = zone repetitions, `loop` = nop body length |
| `ZT_CORES` | `1x1` | core rectangle from (0,0) |

Worked example, the two runs of the campaign:

```bash
cd $TTM/analysis/campaigns
./run_zone_tax.sh t02_zone_tax_1x1 "0:2000:16,3:2000:16,2:2000:16,0:100:16,1:100:16,0:2000:64,3:2000:64,2:2000:64,0:100:64,1:100:64,1:1000:16" 1x1
./run_zone_tax.sh t02_zone_tax_4x4 "0:2000:16,3:2000:16,2:2000:16,0:100:16,1:100:16" 4x4
```

The bare form, which needs sum profiling so that the native `SumN` mode records anything:

```bash
cd $TTM
ZT_GRID="0:2000:16,3:2000:16,2:2000:16,0:100:16,1:100:16" ZT_CORES=1x1 \
    python -m tracy --enable-sum-profiling -m pytest analysis/zone_tax.py::test_zone_tax -s
```

`--enable-sum-profiling` compiles kernels with `-DPROFILE_KERNEL=9` (base plus the DO_SUM option)
instead of `-DPROFILE_KERNEL=1`. The runner prints the observed `-DPROFILE_KERNEL=` values from the
build log as a check.

### 3.3 analysis/sdpa_sweep.py (the July and September single-op sweep form)

Test: `test_sweep`, parametrized over sequence lengths. This is the harness the earlier calibration
used; the campaign reruns it only to reproduce the earlier harness form exactly (the T0.1 diagnostic).

| Env var | Default | Meaning |
|---|---|---|
| `SDPA_SEQ_LENS` | `1024,2048,4096,8192,16384,32768` | comma list; becomes the pytest parametrization |
| `SDPA_HEAD_DIM` | `0` meaning the Llama 8B head dim | head dim override |
| `SDPA_NH` | `0` meaning the module default | query heads override |
| `SDPA_NKV` | `0` meaning the module default | KV heads override |
| `SDPA_QCHUNK` | Llama 8B q chunk | q chunk |
| `SDPA_KCHUNK` | value of `SDPA_QCHUNK` | k chunk |
| `SDPA_CAUSAL` | `1` | causal |
| `SDPA_DTYPE` | `bfp8_b` | `bfp8_b` or `bfloat16` |
| `SDPA_FIDELITY` | `HiFi2` | math fidelity |
| `SDPA_EXP_APPROX` | `1` | exp approx mode |
| `SDPA_SKIP_REF` | `0` | `1` skips the torch reference check (needed at long S) |

Worked example, the T0.1 harness-form diagnostic (card log row 5):

```bash
cd $TTM
SDPA_SKIP_REF=1 SDPA_SEQ_LENS=1024,4096 SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 \
    python -m tracy -m pytest analysis/sdpa_sweep.py::test_sweep -s
```

### 3.4 analysis/r1_regimes.py (cross, windowed, dense mask in one harness)

Test: `test_r1_regime`. Written for block R1a to reproduce the `p2_sweep.py` forms with one point per
process. `nh = nkv` (multi-head, not GQA).

| Env var | Default | Meaning |
|---|---|---|
| `R1_MODE` | `cross` | `cross`, `window` or `mask` |
| `R1_NH` | `16` | heads (query and KV) |
| `R1_D` | `128` | head dim |
| `R1_QCHUNK` | `128` | q chunk; the harness sets `k_chunk_size` to the same value, so there is no separate k knob |
| `R1_ITERS` | `3` | invocations |
| `R1_SQ` | required for `cross` | query length |
| `R1_SK` | required for `cross` | KV length |
| `R1_S` | required for `window` and `mask` | sequence length |
| `R1_W` | required for `window` | window width in tokens |
| `R1_DENSITY` | `0.25` (`mask` mode) | Bernoulli density of the additive mask |

Worked example, the cross point of R1a:

```bash
cd $TTM/analysis/campaigns
R1_MODE=cross R1_SQ=1024 R1_SK=8192 R1_NH=16 R1_D=128 R1_QCHUNK=128 R1_ITERS=3 \
    ./run_regime.sh r1a_cross_1024_8192_nh16_zoff plain "analysis/r1_regimes.py::test_r1_regime"
```

### 3.5 analysis/p2_sweep.py (the July regime harness: window, mask, cross)

Tests: `test_p2_window`, `test_p2_mask`, `test_p2_cross`. Each test body returns early unless
`P2_MODE` selects it, so one process runs one regime. Used for the T2.3 regime table.

| Env var | Default | Meaning |
|---|---|---|
| `P2_MODE` | `window` | selects which test body runs: `window`, `mask` or `cross` |
| `P2_SEQ` | `8192,16384` | sequence lengths (parametrization for window and mask) |
| `P2_WIN` | `1024,4096` | window widths (window mode) |
| `P2_DENSITIES` | `0.25` | mask densities (mask mode) |
| `P2_NH` | `16` | heads |
| `P2_D` | `128` | head dim |
| `P2_QCHUNK` | `128` | q chunk |
| `P2_ITERS` | `2` | invocations |

Worked example, cross at (2048, 8192) with a `-k` filter selecting the parametrization:

```bash
cd $TTM/analysis/campaigns
P2_MODE=cross P2_NH=16 P2_D=128 P2_QCHUNK=128 P2_ITERS=3 \
    ./run_regime.sh t23r_cross_2048_8192_nh16_zoff plain "analysis/p2_sweep.py::test_p2_cross -k 2048-8192"
```

### 3.6 analysis/chunked_sweep.py (chunked and paged prefill)

Test: `test_chunked_prefill`, parametrized over `CK_STARTS`. Uses a paged cache with an identity page
table.

| Env var | Default | Meaning |
|---|---|---|
| `CK_STARTS` | `4096` | comma list of `chunk_start_idx` values; becomes the parametrization |
| `CK_S` | `8192` | total paged cache length |
| `CK_SQ` | `2048` | query length of this chunk |
| `CK_NH` | `16` | query heads |
| `CK_NKV` | `16` | KV heads |
| `CK_D` | `128` | head dim |
| `CK_ITERS` | `2` | invocations |

Worked example:

```bash
cd $TTM/analysis/campaigns
CK_STARTS=2048 CK_S=8192 CK_SQ=2048 CK_ITERS=3 \
    ./run_regime.sh r1a_chunked_start2048_S8192_Sq2048_zoff plain "analysis/chunked_sweep.py::test_chunked_prefill"
```

### 3.7 analysis/mla_perf_sweep.py (FlashMLA prefill)

Test: `test_mla_sweep`, parametrized over `MLA_SEQ`. Asymmetric MLA dims: `d_qk = kv_lora + d_rope`,
`d_v = kv_lora`.

| Env var | Default | Meaning |
|---|---|---|
| `MLA_SEQ` | `1024,2048,4096,8192` | comma list of sequence lengths (parametrization) |
| `MLA_NH` | `16` | query heads |
| `MLA_NKV` | `1` | latent KV heads |
| `MLA_KVLORA` | `512` | latent dim (also `d_v`) |
| `MLA_DROPE` | `64` | rope dim added to `d_qk` |
| `MLA_ITERS` | `2` | invocations |

Worked example:

```bash
cd $TTM/analysis/campaigns
MLA_SEQ=4096 MLA_NH=16 MLA_NKV=1 MLA_KVLORA=512 MLA_DROPE=64 MLA_ITERS=3 \
    ./run_regime.sh r1a_mla_nh16_S4096_zoff plain "analysis/mla_perf_sweep.py::test_mla_sweep"
```

### 3.8 analysis/sparse_sweep.py and sparse_sweep_fresh.py (sparse MLA SDPA)

Test: `test_sparse`, parametrized over `SP_TOPKS`. Calls `ttnn.transformer.sparse_sdpa`: each of S
queries attends its top-k selected latent KV entries.

| Env var | Default | Meaning |
|---|---|---|
| `SP_TOPKS` | `512,1024,2048` | comma list of top-k values (parametrization) |
| `SP_H` | `16` | heads |
| `SP_S` | `2048` | query length |
| `SP_T` | `8192` | latent cache length |
| `SP_KDIM` | `576` | score dim |
| `SP_VDIM` | `512` | output dim |
| `SP_KC` | `128` | k chunk |
| `SP_ITERS` | `2` | invocations |

Worked example on the calibration tree and on the fresh tree (the fresh tree needs
`sparse_sweep_fresh.py` because main tip requires the `kv_format` argument):

```bash
cd $TTM/analysis/campaigns
SP_H=32 SP_S=2048 SP_T=8192 SP_TOPKS=2048 SP_KC=128 SP_ITERS=3 \
    ./run_regime.sh r1a_sparse_H32_S2048_T8192_topk2048_zoff plain "analysis/sparse_sweep.py::test_sparse"
SP_H=32 SP_S=2048 SP_T=8192 SP_TOPKS=2048 SP_KC=128 SP_ITERS=3 \
    ./run_regime_fresh.sh r1e_fresh_sparse_H32_S2048_T8192_topk2048_zoff plain "analysis/sparse_sweep.py::test_sparse"
```

The R1e fresh run in `campaign_r1.sh` targets `analysis/sparse_sweep.py`; on the fresh tree the
runnable module is `analysis/sparse_sweep_fresh.py::test_sparse`, which differs only in the
`kv_format=ttnn.transformer.SparseKVFormat.BF16` argument. Use the `_fresh` module when the plain one
raises on the missing argument.

### 3.9 analysis/joint_sweep.py (joint attention, SD3 and Flux shapes)

Test: `test_joint_sweep`, parametrized over `JT_SEQS`. Calls
`joint_scaled_dot_product_attention` with `joint_strategy` rear.

| Env var | Default | Meaning |
|---|---|---|
| `JT_SEQS` | `2048,4096,8192` | comma list of sequence lengths (parametrization) |
| `JT_NH` | `16` | heads |
| `JT_D` | `128` | head dim |
| `JT_JOINT` | `512` | joint stream length inside the sequence |
| `JT_QC` | `128` | q chunk |
| `JT_KC` | `128` | k chunk |
| `JT_ITERS` | `40` | invocations (the campaign used 1, giving 3 warm-ups plus 1 = 4 invocations) |

Worked example:

```bash
cd $TTM/analysis/campaigns
JT_SEQS=4096 JT_JOINT=333 JT_NH=24 JT_D=128 JT_QC=128 JT_KC=512 JT_ITERS=1 \
    ./run_regime.sh r1a_joint_N4096_L333_nh24_d128_q128k512_zoff plain "analysis/joint_sweep.py::test_joint_sweep"
```

### 3.10 analysis/decode_sweep.py (paged SDPA decode, the tt_transformers form)

Test: `test_decode_sweep`. This is the T2.8 calibration harness: batch users, paged KV cache with a
page block, one position sweep in one process.

| Env var | Default | Meaning |
|---|---|---|
| `DEC_B` | `32` | batch (users) |
| `DEC_NH` | `32` | query heads |
| `DEC_NKV` | `8` | KV heads |
| `DEC_D` | `128` | head dim |
| `DEC_BLOCK` | `32` | paged block size in tokens |
| `DEC_POS` | `1024` | comma list of cache positions to decode at |
| `DEC_MAXSEQ` | `max(8192, 2 * max(DEC_POS))` | page table sizing |
| `DEC_ITERS` | `3` | invocations per position |
| `DEC_KV_DTYPE` | `bfp8_b` | `bfp8_b` or `bfloat16` |
| `DEC_GRID` | `8x8` | core grid |

Worked example, the main decode sweep of T2.8:

```bash
cd $TTM/analysis/campaigns
DEC_B=32 DEC_POS=128,512,1024,2048,4096,8192 DEC_MAXSEQ=16384 DEC_GRID=8x8 \
    ./run_decode.sh t28_decode_b32_g64_pos128to8192
```

`run_decode.sh` runs tracy with `-r` so that the ops report carries `DEVICE KERNEL DURATION` per
invocation, which is the decode basis (section 6).

### 3.11 analysis/r1_decode.py (non-paged decode, the model's constant form)

Test: `test_r1_decode`. Reproduces the `decode_latency_sweep.py` form: Q `[1, b, nh, d]` bf16, K and V
`[b, nkv, cache, d]`, `cur_pos = cache - 1`, full grid, output in DRAM.

| Env var | Default | Meaning |
|---|---|---|
| `DEC_B` | `8` | batch |
| `DEC_NH` | `32` | query heads |
| `DEC_NKV` | `8` | KV heads |
| `DEC_D` | `128` | head dim |
| `DEC_POS` | `1024` | comma list of cache lengths |
| `DEC_ITERS` | `3` | invocations per point |
| `DEC_KV_DTYPE` | `bfloat16` | `bfp8_b` or `bfloat16`; note the default differs from `decode_sweep.py` |

Worked example:

```bash
cd $TTM/analysis/campaigns
DEC_B=32 DEC_KV_DTYPE=bfp8_b DEC_POS=1024,4096 DEC_NH=32 DEC_NKV=8 DEC_D=128 DEC_ITERS=3 \
    ./run_regime.sh r1b_decode_nonpaged_b32_kvbfp8_pos1024_4096_g110_zoff plain "analysis/r1_decode.py::test_r1_decode"
```

### 3.12 analysis/r1_mla_decode.py (paged MLA decode)

Test: `test_r1_mla_decode`. Reproduces the `test_mla_decode.py` row 1 geometry: Q height-sharded on
`MLAD_QCORES` cores, KV bfp8 paged with block 64, `reuse_k` so V is not passed.

| Env var | Default | Meaning |
|---|---|---|
| `MLAD_B` | `4` | batch |
| `MLAD_NH` | `128` | query heads |
| `MLAD_KVLORA` | `512` | latent dim |
| `MLAD_DROPE` | `64` | rope dim |
| `MLAD_QCORES` | `64` | cores the sharded query occupies |
| `MLAD_ITERS` | `3` | invocations per position |
| `MLAD_POS` | `1024` | comma list of cache positions |
| `MLAD_CACHE` | `max(2 * max(MLAD_POS), 2048)` | allocated cache length |

Worked example:

```bash
cd $TTM/analysis/campaigns
MLAD_POS=1024,4096,8192 MLAD_CACHE=16384 MLAD_B=4 MLAD_NH=128 MLAD_KVLORA=512 MLAD_DROPE=64 \
    MLAD_QCORES=64 MLAD_ITERS=3 \
    ./run_regime.sh r1b_mla_decode_paged_b4_nh128_pos1024_4096_8192_zoff plain "analysis/r1_mla_decode.py::test_r1_mla_decode"
```

### 3.13 analysis/decode_latency_sweep.py (non-paged decode, many-iteration latency form)

Test: `test_decode_latency`. The July latency harness. Note it uses `DEC_SEQS`, not `DEC_POS`.

| Env var | Default | Meaning |
|---|---|---|
| `DEC_B` | `8` | batch |
| `DEC_NH` | `32` | query heads |
| `DEC_NKV` | `8` | KV heads |
| `DEC_D` | `128` | head dim |
| `DEC_SEQS` | `1024,2048,4096,8192,16384` | comma list of cache lengths |
| `DEC_ITERS` | `80` | invocations per point |

```bash
cd $TTM
DEC_B=8 DEC_SEQS=1024,4096 DEC_ITERS=80 python -m tracy -r -m pytest analysis/decode_latency_sweep.py::test_decode_latency -s
```

### 3.14 analysis/mla_decode_latency_sweep.py (non-paged MLA decode)

Test: `test_mla_decode_latency`. The model's non-paged MLA decode form: `cur_pos = cache - 1`, K in
bf16 with V a slice of K.

| Env var | Default | Meaning |
|---|---|---|
| `MLAD_B` | `1` | batch |
| `MLAD_NH` | `16` | query heads |
| `MLAD_KVLORA` | `512` | latent dim |
| `MLAD_DROPE` | `64` | rope dim |
| `MLAD_SEQS` | `1024,2048,4096,8192,16384` | comma list of cache lengths |
| `MLAD_ITERS` | `80` | timed invocations; the harness adds 3 warm-ups, so the CSV holds `3 + MLAD_ITERS` invocations per point |

Worked example, the R1b addendum:

```bash
cd $TTM/analysis/campaigns
MLAD_B=8 MLAD_NH=16 MLAD_KVLORA=512 MLAD_DROPE=64 MLAD_SEQS=1024,4096,8192 MLAD_ITERS=1 \
    ./run_regime.sh r1b_mla_decode_nonpaged_b8_nh16_kvbf16_cache1024_4096_8192_zoff plain \
    "analysis/mla_decode_latency_sweep.py::test_mla_decode_latency"
```

### 3.15 analysis/prefill_latency_sweep.py

Test: `test_prefill_latency`, parametrized over `PF_SEQ`. Many-iteration prefill latency, head dim
fixed at 128 in the source.

| Env var | Default | Meaning |
|---|---|---|
| `PF_SEQ` | `2048,4096,8192,16384` | comma list of sequence lengths (parametrization) |
| `PF_NH` | `32` | query heads |
| `PF_NKV` | `8` | KV heads |
| `PF_CAUSAL` | `1` | causal |
| `PF_ITERS` | `50` | invocations per point |

```bash
cd $TTM
PF_SEQ=4096 PF_ITERS=50 python -m tracy -r -m pytest analysis/prefill_latency_sweep.py::test_prefill_latency -s
```

### 3.16 analysis/gap_sweep.py (head dim, batch and attention sink axes)

Tests: `test_gap_headdim`, `test_gap_batch`, `test_gap_sink`. Each body returns early unless
`GAP_MODE` selects it.

| Env var | Default | Meaning |
|---|---|---|
| `GAP_MODE` | unset (every body skips) | `headdim`, `batch` or `sink` |
| `GAP_HEADDIMS` | `64,72,96,160,256` | head dims (parametrization of `test_gap_headdim`) |
| `GAP_BATCHES` | `1,2,4` | batches (parametrization of `test_gap_batch`) |
| `GAP_NH` | `16` | heads |
| `GAP_S` | `4096` for headdim and sink, `2048` for batch | sequence length |
| `GAP_ITERS` | `2` | invocations per point |

```bash
cd $TTM
GAP_MODE=headdim GAP_HEADDIMS=64,96 GAP_NH=16 GAP_S=4096 GAP_ITERS=3 \
    python -m tracy -m pytest analysis/gap_sweep.py::test_gap_headdim -s
```

### 3.17 analysis/membound_test.py (DRAM and matmul bandwidth reference)

Tests: `test_membound` (parametrized over `MEMBOUND_N`) and `test_membound_matmul` (parametrized over
`MEMBOUND_MM_N`). Provides the memory-bound reference the DRAM law is compared against.

| Env var | Default | Meaning |
|---|---|---|
| `MEMBOUND_N` | `8192` | element count or square side for the eltwise membound test |
| `MEMBOUND_MM_N` | `8192` | square side for the matmul test |

```bash
cd $TTM
MEMBOUND_N=8192 python -m tracy -m pytest analysis/membound_test.py::test_membound -s
```

### 3.18 analysis/topk_sweep.py (the standalone TopK grid)

Test: `test_topk`, parametrized over the cross product of `TK_NS` and `TK_KS`. This is the earlier
TopK harness; the September campaign uses `topk_campaign.py` instead (section 7), but the harness is
still the quickest TopK smoke test.

| Env var | Default | Meaning |
|---|---|---|
| `TK_NS` | `256,512,1024,4096,5000,5300,16384,65536,131072` | row lengths N |
| `TK_KS` | `16,32,64,256,512,1024,2048` | k values |
| `TK_ROWS` | `1` | rows |
| `TK_ITERS` | `6` | invocations per cell |
| `TK_MEM` | `dram` | input memory: `dram` or L1 |

```bash
cd $TTM_FRESH
TK_NS=16384 TK_KS=512 TK_ITERS=6 python -m tracy -r -p -m pytest analysis/topk_sweep.py::test_topk -s
```

### 3.19 analysis/indexer_probe.py (indexer_score_dsa)

Test: `test_indexer_probe`. Calls `ttnn.experimental.indexer_score_dsa` with
`chunk_start_idx = T - Sq`, bf16 TILE tensors.

| Env var | Default | Meaning |
|---|---|---|
| `IX_HEADS` | `64` | indexer heads Hi |
| `IX_D` | `128` | head dim D |
| `IX_SQ` | `2048` | query length Sq |
| `IX_T` | `8192` | key length T |
| `IX_ITERS` | `4` | invocations |

Worked example, the three plan points of T2.9a, which had to run on the calibration checkout because
the main-tip op asserts on the probe's weights layout (section 11.5):

```bash
cd $TTM
export TT_METAL_HOME=$PWD ARCH_NAME=blackhole TT_METAL_FORCE_JIT_COMPILE=1
source python_env/bin/activate
IX_SQ=2048 IX_T=8192 IX_ITERS=3 IX_HEADS=64 IX_D=128 \
    python -m tracy -r -m pytest analysis/indexer_probe.py::test_indexer_probe -s
```

### 3.20 analysis/roofline.py (INTERNAL), validate_roofline.py (INTERNAL) and make_charts.py

These three are host-side analysis tools in the same directory, not device harnesses. They carry no
environment variables. The first two are INTERNAL and are not on either pushed branch by design
(section 1.1), so the two commands that use them can only be run in a working tree that already has
those files; `make_charts.py` is tracked and runs anywhere.

`roofline.py` (INTERNAL) is the exploratory first roofline written during TEN-4716. It is imported,
not run:
`predict(SdpaConfig(...))` returns a `RooflineResult`, `sdpa_perf_stats(cfg)` emits ttsim perf stats,
and `sdpa_config_from_shapes(q_shape, k_shape, v_shape, attrs=None, num_cores=110, arch=None)` builds
a config from tensor shapes. `analysis/campaigns/zone_decomp.py` imports it, when it is present, to
compute the model floor column of each decomposition. The model of record for the campaign is the
polaris copy (`ttsim/perf/roofline_sdpa.py`, section 9), which is on a pushed branch and is what every
published number was checked against; this internal copy existed only so that the decomposition
scripts could run without the polaris venv.

```bash
# INTERNAL: needs analysis/roofline.py, which is not on the pushed branch
cd $TTM
python -c "import sys; sys.path.insert(0, 'analysis'); from roofline import SdpaConfig, predict; print(predict(SdpaConfig(S=4096, q_chunk=128, k_chunk=128, num_heads=32, num_kv_heads=8, num_cores=110, is_causal=True)).wall_clock_cycles)"
```

`validate_roofline.py` (INTERNAL) fits that roofline's constants by least squares over the earlier measured matrix
and runs a leave-one-axis-out hold-out. It reads a measured-components JSON that is internal and
stays in the analysis workspace, and writes the fitted constants next to it. The axes it knows are
`baseline`, `hifi4`, `expacc` and `noncausal` at S in 1024 to 32768.

```bash
# INTERNAL: needs analysis/validate_roofline.py and its validation-data inputs, neither of which is in git
cd $TTM
python analysis/validate_roofline.py
```

Both the script and its input are internal: the script is not on either branch, and its input is
earlier validation data that stays in the analysis workspace on the shared filesystem.

`make_charts.py` draws the four TEN-4679 counter charts from a `post_process.py` results CSV.

```bash
cd $TTM
python analysis/make_charts.py analysis/ten4679_results_rerun_2026-07-22_nh32.csv -o /tmp/charts --exp-share 0.5
```


## 4. Zone instrumentation and the ablations

### 4.1 What the instrumentation is

Two zone kinds are added to the SDPA kernels on `mvlahovic/sdpa_topk_harness`, both compiled out
entirely when `SDPA_ZONES` is 0:

- Accumulate zones, `SDPA_ZACC(slot)` in
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zones.hpp`. An RAII scope that reads
  the 64-bit wall clock at entry and exit (`RISCV_DEBUG_REG_WALL_CLOCK_L` high and low words, the
  register `kernel_profiler` itself uses) and adds the difference to `sums[slot]`, incrementing
  `cnts[slot]`. 24 slots per RISC. Nothing reaches L1 until kernel end, where `SDPA_ZFLUSH(slot,
  "NAME")` emits two `TS_DATA` profiler records per slot, `NAME` for the summed cycles and `NAME_N`
  for the occurrence count. The compute kernel source runs on all three TRISCs, so every compute zone
  yields three records per core: the TRISC_0 record is where `cb_wait_front` spins (UNPACK), the
  TRISC_2 record where packer `STALLWAIT` and the pack-thread SFPU work sit (PACK), and the TRISC_1
  record the math thread's own view (MATH).
- Raw zones, `SDPA_ZRAW(name)` = a plain `DeviceZoneScopedN`, placed once per q chunk on every thread
  (`QCHUNK` on compute, `R_QCHUNK` on the reader, `W_QCHUNK` on the writer). These give the
  per-q-chunk timeline, the kernel prologue and the tail.

Native `DeviceZoneScopedSumN1/2` was not used because it offers only two slots per RISC and no
occurrence count.

The compute slot map (23 slots after the T2.1 addition of `EXP_INIT` and `PUSHES`), from the header of
`analysis/campaigns/apply_zone_patch.py`:

```
 0 STEP        1 K_WAIT      2 Q_WAIT      3 RESERVE_QKT  4 RECONFIG   5 SUBEXP   6 EXP
 7 QK_MM       8 MASK        9 REDUCE     10 OUT_RESERVE 11 QKTIM_WAIT 12 V_WAIT 13 PV_MM
14 PACK_DONE  15 SALAD_EXP  16 SALAD_CORR 17 NORM        18 PUSH_HOLD  19 POPS   20 MASK_DIAG
21 EXP_INIT   22 PUSHES
reader (NCRISC): 0 R_K_READ  1 R_V_READ  2 R_Q_READ  3 R_RESERVE  4 R_BARRIER  5 R_KCHUNK
writer (BRISC):  0 W_WAIT    1 W_DRAIN
raw per q chunk: QCHUNK (compute), R_QCHUNK (reader), W_QCHUNK (writer)
```

The per-slot source lines after patching are tabulated in `bh/zone_decomposition.md` section 2.2,
which also records that `EXP` is nested inside `SUBEXP` and `STEP` is the envelope of one k-chunk
step, so those three must not be added to the other parts.

### 4.2 Applying the patch, and porting it to another base

A checkout of `mvlahovic/sdpa_topk_harness` already has the instrumentation, so nothing has to be
applied to reproduce the campaign. This subsection is for putting the same instrumentation on a
different base, a newer main tip being the case that will come up.

Preferred route, the recorded diff, which needs the internal copy of the analysis workspace:

```bash
cd $TTM
git apply $HANDOFF/bh/zone_patch.diff
```

`bh/zone_patch.diff` is `git diff 72620d5` of the instrumentation commits at the end of the campaign,
so it already contains both the streaming port and the non-streaming port, the toggle header, the zone
header, the harnesses and the reducer. The file is in the analysis workspace
(`$HANDOFF/bh/zone_patch.diff`, about 100 KB), not on either branch, and applies cleanly only to
`aae92de5471`; on any other base use the regeneration route below.

Regeneration route, if the kernels have moved and the diff no longer applies. Two anchor-matching
scripts rewrite the kernels in place; both are tracked in `analysis/campaigns/` and both resolve the
kernel directory from `TTM` per `PORTABLE_CONTRACT.md`, so set `TTM` to the tree to be instrumented
before running them:

```bash
cd $TTM/analysis/campaigns
python3 apply_zone_patch.py           # streaming path: compute_streaming.hpp, sdpa.cpp, reader, writer, dataflow_common
python3 apply_zone_patch_common.py    # non-streaming path: compute_common.hpp (fp32_dest_acc_en=True, production)
```

Both refuse to run twice (`apply_zone_patch.py` exits with "already patched" if `SDPA_ZACC` is already
present) and both abort with `anchor count N != 1` if a source line they expect has changed, which is
the intended failure mode when the kernels drift. After a successful run, regenerate the diff:

```bash
cd $TTM
git add -N ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zones.hpp \
           ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zone_config.hpp
git diff 72620d5 > $HANDOFF/bh/zone_patch.diff
```

The `git add -N` step is required for the two new files to appear in the diff; `bh/card_log.md` row 9a
records that the first version of the diff was missing them.

### 4.3 The toggle header

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zone_config.hpp` holds five defines and
nothing else. It is rewritten between runs by `analysis/campaigns/set_zone_config.sh`:

```
#define SDPA_ZONES 0            // 1: accumulate + per-q-chunk raw zones compiled in (PROFILE_KERNEL build only)
#define SDPA_ABL_READER_STUB 0  // A4: reader reserves/pushes K and V CBs without NoC reads
#define SDPA_ABL_MASK_OFF 0     // A2: causal lightweight mask bracket skipped on every k chunk
#define SDPA_ABL_EXP_STUB 0     // A6: softmax exp_packthread_tile calls removed (STALLWAIT kept)
#define SDPA_ABL_BARRIER_THR 0  // A4b: >0 overrides the reader barrier_threshold (reads in flight per barrier)
```

Usage, positional: `set_zone_config.sh ZONES READER_STUB MASK_OFF EXP_STUB BARRIER_THR`. The script
echoes the resulting defines so the run log records what was compiled.

```bash
cd $TTM/analysis/campaigns
./set_zone_config.sh 0 0 0 0 0        # everything off: the unmodified kernel instruction stream
./set_zone_config.sh 1 0 0 0 0        # zones on
./set_zone_config.sh 0 1 0 0 0        # ablation A4, zones off (clean wall)
./set_zone_config.sh 1 1 0 0 0        # ablation A4, zones on (which parts moved)
./set_zone_config.sh 0 0 0 0 64       # ablation A4b, barrier threshold 64
./set_zone_config.sh 0 0 1 0 0        # ablation A2, mask bracket off
./set_zone_config.sh 0 0 0 1 0        # ablation A6, exp stub
```

The five ablations and what each toggle does, from `bh/zone_decomposition.md` section 2.2 and
`campaign_t22.sh`:

| Name | Toggle | Kernel effect | What it isolates |
|---|---|---|---|
| A4 | `SDPA_ABL_READER_STUB=1` | reader `reserve_back` and `push_back` the K and V circular buffers without issuing the NoC reads; the Q reads are kept | exposed DRAM K and V read time |
| A4b | `SDPA_ABL_BARRIER_THR=64` | overrides the reader barrier threshold (reads in flight per barrier; the default at 110 cores is 4) so all 16 K or V reads of a chunk are in flight | whether the DRAM path is barrier limited or bandwidth limited |
| A2 | `SDPA_ABL_MASK_OFF=1` | `should_apply_lightweight_mask` is forced false, so the causal mask begin, apply and end bracket never runs | the cost of the causal mask bracket |
| A6 | `SDPA_ABL_EXP_STUB=1` | the 16 `exp_packthread_tile` calls per k chunk are removed, the `STALLWAIT` and the packs are kept; the SALAD column exp `exp_tile_first_column` is untouched | the softmax SFPU exp cost, separated from the issue stall |
| Zones | `SDPA_ZONES=1` | the accumulate and raw zones are compiled in | the gross instrumentation tax, measured against the zones-off twin |

Setting `SDPA_ZONES` back to 0 after a campaign block is part of every campaign script; leaving it at 1
silently inflates the next zones-off run by the gross tax (0.7 percent at the anchor, larger at q512).

### 4.4 Running a zones-off and zones-on pair

```bash
cd $TTM/analysis/campaigns
./set_zone_config.sh 0 0 0 0 0 >/dev/null
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 ./run_zone.sh t21_causal_q128k128_zoff
./set_zone_config.sh 1 0 0 0 0 >/dev/null
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 ./run_zone.sh t21_causal_q128k128_zon
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 ./run_zone.sh t21_causal_q128k128_zon_mp mp
./set_zone_config.sh 0 0 0 0 0 >/dev/null
```

`run_zone.sh` takes `<tag> [plain|mp]`. `plain` runs `python -m tracy -m pytest`, `mp` runs
`python -m tracy --perf-counter-multipass --profiler-capture-perf-counters=all -m pytest`. It writes,
into `$DD/`:

- `<tag>.log`, the full pytest and tracy stdout
- `<tag>.csv`, a PROVENANCE line followed by the raw `profile_log_device.csv`
- `<tag>_runs.csv`, `<tag>_cores.csv`, `<tag>_raw.csv` and, for `mp` runs, `<tag>_counters.csv`

It fails the run (exit 1) if the log does not contain "1 passed" or if
`generated/profiler/.logs/profile_log_device.csv` is empty, and it prints the compiled zone config into
the log and into the PROVENANCE line so that a mis-set toggle is visible afterwards.

### 4.5 Reading the zone sums back

The reducer is `analysis/zone_reduce.py`. It is called automatically by every runner, and can be run
by hand on any raw capture (strip the PROVENANCE line first; the reducer expects two header lines then
data, which is what the raw tracy CSV has):

```bash
cd $TTM
tail -n +2 $DD/t21_causal_q128k128_zon.csv > /tmp/zr.csv
python analysis/zone_reduce.py /tmp/zr.csv --out /tmp/t21_causal_q128k128_zon
```

What each output holds:

- `_runs.csv`, one row per op invocation: `run_idx`, `run_id`, `wall_dev_cycles`, `wall_core_x`,
  `wall_core_y`, `wall_core_risc`, `n_cores`, `trisc1_mean`, `trisc1_median`, `trisc1_max`,
  `kernel_start_skew`.
- `_cores.csv`, one row per invocation, core and RISC: `kernel_start` and `kernel_end` relative to the
  wall start, `kernel_dur`, `is_wall_core`, plus one column per accumulate zone (`K_WAIT`, `K_WAIT_N`
  and so on) and one `SUM:<zone>` column per native sum zone.
- `_raw.csv`, one row per raw zone instance with `start`, `end` and `dur`, including the `KERNEL` pair
  per RISC so absolute kernel start and end are available.
- `_counters.csv`, per invocation, core and RISC counter values with `ref_cnt`, written only when the
  capture contains counter records (section 5).

The wall basis the reducer implements: `wall_dev_cycles = max(KERNEL zone end) - min(KERNEL zone
start)` over all cores and all RISCs; the wall core is the core whose KERNEL zone ends last. Kernel
start skew across the 110 cores was 630 to 720 cycles in this campaign, so the wall-setting core's own
span equals the device wall within about 1.5k cycles.

### 4.6 The zone tax correction

The per-boundary cost was measured with nop kernels in T0.2 (section 3.2). Results, from
`data/bh_zones/t02_zone_tax_summary.csv` and `bh/zone_decomposition.md` section 2.1, in cycles:

| Zone kind | BRISC | NCRISC | TRISC_0 | TRISC_1 | TRISC_2 |
|---|---|---|---|---|---|
| raw `DeviceZoneScopedN`, per open and close, recorded | 56.2 | 52.0 | 56.7 | 51.6 | 56.3 |
| raw, dropped once the 125-pair buffer is full | 11.5 | 11.5 | 11.5 | 11.5 | 11.5 |
| native `DeviceZoneScopedSumN1` | 15.5 | 15.5 | 15.5 | 15.5 | 15.5 |
| custom `SDPA_ZACC`, whole-thread cost per occurrence (ACC_OUT) | 26.0 | 26.0 | 26.0 | 26.0 | 26.0 |
| custom `SDPA_ZACC`, inflation inside the measured window (ACC_IN) | 2.0 | 2.0 | 0.0 | 2.0 | 2.0 |

Two corrections follow, and every reducer in `analysis/campaigns/` applies them the same way:

- Inside each zone: `corrected = sum - count * ACC_IN[risc]`, with
  `ACC_IN = {TRISC_0: 0.0, TRISC_1: 2.0, TRISC_2: 2.0, BRISC: 2.0, NCRISC: 2.0}`. The constant appears
  literally as `ACC_IN` at the top of `zone_decomp.py`, `r1_tables.py`, `r1g_threads.py` and
  `regime_tables.py`.
- Outside the inner zones but inside the enclosing `STEP`: the remaining
  `(ACC_OUT - ACC_IN) * count` cycles per thread, with `ACC_OUT = 26.0`. These are subtracted from the
  un-zoned remainder as the `ZONE_TAX_IN_STEP` column rather than attributed to a part.

The gross effect on the wall is not estimated at all: it is measured directly by the zones-off twin of
every run and reported as `gross_zone_tax_pct`.

The decomposition driver, which combines the zones-off wall, the zones-on parts, the counters and the
model floor into one long table per tag:

```bash
cd $TTM/analysis/campaigns
python3 zone_decomp.py t21_causal_q128k128 --config "T2.1 anchor causal q128 k128"
python3 zone_decomp.py r1g_noncausal_S2048_q128k128_hd64 --mp-suffix _zon_mp --config "R1g hold-out"
```

It reads `<tag>_zoff_{runs,cores}.csv`, `<tag>_zon_{runs,cores,raw}.csv` and, when present,
`<tag>_zon_mp_{runs,counters}.csv`; discards invocation 0; writes `decomp_<tag>.csv` and merges a
summary row into `decomp_summary.csv` (replacing any row with the same tag and keeping every other
row). The model floor column comes from `predict()` of `analysis/roofline.py`, which is INTERNAL and
not on the pushed branch (section 1.1); without that file `zone_decomp.py` produces every column
except the floor, and the floor for a published table came from the polaris model of section 9.

Other reducers in the same directory, all run with no arguments and all writing next to themselves:

```bash
cd $TTM/analysis/campaigns
python3 report_tables.py       # all T2.1 and T2.2 tables into report_tables.md
python3 regime_tables.py       # T2.3 regime walls and per-thread parts (reads t23r_*_zoff_runs.csv)
python3 reduce_decode.py       # T2.8 decode durations and delivered KV GB/s (reads t28_*_ops_perf_results.csv)
python3 r1_tables.py           # R1 walls into r1_walls.csv and r1_tables.md
python3 r1b_decode_table.py    # R1b decode and MLA decode table (reads r1b_*_ops_perf_results.csv)
python3 r1f_counters.py        # R1f counter table into r1f_counters_table.csv
python3 r1f_table.py           # renders r1f_counters_table.csv as r1f_table.md
python3 r1g_threads.py         # R1g head_dim 64 wall-core thread split into r1g_thread_split.{csv,md}
python3 r1_cardlog.py          # appends a card_log.md row for every r1*.csv not yet logged (idempotent by tag)
```

`r1_tables.py` also encodes the campaign's dominant-thread rule: "reader/DRAM" when the UNPACK-thread
`K_WAIT + V_WAIT` on the zones-on wall core is at least 25 percent of the zones-on wall, otherwise
"compute (PACK)" when the PACK thread's non-wait zone sum exceeds the MATH thread's, else
"compute (MATH)".


## 5. Perf counters

### 5.1 Enabling them

Counters are requested on the tracy command line. The groups and the two constraints are in
`tools/tracy/__main__.py`:

```
group bits: fpu 0, pack 1, unpack 2, l1_0 3, l1_1 4, instrn 5, l1_2 6, l1_3 7, l1_4 8
constraints per pass: at most ONE L1 bank (l1_0..l1_4 share a single count-time mux)
                      at most 3 groups (4 overflow BRISC .text by 4 bytes, 5 by 16 bytes)
```

Single pass, a subset that fits:

```bash
cd $TTM
python -m tracy --profiler-capture-perf-counters=fpu,instrn,l1_0 -m pytest analysis/zone_sweep.py::test_zone_sweep -s
```

All groups, which needs the multipass scheduler because five L1 banks cannot share one pass:

```bash
cd $TTM
python -m tracy --perf-counter-multipass --profiler-capture-perf-counters=all -m pytest analysis/zone_sweep.py::test_zone_sweep -s
```

On Blackhole `all` expands to `fpu, pack, unpack, instrn, l1_0, l1_1, l1_2, l1_3, l1_4` and schedules
five passes. The workload is replayed once per pass; tracy sets `TT_METAL_PROFILE_PERF_COUNTERS` (a
bitfield) plus `TT_METAL_DEVICE_PROFILER=1` per pass, `tt_metal/jit_build/build.cpp` adds
`-DPROFILE_PERF_COUNTERS=<mask>` to the kernel build, and after the last pass the per-pass logs are
merged into one device log. Without `--perf-counter-multipass` an over-large request fails with the
required pass plan printed, which is the intended way to discover how many passes a group list needs.

To select a subset through the runner scripts, edit the `TP=` line of the runner or call tracy
directly; `run_zone.sh` and the other runners hard-code `=all` for their `mp` mode.

### 5.2 What lands on disk

| Path | Content |
|---|---|
| `generated/profiler/.logs/profile_log_device.csv` | the merged device log: zone rows plus the counter rows (marker id 9090) appended from every pass |
| `generated/profiler/.logs/perf_counter_passes/` | one snapshot per pass, `pass_0.csv` to `pass_N.csv`, before the merge |
| `generated/profiler/.logs/zone_src_locations.log`, `new_zone_src_locations.log` | zone name to source location maps |
| `generated/profiler/.logs/cpp_device_perf_report.csv` | the C++ post-processing summary |
| `generated/profiler/.logs/tracy_profile_log_host.tracy` | the host trace |
| `generated/profiler/reports/<date>/ops_perf_results_<date>.csv` | the ops report, written only with `-r` and only outside multipass |

The campaign archived the per-pass snapshots for the multipass captures it cared about, for example
`data/topk/perf_counter_passes_COUNTERS_noindexer/` (`pass_0.csv` to `pass_4.csv`) and
`data/topk/perf_counter_passes_indexer_Sq2048_T8192/`. The archiving command, from `campaign_t29.sh`:

```bash
cp -r generated/profiler/.logs/perf_counter_passes \
      $TOPKOUT/perf_counter_passes_COUNTERS_noindexer
```

### 5.3 The counter set

With `=all` on Blackhole, 155 distinct counters are recorded per core and RISC. Verify the count on
any capture:

```bash
cd $DD
python3 -c "
import csv, collections
c = collections.Counter()
for r in csv.DictReader(open('r1f_causal_q128k128_zoff_mp_counters.csv')): c[r['counter']] += 1
print(len(c), 'distinct counters'); print('\n'.join(sorted(c)))
"
```

The families and the ones the campaign actually reads, from the docstring of
`analysis/post_process.py` and from `handoff/investigations/data/parse_profile.py`:

- Engine busy: `FPU_COUNTER`, `SFPU_COUNTER`, `MATH_COUNTER`, `PACKER_BUSY`,
  `UNPACK0_BUSY_THREAD0/1`, `UNPACK1_BUSY_THREAD0/1`.
- Issue and stall: `MATH_INSTRN_AVAILABLE`, `MATH_INSTRN_STARTED`, `AVAILABLE_MATH`,
  `MATH_NOT_STALLED_DEST_WR_PORT`, `MATH_FIDELITY_STALL`, `DATA_HAZARD_STALLS_MOVD2A`,
  `MATH_SRC_DATA_READY`, `THREAD_INSTRUCTIONS_0/1/2`, `THREAD_STALLS_0/1/2`,
  `{FPU,SFPU,PACK,UNPACK,MOVE,SYNC,THCON,CFG}_INSTRN_AVAILABLE_*`.
- Waits: `WAITING_FOR_{MATH,SFPU,PACK,UNPACK,MOVE,THCON,MMIO}_IDLE_0/1/2`,
  `WAITING_FOR_NONZERO_SEM_0/1/2`, `WAITING_FOR_NONFULL_SEM_0/1/2`,
  `WAITING_FOR_SRC{A,B}_{VALID,CLEAR}`.
- Source and dest ports: `SRC{A,B}_WRITE_{ACTUAL,AVAILABLE,THREAD0,THREAD1}`, `DEST_READ_GRANTED_0`,
  `PACKER_DEST_READ_AVAILABLE`.
- L1 banks 0 to 4, request and grant pairs: `L1_0_UNPACKER_0` and `L1_0_UNPACKER_0_GRANT`,
  `L1_0_NOC_RING0_{INCOMING,OUTGOING}_{0,1}` and their `_GRANT` twins, `L1_0_TDMA_BUNDLE_0/1`,
  `L1_1_EXT_UNPACKER_1/2/3`, `L1_2_EXT_UNPACKER_4..7`, `L1_3_TDMA_PACK_EXT_0..3`,
  `L1_4_TDMA_PACK_EXT_4/5`, `L1_4_TAG_SEARCH_PACKER1`, and the NOC ring counters on banks 1 to 3.

Derived quantities defined in `analysis/post_process.py` (all as a percentage of that core's
`ref_cnt`, averaged over active cores, where "active" means `FPU_COUNTER > 0`):

```
scoreboard_stall   = MATH_INSTRN_AVAILABLE - AVAILABLE_MATH
dest_wr_port_stall = MATH_INSTRN_AVAILABLE - MATH_NOT_STALLED_DEST_WR_PORT
fidelity_stall     = MATH_FIDELITY_STALL
d2a_hazard_stall   = MATH_INSTRN_AVAILABLE - DATA_HAZARD_STALLS_MOVD2A
issue_eff          = MATH_INSTRN_STARTED / MATH_INSTRN_AVAILABLE
overlap            = (FPU_COUNTER + SFPU_COUNTER - MATH_COUNTER) / min(FPU_COUNTER, SFPU_COUNTER)
```

`ref_cnt` needs care. `perf_counters.hpp` packs it into 24 bits while the Blackhole hardware counter is
32 bits, so a kernel longer than about 12 ms at 1.35 GHz wraps. `load_counters` in
`analysis/post_process.py` recovers the full value from the paired profiler id 9091 records by matching
the low 24 bits, and falls back to the truncated value when no 9091 record is present, recording which
happened in the `ref_cnt_full_recovered` column.

### 5.4 Reading counters back

Three readers exist and all three go through the same `load_counters`:

```bash
cd $TTM
# 1. the sweep summary table plus a results CSV
python analysis/post_process.py generated/profiler/.logs/profile_log_device.csv --seq-lens 4096 --out /tmp/res.csv
# 2. zones and counters together, one row per invocation, core and RISC
python analysis/zone_reduce.py /tmp/zr.csv --out /tmp/mytag
# 3. per-run records with a fixed keep list, imported by the analysis scripts
python $WORK/handoff/investigations/data/parse_profile.py generated/profiler/.logs/profile_log_device.csv
```

`parse_profile.py` is the one reader that is not in a tt-metal checkout; it lives in the earlier
investigations workspace, so that one command needs the internal copy of the analysis workspace.

`parse_profile.py` keeps a 21-counter subset (`FPU_COUNTER`, `SFPU_COUNTER`, `MATH_COUNTER`,
`MATH_INSTRN_AVAILABLE`, `MATH_INSTRN_STARTED`, `WAITING_FOR_SRCA_VALID`, `WAITING_FOR_SRCB_VALID`,
`WAITING_FOR_NONZERO_SEM_0/1/2`, `UNPACK_INSTRN_AVAILABLE_0`, `THREAD_INSTRUCTIONS_0`,
`THREAD_STALLS_0`, `UNPACK0_BUSY_THREAD0`, `PACKER_BUSY`, `L1_0_UNPACKER_0` and its grant,
`L1_0_NOC_RING0_INCOMING_0` and its grant, `MATH_SRC_DATA_READY`, `AVAILABLE_MATH`) and reports means
over active cores plus the device wall and the mean TRISC_1 span.

### 5.5 Worked example, capture to counter table

The R1f block: counters on the unmodified kernel (zones compiled out) at twenty configurations, one
capture each.

```bash
cd $TTM/analysis/campaigns
./set_zone_config.sh 0 0 0 0 0 >/dev/null
SDPA_SEQ=4096 SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 ./run_zone.sh r1f_causal_q128k128_zoff_mp mp
python3 r1f_counters.py
python3 r1f_table.py
```

`run_zone.sh ... mp` runs, on the card:

```bash
python -m tracy --perf-counter-multipass --profiler-capture-perf-counters=all -m pytest analysis/zone_sweep.py::test_zone_sweep -s
```

`r1f_counters.py` then reads every `r1f_*_counters.csv` with its `_runs.csv` twin, discards invocation
0, and writes `r1f_counters_table.csv` with a PROVENANCE first line: per configuration and invocation
the `FPU_COUNTER`, `SFPU_COUNTER` and `MATH_COUNTER` on the wall-setting core and as means over active
cores, the device wall of the same multipass run, the active core count, the overlap
`(FPU + SFPU - MATH) / min(FPU, SFPU)` on both bases, and the MATH busy share of the wall.
`r1f_table.py` renders that CSV as the markdown table of `bh/campaign_r1.md` section 9 and prints the
run-to-run spread of every counter between invocations 1 and 2.

Counters do not distort the walls measurably on the plain prefill points: T0.1 recorded the multipass
causal anchor within 0.15 percent of the plain run. They do cost wall time on some shapes; the T2.1
table carries both `wall_zoff` and `wall_mp` so the difference is visible per configuration.

The zoned build and the unmodified build execute the same Tensix instruction stream: at the anchor
`THREAD_INSTRUCTIONS_0/1/2` differ by 11, 0 and 59 instructions, and `FPU_COUNTER` and `SFPU_COUNTER`
are identical (724,508 and 304,463 per core). That cross-check is what licenses reading zone sums from
one build and counters from the other.


## 6. Ops reports and model level runs

### 6.1 The ops report

Add `-r` to the tracy command line. Tracy then post-processes the device log into
`generated/profiler/reports/<date>/ops_perf_results_<date>.csv`, one row per device op invocation.

```bash
cd $TTM
rm -rf generated/profiler/reports/*
rm -f generated/profiler/.logs/profile_log_device.csv
python -m tracy -r -m pytest analysis/decode_sweep.py::test_decode_sweep -s
R=$(ls -d generated/profiler/reports/*/ | sort | tail -1); sleep 2; ls $R
cp $R/ops_perf_results*.csv /tmp/mytag_ops_perf_results.csv
```

The `sleep 2` between locating the directory and copying is not decorative. Every runner script in the
campaign has it, because the report writer finishes after tracy's own exit and a copy issued
immediately can capture an empty file (section 11.6).

Columns the campaign reads:

- `OP CODE`, the op name. SDPA prefill appears as `SDPAOperation`, paged decode as
  `SdpaDecodeDeviceOperation`; the reducers match with regular expressions
  (`SdpaDecode|ScaledDotProductAttentionDecode` in `reduce_decode.py`,
  `ScaledDotProduct|SDPA|Sdpa|sdpa|Attention` in `data/model_level/reduce_ops.py`).
- `DEVICE KERNEL DURATION [ns]`, located by prefix because the unit suffix varies:
  `[c for c in o.columns if c.startswith("DEVICE KERNEL DURATION")][0]`.
- `CORE COUNT`, the core count of that op instance.
- `ATTRIBUTES`, the op's resolved program and compute configuration as a string. This is the column
  the model's shim parses: `q_chunk_size`, `k_chunk_size`, `compute_with_storage_grid_size`,
  `exp_approx_mode`, `math_fidelity`, `fp32_dest_acc_en`, `is_causal`, `math_approx_mode`, and for
  decode `num_q_heads`, `num_kv_heads`, `head_dim`, the output memory config and
  `overlap_qk_coregrid`. `data/model_level/reduce_ops.py` extracts them with one regular expression
  over that string.
- `GLOBAL CALL COUNT`, `DEVICE ID`, `METAL TRACE ID`, `METAL TRACE REPLAY SESSION ID`,
  `PROGRAM CACHE HIT`, the row identity. Traced programs appear once per replay session, so rows with
  the same identity must be aggregated by median.
- `OP TYPE`, which is `signpost` for `tracy.signpost` rows, with the signpost header in `OP CODE`.
  That is how `topk_campaign.py --postprocess` finds the cell boundaries (section 7).

### 6.2 The model level runs (T2.5)

One `Attention` layer of `models/tt_transformers` with real Llama 3.1 8B Instruct weights, driven
through two env-controlled copies of the tt_transformers attention unit tests that exist only on
`mvlahovic/sdpa_topk_harness`: `analysis/attn_prefill_perf.py::test_attn_prefill_perf` and
`analysis/attn_decode_perf.py::test_attn_decode_perf`.

| Env var | Default | Meaning |
|---|---|---|
| `HF_MODEL` | none, required | weights directory, `/proj_sw/user_dev/llama31-8b-data/Llama-3.1-8B-Instruct` |
| `MESH_DEVICE` | none, required | `N150` for a single chip |
| `TT_CACHE_PATH` | none, required in practice | writable weight cache; the campaign used `data/model_level/tt_cache` |
| `ATTN_SEQ` | `4096` | prefill sequence length (`attn_prefill_perf.py`) |
| `ATTN_BATCH` | `32` | decode batch (`attn_decode_perf.py`) |
| `ATTN_POS` | `128,1024,4096` | decode cache positions (`attn_decode_perf.py`) |
| `ATTN_MAXSEQ` | `max(8192, 2 * max(ATTN_POS))` | page table sizing (`attn_decode_perf.py`) |
| `ATTN_ITERS` | `3` | invocations per point |
| `ATTN_SKIP_REF` | `1` | skip the reference comparison |
| `ATTN_PAGED` | `1` | paged attention, as the demo's `--paged_attention 1` |

Prefill. `HF_MODEL` stays an absolute path because the weights are a shared read-only directory
(`/proj_sw/user_dev/llama31-8b-data`, mode `drwxrwxr-x`, owned by neither the reader nor the reference
author) and are not part of any workspace copy:

```bash
cd $TTM
export TT_METAL_HOME=$PWD ARCH_NAME=blackhole
export HF_MODEL=/proj_sw/user_dev/llama31-8b-data/Llama-3.1-8B-Instruct MESH_DEVICE=N150
export TT_CACHE_PATH=$HANDOFF/data/model_level/tt_cache
source python_env/bin/activate
ATTN_SEQ=4096 ATTN_PAGED=1 ATTN_ITERS=3 ATTN_SKIP_REF=1 \
    python -m tracy -r -m pytest analysis/attn_prefill_perf.py::test_attn_prefill_perf -s
```

Decode:

```bash
cd $TTM
ATTN_BATCH=32 ATTN_POS=128,1024,4096 ATTN_MAXSEQ=8192 ATTN_PAGED=1 ATTN_ITERS=3 ATTN_SKIP_REF=1 \
    python -m tracy -r -m pytest analysis/attn_decode_perf.py::test_attn_decode_perf -s
```

`TT_CACHE_PATH` is mandatory because the default cache location is under the read-only weights
directory and the run fails with a `PermissionError` after loading the weights (`bh/card_log.md` row
16a). The same row records the other failure mode: the attention tests originally referenced the
tt_transformers test-directory fixture `ensure_gc` and failed in setup.

The four T2.5 runs are driven by `campaign_t25.sh` (section 8.2). Their outputs land in
`$HANDOFF/data/model_level/`: `<run>_ops_perf_results.csv`,
`<run>_profile_log_device.csv`, `<run>_PROVENANCE.txt` and `<run>.log`. Reduce them with the
following, whose `reduce_ops.py` is an offline script of the analysis workspace and is on no branch:

```bash
cd $HANDOFF/data/model_level
python3 reduce_ops.py
```

which writes `sdpa_ops_summary.csv` (one row per SDPA-family op instance with its duration and
`ATTRIBUTES`) and prints per run the median, min and max over the invocations of each distinct
`OP CODE` and attribute combination, excluding the first invocation when there are at least three.

### 6.3 The positions sidecar

The decode ops report does not carry the cache position: `cur_pos` is a runtime argument, not an
attribute. The validation harness therefore takes a sidecar CSV mapping `GLOBAL CALL COUNT` to
`cur_pos` and, optionally, `page_block_size`. The campaign's sidecar is
`data/model_level/positions_llama8b_decode.csv`, derived by hand from the run's PROVENANCE line
(`ATTN_POS=128,1024,4096`, `ATTN_ITERS=3`, batch 32, paged block 32), with the nine
`SdpaDecodeDeviceOperation` rows in `GLOBAL CALL COUNT` order being three invocations at each position:

```
# PROVENANCE: positions for llama8b_attn_decode_b32_pos128_1024_4096_ops_perf_results.csv ...
GLOBAL CALL COUNT,cur_pos,page_block_size
25600,128,32
45056,128,32
64512,128,32
83968,1024,32
103424,1024,32
122880,1024,32
142336,4096,32
161792,4096,32
181248,4096,32
```

To build one for a new decode run, read `ATTN_POS` and `ATTN_ITERS` out of the run's PROVENANCE line
and pair them with the decode rows in call-count order:

```bash
cd $HANDOFF/data/model_level
head -1 llama8b_attn_decode_b32_pos128_1024_4096_PROVENANCE.txt
python3 -c "
import pandas as pd
d = pd.read_csv('llama8b_attn_decode_b32_pos128_1024_4096_ops_perf_results.csv', low_memory=False)
s = d[d['OP CODE'].astype(str).str.contains('SdpaDecode')]
print(s[['GLOBAL CALL COUNT', 'OP CODE']].to_string(index=False))
"
```

`cur_pos` is the position of the token being decoded, so the attended length is `cur_pos + 1`. The
sidecar is passed to the validation harness with `--positions` (section 9.5).


## 7. The TopK campaign

### 7.1 The driver

`$TTM/analysis/campaigns/topk_campaign.py` is a standalone script (not a
pytest module) that opens device 0, enables the program cache, runs no trace capture, and walks a
fixed grid of 300 cells. It is tracked on the harness branch beside the runners, and is driven from
the fresh tree (section 7.3) because the ops it measures are the main-tip ones. Its plan is
`topk/sweep_plan.md` in the handoff workspace; the routing and kernel background are in `topk/generic_topk_kernel.md` and
`topk/router_topk_kernels.md`.

Command line:

| Flag | Default | Meaning |
|---|---|---|
| `--classes` | `SMOKE` | comma list of `L1..L6`, `A..G`, `R1..R4`, `COUNTERS`, `SMOKE`, or `ALL` |
| `--iters` | `0` meaning per-cell default | override the timed iteration count of every cell |
| `--out` | the `data/topk` directory of the handoff workspace | output directory; pass a directory of your own when writing new cells |
| `--list` | off | enumerate cells and exit without importing torch or ttnn |
| `--postprocess REPORT_CSV CELLS_CSV` | off | join a tracy ops report to a cells CSV and exit |
| `--exclude-op` | empty | comma list of cell ops to skip at run time, for example `indexer` |

It takes no environment variables of its own. It reads `TT_METAL_HOME` (falling back to the working
directory) and `TT_METAL_PROFILE_PERF_COUNTERS` only to build its PROVENANCE line, which records the
git sha, branch, dirty state, arch, grid, core count, host, date, the full command line, the
perf-counter env value and the compiled `MEM_BRISC_FIRMWARE_SIZE` line grepped at run time.

### 7.2 Enumerate before running

`--list` needs no card, no ttnn and no torch, so it can be run anywhere:

```bash
python3 $TTM/analysis/campaigns/topk_campaign.py --list --classes ALL
```

```
A           69      L1          57      R1          25
B            8      L2           8      R2           8
C            9      L3           7      R3           3
D           14      L4           3      R4           3
E           19      L5           6      SMOKE        7
F           21      L6           6      COUNTERS    11
G           16
duration cells (excl. COUNTERS, SMOKE): 282; COUNTERS: 11; SMOKE: 7; total selected: 300
```

With one or two classes selected, `--list` also prints every `cell_id` with its predicted routing
path, which is the cheapest way to check a grid edit:

```bash
python3 $TTM/analysis/campaigns/topk_campaign.py --list --classes L6
```

What the classes cover, from `topk/sweep_plan.md` section 3: `L1` to `L6` are
`ttnn.experimental.topk_large_indices` (row law refit, mode boundaries, anchors); `A` to `G` are the
generic `ttnn.topk` (single-core factory, multi-core factory, composite and sampling routes); `R1` and
`R2` are the MoE gates (`generalized_moe_gate`, `deepseek_moe_gate`, `moe_grouped_topk`); `R3` is
`ttnn.sort` as a cross-check; `R4` is `indexer_score_dsa`; `SMOKE` is one cell per op; `COUNTERS` is
the perf-counter subset.

### 7.3 Running a class group

One tracy invocation per class group, one process each. Options come before the script path.

```bash
export TT_METAL_HOME=$TTM_FRESH
export ARCH_NAME=blackhole
export PYTHONPATH=$TT_METAL_HOME:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools
source $TTM/python_env/bin/activate
cd $TT_METAL_HOME
rm -rf generated/profiler/reports/*
python -m tracy -r -p -v $TTM/analysis/campaigns/topk_campaign.py \
    --classes SMOKE --out $TOPKOUT
```

`-r` writes the ops report, `-p` profiles only enabled zones, `-v` is verbose. The recorded run order
of the campaign, one line per tracy invocation:

```
SMOKE -> L1,L2,L3,L4,L5 -> L6 -> A -> B,C,D -> E,F -> G -> R1 -> R2,R3 -> R4 -> COUNTERS (multipass)
```

Smoke first, so the BRISC firmware size problem and any API drift surface before the grid.

Iteration discipline per duration cell, implemented in the `Runner` class and stated in
`sweep_plan.md` section 0: program cache on, trace off, one cache-warming call plus two further
warmups discarded, then `--iters` timed calls (default 3, anchors 6), `ttnn.synchronize_device` after
each call, and the per-op `DEVICE KERNEL DURATION [ns]` taken from the ops CSV rather than a host
timer. Any exception in a cell (a `TT_FATAL` surfaces as `RuntimeError`, out of memory included) is
caught, its first 400 characters recorded in the `error` column, and the run continues after
re-synchronizing the device.

Each run writes `cells_<TAG>.csv` into `--out`, where `TAG` is the class list with commas replaced by
underscores. Columns: `cell_id`, `cls`, `op`, `params`, `iters`, `predicted`, `status`, `info`,
`error`, `host_seconds`, behind a `# PROVENANCE` first line.

### 7.4 The counter cells

The `COUNTERS` class holds 11 cells (`sweep_plan.md` section 3.4): `topk_large_indices` at `R = C` with
N in {16384, 131072} crossed with K in {512, 2048}, generic class A at (4096, 32) and (4096, 512)
single core, class E at (16384, 32) multi core, class F at (65536, 128) composite,
`generalized_moe_gate` at `B = C` with k = 8 and no softmax, `moe_grouped_topk` at N = 128 and
T = 4096, and the indexer at (2048, 8192).

```bash
cd $TTM_FRESH
python -m tracy -r -p --perf-counter-multipass --profiler-capture-perf-counters=all \
    $TTM/analysis/campaigns/topk_campaign.py \
    --classes COUNTERS --out $TOPKOUT
```

The first attempt of this cell group aborted: the indexer cell raises `TT_FATAL` on main tip
(section 11.5). The rerun excludes it:

```bash
cd $TTM_FRESH
python -m tracy -r -p --perf-counter-multipass --profiler-capture-perf-counters=all \
    $TTM/analysis/campaigns/topk_campaign.py \
    --classes COUNTERS --exclude-op indexer \
    --out $TOPKOUT
```

Multipass emits no ops report (section 11.4), so the counter cells have no `results_*.csv` from the
join; their evidence is the merged device log `profile_log_device_COUNTERS_noindexer.csv` plus the
per-pass snapshots in `perf_counter_passes_COUNTERS_noindexer/` (`pass_0.csv` to `pass_4.csv`), read
with `analysis/post_process.py` or `parse_profile.py` (section 5.4). For the indexer capture on the
calibration checkout the merged log carries KERNEL zones plus 40,920 counter rows.

### 7.5 The postprocess join

`--postprocess` runs offline, needs no card and no ttnn, and joins the ops report to the cells by
signpost order. Each cell is bracketed by `signpost("CELL_BEGIN", cell_id)` and
`signpost("CELL_END", cell_id)`, which appear in the ops report as rows with `OP TYPE = signpost`, the
header in `OP CODE` and the message in `ATTRIBUTES`. Every device op row between a begin and an end is
attributed to that cell.

```bash
cd $TTM_FRESH
OUT=$TOPKOUT
R=$(ls -d generated/profiler/reports/*/ | sort | tail -1)
cp $R/ops_perf_results*.csv $OUT/ops_perf_results_L6.csv
python3 $TTM/analysis/campaigns/topk_campaign.py \
    --postprocess $OUT/ops_perf_results_L6.csv $OUT/cells_L6.csv
mv $OUT/topk_campaign_results.csv $OUT/results_L6.csv
```

The output is one row per (cell, op) with columns `cell_id`, `cls`, `op`, `params`, `iters`,
`predicted`, `status`, `info`, `error`, `op_index`, `op_code`, `n_calls`, `dev_kernel_ns_median`,
`dev_kernel_ns_min`, `dev_kernel_ns_max`, `per_iter_sum_all_ops_ns_median`, `op_codes_in_cell`, behind
the cells PROVENANCE line with `report=<path>` appended. `op_codes_in_cell` is the route
verification: it lists the device ops the call actually dispatched, which is how the routing
predictions of `sweep_plan.md` section 2.3 were checked.

The default output path is `topk_campaign_results.csv` next to the cells CSV; `campaign_t26.sh`
renames it to `results_<TAG>.csv` after each join. The join for `SMOKE`, `L1` to `L5` and `L6` was
redone offline after a fix to the signpost message handling (`bh/card_log.md` row 17a); later groups
used the fixed script.

### 7.6 The indexer points

The `R4` indexer cells of the plan could not run on main tip, so the three points were taken on the
calibration checkout with `analysis/indexer_probe.py` (section 3.19). The block is
`campaign_t29.sh`:

```bash
cd $TTM/analysis/campaigns
./campaign_t29.sh
```

which runs, per point:

```bash
cd $TTM
IX_SQ=2048 IX_T=8192 IX_ITERS=3 IX_HEADS=64 IX_D=128 \
    python -m tracy -r -m pytest analysis/indexer_probe.py::test_indexer_probe -s
IX_SQ=2048 IX_T=32768 IX_ITERS=3 IX_HEADS=64 IX_D=128 \
    python -m tracy -r -m pytest analysis/indexer_probe.py::test_indexer_probe -s
IX_SQ=512 IX_T=8192 IX_ITERS=3 IX_HEADS=64 IX_D=128 \
    python -m tracy -r -m pytest analysis/indexer_probe.py::test_indexer_probe -s
IX_SQ=2048 IX_T=8192 IX_ITERS=3 IX_HEADS=64 IX_D=128 \
    python -m tracy -r --perf-counter-multipass --profiler-capture-perf-counters=all \
    -m pytest analysis/indexer_probe.py::test_indexer_probe -s
```

Results land in `data/topk/t29_indexer_*` (log, ops report, device log, PROVENANCE) and are tabulated
in `topk/indexer_points.md`. The three points ran on 88 cores with `chunk_start_idx = T - Sq`.

### 7.7 The offline TopK analysis and model acceptance

Two further scripts in the `topk/` directory of the analysis workspace, both offline, both reading
the `data/topk/results_*` CSVs, and both needing the internal copy of that workspace:

```bash
cd $HANDOFF/topk
$POLARIS/.venv/bin/python topk_analyze.py
$POLARIS/.venv/bin/python topk_model_acceptance.py
```

`topk_analyze.py` produces the fits of `topk/topk_campaign_results.md` and
`data/topk/topk_fits.csv`; `topk_model_acceptance.py` scores the polaris model against the campaign
cells into `data/topk/topk_model_acceptance.csv`, which is the input of the `tm_` figures
(section 9.6).


## 8. The campaign scripts

All of them live in `$TTM/analysis/campaigns/` on `mvlahovic/sdpa_topk_harness` and are executable.
They must be run from that directory, because they call each other by relative path, and they write
their captures and derived tables into `$DD`, which they take from the environment and which must
therefore be set to a directory the reader can write (section 1.1). Durations below are the observed
spans between the first and last logged run start of that block in `bh/card_log.md`, so they exclude
the last run's own time; add a minute or two per block.

### 8.1 The runners

| Script | Signature | Tree | Harness | Notes |
|---|---|---|---|---|
| `run_zone.sh` | `<tag> [plain\|mp]` | `tt-metal` | `analysis/zone_sweep.py::test_zone_sweep` | config from the `SDPA_*` env vars; `plain` is `tracy -m pytest`, `mp` adds `--perf-counter-multipass --profiler-capture-perf-counters=all` |
| `run_regime.sh` | `<tag> <plain\|mp> "<pytest target incl -k>"` | `tt-metal` | any | `plain` is `tracy -r -m pytest`, so it also copies the ops report; env is carried in and recorded (`P2_`, `MLA_`, `MLAD_`, `CK_`, `SDPA_`, `R1_`, `SP_`, `JT_`, `DEC_` prefixes) |
| `run_decode.sh` | `<tag> [plain\|mp]` | `tt-metal` | `analysis/decode_sweep.py::test_decode_sweep` | always `tracy -r`; config from the `DEC_*` env vars |
| `run_fresh.sh` | `<tag> [plain\|mp]` | `tt-metal-fresh` | `analysis/zone_sweep.py::test_zone_sweep` | same as `run_zone.sh` on the fresh tree; zones do not exist there, so the toggle header is irrelevant |
| `run_regime_fresh.sh` | `<tag> <plain\|mp> "<target>"` | `tt-metal-fresh` | any | same as `run_regime.sh` on the fresh tree |
| `run_zone_tax.sh` | `<tag> "<ZT_GRID>" [ZT_CORES]` | `tt-metal` | `analysis/zone_tax.py::test_zone_tax` | uses `tracy --enable-sum-profiling` |
| `set_zone_config.sh` | `ZONES READER_STUB MASK_OFF EXP_STUB BARRIER_THR` | `tt-metal` | none | rewrites the toggle header |

Every runner exports `TT_METAL_HOME`, `ARCH_NAME=blackhole` and `TT_METAL_FORCE_JIT_COMPILE=1`,
activates the calibration venv, clears `generated/profiler/.logs/profile_log_device.csv` (and, for the
`-r` runners, `generated/profiler/reports/*`), records a PROVENANCE line with the firmware bundle from
the log, the git sha and branch, the timestamp, the exact command, the env and the compiled zone
config, and finally calls `analysis/zone_reduce.py`. The two fresh-tree runners call the reducer from
the calibration tree by absolute path, because the fresh tree's copy is a working-tree file.

### 8.2 The campaign blocks

| Script | Block | What it runs | Runs | Observed span |
|---|---|---|---|---|
| `campaign_t21.sh` | T2.1 grid 1 | seven `(q, k)` points (128:128, 64:128, 256:128, 128:256, 128:512, 512:128, 512:512) crossed with causal and non-causal, each as zones off, zones on and zones on with counters | 42 plus 1 smoke | 22:16 to 23:20 on 2026-09-11, about 1 h |
| `campaign_t22.sh` | T2.2 ablations | four ablations (`a4`, `a4b`, `a2`, `a6`) at three causal points (q128 k128, k256, k512), each zones off and zones on, plus counters at the anchor | 28 | 23:22 to 23:50, about 30 min |
| `campaign_t23.sh` | T2.3 DRAM law | five zones-off variants at the causal anchor: K/V bf16, nkv 32, nkv 1, grid 8x8, all bf16 | 5 | 23:50 to 23:53, about 3 min |
| `campaign_t23r.sh` | T2.3 regimes | cross, window, mask through `p2_sweep.py`, MLA through `mla_perf_sweep.py`, chunked through `chunked_sweep.py`, each zones off and zones on (counters on cross and MLA), plus causal S1024 and S16384 walls | 14 | 00:16 to 00:29 on 2026-09-12, about 14 min |
| `campaign_t24_zoff.sh` | T2.4 production, zones off | the production Llama config (HiFi4, exp accurate, fp32 acc, packer L1 acc, 8x8 grid, q = k = 256) at S 2048, 4096, 8192, plus S1024 at q = k = 64, plus fp32 acc off, plus the 110-core variant | 6 | started 00:04, logged as one card_log row, about 4 min |
| `campaign_t24_zon.sh` | T2.4 production, zones on | the S4096 production point with zones on at 64 and 110 cores, with counters, and with fp32 acc off | 4 | 00:08 to 00:13, about 5 min |
| `campaign_t25.sh` | T2.5 model level | Llama 3.1 8B attention layer, prefill at S 1024, 4096, 8192 and decode at batch 32 positions 128, 1024, 4096, each `tracy -r` | 4 | 00:43 to 00:49, about 6 min plus weight loading |
| `campaign_t26.sh` | T2.6 TopK | the eleven TopK class groups on the fresh tree, each with its postprocess join | 11 | 00:51 to 01:09, about 18 min |
| `campaign_t27.sh` | T2.7 drift check | causal and non-causal anchors and two production points, on the fresh tree, zones off | 4 | 00:32 to 00:34, about 2 min |
| `campaign_t28.sh` | T2.8 decode | paged decode at batch 32 over six positions on 64 cores, batch 32 on 110 cores, batches 8 and 16, K/V bf16, and one counter run | 6 | 01:15 to 01:20, about 5 min |
| `campaign_t29.sh` | T2.9 indexer and TopK counters | three indexer points plus one counter capture on the calibration tree, then the TopK `COUNTERS` group with the indexer excluded on the fresh tree | 5 | 01:37 to 01:43, about 6 min |
| `campaign_r1.sh` | R1 re-measurement | blocks R1d (anchor stability), R1a (prefill regimes), R1b (decode), R1c (hold-outs), R1e (fresh drift for the two non-streaming kernels), interleaved with three R1d anchor repeats and one production end point | 64 | 10:40 to 11:40 on 2026-09-12, about 1 h |
| `campaign_r1_add.sh` | R1b addendum | MLA decode in the model's non-paged form at three cache lengths, plus the paged geometry at batch 8 | 2 | 11:36 to 11:40 |
| `campaign_r1f.sh` | R1f counters | perf-counter multipass captures on the unmodified kernel at twenty configurations | 19 | 12:00 to 12:48, about 48 min |
| `campaign_r1g.sh` | R1g hold-outs | head_dim 64 hold-outs at five shapes, zones off, with one zones-on twin | 5 | 13:14 to 13:17, about 3 min |

Run a block:

```bash
cd $TTM/analysis/campaigns
./campaign_t21.sh 2>&1 | tee $DD/t21_campaign.out
./campaign_t22.sh 2>&1 | tee $DD/t22.out
./campaign_r1.sh  2>&1 | tee $DD/r1.out
./campaign_r1f.sh 2>&1 | tee $DD/r1f.out
./campaign_r1g.sh 2>&1 | tee $DD/r1g.out
```

Three blocks accept overrides through the environment rather than arguments:

```bash
cd $TTM/analysis/campaigns
POINTS="128:128 128:256" MODES=causal WITH_MP=0 ./campaign_t21.sh
POINTS="128:128" ABL="a2:0:1:0:0 a6:0:0:1:0" WITH_MP=0 ./campaign_t22.sh
PROD_Q_DTYPE=bfp8_b PROD_KV_DTYPE=bfp8_b ./campaign_t24_zoff.sh
```

`campaign_t21.sh` takes `POINTS` (space-separated `q:k` pairs, default the seven grid-1 points),
`MODES` (default `causal noncausal`) and `WITH_MP` (default 1, adds the counter capture per point).
`campaign_t22.sh` takes `POINTS` (default `128:128 128:256 128:512`), `ABL` (default
`a4:1:0:0:0 a4b:0:0:0:64 a2:0:1:0:0 a6:0:0:1:0`, each entry `name:READER_STUB:MASK_OFF:EXP_STUB:BARRIER_THR`)
and `WITH_MP`. `campaign_t24_zoff.sh` takes `PROD_Q_DTYPE` (default `bfloat16`) and `PROD_KV_DTYPE`
(default `bfloat8_b`).

`campaign_r1.sh` is built from four helpers worth knowing when reading its output: `anchor` runs the
causal S4096 anchor zones off, `zpair` runs a `zone_sweep.py` point zones off then zones on, `rpair`
does the same for a regime harness target, and `roff` runs one regime point at a chosen tracy mode
with zones off. Every call is followed by `|| true`, so a single failing point does not abort the
block; check the block output for `RUN FAILED` afterwards:

```bash
grep -n "RUN FAILED" $DD/r1.out
```

After a block, append the card log rows and rebuild the tables:

```bash
cd $TTM/analysis/campaigns
python3 r1_cardlog.py
python3 r1_tables.py
python3 r1b_decode_table.py
```

Total card time for the whole campaign, from the two `close` rows of `bh/card_log.md`: 2026-09-11
21:54 to 23:54 for T0 to T2.3, then to 2026-09-12 01:48 for T2.4 to T2.9, then the R1 blocks on
2026-09-12 10:40 to 13:17. `bh/card_log.md` holds 230 table rows, of which 5 are notes or block-close
markers; the remaining 225 rows are device runs, except that a few rows stand for a whole campaign
block (row 9 for `campaign_t21.sh`, the T2.4 zones-off row for all six of its points), so the number of
device runs is somewhat higher than the row count.


## 9. Polaris: the model, its tests, validation and figures

### 9.1 The environment

The campaign used a uv virtual environment inside the checkout rather than the conda path the README
documents. Verify it:

```bash
cd $POLARIS
cat .venv/pyvenv.cfg
.venv/bin/python --version
.venv/bin/python -m mypy --version
```

`pyvenv.cfg` records `uv = 0.11.32`, `version_info = 3.12`, `implementation = CPython`,
`include-system-site-packages = false`, `prompt = polaris`, with the interpreter taken from uv's own
managed CPython (`~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu`). The interpreter reports
3.12.13 and mypy reports 1.15.0.

To recreate it:

```bash
cd $POLARIS
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    deepdiff==9.1.0 einops==0.8.2 gitpython==3.1.50 hydra-core==1.3.2 loguru==0.7.3 lxml==6.0.0 \
    markdown-it-py==3.0.0 matplotlib==3.10.0 networkx==3.4.2 numpy==2.2.3 onnx==1.17.0 \
    openpyxl==3.1.5 pillow==11.1.0 pydantic==2.10.6 pyelftools==0.31 pyyaml==6.0.2 scipy==1.17.1 \
    simpy==4.1.1 sphinx==8.2.1 sphinx-rtd-theme==3.0.1 \
    coverage==7.6.12 isort==8.0.1 mypy==1.15.0 pre-commit==4.6.0 pytest==8.3.4 pytest-cov==6.3.0 \
    pytest-json-report==1.5.0 pytest-mock==3.15.1 pytest-xdist==3.8.0 ruff==0.15.16 \
    types-PYYAML==6.0.12.20241230 types-networkx==3.4.2.20241227 types-requests==2.33.0.20260518 \
    types-openpyxl==3.1.5.20260518 lxml-stubs==0.5.1
```

That list is exactly the dependency set of `envdev.yaml` (the development environment of the repo),
with the conda pins translated to pip pins. The existing `.venv` is a subset of it: it carries the
runtime packages, `mypy 1.15.0`, `types-PyYAML`, `pytest` (9.1.1, newer than the `envdev.yaml` pin
of 8.3.4) and `pytest-mock`, but not `coverage`, `ruff`, `pre-commit`, `pytest-xdist`, `pytest-cov`
or the other stub packages. Anything that needs those, including a faithful reproduction of the CI
static-analysis step, needs the full set.

The README's documented path, which CI follows, is conda through Miniforge:

```bash
cd $POLARIS
conda env create --file envdev.yaml
conda activate polarisdev
pre-commit install
```

`environment.yaml` builds the runtime-only `polaris` environment; `envdev.yaml` builds `polarisdev`
and adds the test and static-analysis tooling. Workloads that pull artifacts from the Large File Cache
need `LFC_SERVER_URLS` set to a comma-separated list of LFC base URLs; there is no default and the SDPA
and TopK model work does not need it.

### 9.2 The test suites

The SDPA and TopK model tests are three files under `tests/test_perf/`:

```bash
cd $POLARIS
.venv/bin/python -m pytest tests/test_perf -q
.venv/bin/python -m pytest tests/test_perf/test_roofline_sdpa.py -q
.venv/bin/python -m pytest tests/test_perf/test_roofline_topk.py -q
.venv/bin/python -m pytest tests/test_perf/test_sdpa_validate.py -q
```

`pytest tests/test_perf -q --collect-only` collects 363 tests on the current branch
(`test_roofline_sdpa.py` 124 test functions, `test_roofline_topk.py` 38, `test_sdpa_validate.py` 23,
the rest parametrizations).

The whole repository suite, as the audits ran it. The deselect is an environmental `coverage` failure
unrelated to this work:

```bash
cd $POLARIS
.venv/bin/python -m pytest tests -q --deselect tests/test_tools/test_compare_proj.py::test_compare_proj
```

The recorded result at the R2 audit commit was 2522 passed, 4 skipped, 1 deselected, 6 xfailed. The six
xfails are the deliberate beyond-5-percent walls (four prefill in `_BEYOND_5`, two MLA decode in
`_MLAD_BEYOND_5`).

`tests/test_perf/data/` holds the validation fixtures: `ops_perf_results_sdpa_fixture.csv` and
`ops_perf_results_sdpa_fixture_positions.csv`. The fixture is partly synthetic (its synthetic rows are
marked as such in the header) and partly two real T2.5 rows, at `GLOBAL CALL COUNT` 28672 and 103424.

### 9.3 Running the model from python

The model of record is `ttsim/perf/roofline_sdpa.py` (SDPA prefill and decode) and
`ttsim/perf/roofline_topk.py` (TopK, MoE gates, indexer). Both are pure python with no device
dependency.

Prefill:

```bash
cd $POLARIS
.venv/bin/python -c "
from ttsim.perf.roofline_sdpa import SdpaConfig, predict
r = predict(SdpaConfig(S=4096, q_chunk=128, k_chunk=128, num_heads=32, num_kv_heads=8,
                       head_dim=128, num_cores=110, is_causal=True))
print(r.wall_clock_cycles, r.wall_regime, r.kernel_path)
print({k: round(v) for k, v in r.components.items()})
"
```

Output on the current branch:

```
2563224 prefill_causal streaming
{'init': 2900, 'compute_floor': 861427, 'fe_issue': 371059, 'reader_wait': 1067384, 'mask_bracket': 8755, 'sfpu_issue': 0, 'dest_roundtrip': 0, 'control': 74803, 'straggler': 176895}
```

`components` is the named wall-term dictionary and its values sum to `wall_clock_cycles`; that is the
contract the tests assert. The other fields of `RooflineResult` worth reading are `regime`,
`wall_regime`, `kernel_path` (`streaming`, `legacy` for the fp32 DEST path, or `thread_split`),
`steps_wall_core`, `active_cores`, `fpu_cycles`, `sfpu_cycles`, `math_active_cycles`,
`compute_latency_cycles`, `overlap_frac`, `dram_in_bytes`, `is_memory_bound`, `low_confidence` with
`low_confidence_reasons`, `defaulted` and `config_echo`.

`SdpaConfig` carries every regime as a field rather than a separate entry point: `kv_seq` for cross
attention, `sliding_window`, `has_attn_mask`, `is_chunked` with `chunk_start_idx`, `paged` with
`page_block_size`, `is_sparse`, `is_joint` with `joint_seq`, `attention_sink`, `v_head_dim` for MLA,
`fp32_dest_acc`, `exp_approx_mode`, `fidelity`, `input_dtype`, `kv_input_dtype`, `batch`,
`dram_scatter_derate` and an `arch` (`ArchConfig`).

Decode:

```bash
cd $POLARIS
.venv/bin/python -c "
from ttsim.perf.roofline_sdpa import predict_decode
r = predict_decode(cache_len=1025, num_q_heads=32, num_kv_heads=8, head_dim=128, batch=32,
                   num_cores=64, paged=True, page_block_size=32,
                   input_dtype='bfp8_b', kv_input_dtype='bfp8_b')
print(r.wall_clock_cycles, r.regime, r.is_memory_bound)
"
```

which prints `380279 decode True`. `predict_decode` is a function, not a config dataclass; its
arguments are `cache_len, num_q_heads, num_kv_heads, head_dim, v_head_dim=0, k_chunk=128, batch=1,
cur_pos=None, sliding_window=0, fidelity="HiFi4", input_dtype="bfloat16", accum_dtype="bfloat16",
num_cores=110, num_cores_per_head=0, arch=None, kv_input_dtype=None, cur_pos_unknown=False,
paged=False, page_block_size=0, is_causal=True, has_attn_mask=False, max_cores_per_head_batch=0,
defaulted="", fallback_reasons=(), q_in_dram=True, q_shard_cores=0, mla_v_read=False`. Either
`cache_len` or `cur_pos + 1` is the attended length; `q_shard_cores` and `mla_v_read` are the MLA
decode knobs.

TopK and indexer:

```bash
cd $POLARIS
.venv/bin/python -c "
from ttsim.perf.roofline_topk import TopkConfig, predict_topk, IndexerConfig, predict_indexer
t = predict_topk(TopkConfig(N=16384, K=512, rows=1))
print(t.route, t.body_mode, t.device_cycles, t.components)
i = predict_indexer(IndexerConfig(Sq=2048, T=8192, Hi=64, D=128, num_cores=88))
print(i.device_cycles, i.components)
"
```

`predict_topk` covers `topk_large_indices`; `predict_generic(GenericConfig(...))` covers the generic
`ttnn.topk` factories; `predict_gate(GateConfig(...))` covers the MoE gates;
`predict_topk_call(TopkCall(...))` routes a call through `route_topk` and prices whichever kernel it
lands on; `predict_indexer(IndexerConfig(...))` covers `indexer_score_dsa`. Each returns a
`TopkResult` whose `components` dictionary sums to `device_cycles`, with `route`, `body_mode`,
`kernel_rev`, `breakdown` and the same `low_confidence` mechanism. `kernel_rev` selects a calibration
set through `calibration_for`; leaving it `None` picks the default and flags
`kernel_rev_defaulted_<sha>`, so set it explicitly when comparing against a specific kernel state.

Every predictor has a `*_perf_stats` twin (`sdpa_perf_stats`, `decode_perf_stats`, `topk_perf_stats`,
`topk_call_perf_stats`, `indexer_perf_stats`) that emits the ttsim `perf_stats` dictionary consumed by
`Device.execute_op`, and `sdpa_config_from_shapes` / `decode_config_from_shapes` build a config from
tensor shapes plus an `ATTRIBUTES` string, which is how the ops-report rows of section 6 reach the
model.

### 9.4 Attributes from shapes, the shim path

```bash
cd $POLARIS
.venv/bin/python -c "
from ttsim.perf.roofline_sdpa import sdpa_config_from_shapes, predict
cfg = sdpa_config_from_shapes((1, 32, 4096, 128), (1, 8, 4096, 128), (1, 8, 4096, 128),
        attrs={'q_chunk_size': 256, 'k_chunk_size': 256, 'is_causal': True,
               'math_fidelity': 'HiFi4', 'exp_approx_mode': False, 'fp32_dest_acc_en': True},
        num_cores=64)
r = predict(cfg)
print(r.kernel_path, r.wall_clock_cycles, r.defaulted, r.low_confidence_reasons)
"
```

Fields absent from `attrs` are echoed in `defaulted` and any model-side fallback in
`fallback_reasons`, so a validation row can always be told apart from a fully specified one. The
attribute-to-field mapping is specified in `model/attr_plumbing_spec.md`.

### 9.5 The validation harness

`tools/sdpa_validate.py` is a thin wrapper over `ttsim/perf/sdpa_validate.py`. It scores the model
against a tt-metal tracy ops report.

| Flag | Default | Meaning |
|---|---|---|
| `csv` (positional) | required | `ops_perf_results_*.csv` from `tools/tracy/process_ops_logs.py` |
| `-o`, `--out-dir` | next to the CSV | output directory |
| `--model` | the CSV stem | label and output file stem |
| `--positions` | none | sidecar CSV with `GLOBAL CALL COUNT,cur_pos[,page_block_size]` (section 6.3) |
| `--archspec` | `config/tt_bh.yaml` | architecture yaml |
| `--device` | `p100a` | package instance name inside the archspec |
| `--no-device` | off | skip the `Device.execute_op` projection column |
| `--no-figure` | off | skip the PNG |
| `--provenance` | empty, repeatable | PROVENANCE line or lines written into the outputs |
| `--resamples` | module default | bootstrap resamples for the confidence interval |
| `--seed` | module default | bootstrap seed |

It writes `<model>_signed_errors.csv`, `<model>_summary.csv` and `<model>_signed_errors.png`, and exits
0 when every regime meets the acceptance criterion of `recon/gap_analysis.md` section 4.3, 1 otherwise.

Prefill, no sidecar needed:

```bash
cd $POLARIS
ML=$HANDOFF/data/model_level
.venv/bin/python tools/sdpa_validate.py $ML/llama8b_attn_prefill_S4096_ops_perf_results.csv \
    -o $ML/validation --model llama8b_attn_prefill_S4096 \
    --provenance "$(head -1 $ML/llama8b_attn_prefill_S4096_PROVENANCE.txt)"
```

Decode, which needs the sidecar because `cur_pos` is a runtime argument and not an attribute:

```bash
cd $POLARIS
ML=$HANDOFF/data/model_level
.venv/bin/python tools/sdpa_validate.py $ML/llama8b_attn_decode_b32_pos128_1024_4096_ops_perf_results.csv \
    -o $ML/validation --model llama8b_attn_decode_b32_pos128_1024_4096 \
    --positions $ML/positions_llama8b_decode.csv
```

The campaign drove all four model-level runs through one wrapper that lives in the analysis
workspace and needs the internal copy of it:

```bash
cd $HANDOFF/data/model_level/validation
$POLARIS/.venv/bin/python run_validation.py
```

Its outputs are the twelve files beside it in `data/model_level/validation/` (a `_summary.csv`, a
`_signed_errors.csv` and a `_signed_errors.png` per run, for the three prefill runs and the decode
run).

### 9.6 The figures

The figure scripts are tracked on the harness branch under `analysis/campaigns/figs/`, are read-only
on their inputs, write PNGs into the `figs/` directory of the analysis workspace only, and are run
with the polaris venv interpreter because they need matplotlib 3.11 and numpy 2.5 and, in two cases,
import the model in place. They read the measured data through `HANDOFF`, so that variable has to
name the analysis workspace that holds it. Two further figure scripts are internal and are absent
from this branch.

```bash
cd $TTM/analysis/campaigns/figs
P=$POLARIS/.venv/bin/python
$P a_figs_part1.py      # zone tax, anchor decomposition, ablations, residual composition, counters vs zones, per-k-tile, utilization ladder
$P a_figs_part2.py      # DRAM law, causal vs non-causal, chain roles, span histogram, grid and pairs, step swimlane, zone method
$P m_figs.py            # compute floor vs counters, the wall law as blocks, measured vs model per-step terms, every wall predicted vs measured, hold-out errors, attribute flow, decode law
$P m_floor_refit.py     # model FPU, SFPU and MATH union against the counters; needs the internal copy of this script
$P tm_fit_figs.py       # TopK model fit figures
```

`tm_fit_figs.py` takes figure names as positional arguments so a subset can be redrawn:

```bash
cd $TTM/analysis/campaigns/figs
P=$POLARIS/.venv/bin/python
$P tm_fit_figs.py tm_signed_errors
```

Every one of them reads the polaris checkout through the `POLARIS` environment variable, whose
default is `$WORK/polaris` per `PORTABLE_CONTRACT.md`. `tm_fit_figs.py`
originally shipped with a default pointing at a `polaris-topk2` worktree that was merged away and no
longer exists anywhere, so set the variable explicitly when reproducing that figure:

```bash
cd $TTM/analysis/campaigns/figs
POLARIS=$POLARIS $POLARIS/.venv/bin/python tm_fit_figs.py
```

The palette every figure uses, from `handoff/revamp/README.md`: blue `#2a78d6`, orange `#eb6834`,
aqua `#1baf7a`, red `#e34948` on `#fcfcfb`, ink `#0b0b0b`, 150 dpi.

The model-side analysis scripts, tracked beside them under `analysis/campaigns/model/`, follow the
same convention, all read-only on their inputs and all run with the polaris venv interpreter:

```bash
cd $TTM/analysis/campaigns
P=$POLARIS/.venv/bin/python
$P model/floor_verification.py     # writes floor_verification_configs.csv and floor_verification_fits.json into the analysis workspace
$P model/refit_r2_floor_r1f.py     # floor constants fitted on the R1f unmodified-kernel counters, prints tables
$P model/refit_r2_fit.py           # refit and evaluation of every wall of the campaign
```

`floor_verification.py` imports the model from the polaris working tree and monkeypatches it in memory
only for the "refit adopted" columns; nothing under `polaris` is written.

### 9.7 mypy the way CI runs it

CI does not run `mypy` directly. `.github/workflows/checkin_tests.yml` calls the composite action
`.github/actions/run-static-analysis`, whose default command is:

```bash
python checkin_tests.py static
```

`checkin_tests.py` expands that into exactly two commands (`prepare_commands_static`):

```bash
mypy ./
python tools/check_pinned_deps.py
```

The mypy settings are not repeated in `checkin_tests.py`; they are read from `[tool.mypy]` in
`pyproject.toml` (`check_untyped_defs = true`, `explicit_package_bases = true`,
`warn_unreachable = true`, `allow_untyped_globals = true`, and an `exclude` regular expression that
skips `tests/`, `ttsim/back/tensix_neo`, `ttsim/front/llk`, several workload trees and several tools).
The version and stubs come from `envdev.yaml`: `mypy=1.15.0` with `types-PYYAML`, `types-networkx`,
`types-requests`, `types-openpyxl` and `lxml-stubs` from pip.

A bare local invocation is not the same check. `checkin_tests.py` refuses to run unless a conda
environment is active (`check_environment_sanity` requires `CONDA_DEFAULT_ENV`) or `--environment` is
passed, and the uv `.venv` is missing several of the stub packages CI installs, so it can report fewer
errors than CI does. The three usable forms, weakest to strongest:

```bash
cd $POLARIS
.venv/bin/python -m mypy ./                                  # quick local check, stub set incomplete
python checkin_tests.py static -e polarisdev                 # the CI command against the conda dev env
python checkin_tests.py static -e polarisdev -n              # print the commands without running them
```

`checkin_tests.py` writes one log per command into `__RUN_TESTS/logs/` and CI uploads `__RUN_TESTS`
(together with `.coverage`, `__ci/json` and `__ci/html`) as the `checkin_artifacts` artifact of the
workflow run. When a mypy result matters, download that artifact from the GitHub Actions run and read
the log in it rather than trusting a bare local run; the repository's mypy badge is generated from the
same CI step.


## 10. Other architectures

The emulator route for an unreleased part is covered in the internal copy of this runbook, which
stays in the analysis workspace. Nothing in that route is needed for any Blackhole measurement in
this document.


## 11. Gotchas and failure modes

### 11.1 Header toggles need a forced JIT recompile

Symptom: `set_zone_config.sh` is run, the next capture has no `TS_DATA` rows (or an ablation has no
effect on the wall), and nothing in the log says why.

Cause: `sdpa_zone_config.hpp` is a compile-time header. Without a forced rebuild the cached kernel
binaries from the previous process are reused and the new defines never reach the device.

Fix: `TT_METAL_FORCE_JIT_COMPILE=1` on every run. Every runner script in the campaign exports it, and
the `bh/card_log.md` header states it was set "on every zone run so header toggles always take
effect".

```bash
export TT_METAL_FORCE_JIT_COMPILE=1
```

Check afterwards, in the run log or in the PROVENANCE line, that the compiled defines are the ones
intended. Every runner prints them:

```bash
grep -m1 "zone_config:" $DD/<tag>.csv
grep -o "\-DPROFILE_KERNEL=[0-9]*" $DD/<tag>.log | sort | uniq -c
```

The related trap is leaving `SDPA_ZONES` at 1 after a zones-on run. The next zones-off wall is then
silently inflated by the gross zone tax. Every campaign script ends with
`./set_zone_config.sh 0 0 0 0 0`.

### 11.2 Do not resize the BRISC firmware region

Symptom, July 2026: a profiler-enabled build on Blackhole failed to link BRISC firmware, `brisc.elf`
coming out at 0x2204 bytes against the 0x2200 region, when all counter groups were asked for in one
pass. The response at the time was to bump `MEM_BRISC_FIRMWARE_SIZE` from `(6 * 1024 + 2560)` to
`(6 * 1024 + 3072)` in `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h`.

Do not carry that bump. It is no longer needed and it breaks profiler runs. Two measurements, 2026-09-16:

- The overflow is gone. With the profiler on, `brisc.elf` builds to 6,904 bytes against the stock
  8,704 region, 1,800 bytes of headroom, and NCRISC and the three TRISCs are all under 2,560. PR 55166
  removed the cause by capping a pass at three counter groups, so nothing asks for the large image any
  more.
- The bump breaks device init. `MEM_NCRISC_FIRMWARE_BASE` and the TRISC bases are derived from
  `MEM_BRISC_FIRMWARE_SIZE`, so changing the header moves every downstream base for JIT-built
  firmware while a host library built before the change still expects the old ones. With the bump in
  place and a cold JIT cache, a profiler run dies in
  `RiscFirmwareInitializer::initialize_and_launch_firmware` with a 10 s timeout on all 119 worker
  cores and `Device 0 init: failed to initialize FW! Try resetting the board.`; a board reset does not
  help, and the same op with the profiler off runs fine. Reverting the header to `(6 * 1024 + 2560)`
  fixes it immediately.

The bump was inert for most of the campaign because the firmware objects were cached from before the
edit; it only bites once `~/.cache/tt-metal-cache` is cleared. If you inherit a workspace that still
carries it and see that init timeout, revert the header first.

A second, independent firmware constraint lives in the perf-counter path: at most three counter groups
fit in one BRISC firmware image (four overflow `.text` by 4 bytes, five by 16), which is why
`PERF_COUNTER_MAX_GROUPS_PER_PASS` is 3 and why `all` needs multipass.

### 11.3 The wall definition

Symptom: walls that disagree with the published ones by 13 to 50 percent, or by 3.5 to 4 percent.

Two separate basis choices caused this.

The first is which span counts as the wall. The campaign basis is the device wall,
`max(KERNEL zone end) - min(KERNEL zone start)` over all cores and all RISCs, which
`analysis/zone_reduce.py` computes as `wall_dev_cycles`. An earlier analysis used the median per-core
KERNEL span instead, which is 13 to 50 percent below it (`bh/targets.md` section 4 item 4). The
wall-setting core's own span equals the device wall in this campaign, so "slowest core" and "device
wall" are one basis; the only remaining choice is a mean-core floor against a wall-core floor, and the
campaign reports the wall-core basis.

The second is the harness form. A loop harness that calls the op three times in one process runs 3.5
to 4 percent slower at every sequence length than a single-op harness, and non-causal loop rows scatter
2 to 3 percent across invocations where single-op runs repeat within 0.3 percent
(`bh/targets.md` section 3). The campaign therefore fixes one form: three invocations in one process,
invocation 0 discarded, mean of invocations 1 and 2, and says so in every PROVENANCE line. Do not mix
a loop wall with a single-op wall in one table.

A third basis trap sits underneath both: the firmware bundle. Today's card at bundle 19.9.0 reproduces
the July loop wall within 0.05 percent but is 3.6 percent (causal) and 2.5 to 4.3 percent (non-causal)
slower than the walls taken on 2026-09-02 at bundle 19.12.0-rc.1. Per-core TRISC1 mean spans agree
within 0.5 percent; the whole offset sits in the straggler core. Firmware as the cause is inferred.
Read the bundle out of the run log before comparing to any earlier number:

```bash
grep -m1 -o "firmware bundle version: [0-9a-z.-]*" $DD/<tag>.log
```

### 11.4 Multipass emits no ops report

Symptom: a `--perf-counter-multipass` run ends with "no ops report" and
`generated/profiler/reports/` is empty or holds nothing useful, even with `-r` on the command line.

Cause: the multipass scheduler replays the workload once per pass and merges the per-pass device logs;
the ops-report post-processing step does not survive that. `bh/card_log.md` rows 17 and 19 both record
it, and `topk/indexer_points.md` states it as a property of the tool.

What to do instead: read the merged device log. It carries the KERNEL zones and every counter row.

```bash
cd $TTM
python analysis/post_process.py generated/profiler/.logs/profile_log_device.csv --seq-lens 4096 --out /tmp/counters.csv
tail -n +2 /path/to/archived_capture.csv > /tmp/zr.csv && python analysis/zone_reduce.py /tmp/zr.csv --out /tmp/mytag
```

Also archive the per-pass snapshots, because they are the only per-pass record:

```bash
TAG=mytag
cp -r generated/profiler/.logs/perf_counter_passes $TOPKOUT/perf_counter_passes_$TAG
```

The practical consequence for a campaign: take the timing from a plain run and the counters from a
separate multipass run of the same configuration, and check that the two walls agree. At the causal
anchor they agree within 0.15 percent; the T2.1 table carries both `wall_zoff` and `wall_mp` per
configuration for exactly this check.

### 11.5 The indexer weights layout changed on main

Symptom: the indexer cell of the TopK campaign fails with a `TT_FATAL` (surfacing as
`RuntimeError`) at `indexer_score_device_operation.cpp:593`, and with it the whole `COUNTERS` group
aborts before the ops report is written.

Cause: main tip requires the weights tensor in layout `[B, 1, Sq, Hi]`. `analysis/indexer_probe.py`
builds `[1, Hi, Sq, 1]`, the layout the op took at the `73e0e25` calibration state (PR 47223
restructured it).

Fix, two parts. Skip the cell in the campaign group:

```bash
cd $TTM_FRESH
python -m tracy -r -p --perf-counter-multipass --profiler-capture-perf-counters=all \
    $TTM/analysis/campaigns/topk_campaign.py \
    --classes COUNTERS --exclude-op indexer --out $TOPKOUT
```

and take the indexer points on the calibration checkout instead, where the probe's layout is the one
the op expects (section 7.6). The (2048, 8192) point there matches the 2026-09-02 probe within 0.2
percent, so that kernel is unchanged; the main-tip op simply could not be timed with this probe. To
time it on main the probe's weights construction has to be updated to `[B, 1, Sq, Hi]` first.

### 11.6 The tracy report writer races the copy

Symptom: the ops report copied out of `generated/profiler/reports/<date>/` is empty (or the campaign
script reports "no ops report") even though tracy exited cleanly and the directory exists.

Cause: the report writer finishes after tracy's own exit, so a copy issued immediately after the
process ends can capture a zero-length file.

Fix: sleep before copying, and check the size. Every runner in the campaign has the sleep:

```bash
TAG=mytag
R=$(ls -d generated/profiler/reports/*/ 2>/dev/null | sort | tail -1)
sleep 2
if [ -n "$R" ] && ls $R/ops_perf_results*.csv >/dev/null 2>&1; then cp $R/ops_perf_results*.csv $DD/${TAG}_ops_perf_results.csv; fi
```

It still bit once. The T2.5 decode ops report was captured empty by `campaign_t25.sh`, and the file in
`data/model_level/` is a manual copy made afterwards. Its `_PROVENANCE.txt` records both the source
directory and the fact, with the calibration tree written out in full where `$TTM` stands below:

```
NOTE: ops report copied manually from $TTM/generated/profiler/reports/2026_09_12_00_51_09 at 2026-09-12T00:52:21Z (the campaign script's copy raced the report writer and captured an empty file)
```

So after any block that produces ops reports, check for empty files and copy by hand from the report
directory if needed:

```bash
find $HANDOFF/data -name "*_ops_perf_results.csv" -size -1k
ls -l $TTM/generated/profiler/reports/*/ops_perf_results_*.csv
```

`data/model_level/reduce_ops.py` guards against it too: it prints `EMPTY <path>` and skips any ops CSV
smaller than 10 bytes.

### 11.7 Model-level attention runs need a writable cache and a self-contained fixture set

Two setup failures cost five runs on 2026-09-12 (`bh/card_log.md` row 16a):

- `analysis/attn_prefill_perf.py` and `analysis/attn_decode_perf.py` originally referenced the
  tt_transformers test-directory fixture `ensure_gc`, which is not visible from `analysis/`, so pytest
  failed in setup with no device op executed.
- The weight cache defaulted into the read-only weights directory, so the run raised `PermissionError`
  after loading the weights.

Fix for the second, which is the one a new user will hit:

```bash
export TT_CACHE_PATH=$HANDOFF/data/model_level/tt_cache
```

### 11.8 Other traps recorded during the campaign

- The zone set changed once mid-campaign. The first T2.1 decomposition showed about 550k un-zoned
  PACK-thread cycles, traced to the two per-k-chunk `exp_packthread_tile_init` calls and the
  `cur.sum` and `out_cb` `push_back` pairs, which had no zone. Slots 21 `EXP_INIT` and 22 `PUSHES`
  were added, the diff regenerated, the three interrupted anchor outputs deleted and the campaign
  restarted from the anchor with the 23-zone compute set. Any decomposition taken with the 21-zone set
  is not comparable.
- Raw zones are dropped once the per-RISC 125-pair buffer is full, at which point they cost 11.5 cycles
  instead of about 52 to 57. Keep the per-q-chunk raw zone count inside the buffer (10 to 19 pairs per
  RISC at the campaign's shapes) or the timeline silently truncates.
- `ref_cnt` is packed into 24 bits while the hardware counter is 32, so kernels longer than about
  12 ms at 1.35 GHz wrap. `load_counters` recovers the full value from the paired profiler id 9091
  records and sets `ref_cnt_full_recovered`; check that column before trusting a percentage on a long
  kernel.
- `SDPA_PACKER_L1_ACC` has no effect on the SDPA prefill op. The program factory reads only
  `fp32_dest_acc_en` and the fidelity out of the compute config, so the runs that set
  `SDPA_PACKER_L1_ACC=1` are equivalent to the production setting of `packer_l1_acc=False`. The knob is
  kept because the production brief mentioned it.
- The non-streaming (fp32 DEST) kernel has drifted on main and the streaming one has not. At the causal
  anchor the fresh main-tip build reproduces the calibration wall within 0.09 percent (non-causal 0.23
  percent), but the production fp32 path is 1.54 percent slower at S4096 and 7.02 percent slower at
  S1024 with q = k = 64. Constants fitted on the calibration checkout carry over to main for the
  streaming path only; the production path needs its own fit on main.
- `ttnn.transformer.sparse_sdpa` gained a required `kv_format` argument on main. Use
  `analysis/sparse_sweep_fresh.py` on the fresh tree (section 3.8).
- The `mask` harness runs a mask-generation op per iteration, and the sparse and joint harnesses can
  run extra ops, so a `_runs.csv` from those harnesses holds more invocations than the op count. The
  reducers filter with `wall_dev_cycles > 0.2 * max(wall_dev_cycles)`; apply the same filter in any new
  analysis.
- `DEC_KV_DTYPE` defaults differ between the two decode harnesses: `bfp8_b` in
  `analysis/decode_sweep.py`, `bfloat16` in `analysis/r1_decode.py`. Always set it explicitly.
- `checkin_tests.py` refuses to run without an active conda environment unless `--environment` is
  given, so it cannot be driven straight out of the uv `.venv` (section 9.7).

### 11.9 Two path traps in a fresh workspace

Neither trap announces itself, because the scripts accept whatever the environment already holds, and
both bite hardest on a workspace that was just cloned per section 1.1.

The first is `TT_METAL_HOME` pointing at the wrong checkout. A runner exports it from `TTM` or
`TTM_FRESH`, but a `TT_METAL_HOME` left over in the shell from an earlier paste, or a `TTM` that was
never re-exported, keeps naming whatever tree it named before. The two trees hold different kernels by
design, so the harness then imports `ttnn` from the wrong one, JIT compiles against the wrong kernels,
and writes `generated/profiler/.logs/` under it. If the stale value names a tree that is not the
reader's, the run also fails with `Permission denied` on the profiler log; if it names the reader's
other checkout, the run is clean and the numbers are the other kernel state's, which is the expensive
case, because the drift between the two trees is exactly what sections 11.3 and 11.8 measure. Check
both the variable and the module that was actually loaded, in the process that will touch the card,
and check the sha the PROVENANCE line recorded afterwards:

```bash
echo "TT_METAL_HOME=$TT_METAL_HOME  TTM=$TTM  TTM_FRESH=$TTM_FRESH"
python -c "import ttnn; print(ttnn.__file__)"
grep -m1 "sha=" $DD/<tag>.csv
```

The second is the `python_env` activation. `PYENV` defaults to `$TTM/python_env/bin/activate`, that
is, the venv inside the calibration checkout, and there is exactly one such venv in the whole workspace
(section 1.7): `tt-metal-fresh` has none and the two fresh-tree runners activate the calibration tree's
venv by that path. A fresh clone has no `python_env` at all until `create_venv.sh` has run, so a runner
fails at the `source` line with "No such file or directory", and a venv built somewhere else is found
only if `PYENV` is set to it:

```bash
ls -d $TTM/python_env || (cd $TTM && ./create_venv.sh --env-dir $TTM/python_env)
export PYENV=$TTM/python_env/bin/activate      # only needed if the venv is not inside $TTM
source $PYENV && python -c "import ttnn; print(ttnn.__file__)"
```


## 12. Provenance: the data folders and what each holds

Commit equivalence: this branch was assembled from the calibration tree with the commit messages
cleaned up, so its shas differ from the ones in the measurement PROVENANCE lines while the trees are byte
identical. The calibration state cited as commit 72620d5332967 is commit aae92de5471 here, and the same
tree is still reachable as 72620d5332967 on branch mvlahovic/analyze_single_chip_sdpa.

### Reference tables in this branch

`analysis/reference_tables/` carries the reduced results of the campaign (148 KB, five files plus a
README): every device wall with its configuration, the same walls priced by the model with signed
error, the perf counter readings per configuration, the measured wall against the model's named
terms, and the fitted floor constants. Diff a rerun against those instead of against prose. The raw
captures behind them are 2.4 GB and stay out of git, as the inventory below says.

Where the data came from. Every measurement inventoried below was produced on 2026-09-11 and
2026-09-12, on one reserved Blackhole p100a at firmware bundle 19.9.0, and the published pages cite
those files. Section 1.1 clones the code that produced them, but there is nothing to clone for the
data: the raw dumps are 2.4 GB and are not in git, so they stay in the analysis workspace on the
shared filesystem, which is what `$HANDOFF` points at in the paths below.

Every raw CSV in this campaign carries a PROVENANCE line as its first line, with the card, firmware
bundle, git sha and branch, the UTC timestamp, the exact command, the environment and the compiled zone
config. Every derived table carries a `# PROVENANCE` comment line naming its inputs and the script that
wrote it. `bh/card_log.md` is the append-only index of every device run.

| Path | Size | Contents |
|---|---|---|
| `handoff/revamp/data/bh_zones/` | 2.0 GB, 1230 files (963 CSVs) | every SDPA capture and its reductions. Raw `<tag>.csv` (PROVENANCE plus the device log) with `<tag>.log`, `<tag>_runs.csv`, `<tag>_cores.csv`, `<tag>_raw.csv`, `<tag>_counters.csv` and, where tracy ran with `-r`, `<tag>_ops_perf_results.csv`. Tag prefixes: `t01_` T0.1 anchor reproduction, `t02_` zone tax, `t21_` grid 1, `t22_` ablations, `t23_` DRAM law, `t23r_` regimes, `t24_` production, `t27_` fresh drift, `t28_` decode, `r1a_` to `r1g_` the R1 re-measurement. Derived: `decomp_<tag>.csv`, `decomp_summary.csv`, `report_tables.md`, `dram_rate_table.csv`, `ablation_deltas_q128k*.csv`, `decode_sweep_table.csv`, `r1_walls.csv`, `r1_tables.md`, `r1b_decode_table.csv`, `r1f_counters_table.csv`, `r1f_table.md`, `r1g_thread_split.{csv,md}`, `prod_parts_table.md`, `t02_zone_tax_summary.csv`. Also the block outputs `t21_campaign.out`, `t22.out`, `r1.out`, `r1_add.out`. The runners, campaign scripts and reducers that wrote all of it are no longer here: they are in git, in `analysis/campaigns/` on `mvlahovic/sdpa_topk_harness` |
| `handoff/revamp/data/model_level/` | 399 MB | the four T2.5 tracy runs: `<run>_ops_perf_results.csv`, `<run>_profile_log_device.csv`, `<run>_PROVENANCE.txt`, `<run>.log`; `sdpa_ops_summary.csv` and `reduce_ops.py`; `positions_llama8b_decode.csv` (the decode sidecar); `tt_cache/` (the Llama weight cache); `validation/` with `run_validation.py` and the eight per-run validation outputs; `README.md` documenting the runs |
| `handoff/revamp/data/topk/` | 780 MB, 89 entries | the T2.6 TopK campaign: per class group `cells_<TAG>.csv`, `results_<TAG>.csv`, `ops_perf_results_<TAG>.csv`, `profile_log_device_<TAG>.csv`, `run_<TAG>.log`, `PROVENANCE_<TAG>.txt`; the three `perf_counter_passes_*` directories with `pass_0.csv` to `pass_4.csv` each; the T2.9a indexer points `t29_indexer_*` plus `indexer_points.csv`; the fits `topk_fits.csv`, `topk_acceptance.csv`, `topk_model_acceptance.csv` |
| `handoff/revamp/data/` (top level) | 148 KB | in git as `analysis/reference_tables/` on `mvlahovic/sdpa_topk_harness`, so a rerun can be diffed against them without the workspace: `targets_table.csv` (the target walls the campaign had to reproduce), `model_components_grid1.csv`, `floor_verification_configs.csv` and `floor_verification_fits.json` (written by `model/floor_verification.py`), `refit_r2_walls.csv`, and the earlier task script sets `t15_scripts/` and `t42_scripts/` |
| `handoff/revamp/bh/` | 752 KB | the Blackhole reports and the build record: `card_log.md` (230 rows), `zone_decomposition.md`, `campaign_r1.md`, `production_config.md`, `regimes.md`, `decode_sweep.md`, `drift_check.md`, `targets.md`, `sdpa_kernel_phase_map.md`, `fresh_build.md` with `fresh_build.log` and `fresh_build_rebuild_memmap.log`, and `zone_patch.diff` |
| `handoff/revamp/topk/` | 348 KB | the TopK plan and driver: `sweep_plan.md`, `topk_campaign.py`, `topk_analyze.py`, `topk_model_acceptance.py`, `topk_campaign_results.md`, `indexer_points.md`, `generic_topk_kernel.md`, `router_topk_kernels.md` |
| `handoff/revamp/model/` | 456 KB | the model notes and the offline fit scripts: `restructure_notes.md`, `refit_notes.md`, `refit_r2_notes.md`, `floor_verification.md` with `floor_verification.py`, `refit_r2_fit.py`, `refit_r2_floor_r1f.py`, `attr_plumbing_spec.md` and `topk_model_notes.md`, plus one further offline script that is internal and is not reproduced here |
| `handoff/revamp/figs/` | 18 MB | every PNG of the pages plus the figure scripts and the per-page figure notes `a_figures_part1.md`, `a_figures_part2.md`, `m_figures.md` and `t_figures.md`; two of the figure scripts are internal and are absent from this branch |
| `handoff/revamp/recon/`, `review/`, `pages/` | 588 KB, 256 KB, 12 MB | the recon reports that scoped the campaign, the review and audit reports, and the published page sources with their check and manifest files |
| `$HANDOFF/../investigations/data/` | 694 MB | the earlier (TEN-4858, TEN-4859) captures and `parse_profile.py`, the third counter reader used in section 5.4. Beside the analysis workspace on the shared filesystem, not in git |
| the internal validation data directory (INTERNAL) | 416 MB | not in git, and the input of the two internal scripts of section 3.20. It stays in the analysis workspace on the shared filesystem: the earlier validation matrix and the fitted constants that `analysis/validate_roofline.py` reads and writes, plus the earlier sweeps and their logs |

Code locations, restated because they matter for reproduction:

- In git, on the pushed branch `mvlahovic/sdpa_topk_harness` of tt-metal (`$TTM`): the kernel
  instrumentation, the toggle header, all 28 harness entries of section 1.4, and the runners, campaign
  scripts, reducers, `topk_campaign.py` and the `figs/` and `model/` script sets under
  `analysis/campaigns/`. The same instrumentation is
  also recorded as `bh/zone_patch.diff` for porting it to another base (section 4.2).
- In git, on the pushed branch `mvlahovic/analyze_sdpa_fresh` of tt-metal (`$TTM_FRESH`): the main-tip
  merge, the BRISC firmware size commit, the 19 harness entries that run on main tip and
  `analysis/sparse_sweep_fresh.py`, which is on no other branch.
- In git, on the pushed branch `mvlahovic/roofline_model_topk` of polaris (`$POLARIS`, PR 530, with
  `mvlahovic/roofline_model_sdpa` as PR 493): `ttsim/perf/roofline_sdpa.py`,
  `ttsim/perf/roofline_topk.py`, `ttsim/perf/sdpa_validate.py`, `tools/sdpa_validate.py` and the three
  test files under `tests/test_perf/`.
- Not in any git repository, analysis workspace only: the reports of sections 4 to 12,
  `$HANDOFF/data/model_level/reduce_ops.py` and `validation/run_validation.py`, the two offline TopK
  analysis scripts beside `topk/sweep_plan.md`, and all of the measured data.
- Internal and in no repository by design: `analysis/roofline.py` and `analysis/validate_roofline.py`
  (sections 1.1 and 3.20) with their validation-data inputs, two of the figure scripts, and the
  material for the other architectures of section 10.


## 13. Appendix: what in this runbook could not be verified against a file

This runbook was written without running anything on the card and without editing any repository.
Everything above was read out of a script, a harness source file, a run log, a report, a PROVENANCE
line or a git object in this workspace, with these exceptions.

- No device command was re-executed. The SDPA, TopK and model-level commands are transcribed
  from the runner scripts, the campaign scripts and `bh/card_log.md`, not re-run. The host-side
  commands that were executed while writing this are the read-only ones: the git queries of section
  1, the `CMakeCache.txt` greps of section 2, `topk_campaign.py --list`, the three polaris `predict`
  calls of section 9.3, `pytest --collect-only` on `tests/test_perf`, `mypy --version` and `pip list`.
- The exact command that created the polaris `.venv` is not recorded anywhere. `pyvenv.cfg` proves it
  was uv 0.11.32 with CPython 3.12, and the installed set was read with `pip list`, but the
  `uv pip install` line in section 9.1 is reconstructed from `envdev.yaml` and is a superset of what
  the `.venv` actually holds.
- The per-block durations in section 8.2 are spans between the first and last logged run start of that
  block in `bh/card_log.md`. They exclude the last run's own duration, and two blocks
  (`campaign_t21.sh`, `campaign_t24_zoff.sh`) are logged as a single row, so their spans are bounded
  by the neighbouring blocks rather than measured directly.
- `campaign_r1.sh` issues 64 device runs by reading the script (1 start anchor, 42 R1a, 1 mid anchor,
  5 R1b, 11 R1c, 2 R1e, 1 end anchor, 1 production end point). `bh/card_log.md` carries 67 rows across
  blocks R1a to R1e plus one row labelled only "R1 R1" with no timestamp; two of the R1b rows are the
  `campaign_r1_add.sh` addendum, leaving one row unaccounted for. The discrepancy was not resolved.
- The `run-static-analysis` CI action and the `checkin_artifacts` artifact of section 9.7 were read
  from the workflow and action YAML in the checkout. No CI run was fetched, so the statement that a
  bare local `mypy` can report fewer errors than CI rests on the package-set difference between the
  `.venv` and `envdev.yaml`, not on a compared pair of logs.
- `analysis/campaigns/` is the layout the runners, the campaign block scripts, the reducers and
  `topk_campaign.py` are being moved into on `mvlahovic/sdpa_topk_harness`. The branch content that
  was verified with `git ls-tree` is the instrumentation and the 28 harness entries under `analysis/`;
  the campaign scripts were still outside the tree when this was written. Run
  `git ls-tree -r --name-only HEAD analysis/campaigns/` once after cloning, and if the directory is
  empty, the scripts are still in `data/bh_zones` of the analysis workspace and the `cd` in front of
  every runner command becomes `cd $DD`.
- The two tt-metal branches were read as local branches and their shas are from
  `git log`. `git ls-remote --heads origin` confirmed `mvlahovic/roofline_model_sdpa` and
  `mvlahovic/roofline_model_topk` on their remotes; the push of
  `mvlahovic/sdpa_topk_harness` and `mvlahovic/analyze_sdpa_fresh` had not landed yet when this was
  written, so run the `ls-remote` check of section 1.1 before assuming a clone can reach them.
