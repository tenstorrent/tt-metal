# DeepSeek-V4.1-Flash prefill on Tenstorrent: start here

For a Claude Code session (and its human) picking up this model. It covers setup, running the model, measuring
accuracy and reproducing the perf numbers, and links to the deeper docs. Facts are as of 2026-11-06, branch
`dnijemcevic/model_bringup` (= `dnijemcevic/model_bringup_rebased`, tip 52107c12262). Items marked **(unverified)**
were not checked against code or a run while this was written.

## 1. What this is

This is chunked prefill of `deepseek-ai/DeepSeek-V4.1-Flash` (text decoder only: no vision and no MTP), all 40 layers,
on a Blackhole Galaxy 8x4 (32 x p150b, `FABRIC_2D_TORUS_XY`, SP = 8 mesh rows, TP = 4 mesh columns). The work is
organized by the gated bring-up framework (`models/demos/common/bringup`). The model code is in
`models/demos/deepseek_v41_flash_d_p/tt/`. The tt-d-gen inference server drives it through the adapter in
`tt/runners/`.

| Item | State |
|---|---|
| Deployment box / spec | Galaxy 8x4: `models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml` (ledger: tasks.yaml, state.json, results/ in that folder) |
| Seed run | QuietBox 2x2, partial (layers 0-3, 14, 20, 21, 24, scaled model): `bringup/spec.yaml`, ledger in `bringup/` |
| Accuracy | X.3 PASS (s56320 ladder, every limit holds). Latest (P.34 gate, rung `last`): err_final_hidden 0.139 (limit 0.242), top1 1.0, top5 1.0 |
| Perf (8x4, 51200, resident engram, P.34 on) | clean warm 5120-token chunk wall ~180-185 ms (median 184.4); device ~166 ms; host-bound |
| Full 56k prefill (X.3 on the rebased build) | 2503 ms (old build 2891) |
| Serving contract (K.1R) | passes against tt-d-gen main d8a194417, under two owner-signed waivers (section 7) |
| In flight | P.34 (POST_COLLAPSE) gate. Then P.35 / P.36 (held), O.1 / Z.1 / F.1 (held) |

## 2. Setup on a new Galaxy

Everything under `/scratch` is node-local. A new box has none of it. The shared masters are on NFS `/data` and are
readable by other users.

```bash
# 1. Checkout (origin = github.com/tenstorrent/tt-metal); steps 1-3 were not re-run for this doc (unverified as a sequence)
git clone https://github.com/tenstorrent/tt-metal.git && cd tt-metal
git checkout dnijemcevic/model_bringup
git submodule update --init --recursive
# tt_ops_code_gen has update=none in .gitmodules. Without it the fork tests that import `eval` fail (the root symlink eval -> it)
git -c submodule.tt_metal/third_party/tt_ops_code_gen.update=checkout submodule update --init tt_metal/third_party/tt_ops_code_gen

# 2. Build: Release, no tests (the build_metal.sh defaults). Afterwards `build` must be a symlink to build_Release.
./build_metal.sh
ls -l build            # build -> build_Release

# 3. venv (./python_env). If ~/.cache/uv is damaged, put uv's cache on /scratch (UV_CACHE_DIR: uv's own variable, unverified here):
UV_CACHE_DIR=/scratch/$USER/uv-cache ./create_venv.sh

# 4. Every session, from the repo root
export PYTHONPATH=$PWD && source python_env/bin/activate
S=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml
```

The data paths come from the spec. `paths.art` is `/scratch/$USER/bringup`, and the framework appends the model
name (`core/spec.py:art`):

| What | Where | Size | How it gets there |
|---|---|---|---|
| HF checkpoint, rev `5254448103b6254fc646dacbca1cbbf20e9216c1`, all 48 shards | `/scratch/$USER/bringup/deepseek_v41_flash_d_p/hf_full` | 476 GiB | You download it once (below). Nothing downloads it automatically. Dense weights and the engram tables are always read from it |
| Goldens (master) | `/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4` (s10240_c5120, s56320_c5120, plus `-official`) | 85 GB | Fixed |
| Goldens (node-local mirror tests read) | `/scratch/$USER/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4` | 85 GB | `reference.golden sync`. The orchestrator syncs at run/resume start. When the mirror is incomplete, tests read NFS (slow) and print a warning |
| Routed-expert weight cache (master) | `/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/tt_cache/8x4/v1` (bank-major `.bm.bin`, bfp4_mse) | 306 GB, 120 files | Fixed |
| Weight cache (node-local mirror the loader reads) | `/scratch/$USER/bringup/deepseek_v41_flash_d_p/tt_cache/8x4/v1` | 306 GB | `core.weight_cache pull`. The orchestrator pulls at run/resume start. A miss rebuilds that layer from the checkpoint (~2-3 min a layer, DS41-121) and writes the cache |

```bash
# checkpoint (the recipe used here: /scratch/dnijemcevic/hf_dl.py; ~7 min on this node's network (unverified elsewhere))
python -c "from huggingface_hub import snapshot_download as d; d('deepseek-ai/DeepSeek-V4.1-Flash', \
  revision='5254448103b6254fc646dacbca1cbbf20e9216c1', \
  local_dir='/scratch/$USER/bringup/deepseek_v41_flash_d_p/hf_full', max_workers=8)"

python -m models.demos.common.bringup.reference.golden status --spec $S      # which dir each rung reads
python -m models.demos.common.bringup.reference.golden sync   --spec $S      # master -> mirror, verified
python -m models.demos.common.bringup.core.weight_cache status --spec $S     # to pull / to push, with a time estimate
python -m models.demos.common.bringup.core.weight_cache pull   --spec $S     # ~50 min at the code's 100 MB/s NFS estimate
```

- The local disk needs about 870 GB: checkpoint, cache and goldens.
- Old `.tensorbin` expert caches are not read any more. To convert one (host only, ~8 min):
  `python models/demos/deepseek_v41_flash_d_p/tools/convert_expert_cache.py --layers $(seq 0 39) [--verify]`.
- `ENGRAM_TABLE=resident` needs ~190-210 GB of host RAM for the two engram tables (the Galaxy host has 566 GB).
- `push` is for the master's owner only. A user who cannot write the master only pulls.

## 3. Run the model

The rules in section 7 apply to every command below. Each device test loads the model: about 1.5-3.5 min with a
warm weight cache (measured model_load_s 82 s and 205 s), plus about 1.5-2.5 min of first-chunk compile.

```bash
# "What is the capital of France?" -> Paris, on the device model (no server)
BRINGUP_SPEC=$S scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_smoke.py

# accuracy: ladder rung `last` (golden prefix 0-51200 loaded, last 5120-token chunk on device, every metric vs the golden)
BRINGUP_SPEC=$S BRINGUP_RESULTS_DIR=/scratch/$USER/ds41_results BRINGUP_TASK=mine BRINGUP_RUNG=last \
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py

# the full 56k prefill, all 11 chunks on device, ~8 min (X.3: 492 s)
BRINGUP_SPEC=$S DS41_ENGRAM_TABLE=resident BRINGUP_RUNG=s56320 \
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py
```

**Serving runner (tt-d-gen).** These are the K.1R tests. They need a tt-d-gen main build (d8a194417, whose H2D stream
descriptor is v3, matching this tt-metal) at the spec's `serving.server_repo`, `/scratch/$USER/tt-d-gen`. The
engine's Python is found at `/scratch/$USER/tt-d-gen-build/venv312/bin/python` (`testing/dgen_engine.py:find_build`).
`BRINGUP_SERVER_REPO=<path>` points at another checkout.

- How to build it: `/scratch/dnijemcevic/tt-d-gen-build/README.md` (this node only). The short version:
  `./build_dgen.sh --bindings --blaze --no-kvm-client`. That build fails at tt-blaze's BFP packer because clang-20
  here has no OpenMP. Build the packer with `CC=g++-12 CXX=g++-12 LDSHARED="g++-12 -shared"`, then re-run with
  `--skip-blaze-build`.
- Alternatively, the `dgen-build` agent (`models/demos/common/bringup/agents/dgen-build.md`) builds it.
- Check the build with: `BRINGUP_SPEC=$S python -m models.demos.common.bringup.testing.dgen_engine find`.

```bash
T=models/demos/deepseek_v41_flash_d_p/tests/bringup/contract
BRINGUP_SPEC=$S scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_contract.py
for t in test_kv_layout test_adapter_acks test_runner_contract test_runner_smoke; do   # smoke ~12 min: Paris via tt-d-gen
  BRINGUP_SPEC=$S scripts/run_safe_pytest.sh --run-all --no-precompile $T/$t.py || break; done
```

## 4. Accuracy

There is no fixed bar. Every gated metric has a limit of `e2e.calib` (2.0) x the official reference's own drift
from full precision. The golden is the fp32 reference. The drift comes from the bf16/fp8/fp4 reference
(`<rung>-official`, `drift-<rung>.json` beside the golden). The code is in `testing/e2e_limits.py` and
`reference/drift.py`.

The ladder (`testing/ladder.py`) writes `$BRINGUP_RESULTS_DIR/$BRINGUP_TASK.json`, by default
`generated/bringup_adhoc/adhoc.json`. Every metric has `{t, value}`, and every gated metric has a `<name>_limit` next
to it. The test fails if any gated metric is over its limit, unless an owner waiver (`tests.waivers`) covers it.

| Metric | Meaning | Latest |
|---|---|---|
| `err_layer_L00..L39` | per-layer output rel L2 (`pcc_layer_*` is diagnostic only) | ~0.02 (L0) to ~0.16 (L18), about half of each limit |
| `err_state_<cache>_L*`, `overlap_state_*` | the served caches: window_kv, compress_kv, index_k, topk_idx (overlap), candidates, engram_tail | all within limits. The rebase moved window_kv L20-37 (explained: one bf16 ulp in one routing weight; see REBASE_STATUS.md) |
| `err_final_hidden` | final norm output rel L2 | 0.139 / limit 0.242 (P.34). X.3: 0.145 / 0.259 |
| `top5_overlap`, `top1_match`, `pcc_logits_tail` | token-level agreement of the last rows' logits | 1.0, 1.0, ~0.9935 |

- Rungs: `s10240` and `seed` (10k), `last` (golden 51k prefix + 1 chunk, the fast check), `s56320` (the full 56k).
- Token-level study: `tests/bringup/test_logit_kl.py` (FIX.20).
- The precision policy is realistic precision only: there is never a max-precision device run. Its rationale is in
  `models/demos/common/bringup/README.md` ("Switches") and in the spec's `agents.rules`.

## 5. Perf: reproduce the headline numbers

All timings use `DS41_ENGRAM_TABLE=resident`. The tests default to `mmap` (`tt/settings.py`: `test="mmap"`, which
gives bit-identical rows but a slower host), so an untagged run is not a perf number. Per the X.3 notes, mmap made
the wall 226-245 ms instead of ~173-185 ms.

```bash
P="BRINGUP_SPEC=$S BRINGUP_RESULTS_DIR=/scratch/$USER/ds41_perf BRINGUP_TASK=perf DS41_ENGRAM_TABLE=resident"
# (a) clean warm wall: compile, warm until no new program and the walls settle, then 10 timed runs -> chunk_wall_ms (median)
env $P BRINGUP_POS=51200 BRINGUP_PROFILE_WALL=1 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py
# (b) profiled run (program real-time profiler): device_ms_total, device_ms_<section>, device_ms_chip<c>; keeps (a)'s wall
env $P BRINGUP_POS=51200 BRINGUP_WALL_CLEAN=1 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py
```

- **Run (b) after (a).** It reads (a)'s `chunk_wall_ms` from the same results file, so it needs the same
  `BRINGUP_TASK` and `BRINGUP_RESULTS_DIR`.
- **`BRINGUP_POS=<start>`.** This times one chunk at that start with a zero KV prefix and random ids. No golden is
  needed, and metrics get the suffix `_p<start>`. Use 51200 for 50k->55k and 512000 for 500k->505k.
- **Without `BRINGUP_POS`.** The profile runs the `last` rung's chunk on the golden prefix and also records
  `pcc_chunk_out`.
- **`BRINGUP_PROFILE_HOST=1`.** Adds 5 runs under cProfile (`<task>_host_p<pos>.json/.pstats`).
- **Full 56k prefill.** Add `BRINGUP_FULL_PREFILL=1 BRINGUP_PROFILE_OPS=1` to a profile run (X.3's form, without
  `BRINGUP_POS`). This records `prefill_ms_full` (warm, no readback) and per-op rows (`<task>_ops.json.gz`).
- **Several start positions.** `test_positions.py` times 0, 51200, 102400, 153600 and 204800 (spec `perf.positions`,
  default 0 and 1-4 x the last chunk's start). It records `pos_ms_<start>`.
- **One process for several runs.** A gate runs several such commands in one process, so the model loads once
  (`tests.one_process`, `core/batch.py`):
  `BRINGUP_BATCH_CMD='<cmd1> && <cmd2>' scripts/run_safe_pytest.sh --no-precompile --run-all models/demos/common/bringup/tests/test_batch.py`.
- **Profile outputs.** These are `<task>_profile.json` (sections, timeline) beside `<task>.json`.

| Number (8x4, rebased build, P.34 on, resident) | Value | Source |
|---|---|---|
| clean warm chunk wall at 51200 | 184.4 ms median (runs 179-198) | `/scratch/dnijemcevic/rebase/p34on_wall/P.34.json` |
| device per chunk at 51200 | 166.5 ms; the chunk is host-bound by ~15-18 ms (wall - device: the engram layer-1 wait is 8-12 ms of it) | `/scratch/dnijemcevic/rebase/p34on/P.34.json`; host breakdown `/scratch/dnijemcevic/rebase/host/NOTES_host_p51200.md` |
| full 56k prefill | 2503 ms | REBASE_STATUS.md (X.3 re-run, before P.34; mmap engram) |
| positions 0..204800 | 219-252 ms | REBASE_STATUS.md (mmap engram) |
| model load | ~1.5-3.5 min with a warm weight cache (owner's figure ~2.5 min unverified) | results `model_load_s` 82-205 s |

The committed `results/X.3.json` holds the pre-rebase numbers: wall 245.2 ms with mmap, and a 2890.7 ms full prefill.
The perf picks' A/B tables (off vs on, at 51200 and 512000) are in
`/scratch/$USER/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/ab/<task>/table.md`. Use them for the
512000 numbers. A rebased 512000 baseline is not restated here.

## 6. Working with the framework

**Read first:** `models/demos/common/bringup/README.md`, which covers the how-to, task ids, gates and exit codes. The
skills are in `models/demos/common/bringup/skill/{bringup,rebase-bringup,bringup-fork-op,bringup-fork-tests}/SKILL.md`;
link them into `~/.claude/skills/`. The agent definitions are in `agents/`; link them into `~/.claude/agents/`.

```bash
B="python -m models.demos.common.bringup"; O="python -m models.demos.common.bringup.orchestrator"
$B status --spec $S                         # the ledger
$O run|resume --spec $S [--until <task>]    # resume after a stop; --until runs a task and its deps, then stops
$O pause --spec $S                          # stop before the next task
$O decide --task P.34 --accept|--reject|--rerun --spec $S   # a perf pick, from its A/B table
$B gate <task> --spec $S [--unit U | --continue | --status | --no-commit]
$B rerun --from <task> --spec $S            # reset it and everything downstream, then sweep
```

- **Ledger.** `tasks.yaml` holds each task's brief, `paths` and `gate.cmd` + `metrics`. `state.json` and
  `state_metrics.json` hold the verdicts and metrics. `results/<task>.json` holds the gate's metrics. `approvals.yaml`
  holds the approvals; any edit to what was approved voids it.
- **The run's working files.** The orchestrator log, agents' transcripts, A/B tables and the device timing log are in
  `/scratch/$USER/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/<run>/`.
- **`gate` commits a PASS by default** (F239). Pass `--no-commit` to test without committing. A gate never restarts:
  each test segment is a unit, kept until the code under test changes.
- **Perf picks (P.\*).** An agent makes the change behind a `tt/settings.py` switch. The orchestrator measures it off
  and on (`ab.env` / `ab.change`) and then waits for the owner's `decide`. Exit codes: 0 done, 3 waiting for a person,
  1 stopped.
- **Settings.** `tt/settings.py` is the only place a switch lives. The environment variable is `DS41_<NAME>`, and each
  entry carries its owner decision and measurements. Read the entry before flipping one. The key switches:

| Switch | Default | Why it matters |
|---|---|---|
| `ENGRAM_TABLE` | resident (tests: mmap) | where the engram tables are read from: the host RAM build vs the checkpoint mmap. Bit-identical. Timed runs set resident |
| `EXPERTS_IMPL` | bfp4_mse | routed experts bfp4 with an MSE-chosen exponent (FIX.22). bfp4 / bfp8 / bfp8_down4 for comparison. Each has its own cache files |
| `MOE_IMPL` | block | block MoE with a fused head on the torus. flat hangs on the torus (DS41-130) |
| `OWN_ROWS` | on | P.23: residual / mHC streams on each chip's own rows |
| `POST_COLLAPSE` | on | P.34: mhc_post writes the next sublayer's collapsed input. Bit-identical to off |
| `WINDOW_KV_KEEP` | full | P.32: full / ring (192 rows a layer) / none: the window-KV memory a slot keeps |
| `DECODER_SKIP` | off | P.30 / P.31: `exact` / `replay128` skip layers 20-39 on non-last chunks |
| `MAX_WORKER_CORES` | 110 | Galaxy power rule: never more than 110 cores an op. Never 0 on a Galaxy |
| `EXPERTS_WEIGHT_CACHE`, `EXPERTS_PREFETCH` | on | the bank-major expert cache and its read-ahead. Pinned host memory is raised only during the expert upload |

**Open design item.** Prefill disaggregation, first cut (today): prefill runs all 40 layers on every chunk and migrates
every cache. Target: prefill stops at an even cut 128 tokens short of the end and skips the decoder. That needs
`DECODER_SKIP=exact` or `replay128`; P.33 checks it in `tests/bringup/test_decoder_cut.py` and it passes. The default
stays off until the owner decides.

## 7. Rules and pitfalls

**Device**
- Run on the device only through `scripts/run_safe_pytest.sh` (`--run-all --no-precompile` for model tests). The
  gates add `--no-precompile` themselves (F151). One device job at a time, and never next to a running orchestrator.
- Before any device job, check that nobody holds a chip. This prints nothing when the chips are free:
  `for f in /dev/shm/TT_UMD_LOCK.CHIP_IN_USE_*_PCIe; do python3 -c "import struct,sys,os; p=struct.unpack_from('<i',open(sys.argv[1],'rb').read(),52)[0]; p and os.path.exists(f'/proc/{p}') and print(sys.argv[1], 'held by pid', p)" "$f"; done`
  Also glance at other users' processes (`ps`).
- Never wrap a device command in `timeout`. The safe runner detects hangs, collects triage and resets the box itself.
- Galaxy reset: the safe runner's `recover-hosts` (`scripts/lib/tt_device_pool.sh`), or `tt-smi -glx_reset`. **Never
  `tt-smi -r`** on a Blackhole Galaxy. (Not exercised while writing this; taken from the bringup skill and the pool
  script.)
- If `/tmp/tt-device.lock` is a stale file of another user, point `TTPOOL_LEGACY_LOCK` / `TTPOOL_LEGACY_DIRTY` at
  files of your own. See `/scratch/dnijemcevic/node_env.sh` for an example.

**Known failures and waivers**, all in the galaxy spec's `tests:` section:
- `unit_waivers`: `test_adapter_acks` / `test_runner_contract` accept "layer N window_kv last block pcc 0.98x < 0.99".
  tt-d-gen's fixed 0.99 rule fails by rounding since P.23 / bfp4_mse; the gated rel-L2 checks pass.
- `fork_waivers`: sparse_sdpa row_bfp4 at H=32 does not build. SFPI 7.90-7.93 has an internal compiler error,
  tenstorrent/tt-metal#64605 (reproducer: `ttnn/ttnn/bringup/sdpa/docs/sfpi_ice/`). H=64 (the 8x4) builds. On a
  2x2 QuietBox, DS41 at TP2 has 32 heads a chip and hits this until there is a fixed SFPI.
- `waivers` (per step): attn_hc / ffn_hc, experts (bfp4), indexer, router. The reasons are written there.
- `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_hybrid_routed_expert_combine.py` hangs under the safe
  runner's precompile plugin, on the old and new builds alike. With `--no-precompile` it passes. It is a framework bug,
  not on DS41's path.
- The Galaxy cannot run 2x2 / 1x4 submesh fork cases: fabric init times out, or they are skipped as "exact match of
  requested num devices". Those cases go to the QuietBox. The list is in `bringup/REBASE_STATUS.md` ("For the
  QuietBox").

**Host**
- A test at its peak plus the agents can exceed a low `ulimit -u` (512 on an earlier node).
- tt-kmd #301 leaks host memory, and only a reboot clears it. Watch `free -g` before resident-engram runs.

**Before changing anything, read** `bringup/known_issues.md` (DS41-n entries; shared ones in
`models/demos/common/bringup/knowledge/known_issues.md`), `bringup/repo_map.md` and the latest sections of
`bringup/BREADCRUMBS.md` (append-only, one section per task attempt).

## 8. Where things are

| Path (relative to `models/demos/deepseek_v41_flash_d_p/` unless absolute) | What |
|---|---|
| `tt/model.py`, `tt/*.py` | the all-device model (`TtDS41Model`) and its steps |
| `tt/settings.py` | every switch, with its default and its why |
| `tt/runners/adapter.py`, `kv_contract.py` | the tt-d-gen prefill adapter and the served KV layout |
| `reference/` | CPU reference: DeepSeek's own inference code with CPU kernels (`hf_oracle.py`, `kernels.py`, `weights.py`) |
| `tests/bringup/contract/` | serving-contract tests (K.1R) |
| `tests/bringup/test_c_*`, `test_swap_*` | frozen per-step component and swap tests (for diagnosis) |
| `tests/bringup/test_decoder_cut.py`, `test_decoder_skip.py`, `test_window_ring.py`, `test_logit_kl.py` | decoder cut / skip, window ring, token-level accuracy |
| `tests/perf/` | matmul sweep, decoder-skip timing, expert-cache warm-up |
| `tools/convert_expert_cache.py` | .tensorbin -> bank-major expert cache |
| `bringup/ledgers/galaxy_8x4-full/` | the 8x4 run: spec, tasks, state, results, supervision.md (the overseer's log; latest at the end), HANDOFF_*.md |
| `bringup/spec.yaml`, `bringup/tasks.yaml` | the 2x2 seed run |
| `bringup/plan.md`, `serving_contract.md`, `ENGRAM_HOST_PATH.md`, `tt/MOE_BLOCK.md` | sharding plan, how tt-d-gen drives prefill, the engram host path, the block MoE |
| `bringup/REBASE_STATUS.md` | the 2026-11-06 rebase: verification, decisions, QuietBox list |
| `bringup/known_issues.md`, `repo_map.md`, `BREADCRUMBS.md`, `opportunities.md` | knowledge files |
| `models/demos/common/bringup/` | the framework: `README.md`, `defaults.yaml` (framework switches), `orchestrator.py`, `tests/` (ladder, profile, positions, contract, smoke, batch), `testing/`, `docs/` |
| `ttnn/ttnn/bringup/` (`INDEX.md`) | the forked ops (`ttnn.bringup.*`) |
| `scripts/run_safe_pytest.sh` | the only way to run on the device |
| `/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/{golden_full/galaxy_8x4,tt_cache}` | shared masters: goldens and weight cache |
| `/scratch/$USER/bringup/deepseek_v41_flash_d_p/{hf_full,golden_full,tt_cache,runs,profiles}` | node-local: checkpoint, mirrors, run dirs |
| `/scratch/$USER/tt-d-gen`, `/scratch/$USER/tt-d-gen-build/venv312` | tt-d-gen checkout and its engine venv |
