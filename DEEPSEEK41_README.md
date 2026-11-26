# DeepSeek-V4.1-Flash prefill on Tenstorrent: start here

For a Claude Code session (and its human) picking up this model. Give this file to your Claude first. It covers what
the bring-up is, its current state and defaults, how to set up, run and measure it, the prefill → decode (PD) serving
hand-off, the open items, and where everything is. It states the current facts; the history is in the ledger
(`supervision.md`, `results/`, `BREADCRUMBS.md`).

Facts are as of 2026-11-26, branch `dnijemcevic/model_bringup_batching` at `e3f48ea0390`, through task PB.MM256 (accepted 2026-11-23, every gate passed). Items marked
**(unverified)** were not checked against code or a run while this was written.

**This is the batching branch.** Sections 1-11 are the `dnijemcevic/model_bringup` text unchanged; section 12 adds packed
prefill and lists every shared fact that is not true on this branch (12.6). Read 12.6 before trusting a number above.

## 1. What this is

Chunked prefill of `deepseek-ai/DeepSeek-V4.1-Flash` (text decoder only: no vision, no MTP), all 40 layers, on a
Blackhole Galaxy 8x4: 32 x p150b chips, fabric `FABRIC_2D_TORUS_XY`, SP = 8 mesh rows (the sequence is split over
them), TP = 4 mesh columns (the hidden size is split over them). "Prefill" = computing the prompt's caches; a separate
decode machine (tt-blaze) generates the tokens after a hand-off.

- The gated bring-up framework (`models/demos/common/bringup`) organizes the work as tasks with gates (tests that must
  pass) in a ledger (section 9).
- The model code is `models/demos/deepseek_v41_flash_d_p/tt/`. Every switch is in `tt/settings.py`.
- The tt-d-gen inference server drives the model through the adapter in `tt/runners/`.

**Branches.** `dnijemcevic/model_bringup` is the single-request prefill (one request a chunk, 5,120-token chunk = 160
rows a chip). `dnijemcevic/model_bringup_batching` forks from it at `bcab6af973b` (2026-11-19 hand-off) and adds packed
prefill (several requests in one 8,192-token chunk); fixes found on either branch are ported to the other as their own
tasks (MB.* on model_bringup, PB.* on the batching branch). Both share one ledger directory layout; each branch's `tasks.yaml` holds its own tasks.

| Item | State |
|---|---|
| Deployment box / spec | Galaxy 8x4: `models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml` (deploy seq 1,044,480, chunk 5,120) |
| Seed run | QuietBox 2x2, partial (layers 0-3, 14, 20, 21, 24, scaled model): `bringup/spec.yaml`, ledger in `bringup/` |
| Accuracy now (MB.9 accept gate, rung `last`, 40 layers) | err_final_hidden 0.1362 (limit 0.2422), top-5 1.0; determinism max abs diff 0.0 (fresh process) |
| Perf now (MB.9 accept gate, 5,120 chunk, layers 0-20) | 70.6k new tok/s at 51,200 history, 68.5k at 139,264, 62.1k at 1,024,000 (section 3) |
| Serving rebuilds | 0 program builds after warm-up, 0 ids uploads a chunk (`tests/test_no_rebuild.py`, cold program cache) |
| Running now | Nothing; the queue is empty after MB.9, waiting on the owner |
| Open in the ledger | FIX.11, FIX.13 (TODO, not queued); section 10 |

## 2. Current defaults

`tt/settings.py` is the source of truth: each entry carries its owner decision and measurements; read it before
flipping one. Environment variable = `DS41_<NAME>`. Call-time switches (read per chunk, listed in
`bringup/hooks.py batch_call_settings`) let both A/B sides run in one model load; load-time ones need a reload.

| Switch | Default | Other values | What it does |
|---|---|---|---|
| `MOE_BLOCK_SRC` | `block` | `main`, `carried` | Marko's MoE block as merged in his stack, on the fork `ttnn.bringup.moe_block` (`tt/moe_block_v2_adapter.py`) |
| `MOE_IMPL` / `MOE_HEAD` / `MOE_TAIL` | `block` / `fused` / `fused` | `flat` / `ops` | Block MoE with fused head and tail. `flat` hangs on the torus (DS41-130) |
| `EXPERTS_IMPL` | `bfp4_mse` | `bfp4`, `bfp8`, `bfp8_down4` | Routed experts bfp4 with an MSE-chosen exponent |
| `ATTN_SP32_PROJ`, `ATTN_SP32`, `OWN_ROWS` | `on` | `off` | Attention and the residual / mHC streams on each chip's own S/(SP TP) rows |
| `HISTORY_FORMAT` | `bfp4` | `bf16` | Prefill's own attention history: served bf16 rows packed to 4-bit records every chunk before the gather |
| `CKV_FORMAT` | `bf16` | `bfp4_rows` | The SERVED compressed KV stays bf16 (decode reads it) |
| `WINDOW_KV_FORMAT` | `bf16_tiles` | `bfp8_tiles`, `bfp8_blocks`, `bf16` | Window KV computed and read in bf16 |
| `INDEX_K_FORMAT` / `INDEX_K_COPY` / `INDEX_K_SERVED` | `bfp8` / `bfp8` / `fp8_row192` | | Indexer scores bfp8 keys; served index keys as Blaze's 192-byte fp8 rows |
| `INDEXER_BLOCK_SELECT`, `INDEXER_BMAX_COMPUTE` | `on` | `off` | Exact block-max pre-selection before the indexer top-k, block maxima in the score's compute kernel |
| `INDEXER_BLOCK_SELECT_MIN` | `auto` (262,144 keys) | a key count | Below it the plain top-k (the block path's extra launches cost more wall than they save) |
| `MATMUL_CONFIG` / `MM_TUNED160` | `galaxy` / `on` | `tuned`, `auto` / `off` | Dense matmul program configs: the 8x4 sweep (`tt/mm_configs.GALAXY`), plus the 5,120-chunk re-sweep (`GALAXY160`, `GMM160`); call-time |
| `POS_DYNAMIC` | `on` | `off` | Every per-chunk value is a runtime value: serving builds no program after warm-up; ids reordered on the device |
| `HOST_HIDE`, `ENGRAM_HOST_IMPL` | `True`, `cpp` | `False`, `numba` | Host work of a chunk hidden behind the device; engram hash / gather in C++ (`tt/csrc/engram_host.cpp`) |
| `MHC_POST_FAST`, `NORM_IN_MHC`, `FFN_FEED`, `POST_COLLAPSE` | on | off | mHC / norm / FFN-input fusions |
| `ENGRAM_TABLE` | `resident` (tests: `mmap`) | `mmap` | Engram tables in host RAM vs the checkpoint mmap. Timed runs set resident |
| `ENGRAM_DEVICE`, `ENGRAM_DEVICE_TABLES` | `off` | `on` | Engram tables on the Galaxy (6.6 GB a chip, ~6 min upload); not shipped |
| `STATIC_TABLES` | `off` | `on` | No per-user, per-chip KV history: each chunk gathers the slot's served rows |
| `DECODER_SKIP` | `off` | `exact`, `replay128`, `decode` | `decode` = prefill runs layers 0-19 + layer 20's KV head only (decode's proposed cut). Every perf measurement runs `decode`; the shipped default stays `off` until the owner decides |
| `PREFILL_END` | `cap` | `end` | The serving runtime stops computing at decode's cap (64-token block) |
| `WINDOW_KV_KEEP` / `REINDEX_SERVE` / `KV_BLOCK_SIZE` | `snapshot128` / `alias` / 64 | | 128-row window snapshot per layer; re-index layers served as aliases of layer 20 |
| `CCL_IMPL` | `fabric_bf16` | `fabric`, `stock` | Gathers on `high_bw_all_gather`, reduce-scatters on the fabric fork, TP partial sums in bf16 |
| `MAX_WORKER_CORES` / `NONMM_MAX_CORES` | 110 / 0 (full 120 grid) | | Galaxy power rule: matmuls and the forks stay within 110 cores; other stock ops may use 120. Never set `MAX_WORKER_CORES=0` on a Galaxy |

## 3. Perf and accuracy headline

**Perf: the 5,120 ladder** (MB.9 accept gate, 2026-11-23, `results/MB.9.json`; method and off/on comparison in
`results/MB.9_ladder.md`). agentic_cc text, layers 0-20 loaded (`DECODER_SKIP=decode`), resident engram table, steady
chunk = mid-request, resumed = first chunk after a prefix hit (128 rows of encoder replay). Wall = clean warm median of 10
timed runs, each preceded by the real previous chunk, program-cache misses refused; device = one profiled run, slowest
chip; exposed host = wall - device. Walls vary by ±1-2 ms run to run: judge changes by device ms.

| History | 51,200 | 131,072 | 139,264 | 262,144 | 524,288 | 786,432 | 1,024,000 |
|---|---|---|---|---|---|---|---|
| steady wall ms | 72.5 | 75.6 | 74.7 | 75.3 | 77.8 | 83.1 | 82.4 |
| steady new tok/s | 70.6k | 67.7k | 68.5k | 68.0k | 65.8k | 61.6k | 62.1k |
| resumed wall ms | 74.9 | 77.8 | 78.7 | 79.5 | 82.0 | 84.6 | 83.4 |
| steady device ms | 60.0 | 62.0 | 62.7 | 63.6 | 67.1 | 73.8 | 75.3 |

- Serving adds the ids readback, ~1.5 ms a chunk (about -1.3k tok/s; `results/agentx_ladder_20261119.md`).
- Device ms by section for a 5,120 chunk at 138,240 history: `results/MB.9_ladder.md` section 3 (chunk 66.06 ms;
  experts 31.5, attention 21.5, ffn_residual 4.7, attn_residual 3.1, indexer 2.1).
- The MoE block's fused matmuls (~25 ms a chunk) take their blocking at load: not tunable by a call-time switch.

**Accuracy** (MB.9 accept gate, rung `last`, all 40 layers, every gated limit holds):

| Metric | Value | Limit |
|---|---|---|
| err_final_hidden | 0.1362 | 0.2422 |
| top5_overlap | 1.000 | >= 0.9943 |
| err_layer_L39 | 0.7554 | 1.5332 (L39 swings 0.70-0.91 on rounding-only changes, DS41-188) |
| determinism_max_abs_diff (fresh process) | 0.0 | == 0 |

## 4. Setup on a new Galaxy

Everything under `/scratch` is node-local. A new box has none of it. The shared masters are on NFS `/data` and are
readable by other users.

```bash
# 1. Checkout (origin = github.com/tenstorrent/tt-metal); not re-run for this doc (unverified as a sequence)
git clone https://github.com/tenstorrent/tt-metal.git && cd tt-metal
git checkout dnijemcevic/model_bringup
git submodule update --init --recursive
# tt_ops_code_gen has update=none in .gitmodules. Without it the fork tests that import `eval` fail (the root symlink eval -> it)
git -c submodule.tt_metal/third_party/tt_ops_code_gen.update=checkout submodule update --init tt_metal/third_party/tt_ops_code_gen

# 2. Build: Release, no tests (the build_metal.sh defaults). Afterwards `build` must be a symlink to build_Release.
./build_metal.sh
# After a C++ change to a bring-up fork: cmake --build build --target install (Python loads the installed copies)

# 3. venv (./python_env)
./create_venv.sh

# 4. Every session, from the repo root
export PYTHONPATH=$PWD && source python_env/bin/activate
S=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml
```

The data paths come from the spec. `paths.art` is `/scratch/$USER/bringup`, and the framework appends the model
name (`core/spec.py:art`):

| What | Where | How it gets there |
|---|---|---|
| HF checkpoint, rev `5254448103b6254fc646dacbca1cbbf20e9216c1`, all 48 shards (476 GiB) | `/scratch/$USER/bringup/deepseek_v41_flash_d_p/hf_full` | Download once (below). Dense weights and the engram tables are read from it |
| Goldens (master, 85 GB) | `/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4` | Fixed |
| Goldens (node-local mirror) | `/scratch/$USER/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4` | `reference.golden sync`; the orchestrator syncs at run/resume start |
| Routed-expert weight cache (master) | `/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/tt_cache/8x4/v1` | Fixed. The default `block` reads the main block's `.moe_main.*` files where its plan lays them out the same, else `.moe_blk.*` (sizes now: unverified; 612 GB for carried + main as of 2026-11-08) |
| Weight cache (node-local mirror) | `/scratch/$USER/bringup/deepseek_v41_flash_d_p/tt_cache/8x4/v1` | `core.weight_cache pull`; the orchestrator pulls at run/resume start. A miss rebuilds that layer from the checkpoint |

```bash
python -c "from huggingface_hub import snapshot_download as d; d('deepseek-ai/DeepSeek-V4.1-Flash', \
  revision='5254448103b6254fc646dacbca1cbbf20e9216c1', \
  local_dir='/scratch/$USER/bringup/deepseek_v41_flash_d_p/hf_full', max_workers=8)"
python -m models.demos.common.bringup.reference.golden status|sync --spec $S
python -m models.demos.common.bringup.core.weight_cache status|pull --spec $S
```

- Local disk: about 1.2 TB (checkpoint, weight cache, goldens).
- `ENGRAM_TABLE=resident` needs ~190-210 GB of host RAM for the two engram tables; a full 40-layer load holds ~200 GB
  host RSS.
- `push` is for the master's owner only. A user who cannot write the master only pulls.

## 5. Run the model

The rules in section 10 apply. **Do all device work through the resident runner** (one model load held in one
process; a later request reaches its first chunk in ~1-2 s). A direct `scripts/run_safe_pytest.sh` makes the runner step
aside, which costs a fresh device open (2-4 min load, plus host RAM the driver leaks until a reboot).

```bash
R="python -m models.demos.common.bringup.resident"
$R start --spec $S          # when the first device work is ready
$R submit --spec $S --cmd "<safe-runner command(s), && joined>" [--task T] [--fresh]
$R status|stop --spec $S
```

New test files mark `BATCH_SHARED = True` (+ `BATCH_ITEM_VARS` for per-run env, `BATCH_LAYERS`) to run in the held
process. The commands below are what the accept gate (`tasks.yaml`, MB.9 `gate.cmd`) runs; prefix `BRINGUP_SPEC=$S`
outside the orchestrator.

```bash
P="BRINGUP_CHUNK=5120 DS41_ENGRAM_TABLE=resident BRINGUP_POS_TOKENS=agentic_cc DS41_DECODER_SKIP=decode"
SR="scripts/run_safe_pytest.sh --run-all --no-precompile"

# accuracy: ladder rung `last` (golden 0-51200 prefix, last chunk on device, all 40 layers, every metric vs its limit)
BRINGUP_RUNG=last $SR models/demos/common/bringup/tests/test_ladder.py
# the 5,120 perf ladder (steady + resumed walls, device ms, exposed host per rung)
$P SKIP_POS=51200,131072,139264,262144,524288,786432,1024000 $SR models/demos/deepseek_v41_flash_d_p/tests/test_skip_positions.py
# no program build after warm-up in serving (PASS/FAIL, run cold: first in a fresh process)
$P $SR models/demos/deepseek_v41_flash_d_p/tests/test_no_rebuild.py
# MB.9's own unit: tuned matmul configs off vs on, one load
$P DS41_MM_TUNED160=on $SR models/demos/deepseek_v41_flash_d_p/tests/test_mm160_offon.py

# "What is the capital of France?" -> Paris, on the device model (no server)
$SR models/demos/common/bringup/tests/test_smoke.py
# the full 56k prefill, all 11 chunks on device
DS41_ENGRAM_TABLE=resident BRINGUP_RUNG=s56320 $SR models/demos/common/bringup/tests/test_ladder.py
```

- **Determinism** (bit-identical repeats) outranks everything. The framework adds it to every model-level gate
  (`core/gate.py with_determinism`, `testing/model_determinism.py`) and runs it in a fresh process, once per accept gate;
  `tests/test_determinism.py` adds the `DECODER_SKIP=decode` sequence the serving runtime takes.
- **Walls.** `test_skip_positions.py` measures its own clean walls (above). For the framework's single-position
  profile (`test_profile.py`) only `BRINGUP_PROFILE_WALL=1` runs give clean walls (the gate splits profile and wall).
- **Box check:** `models/demos/common/bringup/tests/test_box.py` once after a reboot.
- Matmul config sweeps: `tests/perf/test_matmul_sweep.py` (capture, then `-k test_sweep_card` in parallel single-card
  processes, `P23_SHARD=i/N`; resident runner stopped first). Recipe in `results/MB.9_ladder.md` section 2.

**Serving runner (tt-d-gen).** The serving-contract tests drive the model with the real tt-d-gen engine
(`serving.server_repo` in the spec: `/scratch/$USER/tt-d-gen`; tested against `jazpur/dsv4-replay-window` at
`48b79127fb1`). Build: `./build_dgen.sh --bindings --blaze --no-kvm-client`, or the `dgen-build` agent
(`models/demos/common/bringup/agents/dgen-build.md`). Check: `python -m models.demos.common.bringup.testing.dgen_engine find`.

```bash
T=models/demos/deepseek_v41_flash_d_p/tests/bringup/contract
$SR models/demos/common/bringup/tests/test_contract.py
for t in test_kv_layout test_adapter_acks test_runner_contract test_runner_smoke; do $SR $T/$t.py || break; done
```

## 6. Accuracy

There is no fixed bar. Every gated metric has a limit of `e2e.calib` (2.0) x the official reference's own drift
from full precision. The golden is the fp32 reference; the drift comes from the bf16/fp8/fp4 reference
(`<rung>-official`, `drift-<rung>.json` beside the golden). Code: `testing/e2e_limits.py`, `reference/drift.py`.

| Metric | Meaning |
|---|---|
| `err_layer_L00..L39` | per-layer output rel L2 (`pcc_layer_*` is diagnostic only) |
| `err_state_<cache>_L*`, `overlap_state_*` | the served caches: window_kv, compress_kv, index_k, topk_idx (overlap), candidates, engram_tail |
| `err_final_hidden` | final norm output rel L2 |
| `top5_overlap`, `top1_match`, `pcc_logits_tail` | token-level agreement of the last rows' logits |
| `determinism_*` | bit-identical repeats (== 0) |

- Rungs: `s10240`, `last` (golden 51k prefix + 1 chunk, the accept check), `s56320` (the full 56k).
- Owner rule: bit-identity to the previous code is not a criterion; a change may move rounding (e.g. a matmul's fp32
  summation order, DS41-390) as long as every limit holds and repeats are exact.
- The precision policy is realistic precision only: there is never a max-precision device run
  (`models/demos/common/bringup/README.md`, "Switches"; the spec's `agents.rules`).

## 7. The PD serving hand-off (prefill → decode)

Prefill (this model, tt-metal) computes the prompt's caches. tt-d-gen's KV Manager copies ("migrates") them to the
decode machine (tt-blaze). Both sides publish a KV address table; tt-d-gen moves the records between them.

| Document | What |
|---|---|
| `models/demos/deepseek_v41_flash_d_p/bringup/serving_contract.md` | How tt-d-gen drives prefill: input, chunk header, served caches, acks; the `PREFILL_END=cap` and `DECODER_SKIP=decode` contract changes |
| `models/demos/deepseek_v41_flash_d_p/bringup/migration_configs.md` | The served config table (v3, generated from the code): per config the layers, record bytes, entry counts, cap rule, open points |

What is served (config names as tt-d-gen uses them): `0_window_kv` (layers 0-39, a 128-row snapshot at cap),
`1_compress_kv_r2` / `2_index_k_r2` (layers 2, 8, 14), `3_compress_kv_r1` / `4_index_k_r1` (layer 20; layers 24, 28, 32,
36 as aliases of 20's index rows). Compressed KV is served bf16; index keys as fp8 192-byte rows.

- `DECODER_SKIP=decode` (model side implemented and gated; tt-d-gen not yet changed): prefill stops at
  P = (N - 128) // 64 x 64 and never runs layers 21-39; decode runs [P, N) itself. The chunk header gains
  `global_write_start` (encoder replay below it). Under it layers 20-39's snapshot records are not written.
- Serving asks still open on the tt-d-gen side: send the engram token-id tail and the 128-row bounded window replay
  (`bringup/ledgers/galaxy_8x4-full/HANDOFF_20261119.md` section 5).

## 8. Known issues

Model lessons: `models/demos/deepseek_v41_flash_d_p/bringup/known_issues.md` (DS41-n entries; long forms in
`known_issues_detail.md`); shared ones: `models/demos/common/bringup/knowledge/known_issues.md`. Read them before
changing anything. The ones that bite most often:

- Every device open leaks 10-27 GiB of host RAM in the driver until a reboot; check `free -g` before full-model loads.
- New kernels hung the box: bounded waits, every awaited semaphore signalled, NOC writes inside buffers, fabric
  payloads <= 6,208 B, smallest shape first.
- Random test inputs hide a summation-order change in bfp8 matmuls (DS41-390, `results/MB.9_ladder.md`): confirm
  matmul config changes on a real chunk.
- The sparse_sdpa row_bfp4 H=32 cases do not build (SFPI internal compiler error; owner-waived in the spec's
  `fork_waivers`; H=64 on the 8x4 builds).

## 9. Operating the bring-up

**Read first:** `models/demos/common/bringup/README.md` (how-to, task ids, gates, exit codes, resident runner).
Skills: `models/demos/common/bringup/skill/{bringup,rebase-bringup,bringup-fork-op,bringup-fork-tests}/SKILL.md`;
link them into `~/.claude/skills/`. Agent definitions: `models/demos/common/bringup/agents/`; link them into
`~/.claude/agents/`.

**The ledger** (`models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/`):

| File | What |
|---|---|
| `tasks.yaml` | each task's brief, allowed `paths`, `gate.cmd` + `metrics`, A/B env. Escape `'` as `''`, validate after every edit, keep it under 500 KB |
| `approvals.yaml` | the owner's approvals; any edit to what was approved voids it |
| `state.chunks/`, `state_metrics.chunks/` | verdicts and metrics |
| `results/<task>.json`, `results/<task>_*.md` | the gate's metrics; ladder / A/B reports (latest: `MB.9_ladder.md`, `MB.8_ladder.md`, `agentx_ladder_20261119.md`) |
| `supervision.md` | the overseer's log, newest at the end: read its tail first |
| `HANDOFF_20261119.md` | the working rules (fast iteration, owner rules) and packing facts |
| `spec.yaml`, `overseer_tools/` | box, paths, limits, waivers, serving; the overseer's launcher scripts (node-specific) |

**The orchestrator** (runs agents, measures A/B, runs gates, commits):

```bash
B="python -m models.demos.common.bringup"; O="python -m models.demos.common.bringup.orchestrator"
$B status --spec $S                         # the ledger
$O run|resume --spec $S [--until <task>]    # --until runs a task and its deps, then stops
$O pause --spec $S                          # stop before the next task
$O decide --task <T> --accept|--reject|--rerun --spec $S   # a perf pick, from its A/B table
$O measure --task <T> --part <run> --spec $S               # one measurement (e.g. determinism, last)
$B gate <task> --spec $S [--check | --unit U | --continue | --status | --no-commit | --commit-only]
```

- **A/B accept flow (perf picks).** An agent makes the change behind a call-time switch; the orchestrator measures off
  and on in one load and stops with exit code 3; the decision follows the owner's rule (a device win with every gate
  unit passing is accepted); the accept gate (switch default flipped) runs accuracy on all 40 layers plus fresh-process
  determinism, the ladder and the cold no-rebuild check, then commits. Exit codes: 0 done, 3 waiting, 1 stopped.
- **Logs:** `/scratch/$USER/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/` (`orchestrator.log`,
  `logs/<task>.log`, `agents/*.jsonl` (large: extract the last message with a script), `ab/<task>/table.md`).
- **Never edit `tasks.yaml` or `approvals.yaml` while a task runs**, and never switch branches while any orchestrator
  process is alive (check `ps`, not only the log).
- Run measurements yourself on the resident runner; agents only for code changes (they submit to the runner too).

## 10. Rules and pitfalls

**Device**
- Before every device run check that nobody holds a chip: `fuser /dev/shm/TT_UMD_LOCK*` prints nothing when free.
- Run on the device only through `scripts/run_safe_pytest.sh` (`--run-all --no-precompile`) or the resident runner.
  One device job at a time, never next to a running orchestrator.
- Never wrap a device command in `timeout`: the safe runner detects hangs, collects triage and resets the box.
- Reset only with the safe runner's `recover-hosts` (`scripts/lib/tt_device_pool.sh`) or `tt-smi -glx_reset`. **Never
  `tt-smi -r`** on a Blackhole Galaxy.
- Never loosen a limit to make a test pass. No fork regression suites on the 8x4: run the tests you add and the
  model-level checks.
- If `/tmp/tt-device.lock` is a stale file of another user, point `TTPOOL_LEGACY_LOCK` / `TTPOOL_LEGACY_DIRTY` at files of
  your own.

**Owner rules (binding)**
- Fast iteration first: never load what you do not need (CPU, one layer, layers 0-20, the full 40 only in the accept
  gate); write the one-line reason before any full load.
- Do not break tt-d-gen compatibility (served caches, formats, addresses, acks, chunk header, the engine's device
  input). The model does not know the next chunk: no per-prompt precompute, no extra host round trips.
- Serving never rebuilds: every position-, slot- or layout-derived value is a runtime value.

**Known failures and waivers** (the galaxy spec's `tests:` section): `unit_waivers` (tt-d-gen's fixed 0.99 window-KV
rule fails by rounding on a few layers; the gated rel-L2 checks pass), `fork_waivers` (SFPI ICE cases), per-step
`waivers` with their reasons. The Galaxy cannot run 2x2 / 1x4 submesh fork cases; those go to a QuietBox
(`bringup/REBASE_STATUS.md`).

**Open items:** the queue is empty after MB.9 (owner's next direction pending). In the ledger: FIX.11 (layer 0 attention
precision vs the reference), FIX.13 (DS41-141, window-KV rise) are TODO. `DECODER_SKIP` stays `off` until the owner
decides; the tt-d-gen asks in section 7 are pending.

## 11. Where things are

Repo paths relative to `models/demos/deepseek_v41_flash_d_p/` unless absolute.

| Path | What |
|---|---|
| `DEEPSEEK41_README.md` (repo root) | this file |
| `tt/model.py`, `tt/*.py` | the all-device model (`TtDS41Model`) and its steps |
| `tt/settings.py` | every switch, with its default and its why |
| `tt/mm_configs.py` | matmul program configs (`GALAXY`, `GALAXY160`, `GMM160`) |
| `tt/runners/adapter.py`, `tt/runners/kv_contract.py` | the tt-d-gen prefill adapter and the served KV table |
| `tt/moe_block_v2_adapter.py`, `models/demos/moe_block/` | the default MoE block (`MOE_BLOCK_SRC=block`) |
| `reference/` | CPU reference (`ds41_ref.py`, `hf_oracle.py`, `kernels.py`, `weights.py`) |
| `tests/test_skip_positions.py`, `tests/test_no_rebuild.py`, `tests/test_determinism.py`, `tests/fresh_common.py` | perf ladder, no-rebuild check, determinism, fresh-traffic helpers |
| `tests/bringup/contract/` | serving-contract tests |
| `tests/bringup/test_c_*`, `test_swap_*` | frozen per-step component and swap tests |
| `tests/perf/` | probes and sweeps (`test_matmul_sweep.py`) |
| `bringup/ledgers/galaxy_8x4-full/` | the 8x4 run: spec, tasks, approvals, state, results, supervision.md, hand-off |
| `bringup/known_issues.md`, `repo_map.md`, `BREADCRUMBS.md`, `opportunities.md` | knowledge files (BREADCRUMBS: append-only, one section per task attempt) |
| `bringup/serving_contract.md`, `bringup/migration_configs.md` | the PD contract and served table |
| `bringup/spec.yaml`, `bringup/tasks.yaml`, `bringup/plan.md` | 2x2 seed run and sharding plan |
| `models/demos/common/bringup/` | the framework: `README.md`, `resident.py`, `orchestrator.py`, `tests/`, `testing/`, `knowledge/perf_guide.md` |
| `ttnn/ttnn/bringup/` (`INDEX.md`) | the forked ops (`ttnn.bringup.*`) |
| `scripts/run_safe_pytest.sh` | the only direct way to run on the device |
| `/scratch/$USER/bringup/deepseek_v41_flash_d_p/{hf_full,golden_full,tt_cache,runs}` | node-local checkpoint, mirrors, run dirs |
| `/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/{golden_full/galaxy_8x4,tt_cache}` | shared masters: goldens and weight cache |

## 12. Packed prefill (this branch)

`dnijemcevic/model_bringup_batching` = `dnijemcevic/model_bringup` forked at `bcab6af973b` (2026-11-19) + packed prefill
(continuous batching): one 8,192-token chunk (256 rows a chip) carries several requests ("segments"), each continuing
its own slot's served history. Design: `models/demos/deepseek_v41_flash_d_p/bringup/PACKED_DESIGN.md`; binding owner
rules for every PB task: `bringup/PACKED_RULES.md`; the tt-d-gen ask: `bringup/drafts/tt_d_gen_packed_ask.md`.

### 12.1 What the packed path is

- **Chip-aligned lanes** (the one packed path): each segment owns a contiguous run of whole chips (starts, lengths and
  splits multiples of 256 rows; last chip padded; 1-32 chips a segment; up to 6 segments, `tt/packed_glued.GLUED_SEGS`).
  Rows of two requests never share a chip. Typical layouts: 4 x 2,048 (8 chips each) and 4 x 1,792 + a 5th request's
  first 1,024 tokens (every row a real token).
- Token-wise work runs once over all 8,192 rows; attention, compressor, indexer and sparse_sdpa run ONCE a layer with
  per-segment arguments (no row moves). Prefix-hit (replay) segments run in the same call; an unsupported layout raises,
  it never falls back to a slower path.
- Each segment's history and index keys are gathered only to the chips it owns (`tt/owned_gather.py`, `tt/owned_keys.py`).
- No host tables and no program builds per layout: the host uploads token ids + the chunk descriptor (per segment slot,
  start, length, write_start); every per-chunk table is derived on the device (`tt/glued_derive.py`, `tt/lane_rows.py`).
- Serving entry: `tt/runners/adapter.py prefill_packed`. Code: `tt/packed.py` (descriptor, `GluedChunk`),
  `tt/packed_glued.py`, `tt/packed_device.py`; CPU reference `reference/packed_ref.py`.
- Single-request chunks (5,120 or 8,192) still run the model_bringup path.

### 12.2 Switches (all in `tt/settings.py`, call-time)

| Switch | Default | What it does |
|---|---|---|
| `PACKED`, `PACKED_MOE`, `PACKED_GATHER_HIDE`, `PACKED_ATTN_ONECALL`, `PACKED_HOST_FAST`, `PACKED_LANE_BALANCE`, `PACKED_LONG_HIDE`, `PACKED_GLUED`, `PACKED_GLUED_ONECALL`, `PACKED_GLUED_DYNAMIC` | `off` | Select the packed path. Every packed gate and ladder sets them all `on` together with `POS_DYNAMIC=on` (the env `$PK` below) |
| `PACKED_CHIPS` | `on` | Chip-aligned lanes (`off` = back-to-back glued rows, the A/B's old side) |
| `PACKED_HIST_OWNED`, `PACKED_KEYS_OWNED` | `on` | Histories / index keys only to the segment's own chips |
| `PACKED_CWRITE_FAST`, `COMMON_VALUE_SLOTS`, `PACKED_WARM_CONT1` | `on` | Packed served writes without slices; common runtime-arg slots; engine warm-up also covers a continuing, non-aligned one-segment chunk |
| `MM_TUNED256` | `on` | The 8,192-chunk matmul re-sweep (`tt/mm_configs.GALAXY256` / `GMM256`) |
| `MHC_POST_V2` | `True` | mhc_post v2 (bit-identical device win) |
| `INDEXER_BLOCK_SELECT_MIN_PACKED` | `auto` (196,608 keys) | Block-select threshold for packed chunks |

### 12.3 Tests and ladders (from the PB.MM256 and PB.T gates)

```bash
PK="DS41_PACKED=on DS41_PACKED_MOE=on DS41_PACKED_GATHER_HIDE=on DS41_PACKED_ATTN_ONECALL=on DS41_PACKED_HOST_FAST=on \
DS41_PACKED_LANE_BALANCE=on DS41_PACKED_LONG_HIDE=on DS41_PACKED_GLUED=on DS41_PACKED_GLUED_ONECALL=on \
DS41_PACKED_GLUED_DYNAMIC=on DS41_POS_DYNAMIC=on DS41_ENGRAM_TABLE=resident BRINGUP_POS_TOKENS=agentic_cc DS41_DECODER_SKIP=decode"
L=51200,131072,139264,262144,524288,786432,1024000
M=models/demos/deepseek_v41_flash_d_p/tests

# accuracy at 8192 (40 layers, vs the reference limits; from 0 and last)
BRINGUP_CHUNK=8192 $SR $M/packed/test_chunk8192_reference.py
# single 8192 ladder; packed ladders (4x2k, 4x1792s)
BRINGUP_CHUNK=8192 $PK SKIP_POS=$L $SR $M/test_skip_positions.py
BRINGUP_CHUNK=8192 $PK LADDER_POS=$L LADDER_CASES=4x2k,4x1792s $SR $M/packed/test_packed_walls.py
# cold strict no-rebuild on every layout (single8192, one4k, one1600, lanes4x2k, lanes4x1600s, lanes6mix, lanes2x3k)
NO_REBUILD_PACKED_STRICT=1 BRINGUP_CHUNK=8192 $PK $SR $M/test_no_rebuild.py
# adversarial batching tests: no cross-request leakage, bit-exact vs prompt-by-prompt (FAST tier in accept gates; DEEP sweep)
BRINGUP_CHUNK=8192 $PK $SR $M/packed/test_packed_fast_tier.py
BRINGUP_CHUNK=8192 $PK $SR $M/packed/test_packed_deep_sweep.py
# PB.MM256's own unit: tuned 8192 matmul configs off vs on, single and 4x2k, one load
BRINGUP_CHUNK=8192 $PK DS41_MM_TUNED256=on $SR $M/test_mm256_offon.py
```

Ladder method (`tests/packed/ladder_common.py`): every segment of a packed chunk sits at the same history; steady and
resumed (128 replay rows a segment); wall = median of 10 timed runs after the real previous chunk, program-cache misses
refused; tok/s = new tokens / wall.

### 12.4 Latest results (ledger `results/`)

`results/PB.MM256_ladder.md` (2026-11-23, the tip; switch on) and `results/PB.LAD_ladder.md` (2026-11-22, the full
table with device and exposed host per rung). Steady new tok/s, layers 0-20, agentic_cc:

| Case | 51,200 | 139,264 | 262,144 | 524,288 | 1,024,000 |
|---|---|---|---|---|---|
| single 8192 | 83.0k | 82.9k | 81.5k | 76.8k | 68.2k |
| 4 x 2048 (chip-aligned) | 75.0k | 73.2k | 71.0k | 67.1k | 61.3k |
| 4 x 1792 + 1024 | 73.1k | 71.9k | 68.2k | 64.9k | 60.3k |

- Device ms at 139,264 (one load, MM_TUNED256 on): single 85.2, 4 x 2k 95.1. Packed walls carry more exposed host than
  single (19-22 vs ~14 ms at 139,264, PB.LAD).
- Owner target (`PACKED_RULES.md`): 4 x 2k over ~139k >= 67k new tok/s: met (73.2k).
- Accuracy (PB.MM256 accept gate, 8,192 chunk, 40 layers): err_final_hidden 0.1345 / limit 0.2422 (last), 0.1400 /
  0.2586 (from 0); top-5 1.0; determinism max abs diff 0.0; on-repeats exact.

### 12.5 Ledger state on this branch

- Queue empty after PB.MM256 (`d6bc91514d5`, accepted 2026-11-23). TODO in the state: FIX.11, FIX.13, PB.8P.
- Open owner item (supervision.md, hand-off 2026-11-22): no state in the serving path. The default still reads
  per-slot carries (window ring, ring bookkeeping, engram token-id tail); decided: tt-d-gen sends the tail ids and the
  128-row bounded replay every chunk; to do: a permanent stateless gate test, a default that never reads the carries.
- Serving needs tt-d-gen's packer to agree to 256-aligned segments (`PACKED_RULES.md`).
- Framework: `models/demos/common/bringup/selftest/test_perf_ab.py` has 15 pre-existing failures in this checkout (to bisect).

### 12.6 Shared facts that differ on this branch

- MB.9 (`35d79c6720f`) is not on this branch: no `MM_TUNED160`, `GALAXY160` / `GMM160`, `tests/test_mm160_offon.py`,
  `tests/test_mm160_ladder_hits.py`; the 5,120 chunk runs the configs from before MB.9.
- This branch's `tasks.yaml` has the PB.* tasks, not MB.*: no MB.9 `gate.cmd`, no `results/MB.*` files
  (`MB.9.json`, `MB.9_ladder.md`, `MB.8_ladder.md`). The "Perf now" / "Accuracy now" rows of section 1 and the tables of
  section 3 are model_bringup's; this branch's numbers are in 12.4. Its latest 5,120 single-request ladder is
  `results/PB.9C5.json` (2026-11-21, before PB.MB4 / PB.BM / PB.FBM / PB.MM256): steady 69.3k tok/s at 51,200, 66.4k at
  139,264, 56.0k at 1,024,000.
- The model_bringup fixes exist here as their own tasks: MB.1 / MB.2 as PB.MB / PB.MB5, MB.4 as PB.MB4; MB.6 / MB.8
  came from this branch's PB.BM / PB.FBM.
- `INDEXER_BLOCK_SELECT_MIN=auto` is 163,840 keys here (single request; `tt/indexer.py BSEL_MIN_AUTO`), not 262,144.
- `tests/ladder_common.py` does not exist here; the packed ladders use `tests/packed/ladder_common.py`.
- `bringup/known_issues.md` here has DS41-390; on model_bringup it is cited from `results/MB.9_ladder.md` only.
