# Qwen3-VL-4B e2e on Quasar

One pytest (`test_qwen3_vl_e2e.py`) runs Qwen3-VL-4B-Instruct cut down to V vision blocks, T text layers and K decode steps, and compares every stage against the same cut-down HF model on the host. Three wrapper scripts run it on WH/BH (ttsim or hardware), craq-sim and the emu-quasar-2x3 emulator with identical settings.

Ops that work on WH/BH but not on Quasar, simulator issues and model-copy bugs found along the way are logged in [QUASAR_GAPS.md](QUASAR_GAPS.md).

## Run

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) TT_METAL_RUNTIME_ROOT=$(pwd) PYTHONPATH=$(pwd):$PYTHONPATH

E2E=models/experimental/ops/quasar/qwen3_vl/tests/e2e
export QWEN_CRAQ_SIM=/localdev/$USER/quasar-sim/libttsim.so  # the staged craq-sim build; run_craq.sh's default is an old copy
$E2E/run_wh_bh.sh --ttsim wh           # WH baseline on ttsim, emulator-sized grid
$E2E/run_wh_bh.sh --grid native        # WH/BH hardware, full grid
$E2E/run_craq.sh                       # craq-sim, emulator-sized grid
$E2E/run_craq.sh --grid 8x4            # craq-sim, full grid
$E2E/run_emu.sh                        # emulator (needs NNG_SOCKET_ADDR from your IRD reservation)
```

The emulator backs only 64 MiB of DRAM per channel although its descriptor says 1 GiB; larger allocations silently overwrite earlier ones, and the e2e (~1.3 GiB of weights) does not fit (QUASAR_GAPS S7, #55417). Since #60444 tt-metal clamps each emulator channel to those 64 MiB, so the e2e fails at weight upload with an OOM. To run every stage anyway, pass a copy of the emulator build dir under a name tt-metal does not recognise as `QWEN_EMU_DIR` (`/proj_sw/user_dev/$USER/tt-umd-simulators/build/emu-quasar-2x3-uncapped/`, identical except `EMULATOR_TIMEOUT=14400`): tt-metal then uses the descriptor's 1 GiB, DRAM wraps, and PCC is wrong from `vision.block0.attn` on. Whether to keep this workaround or drop the emulator e2e is an open decision.

Script defaults: `--size tiny --vision-layers 2 --text-layers 2 --decode-steps 1 --deepstack-at 0 --debug default` (`--debug fast` on craq-sim, where the watcher slows each op 150-300x).

| Flag | Meaning |
|---|---|
| `--size tiny\|demo` | tiny: 256x256 image, 64 image tokens, 78-token prompt padded to 128. demo: the graph-capture sizes (2752 image tokens, 4096 prefill), WH/BH only: Quasar runs use tiny, and the vision QKV/WO program configs do not fit small Quasar grids at demo sizes. |
| `--vision-layers N`, `--text-layers N` | blocks of the 24-block vision tower / 36-layer text model to run |
| `--decode-steps K` | teacher-forced decode steps after prefill (0 = prefill only) |
| `--deepstack-at I\|real` | move the deepstack tap to vision block I (in TT and HF alike); `real` keeps 5/11/17 |
| `--kv-blocks N` | paged KV-cache blocks of 32 tokens (default: preset) |
| `--host-ops a,b\|all` | run these ops on the host (bisecting only; the run becomes DIAGNOSTIC) |
| `--disable-wa a,b` | turn off named workarounds from `op_overrides.py` |
| `--debug fast\|default\|deep` | fast: no watcher. default: watcher + LLK asserts, NoC sanitize off. deep: plus dump-all, noinline, per-RISC DPRINT, debug logging |
| `--noc-sanitize` | re-enable NoC sanitize (20-30x slower) |
| `--fp32-dest-acc` | keep fp32 dest accumulation in the Quasar config (hardware A/B; undefined on ttsim WH) |
| `--timeout S` | pytest timeout (WH/BH and emulator 14400, craq-sim 10800) |
| `--ttsim wh\|bh`, `--grid 2x3\|native`, `--config quasar\|native` | `run_wh_bh.sh` only; `--config native` runs the unmodified bf8 config as a hardware control |
| `--grid 2x3\|8x4` | `run_craq.sh` only |
| `-- <args>` | passed to pytest, e.g. `-- --qwen-dump-stages` to save golden and TT stage tensors, or `-- --qwen-allow-uncertified` to use host fallbacks not yet checked against the real op (`test_fallbacks.py`) |

Exit codes: 0 PASS, 3 DIAGNOSTIC (some ops ran on the host), anything else FAIL or error.

## Output

Each run writes `generated/qwen3_vl_quasar/<target>/<UTC time>/`:

| File | Content |
|---|---|
| `pcc.md` | verdict, first failing stage, per-stage PCC / threshold / max abs error / seconds, workaround and host-op hit counts |
| `verdict.txt` | `PASS`, `FAIL` or `DIAGNOSTIC` |
| `progress.log` | one `PRE`/`POST` line per ttnn op with shapes, memory configs and model stage. After a hang, the last `PRE` without a `POST` is the op that never finished. |
| `run.log`, `command.txt`, `env.txt`, `git.txt` | full output, exact command, environment, commit and branch commits |
| `ops.pt` | with `-- --qwen-capture-ops STAGE_PREFIX` (e.g. `vision.block0`, `text.decode`): every top-level op output of those stages. `repros/diff_ops.py RUN_A RUN_B` prints where two runs first diverge |
| `prefill_snapshot.pt` | KV cache, decode position, rope delta, page table and prefill stage tensors at the end of prefill. `-- --qwen-resume-prefill <run folder>` skips vision and prefill and decodes from it (same size, layers, grid and config only; the run notes it in `pcc.md`) |

Thresholds per stage and preset live in `thresholds.json`.

`progress.log` also records each op's program, compute and memory configs, so a failing call can be replayed in a repro. `-- --qwen-vision-only` stops after the vision tower (vision stages only), for short runs on slow targets.

## Grids

The emulator build exposes a 2x1 compute grid (`tt_metal/core_descriptors/quasar_simulation_2x3_arch.yaml`). `--grid 2x3` reproduces it elsewhere with `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE`, which sets the grid's *end* coordinate, so the value depends on where each target's compute grid starts:

| Target | Native grid | Override for 2x1 |
|---|---|---|
| ttsim WH | 8x8 | `1,0` |
| ttsim BH | 11x10 | `1,0` |
| craq-sim | 8x4 | `3,2` |
| emu-quasar-2x3 | 2x1 | none |

## ttsim setup (WH/BH baselines)

```bash
for a in wh:wormhole_b0_80_arch bh:blackhole_140_arch; do
  n="${a%%:*}"; y="${a##*:}"; d="/localdev/$USER/ttsim/sim_${n}"
  mkdir -p "$d" && cp "/localdev/$USER/ttsim/src/_out/release_${n}/libttsim.so" "$d/" \
    && cp "tt_metal/soc_descriptors/${y}.yaml" "$d/soc_descriptor.yaml"
done
```

Override locations with `QWEN_TTSIM_DIR`, `QWEN_CRAQ_SIM`, `QWEN_EMU_DIR`. `run_craq.sh` defaults to `/localdev/$USER/sim/libttsim.so`, an old craq-sim copy (2026-10-01); every craq-sim run recorded here used `QWEN_CRAQ_SIM=/localdev/$USER/quasar-sim/libttsim.so`, except the 2026-10-28 2x1 runs, which used a build of craq-sim#538 before it was merged and staged.

On WH/BH the vision-token merge copies rows on the host: `QuasarModelArgs.device_scatter` is on only on Quasar. No simulator can check the device path there yet: ttsim WH aborts on scatter's int32 index (QUASAR_GAPS S3, a ttsim gap, not a WH bug) and ttsim BH aborts at the first op (tt-blaze#5898). The upstream demo (`models/demos/qwen3_vl`) scatters on device on WH.

## Hangs

A hang shows up as the pytest timeout. The script prints the last lines of `progress.log`; the watcher log has per-core waypoints (default/deep profiles). On simulators `TT_METAL_OPERATION_TIMEOUT_SECONDS` is ignored (`tt_metal/llrt/llrt.cpp`, `wait_until_cores_done`). For a hard-to-localize hang, temporarily patch it locally to honor that variable when set; do not commit that change.

## Graph-capture op tests (`qwen3_vl_ops`)

`models/experimental/ops/quasar/tests/qwen3_vl_ops/` (on main) replays every distinct call of a WH N150 demo capture: demo sizes, bfp8/bfp4 dtypes and WH program/shard configs, so some cases do not fit Quasar as captured. On craq-sim we run it with `TTNN_GRAPH_OPS_BF8_AS_BF16=1` (block-float cases run as bf16), one pytest invocation per case (a craq-sim abort or hang loses only that case), and `test_add.py` / `test_multiply.py` a second time through `ttnn.experimental.quasar.add` / `.multiply` (the base ops are not ported to Quasar; that is expected, not a finding).

| Date | Target | Cases | Pass | Skip (captured config too large for the device) | Fail | Timeout | Wall time |
|---|---|---|---|---|---|---|---|
| 2026-10-26 | craq-sim 8x4 (.so md5 74c8eb33), branch on main fb6d34ceaed | 134 (+12 experimental add/mul) | 94 (+10) | 23 (+2): 8x5 L1 shard grids, 12-bank DRAM sharding | 16 | 1 | 1 h 45 min (95 min in tests; 12-50 s per case) |
| 2026-10-26 | craq-sim 8x4, main 5285b13d773 with the DFB drain (no #57646 revert, no tensor_layout droppable) | 134 (+12) | 94 (+10) | 23 (+2) | 16 | 1 | 1 h 34 min (88 min in tests); every case's outcome identical to the run above |

The 16 failures: base add (7) and multiply (3), all passing through the experimental ops; `linear` with a captured 8x8 matmul grid (2); `plus_one` int32 and uint32 (2, Q6); `scatter` (1, Q1); `layer_norm` `[2752, 4096]` at PCC 0.998996 against 0.999 (1). The timeout is prefill SDPA `[16, 12288, 64]` (more than 20 min on craq-sim; the `[32, 4096, 128]` case passes in 13 min).

## Baselines

Latest result per target. Tiny size, 2 vision / 2 text layers, 1 decode step. Wall time is the first-to-last-op span from `progress.log` (setup and weight loading add a few minutes). craq-sim is the staged `release_qsr` build with craq-sim#538 (md5 7f4a2163), except where noted.

| Target | Grid | Verdict | Wall time |
|---|---|---|---|
| craq-sim, main c4725ba800a | 8x4 | PASS (prefill 0.99681, decode0 0.99841; lowest `text.layer1.mlp` 0.99433) | 15 min |
| craq-sim, main 9c941a0cc3a | 2x1 | PASS (prefill 0.99698, decode0 0.99861; lowest `text.layer1.mlp` 0.99427) | 42 min |
| emu-quasar-2x3, DRAM clamp bypassed (`emu-quasar-2x3-uncapped`, see Run), main 1e85fb6a52e with #62744 | 2x1 | runs every stage (1211 device ops; no hang or crash); FAIL on PCC from `vision.block0.attn` on, the accepted DRAM-wrap effect (S7) | 217 min ops (3 h 38 min pytest) |
| ttsim WH (md5 e9c03997); `ttsim_wh_host_cast_fp32_upload` | 2x1 | PASS (prefill 0.99943, decode0 0.99950) | 10 min |
| WH N150 silicon | 8x9 | PASS (all stages >= 0.9993), with the #59510 fix | — |

Blackhole (ttsim and hardware) is not part of the baseline.
