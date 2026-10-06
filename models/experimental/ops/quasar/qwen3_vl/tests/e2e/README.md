# Qwen3-VL-4B e2e on Quasar

One pytest (`test_qwen3_vl_e2e.py`) runs Qwen3-VL-4B-Instruct cut down to V vision blocks, T text layers and K decode steps, and compares every stage against the same cut-down HF model on the host. Three wrapper scripts run it on WH/BH (ttsim or hardware), craq-sim and the emu-quasar-2x3 emulator with identical settings.

Ops that work on WH/BH but not on Quasar, simulator issues and model-copy bugs found along the way are logged in [QUASAR_GAPS.md](QUASAR_GAPS.md).

## Run

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) TT_METAL_RUNTIME_ROOT=$(pwd) PYTHONPATH=$(pwd):$PYTHONPATH

E2E=models/experimental/ops/quasar/qwen3_vl/tests/e2e
$E2E/run_wh_bh.sh --ttsim wh           # WH baseline on ttsim, emulator-sized grid
$E2E/run_wh_bh.sh --grid native        # WH/BH hardware, full grid
$E2E/run_craq.sh                       # craq-sim, emulator-sized grid
$E2E/run_craq.sh --grid 8x4            # craq-sim, full grid
$E2E/run_emu.sh                        # emulator (needs NNG_SOCKET_ADDR from your IRD reservation)
```

Script defaults: `--size tiny --vision-layers 2 --text-layers 2 --decode-steps 1 --deepstack-at 0 --debug default`.

| Flag | Meaning |
|---|---|
| `--size tiny\|demo` | tiny: 256x256 image, 64 image tokens, 78-token prompt padded to 128. demo: the graph-capture sizes (2752 image tokens, 4096 prefill). |
| `--vision-layers N`, `--text-layers N` | blocks of the 24-block vision tower / 36-layer text model to run |
| `--decode-steps K` | teacher-forced decode steps after prefill (0 = prefill only) |
| `--deepstack-at I\|real` | move the deepstack tap to vision block I (in TT and HF alike); `real` keeps 5/11/17 |
| `--kv-blocks N` | paged KV-cache blocks of 32 tokens (default: preset) |
| `--host-ops a,b\|all` | run these ops on the host (bisecting only; the run becomes DIAGNOSTIC) |
| `--disable-wa a,b` | turn off named workarounds from `op_overrides.py` |
| `--debug fast\|default\|deep` | fast: no watcher. default: watcher + LLK asserts, NoC sanitize off. deep: plus dump-all, noinline, per-RISC DPRINT, debug logging |
| `--noc-sanitize` | re-enable NoC sanitize (20-30x slower) |
| `--fp32-dest-acc` | keep fp32 dest accumulation in the Quasar config (hardware A/B; undefined on ttsim WH) |
| `--timeout S` | pytest timeout (WH/BH and emulator 14400, craq-sim 3600) |
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

Thresholds per stage and preset live in `thresholds.json`.

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

Override locations with `QWEN_TTSIM_DIR`, `QWEN_CRAQ_SIM`, `QWEN_EMU_DIR`.

## Hangs

A hang shows up as the pytest timeout. The script prints the last lines of `progress.log`; the watcher log has per-core waypoints (default/deep profiles). On simulators `TT_METAL_OPERATION_TIMEOUT_SECONDS` is ignored (`tt_metal/llrt/llrt.cpp`, `wait_until_cores_done`). For a hard-to-localize hang, temporarily patch it locally to honor that variable when set; do not commit that change.

## Cherry-picks applied

| PR | Commits | Why | Drop when |
|---|---|---|---|
| [#59510](https://github.com/tenstorrent/tt-metal/issues/59510) (not a PR yet) | `[droppable] WH exp: SFPNOP after SFPMAD ...` | WH approx exp returns zeros with `TT_METAL_DISABLE_SFPLOADMACRO=1` (SDPA outputs zeros on WH silicon) | the upstream fix lands |

## Baselines

| Target | Grid | Size | V/T/K | Verdict | Wall time | Run folder |
|---|---|---|---|---|---|---|
| ttsim WH | 8x8 | tiny | 2/2/1 | PASS (all stages >= 0.9993) | 10 min | `generated/qwen3_vl_quasar/wh_bh_wh_native_quasar/20261006T154929Z` |
| ttsim WH | 2x1 | tiny | 2/2/1 | PASS (all stages >= 0.9985) | 10 min | `generated/qwen3_vl_quasar/wh_bh_wh_2x3_quasar/20261006T153843Z` |
| WH N150 silicon | 8x9 | tiny | 2/2/1 | PASS (all stages >= 0.9993), with the #59510 fix commit | — | aus-wh-08 `wh_bh_hw_native_quasar/20261006T182734Z` |

Blackhole (ttsim and hardware) is not part of the baseline.
