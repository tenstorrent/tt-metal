# BGE-M3 S512 Blackhole helper scripts

Working scripts from the p150a B8/B16/B32 optimisation. They are not part of the
model. Remove this directory (or drop its commit) before the merge.

Run from the tt-metal root:

```
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_MODEL=BAAI/bge-m3 TT_VISIBLE_DEVICES=0
H=models/demos/wormhole/bge_m3/tt/helper_scripts
```

Some logs contain binary bytes; use `grep -a`.

## Timing

| script | use |
|---|---|
| `bench_nomask.py <batch> [n]` | Traced forward, no_padding=True, n replays (default 30). Burst time (< 1 s of load). Prints `BENCH`. |
| `bench_nomask_dump.py <batch> [n]` | Same, plus every replay time (`TIMES`). Shows the clock ramp-down. |
| `bench_sustained.py <batch> [n] [soak_s]` | 3 s idle, 20 burst replays, soak, then n timed replays. Standard: `300 10`. Prints `SUSTAINED ... burst_mean=`. |
| `telemetry.sh` | tt-smi telemetry every ~1 s: `time V A W MHz C`. Run with nohup, kill afterwards. Needs `tt-smi` on PATH. |
| `ab_bench.sh <batch> <rounds> <file> <A copy> <B copy>` | A/B timing over alternating fresh processes. B1 varies ~0.1 ms between processes, so use it for any B1 claim below ~0.1 ms. Restores `<file>` from `<A copy>`. |

## Accuracy

| script | use |
|---|---|
| `pcc_batch.py` | End-to-end PCC against HF. `PCC_BATCH=8/16/32`; `PCC_NOPAD=1` runs no_padding=True. Prints `CELL_PCC`. |
| `pcc_b8.py` | The B8 version of the same (default path). |

## Sweeps

| script | use |
|---|---|
| `sweep_inmodel.py <batch> <qkv\|ao\|wo\|wi>` | Builds the model once, sets one 2D multicast matmul config on all 24 layers, and times the traced forward per candidate (grids, in0_block_w, subblocks, out_block_h = pm, pm/2, pm/4). Prints `RESULT` lines and a `SUMMARY`. |
| `sweep_headgroups.py`, `sweep_headgroups.sh` | Head-split and concat head_groups at B8. The .sh edits attention.py in place and restores it. |
| `sdpa_sweep.py` | Isolated SDPA chunk sweep (legacy kernel). In-model results differ; confirm in the model. |
| `sweep_minimal_inmodel.py <batch> <wi\|wo>` | In-model sweep of minimal_matmul configs (M/K/N blocks, subblocks, full device grid) for the MLP wi or wo against the 2D incumbent. |
| `sweep_sdpa_inmodel.py <batch> [mcphb list] [q list] [k list]` | In-model sweep of the S512 SDPA q/k chunks and max_cores_per_head_batch for one batch. Rewrites attention.py per variant and restores it. |
| `sweep_b1_sdpa.py` | In-model sweep of the B1 SDPA q/k chunks and grid (8x8 or device). Rewrites attention.py per variant and restores it. |
| `sweep_bge.py` | Matmul config ranking with the external tt-optimization-loop (host paths inside). |

## Profiling and analysis

| script | use |
|---|---|
| `stack.py <csv> <wall_ms>` | Per-op table from a tt-perf-report CSV. |
| `raw_stack.py <ops_perf_results.csv> [wall_ms]` | Per-op table straight from the raw Tracy CSV (no tt-perf-report), with core counts. |
| `extract_tracy.py` | Per-op-code kernel-time table from one Tracy CSV. |
| `find_cast.py` | Wraps ttnn.typecast and prints each call site. |
| `roofline.py`, `roofline_v2.py`, `roofline_b1.py` | Roofline estimates for the matmuls and SDPA (v2 uses the real L1/DRAM residency). |
| `micro_heads.py`, `probe_variants.py`, `probe_qkv_in_dtype.py` | Short probes of the head-split variants and the QKV input dtype. |

## Porting (one-off, host paths inside)

`port_qkv.py`, `port_concat.py`, `inv.sh`: splice or compare the p150 reference ops against main.
They point at `/local/ttuser/gtobar/p150_reference` on `sjc-snva-tp100`.

Galaxy (tt-galaxy-bh): pass `-t 8086` to tracy. /etc/hosts maps the hostname to an address no
interface has, so the port search fails. tt-perf-report is not in the venv there; use raw_stack.py.

Tracy capture, then report:

```
TT_METAL_DEVICE_PROFILER=1 python_env/bin/python -u -m tracy -p -r --no-runtime-analysis -v -m pytest \
  "models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py::test_bge_m3_tracy_perf[device_params0-batch8-nomask]" -sv
python_env/bin/tt-perf-report <ops_perf_results.csv> --arch blackhole --start-signpost start \
  --end-signpost stop --no-color --no-advice --csv rep.csv
python3 $H/stack.py rep.csv <wall_ms>
```
