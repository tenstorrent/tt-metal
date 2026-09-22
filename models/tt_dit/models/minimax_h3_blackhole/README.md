# MiniMax-H3 on the Blackhole Galaxy: AGMM fused vs unfused, blocking sweeps and roofline — plan

Status: plan, 2026-09-22. Written for an agent starting cold on a 4x8 Blackhole galaxy, on branch
`jameslee/exp_ring_sdpa_wh`. The Wormhole study this repeats is
[`../minimax_h3_wormhole/agmm_fused_vs_unfused.md`](../minimax_h3_wormhole/agmm_fused_vs_unfused.md); read its
Question, Harness and Results sections first. The harness is the same code; the architectural differences are
constants in the tools or values resolved from the device at run time. Results go in `agmm_fused_vs_unfused.md`
beside this file, with the same sections and tables as the Wormhole doc.

## Goal

For the three transformer-block projections that gather the K-sharded activation over the 4-device TP ring
(to_qkv, to_out, ff1: `ttnn.experimental.all_gather_minimal_matmul_async`, `models/tt_dit/layers/linear.py:436-462`),
at the 15 s / 768P / 16:9 per-device shape (M = 13664 rows per device on the 4x8 mesh):

1. the roofline of the whole transformer block with a per-op breakdown, from a Tracy profile of one block;
2. the fastest fused blocking per op, from a sweep with a PCC per combo;
3. the unfused pair, all-gather swept on its hyperparameters and the standalone matmul swept on its blocking, on both
   the AGMM worker grid and the model's full matmul grid;
4. every best time against its compute / DRAM / fabric roofline.

The 4x32 mesh (M = 3424) is out of scope for this pass.

## What differs from Wormhole

| item | Wormhole | Blackhole | where it is decided |
|---|---|---|---|
| production mesh row | (4,8), 4 links, Ring, 4 KB fabric payload, l1_small 32768 | (4,8), 2 links, Ring, 8 KB payload, l1_small 65536, no FSDP | `models/tt_dit/tests/models/minimax_h3/common.py:117-134,152-190`, `models/tt_dit/pipelines/minimax_h3/pipeline_minimax_h3.py:177-209` |
| compute grid | 8x9 | 12x10 | `tests/nightly/sdpa_perf_utils.py:66-67` |
| fused AGMM worker grid, workers per link | 8x8, 2 | 12x9, 6 | `models/tt_dit/utils/matmul.py:407-417,744-750`, `linear.py:397` |
| unfused matmul grid | 8x9 | 11x10 (clamp applied when the mesh has >= 32 devices) | `matmul.py:392-404` |
| `num_buffers_per_channel` | 48 | 24 | `linear.py:451` |
| shipped fused blocking at M = 13664 | ff1 from `grid_88_configs`, others `AGMM_BLOCK_SIZES` | all three `AGMM_BLOCK_SIZES` + subblock (2, 2); the 12x9 table holds M = 3424 rows only | `matmul.py:139-140,385-388,597-624`, `models/tt_dit/models/transformers/minimax_h3/agmm_config.py:49-53` |
| roofline constants | 1.0 GHz, 64 / 72 cores, 288 GB/s, 4 x 12.5 GB/s | 1.35 GHz, 108 / 120 cores, 512 GB/s, 2 x 25 GB/s | `models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py:116-148` |
| SDPA compute cores (block figures) | 63 (7x9) | 110 (11x10) | `models/tt_dit/models/transformers/minimax_h3/attention_minimax_h3.py:189` |
| fabric payload ceiling | 7616 B | 15232 B | `conftest.py:30` |

Handled by the code already: the sweep test resolves the grids and the shipped blocking from the device through the
model's own config functions; the orchestrator detects the arch and picks the production row; the roofline tools
take `--arch bh` / the row's arch. Arch-independent by design, do not change: `L1_BUDGET_KB = 1400`
(`models/tt_dit/utils/sweep_mm_block_sizes.py:585-598`), the all-gather production triple (3, 16, 2)
(`models/tt_dit/parallel/manager.py:933-945`), `AGMM_BLOCK_SIZES`, the two-semaphore-set ping-pong in the sweep.

## Steps

### 0. Environment

```bash
git fetch origin && git checkout jameslee/exp_ring_sdpa_wh   # build as usual; reinstall pinned packages if rebuilt
source python_env/bin/activate
python -c "import ttnn; print(ttnn.get_arch_name())"          # must print: blackhole
tt-smi                                                          # 32 chips, links up
python -m pytest models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py::test_h3_agmm_sweep \
  --collect-only -q -p no:cacheprovider | grep -c "blackhole-.*bh4x8links2_ring"   # 36 ids
```

The pytest id prefix must be `blackhole` (the orchestrator builds node ids from `ttnn.get_arch_name()`). The
Wormhole row is skipped on this machine by its mark.

### 1. Block profile and roofline figures with the per-op breakdown

```bash
scripts/run_safe_pytest.sh --profile \
  "'models/tt_dit/tests/models/minimax_h3/test_performance_minimax_h3.py::test_minimax_h3_transformer_block_perf[blackhole-sp_sim1-15s_768p-4x8sp1tp0nl2_ring_is_fsdp0]'" \
  -s --timeout 3600
# it prints generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv
python models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py --arch bh \
  --profile-csv generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv --dump --figs all \
  --out-dir transformer_roofline_out_bh
```

The test id is the Blackhole row of `GALAXY_RING` (`common.py:165-167`) crossed with the duration and sp_sim
parameters (`test_performance_minimax_h3.py:91-106`); it runs two iterations and profiles the second between the
`start` / `stop` signposts. The node id is quoted twice because `python -m tracy` re-shells its argv
(`tools/tracy/__main__.py:368`). `--dump` prints the per-op block table (measured / ideal / limiter / utilization /
bound formula); `--figs all` writes `roofline_bh_M13664.png`, `block_stacked_bh_M13664.png`,
`block_ops_bh_M13664.png`, `time_bars_M13664.png`, `stacked_M13664.png`, `nstar_links.png`. Record the block table
and the three AGMM rows in the results doc. Op codes are the same as on Wormhole for the 4x8 row (the exp ring SDPA
is a 4x32 feature, `attention_minimax_h3.py:211`).

### 2. PCC gate for the shipped blockings, fused and separate

```bash
python -m pytest models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py \
  -k "bh4x8links2 and h3_15s and check" -p no:cacheprovider --timeout 1800
```

Runs to_qkv and to_out through `test_linear` and ff1 through `test_linear_swiglu`, both ways. Note these unit rows
use the 4 KB fabric payload the Wan2.2 tests were written with; the sweep row uses the production 8 KB.

### 3. Sweeps at M = 13664

```bash
TOOL=models/tt_dit/tests/models/minimax_h3/tools/agmm_unit_sweep.py
# smoke: one shipped combo per mode; arch and device row are detected
python $TOOL run --ops to_out --M 13664 --modes fused,mm_ring,mm_full --combos '[[8,8,6,2,2]]'
python $TOOL run --ops to_out --M 13664 --modes ag --combos '[[3,16,2,0,0]]'
```

Check the sidecars in `generated/agmm_h3_sweep/`: `arch` blackhole, `grid` 12x9 (fused, mm_ring) and 11x10
(mm_full), `full_grid` 12x10, `num_buffers_per_channel` 24, `shipped_blocks` [8, 8, 6, 2, 2] for to_out, and one OK
row each in `agmm_h3_sweep_results.csv`. Then the study, one device session per case, sequentially:

```bash
for mode in ag fused mm_ring mm_full; do
  for op in to_qkv to_out ff1; do
    python $TOOL run --ops $op --M 13664 --modes $mode --quiet --keep-going --timeout 14400
  done
done
```

Run it detached (`setsid nohup ... &`) with a log, as `generated/agmm_h3_sweep/run_all.sh` did on Wormhole, and
watch for `done` lines. Budget: candidate counts are of the Wormhole order (per-core M 36 tiles, K per device 42 or
56, N per core 19 / 5 / 25 or 26), about 300 to 600 combos per matmul case and 48 per all-gather case; per-combo
time on Wormhole was 6 to 20 s and mostly host work (compile, trace capture, PCC on 13664-row tensors), so expect
about an hour per matmul case, 10 minutes per all-gather case, roughly 10 hours for the twelve cases. `--M 9184
4736` afterwards is optional and repeats the Wormhole lengths.

### 4. Tables

```bash
python $TOOL report --top 10 --out generated/agmm_h3_sweep/report_bh_M13664.md
python $TOOL roofline --out generated/agmm_h3_sweep/roofline_bh_M13664.md
```

Both pick the Blackhole row and constants from the CSV. `report` gives the fused ranking, the all-gather ranking,
the two standalone-matmul rankings (each with the blocking the model would use marked) and the fused-vs-unfused
summary; `roofline` gives every best time against its compute / DRAM / fabric bound with cores taken from the row's
grid.

### 5. Write `agmm_fused_vs_unfused.md` in this directory

Mirror the Wormhole doc section for section: Question; Harness (the Blackhole row, the constants above, what the
tools resolved); Results at M = 13664 with the summary table (best fused, best all-gather, best matmul on each grid,
AG + MM, fused vs unfused), the fused top-3 with the shipped rank, the all-gather ranking with the production point,
the standalone matmul rank-1 against the model's blocking, and numbered findings; Best measured vs roofline (the
compact utilization table, then the full table); Caveats. Cite code by file:line. Code comments never name a doc.
Add one paragraph to `../MiniMaxH3.md` where this directory is linked.

## Predictions to check

From the Wormhole results and the Blackhole constants (`transformer_roofline.py --arch bh --no-block --dump`):

1. **Fabric bounds are identical in microseconds** (2 x 25 GB/s = 4 x 12.5 GB/s aggregate) while compute bounds
   shrink 2.3x (298.6 vs 131 TFLOP/s at HiFi2): 1102 us for the K = 5376 gather, 1469 us for K = 7168.
2. **Fused to_out is fabric-bound on Blackhole**: fabric 1469 us against compute 882 us on 108 cores (the
   compute/fabric crossover `n_star` is 2240 columns, N = 1344 is below it). On Wormhole it was compute-bound. Expect
   its fused time to be set by the gather (at Wormhole's 58% link efficiency about 2500 us) and its fused-vs-unfused
   margin to be the smallest of the three, possibly negative once the standalone matmul on 110 cores is added to a
   standalone gather. For to_qkv and ff1 fabric is 42% / 31% of compute (Wormhole 18% / 14%), so the fused op's
   overhead over `mm_ring` should grow from Wormhole's 0 to 3% to several percent.
3. **M padding**: 427 M tiles over 12 columns is 36 per core, exactly Wormhole's M = 9184 case; predict M_block 12
   (three blocks, no padding) or 9 wins the fused sweeps and the shipped M_block 8 (five blocks, pads to 40) trails by
   2 to 5%, as on Wormhole at 9184 (+1.7% / +5.4% / +5.3%). The closest Wormhole analogue for per-core work is
   `mm_full` at M = 9184 (36 x N/9): winners (12, 7, 8), (12, 8, 5, 4, 1), (12, 4, 10).
4. **N per core on 9 rows**: to_qkv 168 tiles -> 19 (odd), to_out 42 -> 5, ff1 224 -> 25 (odd; the SwiGLU op splits
   gate/up pairs, expect 26). The shipped ff1 (8, 3, 14) is likely further from rank 1 than Wormhole's 0.1%.
5. **11x10 vs 12x9**: 1.9% more cores, but N over 10 rows pads to_out's 42 tiles to 50 and to_qkv's 168 to 170, M
   over 11 columns pads 427 to 429; predict `mm_full` roughly equal to or slower than `mm_ring` for to_out and the
   Wormhole "9th row" gain (4 to 10%) mostly gone.
6. **Shipped blockings**: on Blackhole at M = 13664 the model resolves `AGMM_BLOCK_SIZES` + (2, 2) for all three
   ops (`get_agmm_config` -> `get_matmul_config`, table miss); the sidecar's `shipped_blocks` shows what it resolved.
   The 12x9 table entries at `matmul.py:386-388` are for M = 3424 and differ ((4, 7, 14), (9, 8, 5, (3, 1)),
   (6, 7, 12)).
7. **Standalone all-gather**: Wormhole reached 58% of link bandwidth at every size. First Blackhole number; the 8 KB
   payload may move the `chunks_per_sync` optimum (32 won everywhere on Wormhole).

## Gotchas carried over from the Wormhole run

- `python -m tracy` re-shells argv: pass exact node ids (the orchestrator does); `-k "a and b"` breaks.
- `pytest.ini` sets a 300 s timeout; the orchestrator passes `--timeout`, the block profile command needs `--timeout
  3600`.
- Ring calls queued back to back on one semaphore set hang and a hang wedges the ETH heartbeat; the sweep alternates
  two semaphore sets and two persistent buffers per call, do not simplify it. Recovery: `tt-smi -r all`, wait about
  75 s.
- `TT_METAL_PROFILER_MID_RUN_DUMP=1` is needed for `ReadDeviceProfiler`; the orchestrator sets it. Frequent mid-loop
  flushes stalled a Wormhole galaxy; the sweep flushes only after warmup and after the measured pass.
- Host tensors go to the device as bf16 (`_host_tensor`); fp32 tilize of a 13664-row activation takes minutes.
- to_out's gate is per token, shape [M, N] (`transformer_block_minimax_h3.py:231-234,293-294`); a broadcast gate
  understates the epilogue by about 10%.
- The L1 pre-filter (`estimate_l1_kb`) lets a few combos through that the op rejects at program creation; they are
  recorded as `skipped` in the sidecar and CSV, not failures.
