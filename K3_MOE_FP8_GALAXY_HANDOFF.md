# K3 MoE fp8 wire: Galaxy 8×4 handoff

**Goal (G6 in `K3_MOE_FP8_WIRE_PLAN.md`):** validate the fp8 dispatch/combine wire for the Kimi K3 prefill MoE on
the 8×4 Galaxy — numerics (PCC) first, then perf against the bf16 wire — and report the numbers back into the plan.
Everything so far ran on a 2×4 Blackhole LoudBox at Galaxy-equivalent per-chip load.

- Branch: `pjosipovic/k3-moe-fp8-wire` (based on `main`). The plan and full history: `K3_MOE_FP8_WIRE_PLAN.md`.
- Recent commits worth knowing (newest first):
  - `e7e3c59a71f` combine: 4-row ROW_MAJOR batches over all untilizers, pipelined fabric sender (381 → 251 µs)
  - `8f827ae14d2` combine output to L1; fast DQ2 scale staging in post_combine_reduce
  - `2075be95433` DQ1 (`per_token_cast_back` tile output): multicast valid rows, fast scale staging
  - `c536d6b205b` Q2 output may go to L1 (`MOE_Q2_L1`)
  - `76c1537a786` Q2: multicast valid rows, fused SFPU amax+pow2, 1 KB write groups
  - `d541d5b35db` shared expert split across two overlap windows (beside dispatch / beside combine)

## What the fp8 wire is

Tokens cross the fabric as packed rows `[H e4m3 | H/32 fp32 power-of-two scales]` (1×32 scale blocks).

| Point | Op | Where |
|---|---|---|
| Q1 | `per_token_cast_to_fp8` | on each chip's latent slice, before the fp8 `high_bw_all_gather` and dispatch |
| DQ1 | `per_token_cast_back(tile_output=True)` | after dispatch, writes bf8_b tiles for the routed expert |
| Q2 | `per_token_cast_to_fp8(token_count_aware=True)` | after the routed expert, packs the combine source |
| DQ2 | fused into `post_combine_reduce` | dequant + weighted top-k reduce |

## LoudBox results (2×4, K3, 224 experts = 28 per chip, ~2560 dispatched rows/chip, traced)

| | bf16 wire | fp8 wire |
|---|---|---|
| traced T_moe p50 | 4.788 ms | **4.620 ms** (min 4.604) |
| dispatch / combine | 582 / 336 µs | 325 / 251 µs |
| post_combine_reduce | 134 µs | 101 µs |
| Q1 / DQ1 / Q2 | — | 12 / 73 / 69 µs |
| final_output PCC | 0.994239 | 0.994043 |

(bf16 numbers include the combine/PCR changes that help both wires; before them bf16 was 4.93 ms.)

Expected on the Galaxy: dispatch/combine span more hops (sp = 8, up to 4 hops on axis 0), so the wire saving should
be larger than on the LB; the Q/DQ costs are per-chip and should be about the same. Treat LB gains as a lower bound.

## Build

```bash
./build_metal.sh --release
source python_env/bin/activate
```

Run tests through `scripts/run_safe_pytest.sh` (adds dispatch timeouts and cleanup), not bare `pytest`.

## The fp8 wire is selected by environment variables

`run_model` (used by all tests below) reads them; without them the run is the bf16 baseline.

```bash
export MOE_DISPATCH_WIRE=fp8 MOE_COMBINE_WIRE=fp8 MOE_FP8_SCALE_BLOCK=32
```

Feature flags (defaults are the shipped configuration; flip one only to bisect):

| Flag | Default | Meaning |
|---|---|---|
| `MOE_COMBINE_L1` | 1 | combine output (PCR source) in L1 interleaved |
| `MOE_Q2_L1` | opposite of `MOE_COMBINE_L1` | Q2 packed rows in L1 (both together do not fit beside the shared expert) |
| `MOE_SPLIT_SHARED` | on with fp8 dispatch | shared expert split: gate/up beside dispatch, down + RS beside combine |
| `MOE_Q1_PRE_GATHER` | 1 | Q1 before the latent all-gather (segmented rows) |
| `MOE_DQ1_TILE` | 1 | DQ1 writes bf8_b tiles for the expert |
| `MOE_FUSED_DQ2` | 1 | DQ2 inside post_combine_reduce |

## Steps on the Galaxy

1. **PCC (numerics).** The K3 test has an 8×4 case:
   ```bash
   MOE_DISPATCH_WIRE=fp8 MOE_COMBINE_WIRE=fp8 MOE_FP8_SCALE_BLOCK=32 scripts/run_safe_pytest.sh \
     "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_kimi_k3_moe" -k "torus-xy-8x4 and pcc"
   ```
   Pass criteria: `final_output` PCC ≥ 0.994 (LB fp8: 0.994043, bf16: 0.994239), `routed_output` ≥ 0.96. Run the same without the
   three env vars for the bf16 reference numbers.
2. **Traced perf, fp8 vs bf16.** The scratch LB perf test takes the mesh from the environment:
   ```bash
   K3_MESH=8x4 K3_FABRIC=torus_xy K3_TRACE=1 K3_TRACE_ITERS=20 \
   MOE_DISPATCH_WIRE=fp8 MOE_COMBINE_WIRE=fp8 MOE_FP8_SCALE_BLOCK=32 scripts/run_safe_pytest.sh \
     models/demos/deepseek_v3_d_p/tests/perf/test_scratch_k3_moe_perf_lb.py
   ```
   It prints `K3_TRACE traced ms: min .. p50 ..`. Leave `K3_NUM_EXPERTS` unset on the Galaxy (all 896 experts =
   28 per chip; the LB used 224 to get the same per-chip load). Repeat without the three wire vars for bf16.
3. **Per-op device times.** Same command with `--profile` (and `K3_PERF_ITERS=1`, no `K3_TRACE`); the ops perf CSV
   path is printed at the end. Ops of interest: `DispatchDeviceOperation`, `CombineDeviceOperation`,
   `PostCombineReduceDeviceOperation`, `PerTokenCastToFp8DeviceOperation` (bf16 input = Q1, bf8_b input = Q2),
   `PerTokenCastBackDeviceOperation` (DQ1), `UnifiedRoutedExpertFfnDeviceOperation`. Take the median per op.
   Profiler logs are large: delete `generated/profiler/.logs/*.csv`, the `.tracy` file and old
   `generated/profiler/reports/<date>` folders after each run.
4. **CI perf gate (optional, report-only).** `test_kimi_moe_perf.py::test_kimi_moe_perf_galaxy[k3]` on 8×4
   (see the plan's CI-gate row: expected 6.21 ms ± 3% for the bf16 baseline; set the expected value to `None` for a
   report-only recalibration). Run it with and without the fp8 env vars.

**PCC must be run with the three wire env vars set**; without them the run is bf16 (an LB helper script missed this
once and reported the bf16 PCC as fp8).

## Risks and what to do

- **L1 fit.** Two L1-interleaved buffers were placed against per-core budgets measured on the LB. If a run fails
  with `Statically allocated circular buffers ... clash with L1 buffers`, set `MOE_COMBINE_L1=0` (and keep
  `MOE_Q2_L1=0`); everything falls back to DRAM (costs ~30–50 µs).
- **Fabric packet size.** `KimiK3Config.FABRIC_PAYLOAD_SIZE` is used as is; a sweep on the LB showed no gain from
  changing it (4160 / 4352 / 8704 B within noise). `K3_FABRIC_PAYLOAD` overrides it in the scratch perf test.
- **Repeated mesh open/close** in one pytest session tripped ethernet-init timeouts on the LB; reset the boards
  between runs if a run hangs at fabric init (`tt-smi -r` on the LB; use the Galaxy's reset procedure).
- **Hangs** in our own experiments came only from NoC1 multicast/read misuse; nothing on the branch is known to hang.

## What to report back (into `K3_MOE_FP8_WIRE_PLAN.md` §0 status / G6 row)

- PCC for fp8 and bf16 (`final_output`, `routed_output`).
- Traced T_moe p50/min for fp8 and bf16, and the per-op medians above.
- Any L1 clash or flag that had to change.

## Combine on the Galaxy

Combine was found to be imbalance- and per-row-bound on the LB, not link-bound (see the plan's Combine note). If the
Galaxy fp8 combine win is still small, profile per-core spans of `CombineDeviceOperation`: senders are the first core
of each group on the dispatch row; compare sender writer end vs reader end vs untilizer ends.

## Open follow-ups (not needed for G6)

- If the shared expert's CBs shrink by ~111 KB, Q2 and combine outputs can both live in L1 (Q2 −20 µs more).
- DQ1 output in L1 would save ~24 µs but overlaps the routed expert's CBs by ~9 KB (its narrower config costs +42 µs).
- post_combine_reduce at 101 µs has not been re-profiled since its source moved to L1; DQ2's own share is unmeasured.
