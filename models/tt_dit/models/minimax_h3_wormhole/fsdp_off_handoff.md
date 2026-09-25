# Handoff: MiniMax-H3 t2va on the Wormhole 4x8 with DiT FSDP off (15 s / 16:9 / 768P)

Written 2026-09-25 at the end of the session that produced branch `jameslee/minimax_h3_wh_adaln_tables`
(six commits on top of `jameslee/exp_ring_sdpa_wh` @ `cd1a9268d3d`). Raw logs, probe scripts, frame dumps and
JSON records for every run below are on the run host under
`~/h3_wormhole_results/fsdp_off_feasibility_2026-09-24/` (`RESULTS.md` there is the running log; `e2e_50steps/`
holds the 50-step records). The README of this directory, Part 4, has the same material in narrative form.

## 1. Objective

Run MiniMax-H3 text-to-video-and-audio (`t2va`) on the Wormhole Galaxy (4x8, 32 chips, 12 GB DRAM each) for a
15 s, 16:9, 768P (1344x768, 362 frames) video **with DiT FSDP turned off** (`MINIMAX_H3_DIT_FSDP=0`), at 50
scheduler steps, and know what it costs against the shipped preset (TP=4 / SP=8, DiT FSDP on).

Why it was not simply a flag flip: since `816841ddc93` (#57097, rebased in on 2026-09-24) each transformer
block keeps its adaLN projection weight (2688 x 96768 bf16 = 520 MB, TP-fractured) resident and re-projects
the per-step modulation every step. FSDP shards those along with the matmul weights, so the FSDP-on numbers
were unaffected, but unsharded they add 541 MiB/bank at TP=4 to the 796 MiB/bank of matmul weights, and a
DRAM bank is 1021 MiB. The README Part 1 line "FSDP off: 5 s fits" predates that commit and is stale.

Ground facts measured this session with a per-op DRAM high-water probe (`ttnn.get_memory_view` after every
op via `ttnn.register_post_operation_hook`; the hook is silently bypassed unless
`TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'`):

| quantity, per DRAM bank (x12 per chip) | TP4/SP8 | TP8/SP4 |
|---|---|---|
| DiT matmul weights, unsharded, bf16 | 796 MiB | 398 MiB |
| adaLN projection weights, unsharded | 541 MiB | 258 MiB |
| denoise activation peak above the resident set (K/V ring-gather pool, AGMM scratch, five full-hidden activations) | +258 MiB | +263 to +290 MiB |
| audio decoder (fp32, replicated, resident through the denoise until this session) | 42 MiB | 42 MiB |
| residue of the init warmup (a 5 s request) in shape-keyed CCL pools and per-rung state | -- | 99 MiB (74 recoverable) |

TP cannot exceed 8: 56 heads divide by 1, 2, 4, 7, 8, 14, 28, 56, and only 8 fits a 32-chip mesh axis. So
TP8/SP4 is the only unsharded candidate, and everything below is about finding ~100-300 MiB/bank there.

## 2. Proposed ways to turn FSDP off

Ten proposals; seven were attempted. "Ran" means the 15 s / 16:9 generation completed. Times are from the
harness `test_parallel_sweep_minimax_h3.py` at 50 steps (49 forwards), same host, seed 0, fox prompt, init
warmup on, `coresident=False` (stated in each record), RUN_VBENCH=0. **Steady e2e** = the timed request
minus the first denoise step's compile excess (first step minus steady step, ~1.7 s); it *includes* the
per-request component reloads, because with `coresident=False` they recur every request. **Block time** was
*not* profiled per configuration this session (see §3): the column is steady ms/forward divided by 50
blocks, an upper bound that also contains the refiner, embeddings, `norm_out` and output collectives (the
09-17 Tracy profile of the shipped configuration put the block itself at 242-247 ms of a 246 ms/50 forward).

| # | proposal | attempted | ran 15 s? | steady e2e (s) | steady ms/fwd | ~block ms (derived) | CLIP mean / min | text enc load | DiT load | VAE dec load | audio dec load | other per-request |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TP4/SP8, FSDP off, adaLN resident (the plain flag flip) | analysed (probes A/C + arithmetic) | **no** -- DiT alone is 796 + 541 = 1337 MiB/bank, does not load | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 2 | TP8/SP4, FSDP off, adaLN resident, nothing else changed | yes (run D) | **no** -- OOM in block 0 of forward 0: resident 726 MiB/bank, largest free block 17 MB | -- | -- | -- | -- | 3 | 7 | -- | 0 | -- |
| 3 | TP8/SP4, FSDP off, **`adaln_tables`**: projection weights stay on host, one request-wide modulation table (default when the DiT is unsharded on Wormhole) | yes | **yes** (peak 728 MiB/bank, ~290 headroom) | **638.3** | **12130** | ~243 | 37.25 / 35.66 | 3 | 4 | 4 | 0 | adaLN table build 2.6 |
| 4 | TP4/SP8, FSDP off, `adaln_tables` | yes (run G) | **no** -- DiT 805 MiB/bank loads, OOM on the first 392 MB K/V gather buffer | -- | -- | -- | -- | 3 | ~7 | -- | 0 | -- |
| 5 | TP8/SP4, FSDP off, adaLN resident + **audio decoder evicted** for the denoise | yes (probes H, I) | **only in a fresh process** (peak 945 MiB/bank); OOMs once the init warmup has run (resident 802 MiB/bank) | -- | (12154 one steady step) | -- | -- | 3 | 7 | 3 | 1 (reload) | -- |
| 6 | #5 + **`CCLManager.release_persistent_buffers()`** before each DiT reload (drops the warmup's shape-keyed pools) | yes (probe J + 50 steps) | **yes**, fragile: peak 985 MiB/bank, 18 MiB contiguous left; any second sequence shape in the process would consume it | 642.0 | 12176 | ~244 | 37.20 / 35.40 | 3 | 7 | 3 | 1 (reload) | -- |
| 7 | TP8/SP4, FSDP off, **`adaln_fsdp`**: shard only the adaLN projections across SP, rest unsharded (FSDP's own gather path, ~49 MB per block per step) | yes (probe K + 50 steps) | **yes** (peak 802 MiB/bank, 177 MiB contiguous left) | 641.9 | 12192 | ~244 | 37.45 / 35.97 | 3 | 6 | 3 | 0 | -- |
| 8 | Store the adaLN projections in bfp8_b (258 -> 129 MiB/bank at TP=8) | no | -- | | | | | | | | | changes numerics; tt_dit's linear has no fidelity entry for quantized weights |
| 9 | Trim the denoise peak: free intermediates earlier (five full-hidden `[27296, 5376]` activations = 117 MiB/bank live at once), release the remaining ~26 MiB/bank of per-rung warmup state, or warm up on the target shape only | no | -- | | | | | | | | | only matters for #6 |
| 10 | Stream the matmul weights from host per layer, or go above TP=8 | no | infeasible: 4.7 GB/device/forward over PCIe (seconds per 12 s forward); TP>8 impossible with 56 heads | | | | | | | | | |

Reference rows, same harness and day:

| config | steady e2e (s) | steady ms/fwd | ~block ms | CLIP mean / min | text enc | DiT | VAE dec | audio dec |
|---|---|---|---|---|---|---|---|---|
| **shipped**: TP4/SP8, FSDP on, adaLN resident | **633.8** | **12055** | ~241 | 37.65 / 35.99 | 3 | 4 | 3 | 0 |
| TP8/SP4, FSDP on, adaLN resident | 644.4 | 12277 | ~246 | 37.29 / 35.68 | 3 | 4 | 3 | 0 |

Reading the two tables together:

* **FSDP itself is cheap.** At fixed TP8/SP4, turning it off saves 0.7% (#7), 0.8% (#6) or 1.2% (#3) per
  step. The per-block Tracy measurement of 5.8% FSDP overhead is mostly hidden behind compute.
* **TP=8 is the real cost.** At fixed FSDP-on, TP8/SP4 is 1.8% per step slower than the shipped TP4/SP8, with
  TP=8 matmul blockings still untuned (`agmm_config.py` / `matmul.py` tables were swept for TP=4). Every
  FSDP-off configuration inherits this; the best one (#3) lands 0.7% behind the shipped preset per request.
* **The three that run differ by <4 s per request.** #3 is fastest per step but pays a 2.6 s table build;
  #6 and #7 pay 2-3 s more DiT reload (6-8 GB/device instead of 4.9). #7 has the most memory headroom of the
  three and no per-request build; #6 should not ship.
* **Output.** At 3 steps every TP8/SP4 variant (FSDP on, #3, #6, #7) is bit-identical in frames and audio.
  Over 49 forwards the pipeline is **not run-to-run deterministic**: the FSDP-on TP8/SP4 run repeated
  unchanged moved from PCC 0.9997 to 0.9962 against the same #3/#6 runs and CLIP 37.288 -> 37.293, so the
  50-step PCCs between variants (0.996-0.9997 for #3/#6, 0.963 for #7) are noise whose size depends on when
  the first divergence lands, not a per-path difference. Judge numerics at 3 steps or with repeats. TP4 vs
  TP8 differ at PCC 0.91-0.92 (bf16 reduction order), as in the September TP/SP sweep.
* **Load-time detail.** All loads come from `TT_DIT_CACHE_DIR` (`~/tt_dit_cache`): text encoder 3 s every
  request; DiT 4 s at ~2 GB/device (FSDP on) or ~5 GB (#3), 6-7 s at 6.4-8 GB (#6, #7); VAE decoder 3-4 s;
  audio decoder < 1 s (logged as 0 s; 1 s where it is a reload after eviction). These sit between the stage
  timers and are the "gap" of a few seconds between the run total and the sum of its stages.

What the branch adds, all gated on Wormhole and/or `coresident=False`, so Blackhole and the traced paths
are untouched: `MiniMaxH3Pipeline(adaln_tables=, adaln_fsdp=)` with `MINIMAX_H3_ADALN_TABLES` /
`MINIMAX_H3_ADALN_FSDP`; `ColParallelLinear(on_host=)` and `Parameter.staged_on_device()`;
`MiniMaxH3Transformer3DModel.prepare_request_modulation` + per-forward `adaln_step`; audio decoder as a
coresident exclusion of the DiT with reload; `CCLManager.release_persistent_buffers()`; `Snake`/`SnakeBeta`
drop their first-use shard cache on `deallocate_weights` (it aliases the parameter without channel-TP and
was handed to the op after a reload); the sweep harness records `dit_fsdp` / `adaln_tables` / `adaln_fsdp` /
`coresident`, writes the mp4/wav and the CLIP score. Two hangs on the way, both documented in the README:
a 49-input row-major `ttnn.concat` stalls the mesh (the table build now concatenates on host), and after a
dispatch-timeout reset the fabric mapper can refuse the 4x8 once (`tt-smi -r` again clears it).

## 3. What the next agent should do: profile the transformer block

Every number above is end to end. Nothing in this session profiled the block per configuration, so the
"block ms" column is derived and the 1.8% TP=8 penalty and the 0.7-1.2% FSDP saving are not yet attributed
to ops. The tools to do that are in **PR #57941, "MiniMax-H3: Wormhole Galaxy optimizations"**
(branch `minimax_h3_wh_optimizations`, `models/tt_dit/tests/models/minimax_h3/tools/`), and the README
Part 2 of this directory shows them in use on the shipped configuration:

1. **Take a Tracy per-op profile of one block** with the safe runner in profile mode:
   ```bash
   scripts/run_safe_pytest.sh --profile \
     "'models/tt_dit/tests/models/minimax_h3/test_performance_minimax_h3.py::test_minimax_h3_transformer_block_perf[wormhole_b0-sp_sim1-15s_768p-4x8sp1tp0nl4_ring_is_fsdp1]'" \
     -o timeout=1500
   ```
   It prints `SAFE_PYTEST: PROFILER CSV: <path>` (under `generated/profiler/reports/`). The existing
   parametrizations are TP4/SP8 only (`4x8sp1tp0nl4_ring_is_fsdp0` / `_is_fsdp1`, in
   `tests/.../minimax_h3/common.py`); **add TP8/SP4 variants** (`tp_axis=1, sp_axis=0`) and plumb
   `adaln_tables` / `adaln_fsdp` into the block fixture so #3, #6 and #7 can each be profiled against the
   FSDP-on block at the same TP. Two known snags: the `is_fsdp0` profile has aborted inside Tracy before
   (ff1.md, "per-op durations are not additive under FSDP"), and the block test needs the pinned diffusers
   fork that not every host has (the 09-17 sweep host did not, and ran everything end to end instead).
2. **Compare profiles** with `tools/block_profile_stats.py compare` (per-op mean / std / CoV across the
   32 devices and across runs; the README Part 2 tables came from it) and `tools/project_block_perf.py
   fsdp1=<csv> fsdp0=<csv>` (projects a block delta onto the 15 s forward). Use
   `tools/op_time_from_profiler_csv.py` for a single op's time.
3. **Mind the FSDP accounting.** Under FSDP the weight `AllGatherAsync` runs on the CCL sub-device
   concurrently with the matmuls, so per-op durations do not add up; `tools/block_device_busy.py` gives the
   device-busy *union*, which is the right quantity for any change that shifts timing under FSDP or under
   `adaln_fsdp` (#7 has exactly one such gather per block).
4. **Put the ops against the roofline** with `tools/transformer_roofline.py` (whole-block mode reads the
   profile CSV and gives every op group a bound; the small-op groups are folded and figured separately).
   The interesting question for this handoff is where TP=8's extra 1.8% goes: the three AGMMs and ff2 at
   TP=8 shapes (K=5376 gives 21 tiles per device, a ring-safe K_block was added in `get_agmm_config` on
   2026-09-17 and the blockings have not been swept since), or the K/V ring at SP=4 (half the ring, twice
   the rows per device). `tools/agmm_unit_sweep.py`, `tools/transformer_op_mesh_bench.py` and
   `utils/sweep_mm_block_sizes.py` are the sweep tools for the matmul side; `sdpa.md` has the SDPA method.
   Any new M-keyed table must key on the pipeline's logged `rows/device` (27296 at TP8/SP4, 13664 at TP4/SP8),
   not the harness constant: the block perf test's `_packed_sizes` once produced 13632 where the pipeline ran
   13664 and every swept entry missed (the write-up, `MiniMaxH3_rows_per_device_mismatch.md`, was retired after
   the fix in `c0af23ba607`; README Part 2 records the lengths the pipeline actually runs).
5. **Then decide the Wormhole FSDP-off default** between #3 (`adaln_tables`, current default) and #7
   (`adaln_fsdp`), and whether TP=8 blocking work closes the 1.8% to the shipped preset. If bit-reproducible
   50-step output matters to anyone, the run-to-run nondeterminism noted above needs its source found first;
   the async collectives are the natural suspects.

Housekeeping the next agent inherits: six DiT caches from these experiments under `~/tt_dit_cache/minimax-h3/`
(`transformer_resident_adaln{,_fsdp,_adalnfsdp}/TP{4_0_SP8_1,8_1_SP4_0}_mesh4x8_bf16`, ~63 GB each) plus a
TP8 text-encoder cache (47 GB); VBench was off for every run here; the README Part 1 "FSDP off" rows point to
Part 4 but were not rewritten.
