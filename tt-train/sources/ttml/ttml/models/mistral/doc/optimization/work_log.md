# Optimisation loop work log

Model `mistralai/Mistral-Small-24B-Base-2501` @ `b0a2e4e`, LoRA r16/a32 on all block linears, seq 1024, batch 1, AdamW. Branch `agobeljicTT/mistral-bringup`. Evidence under `doc/optimization/evidence/` (`O`).

- Single-chip Tracy at 1/2/4/8 layers for attribution and the fit; full 40-layer MFU and loss runs with TP=4 over the box (user-authorised, `plan.md`).
- Commands from `tt-train/` with `TT_METAL_HOME=TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal`; TP runs also need `TT_MESH_GRAPH_DESC_PATH=tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto`.

## 2026-10-29

### TP path and baselines

- TP path: `train.py --tp N` (1x4 mesh, axes `("dp","tp")`, `use_tp=True`, vocab-parallel embedding and LM head, `vocab_parallel_cross_entropy_loss`, loss via `get_loss_over_devices`, MFU denominator x N, LoRA-init dump gathers the sharded adapters); `frozen_embedding.embed_tokens` keeps the module's own forward for a vocab-parallel table. Mesh descriptor `configs/mgd/bh_p300x2_1_4_ring_ring.textproto` (2 eth links per edge) copied from `umales/qwen38`.
- Baseline, TP=4, random init, 8 steps, 2 warm-up (`O/full_model_tp4.sh baseline` -> `O/full_tp4_baseline_random.log`, tt-smi `O/full_tp4_baseline_random_tt-smi.txt`): **598.7 ms/step (best 594.7), 161.0 TFLOP/s, MFU 27.10 %** of 594 TFLOP/s.
- Baseline, real weights (`O/full_model_tp4_real.sh baseline`): first attempt failed in the LoRA-init dump (`to_numpy` on a mesh-sharded tensor), fixed with `Sharding.gather`; the rerun had picked up the uncommitted LoRA-apply port (stashed since) and was killed; repeated on the committed code.
- Baseline, TP=4, real weights, 10 steps (`O/full_tp4_baseline_real.log`, `.json`, data log, gathered LoRA init `.npz`): **641.9 ms/step (best 639.6), MFU 25.27 %**, loss 0.4453, 0.3281, 0.2520, 0.2070, 0.1436, 0.2734, 0.3477, 0.2676, 0.3066, 0.4551. 43 ms slower than the random-init run minutes earlier: warm chips (composite README).

### Iteration 1: `LoraApply` port

- `git stash` round-trips to measure the baseline TP profile on the committed code. Chain `O/iter_chain.sh iter1_loraapply` -> `O/iter1_loraapply_chain.log`: harness 11 passed, Tracy 1/2/4/8, 8-layer MFU 438.7 ms at 1101 MHz (perf_analysis MFU history: 1104; docs disagree), TP MFU 816.9 ms -> hot-chip artefact, renamed `*_hot.*`.
- Analysis `O/iter1_loraapply_tracy_analysis.{txt,json}`, diff vs the composite baseline with `diff_tracy.py`, `O/iter1_loraapply_clock_analysis.txt`, op sequence `O/iter1_loraapply_op_sequence_1layer.txt`.
- TP profiles `O/baseline_tracy_tp4_4layers` (73.4 ms) and `O/iter1_loraapply_tracy_tp4_4layers_hot` (140.1 ms, chip 3 at 1.68x), per-chip tool `tp_analysis.py`.
- Cool rerun `O/tp_measure.sh iter1_loraapply` with `COOL=600` (`O/iter1_loraapply_tp_measure.log`): TP MFU 574.3 ms / 28.25 %, 4-layer TP profile 70.9 ms, chips equal.
- Tools added: `diff_tracy.py`, `tp_analysis.py`; `iter_chain.sh` now samples tt-smi and takes `COOL`.

### Iteration 2: transposed frozen weights

- Microbenchmarks `O/iter2_matmul_bench{,2,3,4}.log` (`matmul_bench*.py`).
- First chain `O/iter2_transposed_*`: dX routes without `core_grid` (gate_up dX 274 -> 358 ms); TP failed on the column-parallel LM head not honouring the flag; the `tp_measure` TP numbers of that run are iteration-1 code.
- Fixes: `core_grid` in the backward, allow-list of shapes, parallel linears honour the flag, local-shard shape from the mesh placement.
- Second chain `O/iter2b_rerun.log` / `O/iter2b_transposed_*`: Tracy 1/2/4/8 -> 1398.7 ms projection, 8-layer MFU 380.0 ms after 600 s cool-down; harness had one stale test expectation; TP transposed 0 linears because the mesh tensor reports its global shape.
- Third pass `O/iter2c.log`: harness 13 passed, TP smoke 5 linears, TP MFU 564.5 ms after 420 s cool-down, TP profile per chip equal. Analysis `O/iter2b_transposed_tracy_analysis.{txt,json}`, `O/iter2b_transposed_clock_analysis.txt`.

### Iteration 3: fused LoRA (`fused_lora.py`)

- Building-block timings in the session log: fill_cache 0.84 ms per gate_up tail, concat 0.06, slices 0.06-0.01, K = 161 vs 162 tiles.
- Isolation test (`/tmp/.../scratchpad/fused_isolate.py`, later `O/iter3_fused_isolate.py`) found the tail written without the LoRA scale (rel err 0.5 on the tail, 0.5 on dA), fixed; `O/iter3_fused_quick.log` fused block test passed.
- Chain `O/iter3.sh` -> `O/iter3.log`: harness 14 passed, TP smoke, Tracy 1/2/4/8, 8-layer MFU 375.3 ms after 600 s, TP MFU 556.5 ms after 600 s, repeat 557.9, TP 4-layer profile chips equal.
- Two false starts of the chain (a `pkill` that matched its own command line) are in `O/iter3.log`'s predecessors; the stale MGD env var for the single-chip steps is unset in `iter3.sh`.

### Final validation and exit

- `O/final_validation.sh` -> `O/final_validation.log`: 8-layer real-weight run with `--host-loss` (`O/final_train_8layers_hostloss.log`, `.json`; the host read-back makes it 796 ms/step, not a perf number); `compare_loss.py --tol 0.10` vs the composite phase's torch references -> max 0.026, PASS (`O/final_loss_comparison.txt`).
- `--mem-probe` at 8 layers (`O/final_mem_probe_8layers.log`): 13.79 GiB after forward, 11.61 weights + state.
- TP=4 40-layer real-weight run after 300 s idle (`O/final_full_tp4_real.log`, `.json`): 574.8 ms/step, loss 0.4453 -> 0.4531 as the baseline's windows. Loop closed in `perf_analysis.md` ("Loop exit") and `README.md`.

### Review rounds 1 and 2

- Round 1 (`review_round1.md`): required a TP=4 gradient-level check of the fused/transposed paths, the exit restated at the measured clock with the falling-clock anomaly classified, and arithmetic fixes in the gap decomposition; also taken: the 16-layer Tracy point (`O/iter3_fused_tracy_16layers*`), the iteration-3 op sequence (`O/iter3_fused_op_sequence_1layer.txt`), the fused-LoRA isolation script and its after-fix log.
- TP gradient check (`O/review_followup.sh`, `O/tp_grad_check2.sh`, then by hand): `tp_gradient_check.py --tp 1` (`O/tp_grad_check_single.{npz,log}`), `--tp 4 --no-transpose` (`O/tp_grad_check_tp4_plain.*`), `--tp 4` (`O/tp_grad_check_tp4_fused.*`): PASS on every adapter after the rank-interleaved assignment of the fused qkv / gate_up rows was added.
- First attempts without the interleave (uncorrelated gradients on every TP path, equal statistics) and the descent checks `--sgd-lr 0.05` (`O/tp_grad_check_*_descent.log`: loss 12.8125 -> 12.6250 on all three paths, prediction 0.23) diagnosed it; the invalid first-attempt artefacts were removed.
- 16-layer Tracy of the final path (`O/iter3_fused_tracy_16layers.log`, compact CSV gitignored at 0.9 MB): 563.6 ms device, 694.4 ms wall at 1100 MHz, 24.8 GiB after forward; 2/4/8/16 fit `O/iter3_fused_tracy_analysis_2_4_8_16.txt` -> 1345.8 ms.
- Round 2 (`review_round2.md`): clean-pass. Applied: the descent check labelled as taken on the un-interleaved model (its `.npz` removed), big-matmul roofline corrected to 555 ms, the clock-trend ledger admits the thermal / run-order confound (same-code 16-layer run at ~1096 vs 1041 MHz), nits (harness count, file names, iteration-1 MFU clock 1104, RoPE roofline 3.3, the `--no-transpose` label, the `param:` control comment). Loop closed.

### Loop resumed: iteration 4

- User: "why stop? these are good findings, continue; HiFi4 is a must"; later: "not interested in 8 layer single chip, full model MFU is the most important, the subset layers are only used to quickly deduce what is a top priority".
- New metal op `write_tile_block` (`metal/ops/write_tile_block/`, registered in `CMakeLists.txt`, `metal/operations.hpp`, `nanobind/nb_ops.cpp`; `cmake . && ninja _ttml` in `build_Release`; the first build failed because `nb_ops.cpp` includes op headers individually); device test `O/iter4_write_tile_block_bench.{py,log}` (bit-exact, in place, 0.85 -> 0.06 ms gate_up tail).
- 1D program-config sweep `O/iter4_matmul_sweep1d_gateup_fwd.log` (`matmul_sweep1d.py`), stopped at 1564 configs to free the device (perf_analysis: 1624 distinct configs; docs disagree), best 7.35 vs 7.92 ms applied via `FORWARD_PROGRAM_CONFIG`. A 2D-mcast sweep `O/iter4_matmul_sweep_gateup_fwd.log` found every config over L1; a first 1D attempt without the `sub_h x sub_w <= 4` constraint died without a summary.
- Harness `O/iter4_harness.log` 14 passed. Two aborted chain attempts (`O/iter4_aborted_{first,second}_attempt.log`: the first stopped at the user's request to end the session, the second to drop the single-chip MFU and 16-layer steps; a `pkill` whose heredoc contained the pytest command line killed its own shell once more).
- Final chain `O/iter4.sh` -> `O/iter4.log`: Tracy 1/2/4/8 (`O/iter4_tracy_*`, `O/iter4_tracy_analysis.{txt,json}`, diff `O/iter4_diff_vs_iter3.txt`, clock `O/iter4_clock_analysis.txt`: 1007 MHz), TP=4 MFU after 600 s (`O/iter4_tp_mfu_tp4.log`, tt-smi) 558.5 ms / 29.05 %, 4-layer TP profile `O/iter4_tp_tracy_tp4_4layers` (`O/iter4_tp_analysis.txt`; iteration 3's per-chip table saved as `O/iter3_fused_tp_analysis.txt`).
- Doc fixes in the same commit: table rows re-sorted by totals, unescaped `|` inside code spans of table rows (perf_analysis MFU history, plan kernel inventory), two short rows padded.

### Iteration 5

- Patch drafted from the iteration-4 TP profile ranking (`O/iter4_tp_ranked_chip0.txt`, ad-hoc per-op table of chips 0 and 2).
- Chain `O/iter5.sh` -> `O/iter5.log`: harness `O/iter5_harness.log` 14 passed; gradient check `O/iter5_tp_grad_check_tp4_batched.{log,npz}` vs the single-chip reference, PASS; `tp_measure.sh iter5_tp` after 600 s: `O/iter5_tp_mfu_tp4.log` 556.9 ms, tt-smi, 4-layer TP profile `O/iter5_tp_tracy_tp4_4layers` -> `O/iter5_tp_analysis.txt`.
- The chain's trailing TP-shape sweeps were stopped after the first (`O/iter5_matmul_sweep1d_gateup_fwd_tp.log`: ttnn's choice 1.67 ms vs best 1.66 at the TP-local gate_up shape, no win) to free the device for iteration 6.

### Iteration 5b

- Patch `lora.py` / `fused_lora.py`; measured in the shared chain `O/iter6.sh` -> `O/iter6.log`: harness `O/iter6_harness.log` 15 passed with 5b and 6 on; gradient check `O/iter6_tp_grad_check_tp4.{log,npz}` PASS, 4 partials per layer.
- `tp_measure.sh iter5b_tp` with `EXTRA_ARGS=--no-allcore-rmsnorm` (new hook in `tp_measure.sh`) after 600 s: `O/iter5b_tp_mfu_tp4.log` 563.2 ms, tt-smi, profile `O/iter5b_tp_tracy_tp4_4layers` -> `O/iter5b_tp_analysis.txt`, per-op `O/iter5b_tp_ranked_ops.txt` from the new `tp_rank_ops.py`; iteration 5's per-op table `O/iter5_tp_ranked_ops.txt`.
- The iteration-5 commit was amended twice to take the iteration-6 wiring back out of `train.py` / `tp_gradient_check.py` (applied by the queued iteration-6 build script mid-commit); the second amend went through the index only (`git hash-object` + `update-index`, `--no-verify`) so the running chain's files were not touched.

### Iteration 6

- Op exported from `umales/qwen38` with `git show` into the scratchpad, copied to `metal/ops/rmsnorm_bw_allcore/` with namespace / primitive / kernel-path renames (sed), object files compiled first without relinking (the iteration-5 chain was on the device), registered in `CMakeLists.txt`, `metal/operations.hpp`, `nanobind/nb_ops.cpp`.
- Build + test queued behind the iteration-5 chain (`O/iter6_build_and_test.sh` -> `O/iter6_build_and_test.log`: `cmake . && ninja _ttml`, first test run failed on the backward's return arity, fixed, `O/iter6_rmsnorm_test.log` 3 passed).
- Chain `O/iter6.sh` -> `O/iter6.log`: harness 15 passed, gradient check PASS, single-chip 8-layer profile `O/iter6_tracy_8layers` -> `O/iter6_tracy_8layers_ranked_ops.txt` (reference `O/iter4_tracy_8layers_ranked_ops.txt`), `tp_measure.sh iter6_tp` after 600 s: `O/iter6_tp_mfu_tp4.log` 547.5 ms, tt-smi, profile `O/iter6_tp_tracy_tp4_4layers` -> `O/iter6_tp_analysis.txt`, `O/iter6_tp_ranked_ops.txt`.

### Iteration 7 (rejected) and the TP-shape sweeps

- `TRANSPOSED_SHAPES` + `(5120, 1536)`, `(1024, 5120)`; chain `O/iter7.sh` -> `O/iter7.log`: gradient check `O/iter7_tp_grad_check_tp4.{log,npz}` PASS, `tp_measure.sh iter7_tp` after 600 s: `O/iter7_tp_mfu_tp4.log` 551.3 ms, profile `O/iter7_tp_tracy_tp4_4layers` -> `O/iter7_tp_analysis.txt`, `O/iter7_tp_ranked_ops.txt`. Allow-list change reverted (`git checkout`).
- Narrow TP-shape sweeps at the end of `O/iter7.sh` (`matmul_sweep1d.py --ref-grid 11x10 --in0-block-w 2,3,4 --subs 4x1,2x2 --grids 11x10,10x10`): `O/iter7_matmul_sweep1d_*.log`; ttnn's choice within 2 % of the best on every TP-local forward shape, qkv worse under every explicit config, dX shapes without a fitting config. No change.

### Iteration 8

- `FrozenVocabParallelEmbedding` in `frozen_embedding.py`; chain `O/iter8.sh` -> `O/iter8.log`: gradient check `O/iter8_tp_grad_check_tp4.{log,npz}` PASS, `tp_measure.sh iter8_tp` after 600 s: `O/iter8_tp_mfu_tp4.log` 545.9 ms, tt-smi, profile `O/iter8_tp_tracy_tp4_4layers` -> `O/iter8_tp_analysis.txt`, `O/iter8_tp_ranked_ops.txt`.
- The single-chip harness does not exercise the vocab-parallel path (no mesh); the TP gradient check (loss identical to single chip) is its correctness evidence.

### Stage reviews 3-5

- Round 3 (`review_round3.md`, more-work-needed, 7 findings). Follow-up: `frozen_embedding.py` / `rmsnorm_allcore.py` graph-free on no-grad inputs (finding 1); stacking microbenchmark `O/review3_stack_bench.{py,log}` (finding 2: no gain from tile-aligned stacking, cost recorded).
- Chain `O/review3_followup.sh` -> `O/review3_followup.log`: harness `O/review3_harness.log` 15 passed, gradient check `O/review3_tp_grad_check_tp4.{log,npz}` PASS, 8-layer profile `O/review3_tracy_8layers` -> `O/review3_tracy_8layers_ranked_ops.txt`, `tp_measure.sh review3_tp` after 600 s (`O/review3_tp_mfu_tp4.log` 545.7 ms, `O/review3_tp_analysis.txt`, `O/review3_tp_ranked_ops.txt`: 17/17 collectives, 8+8 norm backward).
- Finding 3: 8-layer real-weight host-loss run `O/review3_train_8layers_hostloss.log` + `O/review3_loss_comparison.txt` (max 0.027, PASS), TP=4 real-weight 10 steps `O/review3_full_tp4_real.{log,json}`. Docs reworded per findings 4-7; the traceback file `iter6_diff_vs_iter4_8layers.txt` deleted.
- Round 4 (`review_round4.md`, more-work-needed on five P3 doc / test items): fixed in `ddf3a77be15` (README count and wording, commit rows, the frozen-embedding assertion, the no-grad norm harness case, `O/review4_tests.log` 5 passed, sweep numbers).
- Round 5 (`review_round5.md`): clean-pass; its three P4 wording notes taken in the closing commit.

## 2026-10-30

### Iteration 9

- New op `metal/ops/add_rmsnorm_fw_allcore/`, `residual_grad` input of `rmsnorm_bw_allcore`, `models/mistral/fused_residual_norm.py`, `Mistral.forward` dispatch, `train.py --no-fused-residual-norm`, `tp_gradient_check.py` wiring; build `O/iter9_ninja.log`.
- Kernel tests `O/iter9_kernel_tests.log` (8 passed: fused op at 32 and 64 tile-rows, backward at 64 tile-rows, 2-layer model vs the block path `O/iter9_model_test.log`). The 64-tile-row cases came from the CB wrap hazard found while bringing up iteration 10's op (`O/iter10_skinny_check.log`, first run PCC 0.01 at K=5120): the ported backward had the same latent overrun on a core's second work item; all three kernel sets now move full blocks.
- Chain `O/iter9.sh` -> `O/iter9.log` (the first two launches were stopped for the kernel fixes): harness `O/iter9_harness.log` 20 passed, gradient check `O/iter9_tp_grad_check_tp4.{log,npz}` PASS, single-chip 8-layer profile `O/iter9_tracy_8layers.log` -> `O/iter9_tracy_8layers_ranked_ops.txt`, `tp_measure.sh iter9_tp` after 600 s: `O/iter9_tp_mfu_tp4.log` **537.4 ms / 30.19 %**, tt-smi, profile `O/iter9_tp_tracy_tp4_4layers` -> `O/iter9_tp_analysis.txt`, `O/iter9_tp_ranked_ops.txt`.

### Iteration 10

- New op `metal/ops/skinny_matmul/`, routing in `modules/lora.py` (`skinny_matmul`, `SKINNY_MIN_K`) and `modules/fused_lora.py` (wiring patch `O/iter10_wire.py`); standalone check `O/iter10_skinny_check.{py,log}` (first run PCC 0.01 at K=5120: the CB wrap hazard, fixed; rerun PCC 0.999999 everywhere).
- Chain `O/iter10.sh` -> `O/iter10.log`: harness `O/iter10_harness.log` 20 passed, gradient check `O/iter10_tp_grad_check_tp4.{log,npz}` PASS, `tp_measure.sh iter10_tp` after 600 s: `O/iter10_tp_mfu_tp4.log` **536.1 ms / 30.27 %**, tt-smi, profile `O/iter10_tp_tracy_tp4_4layers` -> `O/iter10_tp_analysis.txt`, `O/iter10_tp_ranked_ops.txt`.
- The iteration-9 commit was made while this chain ran (its files only; the shared registration files were committed with the skinny lines stripped and restored afterwards).

### Iteration 11: SDPA work distribution

- Baseline of the old kernels at six shapes: `O/iter11_baseline.sh` -> `O/iter11_baseline.log` (outputs saved to the scratchpad `.npz`); build + standalone comparison + attention tests: `O/iter11_build_check.sh` -> `O/iter11_build_check.log` (`SKIP_BUILD=1` reruns without the build; kernels compile at runtime).
- First comparison: forward and dQ bit-identical, dK / dV garbage under `HEAD_SPLIT`; the reduce alone checked against numpy through a new binding `ttml.ops.metal.sdpa_kv_bw_reduce` (PCC 0.999997) -> the fault was the KV kernel's conditional packer reconfig (fixed, see `../perf_analysis.md`, "Iteration 11").
- Chain `O/iter11.sh` -> `O/iter11.log`: harness 35 passed, TP=4 gradient check PASS, cooled TP=4 MFU 524.7 ms / 30.92 % (`O/iter11_tp_mfu_tp4.*`), 4-layer TP profile `O/iter11_tp_tracy_tp4_4layers` (`O/iter11_tp_analysis.txt`, `O/iter11_tp_ranked_ops.txt`; chips 1.149-1.273 GHz, matmul rows equal on all four).
- Code: `metal/common/program_utils.hpp` (`lpt_schedule`, `work_list_runtime_args`), `metal/ops/sdpa_fw/`, `metal/ops/sdpa_bw/` (work-list modes, head split, `sdpa_bw_kv_reduce_*` op and kernels), `nanobind/nb_ops.cpp`, `tests/python/test_mistral.py` (`test_sdpa_causal_gqa_forward_and_gradients`).

### Iteration 13: transpose_split

- Op `metal/ops/transpose_split/` (device op, factory, reader / writer / compute kernels), registered in `CMakeLists.txt`, `metal/operations.hpp`, `nanobind/nb_ops.cpp` (`ttml.ops.metal.transpose_split(input, rows, tail_rows, tail_scale)`); used by `fused_lora.py` backward.
- Build + standalone check + LoRA tests: `O/iter13_build_check.sh` -> `O/iter13_build_check.log` (the chain's grep hid the check lines; they are in `O/iter13_transpose_split_check.log`).
- Chain `O/iter13.sh` -> `O/iter13.log`: harness 40 passed, TP=4 gradient check PASS, cooled TP=4 MFU 516.4 ms / 31.42 % (`O/iter13_tp_mfu_tp4.*`), 4-layer TP profile `O/iter13_tp_tracy_tp4_4layers` (`O/iter13_tp_analysis.txt`, `O/iter13_tp_ranked_ops.txt`).

### Iteration 14: vp_cross_entropy_bw

- Op `metal/ops/vp_cross_entropy_bw/` (mesh-aware device op, factory, reader / writer / compute kernels), registered in `CMakeLists.txt`, `metal/operations.hpp`, `nanobind/nb_ops.cpp`; `ops/distributed/losses.cpp` backward calls it (the upstream gradient repeated into a tile).
- Build + standalone check + loss tests: `O/iter14_build_check.sh` -> `O/iter14_build_check.log` (first run: 99.9 % zeros from the padded gradient tile, found by binning the error; second run exact; the test's PCC bar on the near-constant target entries replaced by a relative-error bar).
- Chain `O/iter14.sh` -> `O/iter14.log`: harness 42 passed, TP=4 gradient check PASS, cooled TP=4 MFU 512.5 ms / 31.66 %, 4-layer TP profile, then the loss match: 8-layer host loss (`O/iter14_train_8layers_hostloss.log`, `O/iter14_loss_comparison.txt`, PASS 0.0294) and TP=4 real weights (`O/iter14_full_tp4_real.log`).

### Iteration 15: SDPA forward data flow (kernel boost checklist)

- Roofline: the forward moves 67 MB of K/V per call at the TP shard (8 heads x 528 Q-row x K-tile pairs x 8 tiles x 2 KB) because every query row re-reads its K/V prefix from DRAM; at ~400 GB/s that is 170 us, measured 229 us (cycle-time, iteration 11-14 profiles) = 290 GB/s; the FLOP roofline is ~15 us (2.1 GFLOP causal at 148 TFLOP/s), so the kernel is 15x off its FLOP roofline and bound by bytes it does not need to move.
- Where the bytes go: K and V of one (batch, kv head) stream are identical for the 4 query heads x 32 rows = 128 rows of the stream; 55 cores each read their rows' prefixes independently (~34x redundancy per stream).
- Where the cores go: 256 rows LPT over 110 cores (iteration 11), heaviest core ~40 K-tile units against an even 38.4; no idle cores.
- Structural fix: chain mode (`CHAIN` define): per stream a contiguous run of 55 cores, rows LPT-scheduled over them, chains of 14 cores in which the first core reads each K/V chunk from DRAM and forwards it core to core (variable_matmul's in0 protocol: ready counter, valid flag, unicast write of the chunk to the same CB address on the next core); every core runs the stream's chunks each round and drops the ones past its row's diagonal. DRAM traffic per call 67 MB -> ~8 chains x 3 rounds x 8 chunks x 64 KB = 12 MB.
- Expected: compute-bound at ~40 units x ~3 us = ~120 us per call (from 229), i.e. -0.44 ms per 4 layers per chip for the forward alone; the same restructure then applies to the Q backward (K/V stream) and the KV backward (Q/dO stream). Standalone measurement: `O/iter11_sdpa_check.py` against the pre-iteration-11 outputs (the forward must stay bit-identical).
- Landed: 140 us per call in the step (`O/iter15_tp_ranked_ops.txt`), 150 us standalone (`O/iter15_sdpa_check.log`), bit-identical; build + check `O/iter15_build_check.sh` -> `O/iter15_build_check.log` (two build attempts: a twice-applied patch block); chain `O/iter15.sh` -> `O/iter15.log`: harness 42, TP gradient check PASS, cooled TP=4 MFU 506.0 ms / 32.06 %.

### Iteration 16: SDPA backward data flow (kernel boost checklist)

- Roofline / bytes: the Q backward re-reads K/V per query row like the forward (67 MB per call, 216 us = 310 GB/s); the KV backward re-reads Q, dO, lse and u per (K/V row, head) item (~100 MB, 289 us = 350 GB/s). FLOP rooflines ~40 us each.
- Fix: the forward's chain data flow (K/V stream per (batch, kv group) for the Q kernel; Q / dO / lse / u stream per (batch, query head) for the head-split KV kernel), chunks of 4 rows per handshake, CBs sized for two chunks.
- Expected: -0.3 (Q) and -0.5 (KV) ms per 4 layers if bandwidth-bound like the forward.
- Landed standalone (`O/iter16_build_check.log`): backward 571 -> 533 us at the TP shard (-7 %), 2181 -> 2006 at the single-chip shape, outputs unchanged. Below half the expected gain: both backward kernels are compute-bound at ~5 us per (row, tile) unit (12 / 16 matmul tiles plus the exp, the dS chain and two DEST transposes), so the removed re-reads were mostly hidden. Next structural fix: one backward pass per (K/V row, head) that also accumulates dQ partials (QK^T and the softmax are recomputed twice today, one per kernel), and `matmul_block` over the chunk instead of per-tile matmuls.
- In the step (`O/iter16_tp_ranked_ops.txt`): bwd Q 0.862 -> 0.783, bwd KV 1.151 -> 1.097 ms per 4 layers; chain `O/iter16.sh` -> `O/iter16.log`: harness 42, TP gradient check PASS, cooled TP=4 MFU 508.4 ms / 31.91 % (wall within noise of iteration 15's 506.0).

### Iteration 17: single-phase RMSNorm backward (kernel boost checklist)

- Roofline / bytes: the two-phase all-core backward moves 60 MB per `[1024, 5120]` call (phase A reads input and dL_dout, phase B reads them again plus the residual gradient, gamma and the partials, writes dx): 150 us at 400 GB/s, measured 228 us (92 + 136) = 260 GB/s. The minimum is 50 MB (input, dL_dout, residual gradient, gamma, dx once): 125 us.
- Where the cores go: 96 (row, slice) items of 54 tiles on 96 cores in both phases; a second launch and a DRAM round trip of the partials in between.
- Structural fix: one program, one (row, slice) item per core with the input / gamma / dL_dout slices resident in L1 (3 x 54 tiles = 324 KB), the per-slice dot product exchanged between the row's 3 cores by writing it into each other's partials CB slot and bumping a semaphore, then the gradient pass from the resident tiles with the residual gradient streamed; `metal/ops/rmsnorm_bw_rows/`, dispatched from `rmsnorm_bw_allcore` when rows x slices fit the grid and no dgamma is needed.
- Expected: ~130-150 us per call (bandwidth-bound on 50 MB), -0.6 to -0.8 ms per 4 layers per chip (8 calls).
- Standalone (`O/iter17_rmsnorm_check.log`): 172 us wall with the residual gradient (155 without), PCC 0.999999 / rel 1.7e-3 vs torch fp64 (the two-phase op's accuracy); norm harness tests 8 passed (`O/iter17_rmsnorm_tests.log`; the 2-batch case falls back to the two-phase op).
- In the step (`O/iter17_tp_ranked_ops.txt`): 1.093 + 0.740 -> 1.320 ms per 4 layers (229 -> 165 us per call); chain `O/iter17.sh` -> `O/iter17.log`: harness 42, TP gradient check PASS, cooled TP=4 MFU 505.6 ms / 32.09 %.

### Iteration 35: ttnn matmul kernel attribution and the TP gate_up forward K-block depth (kernel boost checklist)

- Roofline / bytes: the base matmuls 201 ms/step at 72-85 % of the HiFi4 peak; gate_up forward 1.54 ms per call in the step (1.17 at peak). Thread attribution of ttnn's compute kernel (`bmm_large_block_zm_fused_bias_activation_metal2.cpp`, the fork the mcast factories bind; variants `O/iter35_attrib_patch.py` built in a fresh `TT_METAL_CACHE`): with the math removed the kernels still take 68 % (gate_up fwd 1.09 of 1.61 ms), 67 % (gate_up dX^T 1.03 of 1.53), 77 % (w2 dX^T 0.59 of 0.77), 50 % (qkv fwd), 56 % (out fwd) of their time: operand streaming is the floor, the pack costs nothing (`O/iter35_attrib{,2,3}.log`). The gate_up forward's floor halves (1.09 -> 0.75 ms) with one output block per core (the 178 MB weight shard streamed once instead of twice). Tracy per RISC (`O/iter35_tracy_threads.log`): the compute threads busy 1371 of 1539 us, the writer RISC to the end (the output's write-back, 35 MB per call, exposed after the last K block); the compute thread at ttnn's K blocks of 2 tiles runs 71 cycles per tile matmul against the 64 of the MVMUL sequence (8 matmuls per DEST acquire).
- Expected time after the fix: K blocks of 9 tiles at the 16x5 output block (36 matmuls per acquire, 4.5x fewer partial round trips, fits L1 where ttnn's estimate in the iteration-34 sweep did not: it counted the double buffering twice): 1.605 vs 1.68 ms wall, 1456 vs 1537 us device (`O/iter35_kdepth.log`, `O/iter35_sweep_*.log`) -> ~-0.3 ms per 4 layers (~-3 ms/step). Landed: gate_up forward 6.149 -> 5.845 ms per 4 layers on every chip (`O/iter35_tp_ranked_ops.txt`, `O/iter35_tp_analysis.txt`); chain `O/iter35.sh` -> `O/iter35.log`: harness 48, TP gradient check PASS, cooled TP=4 MFU 462.9 ms / 35.05 % (best 459.1), loss match PASS (0.0301, identical to 32), real weights 476.5.
- Structural fixes tried: (a) the activation width-sharded in L1 so every shard core multicasts its K blocks from L1 (`O/iter35_sharded_in0.py`, `O/iter35_sharded_in0_gateup.log`): 1.68 -> 1.68 ms, the single DRAM-reading in0 sender is not the floor; (b) the two writer kernels' barrier + pop per output subblock replaced by one per block with the block's writes left in flight under the next block's K loop (`O/iter35_writer_patch.py`; bit-exact, `O/iter35_writer_check_{before,after}.log`): no wall change (16x5 / 9: 1.605 -> 1.61), the exposed tail is the DRAM write drain and only a longer last K block hides it (29 us at depth 9 vs 77 at depth 2); reverted; (c) output subblock 4x1 / 2x2 / 1x4 at a math-bound 64-core config: 2.37 ms all three (`O/iter35_sweep_subshape.log`), the unpacker's operand reuse is not the limit; (d) one 32x5 output block with deeper K: depth 3 fits (1.67 ms, no gain), depth 6 needs 2.0 MB of the 1.5 MB L1 (`O/iter35_sweep_32x5_deep.log`); (e) the 2D K-depth sweep of the dX^T shapes on the quick grids (`O/iter35_sweep2d_dx_*.log`) found nothing under ttnn's full-grid choice.
- Lesson: ttnn's mcast matmuls are bound by operand streaming and the per-K-block DEST round trip, not by the FPU, the unpacker or the packer; the config knob that matters is the K-block depth (L1-bound), and the kernels' remaining gap to the nominal peak is mostly the delivered clock (wall 1.68 vs device 1.54 ms: 1.23 GHz). The compute kernel the factories compile is the `_metal2` fork (edits to the legacy twin do nothing; the first two attribution passes patched it and measured nothing); kernel-source variants need a fresh `TT_METAL_CACHE`.

### Iteration 34 (rejected): base matmuls with the full-sync DEST (8-tile output subblocks)

- Roofline / bytes: the base matmuls 201 ms/step at 72-85 % of the HiFi4 peak with 4-tile fp32 output subblocks (DEST half-sync); full sync doubles the DEST and would allow 8-tile subblocks (fewer DEST round trips per output tile).
- Expected time after the fix: -5-10 % on the matmuls (~-10-20 ms/step). Landed: none: every 8-tile subblock fails ttnn's validation (the reuse / mcast factories do not support full sync, `matmul_device_operation.cpp:642-657`), and with 4-tile subblocks full sync changes nothing (gate_up forward 1.67 vs 1.69 ms, bit-exact).
- Structural fix tried: `dst_full_sync_en` in the sweep's compute config, subblocks up to 8 (`O/iter34_matmul_sweep1d.py`, `O/iter34_sweep.sh`); the dX shapes (M = 5184 / 8256) have no 1D config with a 4-row subblock (per-core M 162 / 258) and were swept 2D in iteration 19.
- Lesson: the matmul efficiency gap is in ttnn's mcast kernels (half-sync DEST), not in their program configs; closing it is ttnn kernel work.

### Iteration 33 (rejected): the backward's dx all-reduce overlapped with the LoRA gradient products

- Roofline / bytes: 17 collectives per 4 layers at 4.0 ms (the library's transfer time; 40-90 ms/step with the clock skew); the backward's 8 (the column-parallel layers' dx all-reduces) have independent work beside them: the layer's dA / dB products (~0.6 ms per layer against ~0.24 ms of collective).
- Expected time after the fix: ~-19 ms/step. Landed: none (the TP gradient check's first backward failed in the dispatcher: a queue-0 program cannot start while queue 1 owns the sub-device; the two queues serialise programs, only separate sub-devices run concurrently).
- Structural fix tried: two mesh command queues, `all_reduce_async_begin` / `end` (queue 1, mesh events, the reduce-scatter's intermediates and both outputs pre-allocated and held until `end`, a second `CCLResources` set), the fused backward issuing it before the products, the column-parallel fused forward dropping its `broadcast` (`O/iter33_patch.py`). A sub-device split for the collective workers (16-36 cores in the profile) would cost the 110-core matmul grid 9-18 %: +18-36 ms/step against the ~19 hidden. Also tried: dispatch on the Ethernet cores (120 compute cores): not available on this board (`O/iter33_eth_dispatch.log`).
- Lesson: on tt-metal one sub-device runs one program at a time whatever the queue; overlap needs cores set aside, which this step cannot afford.

### Iteration 32: cross-entropy forward statistics in one pass (kernel boost checklist)

- Roofline / bytes: the vocab-parallel CE forward's local statistics: `ttnn::max` over the [1024, 8192] bf16 shard (16 MB, 0.34 ms per step in the profile), `subtract + exp` into a 128 MB fp32 tensor (0.63) and `ttnn::sum` over it (0.34): ~1.3 ms per step for two reads of 16 MB (~0.1 ms at bandwidth).
- Expected time after the fix: ~0.3 ms per step (two passes over the shard). Landed standalone: 500 -> 215 us for the chain (the accurate exp is 150 of it).
- Structural fix: `vp_ce_row_stats` (device op, factory, reader / writer / compute, binding) with two modes; the sum mode's path: FPU matmul-with-ones (2e-4 relative, source-register rounding), FPU `reduce_tile` (NaN in the unwritten columns), SFPU elementwise accumulation + one FPU reduce on a zeroed DEST (still 5.6e-4: the SDPA exp is bf16-grade), finally the accurate `exp_tile<false>` + `sfpu_reduce<SUM, Float32, REDUCE_ROW>` + `mask_tile` to column 0 (5.3e-6, the old path's).
- Lesson: `sdpa_exp_tile` is the SDPA's bf16-grade exp; every FPU route (reduce, matmul) rounds fp32 to 19 bits: fp32-precision reductions stay on the SFPU (`sfpu_reduce`).
- Verification: standalone vs numpy at the shard shape, the CE / loss / embedding harness subset, chain `O/iter32.{sh,log}`.

### Iteration 31 (rejected): skinny_matmul x reads in flight (kernel boost checklist)

- Roofline / bytes: `SkinnyMatmulPartial` 36 us per call for a 10 MB read over 104 cores (~290 GB/s), the reader's x stream with a barrier per 4-tile block.
- Expected time after the fix: 36 -> ~25 us (the read latency amortised over 32-tile chunks). Landed: 48 -> 47 us per call (wall, partial + reduce): no gain, the stream was not latency-bound.
- Structural fix tried: chunks of up to 8 blocks per barrier, a two-chunk x CB, the compute popping the tail padding (`O/iter31_patch.py`); outputs unchanged (the same differing counts against ttnn's matmul). Reverted, Rejected table.
- Verification of the rejection: `O/iter31_skinny_check.py` before / after (`O/iter31_skinny_baseline.log`, `O/iter31_skinny_check.log`), LoRA harness subset 22 passed on the variant (`O/iter31_lora_tests.log`).

### Iteration 30: SDPA forward diagonal chunk prefix (kernel boost checklist)

- Roofline / bytes: with 8-row chunks a row computes round_up(q + 1, 8) K tiles for q + 1 needed: 3.5 masked tiles per row on average (+21 % of the forward's math stages, ~50 us per call).
- Expected time after the fix: ~-8 us per call at the shard shape. Landed: -1 us (107 -> 106): the masked tiles of a `matmul_block` pass and the exp of a pass cost little once the pass runs anyway; the gain is at shapes with more rows per core (32 heads 334 -> 314, S=2048 288 -> 276).
- Structural fix: `n_cols` per chunk (diagonal position + 1 on the diagonal chunk) through the QK^T passes, the mask stamp, the row max, the exp / sum and the PV contraction (`O/iter30_patch.py`); bit-exact.
- Verification: six shapes bit-exact vs the iteration-28 outputs, attention tests, chain `O/iter30.{sh,log}` (harness 48, TP gradient check PASS, loss match PASS 0.0301, real weights 489.2).

### Iteration 29: RMSNorm backward pass 2 on the FPU (kernel boost checklist)

- Roofline / bytes: `RMSNormBackwardRows` 165 us per call vs a 125 us bandwidth roofline (iteration 20's accounting), 1.32 ms per 4 layers; pass 2's SFPU chain one tile per DEST acquire.
- Expected time after the fix: 145-149 us (iteration 20's variant A measurement). Landed: 148 standalone, 1.320 -> 1.111 ms per 4 layers.
- Structural fix: iteration 20's variant A kernel (`O/iter20_rmsnorm_bw_rows_kernel_variantA.cpp.txt`) installed as is, the factory's compile arg carries -1/C (`O/iter29_patch.py`); rejected in iteration 20 on the exact-fp32 rule only, admissible since the user's 10-31 allowance (PCC 0.999993, rel 4.1e-3).
- Verification: standalone vs fp64 (`O/iter29_norm_bw_check.log`), norm harness subset 8, chain `O/iter29.{sh,log}`: harness 48, TP gradient check PASS, loss match PASS (0.0301), real weights 488.4.

### Iteration 28: KV backward in query-row blocks (kernel boost checklist)

- Roofline / bytes: `SDPABackwardKV` 198 -> 172 us per call at the shard shape (0.72 ms per 4 layers) for 91 us of math stages; per query row 5 CB-serialised DEST passes and 12 fp32 packs (8 L1-accumulate).
- Expected time after the fix: ~150 us (the structure share halved: 8 passes / 20 packs per 4 rows instead of 20 / 48). Landed 172 -> 150 (345 -> 323 backward standalone; `SDPABackwardKV` row 0.722 -> 0.630).
- Structural fix: the chunk's rows as a block (scores two per pass, dP / dS in one pass per row with dP in DEST, dV / dK contracted over the block in DEST), block CBs of `q_block` tiles, `q_block` compile arg (chain chunk under CHAIN, 1 otherwise), `O/iter28_patch.py`; the kernel written from scratch (`sdpa_bw_kv_compute_kernel.cpp`).
- Lesson: `mask_tile(idst, mask)` takes the mask from `idst + 1` whatever the argument (the fw utils say so, the bw helper hid it behind `register_idx + 1`): a masked row cannot share a two-row pass. The symptom (one core's first row at S = 0, deterministic per build) looked like a startup race and a DEST priming cycle hid it; the explicit-mask case of review round 10 exposed the real cause. A new kernel path needs a case for every mask mode before it lands.
- Verification: causal six shapes vs the iteration-26 outputs (ulp-level: dP no longer rounds), explicit-mask GQA / MHA cases vs the causal chain path (`O/iter28c_noncausal_check.log`), attention tests, chain `O/iter28.{sh,log}` (first build) and `O/iter28b.{sh,log}` (final).

### Iteration 27: per-round chain stream bounds (kernel boost checklist)

- Expected time after the fix: backward 387 -> ~330 us (half of its ~100 us stream share), forward 114 -> ~105 (its stream mostly hidden since iteration 26: `nofwd,noread` 121 vs 132). Landed: backward 345 (with chains of 7), forward 110: the forward's remaining time is its per-chunk stages (next structural fix: fewer stages per chunk, see iteration 28 for the KV kernel first).
- Roofline / bytes: every round streamed the whole sequence (24 / 24 / 12 chunks per core for KV / Q / forward) where the rounds' rows need 12 / 12 / 6 (replicated schedule); the stream is the larger part of the skeleton time of the backward kernels.
- Structural fix: per chain and round a stream bound from the round's items (max diagonal chunk for forward / Q, min K row for KV), passed after the row list; readers, writers and compute loop over it; chunk counter over the rounds for slots and credits (`O/iter27_patch.py`, written while the iteration-26 chain ran, applied after it).
- Measured standalone: bit-exact; shard backward 387 -> 345 us (with backward chains of 7, re-swept), forward 114 -> 110.
- Verification: standalone six shapes, attention tests, harness 48, TP gradient check PASS, cooled TP=4 MFU + profile, 8-layer loss match PASS (0.0298, unchanged), TP=4 real weights (`O/iter27.{sh,log}`).

### Iteration 26: SDPA chain credits, 8-row forward chunks, forward chain length (kernel boost checklist)

- Expected time after the fix: forward ~100 us (the credits removing the ready round trip from the hop, ~1 us of 2.6 per chunk, and the 8-row chunk halving the correction stages); landed 115 (credits -4, chunks -1, chain length -16): the hop handshake and the correction stages were not the floor at this shape, the per-core span (chain position, chunk granularity) was; the chain-length sweep found it. The per-core Tracy analysis: `O/iter26_percore_spans.{py,json}`, `O/iter26_lpt_fit.py` (schedule replication + fit), `O/iter26_tracy_ops_perf_results.csv`.
- Roofline / bytes: after iteration 25 the SDPA kernels at the shard shape take 136 / 132 / 198 us (forward / Q / KV) for 50 / 53 / 91 us of math stages (stage removal, `O/iter26_attrib.log`); the math-free, pack-free skeletons still 75 / 62 / 99 us. Tracy: device-bound (forward 122 us device), per-core spans 78-122 us, the slowest core sets the time.
- Structural fixes: credit-based chain hops with three chunks in flight (`O/iter26_patch.py`), 8-row K/V chunks in the forward with the QK^T / exp stages in DEST-sized passes and a two-chunk score CB (`O/iter26b_patch.py`), forward chains of 8 cores (sweep `O/iter26c_sweep*.log`).
- Measured standalone: forward 136 -> 132 (credits) -> 131 (chunk 8) -> 115 (chain 8); backward 393 -> 387; 32 heads 391 -> 359; S=2048 forward 374 -> 320, backward 1304 (iteration 22, `O/iter22i_build_check.log`) -> 1184 (23) -> 1153 (credits) -> 1174 (8-row chunks).
- Rejected with evidence: bf16 DEST on the same algorithm (145 us, PCC 0.9999), position-aware LPT loads (slower), chain 4-6 (DRAM-bound injectors).
- Lesson: removing the `*_init` calls is not a valid skeleton (the device hung twice, `tt-smi -r`); the attribution driver patches compute kernels only and restores them from git, the dataflow files from scratch copies.
- Verification: standalone six shapes (credits bit-exact; chunk 8 within one bf16 ulp), attention tests, harness 48, TP gradient check PASS, cooled TP=4 MFU + profile, 8-layer loss match PASS (0.0298), TP=4 real weights (`O/iter26.{sh,log}`).

### Iteration 25: low-rank update fused into transpose_split (kernel boost checklist)

- Roofline / bytes: `LowrankUpdate` 0.88 ms per 4 layers (12 calls: a full read + write of dx, 10-17 MB each, at ~310 GB/s) on top of `TransposeSplit` 0.49 (which already writes every dx tile): the update's whole cost is the second pass.
- Structural fix: one DEST pass per dx tile: matmul (s_dh tile x A tile column, K = one tile) into DEST0, the transposed input tile into DEST1, SFPU add, one pack. The s_dh tile is the transposed, scaled tail tile, recomputed once per row block per core (items are row-major, so a core sees few row blocks) into a 1-tile resident CB; the reader adds the tail tile per new row block and A's tile column per item. The wide dx's extra columns and the tail output keep the plain path.
- Measured standalone: bit-exact with the two-op path; gate_up 180 -> 74 us, w2 shard 277 -> 116 us per call (`O/iter25c_fused_check.log`).
- Lesson: a patch script run inside a background command hides its traceback (the op's factory anchor had a blank line, so the factory, reader, compute and Python parts never landed, and the first check measured the plain transpose with the term missing: `O/iter25_fused_check.log`, `O/iter25b_fused_check.log`); patch scripts run in the foreground and their asserts are read before the build.
- Verification: standalone three shapes, fused-LoRA harness subset (incl. the wide-w2 test), then the full chain (`O/iter25.{sh,log}`).

### Iteration 24: w2 K-concat removed (wide LoRA input buffer)

- Roofline / bytes: the per-linear ``concat(x, pad(h))`` moves 17 MB (w2 shard) at 433 GB/s: 101 us per call, ~4 ms/step for w2, plus gate_up's 65 us per call (kept, see below).
- Structural fix: the producers write the wide buffer (`swiglu_packed_fw out_cols`, `skinny_matmul tail_out`), the consumers take the wide gradient (`transpose_split dx_cols`, `lowrank_update` with a wider acc, `swiglu_packed_bw`), the base matmul reads the buffer as is. Alternatives weighed: a tensor with logical width ``in`` and padded width ``in + 64`` would have kept the backward untouched, but ttnn's matmul validates the logical K and allocates outputs from logical shapes, so the padded view breaks the forward matmul or the dA product; two tensor objects over one buffer would need a C++ view helper. The plain wide tensor costs three small stride changes instead.
- gate_up: the column-parallel layer's wide ``dx`` ([1024, 5184], 162 tiles) would hit the TP all-reduce's tile divisibility (4 ranks); a 128-row tail (+2.5 % K on the two largest matmuls) or a slice before the collective (a 10 MB copy) each cost about the 65 us concat. Left on the concat path.
- Verification: `O/iter24_wide_check.py` compares every piece against the concat path bit for bit at the shard shape and times the removed and the changed ops; fused-LoRA harness subset; then the full chain.
- Lesson: the single-chip harness fuses only transposed shapes, and w2 at 32768 -> 5120 is not one, so no harness case reached the wide path until `test_mlp_fused_lora_wide_w2` (TP shard width) was added: the first chain's TP gradient check found the wrapper bug instead (`O/iter24.log`). A new model-level path needs a harness case at a shape that actually takes it.

### Iteration 23: SDPA chain hop overlap (kernel boost checklist)

- Expected time after the fix (back-filled, review round 10): ~110 / ~100 / ~170 us (Q / forward / KV, the compute shares of iteration 22's attribution); landed 132 / 137 / 198 (18-41 % of the gain): the stream was not the floor once overlapped, the per-stage structure was (found in iteration 26's attribution).
- Roofline / bytes: after iteration 22 the chain hop (ready handshake, 64 KB unicast, flush, relay, serial with the receive on one RISC) set the SDPA kernels' time: Q 147 us per call at the shard shape against ~110 us of compute, forward 145 against ~100, KV 215 against ~170.
- Structural fix: forwarding on the writer RISC (NOC_1) with a received / forwarded counter pair between the two RISCs (plain L1 words, `wait_min`); the reader only receives. First build forwarded upward in core index: backward 427 -> 592 us, every NOC_1 hop to the next core went the long way round the ring (NOC_1 routes in decreasing coordinates). Chains reversed (injector = highest core of its chain): Q 427 -> 412 (`O/iter23b`), KV -> 395 (`O/iter23c`), forward 145 -> 137 (`O/iter23d`); all bit-exact with iteration 22.
- Measured in the step: SDPA rows -0.156 ms per 4 layers (-1.6 ms/step); TP=4 480.3 ms / 33.78 % (the wall drop beyond the attribution is this run's clock mix).
- Lesson: a NoC's routing direction decides the chain direction; a hop against it costs ~10x.
- Verification: standalone six shapes bit-exact, attention tests, harness 47, TP gradient check PASS, cooled TP=4 MFU + profile, 8-layer loss match, TP=4 real weights (`O/iter23.{sh,log}`).

### Iteration 22: SDPA backward data path (kernel boost checklist)

- Roofline / bytes: `SDPABackwardQ` 196 us per call in the step (0.784 ms per 4 layers per chip), `SDPABackwardKV` 273, `SDPAForward` 140, against ~1.5-2.5 us of FPU + SFPU work per (row, K tile) unit x 38 units per core = 60-100 us; K/V stream 12 MB per call through the chains.
- Structural fix, part 1 (compute): the Q kernel's inner loop over single K tiles (four DEST passes and four CB round trips per tile) became K/V blocks of the chain chunk: QK^T + mask + lse + exp for two columns per pass, dP two per pass, dS two per pass (dP - u in DST[0..1], the fp32 P copies in DST[2..3]), dQ += dS[1 x cols] @ K[cols x qWt] in one pass per output block with one L1 accumulate per block. First build produced inf dQ: `reconfig_data_format(in0, in1)` maps in0 to SrcA, but a matmul unpacks in1 through SrcA and in0 through SrcB (the K tiles were unpacked as fp32, the dS tiles as bf16); fixed with `reconfig_data_format(/* srca */ cb_key, /* srcb */ cb_grad_scores)`. Correct at 517 vs 532 us backward (-3 %): expected -40 %.
- Attribution instead of guessing (`O/iter22_attrib.log`): removing the exp, each matmul stage, the dS math or the mask / scale saves 6-11 us each of ~180; a skeleton compute kernel (no math, no packs, the CB traffic kept) saves 10 (`O/iter22_attrib2.log`, `O/iter22_attrib3.log`): the kernel is data-path-bound. Stream modes (`TTML_SDPA_BW_Q_STREAM`): chain 517, multicast 526, list (per-row DRAM reads) 548 (void: the blocked kernel straddled the list mode's CB, review round 9 P1; with one-row blocks outside the chain: 483 us, bit-exact, `O/review9_list_mode_check.log`); chain injector without its DRAM reads 472 (`O/iter22_attrib4.log`); Tracy per op (`O/iter22_tracy_modes.log`): Q 178 chain / 193 multicast.
- Structural fix, part 2 (data path): the chain injectors read a chunk's tiles with one barrier instead of one per row (two per chunk in the forward, four per row in the KV kernel): backward 517 -> 483 (Q) -> 427 us (KV), forward 149 -> 145 (`O/iter22g_sdpa_check.log`, `O/iter22h_sdpa_check.log`, `O/iter22i_build_check.log`).
- Multicast K/V stream tried and removed: correct outputs where it ran, no gain over the chain, deadlocks on some shapes (details in `perf_analysis.md`, "Iteration 22" and the Ledger). The remaining stream cost is the chain hop itself: the next kernel boost overlaps a core's receive (reader, NOC_0) and forward (writer, NOC_1) with a local received / forwarded handshake, or deepens the chunk buffers.
- Verification: standalone vs the iteration-20 outputs on six shapes (forward, dK, dV bit-exact; dQ one bf16 ulp from the block accumulation order), attention harness tests, full harness 47, TP gradient check PASS, cooled TP=4 MFU + profile, 8-layer host-loss match, TP=4 real weights (`O/iter22.{sh,log}`).

### Iteration 21 (rejected): SDPA forward from the `agobeljicTT/sdpa_scheduler` stack (kernel boost checklist)

- Roofline / bytes: `SDPAForward` 140 us per call in the step (0.56 ms per 4 layers per chip) for 8 heads x 528 score tiles = 38 per core: 3.9 us per score tile against ~1.5 us (8 HiFi4 matmul tiles 0.4, exp 0.7, max / sum / rescale / packs 0.4); K/V DRAM traffic 12 MB per call (chain mode) against 67 MB for main's one-row kernel.
- Structural fix tried: the user's `sdpa_scheduler` stack (PRs #61884 / #62646, same base commit as this branch): `git checkout origin/agobeljicTT/sdpa_scheduler -- metal/ops/sdpa_fw metal/ops/ring_sdpa_fw tests/ops/sdpa_fw_op_test.cpp tests/benchmark/sdpa_fw_benchmark.cpp test_utils/{program_cache,mesh_utils}.hpp`, `sdpa_fw_config.cpp` registered, the benchmark target added with a `mistral24b_tp4_h8kv2d128_s1k` scenario. Relation to this branch: iteration 11's LPT work lists are the stack's longest-first placement; the chain mode (iterations 15-16) is not in the stack; the stack's multi-row items, head stacking and item cuts are not on this branch.
- Measured standalone before any full-model run (`O/iter21_build_check.log`, `O/iter21_sdpa_benchmark{,2}.log`): shard shape default pick 186 us, best of 21 configs 166, chain-mode 149-157 (`O/iter21_sdpa_baseline.log`, same session); S=2048: 401 vs 444. Outputs within one bf16 ulp (`O/iter21_sdpa_check.log`), harness attention tests 4 passed (`O/iter21_attention_tests.log`).
- Rejected at this shape: 64 four-row items on 110 cores (46 idle; the cuts recover 20 us), 128 two-row items with cuts 167; the chain kernel's 256 one-row items balance to a row. Reverted with `git checkout HEAD`, rebuilt, bit-exact against the iteration-20 outputs on all six cases (`O/iter21_sdpa_reverted.log`).
- Lesson: measure a candidate kernel standalone at the exact model shape first (done here; 25 minutes end to end), and a design that wins at S=2048 can lose at S=1024 on item granularity alone.

### Iteration 20 (rejected): norm kernels' per-tile math on the FPU (kernel boost checklist)

- Roofline / bytes: `add_rmsnorm_fw_rows` 126 us per call in the step for 40 MB (317 GB/s), `rmsnorm_bw_rows` 165 us for 50 MB (303 GB/s); both bandwidth-bound by ~80 % and compute-bound by the rest: one tile per DEST acquire with 5-10 SFPU / unpack / pack steps each (the SFPU runs fp32 exactly; the acquire / commit / wait / release handshake is paid per tile).
- Structural fix tried: the per-tile chains on the FPU (`mul_tiles_bcast_rows`, accumulating `mul_tiles`, `mul_reuse_dest` / `add_reuse_dest` with the tile already in DEST), 3-4 tiles per acquire. Expected -0.3 ms per 4 layers per chip (forward 126 -> ~105, backward 165 -> ~135).
- Landed standalone (`O/iter20_build_check{,2}.log`, `O/iter20_norm_bw_variants.log`, `O/iter20_norm_bw_hybrid.log`): forward 138 -> 134 us (-3 %); backward 172 -> 145 us with the residual gradient (-16 %) but rel error vs fp64 1.66e-3 -> 4.09e-3: every `*_reuse_dest` op feeds the fp32 DEST tile back through a 19-bit source register, and the precision loss sits entirely in pass 2 (variant A, SFPU pass 1 + FPU pass 2: 4.09e-3 / 149 us; variant B, FPU pass 1 + SFPU pass 2: 1.66e-3 / 171 us). A precision-preserving hybrid (FPU broadcast and accumulating products with t split into an exact bf16 part and an fp32 remainder, scale and residual add on the SFPU) keeps 1.66e-3 but runs at 173 us: no gain. The forward's 4 us is not worth losing bit-exactness with the two-phase op.
- Kernel sources of the measured variants: `O/iter20_rmsnorm_bw_rows_kernel_{fpu,variantA,variantB}.cpp.txt` (the hybrid pass 2 and the forward FPU pass 2 survive only in the logs and this description).
- Rejected: both kernels reverted to the committed SFPU versions (`O/iter20_norm_{bw,fw}_reverted.log`, norm tests `O/iter20_norm_tests_reverted.log`). The remaining norm cost is the per-tile handshake of exact fp32 SFPU math; a larger per-acquire group is capped by the four fp32 DEST tiles.
- Lesson (also in the Ledger): a kernel run against a stale `_ttml` binary (new CBs in the factory not yet built) hangs the chip; the killed run left the box unusable until `tt-smi -r` (bounded reset per the device-usage skill); build logs must use absolute paths when the build step changes directory.

### Iteration 19: single-phase fused add + RMSNorm forward (kernel boost checklist) and the w2 forward matmul config

- Roofline / bytes (norm forward, `[1024, 5120]`, 8 calls per 4 layers): the two-phase `add_rmsnorm_fw_allcore` moves 50 MB per call (phase A reads residual and delta 20 MB, writes the sum 10 MB and the partials; phase B reads the sum 10 MB and gamma, writes out 10 MB): 125 us at 400 GB/s, measured 148 us in the step (74 + 74) = 340 GB/s, 166 us standalone. The minimum is 40 MB (residual, delta once; sum, out written once): 100 us.
- Where the cores go: 96 (row, slice) items of 54 tiles on 96 cores in both phases; a second launch and a DRAM round trip of the sum and the partials in between.
- Structural fix: one program, one (row, slice) item per core, the slice's sum of squares exchanged between the row's 3 cores (iteration 17's partials protocol), the output pass from the bf16 sum kept resident in L1; `metal/ops/add_rmsnorm_fw_rows/`, dispatched from `add_rmsnorm_fw_allcore` when rows x slices fit the grid with at least 2 slices per row. Same arithmetic and order as the two-phase op. First version: residual / delta / gamma / sum all resident (4 x 54 tiles = 432 KB), read before any compute; final version: residual and delta streamed in double-buffered groups of two tiles behind the compute, the sum written to DRAM group by group behind pass 1, gamma and the sum resident.
- Expected: ~120-130 us per call if bandwidth-bound on 40 MB (-0.15 to -0.2 ms per 4 layers per chip).
- First run (`O/iter19_build_check.log`): PCC 0.67, tiles 0 and 3 of every odd group of 4 wrong: the pass-1 groups used fp32 DEST registers 0-5, but fp32 DEST holds four tiles per `tile_regs_acquire` (half-sync double buffering), so registers 4-5 aliased the other half. Groups of 2 (registers 0-3, as the two-phase phase A) fixed it (`O/iter19_build_check2.log`): bit-exact vs the two-phase op at the model shape, 166 -> 158 us standalone.
- Resident version standalone (`O/iter19_build_check2.log`): 166 -> 158 us at the model shape, bit-exact; in the step (first chain, artefacts `O/iter19a_*`): 150 us per call vs the two phases' 70 + 70 = 140 (`AddRMSNormForwardRows` 1.203 vs 0.565 + 0.558 ms per 4 layers), i.e. slower: the resident design reads 324 KB per core before any compute and overlaps nothing, where the two-phase op streams. An FPU `add_tiles` variant of pass 1 ran at 151 us but its fp32 sums differ at ~2^-11 relative from the SFPU add (1-ulp bf16 differences in sum / out, rms rel 2e-4; `O/iter19_build_check3.log`): not kept.
- Streaming version standalone (`O/iter19_build_check5.log`, `O/iter19_norm_check.log`): model shape 166 -> 138 us per call (-17 %), bit-exact vs the two-phase op (sum, out, rms); `[512, 4096]` (6 x 22) 76 -> 67 us, bit-exact; one-slice shapes 320 -> 298 (not dispatched, see below). Still above the 100-125 us band: pass 1 (add, square, accumulate on the SFPU) and pass 2 (broadcast multiply, scale) are compute-bound per tile at ~96 cores; the next structural fix is the square and the scale on the FPU (`mul_tiles` from CBs, which needs the fp32 sum in a CB or the square from the bf16 sum, a numerics change).
- The first chain (`O/iter19a_chain.log`: harness 47, TP gradient check PASS, 498.8 ms / 32.53 %, loss match PASS max |diff| 0.0298, real weights warm 515.5 ms) measured the resident norm op and NO matmul config: `forward_matmul_kwargs` looked the 2D table up with the layer's input width (8192) instead of the fused K (8256), so the w2 forward row stayed 3.693 ms (`O/iter19a_tp_ranked_ops.txt`); fixed by passing `k=x_cat.shape[-1]` (`O/iter19_config_lookup.log`). The 498.8 of that run is noise against 503.9 (the profile's row deltas are collective waiting +1.5 and the norm +0.08).
- One-slice shapes (rows > 55, e.g. T = 2048): the rows op is slower (375 vs 320 us) and less accurate (80 fp32 accumulator round trips through the ~19-bit unpack path, rel 6e-3 on rms vs the two-phase op's 2 slices); the dispatch requires 2 slices, so these keep the two-phase op. Norm + LoRA harness subset 17 passed.
- Matmul config (glue, `modules/frozen_linear.py::PROGRAM_CONFIG_2D`): 2D block-multicast sweep of the four big TP-local base matmuls (`O/iter19_matmul_sweep2d.py`, `O/iter19_sweep2d_*.log`; `O/iter19_w2_ktail_bench.log` first ruled out the fused-K tile count 258 as the cause). Interleaved A/B (`O/iter19_ab_check.{py,log}`, medians): w2 forward `[1024, 8256] x [8256, 5120]` 1174 us (the model's full-grid call) -> 1043 (2D 11x10, `in0_block_w` 6, subblock 4x1; 11x8 1058), -11 %; gate_up forward: no 2D config beats ttnn's 1D choice (per_core_N 47 overflows L1 beyond `in0_block_w` 1); both dX shapes (`[8256, 5120] x g^T`, `[5184, 16384] x g^T`): the model's full-grid call is the fastest (782 / 1592 us vs 2D 989-1052 / 2111); qkv / out forward unchanged. The first dX sweeps (`O/iter19_sweep2d_1024x5120x8256.log`, `...1024x16384x5184.log`) used the wrong orientation and are superseded. Output within one bf16 ulp of the default (max |diff| 2.0 at |y| ~ 90, the fp32 accumulation order changes): loss match rerun in the chain.
- Expected in the step: w2 forward 3.69 -> ~3.3 ms per 4 layers per chip (-0.4), norm forward -0.06; ~-4.5 ms per step.
- Chain `O/iter19.sh` -> `O/iter19_chain.log`: harness 47 passed, TP gradient check PASS, cooled TP=4 MFU 495.4 ms / 32.75 % (best 487.9; `O/iter19_tp_mfu_tp4.log`), loss match PASS (max |diff| 0.0298), TP=4 real weights warm 509.1 ms; in the step w2 forward 3.693 -> 3.249 ms per 4 layers (-12 %), norm forward 1.123 -> 1.010 (126 us per call), `O/iter19_tp_ranked_ops.txt`.

### Stage review round 10 (iterations 23-27)

- `optimization/review_round10.md`: more-work-needed, no P1; P2: the bf16-DEST rejection measured a straw-man variant (the intended one measured after iteration 28), commit 26 bundled three changes, stale ranked table / state statements, document format of sections 23-27, checklists without expected times and two unresolved citations. All doc items answered in the round-10 commit; the device items (bf16 DEST as intended, a non-causal standalone case, the SwiGLU backward stride A/B) follow it.

### Stage review round 8 (iterations 13-18)

- `review_round8.md`: more-work-needed, four P2 doc / hygiene items (iteration-18 attribution counted collective waiting and explained the A/B gap with a clock figure that does not separate the runs; the failed run-B profile unrecorded; ranked table still "after iteration 12"; iteration-13 docs in the round-7 commit, a stray `iter18_cmake.log`, "(this commit)" left in the table). Answered in this commit: attribution restated collectives-excluded with all product rows (1.63 -> 1.47 -> 1.32 ms per 4 layers), the dA row drop attributed to the dropped `core_grid`; Ledger entries for the failed profile and the iteration-13 docs; ranked table regenerated from run B (`O/iter18b_tp_ranked_ops_top80.txt`); `iter18_cmake.log` removed; two `skinny_contract` harness cases added (the routed N = 1536 shape, a short last column group); loss match rerun in the iteration-19 chain.

### Iteration 18: LoRA gradient products `dB = s g^T h`, `dA = dh^T x` (kernel boost checklist)

- Roofline / bytes: per fused layer the four products read the `[1024, N]` activation or gradient once (N = 16384 / 5120 / 1536 for dB, 5120 / 8192 / 1024 for dA): gate_up dB 32 MB -> 81 us at 400 GB/s, measured 144 us (ttnn `transpose_a` matmul, `[N, r]` orientation, 225 GB/s) plus a scale multiply; the out / w2 dB 10 MB -> 25 us, measured 49 us; dA 10 MB, measured 32 us (ttnn's `[r, N]` orientation, 312 GB/s). 1.63 ms per 4 layers per chip across the six rows of the ranked table (`O/iter17_tp_ranked_ops.txt`, `--top 70`).
- Where the cores go: ttnn uses 33-88 cores for these shapes; the `[N, r]` orientation runs the strided transposed read of the wide operand.
- Structural fix tried: `metal/ops/skinny_contract/` (`scale * small^T @ big`, the small operand transposed once on the host and resident in L1; items = groups of up to 4 adjacent output columns over all 110 cores, each K row of the group streamed in 8-row batches; `transpose_dest` for the `[N, r]` output; scale folded into the pack).
- Expected: bandwidth-bound at ~330 GB/s (`rmsnorm_bw_rows`' streaming rate): gate_up dB ~100 us, N = 5120 ~32 us, dA no worse than ttnn; -0.6 to -0.8 ms per 4 layers per chip.
- Standalone (`O/iter18_build_check.log`, prefetch version; PCC 0.999999 / 1.000000 at all seven shapes vs ttnn): dB N = 16384 157 -> 123 us, 5120 56 -> 54, 1536 28 -> 25; dA N = 5120 32 -> 53, 8192 50 -> 71, 1024 18 -> 20; r = 32 N = 2048 dB 44 -> 58. Below half the expected gain: per item one `cols`-tile row of `big` per K step (8 x 8 KB strided reads per barrier) and 32 `matmul_tiles` per output tile for 16 useful rows; the op streams at 190-265 GB/s where ttnn's `[r, N]` orientation reaches 312.
- Orientation check (`O/iter18_orientation_check.{py,log}`): ttnn `[r, N]` product + transpose of the small output 110 / 41 / 39 us at N = 16384 / 5120 / 1536 vs ttnn `[N, r]` 156 / 55 / 28 vs the op 123 / 53 / 25. Router (`modules/lora.py::skinny_contract`): dA and wide dB (N >= 4096) on ttnn's `[r, N]` product (+ transpose for dB), the narrow dB (qkv, N = 1536) on the op.
- Run A (op for all dB with N >= 4096; `O/iter18.sh` -> `O/iter18_chain.log`): harness 45 passed (3 new `skinny_contract` cases), TP gradient check PASS, cooled TP=4 MFU 498.0 ms / 32.58 % (`O/iter18_tp_mfu_tp4.log`); in the step gate_up dB 0.575 -> 0.472, N = 5120 dB 0.389 -> 0.387 ms per 4 layers (`O/iter18_tp_ranked_ops.txt`).
- Run B (final router; `TAG=iter18b bash O/iter18.sh` -> `O/iter18b_chain.log`): harness 45 passed, TP gradient check PASS, cooled TP=4 MFU 503.9 ms / 32.19 % (best 495.1; run A 498.0 / 32.58 % at a different clock mix, the per-step spread inside one run is 484-511 ms) (`O/iter18b_tp_mfu_tp4.log`); in the step gate_up dB 0.369 (92 us per call; run A 0.472, iteration 17 0.575), N = 5120 dB 0.234 (A 0.387, 17 0.389), the `[16, N]` transposes + pads 0.09; the products 1.11 vs 1.28 (A) vs ~1.45 (17) ms per 4 layers; chip 0 kernels 45.9 vs 46.0 (A) vs 48.2 (17) ms per profiled step (`O/iter18b_tp_analysis.txt`) (`O/iter18b_tp_ranked_ops.txt`).
- Next structural fix for these products: none at the kernel level worth its cost for the wide rows (gate_up dB 88 %, N = 5120 86 %, w2 dA 85 % of 400 GB/s after run B); the small rows are launch-bound (out dA 17 %, qkv dB 34 %), a few us each; the `[16, N]` transposes cost ~5 us each.

### Stage review round 7 (iterations 11-12)

- `review_round7.md`: more-work-needed (UMD pointer in the iteration-11 commit; ranked table and attribution cells); answered in the same day's commits (pointer restored `415ec0f1d3e`, docs regenerated, runtime-arg guards, router M check); loss match rerun scheduled in the iteration-14 chain.

### Build environment note (review round 7)

- Every measurement of this branch was built against the worktree's local UMD checkout `292fe208659` (the submodule pointer in `main` is `7eb549a29db`; the worktree has carried the newer checkout since before the bring-up started). The iteration-11 commit `7f0a7f1f896` accidentally recorded that pointer through a blanket `git add -u`; `415ec0f1d3e` restores `7eb549a` in the tree without touching the worktree.

### Iteration 12 candidate: TP collective parameters (rejected)

- Env knobs `TTML_CCL_NUM_LINKS`, `TTML_CCL_{RS,AG}_{CHUNKS_PER_SYNC,WORKERS_PER_LINK,BUFFERS_PER_CHANNEL}` read per call in `ttnn_fixed/distributed/ttnn_ops.cpp` (unset = library defaults); sweep `O/iter12_ccl_sweep.sh` -> `O/iter12_ccl_sweep.log`: defaults fastest (reduce-scatter 122 us, all-gather 114 us standalone vs 281 / 120 in the step: the difference is chip skew).

### Iteration 12: lowrank_update

- Op `metal/ops/lowrank_update/` (device op, factory, reader / writer / compute kernels), registered in `CMakeLists.txt`, `metal/operations.hpp`, `nanobind/nb_ops.cpp` (`ttml.ops.metal.lowrank_update(acc, a, w, scale, transpose_b)`); routed from `modules/lora.py::lowrank_update` (used by `LoraApply.forward`, `_LoraUpdateAdd.forward`, `fused_lora.py` backward).
- Build + standalone check + LoRA tests: `O/iter12_build_check.sh` -> `O/iter12_build_check.log` (first run: the two-image case failed on a missing pop at the image switch, and the router sliced a `ttnn.Shape`; both fixed, rerun with `SKIP_BUILD=1`).
- Chain `O/iter12.sh` -> `O/iter12.log`: harness 38 passed, TP=4 gradient check PASS, cooled TP=4 MFU 518.5 ms / 31.29 % (`O/iter12_tp_mfu_tp4.*`), 4-layer TP profile `O/iter12_tp_tracy_tp4_4layers` (`O/iter12_tp_analysis.txt`, `O/iter12_tp_ranked_ops.txt`).

### Loss match and review round 6

- Loss match after iterations 9 and 10: `O/iter10_loss.sh` -> `O/iter10_loss.log`: 8-layer real-weight host loss `O/iter10_train_8layers_hostloss.log` + `O/iter10_loss_8layers_hostloss.json` -> `O/iter10_loss_comparison.txt` (max |diff| 0.0273, PASS); TP=4 40 layers real weights `O/iter10_full_tp4_real.{log,json}` (baseline trajectory, 543.0 ms warm). Produced after commit `53ec655a9f9`, committed with the review-round-6 items.
- Round 6 (`review_round6.md`, iterations 9 and 10): more-work-needed on attribution (the iteration-9 kernel gain was stated across chips with collective waiting included; restated same-chip, collectives excluded: -0.86 ms per 4 layers, the fused forward costing what the 32-core one did), the uncommitted loss match, a missing harness case for `skinny_matmul` (added: `test_skinny_matmul_matches_torch`, 12 cases, `O/review6_skinny_tests.log`), stale README lines and work-log hashes.
- Docs restructured into the skimmable format (`perf_analysis.md`, this file, `README.md`); long prose moved to `O/perf_analysis_details.md`; disagreements between the docs marked "(docs disagree)" for resolution.

## Commits (local, never pushed)

| iteration | commit | content |
|---|---|---|
| TP support + baselines | `8eb4febbf2f` | `train.py --tp`, mesh descriptor, frozen-embedding guard, TP=4 baselines 598.7 / 641.9 ms |
| 1 | `c4bd49f83bb` | `LoraApply` port (3-op LoRA), `diff_tracy.py`, `tp_analysis.py`, chain tooling |
| 2 | `e7e8355de22` | `frozen_linear.py` (transposed frozen weights), parallel linears and LoRA wrappers honour it, microbenchmarks |
| 3 | `70c5cb9fe6f` | `fused_lora.py` (LoRA fused into the base matmul), harness fused case |
| exit | `f3c61f23df8` | loop exit, final validation, gap decomposition |
| evidence | `0ce39d1a1e0` | run/harness/Tracy logs force-added past the repo's `*.log` ignore (pre-commit stripped trailing whitespace from the harness logs; contents otherwise as produced) |
| review follow-up | `4229a061220` | TP=4 gradient check, 16-layer point, exit restated at the measured clock, arithmetic and attribution fixes |
| review round 2 close | `5bfaf0a9775` | text corrections, phase closed |
| 4 | `dc6ccf3db8e` | `write_tile_block` metal op, `FORWARD_PROGRAM_CONFIG` (swept gate_up config), sweep scripts, iteration-4 evidence, doc table fixes |
| 5 | `7c19e385351` | batched TP all-reduce of the partial LoRA gradients (`reduce_partial_lora_grads`), TP-shape sweep filters, iteration-5 evidence |
| 5b | `758a7300a8d` | batched mode for the composite TP adapters (qkv, out), `tp_rank_ops.py`, `EXTRA_ARGS` in `tp_measure.sh`, iteration-5b evidence |
| 6 | `190f3aad6fd` | `rmsnorm_bw_allcore` metal op, `rmsnorm_allcore.py`, `RMSNormLayer.rmsnorm_op` hook, harness case, iteration-6 evidence |
| 7 (rejected) | `d546567c67f` | evidence and write-up of the rejected qkv / out fused-path trial at TP; no code change |
| TP sweeps | `d03dfffa23c` | TP-local shape sweeps: no config win; logs and write-up |
| 8 | `4777ff7646a` | frozen vocab-parallel embedding table, resumed-loop state section, iteration-8 evidence |
| review 3 | `4ae4eef5bcc` | review round 3 and its follow-up: graph-free frozen paths, loss match on the final path, doc corrections |
| review 4 | `ddf3a77be15` | review round 4 items: README / work-log rows, frozen-embedding test assertion, no-grad norm harness case, sweep numbers |
| review 5 close | `31b070903d3` | round-5 clean-pass recorded, three wording notes |
| 9 | `d5ba1a1b877` | residual add fused into an all-core RMSNorm forward/backward, CB full-block discipline in both all-core norm kernels, two-items-per-core tests, iteration-9 evidence |
| 10 | `53ec655a9f9` | K-split `skinny_matmul` op for the rank-wide LoRA products, routing in `lora.py` / `fused_lora.py`, iteration-10 evidence |
| review 6 | `a492accb3a6` | review round 6 items: iteration-9 attribution restated, loss match after 9 and 10, `skinny_matmul` harness cases, README / work-log corrections |
| review 6 close | `5084c423f54` | round-6 re-check clean-pass recorded, steady-step wording |
| docs restructure | `884e0eedff6` | skimmable format for `perf_analysis.md`, README and this log; `O/perf_analysis_details.md` |
| 11 | `7f0a7f1f896` | SDPA work lists (LPT) in the forward and Q backward, head-split KV backward with the `sdpa_bw_kv_reduce` op, harness cases, iteration-11 evidence |
| 12 candidate | `2c3a0d63846` | TP collective parameter knobs and sweep: no win, recorded as rejected |
| 12 | `a95fcbdf37d` | `lowrank_update` op, LoRA routing, harness cases, iteration-12 evidence |
| UMD pointer | `415ec0f1d3e` | the submodule pointer bumped by the iteration-11 commit restored |
| review 7 | `456f9cf7a8e` | round-7 items: ranked table regenerated, attribution cells, runtime-arg guards, router M check, review record |
| 13 | `a1fc69ebed8` | `transpose_split` op, fused backward routing, harness cases, iteration-13 evidence |
| 14 | `8b86016e718` | `vp_cross_entropy_bw` op, loss backward routing, harness cases, loss match, iteration-14 evidence |
| 15 | `7a21a21258a` | SDPA forward chain mode (`metal/ops/sdpa_fw/`), iteration-15 evidence |
| 16 | `f045fb8eba8` | SDPA backward chain mode (`metal/ops/sdpa_bw/`), iteration-16 evidence |
| 17 | `f07712081ba` | `rmsnorm_bw_rows` op and its dispatch from `rmsnorm_bw_allcore`, iteration-17 evidence |
| 18 | `2a75ffce1bd` | `skinny_contract` op, `skinny_contract` router in `lora.py` (ttnn `[r, N]` orientation for dA and the wide dB), iteration-18 evidence |
| 19 | `11f9d923524` | `add_rmsnorm_fw_rows` op (dispatched from `add_rmsnorm_fw_allcore`), 2D program config for the w2 forward matmul (`frozen_linear.py`), review-round-8 answers, iteration-19 evidence (bundled against the one-change rule: the op, the config and the review answers share one commit; noted by review round 9) |
| 20 | `a3d3ec03947` | rejected FPU norm-kernel experiment: evidence and docs only (kernels unchanged) |
| state | `c219767ba30` | state after iteration 20: where the remaining step time is |
| 21 | `fd17a21a00b` | rejected `sdpa_scheduler` forward port: evidence and docs only (kernels unchanged) |
| 22 | `66b8a997d14` | blocked `sdpa_bw_q` compute kernel, single read barrier per chunk in the three chain injectors, `TTML_SDPA_BW_Q_STREAM` knob, iteration-22 evidence |
| review 9 | `912f9562ba6` | stage review round 9 record and its doc answers |
| 23 | `0be1385dbde` | SDPA chain forwarding on the writer RISC / NOC_1 (forward, Q, KV readers, writers, factories), chains reversed, iteration-23 evidence |
| review 9 P1 | `4041c62353d` | `sdpa_bw_q` factory: one-row K/V blocks outside the chain; list-mode and chain-mode checks, attention tests |
| 24 | `b88de55dd41` | wide fused-LoRA input for w2 (first chain `O/iter24.log` failed in the wrapper's backward, fixed; chain `O/iter24b*`): `swiglu_packed_fw out_cols`, `skinny_matmul tail_out`, `transpose_split dx_cols`, `lowrank_update` / `swiglu_packed_bw` wide gradient, `MLP.swiglu_op` hook, iteration-24 evidence |
| 25 | `0cd8e1213a8` | `transpose_split(w=A)`: the low-rank update fused into the transpose (factory, reader, compute, binding), the fused LoRA backward routed through it, iteration-25 evidence |
| 26 | `c193bd73925` | bundles three changes, each measured on its own (review round 10): chain credits (`CHAIN_SLOTS = 3`, counting semaphores; `O/iter26_build_check.log`), 8-row forward chunks (`kMaxSkChunkT = 8`, passes, two-chunk score CB; `O/iter26b_*`), per-kernel chain-length knobs (forward 8; `O/iter26c_*`); `lpt_schedule` initial loads (dropped in the round-10 commit) + `env_uint32_or` in `program_utils.hpp`, iteration-26 evidence |
| 27 | `46dca91d991` | per-round chain stream bounds in the three SDPA chain kernels (factories compute them per chain and round, readers / writers / compute loop over them), backward chain length 7, iteration-27 evidence |
| 28 | `a891b66ff30` | blocked `sdpa_bw_kv_compute_kernel.cpp` (rows per chunk, dP in DEST, block contractions; masked rows alone), `q_block` compile arg and block CBs in the KV factory, iteration-28 evidence |
| review 10 | `85fc26239b7` | review round 10 (iterations 23-27, `optimization/review_round10.md`): five P2 items answered (docs: ranked table from iteration 27, sections 23-27 in the document format with the full text in `D`, expected-time lines, Ledger entries, Rejected rows), `initial_load` dropped from `lpt_schedule`, the stale forward-kernel comment and the unused valid semaphore removed; hard-check gaps measured: bf16 DEST as intended (rejected on time), the SwiGLU backward stride A/B (no effect), the explicit-mask SDPA cases (found iteration 28's mask-register bug) |
| 29 | `2352f1d3241` | `rmsnorm_bw_rows` pass 2 on the FPU (variant A kernel), factory arg -1/C, iteration-29 evidence |
| 30 | `82f35c84e55` | SDPA forward diagonal chunk prefix (`n_cols` through the QK^T, mask, max, exp and PV stages), iteration-30 evidence |
| 31 (rejected) | `85a06ef9dfe` | docs and evidence only: skinny_matmul read-ahead measured and reverted |
| 32 | `b1b0e107c27` | `metal/ops/vp_ce_row_stats` (new op: device op, factory, kernels, wrapper, binding), `losses.cpp` forward statistics through it, iteration-32 evidence |
| 33 (rejected) | `46944a1d62f` | docs and evidence only: the second-queue collective overlap (implemented, refused by the dispatcher) and the Ethernet dispatch (unavailable), both reverted |
| 34 (rejected) | `cd55c4e3bfc` | docs and evidence only: the full-sync DEST matmul sweep, refused by ttnn's matmul validation |
| 35 | (this commit) | `frozen_linear.py` 1D config for the TP gate_up forward (K blocks of 9, 16x5), the K-divisibility guard; the ttnn matmul kernel attribution (variants, writer-kernel experiment reverted), iteration-35 evidence |

- Note on `O/iter1_loraapply_mfu_tp4_hot.json`: renamed after the fact from `iter1_loraapply_mfu_tp4.json` (its `args.loss_log` field still carries the old name); it is the hot-chip run of `perf_analysis.md`, "Iteration 1" / Ledger.
