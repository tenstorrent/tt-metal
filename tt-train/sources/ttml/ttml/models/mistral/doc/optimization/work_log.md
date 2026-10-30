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
| review 7 | (this commit) | round-7 items: ranked table regenerated, attribution cells, runtime-arg guards, router M check, review record |
| 13 | (next commit) | `transpose_split` op, fused backward routing, harness cases, iteration-13 evidence |

- Note on `O/iter1_loraapply_mfu_tp4_hot.json`: renamed after the fact from `iter1_loraapply_mfu_tp4.json` (its `args.loss_log` field still carries the old name); it is the hot-chip run of `perf_analysis.md`, "Iteration 1" / Ledger.
