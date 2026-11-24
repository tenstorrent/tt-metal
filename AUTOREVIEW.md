# Auto review: branch `fplavec/gpt_oss_quasar_bringup_2`

- Target: all 8 commits on the branch since it left `main` (`git diff 840db3161d0 9b18a87485f`).
- `BASE`: `840db3161d0e2f4e3d760e75e0c895f7d5d5020a` (merge-base with `origin/main`).
- `BASELINE`: `9b18a87485f7c0b5477b98272056253cedd30a44` (branch head; the fixes are uncommitted edits on top of it).
- Date: 2026-11-23.

## 1. Summary per reviewer

| Reviewer | Reviews | Ended with |
|---|---|---|
| 1 | 4 | `NO FURTHER COMMENTS` |
| 2 | 3 | `NO FURTHER COMMENTS` (F3 left at low severity for the branch owner to decide) |
| 3 | 3 | `NO FURTHER COMMENTS` |

Totals over all issues (an issue that a later reviewer raised again is counted once): 22 fixed (3 of them fixed in a different way than suggested), 3 out of scope, 2 partly rejected, 0 unresolved by a reviewer. One issue is left to the branch owner (section 3).

## 2. All issues raised

### Reviewer 1

- **R1-F1. The (6, 3) grid exclusion and its craq-sim comment were stale**: the cause, problem L, was already fixed on `main` by #66243. See [mlp.py:142-145](models/experimental/ops/quasar/gpt_oss/tt/mlp.py#L142-L145).
  - **Fixed.** `excluded_grids` was removed from `mlp.py`, `fit_core_grid`, `ProgramConfig` ([config.py:15-39](models/experimental/ops/quasar/gpt_oss/tt/experts/config.py#L15-L39), [config.py:101-103](models/experimental/ops/quasar/gpt_oss/tt/experts/config.py#L101-L103)) and the companion test.
  - The test header paragraph about implicit sync was removed too.
  - On 8x4 the down projections now get (6, 3). Cases 01 and 03 passed on (6, 3) on ttsim.
- **R1-F2. Three generated tests have a hand-edited `_OP`**, which a regeneration reverts: [test_add.py:29](models/experimental/ops/quasar/tests/gpt_oss_ops/test_add.py#L29-L29), [test_linear.py:32](models/experimental/ops/quasar/tests/gpt_oss_ops/test_linear.py#L32-L32), [test_multiply.py:29](models/experimental/ops/quasar/tests/gpt_oss_ops/test_multiply.py#L29-L29).
  - **Out of scope.** The branch owner made the edit on purpose (commit `8825a557f4e`).
  - The fix belongs in `OP_EXPR` in [generate_from_graph_capture.py:429-439](models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py#L429-L439). That map is shared with the llama suite, so the change needs its own PR. Reviewer 2 raised the same issue (R2-F4).
- **R1-F3. Broken and stale links in [gpt_oss_ops_analysis.md](models/experimental/ops/quasar/gpt_oss/gpt_oss_ops_analysis.md#L1-L20)**: two `metal2_host_api` paths that do not exist, and line ranges in `decode.py`, `graph_case.py`, `model.py` and `sd_mesh_command_queue.cpp` that the branch's own commits had moved.
  - **Fixed.** All links were corrected.
  - A check now resolves every link in the three reports from the report's folder and requires a range inside the file. Reviewers 2 and 3 raised it again (R2-F2, R3-F8).
- **R1-F4. The reports contradicted each other and the code about problem L** (a fix still to come, an `xfail` that no longer exists).
  - **Fixed.** The problem L row and its next steps in [gpt_oss_ops_analysis.md](models/experimental/ops/quasar/gpt_oss/gpt_oss_ops_analysis.md#L157-L157) now name #66243.
  - [gpt_oss_small_grid_analysis.md](models/experimental/ops/quasar/gpt_oss/gpt_oss_small_grid_analysis.md#L104-L104) states the old state in the past tense with dates.
- **R1-F5. The `sparse_matmul` companion always wrote case 01's output to DRAM**, a call the model does not make on 8x4.
  - **Fixed.** The helper is now public: [down_output_memory_config, decode.py:14-31](models/experimental/ops/quasar/gpt_oss/tt/experts/decode.py#L14-L31).
  - The companion uses the same function to choose the placement for the device under test: [test_sparse_matmul_small_grid.py:65-72](models/experimental/ops/quasar/tests/gpt_oss_ops/test_sparse_matmul_small_grid.py#L65-L72).
  - Reviewer 2 raised it again (R2-F5).
- **R1-F6. `merge_heads_on_device` ignored head padding of the input.**
  - **Fixed differently.** The function raises `NotImplementedError` when the heads dimension is not `num_heads`, and the docstring says so: [merge_heads.py:20-22](models/experimental/ops/quasar/gpt_oss/utils/merge_heads.py#L20-L22).
  - gpt-oss has 64 heads and no head padding, so slicing the heads is not needed. Reviewer 3 raised it again (R3-F5).
- **R1-F7. `pcc_log.py` problems:**
  - the layer steps read back without a composer on a mesh of several devices, and the head steps composed along the wrong dimension;
  - the constant-tensor case compared only one element;
  - the golden data was never cleared after a test.
  - **Fixed.**
    - [_readback, pcc_log.py:126-137](models/experimental/ops/quasar/gpt_oss/utils/pcc_log.py#L126-L137) gets the device from the tensor, and on a mesh of several devices it raises an error that `log_op` logs.
    - The constant case compares the full masked tensors ([pcc_log.py:118-121](models/experimental/ops/quasar/gpt_oss/utils/pcc_log.py#L118-L121)).
    - `test_model` clears the golden data in a `finally` block ([test_modules.py:1296-1314](models/experimental/ops/quasar/gpt_oss/tests/unit/test_modules.py#L1296-L1314)).
  - Raised again as R2-F8, R3-F6 and R3-F7.
- **R1-F8. `pcc_log.py` is a near copy of the llama `pcc_log.py`.**
  - **Out of scope.** One shared module would change llama model code. Raised again as part of R2-F7.
- **R1-F9. Model logic in `conftest.py`, imported by a test.**
  - **Fixed.** `merge_heads_on_device` moved to [utils/merge_heads.py:7-32](models/experimental/ops/quasar/gpt_oss/utils/merge_heads.py#L7-L32). [conftest.py:13](models/experimental/ops/quasar/gpt_oss/conftest.py#L13-L13) and the companion import it from there. Raised again as R3-F4.
- **R1-F10. `_EXCLUDED_GRIDS` in the test copied a literal from `mlp.py`.**
  - **Fixed** by R1-F1. Raised again as part of R3-F3.
- **R1-F11. The `fit_core_grid` docstring said "the 1D multicast factories", and the function ignored `excluded_grids` for (1, 1).**
  - **Fixed.** The docstring now says "the sparse_matmul factory", and `excluded_grids` is gone.
- **R1-F12. Comments that describe distant code** (llama and Qwen3-VL, the pad factory, the simulator error text).
  - **Fixed.** The llama sentence was removed. The pad comment cites #55526 ([merge_heads.py:30](models/experimental/ops/quasar/gpt_oss/utils/merge_heads.py#L30-L30)). The simulator text was removed with R1-F1.
- **R1-F13. The reports cite local paths, and one sentence says "applied as uncommitted changes".**
  - **Out of scope, and partly rejected.** The local paths are the owner's choice for a record of runs. The sentence describes how a past run was set up; it says nothing about the report's own commit state.
  - Reviewer 2 raised the paths again (R2-F3, section 3).
- **R1-F14. The companion tests repeated the new skips of `graph_case`.**
  - **Fixed.** The per-test skips were removed from the concat companion and the two SDPA companions (for example [test_scaled_dot_product_attention_small_grid.py:53-54](models/experimental/ops/quasar/tests/gpt_oss_ops/test_scaled_dot_product_attention_small_grid.py#L53-L54)). Their headers now say that `run_case` skips.
  - Raised again as R2-F9 and R3-F10.
- **R1, round 2: one stale `decode.py` link, a short range, and one sentence in the small-grid report.**
  - **Fixed.** A stricter link check also found three single-line links without a range, which were fixed.
- **R1, round 3: a header range had gone stale after the header was shortened.**
  - **Fixed** (`#L4-L40`).

### Reviewer 2

- **R2-F1. `down_output_memory_config` called `element_size()`, which raises for `bfloat8_b`.** So on Wormhole and Blackhole, where the activations are `bfloat8_b`, every decode MoE step would fail.
  - **Fixed.** The helper uses `down_input.spec.compute_packed_buffer_size_bytes()` ([decode.py:25-26](models/experimental/ops/quasar/gpt_oss/tt/experts/decode.py#L25-L26)).
  - Confirmed on the host: `element_size()` raised "datum for bfp2, bfp4, bfp8 is invalid".
  - On Wormhole n150, `test_decoder[decode_low_latency]`, paged and unpaged, passed with the fix. Reviewer 3 raised it again (R3-F1).
- **R2-F2.** The same as R1-F3: **Fixed.** The `decode.py` links were updated again after R2-F1 added one line.
- **R2-F3.** Local machine paths in the reports. See section 3.
- **R2-F4.** The same as R1-F2: **Out of scope.**
- **R2-F5.** The same as R1-F5: **Fixed.**
- **R2-F6.** The same as R1-F1: **Fixed.**
- **R2-F7. The `pcc_log.py` docstring claimed call sites use `x = log_op(...)`, and the module copies the llama one.**
  - The docstring is **fixed** ([pcc_log.py:8-12](models/experimental/ops/quasar/gpt_oss/utils/pcc_log.py#L8-L12)).
  - The shared module is **out of scope** (R1-F8).
- **R2-F8.** The same as R1-F7: **Fixed.**
- **R2-F9.** The same as R1-F14: **Fixed.**
- **R2-F10. Test headers describe distant C++ details.**
  - **Partly fixed.** The `sparse_matmul` paragraph was removed.
  - **Rejected for the rest.** The matmul and paged SDPA headers record why each narrowed or regridded case still takes the same path as the captured call ([test_matmul_small_grid.py:21-34](models/experimental/ops/quasar/tests/gpt_oss_ops/test_matmul_small_grid.py#L21-L34)). That is what makes each companion a faithful stand-in. The reviewer accepted this.
- **R2-F11. No test of `fit_core_grid`.**
  - **Fixed.** New host-only test [test_fit_core_grid.py:1-52](models/experimental/ops/quasar/gpt_oss/tests/unit/test_fit_core_grid.py#L1-L52): fixed examples plus the documented rules over a sweep. Raised again as part of R3-F2.
- **R2-F12. The summary of the small-grid report was in the present tense about problem L.**
  - **Fixed.** It is now in the past tense with dates ([gpt_oss_small_grid_analysis.md:139](models/experimental/ops/quasar/gpt_oss/gpt_oss_small_grid_analysis.md#L139-L139)).

### Reviewer 3

- **R3-F1.** The same as R2-F1: **Fixed.**
- **R3-F2. No test of `fit_core_grid`, of the SRAM/DRAM choice, or of the `bfloat8_b` path.**
  - `fit_core_grid`: **fixed** by R2-F11.
  - The SRAM/DRAM choice: no separate host test, **rejected** with this evidence. The function needs a device for the bank size. Two device tests cover it: the Wormhole `test_decoder` decode cases (`bfloat8_b` input, SRAM result), and companion case 01 on ttsim (SRAM on 8x4, DRAM on 2x1).
  - The reviewer accepted this.
- **R3-F3.** The same as R1-F1 and R1-F10: **Fixed.**
- **R3-F4.** The same as R1-F9: **Fixed.**
- **R3-F5.** The same as R1-F6: **Fixed differently.**
- **R3-F6.** The same as R1-F7 (multi-device readback): **Fixed.**
- **R3-F7.** The same as R1-F7 (constant case), plus the comment "Both constant":
  - **Fixed.** The comment now says "at least one side is constant".
- **R3-F8.** The same as R1-F3, plus the `conftest.py` link ranges that moved when `merge_heads_on_device` was taken out of that file:
  - **Fixed** ([conftest.py:44-108](models/experimental/ops/quasar/gpt_oss/conftest.py#L44-L108), [conftest.py:111-158](models/experimental/ops/quasar/gpt_oss/conftest.py#L111-L158), [conftest.py:150-158](models/experimental/ops/quasar/gpt_oss/conftest.py#L150-L158)).
- **R3-F9.** The same as R2-F7, plus "zero overhead when off":
  - **Fixed.** The docstring now says "when off, each call costs one environment lookup" ([pcc_log.py:7](models/experimental/ops/quasar/gpt_oss/utils/pcc_log.py#L7-L7)).
- **R3-F10.** The same as R1-F14: **Fixed.**
- **R3-F11.** The same as R1-F1 (the implicit-sync paragraph), plus "L1" used for Tensix SRAM in the `sparse_matmul` companion:
  - **Fixed.** The file no longer says "L1".
- **R3-F12. Row G of the ops analysis report implied that the demo passed with the (6, 3) grid.**
  - **Fixed.** The row now says that every demo run used (5, 3), and that the demo has not run with (6, 3) ([gpt_oss_ops_analysis.md:153](models/experimental/ops/quasar/gpt_oss/gpt_oss_ops_analysis.md#L153-L153)).

## 3. Unresolved issues

No reviewer reached its limit with an open issue. One issue is left for the branch owner to decide:

- **R2-F3, low severity. The three reports cite about 40 paths under `/localdev/fplavec/temp/`**, which other readers cannot open. Some of these folders no longer exist ([gpt_oss_ops_analysis.md:291](models/experimental/ops/quasar/gpt_oss/gpt_oss_ops_analysis.md#L291-L291)).
  - Reviewer 2's position: committed docs should not depend on local files that decay. Either move the needed files to a shared place, or describe them without local paths.
  - Parent's position: the reports are run records of the owner, and the text does not depend on the reader opening the paths. Whether to keep the paths is the owner's choice.

## 4. Tests and experiments

- **Host-only:** `element_size()` on a `bfloat8_b` tensor raised `ValueError`, and `spec.compute_packed_buffer_size_bytes()` worked for bfloat16, `bfloat8_b` and `bfloat4_b`.
- **Host-only:** `test_fit_core_grid.py`, 80 passed.
- **Wormhole n150:**
  - `test_modules.py::test_decoder[decode_low_latency]`, paged and unpaged: 2 passed. These cases run the `bfloat8_b` decode MoE path, which needs R2-F1's fix.
  - `test_model[prefill_b1_s128, 2_layers, trained]` with `GPT_OSS_PCC_LOG=1`: passed, with the same PCC values as before the `pcc_log.py` changes (`layer1.mlp` 0.9021, logits 0.9813).
- **Quasar on ttsim v1.16.0, 8x4 and 2x1 grids.** These ran with PR #63905 applied, which was removed from the working tree afterwards.
  - `test_sparse_matmul_small_grid.py`: 8x4: 2 passed (cases 01 and 03 on (6, 3); 01 with an SRAM output), 2 skipped. 2x1: 4 passed (01 with a DRAM output).
  - `test_nlp_concat_heads_decode_small_grid.py`: 2 passed on each grid.
  - The two SDPA companion tests: 2 passed each, on each grid.
  - `test_nlp_concat_heads_decode.py`: skipped on both grids (64 heads need more cores).
- **A first run of the concat and SDPA companions on ttsim failed** with "UnimplementedFunctionality: t_tile_mmio_wr32". #63905 had been taken out of the source tree, but the host library was still built with it. After a rebuild with #63905 applied again, the same tests passed.
- **gpt-oss demo on ttsim after the fixes, with #63905 applied, on 2026-11-23:** 2 layers, 128-token prefill, 3 decode tokens, all ops on the device.
  - 8x4 with the (6, 3) grid for the down projections: passed in 2 h 5 min.
  - 2x1: passed in 2 h 35 min. This run used the fixed size check of R2-F1, which put the decode down projection output in DRAM.
  - The logs are in `/localdev/fplavec/temp/gpt_oss_ops_runs/20261123_run15_demo_ttsim/`.
- **Not run after the fixes:** the full op suites, and `sparse_matmul` case 00 on a (1, 1) grid.
- **Build state:** the current build contains #63905, and #63905 was applied to the working tree again from `stash@{0}` for the demo runs.

## 5. Changed files

All fixes are uncommitted edits in the working tree.

Changed (`git diff --stat 9b18a87485f`):
- `models/experimental/ops/quasar/gpt_oss/conftest.py`
- `models/experimental/ops/quasar/gpt_oss/gpt_oss_ops_analysis.md`
- `models/experimental/ops/quasar/gpt_oss/gpt_oss_small_grid_analysis.md`
- `models/experimental/ops/quasar/gpt_oss/tests/unit/test_modules.py`
- `models/experimental/ops/quasar/gpt_oss/tt/experts/config.py`
- `models/experimental/ops/quasar/gpt_oss/tt/experts/decode.py`
- `models/experimental/ops/quasar/gpt_oss/tt/mlp.py`
- `models/experimental/ops/quasar/gpt_oss/utils/pcc_log.py`
- `models/experimental/ops/quasar/tests/gpt_oss_ops/test_nlp_concat_heads_decode_small_grid.py`
- `models/experimental/ops/quasar/tests/gpt_oss_ops/test_paged_scaled_dot_product_attention_decode_small_grid.py`
- `models/experimental/ops/quasar/tests/gpt_oss_ops/test_scaled_dot_product_attention_small_grid.py`
- `models/experimental/ops/quasar/tests/gpt_oss_ops/test_sparse_matmul_small_grid.py`

New:
- `models/experimental/ops/quasar/gpt_oss/utils/merge_heads.py`
- `models/experimental/ops/quasar/gpt_oss/tests/unit/test_fit_core_grid.py`
