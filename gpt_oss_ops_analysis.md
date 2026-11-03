# gpt-oss op tests on Quasar (craq-sim): results

Test suite: `models/experimental/ops/quasar/tests/gpt_oss_ops/` (119 cases in 38 files, generated from a graph capture of the gpt-oss-20b demo). The tests call the ops directly. The Quasar patches in `models/experimental/ops/quasar/gpt_oss/conftest.py` do not apply to them.

Test cases are named by their pytest ID without the `-ttnn_mesh_device0` that pytest appends, for example `test_matmul[04_32x2880_bf8_int-dram]`. The ID is built from the case number in its file (cases are sorted by how often the model made that call, most frequent first), then the shape, dtype and memory layout of the op's first input as captured (`int` interleaved, `hs` height sharded, `ws` width sharded; `l1` for SRAM, `dram`). The dtype in the ID stays `bf8` when a case runs as bfloat16. To run one case: `pytest models/experimental/ops/quasar/tests/gpt_oss_ops/test_matmul.py -k 04_32x2880`.

Terms: a DFB (dataflow buffer) is a buffer in a core's SRAM that passes tiles from one kernel to another. The NoC is the network on chip that connects the cores and DRAM. A "mainline" op is the regular ttnn op. Its "Quasar copy" is the version in `ttnn.experimental.quasar`. A program factory is the host code of an op that creates the op's kernels and DFBs; one op can have several factories, for example one for interleaved and one for sharded inputs. Multicast means that one core sends the same data to several cores in one NoC transfer. In the 1D multicast matmul, one core reads input A and multicasts it to the other cores. With implicit sync, the default for the DFBs of data movement kernels on Quasar, a NoC read into a DFB or a NoC write out of it also updates the DFB's counts, with help from an interrupt handler in the firmware, so the kernel code is simpler ([dataflow_buffer.inl:883-886](tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl#L883-L886), [dataflow_buffer.inl:936-939](tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl#L936-L939)). With explicit sync, the data movement kernel updates the counts itself with calls such as `reserve_back`, `push_back`, `wait_front` and `pop_front` ([data_movement_hardware_config.hpp:91-100](tt_metal/api/tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp#L91-L100)). Compute kernels always make the `wait_front` and `pop_front` calls themselves.

# Current state (run 6 of 2026-10-30, tests of PRs and local changes on 2026-11-02 and 2026-11-03)

## Setup

Run 6 used:

- branch `fplavec/gpt_oss_quasar_bringup` on `main` `f23b877dadb` (2026-10-30);
- craq-sim built on 2026-10-30 from its `quasar` branch at `450de768`, in `/localdev/fplavec/temp/qsr-sim`, with 3 GiB of DRAM per channel (`TTSIM_DRAM_CHANNEL_SIZE=3G`, and the same size in `soc_descriptor.yaml`);
- bfloat16 for the block float cases (`TTNN_GRAPH_OPS_BF8_AS_BF16=1`);
- `/localdev/fplavec/temp/run_ops_each.sh`: one pytest process per case, 10 min per case.

Run 6 took 71 min. Output: `/localdev/fplavec/temp/gpt_oss_ops_runs/20261030_bf16_newsim/` (`summary.tsv` and one log per case).

On 2026-11-03, the branch was rebased onto `main` `8b53dfa87b6` (2026-11-03), which has PR #63023. The tests of 2026-11-02 and 2026-11-03 ran on the old base `f23b877dadb`, except the reruns in `20261103_rebased_8b53dfa/` and the tests in "Model run with the device embedding", which ran on the new base.

To rerun, from the tt-metal root:

```bash
TTNN_GRAPH_OPS_BF8_AS_BF16=1 /localdev/fplavec/temp/run_ops_each.sh \
    models/experimental/ops/quasar/tests/gpt_oss_ops /localdev/fplavec/temp/gpt_oss_ops_runs/<new folder> 10m
```

The script sets the craq-sim environment, including `TTSIM_DRAM_CHANNEL_SIZE=3G`. A new simulator build comes with a `soc_descriptor.yaml` that says 1 GiB, so after a new build, set `dram_bank_size` and `dram_view_size` in `/localdev/fplavec/temp/qsr-sim/soc_descriptor.yaml` to `3221225472` again.

## Results

| Result | Runs 3 + 4 (`main` `345aebfbe8d`) | Run 5 (`main` `f23b877dadb`, older craq-sim) | Run 6 (`main` `f23b877dadb`, craq-sim `450de768`) |
|---|---|---|---|
| Passed | 63 | 67 | 79 |
| Failed | 45 | 20 | 35 |
| Timeout | 8 | 2 | 3 |
| Simulator error | 1 | 28 | 0 |
| Skipped | 2 | 2 | 2 |
| **Total** | **119** | **119** | **119** |

The letters name the problems in "Open problems" below. J, C and F are fixed; the history describes them.

- **J is fixed.** The new simulator allows the longer MOP loops (hardware loops of Tensix instructions) that tilize on `main` uses. Of the 28 cases that stopped on J in run 5, 12 pass. The other 16 now show their next error: B 5, K 7, G 2, I 1, E 1.
- **Also fixed since runs 3 + 4:** C (compile error, PR #61760), F (simulator error in SDPA, scaled dot product attention), and 5 of the 8 hangs of E (PR #62092).
- **K is a regression on `main`** (PR #61818, see the notes on K). It fails 3 cases that passed in runs 3 + 4: `test_linear[03_128x2880_bf8_int-dram]`, `test_matmul[00_1x2880_bf8_int-l1]` and `test_matmul[02_1x2880_bf8_int-dram]`.
- **K hid two earlier problems.** A case shows only its first error. `test_matmul[01_128x4096_bf8_int-dram]` and `test_matmul[04_32x2880_bf8_int-dram]` hung in runs 3 + 4 (E). `test_sparse_matmul[00_1x2880_bf8_int-l1]` and `test_sparse_matmul[02_4x32x2880_bf16_int-dram]` failed on the self-looped DFB (H). In run 6, all four stop on K before they reach their earlier error. With a local fix for K (tested on 2026-11-03, see the notes on K), both `matmul` cases pass, so their hang is gone on `f23b877dadb`, and both `sparse_matmul` cases fail on H.
- **All 3 timeouts have a known cause.** `fast_reduce_nc` (1 case) and `softmax` (2 cases) wait on a `DataflowBuffer` that a helper function from `ttnn/cpp/ttnn/kernel_lib/` creates for itself. Open PR #62204 fixes both hangs without kernel changes (tested). A two-line kernel change also fixes `fast_reduce_nc` (tested). For `softmax`, removing the hang shows a second problem: the output is all zeros, because the kernel does not point the packer at each new output DFB. Open PR #59572 adds the missing calls. With #62204 and #59572 together, both softmax cases pass (tested, after a fix to 3 lines of #59572 that do not compile; see the notes on softmax).
- `test_matmul[03_1x2880_bf8_int-l1]` (the 1-row `lm_head` matmul, input in SRAM) passes in 381 s, the longest passing case in run 6 (with the local fix for K, `test_matmul[04_32x2880_bf8_int-dram]` took 397 s). `matmul` picks the plain matmul (no multicast) for it, so K does not apply. Keep the timeout at 10 min per case or more.

## Open problems

The column "gpt-oss model" is from reading the model code, except where it cites the model run of 2026-11-03 (see "Model run with the device embedding"). It describes the model with two changes to `models/experimental/ops/quasar/gpt_oss/conftest.py` that the branch did not have on 2026-11-03 (they were in the git stash entry "sparse_matmul host bypass"): a route of `ttnn.untilize` to the Quasar copy, and `ttnn.sparse_matmul` on the host. Without these changes, `conftest.py` does not route `untilize`, and routes `sparse_matmul` to `ttnn.experimental.quasar.sparse_matmul`. In the column "gpt-oss model", "bypassed" means that the model runs the op on the host for now. A host bypass only lets the model run reach later problems; it is not a fix, so the problem still needs one. The column "Recommended action" names the PRs and issues for each problem, as of 2026-11-03. On 2026-11-02 and 2026-11-03, PRs #62204, #60806 (closed since), #59572 and #63023, local changes for D, K and softmax, and `ttnn.experimental.quasar.matmul` on the `matmul` cases, were tested on this branch. Each was applied as uncommitted changes and removed after the test (see the notes). Since the rebase, #63023 is in the branch through `main`. Open PRs #63516, #63401 and #62751 were not tested. "Recommendations in detail" below has the longer version.

| # | Problem | Ops affected | Cases in run 6 | gpt-oss model | Recommended action |
|---|---|---|---|---|---|
| A | Quasar has no block float formats (`Bfp8_b`, `Bfp4_b`) | 23 ops whose captured calls use `bfloat8_b` or `bfloat4_b`: `add`, `clamp`, `embedding`, `fast_reduce_nc`, `linear`, `matmul`, `multiply`, `paged_fill_cache`, `paged_update_cache`, `permute`, `reshape`, `rms_norm`, `scatter`, `sigmoid`, `slice`, `sparse_matmul` (also `bfloat4_b`), `sum`, `transpose`, `typecast`, `unsqueeze`, `unsqueeze_to_4D`, `untilize`, `zeros_like` | none, because the cases run as bfloat16 (38 failed as captured, in run 2) | Not affected: the model runs these ops as bfloat16 (`quasar_block_float_to_bf16`, [conftest.py:44-104](models/experimental/ops/quasar/gpt_oss/conftest.py#L44-L104)) | Keep bfloat16. Later, use MX formats (issue #58157). Open PR #62751 adds MX to the Quasar copy of `matmul`, to `reduce` and to the normalization ops, so it covers only some of the 23 ops; the other ops still need MX support. |
| B | The mainline binary op implementation, `binary_ng`, builds legacy `DataMovementKernel` kernels, which Quasar does not accept | `add`, `multiply`, and `linear` (its bias add runs in `binary_ng`) | 19: all 10 cases of `test_add` that are not skipped, all 7 of `test_multiply`, `test_linear[01_1x2880_bf8_int-l1]` and `[02_128x2880_bf16_int-dram]` | Not affected: `quasar_ops` routes `add`, `mul`, `multiply`, `subtract`, `sub` and `linear` to the Quasar copies ([conftest.py:107-145](models/experimental/ops/quasar/gpt_oss/conftest.py#L107-L145)) | Nothing needed for the model. For the op tests, call the Quasar copies. No PR found that gives the mainline `binary_ng` a Quasar factory. |
| D | tt-metal rejects `UInt16` and `UInt32` as DFB formats on Quasar. It accepts other formats of the same sizes: `Int16` and `RawUInt16` (16 bits), `Int32` and `RawUInt32` (32 bits) ([tt_backend_api_types.cpp:96-122](tt_metal/common/tt_backend_api_types.cpp#L96-L122)). | `topk` (DFB `index`); `scatter`, which fails in `untilize_with_unpadding` (DFB `mci_in`); `tensor_getitem`, which fails in `slice` (DFB `rm_in`) | 5: `test_topk[00_1x32_bf16_int-l1]` and `[01_128x32_bf16_int-l1]`, `test_scatter[00_1x32_bf8_int-l1]` and `[01_128x32_bf16_int-l1]`, `test_tensor_getitem[01_1x32_u32_int-dram]` | Affected: the MoE router (the part of each layer that picks the experts for each token) calls `topk` ([topk.py:26](models/experimental/ops/quasar/gpt_oss/tt/topk.py#L26)) and `scatter` ([topk.py:47](models/experimental/ops/quasar/gpt_oss/tt/topk.py#L47)). `tensor_getitem`: the model runs this slice of the token IDs in decode, before the token embedding, which runs on the device since 2026-11-03 (see "Model run with the device embedding"). With #63023, the slice works (tested). | `topk`: open PR #63516 (not tested). `tensor_getitem`: PR #63023 (`slice`) makes the case pass (tested). #63023 merged on 2026-11-03, and the branch has it since the rebase onto `main` `8b53dfa87b6`. `scatter`: no PR. Declaring the `UInt16` DFBs of the interleaved `untilize_with_unpadding` factory as `Int16` on Quasar, as #63516 does for `topk`, makes both cases pass with correct values (tested; see the D notes). Open PR #63401 does not change these DFB formats. |
| E | Hang in `fast_reduce_nc`: a `kernel_lib` helper that takes only a DFB id waits for that DFB to empty before it returns, but the DFB can only empty after the helper returns (see the E notes) | `fast_reduce_nc`. The `matmul` hangs of runs 3 + 4 do not occur with a local fix for K (see "E for the multicast matmul"). | 1: `test_fast_reduce_nc[00_32x128x2880_bf8_int-dram]` | Affected: `fast_reduce_nc` sums the expert outputs ([experts/operations.py:66](models/experimental/ops/quasar/gpt_oss/tt/experts/operations.py#L66)). The model's multicast matmuls, the `lm_head` with its input in DRAM (`test_matmul[04_...]`) and the prefill attention output projection (`test_matmul[01_...]`), pass with the local fix for K. | Open PR #62204, which moves the wait for the consumer (the DFB drain) from the `DataflowBuffer` destructor to the kernel exit, fixes this hang without a kernel change (tested). A two-line kernel change also fixes it (tested; no PR). |
| G | Op needs more cores than the 8x4 device has | `nlp_concat_heads_decode` (one core per head, 64 heads); `sparse_matmul` (the captured program config uses core row 5, and the device has rows 0 to 3) | 3: `test_nlp_concat_heads_decode[00_64x64_bf16_hs-l1]`, `test_sparse_matmul[01_32x1x2880_bf8_int-l1]` and `[03_32x128x2880_bf8_int-dram]` | Affected in decode: `nlp_concat_heads_decode` ([decode.py:164](models/experimental/ops/quasar/gpt_oss/tt/attention/decode.py#L164)). `sparse_matmul` is bypassed for now (it runs on the host). | No PR found. `nlp_concat_heads_decode` needs a change so that one core can take several heads; until then, use llama's workaround. `sparse_matmul`: give the two test cases, and the model, a program config that fits 8x4. |
| H | Self-looped DFB: one `sparse_matmul` kernel is both the producer and the consumer of DFB `sparsity`, which Quasar rejects | `sparse_matmul` | none in run 6, because `test_sparse_matmul[00_...]` and `[02_...]` stop on K first. With a local fix for K, both fail on H (tested). | Bypassed for now: `ttnn.sparse_matmul` runs on the host (`make_host_sparse_matmul`, one of the two `conftest.py` changes described above the table) | No PR found. Replace the self-looped DFB with a scratchpad (SRAM that only one kernel uses, [scratchpad_spec.hpp:22-24](tt_metal/api/tt-metalium/experimental/metal2_host_api/scratchpad_spec.hpp#L22-L24)), as merged PR #63023 does for `slice`. |
| I | `untilize` gives a DFB 265 entries; with implicit sync, the DFB descriptor holds this count in an 8-bit field (at most 255) | `untilize` (mainline block factory) | 1: `test_untilize[00_32x262144_bf8_int-dram]` | Not affected for `ttnn.untilize` calls: they run as `ttnn.experimental.quasar.untilize` (`quasar_ops`). `ttnn.to_layout` still runs the mainline `untilize` and `untilize_with_unpadding` ([to_layout_op.cpp:132](ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp#L132), [to_layout_op.cpp:181](ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp#L181)) (see the D notes on `scatter`). On one device, the model uses the experts in `tt/experts/` ([mlp.py:51-52](models/experimental/ops/quasar/gpt_oss/tt/mlp.py#L51-L52)), and its only `ttnn.to_layout` call there, [experts/decode.py:57](models/experimental/ops/quasar/gpt_oss/tt/experts/decode.py#L57), converts a [1, 1, 1, 32] tensor. Its case, `test_to_layout[00_1x32_bf16_int-l1]`, passed in run 6. | No PR found. Issue #62783 (open, assigned) tracks it; its author now proposes to fix the limit in the DFB code, because, in the author's view, the 8-bit fields are a software limit, and struck out the other options. Until that fix, turning off implicit sync in the mainline `untilize` factories is a stopgap; merged PR #62730 made that change in the interleaved `untilize_with_unpadding` factory. |
| K | The multicast helpers reject the matmul's NoC on Quasar | `matmul` and `linear` (1D multicast matmul); `sparse_matmul` | 7: `test_matmul[00_...]`, `[01_...]`, `[02_...]` and `[04_...]`, `test_linear[03_128x2880_bf8_int-dram]`, `test_sparse_matmul[00_...]` and `[02_...]` | Affected: the model calls the mainline `ttnn.matmul` in attention ([operations.py:114](models/experimental/ops/quasar/gpt_oss/tt/attention/operations.py#L114), [decode.py:65](models/experimental/ops/quasar/gpt_oss/tt/attention/decode.py#L65)) and in the `lm_head` ([model.py:422](models/experimental/ops/quasar/gpt_oss/tt/model.py#L422)) when the `lm_head` input is in DRAM. With the `lm_head` input in SRAM, `matmul` picks the plain matmul, which K does not affect (`test_matmul[03_1x2880_bf8_int-l1]`). The model run of 2026-11-03 stopped on K at the attention output projection of the first prefill. The model's `linear` runs as the Quasar copy, so it is not affected. The model's `sparse_matmul` is bypassed for now (it runs on the host). | Issue #63758 (open) tracks K for the 1D multicast matmul; no issue covers `sparse_matmul`. No PR fixes K: PR #63576, which would have reverted #61818, was closed. A local fix removes K for `matmul`, `linear` and `sparse_matmul` (tested; see the K notes). Post it in #63758 and make a PR. Until then, the model can route `ttnn.matmul` to the Quasar copy, which passes all 5 `test_matmul` cases (tested). |
| softmax | Hang in the softmax kernels (C hid it until run 5); with the hang removed, the output is all zeros | `softmax` (attention-optimized factory) | 2: `test_softmax[00_1x4_bf16_int-l1]` and `[01_128x4_bf16_int-l1]` | Affected: the router calls `softmax` ([topk.py:41-43](models/experimental/ops/quasar/gpt_oss/tt/topk.py#L41-L43)) | Needs two fixes. Hang: open PR #62204 fixes it (tested), and so does the kernel change in the softmax notes (tested). All-zero output: issue #60745 (open) tracks it. The packer (the hardware unit that writes compute results into a DFB) keeps writing to the DFB of the last `pack_init` call. Open PR #59572 adds a `pack_init` call at each change of output DFB, as the LLK owners require. With a fix to 3 lines of #59572 that do not compile, #59572 makes both cases pass together with #62204, and also together with the kernel change (both tested). |

## Notes on the open problems

### E: `fast_reduce_nc` hangs in an id-only helper; PR #62204 or a two-line kernel change fixes it (both tested)

- In run 6, the case built only the three `fast_reduce_nc` kernels (`reader_reduce_nc`, `reduce_nc`, `writer_reduce_nc`) and then hung. So the hang is in the op, not in the upload of the input.
- The reader puts a zero tile into DFB `in1` with the id-only form `prepare_zero_tile<dfb::in1>()`, which takes only the DFB id, not a `DataflowBuffer` object ([reader_reduce_nc.cpp:33-35](ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reader_reduce_nc.cpp#L33-L35)). That form builds its own `DataflowBuffer` ([l1_helpers.hpp:73-78](ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp#L73-L78)). On Quasar, a `DataflowBuffer` that goes out of scope waits until the consumer has popped everything that was pushed into the DFB (the DFB drain, in the destructor: [dataflow_buffer.inl:532-542](tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl#L532-L542)).
- The compute kernel waits for the zero tile once and pops it only after the whole reduction ([reduce_nc.cpp:28](ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp#L28), [reduce_nc.cpp:53-55](ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp#L53-L55)). The reduction needs the input tiles, and the reader sends them only after the helper returns. So the reader and the compute kernel wait for each other.
- PR #62092 (the fix for #60848) added forms of the helpers that take the kernel's `DataflowBuffer`. It kept the id-only forms for kernels that are not migrated yet, and marks them deprecated on Quasar ([id_only_helpers.hpp:7-21](ttnn/cpp/ttnn/kernel_lib/id_only_helpers.hpp#L7-L21)). `reader_reduce_nc.cpp` is not migrated, and its Quasar compile prints the deprecation warning (run 5 log).
- Test: with `DataflowBuffer cb_in1_obj(dfb::in1);` next to `cb_in0_obj`, and `prepare_zero_tile(cb_in1_obj)` instead of the id-only call, `test_fast_reduce_nc[00_32x128x2880_bf8_int-dram]` passed in 8 s. Log: `/localdev/fplavec/temp/gpt_oss_ops_runs/fast_reduce_nc_kernel_scope_dfb.log`. The change is not on the branch; the kernel file was restored after the test.
- Open PR #62204 ("Quasar DFB: finish DFBs in firmware at kernel exit") removes the wait from the `DataflowBuffer` destructor and drains each DFB in firmware when the kernel exits, "so copies, temporaries and helper-local DataflowBuffer objects no longer drain early".
- Test of #62204 (2026-11-02): I applied the firmware and host changes of #62204 (head `2aa66bb4e2e`, without its test files) to the branch as uncommitted changes and rebuilt. With `reader_reduce_nc.cpp` unchanged, `test_fast_reduce_nc[00_32x128x2880_bf8_int-dram]` passed in 21 s. The kernel compile still printed the deprecation warning, so the kernel ran with the id-only call. Logs: `/localdev/fplavec/temp/gpt_oss_ops_runs/20261102_pr62204/`.
- With #62204, the 6 cases that pass since PR #62092 still pass: the 5 hangs of E that #62092 fixed (both `test_paged_scaled_dot_product_attention_decode` cases, `test_scaled_dot_product_attention[01_64x128x64_bf16_int-dram]`, `test_sum[00_32x2880_bf8_int-l1]` and `test_paged_update_cache[00_64x8x64x64_bf8_int-dram]`), and the F case `test_scaled_dot_product_attention[00_64x128x64_bf16_int-dram]`, which passes since run 5. The other cases of the suite were not run with #62204.
- Test of the current head of #62204 (2026-11-03, head `eab6fdae94f`; the PR was rebased onto a newer `main`, and has 2 more commits after the one that matches `2aa66bb4e2e`; `/localdev/fplavec/temp/gpt_oss_ops_runs/20261103_pr62204_eab6fdae_pr59572/`): `test_fast_reduce_nc[00_...]` passed in 21 s, and the 6 cases of the previous bullet passed again. This run also had the #59572 changes to the softmax kernels (see the softmax notes), which these 7 cases do not use.
- #62204 is not on the branch; the PR changes were removed after the tests and the branch was rebuilt.
- Other kernels in this suite whose Quasar compile prints the same warning: the reader and compute kernel of the attention-optimized softmax factory (see the softmax notes), and the compute kernels `tilize_wh.cpp`, `untilize_metal2.cpp` and `untilize_variable_num_blocks_metal2.cpp`. These three compute kernels did not hang. `tilize_wh.cpp` runs in cases that pass, such as `test_sum[00_32x2880_bf8_int-l1]` and `test_matmul[03_1x2880_bf8_int-l1]`. The two untilize kernels run in `test_scatter[00_...]` and `[01_...]`, which pass with the `Int16` change tested in the D notes.

### E for the multicast matmul: not seen with the local fix for K

- `test_matmul[01_128x4096_bf8_int-dram]` and `test_matmul[04_32x2880_bf8_int-dram]` hung in runs 3 + 4, before PR #61818 moved the matmul factories to the multicast helpers. In run 6, both stop on K.
- With the local fix for K (see the K notes), case 01 passed in 34 s and case 04 in 397 s (2026-11-03, `/localdev/fplavec/temp/gpt_oss_ops_runs/20261103_kfix_int16_pr63023/`). Both cases ran the 1D multicast matmul: in run 6, both failed on K, which the multicast helpers raise from the 1D multicast factory. Which change after runs 3 + 4 removed the hang was not tested.
- Case 01 is the prefill attention output projection: the only matmul in the model with a [4096, 2880] weight is `o_proj` in `apply_output_projection` ([operations.py:101-114](models/experimental/ops/quasar/gpt_oss/tt/attention/operations.py#L101-L114)), and only the prefill path calls it ([prefill.py:179](models/experimental/ops/quasar/gpt_oss/tt/attention/prefill.py#L179)). Case 04 is the `lm_head` ([model.py:422](models/experimental/ops/quasar/gpt_oss/tt/model.py#L422)) with its input in DRAM.

### softmax: the same kind of wait, and a packer problem; PRs #62204 and #59572 together fix both (tested, with a fix to #59572)

- The kernels of the attention-optimized softmax factory, which both `test_softmax` cases and the gpt-oss router use, have the same kind of wait as `fast_reduce_nc`, in two places:
  - The reader fills both reduce scalers with the id-only `calculate_and_prepare_reduce_scaler<...>()` ([reader_unary_interleaved_sm.cpp:69-80](ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/dataflow/reader_unary_interleaved_sm.cpp#L69-L80)). The compute kernel waits for the scalers at its start and pops them only at its end ([softmax.cpp:183-184](ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp#L183-L184), [softmax.cpp:408-409](ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp#L408-L409)).
  - The compute kernel calls the id-only `compute_kernel_lib::reduce<...>()` twice. The max reduce writes DFB `max` ([softmax.cpp:46-53](ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp#L46-L53)), and the sum reduce writes DFB `recip_sum_exps` ([softmax.cpp:360-366](ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp#L360-L366)). The same compute kernel reads each of these two DFBs later ([softmax.cpp:56-66](ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp#L56-L66), [softmax.cpp:384](ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp#L384)). So the helper's own `DataflowBuffer` for the output waits for a pop that can only come after the helper returns.
- Tests (logs in `/localdev/fplavec/temp/gpt_oss_ops_runs/`):
  - With only the reader changed (kernel-scope `DataflowBuffer` objects for the two scalers), `test_softmax[00_1x4_bf16_int-l1]` still did not finish. The case ran 20 million simulated cycles in 8 min, while the 262144-wide `lm_head` matmul in `test_matmul[03_1x2880_bf8_int-l1]` needs 7.7 million, so the case hung and was not only slow (`softmax_kernel_scope_dfb.log`).
  - With the reader changed and both `reduce` calls given the kernel's existing `DataflowBuffer` objects (`dfb_in_obj`, `dfb_max_scaler_obj` and `dfb_max_obj` for the max reduce; `dfb_exps_obj`, `dfb_sum_scaler_obj` and `dfb_recipsumexps_obj` for the sum reduce), both cases finished in 10 s, but the output was all zeros (PCC 0.0: no correlation with the PyTorch reference) (`softmax_kernel_scope_dfb_reader_and_compute.log`).
  - Neither kernel change is on the branch; the kernel files were restored after the tests.
- The all-zero output comes from the packer: on Quasar, `pack_init` sets the DFB that the packer writes, and `pack_tile` does not change it. The compute kernel changes its output DFB several times without a new `pack_init`, so the packer keeps writing to the DFB of the last `pack_init` call. Issue #60745 (open) tracks this for softmax.
- PR #60806 ("Quasar: pack_tile re-aims the packer when the kernel packs into a different DFB") fixed the packer problem in the LLK (the low-level kernel library that drives the compute hardware), for all kernels. Its author closed it on 2026-11-02 without merging, after the LLK owners said in #60745 that on Quasar a kernel must call `pack_init` for each new output DFB ("always re-init for every new dfb"). The LLK owners also said that later they will try to hide as much of these calls as possible in the compiler.
- Open PR #59572 ("softmax: Quasar uplift of the attention and general-H factories") adds a `pack_init` on Quasar at every change of the output DFB in the attention compute kernel `softmax.cpp`. #59572 does not change the id-only calls in the first bullet of this section. Its head `a193b2a575e` does not compile on Quasar: in `cycle_dfb_pad` and `drain_dfb_pad`, 3 calls to `dummy_pack` and `dummy_unpack` use the name `dfb_id`, which these functions do not declare. The PR's own base has the same form of these functions, so the error is in the PR, not from applying it to this branch. `dfb.get_id()` in these 3 calls fixes the compile error. The #59572 changes to the other softmax kernels and to `moreh_softmax` do not apply to this branch; the two test cases do not use those kernels.
- Tests (logs in `/localdev/fplavec/temp/gpt_oss_ops_runs/`). PRs and kernel changes were applied as uncommitted changes and removed after the tests. The cases run at the op's default fidelity, because `test_softmax.py` drops the captured `compute_kernel_config`.
  - With #62204 (applied as in the notes on E) and both kernels unchanged, both cases finished in 20 s or less, with all-zero output (PCC 0.0), as with the reader and compute kernel change above. So #62204 fixes the hang (2026-11-02, `20261102_pr62204/`).
  - With #62204 and #60806 (head `dfa01f18228`), both cases passed (2026-11-02, `20261102_pr62204_pr60806/`). So the packer problem causes the all-zero output.
  - With #62204 and the #59572 changes to the two kernels (with the compile fix), both cases passed, with #62204 at head `2aa66bb4e2e` (2026-11-03, `20261103_pr62204_pr59572/`) and at its current head `eab6fdae94f` (2026-11-03, `20261103_pr62204_eab6fdae_pr59572/`). The kernels still used the id-only helpers: their compile printed the deprecation warning.
  - With the #59572 changes to the two kernels (with the compile fix) and the reader and compute kernel change above, on the branch build without #62204, both cases passed (2026-11-03, `20261103_softmax_pr59572_kernel_change/`).
  - The other cases of the suite were not run with these changes.

### K: the multicast helpers and the two matmul factories disagree about the NoC on Quasar

The cause below is from reading the code. A local fix based on it was tested on 2026-11-03 (see the end of this section).

- **`ttnn.matmul`** (1D multicast, Metal 2.0 path):
  - The in0 sender kernel (the kernel that reads input A, `in0`, and multicasts it) gets `config_1xx.noc = in0_noc` ([matmul_multicore_reuse_mcast_1d_program_factory.cpp:3855-3860](ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp#L3855-L3860)). `in0_noc` is `NOC_1` on every architecture ([matmul_multicore_reuse_mcast_1d_program_factory.cpp:3618-3619](ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp#L3618-L3619), [kernel_types.hpp:141-153](tt_metal/api/tt-metalium/kernel_types.hpp#L141-L153)). The factory's own comment says that `config_1xx` is ignored on Quasar ([matmul_multicore_reuse_mcast_1d_program_factory.cpp:3843-3844](ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp#L3843-L3844)).
  - The multicast configuration uses `NOC_0` on Quasar ([matmul_multicore_reuse_mcast_1d_program_factory.cpp:4521-4524](ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp#L4521-L4524)).
  - The helper reads the kernel's NoC with `spec_noc()`, which returns `config_1xx->noc` whenever `config_1xx` is set, also on Quasar ([mcast_spec_adapter.cpp:132-138](ttnn/cpp/ttnn/kernel_lib/mcast/host/program_adapters/mcast_spec_adapter.cpp#L132-L138)). So it compares `NOC_1` with `NOC_0`, and the check fails ([mcast_spec_adapter.cpp:235-237](ttnn/cpp/ttnn/kernel_lib/mcast/host/program_adapters/mcast_spec_adapter.cpp#L235-L237)).
- **`ttnn.sparse_matmul`:** the factory sets the multicast NoC to `in0_noc`, which is `NOC_1`, also on Quasar ([sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:1070-1075](ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L1070-L1075)). The helper rejects `NOC_1` on Quasar ([mcast_impl.cpp:505-507](ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_impl.cpp#L505-L507)).
- Both factories moved to the multicast helpers in PR #61818 (merged 2026-10-30).
- Issue #63758 (open, filed 2026-11-02) reports the `ttnn.matmul` error, found through `ttnn.experimental.quasar.outer`, which calls the mainline `matmul`. Merged PR #63891 fixed the same check in the 2D multicast matmul factory: there, the kernels now get `NOC_0` in `config_1xx` on Quasar. #63891 does not change the 1D factory or `sparse_matmul`. No issue covers the `sparse_matmul` error, and no PR fixes either error (searched on 2026-11-03).
- PR #63576, which would have reverted #61818 because Stable Diffusion XL hangs on Wormhole since #61818, was closed on 2026-11-03 without merging. Its replacement, open PR #63950, changes only the kernel code of the multicast helpers, for Wormhole, and does not change the NoC checks.
- Test of a fix (2026-11-03, `/localdev/fplavec/temp/gpt_oss_ops_runs/20261103_kfix_int16_pr63023/`): `spec_noc()` takes the architecture and ignores `config_1xx` on Quasar, so it returns `NOC_0` there. The helper class already stores the architecture (`prepared_arch_`, [mcast_impl.hpp:128](ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_impl.hpp#L128)). Also, the `sparse_matmul` factory sets the multicast NoC to `NOC_0` on Quasar, as the matmul factory does. With these two changes:
  - all 5 `matmul` and `linear` cases of K passed: `test_linear[03_128x2880_bf8_int-dram]` (39 s), `test_matmul[00_1x2880_bf8_int-l1]` (20 s), `[01_...]` (34 s), `[02_1x2880_bf8_int-dram]` (19 s) and `[04_...]` (397 s);
  - both `sparse_matmul` cases failed on H (see the next section).

  The change is not on the branch. A fix in the factories only, as #63891 does for the 2D factory, would need the same change in the 1D factory and in the `sparse_matmul` factory; that form was not tested.
- The Quasar copy of the matmul, `ttnn.experimental.quasar.matmul`, does not use the multicast helpers, so K does not apply to it. On the branch, it passes all 5 cases of `test_matmul` (2026-11-03):
  - the three attention cases `test_matmul[00_...]`, `[01_...]` and `[02_...]`, in 36 s or less (`/localdev/fplavec/temp/gpt_oss_ops_runs/20261103_quasar_matmul_attn/`);
  - the two 262144-wide `lm_head` cases, `test_matmul[03_1x2880_bf8_int-l1]` in 283 s and `[04_32x2880_bf8_int-dram]` in 412 s (`20261103_quasar_matmul_lm_head/`).

  The history reports that the Quasar copy hung at a 16384-wide output. That result is from the matmul experiments of 2026-10-20, on `main` `345aebfbe8d` with an older craq-sim, where the mainline multicast matmul also hung. Both 262144-wide cases pass on this branch, so that result is out of date.

### H is still in the code

The `sparse_matmul` in0 sender still binds DFB `sparsity` as both producer and consumer ([sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:592-603](ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L592-L603)). With the fix for K tested in the K notes, `test_sparse_matmul[00_1x2880_bf8_int-l1]` and `[02_4x32x2880_bf16_int-dram]` fail on H: "DataflowBuffer 'sparsity' is self-looped by data-movement kernel 'in0_sender'".

### D: PRs and tests

- **`topk`:** PR #63516 (open) declares the 16-bit index DFBs as `Int16` on Quasar and makes Quasar always use the single-core factory. It targets the error of `test_topk[00_...]` and `[01_...]`. Not tested on this branch.
- **`tensor_getitem`:** the failing DFB `rm_in` belongs to the row-major `slice` factory. PR #63023 (merged on 2026-11-03; in the branch since the rebase onto `main` `8b53dfa87b6`) declares `UInt16` and `UInt32` slice DFBs as `RawUInt16` and `RawUInt32` on Quasar, in that factory too. With #63023 applied to the branch on the old base, `test_tensor_getitem[01_1x32_u32_int-dram]` passed, and `test_tensor_getitem[00_...]`, `[02_...]` and `test_slice[00_128x2880_bf8_int-dram]` still passed (2026-11-03, `/localdev/fplavec/temp/gpt_oss_ops_runs/20261103_kfix_int16_pr63023/`). On the rebased branch, all 3 `test_tensor_getitem` cases, all 3 `test_embedding` cases and `test_slice[00_...]` pass (2026-11-03, `/localdev/fplavec/temp/gpt_oss_ops_runs/20261103_rebased_8b53dfa/`). The note in the history that `ttnn.experimental.quasar.slice` does not help is about the Quasar copy of `slice`, not about the mainline op that #63023 changes.
- **`scatter`:** the error does not come from `scatter` itself. `ttnn.scatter` converts its `UInt16` index tensor to row major with `ttnn.to_layout`, which runs `untilize_with_unpadding`. The interleaved factory of `untilize_with_unpadding` creates its input DFB `mci_in` with the tensor's own format, `UInt16` ([untilize_with_unpadding_multi_core_interleaved_program_factory.cpp:48-51](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_interleaved_program_factory.cpp#L48-L51)). PR #63401 (open, `untilize_with_unpadding` uplift, which ports the op to Quasar) changes the sharded factory and the writer kernel of the block factory (`writer_unary_stick_layout_wh_multicore_metal2.cpp`, the block writer), but not the data formats of the interleaved factory, so it does not fix `test_scatter`.
- **Test of `Int16` in `untilize_with_unpadding`** (2026-11-03, `/localdev/fplavec/temp/gpt_oss_ops_runs/20261103_kfix_int16_pr63023/` and `20261103_kfix_int16_pr63023_scatter_values/`): in the interleaved factory, the input and output DFB formats were changed from `UInt16` to `Int16` on Quasar. With this change:
  - both `test_scatter` cases passed. This suite has no PyTorch reference for `scatter`, so these cases check only the shape, type and finite values of the output.
  - a separate check of `scatter` with the router's pattern (4 different indices in each 32-wide row; 1 row and 128 rows) matched `torch.scatter` in every value.
  - a round trip of `UInt16` data through `ttnn.to_layout` to row major, with the shapes of the `scatter` index ([1, 4] and [128, 4]), returned every value unchanged. The values included 0 to 31, 32767, 32768, 40000 and 65535.

  So `scatter` itself accepts a `UInt16` index on Quasar.
- **`Int16` or `RawUInt16` for `untilize_with_unpadding`:** both formats work. With `RawUInt16` instead of `Int16`, the same three checks also passed, including the values 1 to 31, which are subnormal numbers as `Float16` (2026-11-03, `20261103_rawuint16/`). The other Quasar uplifts use the two formats in different cases.
  - `RawUInt16` and `RawUInt32` are used where only data movement kernels touch the data: in `slice` (#63023) and, on `main`, for the index DFB of the Quasar copy of pool ([pool_multi_core_program_factory.cpp:797-812](ttnn/cpp/ttnn/operations/experimental/quasar/pool_generic/device/pool_multi_core_program_factory.cpp#L797-L812)). The comment in #63023 says to use the raw formats only for a DFB that no compute kernel handles in a way that depends on the element type, because compute kernels see the raw formats as `Float16` and `Float32`.
  - `Int16` is used where a compute kernel passes 16-bit integer data through: in `topk` (#63516, tested on craq-sim by its author), and in the LLK test `QuasarComputePackUntilize` on `main` ([test_untilize_tilize.cpp:1335-1343](tests/tt_metal/tt_metal/llk/test_untilize_tilize.cpp#L1335-L1343)). That test skips `Int16` with 32-bit DEST (DEST is the register file that holds tile data in the compute unit until the packer writes it to a DFB; each value in it has 16 or 32 bits), a mode that Quasar does not support for `Int16`.
  - The `untilize_with_unpadding` compute kernel unpacks and packs every tile, so `Int16` follows this convention.
  - #63516 also adds `Int16` to the formats that a JIT build check lets a kernel mix with formats of another exponent type (`check_consistent_format_across_buffers`, [data_format.cpp:68-99](tt_metal/jit_build/data_format.cpp#L68-L99)), because `topk` has `Int16` and bfloat16 DFBs in one kernel. `untilize_with_unpadding` does not need that change: both of its DFBs are `Int16`, and the tests above passed without it.
- The branch has #63023 through `main` since the rebase. None of the other changes above is on the branch; each was removed after its test, and the branch was rebuilt.

### Model run with the device embedding (2026-11-03)

Until 2026-11-03, the model ran the token embedding on the host on Quasar. Its code comment gave two reasons: no device op could read the `UInt32` token IDs, and the table would take about 1 GiB of DRAM. The first reason was true only for the slice, and only before #63023: the decode step slices the `UInt32` token IDs on the device, and that slice failed on D. `ttnn.embedding` itself already accepts `UInt32` token IDs on Quasar, because both of its factories that the model uses keep the IDs in a scratchpad, not in a DFB. The row-major factory, which runs when the number of IDs is not a multiple of 32 (as in the 1-ID decode), has its scratchpad at [embeddings_rm_program_factory.cpp:181-183](ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp#L181-L183). The fused factory, which also tilizes the output and runs for the 32-ID decode and the 128-ID prefill ([embedding.cpp:45-55](ttnn/cpp/ttnn/operations/embedding/embedding.cpp#L45-L55)), has its scratchpad at [embeddings_fused_program_factory.cpp:211-213](ttnn/cpp/ttnn/operations/embedding/device/embeddings_fused_program_factory.cpp#L211-L213). All 3 `test_embedding` cases pass. With #63023 in the branch, the device embedding was tested on craq-sim in the two ways below. After these tests, the host embedding was removed from the model (see the D recommendation), and the demo was run again.

- **The steps of the model's embedding code, with a table of the real size** (201088 x 2880 bf16, random values; script `/localdev/fplavec/temp/gpt_oss_ops_runs/20261103_check_device_embedding.py`, log `20261103_device_embedding_seq.log` in the same folder). The decode steps slice the [1, 1, 1, 32] `UInt32` token IDs to the first ID and then run `ttnn.embedding` into a bf16 TILE tensor. A second decode check uses all 32 IDs, and the prefill check uses 128 IDs. The run compiled the `slice` reader and writer kernels (`slice_reader_unary_unpad_dims_rm_interleaved_start_id`, `slice_writer_unary_stick_layout_interleaved_start_id`), so the slice ran on the device. All three outputs were bit-exact with a torch lookup, including the first and the last row of the table. The table upload took 0.3 s.
- **The demo** (`test_gpt_oss_demo[quasar-1x1-prefill_128]`, `QSR_N_LAYERS=2`, log `/localdev/fplavec/temp/gpt_oss_ops_runs/20261103_demo_device_embedding.log`). For this run, the model code was changed so that the model took the device path of the embedding on Quasar too, and so that `quasar_block_float_to_bf16` changed the `bfloat8_b` output dtype of `ttnn.embedding` to bfloat16. The second change is needed: with the `bfloat8_b` output of the captured calls, `test_embedding[01_1x1_u32_int-dram]` and `[02_1x128_u32_int-dram]` fail with "DFB 'out' has data format 'Bfp8_b' which is not supported on architecture quasar" (2026-11-03, `TTNN_GRAPH_OPS_BF8_AS_BF16` not set, log in `/localdev/fplavec/temp/gpt_oss_ops_runs/20261103_embedding_bf8_output/`). The run did not have the two `conftest.py` changes described above the "Open problems" table.
  - Model setup, including the conversion of the weights into a new weight cache, finished in about 5 min without an allocation error. So the 1.08 GiB embedding table fits in DRAM next to the bfloat16 weights of 2 layers, their KV cache and the `lm_head`. The run stopped on K in layer 0 (next bullet), so whether the activations of a full prefill and of decode also fit in DRAM next to the table was not tested.
  - The prefill ran the device embedding (in `prepare_inputs_prefill`, [model.py:1106](models/experimental/ops/quasar/gpt_oss/tt/model.py#L1106-L1106)), and then, in layer 0, the attention ops up to SDPA and the head concatenation. It stopped on K in the attention output projection ([operations.py:114](models/experimental/ops/quasar/gpt_oss/tt/attention/operations.py#L114-L114)), the call that `test_matmul[01_128x4096_bf8_int-dram]` reproduces.
  - The run did not reach decode, so the model itself did not run the slice of the token IDs. The first check above covers the decode steps.
  - Until 2026-11-03, the demo turned off traces on Quasar, with a log message that gave the host embedding as the reason. A trace records the device commands of a step and later replays them without the host, so host work in the step is skipped. Traces must stay off for another reason: the simulator runs use slow dispatch, and in slow dispatch the command queue rejects trace capture and replay ([sd_mesh_command_queue.cpp:487-493](tt_metal/distributed/sd_mesh_command_queue.cpp#L487-L493)). So the demo turns traces off when `TT_METAL_SLOW_DISPATCH_MODE` is set ([text_demo.py:486-490](models/experimental/ops/quasar/gpt_oss/demo/text_demo.py#L486-L490)).
- **The demo with the host embedding removed from the model** (same test and settings, log `/localdev/fplavec/temp/gpt_oss_ops_runs/20261103_demo_device_embedding_final.log`). The demo logged that it runs without traces because of slow dispatch. Model setup took about 4 min with the weight cache from the previous run. The prefill ran the device embedding (the run rebuilt the `embeddings_tilize` kernel) and stopped on K at the same attention output projection.

### I: PR #63401 does not fix it

The block writer that PR #63401 changes (see the D notes) is `writer_unary_stick_layout_wh_multicore_metal2.cpp`, which is also the writer kernel of the mainline `untilize` block factory ([untilize_multi_core_block_program_factory.cpp:61-63](ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_block_program_factory.cpp#L61-L63)). PR #63401 does not turn off implicit sync, so it does not fix I.

Merged PR #62730 fixed the same error in the interleaved `untilize_with_unpadding` factory ("threshold 263 / per_txn 263 exceed uint8_t") by turning off implicit sync on its data movement kernels. The same change in the mainline `untilize` factories would be a stopgap until the fix in the DFB code that #62783 proposes.

## Recommendations in detail

On the device, the gpt-oss model works around A with bfloat16 instead of block float, and works around B and I with the Quasar copies of the ops. With the two `conftest.py` changes described above the "Open problems" table, it bypasses G, H and K for `sparse_matmul` by running `sparse_matmul` on the host, which is not a fix. The token embedding runs on the device since 2026-11-03 (see "Model run with the device embedding"): merged PR #63023 fixed D for the slice of the token IDs, and the branch has #63023 since the rebase onto `main` `8b53dfa87b6`. The 1.08 GiB embedding table ([QUASAR_DATA_FORMATS.md:97-109](models/experimental/ops/quasar/gpt_oss/QUASAR_DATA_FORMATS.md#L97-L109)) fits in DRAM next to the weights of 2 layers (tested). Both demo runs stopped on K in layer 0, so whether the activations of a full prefill and of decode also fit in DRAM next to the table was not tested. To run `sparse_matmul` on the device, the model needs G, H and K fixed. From the model code, the problems that the model hits on the device are D (`topk`, `scatter`), E (`fast_reduce_nc`), G (`nlp_concat_heads_decode`, decode only), K (`ttnn.matmul` in attention, and in the `lm_head` when the `lm_head` input is in DRAM) and softmax.

| # | Recommendation |
|---|---|
| A | Keep bfloat16 for now. Later, use MX formats: ttnn has the MX `DataType`s since PR #58175, and the ops still need to accept them (issue #58157). Open PR #62751 adds MX to the Quasar copy of `matmul`, to `reduce` and to the normalization ops. `mxint8` is the closest match to `bfloat8_b`, and `mxfp4` matches the gpt-oss expert weights. |
| B | Nothing needed for the model. For the op tests, run these cases with `ttnn.experimental.quasar.add`, `multiply` and `linear`, as the SDPA test files do, until the mainline `binary_ng` gets a Quasar factory. No PR for that was found. |
| D | `tensor_getitem`: done. The branch was rebased onto `main` `8b53dfa87b6`, which has PR #63023 (merged 2026-11-03). #63023 makes `test_tensor_getitem[01_1x32_u32_int-dram]` pass (tested). Model: the token embedding runs on the device since 2026-11-03, tested up to the K failure in the first prefill (see "Model run with the device embedding"). On 2026-11-03, the host embedding was removed from `model.py`, which made its embedding code the same as on `main` `8b53dfa87b6`; `ttnn.embedding` was added to the ops in `quasar_block_float_to_bf16`; `QUASAR_DATA_FORMATS.md` was updated to match; and the demo was changed to turn traces off in slow dispatch mode, which does not support traces ([text_demo.py:486-490](models/experimental/ops/quasar/gpt_oss/demo/text_demo.py#L486-L490)). `topk`: test open PR #63516 on top of the branch, then rerun `test_topk`. `scatter`: make a PR that declares the `UInt16` DFBs of the interleaved `untilize_with_unpadding` factory as `Int16` on Quasar, as #63516 does for `topk`. With this change, both `scatter` cases pass, the values match `torch.scatter`, and `UInt16` values above 32767 come back unchanged (tested; see the D notes). `RawUInt16` passed the same tests. `Int16` matches other Quasar code where a compute kernel passes 16-bit integers through: `topk` (#63516) and the LLK test `QuasarComputePackUntilize`. In the PR, add a test that converts `UInt16` data with values over the whole 16-bit range to row major and compares every value. |
| E | `fast_reduce_nc`: open PR #62204 fixes the hang without a kernel change (tested), so get #62204 reviewed and merged. If #62204 does not merge, make the tested change in `reader_reduce_nc.cpp` (a `DataflowBuffer` for `in1` at kernel scope, passed to `prepare_zero_tile`) as a PR that refers to #60848, and look for more id-only helper calls in the kernels that the model uses: the Quasar compile prints a deprecation warning for each one. Matmul: nothing needed for E. With the local fix for K, `test_matmul[01_...]` and `[04_...]` pass (tested). |
| G | `nlp_concat_heads_decode` needs a change so that one core can take several heads. Until then, use a workaround such as llama's `_install_quasar_concat_heads_grid_agnostic` ([test_llama_e2e.py:938](models/experimental/llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L938)). `sparse_matmul`: give `test_sparse_matmul[01_...]` and `[03_...]` a program config that fits 8x4. The model needs the same to run `sparse_matmul` on the device. |
| H | Replace the self-looped `sparsity` DFB with a scratchpad, as merged PR #63023 does for `slice`. With K fixed, `test_sparse_matmul[00_...]` and `[02_...]` fail on H (tested). The model needs this fix to run `sparse_matmul` on the device. |
| I | Nothing needed for the model's `ttnn.untilize` calls. For the op: follow issue #62783, whose author proposes a fix in the DFB code. If the op is needed before that fix, pass `disable_dfb_implicit_sync_for_all=true` in the mainline `untilize` factories as a stopgap, as the `tilize` block factory does and as merged PR #62730 did for the interleaved `untilize_with_unpadding` factory. |
| K | Post the cause and the tested fix from the notes on K in issue #63758, and add the `sparse_matmul` error to the issue. Then make a PR with that fix, with `test_matmul[00_1x2880_bf8_int-l1]` and `test_sparse_matmul[00_1x2880_bf8_int-l1]` as reproducers. For the model until the fix merges: in `quasar_ops`, route `ttnn.matmul` to `ttnn.experimental.quasar.matmul`. The Quasar copy passes all 5 `test_matmul` cases: the three attention cases and both 262144-wide `lm_head` cases (tested), so the route needs no exception for the `lm_head`. The route was not tested in a model run. |
| softmax | Hang: get open PR #62204 reviewed and merged (tested). If #62204 does not merge, make the kernel change tested in the softmax notes above (kernel-scope `DataflowBuffer` objects for the two scalers in the reader, and the kernel's own `DataflowBuffer` objects passed to both `reduce` calls in the compute kernel) as a PR that refers to #60848. All-zero output: ask the author of open PR #59572 to fix its 3 calls that use the undeclared `dfb_id` (`dfb.get_id()` works), then get #59572 merged. With that fix, #59572 makes both cases pass together with #62204, and also together with the kernel change (both tested). |

# History: runs 1 to 5

The sections below are as written after each run. Results, links and recommendations in them can be out of date; "Current state" above replaces them.

The runs below used branch `fplavec/gpt_oss_quasar_bringup`, then rebased on `main` `345aebfbe8d`, with the TEMP squash of PR #60187 (DFB drain hang fix, version of 2026-10-19).

## Status on 2026-10-30

Since the runs, the branch was rebased on `main` `f23b877dadb` (2026-10-30), which already contains PR #60187, so the TEMP commit is gone. Several problems found here are now fixed on `main` or worked around in the gpt-oss model. Run 5 reran the op tests on the new base; its results are in the next section.

| # | Category | Change since the runs |
|---|---|---|
| A | Block float format | PR #58175 (merged 2026-10-28) adds MX `DataType`s to ttnn (`ttnn.mxfp8_e4m3`, `mxfp8_e5m2`, `mxfp6_e2m3`, `mxfp6_e3m2`, `mxfp4`, `mxint8`, `mxint4`, `mxint2`). Ops still need their dtype checks extended (issue #58157, open). |
| C | Kernel does not compile | Fixed on `main` by PR #61760 (merged 2026-10-22), which closes #61354 and #61482. Run 5: the 13 `rms_norm`, `rotary_embedding_llama` and `typecast` cases pass; the 2 `softmax` cases compile but hang. |
| E | Hang | Issue #60848 is fixed by PR #62092 (merged 2026-10-24): helpers take the kernel's `DataflowBuffer` by reference. That covers the 6 hangs that #60848 covered (both decode SDPA cases, prefill SDPA, `sum`, `fast_reduce_nc`, probably `paged_update_cache`). For the multicast matmul, issue #61887 (sharded output, 4 output blocks per core) is fixed by PR #62081 (merged 2026-10-26). The two matmul hangs here have a DRAM interleaved output, so whether #62081 or #62092 also fixes them is unknown. Run 5: 5 of the 8 hangs pass; the other 3 stop earlier on the simulator's MOP limit (J). |
| F | Simulator error | Run 5: passes. So the mid-kernel drain fixed by #62092 was most likely the cause. |
| H | Self-looped DFB | Unchanged on `main`: the mainline `sparse_matmul` still binds `sparsity` as producer and consumer of one data movement kernel, and the Quasar copy still creates legacy kernels. In the gpt-oss model, `ttnn.sparse_matmul` now runs on the host on Quasar (`make_host_sparse_matmul` in `models/experimental/ops/quasar/gpt_oss/conftest.py`, not yet committed or run in the model). |
| I | DFB descriptor field too small | Unchanged on `main`: the mainline `untilize` factories still leave implicit sync on. Issue #62783 ("[Quasar] untilize block factory sizes a DFB past the 255-entry implicit-sync limit", open, filed 2026-10-27) now tracks it. In the gpt-oss model, `ttnn.untilize` now runs as `ttnn.experimental.quasar.untilize` on Quasar (`quasar_ops` in the same `conftest.py`, not yet committed or run in the model). |
| B, D, G | | Run 5: unchanged, where the case gets far enough to show it. |

The folders and scripts of runs 1 to 4 in `/localdev/fplavec/temp/` (`gpt_oss_ops.log`, the run folders under `gpt_oss_ops_runs/`, `rerun_ops_from_summary.sh`, `run_matmul_experiments.sh`) no longer exist. `run_ops_each.sh` was rewritten for run 5. `exp_matmul_mcast.py` is only in the git stash named "matmul test" on the branch.

## Run 5: the new base (2026-10-30)

Run 5 used the branch on `main` `f23b877dadb`, the craq-sim build from 2026-10-14 in `/localdev/fplavec/temp/qsr-sim` (3 GiB DRAM per bank), bfloat16 (`TTNN_GRAPH_OPS_BF8_AS_BF16=1`), and `/localdev/fplavec/temp/run_ops_each.sh`: one pytest process per case, 10 min timeout per case, pytest's own 300 s timeout off. It took 49 min. Output: `/localdev/fplavec/temp/gpt_oss_ops_runs/20261030_bf16/`.

| Result | Runs 3 + 4 | Run 5 |
|---|---|---|
| Passed | 63 | 67 |
| Failed | 45 | 20 |
| Timeout | 8 | 2 |
| Simulator error | 1 | 28 |
| Skipped | 2 | 2 |
| **Total** | **119** | **119** |

### Change per case

| Runs 3 + 4 | Run 5 | Cases |
|---|---|---|
| passed | passed | 48 |
| C: compile error | **passed** | 13: all 5 `rms_norm`, all 4 `rotary_embedding_llama`, all 4 `typecast` |
| E: hang | **passed** | 5: `test_paged_scaled_dot_product_attention_decode[00_64x64_bf16_hs-l1]` and `[01_64x64_bf16_hs-l1]`, `test_scaled_dot_product_attention[01_64x128x64_bf16_int-dram]`, `test_sum[00_32x2880_bf8_int-l1]`, `test_paged_update_cache[00_64x8x64x64_bf8_int-dram]` |
| F: simulator error | **passed** | 1: `test_scaled_dot_product_attention[00_64x128x64_bf16_int-dram]` |
| passed | **J: simulator error** (new) | 15: `test_clamp[02_...]` and `[03_...]`, `test_embedding[02_1x128_u32_int-dram]`, `test_linear[03_128x2880_bf8_int-dram]`, `test_matmul[00_...]`, `[02_...]` and `[03_...]`, `test_permute[01_32x1x2880_bf8_int-l1]`, `test_reshape[01_...]`, `[06_...]`, `[07_...]` and `[08_...]`, `test_sigmoid[01_32x128x2880_bf8_int-dram]`, `test_transpose[00_...]` and `[03_...]` |
| C: compile error | **timeout** (hang) | 2: `test_softmax[00_1x4_bf16_int-l1]`, `test_softmax[01_128x4_bf16_int-l1]` |
| E: hang | J: simulator error | 3: `test_fast_reduce_nc[00_32x128x2880_bf8_int-dram]`, `test_matmul[01_128x4096_bf8_int-dram]`, `test_matmul[04_32x2880_bf8_int-dram]` |
| B | B (same error) | 14 |
| B | B, then J | 5: `test_add[05_...]` and `[09_...]`, `test_multiply[03_...]`, `[05_...]` and `[06_...]` |
| D | D (same error) | 5 |
| G | G (same error) | 1: `test_nlp_concat_heads_decode[00_64x64_bf16_hs-l1]` |
| G, H | J: simulator error | 4: all `test_sparse_matmul` cases |
| I | I, then J | 1: `test_untilize[00_32x262144_bf8_int-dram]` |
| skipped | skipped | 2 |

A case shows only its first error, so the 23 cases that now stop on J (and the 4 `matmul` cases with K) are not tested past it.

### New problems in run 5

- **J: the simulator build is older than `main`.** 28 cases end with `ERROR: UnimplementedFunctionality: mop_expander: inner_loop_len=N`, with N = 90, 160 or 265: the number of tiles in a row 2880, 5120 or 8480 values wide. In 21 of them it is the first error; the other 7 first log a `TT_FATAL` (B, I or K). The craq-sim build from 2026-10-14 allows a MOP (a hardware loop of Tensix instructions) an inner loop of at most 64 (`src/tensix.cpp`, `inner_loop_len <= 64`, in the craq-sim checkout at `9f4e56d9`). `main` PR #61421 (merged 2026-10-27, "Unpack-tilize into DEST and one SrcA MOP run per tile row") runs one MOP per tile row in tilize, which needs longer loops. craq-sim commit `9afaf69b` on its `quasar` branch (2026-10-28, "qsr: accept MOP inner loops up to the LOOP_LENGTH maximum (#537)") raises the limit to 1023. My guess is that the tilize runs while the test uploads its inputs, so these cases stop before the op under test runs: in `test_reshape[01_32x1x2880_bf8_int-l1]` the error comes right after device start. This is not a tt-metal bug, and the gpt-oss model run will hit it too, because its activations are 2880 wide.
- **K: the multicast matmul fails a check of the new multicast helpers.** `test_matmul[00_...]`, `[01_...]`, `[02_...]` and `[04_...]`, the cases that use the multicast matmul, log `TT_FATAL: Multicast sender/forwarder NoC differs from the configured multicast NoC` (`ttnn/cpp/ttnn/kernel_lib/mcast/host/program_adapters/`). `main` moved the matmul to these helpers in PR #61818, merged 2026-10-30. The process continues after this `TT_FATAL` and then stops on J, as `test_add[09_...]` and `test_multiply[06_...]` do after their known `DataMovementKernel` error, so my guess is that K is a real failure of the multicast matmul on Quasar. No GitHub issue matches it.
- **softmax hangs.** It compiles now (C is fixed), but both cases make no progress for 10 min. Open PR #59572 ("softmax: Quasar uplift of the attention and general-H factories") describes this: the attention-optimized factory, which the gpt-oss router softmax uses, "hung in the pack-tile LLK re-init guard". #59572 is not in the base.

### Recommendations from run 5

| Problem | Recommendation |
|---|---|
| J | Build craq-sim from its `quasar` branch at `9afaf69b` or later, keep the local change that sets `DRAM_CHANNEL_SIZE` to 3 GiB for Quasar in `src/config.h`, and rerun. Until then, 28 cases, and the gpt-oss model, cannot get past their tilize. |
| K | Rerun the 4 `matmul` cases with the newer simulator. If the `TT_FATAL` stays, file an issue against the multicast helpers on Quasar (#61818), with `test_matmul[00_1x2880_bf8_int-l1]` as the reproducer. |
| softmax | Test PR #59572 on top of the branch, or wait until it merges. |
| B, D, G, H, I | As in the recommendations below. The gpt-oss model already works around B, H and I. |

## Runs

| # | Date | How | Block float cases | Output |
|---|---|---|---|---|
| 1 | 2026-10-19 19:03 | one pytest process for the whole suite | as captured (`bfloat8_b`, `bfloat4_b`) | `/localdev/fplavec/temp/gpt_oss_ops.log` |
| 2 | 2026-10-19 19:31 | `run_ops_each.sh`: one pytest process per case, 5 min timeout | as captured | `/localdev/fplavec/temp/gpt_oss_ops_runs/20261019_193135/` |
| 3 | 2026-10-19 20:47 | `run_ops_each.sh`, 5 min timeout | as bfloat16 (`TTNN_GRAPH_OPS_BF8_AS_BF16=1`) | `/localdev/fplavec/temp/gpt_oss_ops_runs/20261019_204759/` |
| 4 | 2026-10-19 23:06 | `rerun_ops_from_summary.sh`: the 9 TIMEOUT cases of run 3, 30 min timeout | as bfloat16 | `/localdev/fplavec/temp/gpt_oss_ops_runs/20261019_204759_rerun_20261019_230653/` |
| 5 | 2026-10-30 19:41 | `run_ops_each.sh` (rewritten), new base, 10 min timeout | as bfloat16 | `/localdev/fplavec/temp/gpt_oss_ops_runs/20261030_bf16/` (see "Run 5" above) |
| 6 | 2026-10-30 21:17 | as run 5, with the craq-sim build of 2026-10-30 | as bfloat16 | `/localdev/fplavec/temp/gpt_oss_ops_runs/20261030_bf16_newsim/` (see "Current state" at the top) |

## Did all ops run?

- **Run 1: no.** Only 42 cases started. The 42nd case, `test_paged_scaled_dot_product_attention_decode[00_64x64_bf16_hs-l1]`, hung. The `timeout 10m` in the command then stopped pytest (SIGINT had no effect, then SIGKILL after 20 s), so the other 77 cases never ran and pytest printed no summary.
- **Runs 2 and 3: yes.** Each case ran in its own process with its own timeout, so the cases that hung did not stop the others.

The rest of this report uses run 3 together with run 4 (the result of run 4 replaces run 3's result for the 9 cases it reran), and compares them with run 2.

## Results

| Result | Run 2 (as captured) | Run 3 (as bfloat16) | Runs 3 + 4 (as bfloat16, timeouts rerun with 30 min) |
|---|---|---|---|
| Passed | 44 | 62 | 63 |
| Failed | 70 | 45 | 45 |
| Timeout | 3 (5 min) | 9 (5 min) | 8 (30 min): hangs |
| Simulator error | (counted as failed) | 1 | 1 |
| Skipped | 2 | 2 | 2 |
| **Total** | **119** | **119** | **119** |

- All 44 cases that passed in run 2 also pass in run 3.
- 18 cases that failed in run 2 only because of their block float format pass in run 3, and one more (`test_matmul[03_1x2880_bf8_int-l1]`) passes in run 4, given more time.
- The 2 skipped cases (`test_add[03_32x2880_bf16_ws-l1]`, `test_linear[00_32x4096_bf16_ws-l1]`) are skipped on purpose by the suite: their captured width-sharded grid does not fit the 8x4 device.
- Run 2 recorded the simulator error as a failure. The script now reports it separately.

### Per test file (runs 3 + 4)

| Test file | Cases | Passed in run 2 | Passed | Failed | Timeout | Sim. error | Skipped |
|---|---|---|---|---|---|---|---|
| `test_add.py` | 11 | 0 | 0 | 10 | 0 | 0 | 1 |
| `test_clamp.py` | 4 | 0 | 4 | 0 | 0 | 0 | 0 |
| `test_embedding.py` | 3 | 1 | 3 | 0 | 0 | 0 | 0 |
| `test_fast_reduce_nc.py` | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| `test_interleaved_to_sharded.py` | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| `test_linear.py` | 4 | 0 | 1 | 2 | 0 | 0 | 1 |
| `test_matmul.py` | 5 | 0 | 3 | 0 | 2 | 0 | 0 |
| `test_multiply.py` | 7 | 0 | 0 | 7 | 0 | 0 | 0 |
| `test_nlp_concat_heads.py` | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| `test_nlp_concat_heads_decode.py` | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| `test_nlp_create_qkv_heads.py` | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| `test_nlp_create_qkv_heads_decode.py` | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| `test_paged_fill_cache.py` | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| `test_paged_scaled_dot_product_attention_decode.py` | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| `test_paged_update_cache.py` | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| `test_permute.py` | 3 | 2 | 3 | 0 | 0 | 0 | 0 |
| `test_repeat.py` | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| `test_reshape.py` | 14 | 14 | 14 | 0 | 0 | 0 | 0 |
| `test_rms_norm.py` | 5 | 0 | 0 | 5 | 0 | 0 | 0 |
| `test_rotary_embedding_llama.py` | 4 | 0 | 0 | 4 | 0 | 0 | 0 |
| `test_scaled_dot_product_attention.py` | 2 | 0 | 0 | 0 | 1 | 1 | 0 |
| `test_scatter.py` | 2 | 0 | 0 | 2 | 0 | 0 | 0 |
| `test_sigmoid.py` | 2 | 0 | 2 | 0 | 0 | 0 | 0 |
| `test_slice.py` | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| `test_softmax.py` | 2 | 0 | 0 | 2 | 0 | 0 | 0 |
| `test_sparse_matmul.py` | 4 | 0 | 0 | 4 | 0 | 0 | 0 |
| `test_sum.py` | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| `test_tensor_getitem.py` | 3 | 2 | 2 | 1 | 0 | 0 | 0 |
| `test_to_device.py` | 5 | 5 | 5 | 0 | 0 | 0 | 0 |
| `test_to_layout.py` | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| `test_to_memory_config.py` | 2 | 2 | 2 | 0 | 0 | 0 | 0 |
| `test_topk.py` | 2 | 0 | 0 | 2 | 0 | 0 | 0 |
| `test_transpose.py` | 5 | 2 | 5 | 0 | 0 | 0 | 0 |
| `test_typecast.py` | 4 | 0 | 0 | 4 | 0 | 0 | 0 |
| `test_unsqueeze.py` | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| `test_unsqueeze_to_4D.py` | 8 | 8 | 8 | 0 | 0 | 0 | 0 |
| `test_untilize.py` | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| `test_zeros_like.py` | 2 | 1 | 2 | 0 | 0 | 0 | 0 |

## Failure categories (runs 3 + 4)

A case can only show its first error: once that error is fixed, the case may fail in another way.

| # | Category | Error | Cases | Ops |
|---|---|---|---|---|
| A | Block float data format | `DFB '<name>' has data format 'Bfp8_b' which is not supported on architecture quasar` | 0 (38 in run 2) | none in run 3 |
| B | Mainline op builds legacy kernels | `DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead.` | 19 | all 10 run cases of `test_add`, all 7 of `test_multiply`; `test_linear[01_1x2880_bf8_int-l1]` and `test_linear[02_128x2880_bf16_int-dram]` |
| C | Kernel does not compile | `trisc0 build failed`: `chain.inl:852: missing template argument list after 'DataflowBuffer'` | 15 | `rms_norm` 5, `rotary_embedding_llama` 4, `typecast` 4, `softmax` 2 |
| D | Unsigned integer formats | `DFB '<name>' has data format 'UInt16'` (4) or `'UInt32'` (1) `which is not supported on architecture quasar` | 5 | `test_topk` 2 (index output), `test_scatter` 2 (index input), `test_tensor_getitem[01_1x32_u32_int-dram]` (`UInt32` token IDs) |
| E | Hang | no error; no progress for 30 min | 8 | `test_paged_scaled_dot_product_attention_decode[00_64x64_bf16_hs-l1]` and `test_paged_scaled_dot_product_attention_decode[01_64x64_bf16_hs-l1]`, `test_scaled_dot_product_attention[01_64x128x64_bf16_int-dram]`, `test_matmul[01_128x4096_bf8_int-dram]` and `test_matmul[04_32x2880_bf8_int-dram]`, `test_sum[00_32x2880_bf8_int-l1]`, `test_fast_reduce_nc[00_32x128x2880_bf8_int-dram]`, `test_paged_update_cache[00_64x8x64x64_bf8_int-dram]` |
| F | Simulator error | `NonContractualBehavior: rv32_mem_rd: unaligned addr=0x400009 size=4` | 1 | `test_scaled_dot_product_attention[00_64x128x64_bf16_int-dram]` |
| G | Core grid larger than the device | `Target number of cores 64 is greater than total number of available cores 32` (1); `No core coordinate found at location: (4, 5, TENSIX, LOGICAL)` (2) | 3 | `test_nlp_concat_heads_decode[00_64x64_bf16_hs-l1]`, `test_sparse_matmul[01_32x1x2880_bf8_int-l1]` and `test_sparse_matmul[03_32x128x2880_bf8_int-dram]` |
| H | Self-looped DFB in a data movement kernel | `DataflowBuffer 'sparsity' is self-looped by data-movement kernel 'in0_sender' (bound as both PRODUCER and CONSUMER)` | 2 | `test_sparse_matmul[00_1x2880_bf8_int-l1]` and `test_sparse_matmul[02_4x32x2880_bf16_int-dram]` |
| I | DFB descriptor field too small | `Implicit-sync DFB descriptor overflow: threshold 265 / per_txn 265 exceed uint8_t` | 1 | `test_untilize[00_32x262144_bf8_int-dram]` |

In run 2, one more category existed: `test_scatter[00_1x32_bf8_int-l1]` failed with `input_dtype differs from src_dtype`, because it passes a `bfloat8_b` input and a `bfloat16` source. As bfloat16, it gets past that check and fails like `test_scatter[01_128x32_bf16_int-l1]` (category D).

### Notes on the categories

- **A:** the captured cases use `bfloat8_b` or `bfloat4_b` tensors, and Quasar has no block float formats. This is a hardware limit: the Quasar hardware format list has no `Bfp8_b` or `Bfp4_b`. Quasar has the OCP MX formats instead (`MxFp8R/P`, `MxFp6R/P`, `MxFp4`, `MxInt8/4/2`): blocks of 32 values share one 8-bit scale. Metal, its host packers and the Quasar kernel compiler already support them, but at the time of the runs ttnn had no MX `DataType`, so a model could not create an MX tensor. PR #58175 (merged 2026-10-28) adds `ttnn.mxfp8_e4m3`, `mxfp8_e5m2`, `mxfp6_e2m3`, `mxfp6_e3m2`, `mxfp4`, `mxint8`, `mxint4` and `mxint2`; each op still needs its dtype checks extended (issue #58157, open). Until then, the llama Quasar model and the gpt-oss Quasar model both use bfloat16 instead, and run 3 does the same.
- **B:** the program factories of the mainline binary ops build `DataMovementKernel` kernels, which only Wormhole and Blackhole accept. The mainline `linear` also fails this way; my guess is that it adds its bias with the mainline binary op, as seen in the gpt-oss model run.
- **C:** a bug on `main`, from two PRs that met: #60306 (2026-10-15) added `check_bound_dataflow_buffer(const DataflowBuffer&)` to `ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.inl`, and #52360 (2026-10-16) made `DataflowBuffer` a class template on Quasar only. Wormhole and Blackhole still compile. Tracked in issue #61354 ("Quasar: kernel_lib eltwise chain.inl does not compile since DataflowBuffer became a class template"), fixed since by PR #61760 (merged 2026-10-22). Not caused by the TEMP commit, which does not touch `chain.inl` or `dfb_access.h`. The gpt-oss model run stops at this error too, in its first RMS norm.
- **D:** `UInt32` is not a Quasar hardware format, so it is a permanent limit, but `Int32` and `RawUInt32` are supported. `UInt16` is a Quasar hardware format but is missing from Metal's list of supported Quasar formats, so it is probably a software gap.
- **E:** run 4 gave the 9 cases that timed out in run 3 30 minutes each.
  - **One was only slow:** `test_matmul[03_1x2880_bf8_int-l1]`, the `lm_head` matmul with a 2880 x 262144 weight, passed in 5 min 55 s. Its matmul program alone ran for about 3.7 min.
  - **8 are hangs.** In 7 of them the log stops 5 to 7 s after the start, in the first or second device program, and nothing more happens for 30 min. `test_matmul[04_32x2880_bf8_int-dram]` launches the same three programs at the same times as `test_matmul[03_1x2880_bf8_int-l1]` (the same 2880 x 262144 weight, and one tile row of input in both cases), but its matmul program did not finish in 28 min, more than 7 times as long as in `test_matmul[03_1x2880_bf8_int-l1]`.

  | Hung case | Device program that does not finish | Covered by issue #60848? |
  |---|---|---|
  | `test_paged_scaled_dot_product_attention_decode[00_64x64_bf16_hs-l1]` and `test_paged_scaled_dot_product_attention_decode[01_64x64_bf16_hs-l1]` | Quasar decode SDPA (`sdpa_flash_decode`) | yes: 10 `DataflowBuffer(id)` temporaries in `sdpa_flash_decode.cpp`, 8 in its `compute_common.hpp` |
  | `test_scaled_dot_product_attention[01_64x128x64_bf16_int-dram]` | Quasar prefill SDPA | yes: 70 temporaries in `compute_streaming.hpp`, 8 in its `compute_common.hpp` |
  | `test_sum[00_32x2880_bf8_int-l1]` | `reduce` (`reduction/generic/device/kernels/compute/reduce.cpp`) | yes: uses `compute_kernel_lib::reduce` |
  | `test_fast_reduce_nc[00_32x128x2880_bf8_int-dram]` | `reduce_nc` | yes: its reader uses `prepare_zero_tile` |
  | `test_paged_update_cache[00_64x8x64x64_bf8_int-dram]` | `update_cache` | probably: uses `compute_kernel_lib::tilize` and `untilize`, which #60848 lists with the note "may be balanced within the call; check" |
  | `test_matmul[01_128x4096_bf8_int-dram]` and `test_matmul[04_32x2880_bf8_int-dram]` | mainline 1D multicast matmul (`bmm_large_block_zm_fused_bias_activation_metal2`) | no: the kernel uses no listed helper and no temporaries. No GitHub issue matches a hang of this kernel on Quasar. |

  - **Likely cause of 6 of the 8 hangs:** issue #60848 (open at the time of the runs, fixed since by PR #62092, merged 2026-10-24), "Quasar: helper-owned and temporary DataflowBuffers drain mid-kernel". Since #57646, a `DataflowBuffer` waits for its DFB to drain when it goes out of scope. A helper that builds its own `DataflowBuffer`, or a one-line temporary such as `DataflowBuffer(id).wait_front(n)`, therefore drains in the middle of the kernel, and the kernel hangs if the consumer has not popped everything yet.
  - PR #60187 (the TEMP commit) does not fix this. Its description says that on craq-sim, "sdpa decode and prefill with attention sink" pass only "with a local change that defers the drain to kernel exit (needed until #60848 is fixed)". The 2026-10-13 version of #60187 contained that change; the current version does not.
  - The llama Quasar work found a different decode SDPA hang earlier: "The full 8-kv-head decode flash-decode HANGS on the sim once k_num_chunks>1 whenever a core owns >1 kv-head" (`_install_quasar_device_sdpa_split` in `models/experimental/llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py`). The decode SDPA cases have 8 KV heads, so that cause is possible too.
  - **The 2 matmul hangs are in the 1D multicast matmul, and depend on more than the kernel.** `test_matmul[03_1x2880_bf8_int-l1]` (passes) and `test_matmul[04_32x2880_bf8_int-dram]` (hangs) have the same 2880 x 262144 weight and, in tile layout, the same single 32-row tile row of input A, so the same compute work. They differ in three ways:

    | | `test_matmul[03_1x2880_bf8_int-l1]` | `test_matmul[04_32x2880_bf8_int-dram]` |
    |---|---|---|
    | Input A shape | `[1, 1, 1, 2880]` | `[1, 1, 32, 2880]` |
    | Input A location | SRAM (L1 interleaved) | DRAM interleaved |
    | Captured output | missing (`outs: None`) | `[1, 1, 32, 262144]`, DRAM |
    | Kernels built (run 4) | `bmm_metal2`, `reader_bmm_8bank_output_tiles_partitioned_metal2` | `bmm_large_block_zm_fused_bias_activation_metal2`, `reader_bmm_tile_layout_in0_sender_padding_metal2`, `reader_bmm_tile_layout_in0_receiver_metal2`, `reader_bmm_tile_layout_in1_sender_writer_padding_metal2` |
    | Implementation | plain multi-core matmul, no data shared between cores | 1D multicast matmul: one core reads input A and sends it to the others |

    Only the location of input A changes how the op runs. Neither case passes a program config, so `matmul` picks one from the shapes and memory locations (`ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp`): when input A, the weight and the output are all interleaved in DRAM, it takes a multicast config. That is true for case 04 and false for case 03. My guess is that for case 03 the multicast config for a 262144-wide output does not fit, so the selection falls through to the plain matmul at its end.
  - **The multicast matmul does not always hang.** In run 3, `test_matmul[00_1x2880_bf8_int-l1]` and `test_matmul[02_1x2880_bf8_int-dram]` also built the multicast kernels and passed. Both have a 5120-wide output. The two hung cases are larger: case 04 has a 262144-wide output, and `test_matmul[01_128x4096_bf8_int-dram]` has 4 tile rows and K = 4096. Which property causes the hang is not known. No GitHub issue matches a hang of this kernel on Quasar.
  - **The matmul experiments narrowed this down:** with everything else as in case 04, the multicast matmul passes up to an 8192-wide output and hangs from 10240 wide on. The Quasar copy of the matmul hangs too. The plain matmul passes the full 262144-wide case (see "Matmul experiments" below).
- **F:** a 32-bit RISC-V core read 4 bytes at `0x400009`, an address that is not 4-byte aligned, and the simulator stopped the process. On Quasar only the compute cores (TRISCs) are 32-bit, so the read came from the SDPA compute kernel. `0x400009` is 9 bytes past the end of the 4 MiB core SRAM, so my guess is that the kernel read through a pointer that was never set correctly. No GitHub issue matches this error for SDPA; the only `rv32_mem_rd` issues are about GroupNorm on the Wormhole simulator.
  - **F is probably related to E.** The two prefill SDPA cases differ in one setting only: `sliding_window_size` is 128 in `test_scaled_dot_product_attention[00_64x128x64_bf16_int-dram]` (this error) and `None` in `test_scaled_dot_product_attention[01_64x128x64_bf16_int-dram]` (the hang). Shapes (`[1, 64, 128, 64]` queries, 8 KV heads), attention sink, 8x4 grid and chunk sizes (32) are the same.
  - **The same call worked before the rebase.** gpt-oss layer 0 uses sliding attention with a window of 128, and the gpt-oss model run on 2026-10-15 ran its prefill SDPA with these settings and failed only later, in `ttnn.add`. That run used the 2026-10-13 version of the TEMP commit, which defers the DFB drain to kernel exit. My guess is that both E and F come from the mid-kernel drain in issue #60848. The rebase onto a newer `main` (264 commits) is the other difference between the two runs, so this is not proven.
- **G:** the op or its captured program config needs more cores than the 8x4 device has. `nlp_concat_heads_decode` uses one core per head (64 heads). The captured `sparse_matmul` program configs name core row 5, and the device has rows 0 to 3.
- **H:** the mainline `sparse_matmul` reader binds its `sparsity` DFB as both producer and consumer in one data movement kernel. Quasar does not allow that; the #60187 description lists it as known and not covered ("DM self-loop DFBs, which Gen2 rejects (they need a `Scratchpad` port)"). At the time of the runs, the gpt-oss model routed `ttnn.sparse_matmul` to `ttnn.experimental.quasar.sparse_matmul`; it now runs it on the host instead (see Status on 2026-10-30).
- **I:** the `untilize` case is the `lm_head` output, 262144 values wide. The full error is `threshold 265 / per_txn 265 exceed uint8_t (num_entries 265, num_txn_ids 1, num_prods_or_cons 1)`: the program factory sizes a DFB with 265 entries, and with implicit sync the per-transaction counts in the DFB descriptor are 8 bits, so at most 255. The check is in `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp` and came in with #52360 (2026-10-16), and its message suggests "Reduce num_entries or use explicit sync".
  - **The check is intentional, the limit is older.** Before #52360 the same values were already stored in 8-bit fields, with `static_cast<uint8_t>` and no check, so 265 silently became 9. #52360 turned that silent truncation into an error.
  - **Quasar only.** The transaction-ID descriptors, and with them this check, are set up only on chips with tile counter registers. On Wormhole and Blackhole, `ProgramImpl::finalize_dataflow_buffer_configs` skips that step ("On WH/BH there are no tile counters or remapper hardware").
  - **Why 265 fails and 264 would not.** A DFB can use up to 4 transaction IDs (`NUM_TXN_IDS = 4`), and the 255 limit applies per transaction ID. The host picks the smallest count from 2 to 4 that divides the entries evenly, and falls back to 1 if none does. 265 = 5 x 53 is not divisible by 2, 3 or 4, so it gets 1 ID with 265 entries. A block of 264 entries would get 2 IDs with 132 each and pass. So the effective limit is up to 1020 entries, depending on how the entry count divides.
  - **Explicit sync avoids the check, and is a two-argument change.** The transaction-ID descriptors are computed only for DFB sides with implicit sync on. The `untilize` block factory (`ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_block_program_factory.cpp`) creates its reader and writer with `create_reader_datamovement_config()` and `create_writer_datamovement_config()`, which leave implicit sync on. Passing `/*disable_dfb_implicit_sync_for_all=*/true`, as the sibling `tilize` block factory does since #57616 (2026-09-28), turns it off. Both kernels already do their own `reserve_back`/`push_back`, `wait_front`/`pop_front` and NoC barriers, which is what explicit sync needs. The helper's comment also says that DM kernels doing many sub-tile ("stick") NoC transfers stall the implicit credit accounting, and this writer writes sticks.
  - **The Quasar copy already does this.** All 8 factories of `ttnn.experimental.quasar.untilize` pass `disable_dfb_implicit_sync_for_all=true`; only the mainline `ttnn.untilize` factories leave implicit sync on. #57616 described all Metal 2.0 Quasar tilize/untilize factories as opting out, which holds for the Quasar copies but not for mainline `untilize`.
  - **Implicit sync is not new; the check is.** Implicit sync has been on by default for Quasar DFBs with a data movement endpoint since the first Metal 2.0 host API for Quasar (#41169, 2026-04-02, `disable_implicit_sync = false`). The mainline `untilize` block factory got its Metal 2.0 port on 2026-09-23 (#56280) and kept that default. What is new is the check from #52360 (2026-10-16); before it, the oversized value was cut to 8 bits without an error.
  - **The limit comes from the software descriptors, not from a hardware register, as far as the code shows.** The host stores the threshold in `dfb_dm0_isr_txn_threshold_t::threshold` (`uint8_t`, followed by 3 padding bytes) and the per-transaction count in `TxnDFBDescriptor::tiles_to_post` (`uint8_t`, 12 padding bytes in the struct). The firmware writes the threshold to the data movement command buffer register as a 32-bit value (`dataflow_buffer_init.h`), and craq-sim models that register as 16 bits. So my guess is that the hardware allows at least 16 bits, and widening the two fields would remove this limit; the hardware register width is not checked against the Quasar specification. At the time of the runs, no GitHub issue covered this error; issue #62783 (filed 2026-10-27) now does. Open PR #61406 ("untilize: split wide interleaved rows across the grid instead of one core") changes which factory wide rows use, but only in the mainline untilize and as a speed change, so whether it avoids this error is unknown.

## Recommendations

| # | Category | Recommendation |
|---|---|---|
| A | Block float format | Now: run as bfloat16, as in run 3 (`TTNN_GRAPH_OPS_BF8_AS_BF16=1`). This matches what the gpt-oss model run does on Quasar. Later: use MX formats. ttnn has the MX `DataType`s since PR #58175 (merged 2026-10-28); the ops still need to accept them (#58157). `mxint8` is the closest match to `bfloat8_b`, according to #58157, and `mxfp4` matches the gpt-oss expert weights, which the original checkpoint ships as MXFP4. MX also uses about as little DRAM as the block float formats (per 32x32 tile: `mxint8` 1056 B, `mxfp4` 288 B, bfloat16 2048 B), which bfloat16 cannot. |
| B | Legacy kernels | Test the `ttnn.experimental.quasar` versions (`add`, `multiply`, `linear`), as the two SDPA test files already do. The gpt-oss model run already routes these ops to the Quasar versions. |
| C | Compile error | Fixed on `main` by PR #61760 (closes #61354). Rerun the 15 cases on the new base. The gpt-oss model needs all four ops (`rms_norm`, `rotary_embedding_llama`, `softmax`, `typecast`). A route of `ttnn.typecast` to `ttnn.experimental.quasar.typecast` is no longer needed for this error; add one only if the mainline `typecast` still fails on Quasar. |
| D | Unsigned integer formats | `topk` and `scatter`: run on the host in the model for now (a host router), and ask whether `UInt16` can be added to Metal's Quasar format list. `tensor_getitem` on `UInt32`: the model's host embedding already avoids this slice. `ttnn.experimental.quasar.slice` does not help here: it has no `UInt32` handling. `topk` and `scatter` have no Quasar version. |
| E | Hang | For the 6 hangs that #60848 covers: #60848 is fixed by PR #62092, so rerun them on the new base. If the decode SDPA cases still hang, try llama's workaround: one call per KV head, so that each core handles one KV head. For the 2 matmul hangs: first rerun `04_n10240` and case 04 on the new base, which has PR #62081 (fix for #61887, sharded output) and PR #62092. If they still hang, report a new issue against the 1D multicast matmul on Quasar with a DRAM interleaved output (mainline and Quasar copy), with the experiment variants `04_n8192` (passes) and `04_n10240` (hangs) as the reproducer pair. For the gpt-oss `lm_head`, use one of the workarounds in "Matmul experiments" until it is fixed. Give large cases such as the `lm_head` matmul at least 10 min on the simulator. `ttnn.experimental.quasar.matmul` is a further implementation to compare for the matmul hangs. `sum`, `fast_reduce_nc` and `paged_update_cache` have no Quasar version. |
| F | Simulator error | First rerun this case together with the E cases on the new base, which has the #60848 fix (PR #62092). If it still fails, report it as a bug in the Quasar SDPA compute kernel, with this case as the reproducer and the note that it happens only with `sliding_window_size=128`. |
| G | Core grid | Expected on the 32-core simulated device. `nlp_concat_heads_decode` needs a change so that it can put several heads on one core; until then, use a workaround such as llama's `_install_quasar_concat_heads_grid_agnostic`. For `sparse_matmul`, the captured program config needs core row 5, so give the case a program config that fits 8x4. Its Quasar version does not help (see H). |
| H | Self-looped DFB | `ttnn.experimental.quasar.sparse_matmul` does not help: it is not ported to Metal 2.0 and still creates legacy `DataMovementKernel` kernels, so it fails like category B. The mainline op needs a scratchpad instead of the self-looped `sparsity` DFB, or the Quasar copy needs a Metal 2.0 port. For the model: done. `ttnn.sparse_matmul` runs on the host on Quasar (`make_host_sparse_matmul` in the gpt-oss `conftest.py`), until one of these fixes lands. |
| I | Descriptor overflow | For the model: done. `ttnn.untilize` runs as `ttnn.experimental.quasar.untilize` on Quasar (`quasar_ops` in the gpt-oss `conftest.py`), since the Quasar copy already uses explicit sync. For the op: pass `disable_dfb_implicit_sync_for_all=true` to the reader and writer configs of the mainline `untilize` factories, as the `tilize` block factory does. Issue #62783 now tracks this; add this case there as a second reproducer if it is not already in it, and ask for a Quasar test with more than 255 tiles in one core's block. |

### Matmul experiments

The file `models/experimental/ops/quasar/tests/gpt_oss_ops/exp_matmul_mcast.py` (not part of the generated suite; its name does not match `test_*.py`, so suite runs do not collect it) has variants of `test_matmul[04_32x2880_bf8_int-dram]`, each with one change:

- `04_a_in_l1`: input A in SRAM instead of DRAM, so `matmul` picks the plain matmul.
- `04_n<N>`: a weight N columns wide instead of 262144, with all tensors in DRAM as in case 04, so `matmul` picks the multicast matmul.
- `04_qsr_n<N>`: as `04_n<N>`, but run with `ttnn.experimental.quasar.matmul`.

Run them with `/localdev/fplavec/temp/run_matmul_experiments.sh [timeout] ["-k expression"]`, one pytest process per variant. Its summary has a `matmul_kernels` column (`multicast` or `plain`), taken from the kernels each variant built, so it shows which implementation ran. It cannot tell the Quasar matmul from the mainline one, because their kernel files have the same names.

```bash
TTNN_GRAPH_OPS_BF8_AS_BF16=1 EXP_MATMUL_ALL=1 /localdev/fplavec/temp/run_matmul_experiments.sh 3m "04_n8192 or 04_n10240"
# one variant directly
TTNN_GRAPH_OPS_BF8_AS_BF16=1 EXP_MATMUL_ALL=1 pytest -s --timeout=0 models/experimental/ops/quasar/tests/gpt_oss_ops/exp_matmul_mcast.py -k 04_n8192
```

The file collects only variants without a conclusive result; `EXP_MATMUL_ALL=1` collects all of them.

#### Results (2026-10-20, as bfloat16)

Three runs, all in `/localdev/fplavec/temp/gpt_oss_ops_runs/`: `matmul_exp_20261020_144453/` (436 s timeout), `matmul_exp_20261020_152129/` (3 min timeout) and `matmul_exp_20261020_153500/` (15 min timeout). "Matmul program" is the time from the start of the matmul program (the last program the test launches) to the end of the test or the timeout.

| Variant | Op | Implementation | Output width (tiles) | Result | Matmul program |
|---|---|---|---|---|---|
| `04_n5120` | `ttnn.matmul` | multicast | 160 | passed in 25 s | about 8 s |
| `04_n6144` | `ttnn.matmul` | multicast | 192 | passed in 32 s | about 8 s |
| `04_n8192` | `ttnn.matmul` | multicast | 256 | passed in 34 s | about 10 s |
| `04_n10240` | `ttnn.matmul` | multicast | 320 | **hang** | no progress in 3 min; about 10 s expected |
| `04_n12288` | `ttnn.matmul` | multicast | 384 | **hang** | no progress in 3 min |
| `04_n16384` | `ttnn.matmul` | multicast | 512 | **hang** | no progress in 7.2 min |
| `04_n65536` | `ttnn.matmul` | multicast | 2048 | **hang** | no progress in 6.4 min |
| `04_n262144` | `ttnn.matmul` | multicast | 8192 | **hang** (same as case 04) | no progress in 5.3 min |
| `04_qsr_n16384` | `ttnn.experimental.quasar.matmul` | (kernel file names are the same, so not visible) | 512 | **hang** | no progress in 3 min |
| `04_a_in_l1` | `ttnn.matmul` | plain | 8192 | **passed** in 8 min 49 s | about 6.5 min |
| `04_qsr_n262144` | `ttnn.experimental.quasar.matmul` | | 8192 | not run | most likely hangs, as at 16384 |

Conclusions:

- **The mainline 1D multicast matmul hangs when the output is wider than 8192 values (256 tiles)** and passes up to that width, with everything else as in case 04. My guess at the reason: the 8x4 grid has 32 cores, so 256 tiles give each core 8 output tiles and 320 tiles give it 10. A limit at 8 tiles per core suggests that the hang starts when the output of one core no longer fits in one block and the kernel loops over several blocks. This is not checked: the program config that `matmul` chose is not in the logs.
- **The Quasar copy of the matmul (`ttnn.experimental.quasar.matmul`) hangs too**, already at 16384. So routing `ttnn.matmul` to it in the `quasar_ops` patch would not help.
- **The plain multi-core matmul works for the full `lm_head` case.** `04_a_in_l1` passed with a 262144-wide output. It is slow on the simulator: its matmul program took about 6.5 min, compared with 3.7 min in `test_matmul[03_1x2880_bf8_int-l1]`. The earlier 436 s timeout was too short for it.
- **Effect on the gpt-oss model:** its `lm_head` runs `ttnn.matmul(hidden_states, self.lm_head_weight, dtype=ttnn.bfloat8_b)` (`models/experimental/ops/quasar/gpt_oss/tt/model.py`, `_forward_layers_and_head`) with this 262144-wide weight. Unless `hidden_states` is in SRAM, `matmul` picks the multicast matmul and the model will hang there. Possible workarounds, none tested in the model yet:
  1. On Quasar, move `hidden_states` to SRAM (L1 interleaved) before the `lm_head` matmul. That is what made `04_a_in_l1` take the plain matmul. In prefill, `hidden_states` has 128 rows (4 tile rows) instead of 1, so the selection may still pick a multicast config there. That needs a check.
  2. Split the `lm_head` matmul into pieces of at most 8192 columns (32 pieces) and concatenate them. The multicast matmul passes at that width, but this adds 64 ops per call.
  3. Pass an explicit program config for the plain matmul. `MatmulMultiCoreProgramConfig` is not exposed in Python (`ttnn.MatmulMultiCoreProgramConfig` does not exist), so this needs a change in ttnn.
- **For a bug report** against the multicast matmul on Quasar: `04_n8192` (passes) and `04_n10240` (hangs) are the closest pair. Both are variants of `test_matmul[04_32x2880_bf8_int-dram]` with a smaller weight, in `exp_matmul_mcast.py`.

The experiment file now collects only `04_qsr_n262144` by default; all other variants have a conclusive result and run only with `EXP_MATMUL_ALL=1`.

### Timeouts

`run_matmul_experiments.sh` and `rerun_ops_from_summary.sh` default to the longest runtime of any passing case in `/localdev/fplavec/temp/gpt_oss_ops_runs/*/summary.tsv`, plus 20%. Today that is `test_matmul[03_1x2880_bf8_int-l1]` with 363 s, so 436 s. The 262144-wide variants do as much work as that case, so they have about 20% of extra time before they count as hung.

### Which failures a `ttnn.experimental.quasar` op could fix

The op tests call the mainline ops. `ttnn.experimental.quasar` has Quasar versions of some of them. Checked by reading the Quasar ops' code, not by running them:

| Failing cases (run 3) | Quasar version | Would it fix the case? |
|---|---|---|
| `add` 10, `multiply` 7, `linear` 2 (B) | `quasar.add`, `quasar.multiply`, `quasar.linear` | **Yes, most likely.** They use Quasar data movement kernels, and the gpt-oss model run already uses `quasar.add` and `quasar.linear` successfully. |
| `typecast` 4 (C) | `quasar.typecast` | **Probably, but no longer needed.** Its compute kernel (`experimental/quasar/typecast/device/kernels/compute/eltwise_typecast.cpp`) does not include the eltwise chain helpers, so it avoids the `chain.inl` compile error. The mainline kernel includes `kernel_lib/eltwise/api/chain.hpp`, and PR #61760 fixes the compile error on `main`. |
| `untilize` 1 (I) | `quasar.untilize` | **Yes, most likely.** All 8 of its factories turn implicit sync off, so the 8-bit descriptor check does not apply. |
| `matmul` 01, 04 (E, hangs) | `quasar.matmul` | **No, for the wide case.** In the matmul experiments, `quasar.matmul` hangs at a 16384-wide output (`04_qsr_n16384`), as the mainline multicast matmul does from 10240 on. `matmul` 03 was only slow and passes. |
| `tensor_getitem` 01 (D, `UInt32` slice) | `quasar.slice` | **Probably not.** I found no handling of `UInt32` in it, so its DFB would still be `UInt32`. The model's host embedding avoids this slice anyway. |
| `sparse_matmul` 4 (G, H) | `quasar.sparse_matmul` | **No.** It is not ported to Metal 2.0: its factory (`experimental/quasar/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`) still creates its kernels with the legacy `tt_metal::CreateKernel(..., tt_metal::DataMovementConfig{...})`, so it would fail with the `DataMovementKernel` error. The mainline op is ported (#58129, 2026-10-07), which is why it gets further, to the self-looped DFB. |
| SDPA 3 (E, F) | already the Quasar versions | not applicable |
| `rms_norm`, `rotary_embedding_llama`, `softmax` (C); `topk`, `scatter` (D); `sum`, `fast_reduce_nc`, `paged_update_cache` (E); `nlp_concat_heads_decode` (G) | none with a Python binding | no; the Quasar `reduction/generic` code exists but is internal (used by `quasar.avg_pool2d`) |

**Consequence for the gpt-oss model run:** at the time of the runs, its `quasar_ops` fixture routed `ttnn.sparse_matmul` to `quasar.sparse_matmul`, which, from the code, will fail with the `DataMovementKernel` error, while the mainline `sparse_matmul` fails on the self-looped `sparsity` DFB. So the MoE expert matmul has no working device path on Quasar. The options are a host version of the expert matmul in the model (now done, see Status on 2026-10-30), a scratchpad instead of the self-loop in the mainline op (as a comment in the Quasar 2D matmul factory suggests: "replace the self-loop with a scratchpad/LocalTensorAccessor"), or a Metal 2.0 port of the Quasar copy.

Issue #56625 ("Analyze metal 2.0 ported gpt-oss ops on emulator/simulator", assigned to vsureshTT) tracks running this exact test suite on craq-sim and the emulator. These results belong there.

### Rerunning

All cases, as bfloat16:

```bash
TTNN_GRAPH_OPS_BF8_AS_BF16=1 /localdev/fplavec/temp/run_ops_each.sh \
    models/experimental/ops/quasar/tests/gpt_oss_ops /localdev/fplavec/temp/gpt_oss_ops_runs/bf16 5m
```

Only the cases that timed out, with more time (run 4 used 30 min; the default is now the longest passing runtime plus 20%):

```bash
TTNN_GRAPH_OPS_BF8_AS_BF16=1 /localdev/fplavec/temp/rerun_ops_from_summary.sh \
    /localdev/fplavec/temp/gpt_oss_ops_runs/20261019_204759/summary.tsv TIMEOUT
```
