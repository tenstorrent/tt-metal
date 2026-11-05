# Auto review: branch `fplavec/quasar_sparse_matmul`

- **Target:** all commits on the branch since it left `main`: `git diff 5a249c7b528 fa2281c70d6` (6 files). Reviewers 2 and 3 also saw the fixes of the earlier reviewers as uncommitted edits.
- **BASE:** `5a249c7b5289339d32eb63f8f98d06b89f4d7681` (`git merge-base origin/main HEAD`).
- **BASELINE:** `fa2281c70d62dfed49d02134c85150de26aeb046` (HEAD). All fixes are uncommitted edits on top of it.
- **Date:** 2026-11-05.

## 1. Summary per reviewer

| Reviewer | Reviews | Ended with | Issues raised |
|---|---|---|---|
| 1 | 3 of 5 | `NO FURTHER COMMENTS` | 5 |
| 2 | 3 of 4 | `NO FURTHER COMMENTS` | 4 (1 repeats reviewer 1) |
| 3 | 3 of 3 | `NO FURTHER COMMENTS` | 5 (1 repeats reviewer 1) |

Of the 12 distinct issues, 10 were fixed, 2 are out of scope, none were rejected, and none are unresolved. All fixes change comments only, except the two new `static_assert`s for R1-F3.

## 2. All issues raised

### Reviewer 1

- **R1-F1 (test coverage, medium).** No in-tree test can run the Quasar-only code, because every test of this op uses BFP weights and Quasar has no BFP format ([test_sparse_matmul.py:123-125](tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py#L123-L125), [test_sparse_matmul.py:638-645](tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py#L638-L645)). Untested on Quasar: the `batch_valid` handoff, the fp32 partials reload through SrcA, the NOC_0 multicast, and the scratchpad reads under `scoped_lock`.
  - **Out of scope.** The porting guide that the branch followed asks for two things: the existing suite passes unchanged on WH/BH, and the op builds and runs on Quasar (§9 and §10 of `git show e6f04315e01:docs/source/ttnn/ttnn/ai/quasar_porting.md`). It does not ask for new tests. All three reviewers accepted this verdict.
  - **Condition from reviewers 2 and 3:** this report is not committed, so put the follow-up in the PR description or a GitHub issue to keep it.
  - **Follow-up, as the reviewers suggested:**
    - Add `ttnn.bfloat16` to `in1_dtype` of the three `*_without_nnz` tests ([test_sparse_matmul.py:123-125](tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py#L123-L125), [333-335](tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py#L333-L335), [534-536](tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py#L534-L536)).
    - Add an in1 dtype argument to `_make_sparse_inputs`, so that `test_sparse_matmul_interm0_format_tile_size` covers fp32 with a K spill.
    - Add `ttnn.bfloat16` to `weight_dtype` of `test_indexed_bias_and_cache_rebinding` ([test_sparse_matmul_indexed.py:419-423](tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py#L419-L423)), to cover fused bias with fp32.
    - Skip the BFP values on Quasar. A `sim_quasar` CI entry is possible too (that leg uses a 3x2 grid).
  - **Script for the Quasar results so far.** They come from `qsr_sparse_check.py`, `qsr_env.sh` and `run_cases.sh` in `/tmp/claude-1211409539/-localdev-fplavec-git-2026-11-04-quasar-sparse-matmul-tt-metal/960470a7-da8c-490c-bbba-d06674f147f8/scratchpad`. That folder is temporary, so keep the script, or at least the case list in [section 4](#4-tests-and-experiments), with the follow-up. The earlier result table is in `git show e6f04315e01:ttnn/cpp/ttnn/operations/matmul/device/sparse/QUASAR_UPLIFT_REPORT.md`.
  - Raised again by reviewers 2 and 3.
- **R1-F2 (stale comment, low).** The comment on the `SPARSITY` define still said "dataflow buffer" and "dfb::/tensor::" after the sparsity page became a scratchpad. **Fixed:** it now says "scratchpad" and "scratch::/tensor::" ([sparse factory:575-579](ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L575-L579)).
- **R1-F3 (misleading comment, low).**
  - **Problem:** both in0 kernel headers said the host binds `batch_valid` exactly when `get_batch_from_reader` is set on Quasar, but only the sparse factory does that. Only the compute kernel checked its own define. So a factory that missed the define on the in0 receiver would hang compute on the receiver nodes instead of failing to build.
  - **Fixed:** both data movement kernels got a Quasar-only `static_assert(!get_batch_from_reader)` for builds without the define ([in0 sender:202-207](ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp#L202-L207), [in0 receiver:54-59](ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp#L54-L59)). The headers were rewritten as a contract, then shortened under R2-F4.
- **R1-F4 (ambiguous comment, low).**
  - **Problem:** in the `reload_partials_to_dest` comment, "then" could be read as "with the mode on", and the comment described LLK packer internals.
  - **Fixed:** the first rewrite named the wrong cause; R3-F2 corrected it ([sparse factory:254-259](ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L254-L259)).
- **R1-F5 (long comments, low).**
  - **Problem:** four comments in the sparse factory repeated the `batch_valid` explanation, and "the readers" was ambiguous.
  - **Fixed:**
    - The full explanation is now only at `use_batch_valid_dfb` ([sparse factory:143-147](ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L143-L147)).
    - The DFB spec comment ([sparse factory:506-509](ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L506-L509)) and the define comment ([sparse factory:583-588](ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L583-L588)) keep only what is specific to them.
    - The comment above `batch_valid_binding` is removed.

### Reviewer 2

- **R2-F1 (test coverage, medium).** The same issue as R1-F1. **Out of scope**, with the same reason. The reviewer accepted it on the condition that the follow-up keeps the case list and the script location.
- **R2-F2 (performance, low).**
  - **Problem:** on Quasar, `read_sparsity` takes one `scoped_lock` per 2-byte flag. Each lock does a cache invalidate on acquire and a flush on release, with fences around both. This happens in the in0 sender ([in0 sender:193-199](ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp#L193-L199)) and in the in1 sender/writer ([in1 sender/writer:310-316](ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp#L310-L316)).
  - **Out of scope:**
    - The cost is a guess. This branch is a functional bring-up, and the simulator gives no useful timing.
    - The fix is one lock per page, declared after the page's `noc.async_read_barrier()` and held across the batch body, in both kernels.
    - After such a fix, the Quasar cases must run again.
    - On WH/BH the lock does nothing.
  - The reviewer accepted, and corrected my first description of how big the fix is.
- **R2-F3 (inaccurate comment, low).**
  - **Problem:** the DFB spec comment said two entries let an in0 kernel post the next flag while compute works on the current batch. Compute pops each flag before the batch body, so one entry already allows that.
  - **Fixed:** the comment now says two entries let an in0 kernel run up to two batches ahead, for example past a skipped batch ([sparse factory:506-509](ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L506-L509)).
- **R2-F4 (comment about distant code, low).**
  - **Problem:** the in0 kernel headers listed the `static_assert`s of the other kernels. I had added that text for R1-F3.
  - **Fixed:** each header keeps the contract and speaks only for its own kernel ([in0 sender:24-29](ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp#L24-L29), [in0 receiver:13-18](ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp#L13-L18)).

### Reviewer 3

- **R3-F1 (test coverage, medium).** The same issue as R1-F1. **Out of scope**, with the same reason. The reviewer accepted it on the condition that the owner puts the follow-up in the PR description or an issue.
- **R3-F2 (wrong cause in a comment, low).**
  - **Problem:** the R1-F4 rewrite said the matmul's SrcA/SrcB unpacks "do not work" in unpack-to-dest mode. The real cause is that DEST is not cleared after each pack in that mode, while `matmul_block` adds into DEST.
  - **Fixed:** the comment now names that cause, without naming LLK functions ([sparse factory:254-259](ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L254-L259)).
- **R3-F3 (stale comments, low).**
  - **Problem:** two comments in the compute kernel said the reload goes through the UnpackToDestFp32 alias whenever fused bias reads the partials through SrcA. The sparse factory on Quasar never makes the alias.
  - **Fixed:** one comment now states both conditions ([compute:111-117](ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp#L111-L117)), and the other names the define that selects the alias ([compute:229-231](ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp#L229-L231)).
- **R3-F4 (long comment about distant code, low).**
  - **Problem:** the Quasar `#else` comment repeated the `static_assert` message. It also said "Mirrors the experimental Quasar bmm kernel", which is no longer true.
  - **Fixed:** the comment is now one line ([compute:345-351](ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp#L345-L351)).
- **R3-F5 (stale comment, low).**
  - **Problem:** the sparse factory comment above `bias_reload_alias` made the alias depend on bias alone.
  - **Fixed:** the comment now also names `reload_partials_to_dest` as a condition ([sparse factory:261-267](ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L261-L267)).

## 3. Unresolved issues

No unresolved issues.

## 4. Tests and experiments

**Before the review.** These ran on the code of `e6f04315e01`; HEAD `fa2281c70d6` only deletes two `.md` files.

- **Quasar simulator (craq-sim), LLK asserts on and off.** All 14 cases pass:
  - `--mode b_sparse --no-nnz`, with `--zero-every 0` and with `--zero-every -1`
  - `--mode a_sparse --no-nnz`
  - `--mode ab_sparse --no-nnz`
  - `--mode b_sparse --no-nnz` with `--grid 4 2`, and with `--grid 4 4 --nt 16 --s 4 --experts 8`
  - `--mode b_sparse --no-nnz`: `--tile-h 16`; `--ckc --fp32`; `--ckc --l1acc`; `--repeat 2`
  - `--mode b_sparse`
  - `--mode gather --experts 8 --bias`, with and without `--ckc --fp32`
- **Wormhole n300, sparse suite.** `test_sparse_matmul.py` and `test_sparse_matmul_indexed.py`: 93 passed and 2 skipped, with the same outcome for every test as before the uplift. The 2 skipped tests are Quasar-only tests of `ttnn.experimental.quasar.sparse_matmul`.
- **Wormhole n300, script.** The script's `nnz=None` cases also pass.

**During the review.**

- **After R1-F3 (new `static_assert`s).** On the Quasar simulator with `TT_METAL_LLK_ASSERTS=1`, five cases build and pass: B sparse with `nnz`; B sparse with `nnz=None`; A sparse with `nnz=None`; indexed with bias; and a dense `ttnn.matmul` with a 1D mcast_in0 config, which uses the same two in0 kernels.
- **After R3-F2 to R3-F4.** On the Quasar simulator, B sparse with `nnz=None` and indexed with bias and fp32 pass. On Wormhole, `pytest tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py -k "without_nnz or interm0_format"` gives 8 passed and 2 skipped; the 2 skipped tests are the Quasar-only tests above.
- **Final build after the last edit (R3-F5):** `./build_metal.sh -e --enable-fake-kernels-target` passes, with no errors, and recompiles the sparse factory.

**Not run:**

- The full Wormhole sparse suite after the review edits. Since its last full run, the code changed only in comments and in the two Quasar-only asserts, which WH/BH do not compile.
- Dense matmul suites. WH/BH compile the same code as before.
- Blackhole, because this machine has none. The "Sanity tests" workflow covers it.

No temporary instrumentation was added.

## 5. Changed files

All fixes are uncommitted edits on top of `fa2281c70d6`; nothing is staged. `git diff --stat fa2281c70d6`:

```
 ...large_block_zm_fused_bias_activation_metal2.cpp | 21 +++++------
 .../reader_bmm_tile_layout_in0_receiver_metal2.cpp |  7 ++--
 ...r_bmm_tile_layout_in0_sender_padding_metal2.cpp |  6 +++-
 ...e_matmul_multicore_reuse_mcast_1d_optimized.cpp | 42 +++++++++++-----------
 4 files changed, 39 insertions(+), 37 deletions(-)
```

Full paths:

- `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`

No files were created.
