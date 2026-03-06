# MTP Debug Diary

Date: 2026-03-06
Repo: `tt-metal`
Focus: `models/demos/deepseek_v3/tests/test_mtp.py::test_mtp_verify_batching_aliasing`

## Goal
Debug the low MTP acceptance rate / prompt-lane corruption seen in the DeepSeek verify-batching + aliasing path.

## Starting Point
Observed before this session:
- `test_mtp_accept_rate_and_perf`: about `0.8` accept rate
- `test_mtp_prefill_priming`: about `0.8` accept rate
- `test_mtp_verify_batching_aliasing`: about `0.5` accept rate, and in the latest short run prompt-lane parity also regressed

Initial hypothesis:
- The core MTP predictor is likely correct.
- The bug is likely in the aliased verify decode path, especially cache updates or cache visibility under aliased page tables.

## Concrete Plan
1. Consolidate debug logging under `DEBUG_MTP=1` so the debug workflow is reproducible with one switch.
2. Re-run the short verify-batching test with unified debug enabled and capture:
   - aliased page-table layout
   - split cache-update masks / positions
   - prompt-lane behavior under verify aliasing
3. If the first debug pass fails in the debug harness itself, remove the failing debug-only path and rerun.
4. Use the result to separate:
   - debug harness failures
   - bad split-update behavior
   - broader aliasing / cache corruption

## Code Changes Before Experiments
- Added `DEBUG_` env passthrough to `ttnn/ttnn/distributed/ttrun.py` so `DEBUG_MTP=1` reaches all ranks.
- Consolidated verify-batching debug controls in `models/demos/deepseek_v3/tests/test_mtp.py` under `DEBUG_MTP=1`.
- Consolidated MTP alias / page-table debug controls in `models/demos/deepseek_v3/tt/generator.py` under `DEBUG_MTP=1`.
- Consolidated MTP alias / update debug controls in `models/demos/deepseek_v3/tt/mla/mla1d.py` under `DEBUG_MTP=1`.
- Removed remaining per-feature debug env toggles so the workflow uses `DEBUG_MTP=1` instead of a mix of `DEEPSEEK_MTP_*DEBUG` env vars.

## Environment / Run Workflow
Per `tt-machines`, all TT runs were done with:
```bash
source ../setup_metal.sh
source python_env/bin/activate
```

Verified from `../setup_metal.sh`:
- `DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528`
- `DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/dev`
- `MESH_DEVICE=DUAL`
- `TT_METAL_HOME=/home/jrock/wa/tt-metal`

## Syntax Check
Ran:
```bash
python3 -m py_compile \
  ttnn/ttnn/distributed/ttrun.py \
  models/demos/deepseek_v3/tests/test_mtp.py \
  models/demos/deepseek_v3/tt/mla/mla1d.py \
  models/demos/deepseek_v3/tt/generator.py
```

Result: pass.

---

## Experiment 1: First Unified Debug Run
Command:
```bash
source ../setup_metal.sh
source python_env/bin/activate
DEBUG_MTP=1 TT_LOG_HOST_RANK=0 DEEPSEEK_V3_MTP_VERIFY_MAX_PROMPTS=2 \
  /home/shared/scripts/ds-run pytest models/demos/deepseek_v3/tests/test_mtp.py::test_mtp_verify_batching_aliasing -s \
  2>&1 | tee logs/mtp_debug_exp1.log
```

Outcome:
- Run did not reach the MTP test body.
- It hit a TT init failure and then stalled.
- Per `tt-machines`, I stopped the run and reset the machine.

Relevant log excerpt from `logs/mtp_debug_exp1.log`:
```text
Read unexpected run_mailbox value: 0x40 (expected 0x80 or 0x0)
TT_FATAL: Read unexpected run_mailbox value from core (x=20,y=16)
```

Conclusion:
- This run was not useful for MTP behavior.
- The machine needed reset before continuing.

Follow-up:
- Ran `/home/shared/scripts/reset.sh` after sourcing `../setup_metal.sh`.
- Waited 30 seconds before rerunning, per `tt-machines`.

---

## Experiment 2: Clean Rerun With `source ../setup_metal.sh`
Command:
```bash
source ../setup_metal.sh
source python_env/bin/activate
DEBUG_MTP=1 TT_LOG_HOST_RANK=0 DEEPSEEK_V3_MTP_VERIFY_MAX_PROMPTS=2 \
  /home/shared/scripts/ds-run pytest models/demos/deepseek_v3/tests/test_mtp.py::test_mtp_verify_batching_aliasing -s \
  2>&1 | tee logs/mtp_debug_exp1_reset.log
```

Outcome:
- This run reached the test body.
- It failed before any aliased verify decode because the debug-only base-compare path tried to clone the full on-device KV cache.
- The failure is in the debug harness, not the original aliasing bug.

Relevant log excerpt from `logs/mtp_debug_exp1_reset.log`:
```text
> kv_cache_snapshot = [ttnn.clone(tensor) for tensor in kv_cache_current]
E RuntimeError: TT_FATAL @ /home/jrock/wa/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:434: false
E Out of Memory: Not enough space to allocate 5013504 B DRAM buffer across 12 banks
```

Conclusion:
- `compare_base` implemented as device-side `ttnn.clone(...)` is too expensive on this configuration.
- This does not provide information about the original MTP aliasing failure.

Action taken:
- Disabled `compare_base` in the unified `DEBUG_MTP` path.
- Kept a log reason in the test so the run records why it is off.

---

## Experiment 3: Rerun With Base Compare Disabled
Command:
```bash
source ../setup_metal.sh
source python_env/bin/activate
DEBUG_MTP=1 TT_LOG_HOST_RANK=0 DEEPSEEK_V3_MTP_VERIFY_MAX_PROMPTS=2 \
  /home/shared/scripts/ds-run pytest models/demos/deepseek_v3/tests/test_mtp.py::test_mtp_verify_batching_aliasing -s \
  2>&1 | tee logs/mtp_debug_exp2_no_base_compare.log
```

Outcome:
- This run no longer OOMed in `compare_base`.
- It stalled before the first `before_seed_decode` log.
- The most likely culprit was the full KV host dump path in `_dump_kv_cache`.

Observation:
- `logs/mtp_debug_exp2_no_base_compare.log` stopped advancing after the alias-table setup logs at `2026-03-06 14:43:44 UTC`, while the TT pytest process kept burning CPU.

Conclusion:
- The full KV host dump path is too heavy for the interactive debug loop.
- It is not required to reproduce the original prompt-lane mismatch.

Action taken:
- Disabled the KV dump path in the unified `DEBUG_MTP` flow.
- Kept a log reason in the test so the run records why it is off.

---

## Experiment 4: Rerun With Only Lightweight Debug Enabled
Command:
```bash
source ../setup_metal.sh
source python_env/bin/activate
DEBUG_MTP=1 TT_LOG_HOST_RANK=0 DEEPSEEK_V3_MTP_VERIFY_MAX_PROMPTS=2 \
  /home/shared/scripts/ds-run pytest models/demos/deepseek_v3/tests/test_mtp.py::test_mtp_verify_batching_aliasing -s \
  2>&1 | tee logs/mtp_debug_exp3_no_base_no_kv.log
```

Outcome:
- This run completed quickly enough to be useful.
- It reproduced the original prompt-lane corruption cleanly.
- Final result:
  - prompt match rate `23/32 = 0.719`
  - accept rate `18/32 = 0.562`

Relevant setup log from `logs/mtp_debug_exp3_no_base_no_kv.log`:
```text
DEBUG_MTP enabled: compare_base=False compare_steps=0 enable_verify_table=True enable_kv_log=False
compare_base_reason=disabled because device-side KV snapshot cloning OOMs in verify debug runs
kv_log_reason=disabled because full KV host dumps stall before verify decode
```

### 4A. Aliased Page Tables Are As Expected
Relevant log excerpt:
```text
decode_page_tables[0] host alias debug shape=(4, 64)
row0[:8]=[0, 1, 2, 3, 4, 5, 6, 7]
row1[:8]=[0, 1, 2, 3, 4, 5, 6, 7]
row2[:8]=[128, 129, 130, 131, 132, 133, 134, 135]
row3[:8]=[128, 129, 130, 131, 132, 133, 134, 135]
eq01=True eq23=True
```

Conclusion:
- The verify alias page-table construction itself looks correct for the tested 2-prompt setup.
- The bad behavior is happening after the aliasing is constructed, not because alias rows are mapped incorrectly at the high level.

### 4B. The Aliased Update Path Definitely Activates
Relevant log excerpt:
```text
MTP alias_mask.any() detected: [0, 1, 0, 1]
```

Conclusion:
- The special aliased cache-update path in `MLA1D._fwd_decode_paged_update_cache` is definitely being exercised.
- This keeps the split-update path as the primary suspect.

### 4C. The Split-Update Debug Print Is Broken Right On The Case We Care About
Relevant log excerpt:
```text
MTP update mask setup: alias_mask=[0, 0, 0, 0] per_shard=4 total_elems=4 logical_shape=(4,) num_devices=64 num_devices_eff=1 is_sharded=False row_idx=None
MTP split update tensor debug failed: shape '[4]' is invalid for input of size 256
```

Conclusion:
- The existing debug print for `MTP split update tensors` is itself buggy.
- It is trying to reshape a replicated host tensor without first slicing a single replica.
- This is a debug-only bug, but it blocks the most useful internal visibility into `prompt_mask`, `spec_mask`, `prompt_pos`, and `spec_pos`.

### 4D. Prompt-Lane Corruption Starts In Row 2 Early
Relevant verify-table excerpts:
```text
step 0
req 0 row 0 pos 1 next_pred 223   match_gt MATCH spec_prev 223   accept ACCEPT
req 1 row 2 pos 1 next_pred 16363 match_gt MATCH spec_prev 76    accept REJECT

step 1
req 0 row 0 pos 2 next_pred 643   match_gt MATCH spec_prev 643   accept ACCEPT
req 1 row 2 pos 2 next_pred 5620  match_gt MISMATCH spec_prev 57575 accept REJECT

step 2
req 0 row 0 pos 3 next_pred 27    match_gt MATCH spec_prev 27    accept ACCEPT
req 1 row 2 pos 3 next_pred 5034  match_gt MISMATCH spec_prev 71391 accept REJECT

step 3
req 0 row 0 pos 4 next_pred 695   match_gt MATCH spec_prev 695   accept ACCEPT
req 1 row 2 pos 4 next_pred 28    match_gt MISMATCH spec_prev 13598 accept REJECT
```

Conclusion:
- Row `2` breaks immediately after the first reject.
- The pattern is consistent with cache corruption localized to the aliased prompt/spec pair `row 2 / row 3`.
- This is exactly the behavior expected if the aliased split-update path is writing or masking incorrectly.

### 4E. Corruption Later Spreads Beyond Row 2
Later verify-table excerpts:
```text
step 7
req 0 row 0 pos 8 next_pred 7929  match_gt MISMATCH spec_prev 7929  accept ACCEPT
req 1 row 2 pos 8 next_pred 39898 match_gt MATCH    spec_prev 39898 accept ACCEPT

step 8
req 0 row 0 pos 9 next_pred 25    match_gt MISMATCH spec_prev 20    accept REJECT
req 1 row 2 pos 9 next_pred 10060 match_gt MATCH    spec_prev 33522 accept REJECT

step 14
req 0 row 0 pos 15 next_pred 929   match_gt MISMATCH spec_prev 736   accept REJECT
req 1 row 2 pos 15 next_pred 57212 match_gt MISMATCH spec_prev 16363 accept REJECT
```

Conclusion:
- The failure is not permanently isolated to one pair.
- Row `0` is initially healthy, but later also becomes corrupted.
- That suggests corruption accumulates over steps, consistent with bad cache updates rather than a pure one-step logits bug.

### 4F. Final Assertion
Relevant log excerpt:
```text
MTP verify batching prompt match rate: 23/32 = 0.719
MTP verify batching accept rate: 18/32 = 0.562
AssertionError: Prompt-lane mismatch under verify batching: 23/32
```

Conclusion:
- The original issue is reproduced under the cleaned-up `DEBUG_MTP` flow.
- The failure is prompt-lane corruption first, not just low speculative acceptance.

---

## Experiment 5: Fix Split-Update Debug And Re-Run
Command:
```bash
source ../setup_metal.sh
source python_env/bin/activate
DEBUG_MTP=1 TT_LOG_HOST_RANK=0 DEEPSEEK_V3_MTP_VERIFY_MAX_PROMPTS=2   /home/shared/scripts/ds-run pytest models/demos/deepseek_v3/tests/test_mtp.py::test_mtp_verify_batching_aliasing -s   2>&1 | tee logs/mtp_debug_exp4_alias_masks.log
```

Code change before this run:
- Fixed the split-update debug path in `models/demos/deepseek_v3/tt/mla/mla1d.py` so it can flatten replicated host tensors first and then slice one logical replica.
- Changed the debug gating so it logs the first few aliased calls, not just the first few total calls.

Outcome:
- The test still failed with the same final rates:
  - prompt match rate `23/32 = 0.719`
  - accept rate `18/32 = 0.562`
- The split-update inputs for the aliased call are now visible.

Relevant log excerpt from `logs/mtp_debug_exp4_alias_masks.log`:
```text
MTP alias_mask.any() detected: [0, 1, 0, 1]
MTP update mask setup: alias_mask=[0, 1, 0, 1] per_shard=4 total_elems=4 logical_shape=(4,) num_devices=64 num_devices_eff=1 is_sharded=False row_idx=None kind=alias
MTP split update tensors [alias]: raw_shape=(4,) raw_numel=4 positions=[[1, 2, 1, 2]] prompt_mask=[[1, 0, 1, 0]] spec_mask=[[0, 1, 0, 1]] prompt_pos=[[1, -1, 1, -1]] spec_pos=[[-1, 2, -1, 2]]
```

Conclusion:
- The aliased split-update inputs are logically correct for the first failing verify step.
- This weakens the earlier hypothesis that the host-side mask construction itself is wrong.
- It also makes `paged_update_cache` handling of the generated tensors, or something above it in full decode, more likely.

---

## Experiment 6: Insert Debug-Only Sync Between Prompt And Spec Cache Updates
Command:
```bash
source ../setup_metal.sh
source python_env/bin/activate
DEBUG_MTP=1 TT_LOG_HOST_RANK=0 DEEPSEEK_V3_MTP_VERIFY_MAX_PROMPTS=2   /home/shared/scripts/ds-run pytest models/demos/deepseek_v3/tests/test_mtp.py::test_mtp_verify_batching_aliasing -s   2>&1 | tee logs/mtp_debug_exp5_alias_sync.log
```

Code change before this run:
- Added a temporary debug-only `ttnn.synchronize_device(mesh_device)` between the prompt and spec `paged_update_cache` calls in `models/demos/deepseek_v3/tt/mla/mla1d.py`.
- This was only a diagnostic probe and has been removed after the run.

Outcome:
- The sync inserted successfully.
- It did not change the failure at all.
- Final result stayed exactly the same:
  - prompt match rate `23/32 = 0.719`
  - accept rate `18/32 = 0.562`

Relevant log excerpt from `logs/mtp_debug_exp5_alias_sync.log`:
```text
MTP split update tensors [alias]: raw_shape=(256,) raw_numel=256 positions=[[1, 2, 1, 2]] prompt_mask=[[1, 0, 1, 0]] spec_mask=[[0, 1, 0, 1]] prompt_pos=[[1, -1, 1, -1]] spec_pos=[[-1, 2, -1, 2]]
MTP alias split update debug sync inserted between prompt/spec cache updates
MTP verify batching prompt match rate: 23/32 = 0.719
MTP verify batching accept rate: 18/32 = 0.562
```

Important interpretation:
- `raw_shape=(256,) raw_numel=256` is expected here because the full-model `position_idxs` tensor is replicated across the 64-device mesh.
- The debug path slices the first logical replica, which still gives the same logical aliased positions: `[[1, 2, 1, 2]]`.

Conclusion:
- Simple prompt/spec launch ordering or lack of immediate device visibility is not the issue.
- The prompt corruption survives even when the two update launches are explicitly separated by a device sync.

---

## Experiment 7: Run The Existing Low-Level Aliasing Unit Test
Command:
```bash
source ../setup_metal.sh
source python_env/bin/activate
/home/shared/scripts/ds-run pytest   models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_paged_update_cache_deepseek.py::test_paged_update_cache_verify_aliasing   -s 2>&1 | tee logs/mtp_debug_exp6_unit_alias.log
```

Outcome:
- The dedicated low-level aliasing unit test passed.

Relevant log excerpt from `logs/mtp_debug_exp6_unit_alias.log`:
```text
PASSED models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_paged_update_cache_deepseek.py::test_paged_update_cache_verify_aliasing[device_params0-1x2_grid]
1 passed, 1 warning in 7.08s
```

Conclusion:
- The basic split prompt/spec `paged_update_cache` strategy with aliased page tables works in isolation.
- That materially lowers the probability that the root cause is in the bare low-level cache op itself.

Important caveat:
- The existing unit test does **not** fully match the full-model layout.
- In the unit test, the prompt/spec index tensors are explicitly sharded.
- In the full model, the debug logs show:
  - `logical_shape=(4,)`
  - `is_sharded=False`
  - `raw_numel=256` on host because the tensor is replicated across 64 devices.
- So the low-level unit test passing does **not** clear the replicated, non-sharded `update_idxs_tensor` path used by the full model.

---

## Experiment 8: Run A Matching Low-Level Unit Test For Replicated Update Indices
Code change before this run:
- Added `test_paged_update_cache_verify_aliasing_replicated_update_idxs` to `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_paged_update_cache_deepseek.py`.
- This new test mirrors the full-model layout more closely:
  - local logical `update_idxs_tensor` shape `(4,)`
  - replicated prompt/spec update indices across the mesh
  - batch-sharded update tensor
  - same aliased prompt/spec page-table pattern
- Also switched the unit-test debug gate to `DEBUG_MTP=1` so the unit test uses the same debug knob as the full-model path.

Command:
```bash
source ../setup_metal.sh
source python_env/bin/activate
DEBUG_MTP=1 /home/shared/scripts/ds-run pytest \
  models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_paged_update_cache_deepseek.py::test_paged_update_cache_verify_aliasing_replicated_update_idxs \
  -s 2>&1 | tee logs/mtp_debug_exp8_replicated_unit.log
```

Outcome:
- The new replicated-index unit test passed.
- During test bringup, the first draft of the Python reference indexed per physical device instead of per logical mesh column; I corrected that to `shard_idx = device_idx % dp_factor` before the final run.

Relevant log excerpts from `logs/mtp_debug_exp8_replicated_unit.log`:
```text
replicated update_idxs (local): tensor([ 0, -1,  1, -1], dtype=torch.int32)
```

```text
after_replicated_spec_update kvcache device0:
tensor([[[[2., 2., 2., ..., 2., 2., 2.],
          [3., 3., 3., ..., 3., 3., 3.],
```

```text
after_replicated_spec_update kvcache device1:
tensor([[[[4., 4., 4., ..., 4., 4., 4.],
          [5., 5., 5., ..., 5., 5., 5.],
```

```text
after_replicated_spec_update kvcache device8:
tensor([[[[2., 2., 2., ..., 2., 2., 2.],
          [3., 3., 3., ..., 3., 3., 3.],
```

```text
after_replicated_spec_update kvcache device63:
tensor([[[[16., 16., 16., ..., 16., 16., 16.],
          [17., 17., 17., ..., 17., 17., 17.],
```

```text
PASSED models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_paged_update_cache_deepseek.py::test_paged_update_cache_verify_aliasing_replicated_update_idxs[device_params0-1x2_grid]
1 passed, 1 warning in 3.41s
```

Conclusion:
- The low-level aliased split-update path also works when `update_idxs_tensor` is replicated and local, matching the full-model layout more closely.
- The log also confirms the expected mesh behavior:
  - device `0` and device `8` match because the update tensor is replicated over mesh rows
  - device `1` differs from device `0` because updates are sharded over mesh columns
  - device `63` carries the last column’s distinct prompt/spec values
- This removes the replicated, non-sharded `update_idxs_tensor` path as the most likely low-level root cause.

---

## Current Conclusions
1. The original verify-batching + aliasing failure is reproduced cleanly under `DEBUG_MTP=1`.
2. The first two debug-only mechanisms were too heavy and had to be removed from the live debug loop:
   - device-side KV snapshot clone for base compare: OOM
   - full KV host dump: stalled before verify decode
3. The aliased page-table construction itself looks correct.
4. The aliased split-update path definitely activates, and the generated alias masks / positions look logically correct for the first failing step:
   - `positions=[[1, 2, 1, 2]]`
   - `prompt_pos=[[1, -1, 1, -1]]`
   - `spec_pos=[[-1, 2, -1, 2]]`
5. Adding an explicit device sync between the prompt and spec cache-update calls does not change the result.
6. The existing low-level unit test for aliased split `paged_update_cache` passes.
7. The new low-level unit test for replicated, local `update_idxs_tensor` also passes.
8. Therefore the most likely failure surface has moved further upstack:
   - read-side / decode-side behavior under aliased page tables
   - or a full verify-batching lane-selection / aliasing pattern difference that the low-level unit tests still do not cover

## Most Likely Issue Now
Most likely root-cause buckets, in order:
1. `Read-side / attention-side aliasing bug during full decode`
- `MLA1D.forward_decode` updates the paged cache and then immediately runs FlashMLA with the aliased page table.
- Both low-level aliased cache-update microtests now pass, including the replicated-index case.
- The remaining suspect is how the full decode path consumes that aliased cache.

2. `The full verify-batching test aliases more lanes than the microtests`
- The short full-model run still aliases odd local rows globally, while the selected prompt/spec pairs are only a subset of users.
- Even if that is not the final root cause, it is still a meaningful difference from the passing unit test.

3. `A higher-level page-table / lane-selection mismatch outside bare paged_update_cache`
- The low-level op is now tested in both sharded-index and replicated-index forms.
- What is still untested in isolation is the exact full-model combination of:
  - aliased verify page tables
  - immediate decode readback
  - prompt/spec accept-reject stepping across multiple verify iterations

## Next Steps
1. Focus on the read side in `models/demos/deepseek_v3/tt/mla/mla1d.py`:
   - confirm how the aliased cache is consumed by `_fwd_decode_flash_mla`
   - confirm whether rejected spec positions can still influence prompt-lane attention on the next step
2. Add one more focused full-model probe to reduce test-harness differences:
   - make the verify aliasing test alias only the specific prompt/spec rows being checked, not every odd row globally
   - then re-run the short `DEBUG_MTP=1` command and compare prompt parity
3. Keep the improved split-update debug logging in place; it is now useful and stable under `DEBUG_MTP=1`.
4. Keep the new replicated-index unit test; it closes an important low-level gap and should guard against future regressions in the cache-update path.

## Files Touched During Debugging
- `ttnn/ttnn/distributed/ttrun.py`
- `models/demos/deepseek_v3/tests/test_mtp.py`
- `models/demos/deepseek_v3/tt/generator.py`
- `models/demos/deepseek_v3/tt/mla/mla1d.py`
- `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_paged_update_cache_deepseek.py`

## Run Logs
- `logs/mtp_debug_exp1.log`
- `logs/mtp_reset_after_exp1.log`
- `logs/mtp_debug_exp1_reset.log`
- `logs/mtp_reset_after_exp2.log`
- `logs/mtp_debug_exp2_no_base_compare.log`
- `logs/mtp_reset_after_exp3.log`
- `logs/mtp_debug_exp3_no_base_no_kv.log`
- `logs/mtp_debug_exp4_alias_masks.log`
- `logs/mtp_debug_exp5_alias_sync.log`
- `logs/mtp_debug_exp6_unit_alias.log`
- `logs/mtp_debug_exp8_replicated_unit.log`
