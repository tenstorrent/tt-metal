# DeepSeek v3 fused-op unit tests

This table enumerates pytest tests under this folder and the cases they cover.

| Module | Operation | Unit test exists | Decode | Prefill 128 | Prefill 1k | Prefill 8k | Prefill 32k | Prefill 128k | ATOL D | ATOL P-128 | E2E Perf D (us) | E2E Perf P-128 (us) | Device Perf D (us) | Device Perf P-128 (us) | PCC D | PCC P-128 | Single device test | Random weights | Program_cache on/off | Tracing on/off | 100 iterations | Added to CI | Perf shown in superset | all tests passing | Path to test file | Last updated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---::--- | --- | --- |
| moe | ds_moe_all_gather | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 0.2 | 0.2 | 74.646 | 78.704 | kernel=75.073 us; op2op=838.841 us | kernel=82.140 us; op2op=55362.835 us | 1.0 | 1.0 | ⛔️ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❌ | models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_all_gather.py | 2026-02-02 |
| moe | ds_moe_fwd_moe | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | 0.2 | 0.2 | 9367.430 | 9884.181 | kernel=5420.380 us; op2op=1319.357 us | kernel=4013.948 us; op2op=113307.307 us | 0.9926398353764313 | 0.9926398353764313 | ⛔️ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_fwd_moe.py | 2026-02-01 |
| moe | ds_moe_gate_projection_sigmoid | ✅ | ✅ | ✅ | ⛔️ | ⛔️ | ✅ | ❌ | 0.18359375 | 0.0078125 | 229.611 us | 83.907 us | ❌ | ❌ | 0.9967648854123857 | 0.9999874209205265 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| moe | ds_moe_add_score_correction_bias | ✅ | ❌ | ❌ | ⛔️ | ⛔️ | ❌ | ❌ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | 2026-01-27 |  |
| moe | ds_moe_expert_selection | ✅ | ❌ | ❌ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❌ | ❌ | ❓ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | 2026-01-27 |  |
| moe | ds_moe_reduce_scatter | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | 706.534 | 659.518 | ❓ | ❓ | 0.999994234669188 | 0.999994234669188 | ⛔️ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| moe | ds_moe_repeat_permute_expert_weights | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| embedding | ds_embedding_fwd_embedding | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 1.0 | 1.0 | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| embedding | ds_embedding_fwd_typecast | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 1.0 | 1.0 | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| embedding | ds_embedding_fwd_all_gather | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 1.0 | 1.0 | ⛔️ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| embedding | ds_embedding_fwd_reduce_scatter | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 1.0 | 1.0 | ⛔️ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| embedding | ds_embedding_fwd_slice | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 1.0 | ❓ | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| lm_head | ds_lm_head_fwd_mesh_scatter | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 314.016 | 1364.877 | k=143.843/op2op=3936.990 | k=588.452/op2op=3322.758 | 1.0 | 1.0 | ⛔️ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| lm_head | ds_lm_head_projection_decode | ✅ | ✅ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | 94.550 | ❓ | k=90.649/op2op=978.210 | ❓ | 0.993489 | ❓ | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| lm_head | ds_lm_head_projection_prefill | ✅ | ⛔️ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 225.797 | ❓ | ❓ | ❓ | 0.993468 | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| mlp | ds_mlp_fwd_all_gather | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 671.926 | 920.434 | ❓ | ❓ | 1.0 | 1.0 | ⛔️ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| mla | paged_update_cache | ✅ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| mla | ds_mla_norm_and_rope | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| mla | ds_ag_reshape | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| mla | ds_mla_all_to_all_before_flash_mla | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| mla | ds_flash_mla | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| mla | ds_wo | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ✅ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| mla | ds_fused_wqkva | ✅ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| mla | ds_linear_with_input_dim | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| mla | ds_fused_q_rope_nope | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | ❓ | 2026-01-26 |  |
| moe | ds_moe_op | ✅ | ✅ | ✅ | ⛔️ | ⛔️ | ⛔️ | ❌ | 0.2 | 0.2 | 9368.412 | 9884.107 | kernel=5419.860 us; op2op=1278.531 us | kernel=4009.673 us; op2op=116188.915 us | 0.992639835 | 0.992639835 | ⛔️ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | 2026-01-26 |  |



Each row referrs to a fused op and it's related unit tests.

*Legend*

Decode/prefill categories
✅: verified to exist in code and test_results and pass
❌: verified to exist in code and test_results and fail
⛔️: verified to not exist based on the code and test_results
❓: can not be verified, data missing or inconsistent

atol/pcc/perf categories
- values if present (using option with program_cache, with trace, real weights; represents the actaul measurement, not the target), ❓ otherwise

Feature categories
✅: verified to exist in code and test_results
⛔️: verified to not exist based on the code and test_results
❓: can not be verified, data missing or inconsistent

Test cases / features:
- Unit test exists: whether the unit tests exists
- Decode: whether decode setting is tested
- Prefill (128,1k,8k,32k,128k): whether prefill is tested with the specified seqquence length
- ATOL D/P-128: latest reported ATOL target from logs or config for decode/prefill-128
- E2E Perf D/P-128 (us): latest e2e perf values from logs for decode/prefill-128
- Device Perf D/P-128 (us): latest device perf totals from logs (kernel/op-to-op) for decode/prefill-128
- PCC D/P-128: latest reported PCC from logs for decode/prefill-128
- Single deivce test: single device test that tests the first device's chunk of the workload (only for non-CCL ops)
- Random weights: random weight option available for tests
- Program_cache on/off: option available
- Tracing on/off: option available
- 100 iterations: does the test run 100 iterations when checking PCC/ATOL
- Added to CI: whether it's added to any CI pipeline; reflects special fused op unit pipeline in `.github/workflows/galaxy-deepseek-tests-impl.yaml`
- Perf shown in superset: whether any superset dashboard cisualizes the data
- Path to test file: path to the test file

Superset dashboard: https://superset.tenstorrent.com/superset/dashboard/4fa0fef8-cced-4a8e-8819-48aeed75dcee/?permalink_key=6JbY9pNEQaZ

Update instructions:
Go through the current fused op unit tests readme (table above) and verify each cell for it's status; update each cell to it's current status. Read the guide for creating the tests (AGENTS_GUIDE_ADD_TEST.md) for more details about the options, and test results files.
