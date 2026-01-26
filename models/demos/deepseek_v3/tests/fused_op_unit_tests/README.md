# DeepSeek v3 fused-op unit tests

This table enumerates pytest tests under this folder and the cases they cover.

| Module | Operation | Unit test exists | Decode | Prefill 128 | Prefill 1k | Prefill 8k | Prefill 32k | Prefill 128k | ATOL D | ATOL P-128 | E2E Perf D (us) | E2E Perf P-128 (us) | Device Perf D (us) | Device Perf P-128 (us) | PCC D | PCC P-128 | Single device test | Random weights | Program_cache on/off | Tracing on/off | 100 iterations | Added to CI | Perf shown in superset | Path to test file | Last updated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| moe | ds_moe_op | ✅ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_op.py` | 2026-01-26 |
| moe | ds_moe_fwd_moe | ✅ | ✅ | ✅ | ❓ | ❓ | ⛔️ | ❓ | ❓ | ❓ | 11267.595 | 11754.253 | ❓ | ❓ | 0.992639835 | 0.992639835 | ⛔️ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_fwd_moe.py` | 2026-01-26 |
| moe | ds_moe_gate_projection_sigmoid | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | 0.18359375 | 0.0078125 | 229.800 us | 1037.366 us | kernel=228.541 us; op2op=821.123 us | kernel=84.189 us; op2op=632.061 us | 0.9967649 | 0.9999874 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_gate_projection_sigmoid.py` | 2026-01-26 |
| moe | ds_moe_add_score_correction_bias | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | 0.1875 | 0.0625 | 15.874 us | 455.005 us | kernel=15.326 us; op2op=978.213 us | kernel=13.926 us; op2op=407.632 us | 0.9948338 | 0.9984173 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_add_score_correction_bias.py` | 2026-01-26 |
| moe | ds_moe_expert_selection | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | 0.015625 | 0.01171875 | 237119.722 us | 236679.265 us | kernel=120.884 us; op2op=84292.847 us | kernel=127.346 us; op2op=83854.761 us | 0.9603181 | 0.9666768 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_expert_selection.py` | 2026-01-26 |
| moe | ds_moe_all_gather | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_all_gather.py` | 2026-01-26 |
| moe | ds_moe_reduce_scatter | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | 706.534 | 659.518 | ❓ | ❓ | 0.999994234669188 | 0.999994234669188 | ⛔️ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_reduce_scatter.py` | 2026-01-26 |
| moe | ds_moe_repeat_permute_expert_weights | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_repeat_permute_expert_weights.py` | 2026-01-26 |
| embedding | ds_embedding_fwd_embedding | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 1.0 | 1.0 | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding_fwd_embedding.py` | 2026-01-26 |
| embedding | ds_embedding_fwd_typecast | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 1.0 | 1.0 | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding_fwd_typecast.py` | 2026-01-26 |
| embedding | ds_embedding_fwd_all_gather | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 1.0 | 1.0 | ⛔️ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding_fwd_all_gather.py` | 2026-01-26 |
| embedding | ds_embedding_fwd_reduce_scatter | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 1.0 | 1.0 | ⛔️ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding_fwd_reduce_scatter.py` | 2026-01-26 |
| embedding | ds_embedding_fwd_slice | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 1.0 | ❓ | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding_fwd_slice.py` | 2026-01-26 |
| lm_head | ds_lm_head_fwd_mesh_scatter | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 314.016 | 1364.877 | k=143.843/op2op=3936.990 | k=588.452/op2op=3322.758 | 1.0 | 1.0 | ⛔️ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head_fwd_mesh_scatter.py` | 2026-01-26 |
| lm_head | ds_lm_head_projection_decode | ✅ | ✅ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | 94.550 | ❓ | k=90.649/op2op=978.210 | ❓ | 0.993489 | ❓ | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head_projection_decode.py` | 2026-01-26 |
| lm_head | ds_lm_head_projection_prefill | ✅ | ⛔️ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 225.797 | ❓ | ❓ | ❓ | 0.993468 | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head_projection_prefill.py` | 2026-01-26 |
| mlp | ds_mlp_fwd_all_gather | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 671.926 | 920.434 | ❓ | ❓ | 1.0 | 1.0 | ⛔️ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_mlp_fwd_all_gather.py` | 2026-01-26 |
| mla | paged_update_cache | ✅ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_paged_update_cache.py` | 2026-01-26 |
| mla | ds_mla_norm_and_rope | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_mla_norm_and_rope.py` | 2026-01-26 |
| mla | ds_ag_reshape | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_ag_reshape.py` | 2026-01-26 |
| mla | ds_mla_all_to_all_before_flash_mla | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_mla_all_to_all_before_flash_mla.py` | 2026-01-26 |
| mla | ds_flash_mla | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_flash_mla.py` | 2026-01-26 |
| mla | ds_wo | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ✅ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_wo.py` | 2026-01-26 |
| mla | ds_fused_wqkva | ✅ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_fused_wqkva.py` | 2026-01-26 |
| mla | ds_linear_with_input_dim | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ⛔️ | ⛔️ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_linear_with_input_dim.py` | 2026-01-26 |
| mla | ds_fused_q_rope_nope | ✅ | ❓ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ✅ | ✅ | ✅ | ✅ | ❓ | ✅ | ❓ | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_fused_q_rope_nope.py` | 2026-01-26 |

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
