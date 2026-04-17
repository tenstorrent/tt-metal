# MTP Precision Experiments

## Baseline

- Date: 2026-04-17
- Test: `models/demos/deepseek_v3/tests/test_mtp.py::test_mtp_accept_rate_and_perf[False-0.75-device_params0]`
- Command:
  - `/home/shared/scripts/ds-run pytest -svv models/demos/deepseek_v3/tests/test_mtp.py -k test_mtp_accept_rate_and_perf`
- Reference:
  - `/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/dev/test_io_cache/mtp_full_model_seq128_baseline_20260417.pt`
- Result: `PASSED`

### Acceptance

- Accept rate: `29646 / 32512 = 0.912`
- Minimum required: `0.750`

### Throughput

- Rank `[1,0]`: `198.89 tokens/s` over `163.468s`
- Rank `[1,1]`: `188.03 tokens/s` over `172.905s`

### Pytest Timing

- Rank `[1,0]`: `206.76s`
- Rank `[1,1]`: `212.96s`

### Artifacts

- Log: `/tmp/mtp_accept_baseline_20260417.log`
- XML: `generated/test_reports/most_recent_tests.xml`
- Preserved baseline reference: `/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/dev/test_io_cache/mtp_full_model_seq128_baseline_20260417.pt`

## Experiment 1: BFP4 Attention Weights Only

Goal:
- Move MLA attention weights in the MTP module to `bfloat4_b`
- Keep matmul outputs at the baseline dtype behavior
- Keep MoE, norms, embedding, `eh_proj`, and LM head unchanged

Configuration prepared in code:
- `DEEPSEEK_V3_MLA_ATTN_WEIGHT_DTYPE=bfloat4_b`

Cache isolation:
- Attention precision overrides append a cache suffix so experiment weights do not reuse the baseline MTP cache.

### Status

- Date: 2026-04-17
- Result: `PASSED`
- Accept rate: `29129 / 32512 = 0.896`
- Minimum required: `0.750`

### Command Used

- Current intended run:
  - `DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized`
  - `DEEPSEEK_V3_MLA_ATTN_WEIGHT_DTYPE=bfloat4_b`
  - `DEEPSEEK_V3_CACHE_SPECS_JSONL=/tmp/mtp_bfp4_weight_only_specs_20260417.jsonl`
  - `/home/shared/scripts/ds-run pytest -svv models/demos/deepseek_v3/tests/test_mtp.py -k test_mtp_accept_rate_and_perf`

### Notes

- Cache isolation for this run used the suffix `61_layers_mtp_module_attn_w_bfp4`.
- This run used the baseline reference stream, which is now preserved at `/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/dev/test_io_cache/mtp_full_model_seq128_baseline_20260417.pt`.
- Cache-spec records show MLA attention weights saved as `BFLOAT4_B`, for example:
  - `decoder_block/mla/wq_kv_a.input_tensor_b.tensorbin`
  - `decoder_block/mla/wq_b.input_tensor_b.tensorbin`
  - `decoder_block/mla/wkv_b1.input_tensor_b.tensorbin`
  - `decoder_block/mla/wkv_b2.input_tensor_b.tensorbin`
  - `decoder_block/mla/wo.input_tensor_b.tensorbin`

### Throughput

- Rank `[1,0]`: `344.03 tokens/s` over `94.503s`
- Rank `[1,1]`: `371.28 tokens/s` over `87.567s`

### Pytest Timing

- Rank `[1,0]`: `153.27s`
- Rank `[1,1]`: `158.49s`

## Experiment 2: BFP4 Attention Weights With BFP4 Base Reference

Goal:
- Regenerate the base-model MTP reference with full-base MLA attention weights in `bfloat4_b`
- Re-run the BFP4-attention MTP acceptance test against that updated reference

Configuration:
- `DEEPSEEK_V3_MLA_ATTN_WEIGHT_DTYPE=bfloat4_b`

Cache isolation:
- Full base cache: `61_layers_attn_w_bfp4`
- MTP module cache: `61_layers_mtp_module_attn_w_bfp4`

### Reference Generation

- Test: `models/demos/deepseek_v3/tests/test_mtp.py::test_generate_mtp_reference_io[device_params0]`
- Result: `PASSED`
- Output reference:
  - `/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/dev/test_io_cache/mtp_full_model_seq128.pt`
- Preserved prior baseline reference:
  - `/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/dev/test_io_cache/mtp_full_model_seq128_baseline_20260417.pt`

### Notes

- `test_generate_mtp_reference_io` runs the full base generator with `enable_mtp=False` and saves `hidden_states`, `next_tokens`, and `start_tokens`.
- To avoid a full recache of unchanged MoE weights, the BFP4 base cache was built by cloning `61_layers` and rewriting only each block's `mla/` subtree.
- Cache-spec records for `/tmp/mtp_surgical_base_bfp4_specs_full_20260417.jsonl` show full-base MLA weights saved as `BFLOAT4_B` while norms remain `BFLOAT16`.

### Acceptance

- Date: 2026-04-17
- Result: `PASSED`
- Accept rate: `28441 / 32512 = 0.875`
- Minimum required: `0.750`

### Commands Used

- Reference generation:
  - `DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized`
  - `DEEPSEEK_V3_MLA_ATTN_WEIGHT_DTYPE=bfloat4_b`
  - `DEEPSEEK_V3_MTP_GENERATE_REFERENCE=1`
  - `/home/shared/scripts/ds-run pytest -svv models/demos/deepseek_v3/tests/test_mtp.py -k test_generate_mtp_reference_io`
- Acceptance test:
  - `DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized`
  - `DEEPSEEK_V3_MLA_ATTN_WEIGHT_DTYPE=bfloat4_b`
  - `/home/shared/scripts/ds-run pytest -svv models/demos/deepseek_v3/tests/test_mtp.py -k test_mtp_accept_rate_and_perf`

### Throughput

- Rank `[1,0]`: `390.47 tokens/s` over `83.264s`
- Rank `[1,1]`: `382.88 tokens/s` over `84.914s`

### Pytest Timing

- Reference generation rank `[1,0]`: `689.92s`
- Reference generation rank `[1,1]`: `695.02s`
- Acceptance rank `[1,0]`: `93.96s`
- Acceptance rank `[1,1]`: `99.18s`

### Artifacts

- Base-cache rewrite log: `/tmp/mtp_surgical_base_bfp4_specs_full_20260417.jsonl`
- Reference generation log: `/tmp/mtp_generate_reference_bfp4_base_fast_20260417.log`
- Acceptance log: `/tmp/mtp_accept_bfp4_weight_only_new_ref_20260417.log`
