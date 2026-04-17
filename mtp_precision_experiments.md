# MTP Precision Experiments

## Baseline

- Date: 2026-04-17
- Test: `models/demos/deepseek_v3/tests/test_mtp.py::test_mtp_accept_rate_and_perf[False-0.75-device_params0]`
- Command:
  - `/home/shared/scripts/ds-run pytest -svv models/demos/deepseek_v3/tests/test_mtp.py -k test_mtp_accept_rate_and_perf`
- Reference:
  - `/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/dev/test_io_cache/mtp_full_model_seq128.pt`
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
