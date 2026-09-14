# Tests without DEVICE_FP32 or DEVICE in GateComputeMode parametrization

## Sigmoid-family tests (no DEVICE_FP32, no DEVICE)

| Test | File | Gate modes | Notes |
|------|------|-----------|-------|
| `test_kimi_prefill_block_chunked` | `tests/test_prefill_block_chunked.py:865` | `HOST_ALL` only | Kimi has a single expert group; comment says "validated only with the host gate" |
| `test_kimi_prefill_block_chunked_padded` | `tests/test_prefill_block_chunked.py:912` | `HOST_ALL` only | Same as above |

These two are the only sigmoid-family parametrized tests that have neither `DEVICE_FP32` nor `DEVICE`.

## GPT-family tests (no DEVICE_FP32, no DEVICE -- by design)

These use GPT-specific routing (`GPT_DEVICE` / `GPT_HOST`) which is a different routing family.
Left as-is per earlier decision.

| Test | File | Gate modes |
|------|------|-----------|
| `test_mistral4_prefill_block` | `tests/test_prefill_block.py:804` | `GPT_DEVICE` only |
| `test_mistral4_prefill_transformer` | `tests/test_prefill_transformer.py:1308` | `GPT_DEVICE` only |
| `test_mistral4_moe` | `tests/pcc/test_ttnn_moe.py:1170` | `GPT_DEVICE` only |
| `gpt_oss_120b` entries in `REGULAR_GATE_CASES` | `tests/pcc/test_moe_gate_prefill2d.py:268-269` | `GPT_HOST`, `GPT_DEVICE` |
| `mistral_small_4` entries in `REGULAR_GATE_CASES` | `tests/pcc/test_moe_gate_prefill2d.py:276-277` | `GPT_HOST`, `GPT_DEVICE` |
