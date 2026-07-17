# DeepSeek V3: Disaggregated prefill

This directory will contain the implementation of prefill stage for DeepSeek V3 model. More details TBD.

## Multiple models

The prefill runner is a model-agnostic engine in the common package
(`models/demos/common/prefill/`). It drives any model through a `PrefillModelAdapter`; the
DeepSeek-V3 family's concrete adapters live here in `tt/runners/adapters/`, selected by the
**`PREFILL_MODEL`** env var (default `deepseek_v3_d_p`; Kimi is `kimi_k2_6`). To integrate a new
model, see
[models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md](../common/prefill/docs/ADDING_A_PREFILL_MODEL.md).

## Environment Variables

- **`PREFILL_MODEL`** — Which model adapter the runner / producers use (`deepseek_v3_d_p` | `kimi_k2_6`). Defaults to `deepseek_v3_d_p`. Replaces the former `PREFILL_MODEL_VARIANT`.

- **`DEEPSEEK_V3_HF_MODEL`** — Path to DeepSeek-R1-0528 weights directory. Falls back to `models/demos/deepseek_v3/reference/` then `/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528`.
- **`TT_DS_PREFILL_TTNN_CACHE`** — Directory for cached TTNN weight tensors (`.tensorbin` files). First run writes cache, subsequent runs load directly. Defaults to `{model_path}/tensor_cache_{arch}_{num_devices}dev/`.
- **`TT_DS_PREFILL_HOST_REF_CACHE`** — Directory for cached host reference snapshots used in PCC validation. Defaults to `/tmp/deepseek_v3_transformer_ref_cache`.
- **`TT_DS_PREFILL_INFINITEBENCH_CACHE`** — Directory for cached InfiniteBench prompt data. Defaults to `/tmp/deepseek_v3_transformer_inputs`.
- **`TT_DS_PREFILL_DEBUG_TOKEN_COUNT`** — Enable debug output for per-expert token counts in MoE forward pass. Set to `1`, `true`, or `yes` to enable. Defaults to disabled. Warning: enabling this adds device-to-host transfer overhead on every MoE layer forward.

## Weight Loading and TTNN Cache

See [tt/WEIGHTS_AND_CACHE.md](tt/WEIGHTS_AND_CACHE.md) for the weight loading contract that every TT module implements.
