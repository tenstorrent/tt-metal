# DeepSeek V3: Disaggregated prefill

This directory will contain the implementation of prefill stage for DeepSeek V3 model. More details TBD.

## Multiple models

The prefill runner is a model-agnostic engine in the common package
(`models/demos/common/prefill/`). It drives any model through a `PrefillModelAdapter`; the
DeepSeek-V3 family's concrete adapters live here in `tt/runners/adapters/`, selected by the
**`PREFILL_MODEL`** env var (default `kimi_k2_7`). To integrate a new
model, see
[models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md](../common/prefill/docs/ADDING_A_PREFILL_MODEL.md).

## Environment Variables

- **`PREFILL_MODEL`** — Which model adapter the runner / producers use (`deepseek_v3_d_p` | `kimi_k2_6` | `kimi_k2_7` | …). Defaults to `kimi_k2_7`. Replaces the former `PREFILL_MODEL_VARIANT`.

- **`DEEPSEEK_V3_HF_MODEL`** — Path to DeepSeek-R1-0528 weights directory. Falls back to `models/demos/deepseek_v3/reference/` then `/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528`.
- **`TT_DS_PREFILL_TTNN_CACHE`** — Directory for cached TTNN weight tensors (`.tensorbin` files). First run writes cache, subsequent runs load directly. Defaults to `{model_path}/tensor_cache_{arch}_{num_devices}dev/`.
- **`TT_DS_PREFILL_HOST_REF_CACHE`** — Directory for cached host reference snapshots used in PCC validation. Defaults to `/tmp/deepseek_v3_transformer_ref_cache`.
- **`TT_DS_PREFILL_INFINITEBENCH_CACHE`** — Directory for cached InfiniteBench prompt data. Defaults to `/tmp/deepseek_v3_transformer_inputs`.
- **`TT_DS_PREFILL_DEBUG_TOKEN_COUNT`** — Enable debug output for per-expert token counts in MoE forward pass. Set to `1`, `true`, or `yes` to enable. Defaults to disabled. Warning: enabling this adds device-to-host transfer overhead on every MoE layer forward.
- **`PREFILL_COMPRESSED_FP8_DISPATCH`** — Kill switch for compressed FP8 MoE dispatch (e4m3 compression of activations around dispatch, per-token scales in the metadata tail). The feature is **default ON** for validated models (DeepSeek-V3, Kimi) on Blackhole and always OFF elsewhere; set `0` to disable it locally (debugging / bf16 comparison runs). The env var can only disable — it never enables fp8 for unvalidated models or non-Blackhole hardware. Note: under `tt-run`, shell-exported `PREFILL_*` vars are not propagated — set it via the manifest's `env` map.
  Scope: **runner / producer only** — they resolve the flag in their own process, where no test param can reach. **It does not affect any test.** Tests take the mode as a parametrize axis: every prefill test collects a `fp8_dispatch` and a `bf16_dispatch` instance, so pick one with `-k fp8_dispatch` / `-k bf16_dispatch` instead of exporting anything. Their gate is `can_compressed_fp8_dispatch()` (validated model + Blackhole, capability only), so an `fp8_dispatch` case skips only when it genuinely cannot run.

## Weight Loading and TTNN Cache

See [tt/WEIGHTS_AND_CACHE.md](tt/WEIGHTS_AND_CACHE.md) for the weight loading contract that every TT module implements.
