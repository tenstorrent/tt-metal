# AutoFix Report

## Starting Evidence

- Source: `AUTODEBUG.md`, headline findings 1 and 2.
- Original failures: BFP8 LoFi and HiFi2 LM-head linear both required
  `1778432 B > 1572864 B`; BF16 HiFi4 required
  `2745088 B > 1572864 B` with the BFP4-tuned `32768/3` geometry.
- Focus: real Falcon3 weights, inherited 32-core terminal residual geometry,
  strict TTNN fallback, one decoder layer, traced two-token generation.

## Hypothesis Experiments

- Hypothesis: BFP8 fits if the legal K block width is reduced from 3 to 1.
  Experiment: run the normal real-model construction and traced token path with
  `FALCON3_PRECISION_CONFIG=.../bfp8_lofi_bfp8_act_ccl_kv.json`, 32768 columns,
  and `in0_block_w=1`.
  Result: pass; generated tokens `[95671, 80947]`; all devices closed.
  Verdict: verified.
  Evidence: `results/autofix_lm_head/bfp8_32768_w1.log`.
  Fix: derive LM-head `in0_block_w` from the consumed weight dtype: BFP4 keeps
  the optimized width 3; BFP8 and BF16 request the smallest legal width 1.
  Verification: the normal selected-config construction path, without geometry
  environment overrides, passed and reported
  `lm_head_geometry={columns_per_device:32768,in0_block_w:1}` in
  `results/autofix_lm_head/bfp8_normal_path_verify.log`.

- Hypothesis: BF16 fits at width 1, or after reducing the column split.
  Experiment: serialized real-model strict-fallback runs at `(32768,1)`, then
  `(16384,1)`, then `(8192,1)`.
  Result: all three failed with the identical static CB allocation
  `2003712 B > 1572864 B`; every run closed all devices.
  Verdict: refuted. The limiting BF16 buffers are independent of these N splits.
  Evidence: `results/autofix_lm_head/bf16_{32768,16384,8192}_w1.log`.
  Fix: none for BF16; exact runtime blocker retained as rejected-candidate evidence.

## Commands

Each run used the same inline real-model harness and changed only the shown
environment fields:

```text
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
FALCON3_PRECISION_CONFIG=<bfp8-or-bf16-config.json> \
FALCON3_LM_HEAD_COLUMNS_PER_DEVICE=<32768|16384|8192> \
FALCON3_LM_HEAD_IN0_BLOCK_W=1 python - <<'PY' ...
```

The harness opened a `1x4` mesh, called `build_generator(...,
override_num_layers=1, max_context_len=32768)`, generated two traced device-
sampled tokens from the first 128 AIME24 reference tokens, called
`generator.teardown()`, and closed the mesh in `finally`.

The post-fix verification omitted both geometry overrides:

```text
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
FALCON3_PRECISION_CONFIG=.../bfp8_lofi_bfp8_act_ccl_kv.json python - <<'PY' ...
```

## Final Status

- Fixed for BFP8 with real-model traced evidence through the normal precision-
  config construction path.
- BF16 remains a measured hardware/runtime L1 limitation for all requested
  geometries; it is not silently classified as numerical instability.
- Remaining risk: this focused experiment proves construction, terminal logits,
  traced token execution, and policy propagation for one real decoder layer.
  Full 28-layer accuracy and teacher-forcing performance remain the parent
  datatype-sweep candidate evaluation.
