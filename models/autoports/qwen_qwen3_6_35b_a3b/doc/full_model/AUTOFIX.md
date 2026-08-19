# AUTOFIX - full-model watcher sampling gate

Date: 2026-08-19

## Goal

Make the Qwen/Qwen3.6-35B-A3B full-model watcher gate pass while preserving:

- optimized 2x2 multichip decoder execution
- flat 4-way vocab-sharded LM head
- on-device traced greedy decode
- no host sampling, no single-chip fallback, no replicated full logits, no untraced Python token feedback

## Fix

Implemented an opt-in composite top-k all-gather path in common sampling:

- `models/common/sampling/tt_sampling.py`
  - Adds `args.use_composite_topk_all_gather`.
  - When enabled and the gather uses the public `ttnn.all_gather` fallback, converts the tiny tile-layout top-k tensor to row-major with `ttnn.untilize`.
  - Calls the existing `ttnn.all_gather`; row-major input forces TTNN's composite all-gather implementation (`all_broadcast` + `concat`) instead of the native minimal-default all-gather writer.
  - Converts the gathered tensor back to tile layout with `ttnn.tilize` before the existing sampling path continues.

Enabled it only for this model:

- `models/autoports/qwen_qwen3_6_35b_a3b/tt/model.py`
  - Sets `args.use_composite_topk_all_gather = True` in `_make_sampling_args`.
  - Updates `iter_runtime_fallback_audit()` to declare `common_sampling_generator_flat_4way_topk1_composite_gather`.

Updated the stage smoke assertion:

- `models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py`

## Verification

Syntax:

```bash
./python_env/bin/python -m py_compile models/common/sampling/tt_sampling.py models/autoports/qwen_qwen3_6_35b_a3b/tt/model.py models/autoports/qwen_qwen3_6_35b_a3b/tt/generator.py models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py
```

Log:

```text
models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/py_compile_composite_gather.log
```

Focused watcher probe:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 timeout 180 ./python_env/bin/python - <<'PY' 2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/probe_watcher_rm_composite_all_gather.log
...
PY
```

Result:

```text
RM_COMPOSITE_AG_OK ... maxdiff 0.0
```

Original watcher gate:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 RUN_QWEN36_FULL_MODEL_SMOKE=1 ./python_env/bin/python -m pytest -q --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py
```

Log:

```text
models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/watcher_synthetic_composite_gather.log
```

Original result at the time of the watcher fix:

```text
4 passed, 2 warnings in 37.88s
```

Final post-remediation rerun after adding traced teacher-forcing and changed
page-table coverage:

```text
6 passed, 2 warnings in 50.05s
```

Fallback-guarded synthetic run:

```bash
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' RUN_QWEN36_FULL_MODEL_SMOKE=1 ./python_env/bin/python -m pytest -q --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py
```

Original log:

```text
models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/synthetic_no_watcher_throw_on_fallback_composite_gather.log
```

Original result:

```text
4 passed, 2 warnings in 27.75s
```

Final post-remediation fallback-guarded rerun:

```text
6 passed, 2 warnings in 46.56s
```

## Status

Watcher gate passes. The measured decode path remains device-side and traced; only the internal top-k gather implementation changed.
