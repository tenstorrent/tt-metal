# AUTOTRIAGE - full-model watcher sampling gate

Date: 2026-08-19

## Symptom

The full-model watcher gate aborted in the traced token-out smoke:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 RUN_QWEN36_FULL_MODEL_SMOKE=1 ./python_env/bin/python -m pytest -q --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py
```

Existing failure log:

```text
models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/watcher_synthetic.log
```

Key evidence:

```text
Device 0 worker core(x= 1,y= 0) virtual(x= 2,y= 2): BRISC tripped an assert on line 123.
Current kernel: ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp.
BRISC: ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp
NCRISC: ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_reader.cpp
File ".../models/common/sampling/tt_sampling.py", line 339 in _perform_all_gather
```

The failure occurs in `TTSampling.forward` while gathering local top-k values for `SamplingGenerator`. The decoder stack, LM head, and no-watcher traced generation had already passed.

## Scope Isolation

Evidence that this is isolated to the CCL gather used by sampling:

- `test_full_model_non_aligned_prompt_smoke` passed under watcher in the original log before the traced sampling smoke failed.
- The failing stack reaches `models/common/sampling/tt_sampling.py::_perform_all_gather`, then `ttnn.all_gather`.
- Existing minimal probes also trip the same watcher assert in `minimal_default_writer.cpp`:
  - `models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/probe_watcher_axis_explicit_all_gather.log`
  - `models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/probe_watcher_noinline_all_gather.log`
- No-watcher generation passes, so the full-model tensor shapes and sampler semantics are otherwise valid.

## Source Findings

`ttnn.all_gather(cluster_axis=None)` on a non-line mesh recursively runs native all-gather over mesh axes. For the Qwen full model this is used on small tile-aligned top-k tensors from a flat 4-way vocab-sharded LM head.

The public all-gather path selects `ttnn::prim::all_gather`, which uses the `all_gather_async` minimal default reader/writer kernels. That is the same writer kernel reported by watcher.

`ttnn.experimental.all_reduce_async` exposes a `math_op` argument in Python, but this checkout's C++ wrapper comments it out in the relevant overloads and performs sum reduction. It is not usable for a global max/argmax sampling rewrite in Python.

## Candidate Results

Rejected:

- One-step `ttnn.experimental.all_gather_async(..., cluster_axis=None, topology=Ring, use_broadcast=True)` fails at program construction on this 2x2 topology:
  `TT_FATAL: Could not find any forwarding direction from src (M0, D1) to dst (M0, D2)`.
- Two-axis `all_gather_async(..., use_broadcast=True)` is watcher-clean but produces incorrect shard ordering for this `ShardTensorToMesh(dim=3)` tensor. Probe result showed `maxdiff=4096.0`, so it is not semantically valid.
- All-reduce max is not available through Python in this checkout because the experimental wrapper ignores the requested reduce type.

Accepted:

- Convert only top-k gather inputs to row-major before `ttnn.all_gather`, forcing the existing composite all-gather path (`all_broadcast` + `concat`), then tilize the gathered result before the existing downstream sampling logic.
- This still gathers only top-k values/indices, not full logits.
- This remains on-device and trace-capturable. It does not introduce host sampling, replicated logits, single-chip execution, or Python token feedback.

Passing focused probe:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 timeout 180 ./python_env/bin/python - <<'PY' 2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/probe_watcher_rm_composite_all_gather.log
...
PY
```

Key output:

```text
RM_COMPOSITE_AG_OK (4, 1, 32, 128) (4, 32, 128) 0.0 [0.0, 31.0, 32.0, 63.0, 64.0, 95.0, 96.0, 127.0]
```

## Conclusion

Root cause is a watcher-visible BRISC assert in the native minimal-default all-gather writer for the small tile-aligned sampling top-k gather on this 2x2 Blackhole mesh. The full-model stage can avoid that external CCL kernel path without changing greedy semantics by forcing composite all-gather for the top-k sampling gathers only.
