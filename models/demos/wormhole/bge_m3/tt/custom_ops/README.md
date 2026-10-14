# BGE-M3 model-local custom ops

`ttnn.generic_op` kernels and their Python wrappers. The data-parallel serving
path imports these from `attention.py`, so they are production code.

Building the program from Python means a change to a descriptor or a kernel
needs no `_ttnn.so` rebuild.

## Layout

    custom_ops/
        __init__.py              package surface
        README.md                this file
        encoder_sdpa/            encoder SDPA (see its own README)
            op.py, config.py, kernels/
        fused_qkv_heads/         fused QKV to Q/K/V head split
            op.py, kernels/
        fused_concat_heads/      attention head concatenation
            op.py, kernels/
        qkv_scatter_matmul.py    fused projection, head split, BF4 conversion

## What the serving path calls

| Op | Role |
| --- | --- |
| `qkv_scatter_matmul` | one program for the QKV projection, the head split, and the BF4 conversion |
| `encoder_sdpa` | q256/k2048 attention over BF4 K and V, with compact valid-length masking |

`encoder_sdpa` writes the concat-head order directly, so the serving path does
not run `fused_concat_heads`. `quality_mode` turns the scatter off and uses
`fused_qkv_heads` and `fused_concat_heads` instead.

## Conventions

- A kernel path in `op.py` is relative to `TT_METAL_HOME`, for example
  `"models/demos/wormhole/bge_m3/tt/custom_ops/fused_qkv_heads/kernels/writer_qkv_scatter.cpp"`.
- Each sub-package exposes a callable that takes and returns ttnn tensors.
- An op validates its shape and raises when the shape does not match. It does
  not fall back to a stock op, because a silent fallback hides a broken path.

## Tests

| Command | Purpose |
| --- | --- |
| `pytest models/demos/wormhole/bge_m3/tests/pcc/model_dp.py -s` | accuracy of the full serving path |
| `pytest models/demos/wormhole/bge_m3/tests/perf/perf.py -s` | wall-clock latency |
| `pytest models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py -k n300_dp_tracy` | per-op device time |
