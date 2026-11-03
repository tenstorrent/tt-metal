# Device cache initialization

Qwen3.8-27B allocates qualified TP4 cache buffers directly on the device and
zeros them with the existing TTNN fill operation. This avoids constructing and
transferring large host zero tensors. Each fill completes before the buffer is
returned to decoder or model setup.

The default eligibility is exact:

| Buffer | Shape | Dtype |
| --- | --- | --- |
| KV | `[8192, 1, 32, 256]` | BF8 or BF16 |
| Recurrent | `[1, 12, 128, 128]` or `[2, 12, 128, 128]` | FP32 |

These tensors use standard tiled, interleaved DRAM storage on a Blackhole
P300_X2 QB2 with a `(1, 4)` mesh. Other topologies, shapes, dtypes and
row-major convolution history retain
`ttnn.zeros`. Shape, dtype, layout and cache ownership are unchanged.

For an allocation baseline, set the boolean policy before allocating a fresh
cache:

```python
for layer in generator.model.layers:
    layer.policy["device_zero_cache"] = False
```

The default is `True`; non-boolean values raise an error. This affects future
allocations only. Cache reset and token generation are unchanged.

If fill or its completion fence fails, cleanup drains device work before
releasing the buffer. If draining or release also fails, the decoder retains
the buffer and refuses further cache allocation, including with the policy
disabled. Close the model mesh before reuse. Allocation is a setup operation
and must run outside trace capture.

## Historical measurements

The source checkpoint `0327c84eeab20075b0bdf95b8917363a12db1974` was measured on
Blackhole QB2 with the completion fence included. Ordinary B1 used a fresh
262144-token-capacity cache for each request, a 129-token prompt and 128 output
tokens. Across three warm requests per arm, median cache readiness fell from
5.790604 s to 0.017676 s and complete request time from 9.333312 s to 3.568067 s.
Decode time was unchanged. The 16-request timing set included B1 and B2;
B2's 16384-page KV allocation retained the host path and its request time was
unchanged. Runs used a fixed baseline-first order.

These measurements describe that checkpoint, not qualification of later PR
heads. The historical component, ordinary B1/B2 and 128K correctness checks
passed. Persistent HTTP measurements at 100K/128K showed no material gain:
vLLM allocated 8193 KV pages, outside this allowlist, before serving requests.
The result is an allocation improvement for the listed shapes, not a sustained
decoding, serving or long-context prefill speedup.

## Validation

Host-only policy, completion-order and failure-ownership checks:

```bash
python3 -m unittest models.demos.qwen38_27b_qb2.tests.unit.test_cache_initialization -v
```

On a reserved QB2 or a QB2 CI runner, check all four allowed configurations,
all replicas and poisoned trace replay of the in-place fill:

```bash
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
python3 -m pytest models/demos/qwen38_27b_qb2/tests/test_cache_initialization.py -v
```
