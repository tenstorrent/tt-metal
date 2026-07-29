# AutoFix: sparse-matmul routed MLP

## Starting evidence

`AUTODEBUG.md` proposed replacing the dense all-expert packed gate/up and down
matmuls with `ttnn.sparse_matmul`.  The focused experiment used BF16 synthetic
nonzero layer-1 weights, exact sigmoid top-8 routing, the existing post-down
route multiply/reduce, and trace replay.

## Hypothesis experiment

Command, batch 1:

```bash
pytest -q 'models/autoports/coherelabs_north_mini_code_1_0/tests/test_fused_decoder.py::test_fused_sparse_dynamic_trace_replay_matches_reference[blackhole-1-1-sparse_matmul-mesh_device0]' -s
```

Result: passed, trace-replay PCC `0.9997952403828803`.

The first batch-32 probe supplied a per-token `[1,1,32,128]` mask and was
rejected by the device op before dispatch:

```text
sparsity.logical_volume() (4096) must be equal to the product of all batch dimensions (128)
```

This verifies that sparsity is per M tile/expert, not per logical token.  The
corrected probe reduced the 32 route rows into one expert-union mask
`[1,1,1,128]`, omitted `nnz` for runtime inference, retained the individual
sigmoid scores for the final multiply/reduce, and used the Gemma-style 6D
output reshape/transpose.

Command, batch 32:

```bash
pytest -q 'models/autoports/coherelabs_north_mini_code_1_0/tests/test_fused_decoder.py::test_fused_sparse_dynamic_trace_replay_matches_reference[blackhole-1-32-sparse_matmul-mesh_device0]' -s
```

Result: failed reproducibly with trace-replay PCC `0.9383658910883104`, below
the functional acceptance bar `0.995`.

## Verdict

Refuted for the serving decode contract.  Batch 1 alone is insufficient to
validate the 6D/tile routing behavior, and batch 32 fails correctness under
dynamic trace replay.  The candidate implementation and temporary tests were
removed.  No latency benchmark was used to override the failed correctness
gate; the packed dense-expert path remains selected.
