# AutoFix: sparse routed experts

## Starting evidence

- Source review verified that the initial functional decoder repeated every
  token across all 128 experts and used dense `ttnn.matmul`.
- The functional-decoder contract requires the non-Galaxy active-expert
  pattern: dense top-k routing weights as a sparsity tensor and
  `ttnn.sparse_matmul` gate/up/down projections.
- Canonical implementation:
  `models/demos/gemma4/tt/experts/{decode,prefill}.py`.

## Hypothesis experiments

### Canonical sparse topology

- Hypothesis: canonical Gemma4 sparse experts preserve the dense expert result.
- Change: use top-8 sparse gate/up/down for decode and canonical all-expert
  sparse prefill followed by router weighting.
- Result: real layer-0 prefill passed, but decode PCC was
  `0.9949699919511993`, below the unchanged `0.995` bar.
- Verdict: topology verified; default sparse accumulation alone was not
  sufficient at the acceptance edge.

### Down-projection accumulation

- Hypothesis: HiFi4 with FP32 destination accumulation on only the
  accumulation-heavy sparse down projection would recover the PCC.
- Result: decode PCC decreased to `0.99494019`.
- Verdict: refuted; the change was removed.

### Gate-projection accumulation

- Hypothesis: the sparse gate projection is the sensitive leaf because its
  accumulation feeds the nonlinear GELU.
- Change: HiFi4 with FP32 destination accumulation on sparse gate only.
  Sparse up/down and all other expert operations retain framework defaults.
- Result: real layer-0 prefill PCC `0.99938856`; decode PCC `0.99501013`.
  Real full-attention natural/shared-cache cases also passed.
- Verdict: verified and retained.

### Serving decode batch

- Hypothesis: canonical sparse decode accepts 32 independent 128-expert masks
  in one call.
- Result: refuted. The sparsity volume was 4096 while the expert-weight batch
  length was 128.
- Fix: serialize decode users through batch-1 sparse calls using TTNN slices,
  then concatenate outputs on device. There is no host materialization and the
  operation sequence remains trace-capturable.
- Verification: traced batch 1 and batch 32 passed for sliding and full layer
  kinds; mutable-buffer A/B/A batch-32 replay also passed for both.

### Multi-user prefill

- Change: public prefill accepts `[batch, 1, sequence, hidden]`, slices each
  user and page-table row on device, invokes the single-user paged path, and
  concatenates outputs on device.
- Verification: real-shape batch-2 prefill passed for sliding and full layer
  kinds. The test writes
  `prefill_batch2_layer{0,5}_{layer_type}.json`.

## Static verification

```text
python -m py_compile \
  models/autoports/google_gemma_4_26b_a4b_it/tt/functional_decoder.py \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py

pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k "sparse_moe_prefill_delegates or sparse_moe_canonical_hot_path_audit or hot_path_fallback_audit"
```

The focused static suite passed. It asserts three sparse decode projections,
exactly one explicit compute config (gate only), canonical sparse prefill,
device-only batch serialization, and absence of torch/from-Torch/to-Torch in
the measured path.

## Final status

Fixed. Decode uses real top-8 active experts, prefill uses the canonical sparse
prefill topology, real-weight PCC meets the unchanged bar, and batch-32 traced
execution is covered. The only non-default expert precision is the
hardware-verified sparse gate projection.
