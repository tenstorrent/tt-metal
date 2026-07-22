# AutoDebug: real-weight sparse decode regression

## Headline

The regression is localized to the newly selected sparse MoE decode path, not
to attention or KV-cache handling. The synthetic result is not evidence that
the sparse expert matmuls are correct: `_synthetic_state()` makes both
`gate_up` and `down` identically zero. Its 0.999770 PCC exercises routing,
biases, and residual addition, but every weight-bearing sparse multiplication
has a zero result.

The highest-probability fault is the split sparse gate/up weight path. The
fused decoder creates `gate_weight` and `up_weight` by a strided device
`ttnn.slice` of an already tiled `[E, H, 2I]` tensor, reshapes the results to
`[1, E, H, I]`, and hands those tensors to `sparse_matmul`. The established
GPT-OSS sparse implementation instead performs `[..., ::2]` / `[..., 1::2]`
on the host and uploads independent, contiguous, rank-4 TILE tensors. A
real-weight packed-sparse (`policy=sparse`) versus split-sparse
(`policy=sparse_split`) A/B is the decisive first experiment.

## Evidence from source

1. `tests/test_functional_decoder.py::_synthetic_state` explicitly initializes
   `gate_up` and `down` with `torch.zeros`. Therefore the synthetic candidate
   cannot detect corrupt, swapped, mis-tiled, or otherwise misread sparse
   expert weights.
2. `FusedDecoder._moe_forward` selects `_sparse_moe_forward(...,
   split_projection=True)` for `moe_policy == "auto"` at decode length one.
   Real prefill takes the dense split-linear branch instead. The observed
   passing prefill/cache checks and failing decode output therefore isolate the
   new decode-only branch.
3. The real-state provenance is internally consistent. `_real_state()`
   dequantizes canonical MXFP4 blocks with
   `convert_moe_packed_tensors(..., dtype=torch.bfloat16)` into dense
   `gate_up_proj` / `down_proj`; `_hf_layer()` and
   `FunctionalDecoder.from_state_dict()` consume those same tensors. The latter
   also enforces `[E,H,2I]` and `[E,I,H]`. A model-weight orientation mismatch
   between HF and TT is consequently unlikely.
4. The interleaved feature convention itself is correct: the canonical
   `models/demos/gpt_oss/tt/experts/weights.py` loader defines gate as
   `gate_up[..., ::2]` and up as `gate_up[..., 1::2]`, matching the fused
   decoder's intended indices.
5. The sparse input/output shapes match the documented one-token modes after
   singleton dimensions are removed. No static source mismatch was found in
   the routing mask or expert-axis permutation. The remaining material
   differences from the canonical GPT-OSS path are weight materialization and
   the exact BF16/program configuration.
6. Repository unit coverage for `sparse_matmul` uses BFLOAT8_B weights and much
   smaller K. The failing path uses BF16 weights, K=2880, HiFi4, FP32
   accumulation, and `in0_block_w=2`; this combination is not covered by those
   unit tests. This becomes the next suspect only if packed and split sparse
   both fail.

## Ranked verify/refute experiments

### 1. Real-weight packed sparse versus split sparse (decisive)

Use the exact layer-12 and layer-13 inputs/seeds from
`test_real_weight_layer_kind_prefill_and_decode`, changing only `moe_policy`:

- `sparse`: one sparse `[E,H,2I]` projection using the original uploaded TILE
  `gate_up_weight`, then logical even/odd output slices.
- `sparse_split`: two sparse `[E,H,I]` projections using the construction-time
  strided split tensors.
- `split`: dense split linears, as a known-shape control.

Interpretation:

- packed sparse passes and split sparse fails: confirmed split tensor
  materialization/layout fault. Keep packed sparse if it still beats the best
  traced baseline, or host-split before device upload and create independent
  rank-4 TILE gate/up tensors as the canonical loader does.
- dense split passes, but both sparse variants fail: sparse-matmul computation
  (most likely the down projection or BF16/config combination), not weight
  provenance.
- dense split also fails: probe the construction-time split tensors directly;
  the prior whole-layer prefill PCC was not a sufficiently isolated check
  because residual addition can mask a bad MoE contribution.

Record output PCC and also PCC/max error for the MoE result before residual
addition; end-to-end output alone is less diagnostic.

### 2. Probe the three sparse matmuls independently with real data

For one normalized real decode token, retain the same router/top-k mask and
compare selected experts at these boundaries against dense TTNN or Torch:

1. gate projection before bias;
2. up projection before bias;
3. activated `down_input`;
4. down projection before bias and routing reduction.

Read only this temporary diagnostic data back to the host; do not add a host
read to the delivered runtime. If gate/up diverge only for split sparse, the
host-split TILE fix is sufficient. If gate/up agree but down diverges, inspect
the `[tokens,E,1,I]` / `[1,1,tokens,E]` down-mode pair and its program config.

### 3. Validate split tensors, including physical layout

Compare host round-trips of `gate_weight` and `up_weight` against the same real
state tensor's `[..., ::2]` and `[..., 1::2]`, respectively. Print logical
shape, padded shape, layout, dtype, and memory config before each sparse call.
Also try host-splitting before `_to_device_tensor(..., layout=TILE)` without
changing any math. This distinguishes bad slice contents from a sparse reader
incompatibility with a strided-slice/reshape product.

### 4. If both sparse variants fail, reduce the sparse kernel variables

Run an isolated real gate/up or down projection while varying one item at a
time:

- `in0_block_w=1` versus `2`;
- default compute-kernel config versus the decoder's HiFi4/FP32-acc config;
- BFLOAT8_B versus BF16 weight storage, with an appropriate PCC bar;
- the current 5x6 grid versus a known sparse unit-test-style grid/config.

This should be done on the first failing projection, not through the whole
decoder, so a kernel/config defect is immediately visible.

### 5. Close the test blind spot

Add a fused-path correctness case whose expert matrices are nonzero. A
deterministic structured or low-scale nonzero state is adequate, provided both
gate/up and down products materially affect the output. Require sparse MoE
boundary PCC as well as final decoder PCC. Keep the two real-weight layer-kind
tests as the authoritative acceptance gate.

## Lower-probability hypotheses

- Router/top-k mismatch is unlikely: the same FP32 router, BF16 top-k/softmax,
  and scatter code is used by the dense and sparse branches, and the synthetic
  bias result exercises expert selection.
- KV-cache or attention is not causal for this failure: key/value checks pass,
  and only the post-attention sparse MoE selection changed.
- Gate/up parity is unlikely to be reversed in source because it matches the
  canonical GPT-OSS loader. It is still covered by the split-tensor round-trip.
- The static test's expected sparse call count is a separate mechanical test
  failure and does not explain the 0.60--0.67 real-weight PCC.

## Recommended repair order

Run experiment 1 first. If packed sparse restores both real layer kinds and
still beats the correct traced baseline, select it immediately and add the
nonzero synthetic guard. If split sparse alone fails and its latency is needed,
materialize gate/up on the host as independent contiguous TILE weights, then
repeat experiments 1--3. Investigate sparse kernel configuration only if both
real-weight sparse forms fail.
