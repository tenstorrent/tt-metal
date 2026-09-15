# Adaptive-BFP4 full-chip comparison driver

Implemented in `adaptive_fullchip.py`; no attention kernel or shared driver changes. Device/JIT qualification is pending with the parent.

Controls:

- `--b4-search native|minus|pm`: qualified native-group RNE, E/E−1 search, or E/E−1/E+1 search.
- `--kv-formats b8_b4|b4_b4`: K8 is always RNE5/native B8; every B4 input uses the selected quantizer. Q remains RNE7/BF16.
- `--mode denom_bf16` (default), `main_bf16`, `fast_bf16`, or `fp32`. Denominator-only/full-FAST retain the private correction-reset fix.

The driver is an independent copy of the qualified asymmetric host path, with the original source hash recorded. Q256/K512/D128, all-Q square noncausal attention, per-head chain assignments, input buffering, formats, and existing kernel paths remain unchanged. Native B4 uses the original optimized batch4/BF16-DST preprocessor; adaptive search uses its private batch1/FP32-DST implementation. Their preprocessing-time difference is intentionally included rather than hidden.

Every invocation creates original BF16 inputs and an original-input FP64 sampled-Q reference. Exact checks of actual device-preprocessed Q/K/V are on by default; CPU oracles run in524,288-value chunks after downloading outputs. No CPU search result or mean is uploaded. Checks compare decoded FP32 bits with signed zeros canonicalized, save failures, and record source pins before/after the run.

JSON records include ordinary L2/PCC and centered-output error against the same original-V FP64 column mean, with no gain fitting; preprocessing, attention, and combined trace times; useful attention TFLOP/s; all-output finite checks; and trace replay equality. Constant-V centered-relative error is explicitly undefined. Disabling exact checks requires `--no-check-preprocess` and is recorded.

## Suggested qualification

Run these only on the parent's configured device environment:

```sh
for kv in b8_b4 b4_b4; do
  for search in native minus pm; do
    python experiments/sdpa-l2/bfp4-lofi-v2/adaptive_fullchip.py \
      --label adaptive-smoke-${kv}-${search}-v1 --kv-formats "$kv" --b4-search "$search" \
      --length 1024 --heads 2 --cores 4 --sample-rows 1024 --iters 0
  done
done
```

Then use the same six controls with `--length 32768 --heads 10 --cores 110 --sample-rows 128 --iters 10`. Optional distribution controls include normal, sparse outliers, Q/K×2, common modes, and K-only/V-only channel outliers. The last two multiply every sixteenth feature by32 as part of original input construction.

## Host checks completed

CPU-only mocks covered all24 combinations of four modes, two K/V formats, and three searches on H10/N8192/110 cores. Attention CB audits, defines, and compile-time arguments were identical across searches within each mode/format. All chain assignments covered320 Q jobs with unchanged slot counts. Five exact-check routes crossed a chunk boundary at1,048,576 values each. CLI entrypoint/help and constant-V centered-error handling passed. Python syntax passed. These are host planning/oracle checks, **not JIT, device correctness, or performance results**.

Default denominator-only attention CB payload is890,880B/core for K8/V4 and825,344B/core for K4/V4, independent of search. Adaptive preprocessing separately uses5,248B/core; it does not add attention CBs or remove existing K/V slots.
