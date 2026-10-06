# Text-derived maximum-context fixtures

The maximum-context CPU capture evaluates the pinned checkpoint's embedding and
layers 0–4, all sliding-window layers, to record actual inputs for layers 0 and
5. It is a decoder-input fixture task, not full-model generation. The three
existing source documents are repeated as **text before tokenization** to cover
262144 tokens. Activations are computed for every position; none are repeated or
randomly generated. Exact expanded text, token IDs, source hashes, pinned weight
revision and generator hash are recorded under `actual_text_long/`.

Checkpoint: `google/gemma-4-26B-A4B-it`, revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Hidden width is 2816; preceding layers
have 16 Q heads, 8 KV heads, head width 256 and a 1024-token causal window. PLE and
shared KV layers are absent. The oracle retains FP32 checkpoint weights,
embedding scale, hidden states and cache values, and the real HF `layer_scalar`.
Only the recorded target-layer transport is BF16-roundtripped FP32.

## Streaming semantics

The installed Transformers `cache_utils.py:1469–1470` uses
`layer_types[:-num_kv_shared_layers]`. Here the shared count is zero, so ordinary
`DynamicCache(config)` creates unbounded `DynamicLayer` objects. Chunking queries
alone would therefore retain growing KV and eventually rebuild large masks.

The explicit streaming mode constructs HF `Cache` objects containing
`DynamicSlidingWindowLayer` instances. Their `update` keeps the last 1023 keys
while returning the complete current query chunk plus its valid history
(`cache_utils.py:241–263`). For an absolute query position `p`, the fixture mask
allows exactly `p-1023 <= key_position <= p`, using the cache's reported absolute
offset. RoPE receives absolute positions and the original HF rotary module on
each chunk. The original decoder layer performs the remaining attention, norms,
router, experts, shared MLP and layer scaling.

With 1024 queries per chunk, the largest mask is 1024×2047 (about 8 MiB), the
16-head FP32 attention score tensor is about 128 MiB, and persistent K/V are
about 16 MiB per loaded layer. A full hidden-state stream is 2.75 GiB. Only one
checkpoint layer is loaded at a time. Capture uses two hidden-state buffers plus
bounded layer temporaries; serialization temporarily adds raw/transport copies.
Observed RSS and peak reference RSS are recorded by `cpu_workflow.json`.

## Small actual-input validation

`streamed_fixture_pilot.json` records a four-thread 4096-token pilot:

- Layer 0 streamed vs monolithic minimum per-token PCC:
  `0.9999999999999045`; maximum absolute difference `2.2888e-5`; zero changed
  top-8 route sets.
- Layer-5 input after five streamed layers vs the earlier recorded fixture:
  minimum per-token PCC `0.9999999999875995`; maximum absolute difference
  `1.1253e-4`.
- Five streamed layers took 18.215 seconds. Linear extrapolation to 262144
  tokens is 1165.75 seconds, about 19.4 minutes; 20–30 minutes was budgeted for
  capture and serialization. This estimate is not a device performance claim.

CPU-only fixture-loading checks cover the 262144/262143-style prefix contract,
layer validation and rejection of non-BF16-roundtripped transport. Formatting
and syntax checks pass. Importing the standalone reference helper does not import
TTNN.

## Capture and references

The authorized capture runs with four threads:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.create_optimized_activation_fixture --output-dir models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_long --case 262144 0 --layers 0 5 --threads 4 --prefill-chunk-size 1024 --repeat-text-to-length
```

One canonical 262144-token input fixture per layer serves both requested lengths
by causal prefix slicing. No second pass through the preceding layers is needed
for 262143. Raw `.pt` tensors are ignored locally; compact manifests and logs
record provenance and completion.

After capture, four CPU-only sampled references are generated serially: layers
0 and 5, lengths 262144 and 262143. `create_optimized_long_reference.py` projects
all K/V positions with real HF weights, then evaluates the same 291 sampled query
rows as the existing long-context harness. Full attention includes every allowed
causal key; sliding attention uses the exact window. It retains the existing
FP32 HF SDPA reference policy. Complete K/V storage is 4 GiB for layer 0 and
2 GiB for layer 5; HF's GQA expansion can temporarily add 8/16 GiB respectively.
This remains bounded by available host memory and avoids an S×S attention matrix.

Reference command pattern (the workflow manifest records every exact invocation):

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.create_optimized_long_reference --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_long/actual_text_layer5_262144_0.pt --layer 5 --length 262144 --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_long/actual_text_layer5_262144_reference.pt --threads 4
```

`long_context.py --input-fixture ... --reference-file ...` requires the saved
reference's fixture SHA256 and requested slice to match. The reference carries
model revision, layer, length, sampled rows and the input provenance. Results
continue to disclose `scope=subset`; this is full-context execution with sampled
output comparisons, not an all-output HF parity claim. No device work is run by
the capture/reference helper.

Current status and PIDs: `actual_text_long/cpu_workflow.json`. Final capture
timings: `actual_text_long/actual_text_fixture_manifest.json`. Each sampled
reference gets a compact JSON sidecar with its hash, command and elapsed time.

## Completed capture

The capture exited successfully in 1157.82 seconds (19.30 minutes), including
layer evaluation and fixture serialization. The five preceding-layer forwards
took 230.99, 244.98, 234.81, 209.23 and 220.27 seconds. The sampled capture RSS
high-water mark was 9.50 GiB; the controller samples capture RSS every 15 seconds
and reference RSS every 10 seconds, so these measurements can miss short peaks.

Canonical fixture hashes:

| Layer | SHA256 |
| --- | --- |
| 0 | `74cfb212f7f18fbc2bb7ebc3aa5f8f10619766a437b0ea93f86c8ad09f15c596` |
| 5 | `52d5a811ff0735478d3fff537be518fc10c098c812c8a34ffede705b0385352a` |

Both contain `[1,262144,2816]` raw FP32 and BF16-roundtripped transport inputs,
zero decode tokens, and the same exact 262144-token corpus prefix. Twenty source
text repetitions were needed before tokenization. All preceding-layer outputs
and both saved boundary streams passed finite-value checks.

## Completed references and checks

All four reference jobs exited successfully. Their individual helper elapsed
times include input validation/hash calculation, weight loading, K/V projection,
sampled attention and output serialization. The workflow timings additionally
include Python startup and the controller's polling interval. The sampled
reference RSS high-water mark was 29.26 GiB.

| Layer | Context | Helper seconds | Reference file |
| --- | --- | --- | --- |
| 0 | 262144 | 41.45 | `actual_text_layer0_262144_reference.pt` |
| 0 | 262143 | 41.32 | `actual_text_layer0_262143_reference.pt` |
| 5 | 262144 | 59.29 | `actual_text_layer5_262144_reference.pt` |
| 5 | 262143 | 50.51 | `actual_text_layer5_262143_reference.pt` |

`actual_text_long/verification.json` records passing shape, finite-output,
token-ID, source-provenance, hash and sampled-row checks. Every reference has
shape `[1,291,2816]`. The 290 shared causal-prefix rows have minimum per-token
PCC `0.9999999999999272` for layer 0 and `0.9999999987744099` for layer 5 between
the two lengths. Maximum absolute differences are `1.5259e-5` and `0.0014534`,
respectively. These are CPU reference-consistency checks, not device accuracy
results. The helper, capture and validation paths do not import TTNN.

The completed CPU workflow is idle. No hardware validation was performed by
this task. The parent can supply the corresponding input fixture and reference
to the existing long-context device contract, which still labels its comparison
scope as a subset of output rows.
