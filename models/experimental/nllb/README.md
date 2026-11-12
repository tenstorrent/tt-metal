# NLLB-200 on Blackhole

Experimental TTNN translation for `facebook/nllb-200-distilled-600M`.
Greedy decoding supports batches 1–4, source lengths through 256 tokens, and
up to 256 generated tokens (including the forced target-language token).
BF16 is the default. Mixed `bfp8_b` stores the vocabulary projection in BFP8_B;
transformer weights and activations remain BF16 for numerical accuracy.

The backend uses TTNN attention, cross-attention KV reuse, decoder trace replay,
and packed last-token vocabulary projection. Self-attention recomputes the
prefix. Requests must be serialized per backend; beam search and concurrent
serving are not supported. The configuration-driven loader also handles distilled
1.3B and 3.3B, but those sizes are **not qualified by the 600M results**.

## Layout

| Directory | Contents |
|---|---|
| `tt/` | Backend, trace kernels, runtime ownership, input/checkpoint validation |
| `demo/` | Translation CLI and text API |
| `reference/` | Independent CUDA FP32 fixture generator, comparison helpers, asset pins |
| `tests/` | CPU regressions and opt-in Blackhole integration tests |
| `benchmarks/` | Paired benchmark, shared input cases, model-specific configurations |
| `docs/validation.md` | Detailed setup, measurement protocol, evidence and limitations |

## Run a translation

Use a built TT-Metal environment with TTNN, PyTorch, Transformers and NumPy.
Provide local official checkpoint and tokenizer files; pinned revisions and
hashes are in [reference/official-assets.json](reference/official-assets.json).
Run commands from the TT-Metal checkout root:

```sh
export NLLB_ASSETS=/path/to/nllb-200-distilled-600M
export NLLB_DEVICE=0  # visible index of an allocated Blackhole card
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m models.experimental.nllb.demo.translate \
  --checkpoint "$NLLB_ASSETS" --config "$NLLB_ASSETS/config.json" \
  --tokenizer-directory "$NLLB_ASSETS" --device "$NLLB_DEVICE" \
  --source-language eng_Latn --target-language fra_Latn \
  --precision bf16 --max-new-tokens 64 --text 'Hello world.' \
  --output translation.json
```

Repeat `--text` for a batch. The CLI owns and closes its device. For a reusable
backend and caller-owned devices, see [the API notes](docs/validation.md#translate).

## Test and benchmark

Keep upstream pytest conftests enabled. Native tests require explicit local
assets and an allocated card; a skip does not establish hardware correctness.

```sh
export NLLB_TEST_CHECKPOINT="$NLLB_ASSETS"
export NLLB_TEST_CONFIG="$NLLB_ASSETS/config.json"
export NLLB_TEST_TOKENIZER="$NLLB_ASSETS"
export NLLB_TEST_DEVICE="$NLLB_DEVICE"
python -m pytest models/experimental/nllb/tests --tt-arch blackhole --collect-only -q
python -m pytest models/experimental/nllb/tests/test_packed_integration.py::test_portable_packed_native \
  --tt-arch blackhole -ra
python -m models.experimental.nllb.benchmarks.benchmark_paired \
  --checkpoint "$NLLB_ASSETS" --config models/experimental/nllb/benchmarks/configs/600m.json \
  --input-case models/experimental/nllb/benchmarks/cases/natural_long_b2.json \
  --device "$NLLB_DEVICE" --precision bf16 --pairs 6 --output nllb-benchmark
```

Packaging regression through CHIA passed **483 CPU tests and 117 subtests**,
plus native CLI/B1/B2/B4, masking, EOS and cleanup checks on the pinned runtime.
Native checks use a four-token cap and same-TT parity; they do not repeat the
full multilingual evaluation.

The native packed probe uses BF16. Independent reference generation and
precision-aware envelope tests are described in [the validation guide](docs/validation.md).

Benchmark inputs are shared across model sizes; configuration hashes retain
explicit model binding. Small synthetic/boundary cases exercise shapes and
termination, not multilingual quality. The earlier 8,096-translation evaluation
is separate evidence from an earlier precision policy, not a result of these
small fixtures or proof of the revised policy's full-corpus quality.

## Measured performance and limits

On a fixed long B2 workload, six balanced FULL/LAST pairs measured **6.453 →
3.164 seconds (2.04×)** for BF16 and **5.635 → 2.960 seconds (1.90×)** for mixed
BFP8_B. Loaded device allocation was **1.780 / 1.534 GB**, respectively; host RSS
increased with mixed precision. FULL is an already optimized TT control, not a
PyTorch or NVIDIA baseline. See [measurement details](docs/validation.md#latest-targeted-measurements).

Measurements used TT-Metal `86b55b92`, Python 3.10, Torch 2.14+cpu,
Transformers 5.12.1 and NumPy 1.26.4. They predate this packaging refactor;
upstream's default dependency stack and newer revisions need CI validation.
No production latency, peak-memory, or larger-model acceptance is claimed.

Code: Apache-2.0. Checkpoints are supplied separately under CC-BY-NC-4.0;
this experimental port does not establish commercial suitability.
