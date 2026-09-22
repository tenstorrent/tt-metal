# NLLB translation on Blackhole

Experimental TTNN inference for Facebook NLLB-200. The optimized path targets
distilled 600M, greedy decoding, batches of 1–4, source lengths of 1–256 tokens,
and generation caps of 1–256 tokens. The cap includes the forced target-language
token and excludes the decoder-start token. BF16 is the default. For 600M, BFP8_B stores the output-projection weights in block floating point
while retaining BF16 transformer weights, activations, normalization, biases,
and embeddings. This global policy applies to every request.
The configuration-driven backend also loads distilled 1.3B and 3.3B, but their
sampled quality results do not establish full acceptance. The low-level `forward`
API retains its separate 64-token decoder-input limit.

The implementation uses TTNN attention, request-local decoder trace replay,
cross-attention KV reuse, and last-token vocabulary projection. Eligible 600M
multirow requests share a packed vocabulary projection. Self-attention still
processes the full decoder prefix. Serialize requests per backend; concurrent
serving and beam search are not supported.

## Environment and assets

Place this directory at `models/experimental/nllb` in a built TT-Metal checkout.
Use its configured TTNN Python environment with PyTorch, NumPy, Transformers,
and official local tokenizer assets. Tests also need pytest and the dependencies
of the checkout's upstream test configuration.

The reproduction environment is TT-Metal
`86b55b92d0dc2094bee4f78dca0e31f37ea9673a`, Python 3.10, PyTorch 2.14+cpu,
Transformers 5.12.1, and NumPy 1.26.4. Earlier model evaluations used
`fd80faa3b35fa6d38a92d08326205fc4168284ec`. These environments differ from
upstream's default Python dependency versions; runtime-specific results are
reported separately with the contribution.

`official-assets.json` records checkpoint revisions, sizes, and hashes for all
three sizes. Provide official weights, `config.json`, and the four tokenizer
files listed there. Weights are not bundled. Their license is CC-BY-NC-4.0;
this experimental port does not establish commercial suitability.

Set these before any TTNN import, including pytest conftests:

```sh
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NLLB_ASSETS=/path/to/official/nllb-600m
export NLLB_DEVICE=0  # visible index of your allocated Blackhole device
```

Run commands from the TT-Metal checkout root. A visible device index may map to
a different physical card on another host.

## Translate

```sh
python -m models.experimental.nllb.translate \
  --checkpoint "$NLLB_ASSETS" --config "$NLLB_ASSETS/config.json" \
  --tokenizer-directory "$NLLB_ASSETS" \
  --source-language eng_Latn --target-language fra_Latn \
  --device "$NLLB_DEVICE" --precision bf16 --max-new-tokens 64 \
  --text 'Hello world.' --output translation.json
```

Repeat `--text` for batches of up to four. Output includes translations, complete
token rows, precision policy, and synchronized generation time. Loading and
tokenization are excluded from that timing. The CLI owns and closes its device.

For repeated requests, create a backend once inside `runtime_setup.RuntimeOwner`,
bind it with `owner.bind(create_backend(...))`, then call `model.generate` with
validated token/mask arrays and the target-language ID. Call
`runtime_setup.configure_tracking()` before importing TTNN or the backend.
`RuntimeOwner.open` reserves the required 64 MiB trace region and enables program
cache. The text API `translate.translate` borrows a caller-owned device and
creates a model per call; that device must already have these settings.

Normal completion releases request traces and caches. If cleanup cannot be
confirmed, the runtime retains the owner/model/device and reports failure.
Inspect `runtime_setup.retained_owners()`; do not close or reset a retained device
and claim recovery.

## Targeted validation

```sh
export NLLB_TEST_DEVICE="$NLLB_DEVICE"
export NLLB_TEST_CHECKPOINT="$NLLB_ASSETS"
export NLLB_TEST_CONFIG="$NLLB_ASSETS/config.json"
export NLLB_TEST_TOKENIZER="$NLLB_ASSETS"
export NLLB_TEST_PRECISION=bf16
export NLLB_TEST_TIMEOUT=300
python -m pytest models/experimental/nllb --tt-arch blackhole --collect-only -q
python -m pytest models/experimental/nllb/test_generation_envelope.py \
  --tt-arch blackhole -ra
python -m pytest models/experimental/nllb/test_packed_integration.py::test_portable_packed_native \
  --tt-arch blackhole --capture=tee-sys -ra --junitxml=nllb-packed.xml
```

Keep upstream conftests enabled. Device tests are opt-in; missing assets can
produce skips, which do not demonstrate correctness. The envelope test checks
same-device consistency. To compare against an independent reference, generate a
fixture on an allocated NVIDIA GPU using `generate_reference.py --help`, then set
`NLLB_TEST_FP32_ENVELOPE` and `NLLB_TEST_FP32_ENVELOPE_SHA256`. The helper binds
reference outputs to the checkpoint, inputs, and generation policy.

CPU tests cover validation, checkpoint loading, tokenizer assets, trace lifecycle,
and simulated failures. Native probes cover actual kernels, public batching, and
cleanup. Simulated failures do not prove native failure recovery.

The packaged `test_portable_packed_native` probe always uses BF16. Setting
`NLLB_TEST_PRECISION` does not change that probe; mixed-precision validation
requires the precision-aware envelope/reference checks.

## Compare performance

```sh
python -m models.experimental.nllb.benchmark_paired \
  --checkpoint "$NLLB_ASSETS" \
  --config models/experimental/nllb/benchmark_cases/600m/config.json \
  --input-case models/experimental/nllb/benchmark_cases/600m/natural_long_b2.json \
  --output new-benchmark-output --device "$NLLB_DEVICE" \
  --precision bf16 --pairs 6
```

This compares full versus last-token projection on identical requests; multirow
LAST requests also use packing. `--validation` limits generation to four tokens
and checks the protocol, not representative translation performance. Retain raw
timings, token counts, balanced ordering, and lifecycle failures. Six cases per
model cover short/long sources, B1/B2, and source-length-256 B1/B4. The extra
600M `natural_long_b2` case contains two fixed, ordinary English paragraphs
with a 256-token cap; its inputs were fixed before reference generation.

### Latest targeted measurements

On the pinned runtime above, the fixed `natural_long_b2` case has two 134-token
sources and produces 48/175 learned predictions including EOS. Six balanced
FULL/LAST pairs per precision measured the following synchronized request times:

| Precision | FULL median | Packed LAST median | Within-run speedup | Loaded device allocation |
|---|---:|---:|---:|---:|
| BF16 | 6.45293 s | 3.16389 s | 2.0396× | 1,779,531,776 bytes |
| Mixed BFP8_B | 5.63496 s | 2.96033 s | 1.9035× | 1,533,556,736 bytes |

All saved outputs matched the independent A100 FP32 reference exactly. Precision
runs were sequential; their comparison is not another interleaved experiment.
Device allocation fell 13.82%, while loaded host RSS increased from 1.774 to
2.052 GB. These are allocation snapshots, not peak serving memory. First FULL
requests took about 56–58 seconds including compilation; load and initialization
are excluded from the request medians. FULL already includes other TT
optimizations, so these ratios are not comparisons against PyTorch or NVIDIA.

The source package passed 363 CPU tests and 34 subtests, with no execution skips
or xfails (471 collected, not all run). The corrected mixed-precision policy
passed 22 long numerical checks at NRMSE ≤0.04, 13 short FP32-exact rows, and
natural B1/B2/B4 reference parity with EOS/PAD and cleanup checks. One synthetic
argmax differed; numerical tolerance and natural generation equality are separate
checks. These are targeted results, not renewed full-corpus qualification.
The contribution adds source headers and updates documentation and its integrity
manifest; executable Python ASTs and embedded kernel strings remain unchanged
from that tested package. Hardware results apply to the pinned runtime above,
not automatically to a newer checkout or upstream's default dependency stack.

Historical five-workload measurements found 1.568× speedup over an earlier
optimized TT implementation and 1.062× over projected trace decoding. The earlier BFP8_B policy
reduced loaded-device storage by 32.2% versus BF16; that value must not be
attributed to the newer policy retaining all 600M transformer weights in BF16. These are source-specific
results, not fresh measurements of every packaged revision, NVIDIA speedups,
isolated peak-memory measurements, or production latency guarantees.

The optimized donor, with its earlier BFP8_B policy, was evaluated on 8,096
translations in eight directions per precision. Subsequent packaging used targeted regression checks. Larger-model
quality flags, concurrent serving, cold-start comparisons, and isolated serving
peak memory remain open. Generation beyond 256 tokens is not admitted. Longer
output and newer-runtime checks are targeted tests, not a repeated full-corpus
evaluation. `PACKAGE_FILES.json` binds exported files;
it is an integrity inventory, not a correctness certificate.
