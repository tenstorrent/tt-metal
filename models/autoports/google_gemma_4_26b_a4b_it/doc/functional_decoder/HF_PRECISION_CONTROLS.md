# HF precision controls

CPU-only AutoFix controls for the batch-32 decode failures, using checkpoint revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, real layers 0 and 5, seed 42,
prefill length 33, decode position 33, and eight CPU threads. The original
**0.995 PCC threshold remains unchanged**.

The preserved [script](../../tests/hf_precision_controls.py) loads HF layers
directly from local safetensors into meta-created modules. It reproduces
`tests/batched.py`'s initialization and input RNG order without importing TTNN.
Input hashes, source hashes, all per-slot PCCs/expert IDs, and intermediate PCCs
for relevant slots are recorded in the JSON artifacts.

## Reproduction

Run from the repository root with the pinned checkpoint already cached:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.hf_precision_controls --control all
```

This exact command was run successfully. Its input hashes, per-slot prefill and
decode PCCs, and expert IDs reproduced the original inline experiments exactly.
Use `--control bf16`, `--control cache`, or `--control rope-cache` to run one
control independently. The added RoPE control was also run directly:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.hf_precision_controls --control rope-cache
```

The script records expected numerical failures; successful process exit means
the diagnostic completed, not that the decoder passed acceptance.

## Results

| Control | Layer 0, sliding attention | Layer 5, full attention |
| --- | --- | --- |
| Full HF BF16 computation | Only slot 14 fails: PCC 0.96967584 | Slots 11/29 fail: PCC 0.98607144 / 0.99315290 |
| BF16 cache rounding only | All decode slots pass; minimum PCC 0.99999275 | All decode slots pass; minimum PCC 0.99999956 |
| BF16 RoPE tables plus cache rounding | All decode slots pass; minimum PCC 0.99998745 | All decode slots pass; minimum PCC 0.99999954 |

The [full BF16 control](batch32_hf_precision_control.json) compares FP32 HF with
BF16 weights, activations, cache and RoPE. HF's built-in FP32 RMSNorm reductions
and attention softmax remain in place. All five recorded failing TT sliding
slots (5/15/17/25/27), and full-attention slots 9/21, pass this HF control with
unchanged expert sets. The three HF failures each change one of eight experts.
All per-slot prefill comparisons pass.

The [cache-only control](batch32_hf_cache_precision_control.json) keeps all HF
computation FP32. A `DynamicCache` subclass rounds incoming K/V through BF16
before storing and returning FP32 values, during both prefill and decode. All
64 decode outputs pass and retain the oracle's expert sets. All per-slot
prefill comparisons pass, although some prefill token routes change.

The [RoPE-plus-cache control](batch32_hf_rope_cache_precision_control.json) adds
BF16-roundtripped FP32 cos/sin tables to the cache-only control. It independently
reruns the cache-only baseline and records both comparisons for every decode
slot. All 64 decode outputs pass and retain the FP32 oracle's expert sets.
Sliding slot 17 reaches PCC **0.99999228**, versus **0.99999679** with cache
rounding alone; its selected expert set remains unchanged. All prefill slots
pass. Input hashes and cache-only results exactly match the earlier control.

For these batch-32 inputs, cache representation alone does not reproduce the
recorded decode failures.
Adding BF16 RoPE table quantization also does not reproduce sliding slot 17's
failure in this CPU control.
BF16 HF independently reproduces two full-attention routing discontinuities,
but these CPU controls do not establish unavoidable precision limits or the
cause of the remaining TT failures. CPU top-k tie-breaking and TT kernels may
differ; this cache control's prefill also attends directly to rounded K/V.

Only diagnostic files were changed. No device access, implementation changes,
or acceptance changes were made.

## Long-context controls

The preserved [long-context script](../../tests/hf_long_decode_precision_controls.py)
uses real layer 0, the exact local `tests/config.json`, seed 42, prefill length
4096, and 128 successive decode positions 4096–4223. It reads the actual
device-harness tensors from `headline_inputs.pt`, verifies every tensor is
bitwise equal to the CPU RNG reconstruction, and asserts that HF forwards do
not consume RNG. All 129 input hashes also match the device-captured
`headline_inputs.sha256.json` manifest. This rules out input-stream drift from
the TT setup between prefill input generation and the first decode input.

Exact command executed:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.hf_long_decode_precision_controls --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/headline_inputs.pt --length 4096 --steps 128
```

The [evidence artifact](long_decode_hf_precision_control.json) contains every
position's PCC, selected expert IDs, routing margins, input hashes, and detailed
stage comparisons/candidate logits for the targeted or changed routes.
All model computation remains FP32 in both controls; only the stated storage
roundtrips differ from the FP32 HF oracle.

| Control | Position 4110 PCC | Position 4149 PCC | Failures across 128 positions |
| --- | --- | --- | --- |
| BF16 cache only, FP32 RoPE | 0.99999751 | **0.99340531** | 4149 only |
| BF16 RoPE tables and cache | 0.99998525 | 0.99999624 | None; minimum PCC 0.99909410 |

Cache rounding alone reproduces the position-4149 discontinuity: expert 109
is replaced by expert 49. The FP32 rank-8/9 logit margin is 0.00070157647;
attention PCC remains 0.99999859, while raw-expert PCC drops to 0.95303319.
This demonstrates a cache-sensitive routing discontinuity for these exact
inputs, without establishing a general precision limit.

The combined RoPE/cache control retains the original expert sets at both
4110 and 4149. It changes one expert at positions 4175 and 4222, with both
outputs still passing 0.995. Thus the combined rounding control does not
reproduce the device's position-4110 failure. Both prefill comparisons pass
(PCC 0.99978088 and 0.99978878). No acceptance criteria were changed.
