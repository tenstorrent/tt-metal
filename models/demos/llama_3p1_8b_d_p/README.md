<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B disaggregated prefill

TTNN prefill for the Llama-3.1-8B-Instruct checkpoint on one Blackhole Galaxy,
with a 4×8 mesh: sequence parallelism (SP) across four rows and tensor parallelism
(TP) across eight columns. The model runs all 32 decoder layers, embedding,
final RMSNorm, and the vocabulary head. Calls process 1024-token chunks, with
explicit valid ranges and two independent cache slots.

## Model and cache

`tt/model.py::PrefillModel` owns weights and shared attention/RoPE resources.
The caller owns token tensors, the cache returned by `allocate_kv_cache`, and
each returned output. Calls through a model instance must be sequential.
`upload_token_chunk` packs natural-order host IDs into SP-row order.
`prefill_chunk(..., skip_lm_head=False)` returns vocabulary-sharded logits;
the default returns hidden states. Call `model.close()` after use.

Weights and activations are BF16. Indexed RoPE uses the stock BF16 destination
setting, with BF16 tables and outputs. KV defaults to BF8_B, with BF16 also covered
by the accuracy tests. Each TP column owns one KV head; SP rows own successive
256-token stripes. The local cache shape is `[64, 1, max_seq_len / 4, 128]`,
with user-major planes (`slot * 32 + layer`), 32-token round-robin DRAM pages,
and Meta-interleaved K after RoPE. `max_seq_len` defaults to 2048; the model
accepts a chunk-aligned capacity. Full-model book tests currently cover 2K and
4K, with 8K–64K explicitly skipped.

Attention gathers the selected cache prefix, restores natural token order,
and applies standard SDPA with an absolute-position causal mask, accurate
exponential mode, and FP32 destination accumulation. It requires the standard
SDPA accurate-exponential correction tracked in [PR #57180](https://github.com/tenstorrent/tt-metal/pull/57180).
Check that correction is present before interpreting failures of the strict attention accuracy gate.

## Run the tests

Use a built tt-metal environment and a local checkpoint containing config,
tokenizer, safetensors index, and all shards. All checkpoint tests honor:

```bash
export LLAMA31_8B_CHECKPOINT=/mnt/models/meta-llama/Llama-3.1-8B-Instruct
export OMP_NUM_THREADS=16
```

On an allocated Blackhole Galaxy with the SP/TP ring links available:

```bash
python3 -m pytest --noconftest models/demos/llama_3p1_8b_d_p/tests/host -v
python3 -m pytest models/demos/llama_3p1_8b_d_p/tests/unit -v --timeout=600
python3 -m pytest models/demos/llama_3p1_8b_d_p/tests/model/test_prefill_model_vs_ref.py -v --timeout=1200
python3 -m pytest models/demos/llama_3p1_8b_d_p/tests/model/test_prefill_native_input_accuracy.py -v --timeout=1200
python3 -m pytest models/demos/llama_3p1_8b_d_p/tests/model/test_prefill_book_top5.py -v --timeout=600
```

The CI registration in `tests/pipeline_reorg/blaze_models_prefill_tests.yaml`
runs CPU reference checks first in the components/decoder allocation, plus a
separate book 2K/4K job. Select `llama31` in the Blaze Models Prefill tests workflow
to run the component, book and runner jobs. Select `llama31_prefill_runner` for
the runner acceptance stage alone. CPU checks use `--noconftest` so collection does not load device
fixtures. Shared CPU helpers are in `tests/utils.py`; TTNN helpers are in
`tests/device_utils.py` and `tests/model/device_utils.py`.

### Full 2K accuracy: required local validation

The approximately 33-minute full-model/all-layer accuracy suite is temporarily
omitted from CI. Before pushing changes to model execution, numerical policy,
reference calculations, or these accuracy tests, run both
`test_prefill_model_vs_ref.py` and `test_prefill_native_input_accuracy.py` with the
commands above. Keep all 11 cases and their existing thresholds. Record the
results and tested commit in the PR; a green CI run alone does not establish full
2K KV accuracy. Documentation-only and CI-selection changes do not require a
repeat device run.

Set `LLAMA_PREFILL_EVIDENCE_DIR` to a fresh directory to retain accuracy reports;
the native-input test creates one directory per prompt/dtype. Set
`PREFILL_SUMMARIES` to also retain compact per-layer PCC summaries.

**Before project completion:** restore the full 2K accuracy CI job and obtain a
passing run on the final integrated revision.

## Accuracy contract

- Component tests compare norm, QKV, RoPE, cache writes, attention, MLP,
  embedding, and vocabulary projection against independent Torch/HF references.
  Decoder tests cover residual composition, real layers, slot isolation, replay,
  and branch ablations.
- Real-token attention composition separately gates rotated Q (PCC ≥ 0.9999,
  normalized L2 ≤ 0.01), source-to-stored K/V (PCC ≥ 0.9999/0.999 and normalized
  L2 ≤ 0.01/0.02 for BF16/BF8_B), and SDPA against an independent CPU reference
  using actual device Q and stored K/V (PCC ≥ 0.9999, normalized L2 ≤ 0.01).
  Source-head PCC, BF16 source-head normalized L2, and both source-output limits
  remain enforced. BF8 source-head normalized L2 is now characterization, replacing
  its previous hard gate: repeated-BOS inputs can amplify upstream rounding beyond 5% before SDPA,
  and SDPA error can cancel that drift. Both source-head drift and the independent
  exact-input reference's drift from the source remain in the metrics; the existing
  ideal-Q/cache-readback reference is retained for attribution.
- The full 2K boundary test scores every layer and KV head. Layer-zero checks
  and final logits are enforced. Later accumulated raw-FP32 hidden/KV drift is
  recorded with its original PCC ≥ 0.99 and normalized L2 ≤ 0.15 limits;
  those diagnostic rows are not all required to pass.
- Full 2K acceptance also requires the native-input companion for both cache
  dtypes and both prompts. It independently recomputes each layer from its actual
  device input. KV requires PCC ≥ 0.9999 and normalized L2 ≤ 0.01 for BF16,
  or PCC ≥ 0.999 and normalized L2 ≤ 0.02 for BF8_B. Hidden outputs require
  PCC ≥ 0.999 and normalized L2 ≤ 0.025/0.05 respectively. Final logits,
  cache boundaries, other slots, and uninstrumented replay remain enforced.
- The book test runs 32 layers at 2048 and 4096 tokens and requires the independent
  FP32 HF reference's top-1 token at the final position to appear in the device's
  top five. The committed public-domain input and goldens pin text, token IDs,
  checkpoint/tokenizer hashes, and reference settings. `scripts/generate_book_golden.py`
  regenerates the reference on a compute host. There is no 4K cache comparison.

Test registrations and thresholds describe coverage; passing results must come
from a run of the branch being reviewed.

## Common prefill runner

See the [runner acceptance guide](docs/prefill-runner.md) for the two-slot 2K
producer/runner tests, source KV address-table contract, and SC1 CI command.
