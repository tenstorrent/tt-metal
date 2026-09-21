<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B standalone prefill

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

Weights and activations are BF16. KV defaults to BF8_B, with BF16 also covered
by the accuracy tests. Each TP column owns one KV head; SP rows own successive
256-token stripes. The local cache shape is `[64, 1, max_seq_len / 4, 128]`,
with user-major planes (`slot * 32 + layer`), 32-token round-robin DRAM pages,
and Meta-interleaved K after RoPE. `max_seq_len` defaults to 2048; the model
accepts a chunk-aligned capacity. Full-model book tests currently cover 2K and
4K, with 8K–64K explicitly skipped.

Attention gathers the selected cache prefix, restores natural token order,
and applies standard SDPA with an absolute-position causal mask, accurate
exponential mode, and FP32 destination accumulation. It requires the standard
SDPA accurate-exponential prerequisite included in this branch's base.

## Run the tests

Use a built tt-metal environment and a local checkpoint containing config,
tokenizer, safetensors index, and all shards. All checkpoint tests honor:

```bash
export LLAMA31_8B_CHECKPOINT=/mnt/models/meta-llama/Llama-3.1-8B-Instruct
export OMP_NUM_THREADS=16
```

On an allocated Blackhole Galaxy with the SP/TP ring links available:

```bash
python3 -m pytest models/demos/llama_3p1_8b_d_p/tests/unit -v
python3 -m pytest models/demos/llama_3p1_8b_d_p/tests/full_model/test_prefill_model_vs_ref.py -v
python3 -m pytest models/demos/llama_3p1_8b_d_p/tests/full_model/test_prefill_native_input_accuracy.py -v
python3 -m pytest models/demos/llama_3p1_8b_d_p/tests/full_model/test_prefill_book_top5.py -v
```

The CI registration in `tests/pipeline_reorg/blaze_models_prefill_tests.yaml`
separates CPU reference checks, components/decoder, full 2K accuracy, and book
2K/4K. The host entry uses `--noconftest` so collection does not load device
fixtures. Set `LLAMA_PREFILL_EVIDENCE_DIR` to a fresh directory to retain full
accuracy reports; the native-input test creates one directory per prompt/dtype.

## Accuracy contract

- Component tests compare norm, QKV, RoPE, cache writes, attention, MLP, residual,
  embedding, and vocabulary projection against independent Torch/HF references.
  Decoder tests cover real layers, slot isolation, replay, and branch ablations.
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
