<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama 3.1 8B — disaggregated prefill

First working prefill implementation for Llama-3.1-8B-Instruct on a Blackhole Galaxy, built by
following [`common/prefill/docs/MODEL_BRINGUP_RECIPE.md`](../common/prefill/docs/MODEL_BRINGUP_RECIPE.md)
against the prefill spec `spec_llama31_8b_v0.json`.

Everything for this model lives in this one directory, and
[`run_bringup.sh`](run_bringup.sh) re-runs the whole ladder.

**This is the family's first GQA-DENSE model.** Every other `_d_p` package is MLA+MoE, except
`gpt_oss_d_p`, which is GQA+MoE with attention sinks, sliding-window alternation and QKV bias — none
of which Llama has. What that meant in practice is in [Reuse vs fresh](#reuse-vs-fresh).

---

## Architecture

| | |
|---|---|
| Attention | GQA, 32 Q heads / 8 KV heads (group 4), head_dim 128, **no** bias / sinks / sliding window / QK-norm |
| FFN | dense gated SwiGLU (silu), 4096 -> 14336 -> 4096, 3 matrices, no bias |
| Layers | 32, all identical — no MoE, no layer-type schedule |
| RoPE | full rotary (128), theta 500000, **llama3** smooth-ramp scaling (factor 8, low 1.0, high 4.0, orig 8192) |
| Norm | plain RMSNorm (no Gemma fold), eps 1e-5 |
| Vocab | 128256, untied lm_head |

## Topology

Spec `topology`: mesh **8x4**, **SP=8** on rows (axis 0), **TP=4** on cols (axis 1), one pipeline
stage. On a plain-MESH Galaxy that means `FABRIC_1D` + `ttnn.Topology.Linear` — there are no
wrap-around links, so `FABRIC_1D_RING` cannot be opened.

**TP=4 with 8 KV heads gives 2 KV heads per chip.** Both donor packages have exactly one, so the
canonical cache shape's `1` in the head dimension had to become `n_kv_heads // tp`, and the migration
bank walk had to step over the head dim. See [`docs/SPEC_NOTES.md`](docs/SPEC_NOTES.md) §1-2.

---

## Status

### Module PCC (random weights, identical both sides)

| Block | Mesh | PCC | Test |
|---|---|---|---|
| RMSNorm | 1x1, 8x4 | 0.99997 | `unit/test_norm_vs_ref.py` |
| SwiGLU activation | 1x1, 8x4 | 0.99999 | `unit/test_dense_mlp_vs_ref.py` |
| Dense MLP (bf16) | 1x1, 8x4 | 0.9995 – 0.9998 | `unit/test_dense_mlp_vs_ref.py` |
| Dense MLP (**bfp4**) | 1x1, 8x4 | **0.9894 – 0.9897** | `unit/test_dense_mlp_vs_ref.py` |
| RoPE one-shot, pos 0 / 65536 | 1x1 | 0.99999 | `unit/test_rope_vs_ref.py` |
| RoPE indexed SP, cached_len 0 / 4k / 64k | 8x4 | 0.99999 | `unit/test_rope_vs_ref.py` |
| Attention block | 1x1 | 0.9998 | `unit/test_attention_vs_ref.py` |
| KV cache write (seam) | 8x4 | K 0.99987 / V 0.99987 | `unit/test_kv_cache_write_vs_ref.py` |
| KV block-cyclic round-trip, 1 & 3 layers | 8x4 | 0.99998 | `unit/test_kv_cache_gqa_sp_vs_ref.py` |
| Ring SDPA, live Q/K/V | 8x4 | 0.99997 | `unit/test_ring_joint_sp_vs_ref.py` |
| Ring SDPA, cache-read | 8x4 | 0.99990 | `unit/test_ring_joint_cache_read_sp_vs_ref.py` |
| 2-chunk attention (production module) | 8x4 | KV 0.99987, out 0.9957 | `unit/test_attention_chunked_vs_ref.py` |
| Decoder layer (composition) | 1x1 | 0.9986 | `unit/test_decoder_layer_vs_ref.py` |
| Decoder layer (composition) | 8x4 | 0.9989 | `unit/test_decoder_layer_vs_ref.py` |
| Parallel embedding, 1D **and** 2D vocab-parallel | 8x4 | 1.0 | `unit/test_parallel_embedding_vs_ref.py` |
| Final norm + lm_head (real 128256 vocab) | 8x4 | 0.99986 | `unit/test_lm_head_vs_ref.py` |
| **Whole model**, 2 layers — final hidden | 8x4 | 0.99996 | `unit/test_model_sp_vs_ref.py` |
| **Whole model**, 2 layers — per-layer KV | 8x4 | 0.99981 – 0.99984 | `unit/test_model_sp_vs_ref.py` |
| **Whole model**, 4 layers — final hidden | 8x4 | 0.99991 | `unit/test_model_sp_vs_ref.py` |
| **Whole model**, 4 layers — per-layer KV | 8x4 | 0.99976 – 0.99984 | `unit/test_model_sp_vs_ref.py` |

Host-only (no device): 34 tests in `tests/torch/` pin the vendored torch reference against upstream
`transformers` — RMSNorm, MLP, llama3 rope frequencies, attention, decoder layer, whole model, the
chunked==one-shot invariant, and the golden-cache round-trip.

### Known gaps

- **bfp4 cannot meet the spec's 0.999 module gate** (measured 0.9895). The spec's `numerics` and
  `acceptance.pcc` are jointly unsatisfiable; both dtypes are constructor arguments here so the
  trade-off can be measured. Defaults: attention **bfp8**, MLP **bfp4**.
- **Attention weight precision is still contested** (spec `known_risks`): llm_perf says bfp4,
  tt_transformers' accuracy path says bfp8. Defaulted to the conservative bfp8.
- **No performance numbers.** Out of bring-up scope, and the spec carries no targets to gate against.
- **A `(1,4)` submesh cannot bring fabric up on this Galaxy**, so TP-without-SP is untested. Not a
  configuration the spec targets.

---

## Running

```bash
source ttenv.sh                                   # TT_METAL_HOME, PYTHONPATH, LD_LIBRARY_PATH

./models/demos/llama3_1_8b_d_p/run_bringup.sh          # everything, in recipe order
./models/demos/llama3_1_8b_d_p/run_bringup.sh d1       # host-only reference tests
./models/demos/llama3_1_8b_d_p/run_bringup.sh d3       # the decoder suite on the mesh
./models/demos/llama3_1_8b_d_p/run_bringup.sh m p      # model suite + migration table
```

Each group runs in its own pytest process. **This is required, not tidiness**: one fabric config per
process, one mesh shape per process — otherwise fabric bring-up dies with a misleading
`Fabric Router Sync: Timeout ... Ethernet handshake likely failed`. Per-group logs land in
`.bringup_runs/`.

### Real weights

```bash
export HF_MODEL=/path/to/Meta-Llama-3.1-8B-Instruct   # any ungated mirror; config must match configs/
PREFILL_CHUNKED=0 pytest models/demos/llama3_1_8b_d_p/tests/galaxy_prefill_kv_pcc.py   # P1 one-shot
PREFILL_CHUNKED=1 pytest models/demos/llama3_1_8b_d_p/tests/galaxy_prefill_kv_pcc.py   # P2 chunked
```

### Serving (two terminals, both local)

```bash
# A — runner
PREFILL_MANIFEST=models/demos/llama3_1_8b_d_p/tt/runners/manifests/llama3_1_8b.json \
PREFILL_H2D_SERVICE_ID=llama_prefill PREFILL_MOCK_MIGRATION=1 \
  python -m models.demos.common.prefill.runners.prefill_runner

# B — producer
PREFILL_MANIFEST=models/demos/llama3_1_8b_d_p/tt/runners/manifests/llama3_1_8b.json \
PREFILL_H2D_SERVICE_ID=llama_prefill PREFILL_PRODUCER_CHECK_PCC=1 \
  python -m models.demos.common.prefill.runners.prefill_producer
```

---

## Reuse vs fresh

| Part | Source | Mode |
|---|---|---|
| MeshConfig, CCLManager, CCL wrappers | `gpt_oss_d_p/tt/{config,ccl}.py` | copy (retargeted 4x8/TP=8 -> 8x4/TP=4) |
| RMSNorm | `gpt_oss_d_p/tt/rms_norm.py` | copy |
| Parallel embedding | `minimax_m3/tt/parallel_embedding.py` | copy (default 1D; dropped the sharded-residual coupling) |
| Weight cache | `minimax_m3/tt/weight_cache.py` | copy |
| Dense MLP | `minimax_m3/tt/dense_mlp.py` | copy; activation rewritten (clamped swigluoai -> plain SwiGLU) |
| KV cache cluster (7 roles) | `gpt_oss_d_p` — all seven from ONE package | copy; **head dim 1 -> `n_kv/tp`** |
| Attention plumbing | `gpt_oss_d_p/tt/attention/` | copy; biases, sinks, sliding window, o_proj padding all deleted |
| Ring SP attention | `gpt_oss_d_p/tt/attention/dense_sp.py` | copy; sink/sliding arguments removed, halo helper dropped |
| Runtime | `gpt_oss_d_p/tt/tt_prefill_runtime.py` | copy |
| Adapter, manifest | `ADDING_A_PREFILL_MODEL.md` | fresh (contract) |
| **RoPE frequencies** | — | **fresh**: the donor is YaRN, Llama is llama3 scaling |
| **Torch reference** | — | **fresh**: trimmed from `transformers` LlamaForCausalLM, torch-only |
| **Migration address walk** | `gpt_oss_d_p/tt/runners/kv_chunk_table.py` | **rewritten**: closed-form shard index, 16 configs |
| **Checkpoint loader** | `gpt_oss_d_p/tt/model_config.py` | rewritten: plain safetensors, no MXFP4 dequant |

The single most important donor decision: **all seven KV-cluster roles come from `gpt_oss_d_p`**, the
only GQA package. Mixing the allocator from one package with the ring SDPA from another produces a
silently broken model, never a crash.

---

## Layout

```
reference/       torch-only oracle: config constants, model, golden runner + reference cache
tt/              device implementation
  attention/     config, weights, kv_cache, operations, prefill, dense_sp
  runners/       adapter, KV chunk address table, serving manifest
tests/
  torch/         host-only (D1/M1): reference vs upstream HF, golden-cache round-trip
  unit/          per-module PCC on the target mesh (D2/D3, M2/M3)
  test_kv_cache_table.py     P4: address table vs device DRAM, bit-exact
  galaxy_prefill_kv_pcc.py   P1/P2: real-weights per-layer KV vs the CPU golden
docs/SPEC_NOTES.md           what the prefill spec template should carry next time
bringup_log.jsonl            append-only process log (recipe §7)
run_bringup.sh               one-command re-run
```
