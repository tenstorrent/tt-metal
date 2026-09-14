<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B-Instruct — prefill bring-up

TTNN implementation of **Llama-3.1-8B-Instruct** prefill on a **Blackhole Galaxy**, running
**SP=8 × TP=4** on the 8×4 mesh, validated per layer against a CPU golden KV trace.

Binding spec: [`configs/prefill_spec.json`](configs/prefill_spec.json) (a copy of the snapshot the
pipeline prepared; `PREFILL_SPEC` overrides it). Dimensions:
[`configs/config.json`](configs/config.json), a byte copy of the checkpoint's own.

Scope is the recipe's **E → D1–D3 → M1–M3 → P1–P2**: package scaffold, torch oracle, module-by-module
PCC, whole model, KV cache, real weights, chunked prefill. **Not** in scope and **not** implemented:
serving (adapter/manifest/runner), KV migration, perf tuning, decode.

## Architecture

| | |
|---|---|
| Decoder layers | 32, all identical — no hybrid schedule, no sliding/full alternation, no per-layer dispatch |
| Hidden / intermediate | 4096 / 14336 |
| Attention | GQA: 32 q / 8 kv heads, head_dim 128, **full** RoPE, **no** QK-norm, no sinks, no biases |
| RoPE | θ = 5e5 with `llama3` rescaling (factor 8, low 1, high 4, original context 8192) |
| MLP | dense SwiGLU, plain `silu(gate) * up` — **not** the clamped `swigluoai` of gpt-oss/M3 |
| Norm | plain RMSNorm (`x_normed * w`) — **no** Gemma `(1 + w)` fold |
| Vocab | 128256, `tie_word_embeddings: false` |
| Checkpoint | plain bf16 safetensors, **unquantized** (no dequant step exists or is needed) |

### Deployment on the 8×4 Blackhole Galaxy

* **TP=4 on the columns**: q/k/v and gate/up are column-parallel; o_proj and down_proj are
  row-parallel and close with a TP all-reduce. `hidden/tp` = 1024 and `intermediate/tp` = 3584 are
  both tile-aligned, so no projection needs output-dim padding.
* **SP=8 on the rows**: the sequence is sharded across the mesh rows (640 tokens per row at
  chunk 5120), and attention is `ring_joint_scaled_dot_product_attention` over the SP axis — the ring
  gathers the other rows' K/V internally by online softmax, so there is no explicit all-gather and no
  host-built mask.
* **Residual stream: replicated** full `hidden_size` on every TP column. The `emb/tp`-sharded
  residual MiniMax-M3 offers is a perf lever and is deliberately **not** carried over (see Gaps).
* **KV cache**: SP-sharded block-cyclic on the DRAM ND-shard substrate, 2 tensors (`k`, `v`), bf8_b,
  written by `update_padded_kv_cache`. Chunk N > 0 reads the accumulated prefix on device.
* **Embedding**: sharded, 1D (`emb_dim` on TP) by default; 2D vocab-parallel under
  `LLAMA_EMBED_SHARD_VOCAB=1`. Both are tested and agree to PCC 1.000000.
* **LM head**: column-parallel on vocab. Prefill is headless — the populated cache is the output.

### KV cache layout

Canonical by convention across the prefill packages, because the chunked ring SDPA reads it:

| Element | Value |
|---|---|
| Per-chip shape | `[num_users * num_layers, num_kv_heads/tp, max_seq_len/sp, head_dim]` = `[32, 2, 1280, 128]` |
| Slot packing | `slot = user_id * num_layers + layer_idx` (user-major) |
| DRAM | `NdShardSpec`, shard `[1, 1, 32, 128]`, `ROUND_ROBIN_1D` over `get_num_dram_banks()` |
| Sequence | SP-sharded block-cyclic; alignment `max_seq_len % (32 * sp) == 0` |
| dtype | `bfloat8_b`, from the spec's `dataformats.kv_cache.default` |

**The one deviation from the sources.** Both `gpt_oss_d_p` and `minimax_m3` allocate `dim1 = 1` — one
KV head per chip — because GPT-OSS has 8 KV heads at TP=8 and M3 has 4 at TP=4. Llama-3.1-8B has
**8 KV heads at TP=4, i.e. 2 per chip**, and `update_padded_kv_cache` enforces
`cache_shape[1] == input_shape[1]`. The literal `1` encodes the *source's* head count, not the
layout, so it is re-derived; everything else about the layout is copied verbatim. This is the single
place this package departs from the canonical shape, and it fails loudly rather than silently — which
is why it cost one iteration and not a day.

## Exploration: what was reused, and the envelope it was measured at

Candidates were gated on the spec's **target hardware** first, from `ADAPTER_PATHS` in
`models/demos/common/prefill/adapter.py`. Only **`minimax_m3`** has run on a Blackhole Galaxy at
**(8, 4) with TP=4** — the exact mesh this spec binds — so it is the primary source despite being
sparse-attention + MoE: its *dense* layers are GQA at head_dim 128 and its SP/CCL/KV substrate is
measured at this shape. `gpt_oss_d_p` is the closer architectural match (GQA, 2 caches) but ran at
**sp4 × tp8, head_dim 64, chunk 1024**, so it was used for the math and the layout rationale only,
never for a copy. Everything else in `ADAPTER_PATHS` is MLA (DeepSeek / Kimi / GLM), which has the
wrong cache count.

| Part | Chosen | Envelope it was measured at | Rejected |
|---|---|---|---|
| Weight loading | fresh; walk structure from `minimax_m3/tt/model_config.py` | hidden 6144, hd 128, chunk 5120, sp8×tp4 | `gpt_oss_d_p` mxfp4 loader — wrong quantization scheme |
| Dequant | **none** — write-fresh not needed | — | checkpoint is plain bf16; asserted, not assumed |
| Norm / embedding | `minimax_m3/tt/{rms_norm,parallel_embedding}.py` | hidden 6144, sp8×tp4 | `gpt_oss_d_p` — sp4×tp8, wrong mesh orientation |
| MLP | `minimax_m3/tt/dense_mlp.py` — **structure only** | hidden 6144, inter 12288, sp8×tp4 | its `swigluoai` activation — Llama is plain silu |
| Attention | `minimax_m3/tt/attention/{operations,weights,dense_sp}.py` | hidden 6144, hd 128, chunk 5120, sp8×tp4 | `gpt_oss_d_p` — sinks + sliding, hd 64, sp4×tp8 |
| RoPE | `tt_transformers` `gather_cos_sin` / `get_rot_transformation_mat`; frequencies from this package's own reference | hd 128, llama3 scaling | M3's partial-rotary wrapper — Llama rotates the full head_dim |
| KV cache | `minimax_m3/tt/attention/kv_cache.py` — all five coupled roles from one package | hd 128, cache 10240, chunk 5120, sp8×tp4 | M3's third `index_k` cache — GQA needs 2, not 3 |
| Runtime | `minimax_m3/tt/tt_prefill_runtime.py` | chunk 5120, sp8×tp4 | DeepSeek's — MLA cache lifecycle |

**Imported, never copied**: `get_num_dram_banks` (`common/prefill/runners/migration.py`),
`blockcyclic_positions` / `block_cyclic_reorder` (`models/common/utils.py`), the DeepSeek golden-cache
helpers `ReferenceCacheKey` / `save_reference_cache` / `load_reference_cache`
(`deepseek_v3_d_p/utils/transformer_helpers.py`), and `gather_cos_sin` /
`get_rot_transformation_mat` (`tt_transformers/tt/common.py`).

**A source deliberately not used.** A previous bring-up of this model exists on the branch
`dgolubovic/llama-3-1-8b-prefil-run-1` as `models/demos/llama_3_1_8b_d_p` (only stale `__pycache__`
remains in this working tree). It was excluded: `prepared.yaml` branched this workspace from
`origin/main`, not from that run; the package path differs; and the recipe gates candidates on
`ADAPTER_PATHS`, which that package is not in on this branch.

## Topology

The pod is torus-wired, so both mesh-graph descriptors map. The descriptor is chosen in
`conftest.py` **before the cluster initialises**; the default is the plain mesh
(`single_bh_galaxy_mesh_graph_descriptor.textproto`, `FABRIC_1D`, `Topology.Linear`), which maps on
any galaxy. `LLAMA_TORUS=1` selects `single_bh_galaxy_torus_x_graph_descriptor.textproto`
(`FABRIC_1D_RING`, `Topology.Ring`).

**Every number in this file was measured on the `linear` topology.** The mesh smoke test passes on
both; linear and torus collective costs are not comparable, so no number here is a torus number.

## PCC status

Spec thresholds: **`pcc_lower_bound` 0.85** (the assert in every test) and **`pcc_target` 0.99**
(what every component aims for).

### Acceptance — real weights, **full 32-layer depth, full 4096 width**, `synthetic_10240`

Both modes run the same model against the same CPU golden trace; they differ only in `chunk_size`.

| Mode | min K | min V | layers < target | verdict | wall (tok/s) |
|---|---|---|---|---|---|
| `one_shot` (1 × 10240) | **0.9800152** | **0.9777526** | 21 / 32 | **PASS** | 18.9 s (541) |
| `chunked` (2 × 5120) | **0.9800080** | **0.9782111** | 21 / 32 | **PASS** | 25.6 s (400) |

The two modes flag the identical 21 layers, and layer 0 is bit-identical between them
(K 0.9859747, V 0.9998163).

* Every layer in both modes clears `pcc_lower_bound` 0.85 with a wide margin.
* **P2's goal is met**: chunked and one-shot agree per layer to ~1e-5 on K and ~5e-4 on V — multi-chunk
  prefill produces the same KV as an equal-length one-shot run.
* Throughput is context only; **nothing was tuned** and perf is out of scope.

### Why 21 layers land below `pcc_target` — root-caused

**The golden trace's RoPE inverse frequencies were rounded to float16 before the position outer
product.** Recomputing layer 0's post-RoPE K from the real checkpoint with fp16-rounded frequencies
matches the trace at PCC **1.000000**; with the fp32/fp64 frequencies HuggingFace actually uses it
matches at 0.9861 and decays monotonically with position. Evidence, all reproducible:

| Measurement | Result |
|---|---|
| Layer-0 K vs trace, full-precision frequencies | 0.986164 |
| Layer-0 K vs trace, **fp16-rounded frequencies** | **1.000000** |
| Per-2048-token K, full precision | 0.99938, 0.99576, 0.98882, 0.97918, 0.96765 |
| `inv_freq[0]` = 1.0 exactly representable in fp16 | **zero** drift, at any position |
| Worst accumulated phase error at token 10239 (j=1) | **1.68 rad** |
| Acceptance, `chunked`, with `LLAMA_ROPE_FREQ_FP16=1` | min K **0.993592** (from 0.980), layer-0 K **0.999743** (from 0.985975) |

Rounding `inv_freq` perturbs a *frequency*, so the phase error is `position × Δinv` — linear in
position rather than bounded. That is why K degrades with context and j=0 does not move at all.

**The device is not the limit.** Measured against this package's own reference (itself pinned to
HuggingFace at PCC 0.999989 on real weights), with real weights over the full 10240 tokens:

| Comparison | pos 0–2047 | 2048–4095 | 4096–6143 | 6144–8191 | 8192–10239 | all |
|---|---|---|---|---|---|---|
| **device vs reference**, layer 0 K, one-shot | 0.999743 | 0.999741 | 0.999743 | 0.999743 | 0.999744 | **0.999743** |
| **device vs reference**, layer 0 K, chunked | 0.999743 | 0.999741 | 0.999743 | 0.999743 | 0.999744 | **0.999743** |
| **device vs reference**, layer 0 V, both modes | — | — | — | — | — | **0.999817** |
| reference vs *trace*, layer 0 K | 0.99938 | 0.99576 | 0.98882 | 0.97918 | 0.96765 | 0.98616 |

The device's error is **flat** across position; the trace's is not.

**The default is not changed.** `tt/rope.py` and `reference/model.py` read one function,
`reference.model.llama3_inv_freq`, which computes full-precision frequencies. `LLAMA_ROPE_FREQ_FP16=1`
flips both sides onto the trace's convention for an A/B. Baking the fp16 rounding in by default would
give the device a 1.7-radian phase error against HuggingFace at 10k context, which is a real accuracy
loss for any other consumer.

The residual **V** decay with depth (layer 0 → 0.9998, layer 12 → 0.978) is unaffected by that knob
and is ordinary accumulation of bf8 weights / bf8 KV cache / bf16 activations against an fp16 CPU
reference over 32 layers. The whole-model random-weight test measures the same shape of decay against
a host reference (L0 0.99991 → L3 0.99961) with no trace involved.

### Component PCC — target mesh (8×4, linear), random weights, real dims

Everything below is at or above `pcc_target`; nothing landed in the accept-and-explain band.

| Component | Test | PCC |
|---|---|---|
| RMSNorm, hidden 4096 (s=256 / 1024) | `test_norm_vs_ref.py` | 0.999964 / 0.999962 |
| RMSNorm is plain, not Gemma (control) | `test_norm_vs_ref.py` | plain 0.999965 vs gemma-fold 0.892425 |
| SwiGLU activation | `test_mlp_vs_ref.py` | 0.999995 |
| Activation is silu-swiglu, not swigluoai (control) | `test_mlp_vs_ref.py` | 0.999995 vs swigluoai 0.954912 — the wrong one is **above** the 0.85 bound |
| Dense MLP 4096→14336→4096 (s=1024 / 5120) | `test_mlp_vs_ref.py` | 0.999899 / 0.999899 |
| RoPE per-chunk / indexed (chunk 0 and 1) | `test_rope_vs_ref.py` | 0.999994 |
| RoPE indexed == per-chunk | `test_rope_vs_ref.py` | > 0.9999 |
| KV cache write + read-back (L0 / L3) | `test_kv_cache_vs_ref.py` | 0.999974 / 0.999973 |
| KV cache 2-chunk append @ 10240 | `test_kv_cache_vs_ref.py` | 0.999973 |
| KV cache layer-slot isolation | `test_kv_cache_vs_ref.py` | 0.999974 (and < 0.5 against every other layer) |
| Ring SDPA, live K/V (s=1024 / 5120) | `test_ring_sdpa_vs_ref.py` | 0.999830 / 0.999809 |
| Ring SDPA, cache-read (2 × 1024 / 2 × 5120) | `test_ring_sdpa_vs_ref.py` | 0.999767 / 0.999731 |
| Ring SDPA reads the right layer slot | `test_ring_sdpa_vs_ref.py` | 0.999780 (and < 0.9 against layer 0's cache) |
| Attention block, one-shot | `test_attention_vs_ref.py` | 0.999733 |
| Attention cached K / V | `test_attention_vs_ref.py` | 0.999942 / 0.999945 |
| Attention chunk 1 (cache-read path) | `test_attention_vs_ref.py` | 0.999647 |
| **Decoder layer** / chunked / residual-only | `test_decoder_layer_vs_ref.py` | **0.999997** / 0.999998 / 0.999999 |
| Parallel embedding 1D / 2D / agreement | `test_embedding_lm_head_vs_ref.py` | 0.999999 / 0.999999 / 1.000000 |
| LM head (real vocab / unaligned vocab) | `test_embedding_lm_head_vs_ref.py` | 0.999973 / 0.999973 |
| Final norm | `test_embedding_lm_head_vs_ref.py` | 0.999962 |
| Runtime: compile-then-run == run | `test_runtime_contract.py` | 1.000000 |

The ~0.99997 floor on everything that round-trips through the cache is the bf8 quantisation limit of
the spec's `kv_cache` dtype, not an implementation loss.

### Reduced runs — **diagnostics, never results**

Labelled `REDUCED` wherever they appear, in the logs and here (recipe §4).

| Run | Reduction | Result |
|---|---|---|
| `test_model_sp_vs_ref.py` whole-model KV | **4 of 32 layers**, vocab 8192; full width, full chunk, full mesh | K/V 0.99991 (L0) → 0.99961 (L3) |
| `test_model_sp_vs_ref.py` e2e logits | same | 0.998797 |
| `test_model_sp_vs_ref.py` chunked == one-shot | same | 1.000000 (L0) → 0.999962 (L3) |
| `test_runtime_contract.py` | 2 of 32 layers | contract asserts + compile idempotence |
| `test_real_weights_vs_ref.py` | layer 0 only, **full 10240-token length**, real weights | K 0.999743, V 0.999817 |
| `test_reference_vs_hf_real_weights` | 2 of 32 layers, real weights | logits 0.999989, identical argmax |
| `test_reference_matches_golden_trace_convention` | 32 layers, first **512** tokens | min K 0.999945, min V 0.999990 |

A host cannot hold two copies of a 32-layer 8B model as random weights, which is exactly the case the
recipe describes. The graded whole-model number is the acceptance run above.

## Running it

```bash
cd <tt-metal>
export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH"
export PREFILL_HF_MODEL=/mnt/models/meta-llama/Llama-3.1-8B-Instruct
export PREFILL_TRACE_DIR=$PREFILL_HF_MODEL/golden/synthetic_10240
export PREFILL_SPEC=$TT_METAL_HOME/models/demos/llama_3_1_8b/configs/prefill_spec.json

# acceptance, one-shot and multi-chunk against the same trace
PREFILL_CHUNKED=0 PREFILL_ACCEPTANCE_OUT=/tmp/acceptance_one_shot.json \
  scripts/run_safe_pytest.sh models/demos/llama_3_1_8b/tests/test_prefill_acceptance.py
PREFILL_CHUNKED=1 PREFILL_ACCEPTANCE_OUT=/tmp/acceptance_chunked.json \
  scripts/run_safe_pytest.sh models/demos/llama_3_1_8b/tests/test_prefill_acceptance.py

# full package suite
scripts/run_safe_pytest.sh --run-all models/demos/llama_3_1_8b/tests/

# host-only subset (no device, no chip lock)
python3 -m pytest models/demos/llama_3_1_8b/tests/torch_ref/ \
                  models/demos/llama_3_1_8b/tests/test_bringup_log.py \
                  models/demos/llama_3_1_8b/tests/unit/test_weight_loader.py
```

`conftest.py` selects the mesh-graph descriptor before the cluster initialises, so
`TT_MESH_GRAPH_DESC_PATH` does not need to be exported. **Run every device test through
`scripts/run_safe_pytest.sh`** — it `flock`s chip access so several agents can share the pod, and
calling `pytest` directly turns a second job into an apparent hang.

The tilized weight cache defaults to `$TT_METAL_HOME/generated/llama_3_1_8b` (~8 GB) rather than a
sibling of the checkpoint, so a run never writes into the shared `/mnt/models` NFS store; set
`TT_CACHE_PATH` to move it. The first run builds it (~40 s); later runs reuse it.

### Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `LLAMA_TORUS` | `0` | `1` selects the torus mesh descriptor + `FABRIC_1D_RING` + `Topology.Ring` |
| `LLAMA_EMBED_SHARD_VOCAB` | `0` | `1` selects the 2D vocab-parallel embedding (less memory, two SP collectives) |
| `LLAMA_ROPE_FREQ_FP16` | `0` | `1` rounds the rope inverse frequencies to fp16, reproducing the golden trace's convention (see above) |
| `LLAMA_RING_FP32_ACC` | `0` | `1` turns on `fp32_dest_acc_en` for the ring SDPA (off by default; the projections keep it on) |
| `LLAMA_LOAD_NLAYERS` | unset | load only layers `0..N-1` — a debugging aid; any number under it is a **reduced** run |
| `LLAMA_NUM_LINKS` | auto | fabric links per CCL call |
| `TT_CACHE_PATH` | `$TT_METAL_HOME/generated/llama_3_1_8b` | tilized weight cache location |

## Layout

```
configs/          vendored config.json (byte copy of the checkpoint's) + the prefill spec snapshot
reference/        torch-only oracle (config, model, golden cache) — no ttnn, no HF at import
utils/            state-dict slicing, weight-cache paths, and the HF<->Meta RoPE column permutation
tt/               ccl, mesh, compute configs, rms_norm, mlp, rope, parallel_embedding, lm_head,
                  layer, model, model_config (checkpoint loader), tt_prefill_runtime
tt/attention/     config, weights (QKV fusion + Meta permute), operations, kv_cache, dense_sp, prefill
tt/runners/       per-layer KV PCC against the golden trace
tests/torch_ref/  host-only: reference vs HF, golden cache, golden-trace rope root cause
tests/unit/       module-by-module PCC on the target mesh
tests/            acceptance test + the recipe §7 log lint
bringup_log.jsonl the append-only process log (recipe §7)
```

## Known gaps

No blockers: the full-depth, full-width, real-weights end-to-end number exists and passes in both
modes. In priority order:

1. **The golden trace's fp16 rope frequencies.** Root-caused above, but the trace itself is not
   fixed — that belongs to whoever owns the generator for
   `/mnt/models/meta-llama/Llama-3.1-8B-Instruct/golden/`. Until it is regenerated, the acceptance K
   numbers understate this implementation by ~0.014 PCC.
2. **Whole-model random-weight test is depth- and vocab-reduced** (4 of 32 layers, vocab 8192). Host
   memory, not a correctness gap; the full-depth path is covered by acceptance on real weights.
3. **No sharded (`emb/tp`) residual stream.** One layout, no knob. The sibling package measures the
   sharded layout as a perf win (all-reduce → reduce-scatter, cheaper norms). Perf is out of scope;
   this is the obvious first move in a perf pass.
4. **Serving and KV migration are not implemented** — explicitly a separate follow-on. There is no
   `PrefillModelAdapter`, no `ADAPTER_PATHS` entry and no manifest. The per-layer seam exists
   (`Model.prefill_forward(on_layer_complete=...)`, `TtPrefillRuntime.set_layer_ack_channel`) and is
   unused.
5. **No perf work at all.** No ttnn trace capture, no tuned program configs, no fused matmul +
   reduce-scatter (that op is Ring-only and this galaxy runs `FABRIC_1D` Linear by default). The
   throughput figures above are untuned.
6. **Top-1 / logits agreement with HF is not tracked end to end on device.** The recipe excludes it
   from done and KV PCC is the proxy. The LM head is implemented and unit-tested but is off the
   acceptance path, since prefill runs headless.
7. **Single user slot** (`num_users=1`). The cache layout is user-major and sized for more, but only
   slot 0 is exercised.
8. **Only the 10240-token trace is used.** `synthetic_5120b` exists, but at 5120 tokens the chunked
   mode degenerates to one chunk, so it is not a valid acceptance trace for P2.
9. **`bringup_digest.py` does not exist on this branch**, so recipe §8's `--lint` cannot be run as
   written. The §7 lint is implemented as `tests/test_bringup_log.py` and runs in the suite instead.
