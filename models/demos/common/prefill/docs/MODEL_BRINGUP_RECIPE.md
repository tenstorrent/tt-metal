<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prefill model bring-up recipe

End-to-end path from a HuggingFace checkpoint to a model served by the
`models/demos/common/prefill` engine.

**Scope:** first working implementation — package scaffold, torch oracle, module-by-module PCC
bring-up, full model, KV cache, runtime, adapter, migration table.
**Not in scope:** perf tuning, sharding/memory optimization, dtype accuracy trade-offs, new C++
kernels, decode.

Companion: [`ADDING_A_PREFILL_MODEL.md`](ADDING_A_PREFILL_MODEL.md) — the serving contract this
recipe exists to make satisfiable.

> **Status: outline.** Sections marked `TODO` are stubs to be filled in one at a time.

---

## 1. Inputs to the bring-up

Two JSON files. Everything the agent needs comes from one of them; nothing is inferred.

### 1.1 Donor map — `donor_template.json` (advisory)

*How this repo builds a prefill model.* For each part that is copied rather than written fresh, it
names the existing file to copy from: weight loading, dense MLP, MoE wrapper, attention, rope, the
runtime, migration hooks, and the KV cache cluster (one donor package resolving seven coupled roles).
Every entry is copy-and-adapt — see §2.1 for how far that goes.

Pointers that never vary by model are **not** in this file — they live in
[Fixed references](#3-fixed-references) below.

### 1.2 Prefill spec — `TODO: template not written yet` (binding)

*What this model is.* The model-specific facts: HF reference and config, checkpoint dtype and
quantization, tensor naming, layer map, attention parameters, MoE parameters, target arch and mesh,
TP/SP/EP split, chunk size, sequence length, user count, PCC thresholds, golden trace location.

`TODO` — write `prefill_spec_template.json` and replace this paragraph with a field list plus the
fields that fail *silently* when guessed.

### 1.3 Precedence

The **spec is binding** and has the highest priority of any input. Every value in it must be
respected exactly, at every stage. Nothing in a donor, and no convention in this document, overrides
it.

The **donor map is advisory**. It points at existing implementations to guide and speed up the work.
Where a donor conflicts with the spec, or where following it would violate the spec, the donor gives
way — and that is a signal the donor was the wrong choice, not a reason to bend the spec.

---

## 2. Donors

### 2.1 What a donor is for

A donor is a working prefill package that already solved the same problem for a different model.
Reading it before writing a part serves two purposes:

- **Don't reinvent what exists.** Most of a prefill package is plumbing every model needs —
  sharding, collectives, weight caching, KV layout, the serving contract. It is already written and
  debugged several times over.
- **Don't diverge from patterns that are load-bearing.** Some conventions look arbitrary and are
  not: the KV cache layout is read by the ring SDPA op and the migration address walk, collectives
  must go through the CCL manager's semaphore ping-pong, weight-cache keys must match what the
  cache-populate run wrote. Deviating from these does not raise an error — it degrades PCC, corrupts
  output nondeterministically, or silently misses the cache.

**Start from the donor's file: copy it into the new package and adapt it.** It is a template, not a
reference you merely consult — but adapting is the point, not transcribing. Take its structure, its
sharding and CCL placement, its program configs, its lifetime and deallocation discipline. Do not
carry over its model-specific content: attention math, config constants, the HF weight-name map, or
anything the spec fixes.

A donor is advisory (§1.3). If following one would violate the spec, the donor is the wrong choice.

### 2.2 What the donor map covers

`donor_template.json` is the single source for which package each part comes from — filled per
bring-up, one pointer per part whose donor depends on the model:

| Entry | What you are reading it for |
|---|---|
| `weights.loading` | safetensors iteration, prefix filtering, dtype conversion |
| `weights.dequant_and_permute` | qkv fusion/swizzle, tile-alignment padding, bias handling |
| `compute.mlp_dense` | column/row-parallel split and where the CCL lands |
| `compute.moe_wrapper` | thin wrapper over the imported EP substrate: router, activation, shared expert |
| `compute.attention` | head split, projection sharding, program configs, output-proj CCL tail — **not** the attention math |
| `compute.rope` | variant plumbing, and the whole-cache indexed rope built once |
| `kv_cache` | one donor package resolving seven coupled roles — see §5 |
| `serving.runtime` | `compile` / `make_chunk_input` / `prefill_chunk` and its chunk-range assertions |
| `migration.hooks` | `kv_migration_base_address` / `kv_migration_stages` / `set_layer_ack_channel` |

Parts whose pointer never varies by model — norm, embedding, MoE substrate, MeshConfig, CCLManager,
weight cache, test scaffolds, golden cache — are not in the donor map. They are fixed, and listed
below.

---

## 3. Fixed references

Pointers that do NOT vary from model to model. Take them as-is; the donor map
(`donor_template.json`) carries only the entries whose donor depends on the model.

> Provisional home. These belong in their per-stage sections (D3, M3, P1, ...) and will be moved
> there once those sections are written. Kept in one block for now so the donor template stays clean.

| What | Pointer | Mode |
|---|---|---|
| RMSNorm | `models/demos/gpt_oss_d_p/tt/rms_norm.py` | copy |
| Embedding | `models/demos/minimax_m3/tt/parallel_embedding.py` | copy |
| MoE substrate | `models/demos/deepseek_v3_d_p/tt/moe/` | **import** |
| MeshConfig | `models/demos/gpt_oss_d_p/tt/config.py` | copy |
| CCLManager | `models/demos/gpt_oss_d_p/tt/ccl.py` | copy |
| CCL wrappers | `models/demos/gpt_oss_d_p/tt/config.py` | copy |
| Weight cache | `models/demos/minimax_m3/tt/weight_cache.py` | copy |
| Migration helpers | `models/demos/common/prefill/runners/migration.py` | **import** |
| Runtime contract | `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` | contract |
| Reference trimming | `models/demos/deepseek_v3_d_p/reference/kimi_k3/modeling_kimi_k3_mla.py` | pattern |
| Reference purity | `models/demos/deepseek_v3_d_p/reference/kda/README.md` | contract |
| PCC test scaffold | `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py` | pattern |
| Config diff test | `models/demos/deepseek_v3_d_p/tests/torch/test_kimi_k3_mla_reference.py` | pattern |
| KV table test | `models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py` | pattern |
| Mesh KV-PCC | `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py` | pattern |
| CPU golden cache | `models/demos/deepseek_v3_d_p/utils/transformer_helpers.py:762` | import |
| Golden trace generation | `models/demos/deepseek_v3_d_p/tt/runners/generate_prompt_trace.py` | pattern |

Two rules the golden cache embodies, worth keeping: key the cache on **every** field that changes the
output (`ReferenceCacheKey` is frozen so a changed field yields a different filename and stale results
are never reused silently), and **assert rather than recompute** where a CPU run is expensive — see
`tests/test_mla.py:293`, which fails loudly on a cache miss in CI instead of burning an hour.
Per-module goldens are cheap to regenerate and are deliberately not cached.

---

## 4. Stages and order

Three ladders, run in order and one at a time: the decoder (D1-D3), then the whole model around it
(M1-M3), then the prefill pipeline (P1-P4). D and M have the same three-step shape — torch golden,
then a mock outline plus PCC tests, then implement in ttnn, with a torch CPU fallback only where
ttnn genuinely cannot.

**Prerequisite for D3 — mesh up.** Bring-up goes straight to the target mesh; there is no
single-card step at any point. Before the first module is written: the target mesh opens,
`MeshConfig` and `CCLManager` are in place, and an all-gather + all-reduce smoke test passes. Every
module PCC test therefore exercises sharding and collectives from the first one — more setup cost,
and the "worked on one card, broke on the mesh" class of bug disappears entirely.

All PCC tests up to P1 run on **random weights**, identical on both sides. Real checkpoint loading
is not a dependency of any module test and is deferred to P1.


Decoder bringup stages:

- **D1 — Torch golden impl**

  No TTNN yet.

- **D2 — Mock decoder + PCC tests for each component**

  Write mock decoder layer using mock building blocks that should be implemented later (like here: models/demos/gpt_oss_d_p/tt/layer.py). Write PCC tests for each module and whole decoder.

- **D3 — Implement 1 Decoder as composition of big blocks (Attn, MLP, Emb,...)**

  Bringup previously defined modules on target device straight away (no need for single chip bringup first). Where no single ttnn op matches the block, write the mathematical equivalent out of the ttnn ops that do exist — there is no `ttnn.mlp`, and an MLP is a matmul, an activation and a second matmul, so compose it rather than dropping to CPU. Fall back to pytorch CPU only when the math cannot be expressed in ttnn at all; if the fallback is inevitable, take it, log it and move on.


When each stage is finished - goals:
- D1 goal → Have torch implementation for each building block of model Attn,MLP,Norms,Emb.
- D2 goal → Have layer.py outline of one decoder layer in high level blocks so we can write PCC tests for each building block and decoder. 
- D3 goal → Have all building modules of decoder passing PCC tests in isolation (ttnn wherever the math can be composed from ttnn ops; torch CPU only where it cannot). Have one decoder block passing PCC test.

We switch from one stage to next only when the goal is reached.


Whole model bringup stages:

Same shape as decoder bringup. The decoder is already implemented and PCC-tested from D1-D3, so it
enters as a finished block; only the parts around it are new.

- **M1 — Torch golden impl (whole model)**

  Write what is missing for the whole model pytorch golden reference: embedding, final norm, lm head, and the layer stack itself. No TTNN yet.

- **M2 — Mock prefill pipeline + PCC tests for each new component**

  Mock the prefill pipeline: call decoder N times, embedding, final norm, lm head (like here: models/demos/gpt_oss_d_p/tt/model.py). Nothing needs to be implemented in this stage, just have the outline to generate tests. Generate PCC tests for the components that need to be implemented (decoder already has impl and tests).

- **M3 — Implement the whole-model components**

  Bringup embedding, final norm, lm head and the N-layer stack on target device. Same rule as D3: where no single ttnn op matches the block, write the mathematical equivalent out of the ttnn ops that do exist before considering a fallback. Fall back to pytorch CPU only when the math cannot be expressed in ttnn at all; if the fallback is inevitable, take it, log it and move on.


When each stage is finished - goals:
- M1 goal → Have torch implementation for every remaining building block: embedding, final norm, lm head, and a full-model forward that matches HF.
- M2 goal → Have model.py outline of the whole model in high level blocks so we can write PCC tests for each new building block and for the e2e model.
- M3 goal → Have all new building blocks passing PCC tests in isolation (ttnn wherever the math can be composed from ttnn ops; torch CPU only where it cannot). Have the whole model passing e2e PCC test. Decoder blocks from D3 still pass.

We switch from one stage to next only when the goal is reached.

New at model scale (did not appear in decoder bringup, define in M1/M2):
- weight loading for all N layers, not one — per-layer state_dict slicing and naming
- KV cache sized for N layers instead of 1
- per-layer type dispatch if the model is hybrid (which layer index gets which block)


Prefill pipeline stages:

The decoder and the model now run; what is left is real weights, chunking, and the serving contract.

- **P1 — Real weights**

  Load the actual checkpoint: safetensors iteration, dequant, qkv fusion/permutation. First run with real weights on the target mesh, one-shot (no chunking).

- **P2 — Chunked prefill**

  KV cache read-back path (ring SDPA over the block-cyclic cache) and the runtime that drives it: compile, make_chunk_input, prefill_chunk with actual_start/actual_end.

- **P3 — Serving**

  KvCaches handle, adapter, ADAPTER_PATHS line, manifest. Runner + producer in two terminals, both local: no inference server and no decode side needed.

- **P4 — Migration**

  KV chunk address table, kv_migration_base_address / kv_migration_stages, layer acks.


When each stage is finished - goals:
- P1 goal → Model runs with real checkpoint weights on target mesh, one-shot. Per-layer KV PCC vs golden trace passes. Random-weight tests from D3/M3 still pass.
- P2 goal → Multi-chunk prefill produces the same KV as an equal-length one-shot run. Chunked per-layer KV PCC vs golden passes.
- P3 goal → `prefill_runner` serves chunks pushed by `prefill_producer`; `PREFILL_PRODUCER_CHECK_PCC=1` passes (runner with `PREFILL_MOCK_MIGRATION=1`).
- P4 goal → Host-only KV address-table test passes; table + device map export to file (`PREFILL_MIGRATION_EXPORT_TO_FILE=1`) and validate offline; mock/loopback migration manifests run.

We switch from one stage to next only when the goal is reached.


Where the KV cache lands: the **layout is decided in D2** (both the attention read path and the
address-table math encode it), **allocation + write** are implemented in D3 (the layer writes K/V
even one-shot), and the **cache read** arrives in P2. GPT-OSS shipped it the same way — KV cache at
P2 of its stack, chunked ring SDPA at P6.

## 5. KV cache allocation (the critical decision)

The highest-leverage decision in the build: bound simultaneously to the attention op, the SP/TP
split, DRAM bank geometry, and the migration address walk.

### 5.1 Do not invent a layout

The layout is already **canonical by convention** across the prefill packages, because the chunked
ring SDPA and the migration address-table walk both read it. `gpt_oss_d_p/tt/attention/kv_cache.py`
and `minimax_m3/tt/attention/kv_cache.py` are near-identical for this reason.

Fixed — copy verbatim:

| Element | Value |
|---|---|
| Per-chip shape | `[num_users * num_layers, 1, seq_local, head_dim]` |
| Slot packing | `slot = user_id * num_layers + layer_idx` (user-major, layers contiguous) |
| DRAM memory config | `NdShardSpec`, shard `[1, 1, 32, head_dim]`, `ROUND_ROBIN_1D` over the DRAM bank grid |
| Contiguous tokens per bank | `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32` |
| Sequence sharding | SP-sharded block-cyclic on the SP axis; `seq_local = max_seq_len // sp` |
| Alignment constraint | `max_seq_len % (TILE_SIZE * sp) == 0` |
| Allocation | zeroed, `ReplicateTensorToMesh` (content diverges on first write) |
| Write op | `ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ...)` |
| Bank count | `get_num_dram_banks(mesh_device)` from `common/prefill/runners/migration.py` |

Varies per model — the only things to change:

| Element | Examples |
|---|---|
| Number of cache tensors | MLA: 1 latent (`kvpe`) · GQA: 2 (`k`, `v`) · M3: 3 (`k`, `v`, `index_k`) |
| Which heads a chip holds | decided at *write* time by the mesh mapping, not at allocation |
| `head_dim`| 64 / 128|
| `cache_dtype` | `bfloat8_b` typical |
| Extra replicated caches | e.g. M3's `index_k` is TP-replicated, not head-sharded |

`init_kvpe_cache` is MLA-specific (single latent cache) — the GQA packages deliberately do not use
it, but they do reuse its NdShard spec so `update_padded_kv_cache` works unchanged.

### 5.2 What the human must define

Four decisions. Everything else follows from the fixed layout above.

**1. Number of cache tensors** — falls out of the attention family:

| Family | Tensors | Why |
|---|---|---|
| MLA | 1 — `kvpe` | one latent row per token, no separate K/V |
| GQA / MHA | 2 — `k`, `v` | |
| Sparse (MSA / DSA) | 3 — `k`, `v`, `index_k` | the indexer needs its own key cache |

**2. `head_dim`** — the per-chip row width, not the model's head dim:

- GQA / MHA: the head dim as-is (64 for GPT-OSS, 128 for M3).
- MLA: `kv_lora_rank + qk_rope_head_dim` — the whole latent row (512 + 64 = 576 for DeepSeek and
  Kimi). The rope columns are cached even when they are never rotated.

**3. `cache_dtype`** — specified in Prefic Spec json.

**4. Auxiliary caches** — any extra cache the attention needs, and whether it is head-sharded or
TP-replicated. M3's `index_k` is replicated across TP columns and only the sparse layers write it;
dense-layer slots stay zeroed, because capacity is cheap and uniform packing keeps the slot math
simple.

Not a decision: **which head a chip holds.** That is set at write time by how the input chunk is
mesh-mapped, not at allocation — every chip is allocated the same zeroed buffer with
`ReplicateTensorToMesh` and the content diverges on the first write.

---

## 6. Stage detail

One subsection per stage. Fill in individually.

Each stage below ends in a **Testing** table. The first column references an existing test that the
agent implements an equivalent of for this model — copy its structure, not its content.

Rows marked **SHARED** are the exception: those tests are already model-agnostic and live in
`common/prefill`. Do **not** rewrite them per model — select the model with env and run them as they
are.

**A stage is complete when, and only when, every test in its Testing table passes.** No stage is
entered before the previous stage's table is green.

---

### D1 — Torch golden impl (decoder)

**Steps**
1. Vendor the model's `config.json`; write the config constants class.
2. Get a torch reference for the decoder blocks. Import the HF modeling file directly if it imports
   and constructs standalone; otherwise trim and vendor the classes you need, recording upstream
   line numbers as provenance. The reference imports torch only — no ttnn, no device code.
3. Write the golden runner: run the reference and dump each block's inputs/outputs to disk, keyed on
   everything that changes the result. Reuse `ReferenceCacheKey` + `save_/load_reference_cache`.
4. Create the tests in the Testing table.

**Testing** — host only, no device.

| Reference for the test to implement | What it compares |
|---|---|
| `deepseek_v3_d_p/tests/torch/test_kimi_k3_mla_reference.py` | Every constant in the config class against the vendored `config.json`, and the vendored torch reference against the upstream HF math. No TTNN. |
| `minimax_m3/tests/unit/test_reference_model.py` | The standalone CPU reference against the inline torch golden, on a reduced config with random weights — so the two oracles cannot drift apart. |

**Goal** D1 passes when both tests above pass.

---

### D2 — Mock decoder + PCC tests

**Steps**
1. Write `layer.py` composing the decoder from named blocks; every block is a mock. Nothing is
   implemented in this stage.
2. Fix each block's constructor and `__call__` signature — this is the interface commitment the
   tests are written against.
3. Decide the KV cache layout (§5.2). It is encoded by both the attention read path and the address
   table, so it cannot wait.
4. Write the whole decoder test suite (table below) against those signatures.

**Testing** — this is where **the decoder test suite** is written. Target mesh, random weights,
identical weights on both sides. Drop rows for features the model does not have.

the model does not have.

| Reference for the test to implement | What it compares |
|---|---|
| `minimax_m3/tests/unit/test_norm_vs_ref.py` | RMSNorm output vs a torch reference, including the Gemma `(1 + weight)` fold if the model uses it. |
| `minimax_m3/tests/unit/test_swiglu_vs_ref.py` | The activation vs a torch reference at the model's exact variant and constants (e.g. swigluoai alpha / clamp limit). |
| `minimax_m3/tests/unit/test_qk_norm_vs_ref.py` | Per-head QK-norm vs a torch reference. Only if the model has QK-norm. |
| `minimax_m3/tests/unit/test_attention_vs_ref.py` | The whole attention block vs a torch reference: QKV proj → head split → QK-norm → RoPE → causal SDPA → o_proj. Same random weights both sides, shared cos/sin so the test measures attention and not the RoPE constants. |
| `minimax_m3/tests/unit/test_ring_joint_sp_vs_ref.py` | The SP-sharded ring SDPA against a torch reference with **live Q/K/V** (no cache): that gathering KV across the SP axis by online softmax gives the same answer as unsharded attention. |
| `minimax_m3/tests/unit/test_ring_joint_cache_read_sp_vs_ref.py` | The same op reading K/V **out of the block-cyclic KV cache** — short Q against a longer accumulated prefix. This is the mechanism chunked prefill depends on. |
| `minimax_m3/tests/unit/test_kv_cache_write_vs_ref.py` | Cache contents after a write through the production prefill seam, read back and PCC'd against the torch reference's K/V — i.e. the write landed at the right slot and offset. |
| `minimax_m3/tests/unit/test_kv_cache_gqa_sp_vs_ref.py` | Write **and** read-back for the model's own cache shape on the chunked-KV substrate, at target SP × TP. |
| `minimax_m3/tests/unit/test_attention_chunked_vs_ref.py` | A 2-chunk sequence pushed through the **same** `Attention` module two ways; asserts the second chunk's output matches. Proves the cache-read path is wired, not just callable. |
| `minimax_m3/tests/unit/test_dense_mlp_vs_ref.py` | Dense MLP vs a torch reference at real dims. |
| `minimax_m3/tests/unit/test_fused_gate_vs_ref.py` | The fused MoE gate (`moe_grouped_topk`) vs the model's routing rule at its exact expert count and top-k. |
| `minimax_m3/tests/unit/test_ep_moe_vs_ref.py` | Router + shared expert + expert-parallel routed experts vs a torch reference, at real dims and the production EP dispatch. |
| `minimax_m3/tests/unit/test_decoder_layer_vs_ref.py` | One complete decoder layer with residuals vs a torch reference — the composition, after every piece above passes alone. |

**Goal** D2 passes when the decoder test suite **collects and runs**, and every failure is a PCC or
`NotImplementedError` — not an import error, a missing fixture, or a signature mismatch.

---

### D3 — Implement decoder on target mesh

**Steps**
1. Mesh prerequisite first: target mesh opens, `MeshConfig` and `CCLManager` in place, all-gather +
   all-reduce smoke test passes. No module work before this.
2. Implement in dependency order — norm, activation/MLP, rope, attention (one-shot and cache-read),
   KV cache allocation + write, MoE, then the composed layer.
3. Where no single ttnn op matches a block, write the mathematical equivalent out of the ttnn ops
   that do exist — `ttnn.mlp` does not exist, but an MLP is a matmul, an activation and a second
   matmul, so compose it. Same for every other block with no one-call equivalent: decompose it
   before reaching for a fallback.
4. Fall back to torch on CPU only where the math cannot be expressed in ttnn at all. If the fallback
   is inevitable, take it, log it, and move on — a new kernel is out of scope for bring-up.

**Testing** — the decoder test suite written in D2. No new tests.

**Goal** D3 passes when every applicable test in the decoder test suite passes, with each block in
ttnn wherever its math can be composed from ttnn ops and a logged torch CPU fallback only where it
cannot.

---

### M1 — Torch golden impl (whole model)

**Steps**
1. Extend the D1 reference with what the decoder did not need: embedding, final norm, lm head, and
   the layer stack.
2. Extend the golden cache to the end-to-end output. A CPU forward of the full model is expensive —
   it must run once, not per test.
3. Create the tests in the Testing table.

**Testing** — host only.

| Reference for the test to implement | What it compares |
|---|---|
| `minimax_m3/tests/unit/test_reference_model.py`, widened to the full model | The whole-model CPU reference forward against the inline torch golden, all layers, random weights. |
| `minimax_m3/tests/golden_hf_first_token.py` | The reference against the **real HF checkpoint** loaded on CPU — ground truth for the whole model, not just self-consistency. |
| No donor — author it | That the golden cache round-trips: a second run loads from disk instead of recomputing, and a changed `ReferenceCacheKey` field forces a miss rather than silently reusing a stale result. |

**Goal** M1 passes when all three pass.

---

### M2 — Mock prefill pipeline + PCC tests

**Steps**
1. Write the model outline: embedding, decoder x N, final norm, lm head. The decoder slot is the
   real D3 module; everything else is a mock.
2. Fix the signatures of the new components, as in D2.
3. Write the whole model test suite (table below).

**Testing** — this is where **the model test suite** is written. Target mesh, random weights.

| Reference for the test to implement | What it compares |
|---|---|
| `minimax_m3/tests/unit/test_parallel_embedding_vs_ref.py` | Embedding lookup vs `torch.nn.functional.embedding`, for **both** sharding modes (emb-on-TP with vocab replicated, and vocab-sharded-on-SP). |
| `gemma4/tests/unit/test_lm_head.py` | LM-head projection vs a torch reference, with the vocab shard layout the model uses. |
| `minimax_m3/tests/unit/test_norm_vs_ref.py` (final-norm instance) | The model's final norm — same test, applied to the tail instance rather than a layer's. |
| `minimax_m3/tests/unit/test_model_sp_vs_ref.py` | The **whole model** at target SP × TP vs a composed torch reference: sequence sharded across SP rows, residual stream SP-sharded through every layer. Catches per-layer weight-slicing and layer-type-dispatch errors that single-layer tests cannot. |

**Goal** M2 passes when the model test suite collects and runs, failing only on PCC or
`NotImplementedError`.

---

### M3 — Implement whole-model components

**Steps**
1. Implement embedding, final norm, and lm head on the target mesh.
2. Wire the N-layer stack: per-layer weight slicing and naming, per-layer type dispatch if the model
   is hybrid, and the KV cache sized for N layers rather than 1.
3. Same rule as D3: where no single ttnn op matches a block, compose the mathematical equivalent
   from the ttnn ops that do exist before reaching for a fallback. Torch CPU only where the math
   cannot be expressed in ttnn at all — log it and move on if it is inevitable.

**Testing** — the model test suite written in M2, plus the decoder test suite as a regression.

**Goal** M3 passes when every test in the model test suite passes, with each new block in ttnn
wherever its math can be composed from ttnn ops and a logged torch CPU fallback only where it
cannot, and the decoder test suite still green.

---

### P1 — Real weights

**Steps**
1. Write the checkpoint loader: safetensors iteration, prefix filtering, dtype conversion.
2. Add dequantization and any qkv fusion / permutation the TT modules expect.
3. Create the tests in the Testing table.

**Testing** — target mesh, **real weights**, golden trace required.

| Reference for the test to implement | What it compares |
|---|---|
| `gpt_oss_d_p/tests/unit/test_mxfp4_loader.py` | Dequantized expert weights against a reference dequantization of the packed blocks + scales. Only if the checkpoint is quantized. Host-only. |
| `minimax_m3/tests/galaxy_prefill_kv_pcc.py`, run `PREFILL_CHUNKED=0` | Every layer's on-device K/V after a one-shot real-weights prefill, against the CPU golden trace. First test where real weights, full layer count, target parallelism and MoE all interact. Also reports throughput. |

**Goal** P1 passes when the loader test passes and the one-shot per-layer KV PCC clears the spec's
threshold — with the D3 and M3 random-weight tables still green.

---

### P2 — Chunked prefill

**Steps**
1. Implement the runtime: `compile`, `make_chunk_input`, `prefill_chunk`, with assertions on
   `actual_start` / `actual_end` so an out-of-contract chunk fails loudly.
   The cache-read op itself is already implemented and tested in D3.

**Testing** — target mesh, real weights.

| Reference for the test to implement | What it compares |
|---|---|
| `minimax_m3/tests/galaxy_prefill_kv_pcc.py`, run `PREFILL_CHUNKED=1` | Per-layer K/V after a **multi-chunk** prefill against the same golden trace P1 used one-shot — i.e. chunk N attending the prefix chunks 0..N-1 left in the cache produces the same result as processing the whole sequence at once. |

**Goal** P2 passes when the chunked run reaches the same per-layer PCC as P1's one-shot run.

---

### P3 — Serving

**Steps**
1. Write the `KvCaches` subclass and `allocate_kv_cache`.
2. Write the adapter: `load_hf_config`, `weight_cache_path`, `allocate_kv_cache`, `build_runtime`.
   Keep imports lazy — the producers import this module too.
3. Add the `ADAPTER_PATHS` line and the manifest JSON.
4. Add a KV read-back branch for your cache layout in
   `prefill_producer.py::_read_slot_kv_and_check_pcc`. It dispatches on `ADAPTER.name` and is
   deliberately **not** adapter-pluggable, so a new layout needs a branch there and its own decode.
   Without it the shared test below cannot validate anything.

**Testing** — no tests to author.

| Reference for the test to implement | What it compares |
|---|---|
| **SHARED — do not rewrite:** `models/demos/common/prefill/tests/test_producer_runner_e2e.py::test_producer_runner_pcc` | Spawns the runner and producer itself. The producer pushes token chunks over the H2D socket, then reads the KV back **device-lessly over UMD through the published address table** and PCCs it against the golden trace. Select the model with `PREFILL_MODEL` / `PREFILL_TRACE_DIR`. Scenarios cover full-depth single user, deterministic round-robin over 4 users, and seeded random interleave over 8 users with slot recycling. |
| **SHARED — do not rewrite:** `models/demos/common/prefill/tests/test_prefill_producer_kv_decode.py` | Decoding of a KV chunk's raw bytes for each supported cache format (row-major, FP8 with page padding, packed scaled FP8), and that an unknown format is rejected rather than silently misread. |

**Goal** P3 passes when the shared e2e test passes for this model on every scenario that applies to
it, and the decode test passes. No inference server and no decode side are involved.

---

### P4 — Migration

**Steps**
1. Write the KV chunk address table for your cache layout (bank walk, config ordering).
2. Implement `kv_migration_base_address` (or `kv_migration_stages` for several caches) and
   `set_layer_ack_channel`.
3. Create the tests in the Testing table.

**Testing**

| Reference for the test to implement | What it compares |
|---|---|
| `deepseek_v3_d_p/tests/test_kv_cache_table.py` — **port this one, not gpt_oss's** | Allocates a real cache on the target mesh, writes it, then reads raw bytes back **from device DRAM at the addresses the table computed** and compares **bit-exactly**. Proves the address arithmetic, the DRAM bank walk, the packed-byte decode, and that a protobuf round-trip preserves lookups. Runs no model and moves nothing over fabric. |
| `minimax_m3/tests/test_kv_chunk_table_merge.py` | The multi-stage (pipeline-parallel) table merge, driven with synthetic per-stage layouts. Device-free. Only if the model runs across multiple pipeline ranks. |
| Manual — Gate 2 in [`PREFILL_MIGRATION_TESTING.md`](PREFILL_MIGRATION_TESTING.md) | The real DRAM → transport → DRAM copy, source and destination sharing one table via loopback. The only gate needing external binaries (`migration_endpoint`, `migration_worker`). |

`gpt_oss_d_p/tests/test_kv_cache_table.py` is parametrized only for a `(2,4)` submesh, which cannot
bring up fabric on a Galaxy — port the deepseek variant, which uses the full mesh and has a
no-weights `random` case.

**Goal** P4 passes when the ported address-table test passes (plus the merge test if
pipeline-parallel). Gate 2 is the sign-off that bytes actually move, and is tracked separately
because of its external dependency.

---

## 7. Definition of done in terms of this model bringup

- [ ] Every module has a `*_vs_ref` test at or above its `unit_pcc` threshold
- [ ] Full model runs at target mesh shape with real weights; per-layer KV PCC recorded in `README.md`
- [ ] Adapter implements all four abstract methods; registered in `ADAPTER_PATHS`; manifest exists
- [ ] Runtime satisfies `ADDING_A_PREFILL_MODEL.md` §2 and asserts on out-of-contract chunk ranges
- [ ] Two-terminal producer PCC passes
- [ ] `README.md` records architecture, reuse-vs-fresh, PCC status, run commands, and known gaps

Explicitly **not** required to call bring-up done: perf numbers, registration in a CI tier, and
top-1 / logits agreement with HF. KV-cache PCC is a proxy for correctness, not a substitute for that
last one — track it as follow-on work, not as a blocker.
