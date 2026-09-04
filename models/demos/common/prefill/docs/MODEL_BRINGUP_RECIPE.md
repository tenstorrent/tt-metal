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

| What | Pointer | Mode | Note |
|---|---|---|---|
| RMSNorm | `models/demos/gpt_oss_d_p/tt/rms_norm.py` | copy | SHOULD BE SHARED: `models/common/rmsnorm.py` exists (258 lines) and no prefill package uses it |
| Embedding | `models/demos/minimax_m3/tt/parallel_embedding.py` | copy | Deliberately NOT gpt_oss_d_p: it replicates the table and carries a TODO to shard it |
| MoE substrate | `models/demos/deepseek_v3_d_p/tt/moe/` | **import** | dispatch/combine/reduce/routing_setup/routed_expert. Copying it into the new package is the failure this pointer prevents |
| MeshConfig | `models/demos/gpt_oss_d_p/tt/config.py` | copy | SHOULD BE SHARED: 5 copies. TP the only knob; SP/EP derived |
| CCLManager | `models/demos/gpt_oss_d_p/tt/ccl.py` | copy | SHOULD BE SHARED: 4 copies; gpt_oss_d_p and minimax_m3 are both exactly 139 lines, differing almost only in docstrings. Wrong semaphore ping-pong is silent corruption |
| CCL wrappers | `models/demos/gpt_oss_d_p/tt/config.py` | copy | allgather/allreduce composition + dealloc ordering. Never call the async CCL ops bare |
| Weight cache | `models/demos/minimax_m3/tt/weight_cache.py` | copy | SHOULD BE SHARED. Also take `get_cache_file_name` from the donor's `utils/general_utils.py`. Cache-populate and serving runs must agree on key naming or the cache silently misses |
| Migration helpers | `models/demos/common/prefill/runners/migration.py` | **import** | `serialize_kv_chunk_table`, `KvCacheStage`, `get_num_dram_banks`. Already common |
| Runtime contract | `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` | contract | Section 2 is the authority |
| Reference trimming | `models/demos/deepseek_v3_d_p/reference/kimi_k3/modeling_kimi_k3_mla.py` | pattern | Provenance header with upstream line numbers; why each replacement was made |
| Reference purity | `models/demos/deepseek_v3_d_p/reference/kda/README.md` | contract | A reference imports torch only. No ttnn, no device code, no mesh fixtures, no checkpoints |
| PCC test scaffold | `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py` | pattern | Identical random weights both sides, shared rope, `comp_pcc`, no checkpoint or network. The instrument every other PCC number depends on. The donor runs it on one card; bring-up runs every PCC test on the target mesh |
| Config diff test | `models/demos/deepseek_v3_d_p/tests/torch/test_kimi_k3_mla_reference.py` | pattern | Asserts every constant equals the vendored `config.json` |
| KV table test | `models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py` | pattern | Isolates address-table correctness: no model run, no fabric migrate. Needs a galaxy |
| Mesh KV-PCC | `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py` | pattern | Golden trace layout, env knobs, per-layer PCC reporting |
| CPU golden cache | `models/demos/deepseek_v3_d_p/utils/transformer_helpers.py:762` | import | `ReferenceCacheKey` + `check_/save_/load_reference_cache`. Caches e2e snapshots + KV so a CPU forward runs once, not per test. Worked usage: `tests/test_prefill_transformer.py` (check :258, save :361, load :630) |
| Golden trace generation | `models/demos/deepseek_v3_d_p/tt/runners/generate_prompt_trace.py` | pattern | Only if no golden trace is staged for this model yet |

`grep "SHOULD BE SHARED"` here for the hoist backlog (P7).

Two rules the golden cache embodies, worth keeping: key the cache on **every** field that changes the
output (`ReferenceCacheKey` is frozen so a changed field yields a different filename and stale results
are never reused silently), and **assert rather than recompute** where a CPU run is expensive — see
`tests/test_mla.py:293`, which fails loudly on a cache miss in CI instead of burning an hour.
Per-module goldens are cheap to regenerate and are deliberately not cached.

---

## 4. Stages and order

Three ladders, run in order and one at a time: the decoder (D1-D4), then the whole model around it
(M1-M4), then the prefill pipeline (P1-P4). D and M have the same four-step shape — torch golden,
then a mock outline plus PCC tests, then implement with a torch CPU fallback where ttnn cannot yet,
then replace every fallback with ttnn.

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

  Bringup previously defined modules on target device straight away (no need for single chip bringup first). What is not supported in ttnn fallback to pytorch CPU to get working first version.

- **D4 — Implement torch fallbacks in ttnn**

  Each component that is in torch has to be rewriten/implemented in ttnn.


When each stage is finished - goals:
- D1 goal → Have torch implementation for each building block of model Attn,MLP,Norms,Emb.
- D2 goal → Have layer.py outline of one decoder layer in high level blocks so we can write PCC tests for each building block and decoder. 
- D3 goal → Have all building modules of decoder passing PCC tests in isolation (doesn't matter if the impl is ttnn or torch). Have one decoder block passing PCC test.
- D4 goal → Rewrite all torch fallbacks to ttnn (everything is on tt hardware). All blocks rewriten from torch to ttnn in this stage should pass PCC tests. Old ttnn blocks should stil pass and e2e model should pass PCC test.

We switch from one stage to next only when the goal is reached.


Whole model bringup stages:

Same shape as decoder bringup. The decoder is already implemented and PCC-tested from D1-D4, so it
enters as a finished block; only the parts around it are new.

- **M1 — Torch golden impl (whole model)**

  Write what is missing for the whole model pytorch golden reference: embedding, final norm, lm head, and the layer stack itself. No TTNN yet.

- **M2 — Mock prefill pipeline + PCC tests for each new component**

  Mock the prefill pipeline: call decoder N times, embedding, final norm, lm head (like here: models/demos/gpt_oss_d_p/tt/model.py). Nothing needs to be implemented in this stage, just have the outline to generate tests. Generate PCC tests for the components that need to be implemented (decoder already has impl and tests).

- **M3 — Implement the whole-model components**

  Bringup embedding, final norm, lm head and the N-layer stack on target device. What is not supported in ttnn fallback to pytorch CPU to get working first version.

- **M4 — Implement torch fallbacks in ttnn**

  Each component that is in torch has to be rewriten/implemented in ttnn.


When each stage is finished - goals:
- M1 goal → Have torch implementation for every remaining building block: embedding, final norm, lm head, and a full-model forward that matches HF.
- M2 goal → Have model.py outline of the whole model in high level blocks so we can write PCC tests for each new building block and for the e2e model.
- M3 goal → Have all new building blocks passing PCC tests in isolation (doesn't matter if the impl is ttnn or torch). Have the whole model passing e2e PCC test.
- M4 goal → Rewrite all torch fallbacks to ttnn (everything is on tt hardware). All blocks rewriten from torch to ttnn in this stage should pass PCC tests. Decoder blocks from D4 should still pass and e2e model should pass PCC test.

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
- P1 goal → Model runs with real checkpoint weights on target mesh, one-shot. Per-layer KV PCC vs golden trace passes. Random-weight tests from D4/M4 still pass.
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

### D1 — Torch golden impl (decoder)

**Steps**
1. Vendor the model's `config.json`; write the config constants class and the test that diffs every
   constant against it.
2. Get a torch reference for the decoder blocks. Import the HF modeling file directly if it imports
   and constructs standalone; otherwise trim and vendor the classes you need, recording upstream
   line numbers as provenance. The reference imports torch only — no ttnn, no device code.
3. Write the golden runner: run the reference and dump each block's inputs/outputs to disk, keyed on
   everything that changes the result. Reuse `ReferenceCacheKey` + `save_/load_reference_cache`.

**Goal** Torch implementation for each decoder building block (attention, MLP, norms).

**Verified by** Host pytest, no device: golden matches an HF forward, and config constants equal the
vendored `config.json`.

---

### D2 — Mock decoder + PCC tests

**Steps**
1. Write `layer.py` composing the decoder from named blocks; every block is a mock. Nothing is
   implemented in this stage.
2. Fix each block's constructor and `__call__` signature — this is the interface commitment the
   tests are written against.
3. Decide the KV cache layout (§5.2). It is encoded by both the attention read path and the address
   table, so it cannot wait.
4. Write a PCC test per block and one for the whole decoder, driving reference and TT module with
   identical random weights.

**Goal** A `layer.py` outline in high-level blocks, and PCC tests for each building block and for
the decoder.

**Verified by** The suite collects and runs. Tests fail on PCC, not on import or signature errors.

---

### D3 — Implement decoder on target mesh

**Steps**
1. Mesh prerequisite first: target mesh opens, `MeshConfig` and `CCLManager` in place, all-gather +
   all-reduce smoke test passes. No module work before this.
2. Implement in dependency order — norm, MLP/activation, rope, attention (one-shot), KV cache
   allocation + write, MoE, then the composed layer.
3. Where ttnn has no op for something, fall back to torch on CPU to get a first working version.

**Goal** Every decoder building block passes its PCC test in isolation (ttnn or torch), and one
decoder block passes its PCC test.

**Verified by** The D2 suite on the target mesh with random weights.

---

### D4 — Decoder torch fallbacks to ttnn

**Steps**
1. Replace one torch fallback with ttnn at a time. Rewrite each block in torch as mathematical equivalent in ttnn operations. Re-run the edited block PCC check and e2e check after each block rewrite.
2. If you can't map torch implementation to ttnn operations, log this and move on — a new kernel is currently out of scope for bring-up.

**Goal** No torch left in the decoder path.

**Verified by** The D2 suite still passes, and nothing in the decoder path imports a torch fallback.

---

### M1 — Torch golden impl (whole model)

**Steps**
1. Extend the D1 reference with what the decoder did not need: embedding, final norm, lm head, and
   the layer stack.
2. Extend the golden cache to the end-to-end output. A CPU forward of the full model is expensive —
   it should run once, not per test.

**Goal** Torch implementation for every remaining building block, and a full-model forward that
matches HF.

**Verified by** Host pytest: full-model golden matches an HF forward; a second run loads from cache
rather than recomputing.

---

### M2 — Mock prefill pipeline + PCC tests

**Steps**
1. Write the model outline: embedding, decoder x N, final norm, lm head. The decoder slot is the
   real D4 module; everything else is a mock.
2. Fix the signatures of the new components, as in D2.
3. Write a PCC test per new component and one end-to-end model test.

**Goal** A `model.py` outline in high-level blocks, with PCC tests for each new building block and
for the e2e model.

**Verified by** The suite collects and runs; new-component tests fail on PCC, not on wiring.

---

### M3 — Implement whole-model components

**Steps**
1. Implement embedding, final norm, and lm head on the target mesh.
2. Wire the N-layer stack: per-layer weight slicing and naming, per-layer type dispatch if the model
   is hybrid, and the KV cache sized for N layers rather than 1.
3. Torch CPU fallback where ttnn cannot yet.

**Goal** Every new building block passes PCC in isolation (ttnn or torch), and the whole model
passes its e2e PCC test.

**Verified by** The M2 suite on the target mesh with random weights.

---

### M4 — Model torch fallbacks to ttnn

**Steps**
1. Replace remaining torch fallbacks with ttnn, one at a time. Rewrite each block in torch as mathematical equivalent in ttnn operations. Re-run the edited block PCC check and e2e check after each block rewrite.
2. If you can't map torch implementation to ttnn operations, log this and move on — a new kernel is currently out of scope for bring-up.


**Goal** Everything runs on TT hardware.

**Verified by** The M2 suite passes, the D4 decoder tests still pass, and the e2e model PCC test
passes.

---

### P1 — Real weights

**Steps**
1. Write the checkpoint loader: safetensors iteration, prefix filtering, dtype conversion.
2. Add dequantization and any qkv fusion / permutation the TT modules expect.
3. Run the full model with real weights on the target mesh, one-shot (no chunking).

**Goal** The model runs on real checkpoint weights at the target mesh shape.

**Verified by** — port the donor's `tests/galaxy_prefill_kv_pcc.py` and run it one-shot:

```bash
PREFILL_TRACE_DIR=<golden> PREFILL_CHUNKED=0 \
  python3 models/demos/<model>/tests/galaxy_prefill_kv_pcc.py
```

Every layer's K/V must clear the spec's PCC threshold, and the random-weight suites from D4 and M4
must still pass.

---

### P2 — Chunked prefill

**Steps**
1. Wrap the cache-read op. It is a shared ttnn primitive —
   `ttnn.transformer.ring_joint_scaled_dot_product_attention` — which gathers KV across the SP axis
   internally by online softmax, so there is no explicit all-gather. Both donors wrap it the same
   way, in `tt/attention/dense_sp.py`; copy that wrapper. Q is the current chunk, K/V is the
   accumulated prefix `[0:logical_n]` read straight out of the block-cyclic cache.
2. Thread the model's attention features into the op: attention sinks, per-layer sliding window,
   and whatever else the spec lists. A donor without that feature will not show you the argument.
3. Take the persistent ring-gather scratch buffers from the CCL manager
   (`get_ring_gather_buffer`) rather than allocating per call.
4. Implement the runtime: `compile`, `make_chunk_input`, `prefill_chunk`, with assertions on
   `actual_start` / `actual_end` so an out-of-contract chunk fails loudly.

Two constraints worth knowing before you start: the cache must be **bf8** on this path (the sliding
ring path and its gather buffers are bf8, and the donor asserts it), and no cache re-layout is
needed — the ring reads the same block-cyclic layout the write op already produced (§5.1).

Validate the ring building block on its own before wiring it into the model; M3 has
`test_ring_joint_sp_vs_ref.py` and `test_ring_joint_cache_read_sp_vs_ref.py` for exactly that.

**Goal** Multi-chunk prefill produces the same KV as an equal-length one-shot run.

**Verified by** two things, in this order.

1. The ring op in isolation — port M3's pair, which exist for exactly this:

```bash
pytest models/demos/minimax_m3/tests/unit/test_ring_joint_sp_vs_ref.py
pytest models/demos/minimax_m3/tests/unit/test_ring_joint_cache_read_sp_vs_ref.py
```

2. The same P1 script in chunked mode, which must produce the per-layer PCC P1 did:

```bash
PREFILL_TRACE_DIR=<golden> PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=<spec> \
  python3 models/demos/<model>/tests/galaxy_prefill_kv_pcc.py
```

---

### P3 — Serving

**Steps**
1. Write the `KvCaches` subclass and `allocate_kv_cache`.
2. Write the adapter: `load_hf_config`, `weight_cache_path`, `allocate_kv_cache`, `build_runtime`.
   Keep imports lazy — the producers import this module too.
3. Add the `ADAPTER_PATHS` line and the manifest JSON.
4. Add a KV read-back branch for your cache layout in
   `prefill_producer.py::_read_slot_kv_and_check_pcc`. It dispatches on `ADAPTER.name` — today
   `minimax_m3`, `gpt_oss_d_p`, and an MLA fallback — and is deliberately **not**
   adapter-pluggable, so a third layout needs a branch there and its own decode.

**Goal** The engine serves chunks pushed by the producer and the KV it wrote reads back correct. No
inference server and no decode side are needed.

**Verified by** a model-agnostic test that already exists — it spawns the runner and the producer
itself:

```bash
PREFILL_MODEL=<model> PREFILL_TRACE_DIR=<golden> \
  pytest models/demos/common/prefill/tests/test_producer_runner_e2e.py::test_producer_runner_pcc
```

Three scenarios must pass: `single_user_full_depth` (11 x 5120, the deepest correctness gate),
`round_robin_4users` (deterministic interleave), `random_8users` (seeded chaotic interleave with slot
recycling). Look for `[producer] KV cache PCC PASSED`; the threshold is
`PREFILL_STANDALONE_CHUNKED_PCC`, producer default `0.93`.

The manual two-terminal equivalent is Gate 1 in
[`PREFILL_MIGRATION_TESTING.md`](PREFILL_MIGRATION_TESTING.md) — use it to debug, not as the gate.

---

### P4 — Migration

**Steps**
1. Write the KV chunk address table for your cache layout (bank walk, config ordering).
2. Implement `kv_migration_base_address` (or `kv_migration_stages` for several caches) and
   `set_layer_ack_channel`.

**Goal** Another process can locate this model's KV bytes from the published table and copy them.

**Verified by** two things.

1. The address table in isolation — no model run, no fabric migrate. Port the donor's test:

```bash
pytest models/demos/<model>/tests/test_kv_cache_table.py -k smoke
pytest models/demos/<model>/tests/test_kv_cache_table.py -k readback
```

`smoke` checks the multi-config layout, block-cyclic SP positions and the DRAM bank walk;
`readback` checks the bytes match the live device cache after a write. Both also cover the protobuf
round-trip preserving lookups.

2. The real DRAM -> transport -> DRAM copy: Gate 2 (loopback migration) in
[`PREFILL_MIGRATION_TESTING.md`](PREFILL_MIGRATION_TESTING.md). Loopback means
`dest_endpoint_id` is the endpoint's own id, so source and destination slots share one table. This
needs the tt-llm-engine binaries (`migration_endpoint`, `migration_worker`) built against the same
tt-metal — the only gate in the ladder with an external dependency.

---

## 7. Definition of done

- [ ] Every module has a `*_vs_ref` test at or above its `unit_pcc` threshold
- [ ] Full model runs at target mesh shape with real weights; per-layer KV PCC recorded in `README.md`
- [ ] Adapter implements all four abstract methods; registered in `ADAPTER_PATHS`; manifest exists
- [ ] Runtime satisfies `ADDING_A_PREFILL_MODEL.md` §2 and asserts on out-of-contract chunk ranges
- [ ] Two-terminal producer PCC passes
- [ ] `README.md` records architecture, reuse-vs-fresh, PCC status, run commands, and known gaps

Explicitly **not** required to call bring-up done: perf numbers, registration in a CI tier, and
top-1 / logits agreement with HF. KV-cache PCC is a proxy for correctness, not a substitute for that
last one — track it as follow-on work, not as a blocker.
