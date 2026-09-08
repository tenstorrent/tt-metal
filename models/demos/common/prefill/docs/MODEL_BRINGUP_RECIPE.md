<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prefill model bring-up recipe

Path from a HuggingFace checkpoint to a model running chunked prefill on the
`models/demos/common/prefill` engine, validated against a CPU golden trace.

**Scope:** first working implementation — package scaffold, torch oracle, module-by-module PCC
bring-up, full model, KV cache, real weights, chunked prefill. Bring-up **ends at P2**.
**Not in scope:** serving (adapter, manifest, the two-process runner) and KV migration, which are a
separate follow-on; perf tuning, sharding/memory optimization, dtype accuracy trade-offs, new C++
kernels, decode.

> **Status: outline.** Sections marked `TODO` are stubs to be filled in one at a time.

---

## 1. Input to the bring-up

**One file: the prefill spec** (`models/demos/common/prefill/specs/prefill_spec_template.json`).

*What this model is, and what it is graded against.* It carries only what the HF config and the
checkpoint cannot tell you: target hardware, TP/SP split, chunk size and sequence length, user
count, data formats, the two PCC thresholds (`acceptance.pcc_target`, `acceptance.pcc_lower_bound`,
§4), golden trace location. Anything readable from `config.json` — dims, layer count, attention
family, rope parameters, vocab, norm eps, expert counts, checkpoint quantization — is **not** in the
spec; read it from the config and assert it (see D1).

The **spec is binding**. Every value in it must be respected exactly, at every stage; no convention
in this document overrides it. Where an existing implementation in the repo conflicts with the spec,
the spec wins and that implementation was the wrong thing to borrow from.

Everything else — which existing code to learn from for each part — the agent finds itself (§2).

---

## 2. Exploration — finding what to borrow

Most of a prefill package is plumbing every model needs: sharding, collectives, weight caching, KV
layout, chunking. It is written and debugged several times over in this repo. Before writing a part,
find the existing code that already solved it — but find it, do not be handed it: a pointer with no
shape attached is what makes a mismatch invisible until PCC drops.

### 2.1 Gate the candidates first

Enumerate the packages that implement prefill on the `common/prefill` engine — `ADAPTER_PATHS` in
`models/demos/common/prefill/adapter.py` is the registry, and a package not in it has never run on
this engine — then keep only those that have actually run on the spec's **target hardware and
mesh**. A package that never ran on the
target teaches nothing reliable about sharding, collectives or L1 budgets — its program configs are
tuned for a different machine.

Rank what survives by, in order: attention family (MLA / GQA / sparse — this decides the cache
count), MLP density (dense / MoE), checkpoint quantization. Record the ranking and the reason in
`README.md`; if the best match for a part is a package that is *not* on the target hardware, treat it
as a source for the **math only**, never for a copy.

### 2.2 What to look for, per part

One row, one search. The reject column matters as much as the first: it is what stops a plausible
but wrong source being adopted.

| Part | Look for | Reject if |
|---|---|---|
| Weight loading | safetensors walk, prefix filtering, the single dtype exit | the quantization scheme differs from the spec's |
| Dequant | the spec's exact scheme (per-tensor / blockwise / packed) | it differs — write the scale application fresh, reuse only the sorted key walk and the fail-loud on a missing scale |
| Norm / embedding | a package that ran at this spec's `hidden` and tokens-per-chunk | it never ran at this width — expect to compose the op from primitives (§2.3) |
| MLP | the column/row-parallel split and where the CCL lands | the activation variant differs — port the structure, never the math |
| Attention | head split, projection sharding, the output-proj CCL tail. **Not** the attention math | the cache count differs (MLA vs GQA vs sparse) |
| RoPE | variant plumbing and the whole-cache indexed build | the source was written against a different `transformers` major — re-read the HF branch the model actually takes |
| KV cache | **one** package for all five coupled roles: 32-token bank walk, slot packing, block-cyclic SP sharding | any one role is sourced from a different package |
| Runtime | `compile` / `make_chunk_input` / `prefill_chunk` and its chunk-range assertions | — |

Shared code is **imported, never copied**: the MoE substrate
(`models/demos/deepseek_v3_d_p/tt/moe/`), the migration helpers
(`models/demos/common/prefill/runners/migration.py` — `get_num_dram_banks` sizes the KV allocator),
and the golden-cache helpers (§3).

### 2.3 Structural vs shape-tuned

A borrowed file holds two kinds of thing, and they transfer differently.

**Structural** — sharding, CCL placement, cache layout, slot packing, lifetime and deallocation
discipline. Load-bearing: some of it looks arbitrary and is not, because the KV layout is read by the
ring SDPA op and collectives must go through the CCL manager's semaphore ping-pong. Copy it, then
verify it mechanically (grep the copied files for the layout invariants in §5 and assert they agree).
Deviating here does not raise an error — it degrades PCC or corrupts output nondeterministically.

**Shape-tuned** — compute configs, dtype casts, program-config constants, and op choices that fit
inside the source's L1 budget. These encode the *source's* hidden size, head_dim and chunk size.
**Re-derive them for this model's dimensions.** Copying a tuned value to a wider model is not the
conservative choice; it silently costs PCC.

The bring-up default for every projection matmul and the plain SDPA is **HiFi4 with
`fp32_dest_acc_en=True`**. A narrower setting is a measurement, not an inheritance — record what it
costs. Where such a constraint is real it is also local: `fp32_dest_acc_en=False` belongs to the ring
cache-read op, not to SDPA in general, and a source that sets it globally is over-broad.

### 2.4 Record the envelope

Exploration is stage **E** and it is logged: one `source` record per part, naming what was chosen and
the shape it was measured at (§7.2). That record is the stage's deliverable — it is what makes a
later `reference_gap` traceable to the choice that caused it, and it is the only reason a wrong
choice is visible at all.

For every part, write down in `README.md` where it came from and the `(hidden, head_dim, chunk,
sp × tp)` that source was **measured** at. Outside that envelope, expect to compose the op from
primitives or to re-measure it — and where a borrowed module has a known ceiling, pin it with a test
at the boundary so the next bring-up inherits the knowledge instead of the bug.

---

## 3. Shared helpers and test patterns

Genuinely model-independent, and the only pointers this document hands out. Everything else is found
by exploration (§2), with its envelope recorded.

| What | Pointer | Mode |
|---|---|---|
| MoE substrate | `models/demos/deepseek_v3_d_p/tt/moe/` | **import** |
| DRAM bank count | `models/demos/common/prefill/runners/migration.py` | **import** |
| CPU golden cache | `models/demos/deepseek_v3_d_p/utils/transformer_helpers.py:762` | **import** |
| Reference purity | `models/demos/deepseek_v3_d_p/reference/kda/README.md` | contract |

Test *patterns* are named per stage in each Testing table below (§6) — read them for structure, not
for content.

Two rules the golden cache embodies, worth keeping: key the cache on **every** field that changes the
output (`ReferenceCacheKey` is frozen so a changed field yields a different filename and stale results
are never reused silently), and **assert rather than recompute** where a CPU run is expensive — see
`tests/test_mla.py:293`, which fails loudly on a cache miss in CI instead of burning an hour.
Per-module goldens are cheap to regenerate and are deliberately not cached.

---

## 4. Stages and order

**Step 0 — resolve the weights, before anything else.** Do this first, so a missing checkpoint is
known on minute one rather than discovered at P1. In order:

1. `PREFILL_HF_MODEL` / `HF_MODEL`, if set — an explicit override always wins.
2. **`/mnt/models/<hf-org>/<Model-Name>`** — the shared store (NFS, org-named dirs, TTNN caches as
   siblings). Look here first; several models are already staged and cost nothing to reuse. A second
   copy of some models sits in the shared hub cache `/mnt/models/huggingface/hub`, which is
   read-only — prefer the flat org dir.
3. Otherwise **download it** to `/mnt/models/<hf-org>/<Model-Name>` (needs an HF token:
   `huggingface-cli login` or `HF_TOKEN`). Creating a new org dir there needs `sudo install -d -m 2775
   -g 50000 /mnt/models/<hf-org>`.
4. If none of that works, **bring up on synthetic weights — but say so, loudly.** A
   format-identical random checkpoint (same on-disk layout: shard names + index, key prefixes,
   quantization and scale tensors) fully validates the loader, the dequantizer, the key mapping and
   the golden pipeline. It says **nothing** about the model's accuracy, and a PCC table produced from
   it is not a result. So it is a legitimate path, never a silent one:
   - log it — a `source` record with `part: "weights"`, `chosen` the path, and `envelope`
     `"SYNTHETIC - format-identical, random values"`;
   - state it in the first paragraph of `README.md`, not in a footnote;
   - label every number it produced as synthetic wherever that number appears;
   - treat the missing full-depth real-weights e2e PCC as a **blocker to log**, not as done.

   Mistral-Medium-3.5 went synthetic because its checkpoint was unreachable, and the whole accuracy
   table it produced turned out to describe nothing. That is the failure this step exists to prevent.

**The golden trace lives beside the weights.** The trace is the reference forward run **once** and
saved, so no test reruns a CPU model: a dir holding `metadata.json` (`{"token_ids": [...]}`, the exact
input tokens) plus `kv_cache/layer_N.safetensors` (the reference KV per layer). P1/P2 feed those same
token ids to the device and PCC its KV against those tensors — it is the graded artifact.

Convention on the shared store is `<weights_dir>/golden/<prompt>_<isl>/`, e.g.
`/mnt/models/MiniMaxAI/MiniMax-M3-ref/golden/longbook_5120/`, with siblings for other lengths
(`longbook_10240`, `longbook_56320`). Point `PREFILL_TRACE_DIR` at one. Look there before generating:
a trace is expensive, and per-model dirs like `/mnt/models/deepseek-prefill-cache/golden/` and
`/mnt/models/kimi-prefill-cache/golden/` hold more.

Three properties worth knowing before you rely on one:

* **It needs real weights.** The generator refuses to run without a safetensors checkpoint, so the
  trace is strictly downstream of step 0 — synthetic weights give a synthetic trace, and a PCC
  against it proves only that two random-valued pipelines agree.
* **It is per (model, prompt, ISL, depth)**, not per model. A different sequence length or layer
  count needs a different trace. That is why the dirs are named for prompt and token count.
* **Do not confuse it with a ttnn trace.** `use_trace` / `trace_region_size` capture the per-chunk
  forward as a device command buffer for replay — a perf mechanism with no goldens in it.

The per-module goldens of D1/M1 are the same compute-once idea at block granularity, keyed on every
field that changes the output (§3, `ReferenceCacheKey`) — the flat `.pt` files under
`/mnt/models/kimi-prefill-cache/golden/` are that family, not traces.

Exploration (**E**, §2) comes first and once: gate the candidate packages, rank them, and record a
`source` per part. Then three ladders, run in order and one at a time: the decoder (D1-D3), then the
whole model around it (M1-M3), then the prefill pipeline (P1-P2). D and M have the same three-step shape — torch golden,
then a mock outline plus PCC tests, then implement in ttnn, with a torch CPU fallback only where
ttnn genuinely cannot.

**Prerequisite for D3 — mesh up.** Bring-up goes straight to the target mesh; there is no
single-card step at any point. First check the runtime env —
`models/demos/common/prefill/tools/check_runtime_env.sh`, which fails with the exports to fix it. A
stale `TT_METAL_RUNTIME_ROOT` is silent: `ttnn` auto-detects the root for an editable install only
when that var is *unset*, and the device-less UMD reader resolves it *before* `TT_METAL_HOME`, so a
dangling root beats a correct `HOME`. Machine provisioning under `/etc/profile.d` is a common source.
It also checks the **interpreter**, which is the other half of the same trap: run everything through
the project venv, and never read a bare `import ttnn` as proof the env is good — the repo has a
`ttnn/` directory, so from the repo root *any* python imports it as an empty namespace package while
`torch` and `transformers` fail, which looks like a half-broken install rather than the wrong python.

**If that check fails, build the env before going further — do not work around it.** Setting up the
venv is not this recipe's job and the commands differ per machine: load the **`build-metal` skill**
if the box has one (it owns build + venv + the per-machine gotchas), otherwise follow the repo's own
`INSTALLING.md` §"Virtual Environment Setup" / `create_venv.sh`. Two traps that recur regardless:
`ttnn` must be *installed* into the venv (editable) rather than reached via `PYTHONPATH`, and a venv
is often container-local, so nothing carries over to a new machine and it has to be redone there.
Then, before the first module is written: the target mesh opens,
`MeshConfig` and `CCLManager` are in place, and an all-gather + all-reduce smoke test passes.

Every module PCC test therefore exercises sharding and collectives from the first one — more setup
cost, and the "worked on one card, broke on the mesh" class of bug disappears entirely.

**Fabric topology: run on whatever galaxy you get.** The mesh-graph descriptor is chosen **before
the cluster initialises** (set it in the package's `conftest.py`), and a torus descriptor cannot map
on a pod without wrap-around links — so this is settled at mesh-up, not later. Default to the plain
mesh descriptor with `FABRIC_1D` + `Topology.Linear`, which maps on **any** galaxy, torus-wired or
not. Put the torus behind one env knob: it is the one significant perf lever bring-up has, so take
it where the pod offers it, but it is never a correctness gate and a bring-up must not fail or block
for the want of it. If the torus descriptor will not map, log it as `env`, fall back to linear, and
carry on. Run the smoke test in whichever topology the pod supports, and both where both work.

Record which topology each measurement was taken on, in `README.md` — collective cost differs enough
that linear and torus numbers are not comparable. It stays out of the spec: the spec fixes what the
*model* is, and the wiring is a property of the pod you happened to get.

All PCC tests up to P1 run on **random weights**, identical on both sides. Real checkpoint loading
is not a dependency of any module test and is deferred to P1.

**Reduced runs are diagnostics, never the result.** Cutting depth (`PREFILL_NUM_LAYERS`), width or
vocab to fit a host-side comparison is legitimate and often necessary — a host cannot materialise a
100B-parameter model as random weights on both sides. But a reduced run is a debugging aid, not a
grade: the model must be run **end to end at its full layer count and full width with real weights**,
with the e2e PCC measured against a full-depth golden trace, whatever the partial runs said. Any
reduced run is logged (§7), recorded in `README.md`, and labelled as reduced wherever its numbers
appear — an unlabelled PCC table reads as a full-model result. If the full-depth
e2e number cannot be produced, that is a **blocker to log**, not something a reduced run substitutes
for.

All references and goldens are **fp16** (`torch.float16`), regardless of the checkpoint dtype and
of the ttnn dtypes under test: the D1/M1 torch references compute in fp16 (input, weights, cos/sin),
the per-module goldens they dump are fp16, and the golden trace P1-P2 compare against is written to
disk as fp16. This is a fixed convention, not a per-model choice. The package you borrow a reference
or a golden runner from might **not** follow it — its casts are shape-tuned, not structural (§2.3),
so replace them rather than carrying them over.

**Two PCC numbers, from the spec, for every component.** The spec's `acceptance` block carries
`pcc_target` (0.99) and `pcc_lower_bound` (0.85). They apply unchanged to every test in every
Testing table below — each `*_vs_ref` module test, the decoder layer, the whole model, and the
per-layer KV check.

- `pcc_target` is what every component **aims for**. A component at or above it is finished; move
  on.
- `pcc_lower_bound` is the **assert** in every test. A component below it is not accepted, the
  stage's Testing table is red, and the work is not done, try to fix before anything else.
- **Between the two** the test passes, accept it and record the measured value and the reason
in the `README.md` PCC status table (§8).

- **E — Exploration**

  Per §2: gate candidates on the spec's target hardware, rank them, and search per part. No code yet.

  E goal → every part in §2.2 has a logged `source`: what was chosen, the envelope it was measured at, and what was rejected. Parts with no usable source are named as write-fresh.

Decoder bringup stages:

- **D1 — Torch golden impl**

  No TTNN yet.

- **D2 — Mock decoder + PCC tests for each component**

  Write mock decoder layer using mock building blocks that should be implemented later (like here: models/demos/gpt_oss_d_p/tt/layer.py). Write PCC tests for each module and whole decoder.

- **D3 — Implement 1 Decoder as composition of big blocks (Attn, MLP, Emb,...)**

  Bringup previously defined modules on target device straight away (no need for single chip bringup first). Where no single ttnn op matches the block, write the mathematical equivalent out of the ttnn ops that do exist — there is no `ttnn.mlp`, and an MLP is a matmul, an activation and a second matmul, so compose it rather than dropping to CPU. Fall back to pytorch CPU only when the math cannot be expressed in ttnn at all; if the fallback is inevitable, take it, log it as a `fallback` (§7) and move on.


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

  Bringup embedding, final norm, lm head and the N-layer stack on target device. Same rule as D3: where no single ttnn op matches the block, write the mathematical equivalent out of the ttnn ops that do exist before considering a fallback. Fall back to pytorch CPU only when the math cannot be expressed in ttnn at all; if the fallback is inevitable, take it, log it as a `fallback` (§7) and move on.


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

The decoder and the model now run; what is left is real weights and chunking.

- **P1 — Real weights**

  Load the actual checkpoint: safetensors iteration, dequant, qkv fusion/permutation. First run with real weights on the target mesh, one-shot (no chunking).

- **P2 — Chunked prefill**

  KV cache read-back path (ring SDPA over the block-cyclic cache) and the runtime that drives it: compile, make_chunk_input, prefill_chunk with actual_start/actual_end.


When each stage is finished - goals:
- P1 goal → Model runs with real checkpoint weights on target mesh, one-shot. Per-layer KV PCC vs golden trace passes. Random-weight tests from D3/M3 still pass.
- P2 goal → Multi-chunk prefill produces the same KV as an equal-length one-shot run. Chunked per-layer KV PCC vs golden passes.

We switch from one stage to next only when the goal is reached.


Where the KV cache lands: the **layout is decided in D2** (the attention read path encodes it),
**allocation + write** are implemented in D3 (the layer writes K/V even one-shot), and the
**cache read** arrives in P2. GPT-OSS shipped it the same way — KV cache at
P2 of its stack, chunked ring SDPA at P6.

## 5. KV cache allocation (the critical decision)

The highest-leverage decision in the build: bound simultaneously to the attention op, the SP/TP
split, DRAM bank geometry, and the migration address walk.

### 5.1 Do not invent a layout

The layout is already **canonical by convention** across the prefill packages, because the chunked
ring SDPA reads it. `gpt_oss_d_p/tt/attention/kv_cache.py`
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

**Run every device test through `scripts/run_safe_pytest.sh`.** It `flock`s device access so
several agents can share one pod, sets `TT_METAL_OPERATION_TIMEOUT_SECONDS` for hang detection at
the dispatch layer, and resets the device after a hang so the next run starts clean. Calling `pytest`
directly on a shared pod is what turns a second job into an apparent hang: it blocks on the chip lock
and logs nothing but a lock warning. Host-only tests need no wrapper.

**A stage is complete when, and only when, every test in its Testing table passes** and the
stage's log lines are written (§7). No stage is entered before the previous stage's table is green.

---

### D1 — Torch golden impl (decoder)

**Steps**
1. Vendor the model's `config.json`; write the config constants class.
2. Get a torch reference for the decoder blocks. Import the HF modeling file directly if it imports
   and constructs standalone; otherwise trim and vendor the classes you need, recording upstream
   line numbers as provenance. The reference imports torch only — no ttnn, no device code, and
   computes in fp16 (§4).
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
   is inevitable, take it, log it as a `fallback` (§7), and move on — a new kernel is out of scope for bring-up.

**Testing** — the decoder test suite written in D2. No new tests.

**Goal** D3 passes when every applicable test in the decoder test suite passes at or above
`pcc_lower_bound`, every component below `pcc_target` has its measured value and reason recorded
(§4), each block is in ttnn wherever its math can be composed from ttnn ops, and a torch CPU
fallback is logged only where it cannot.

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
| No pattern in the repo — author it | That the golden cache round-trips: a second run loads from disk instead of recomputing, and a changed `ReferenceCacheKey` field forces a miss rather than silently reusing a stale result. |

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
   cannot be expressed in ttnn at all — log it as a `fallback` (§7) and move on if it is inevitable.

**Testing** — the model test suite written in M2, plus the decoder test suite as a regression.

**Goal** M3 passes when every test in the model test suite passes at or above `pcc_lower_bound`,
every new component below `pcc_target` has its measured value and reason recorded (§4), each new
block is in ttnn wherever its math can be composed from ttnn ops, a torch CPU fallback is logged
only where it cannot, and the decoder test suite is still green.

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

**Goal** P1 passes when the loader test passes and the one-shot per-layer KV PCC is at or above
`pcc_lower_bound` on every layer — aiming for `pcc_target`, with the minimum recorded in `README.md`
— and the D3 and M3 random-weight tables are still green.

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

**Goal** P2 passes when the chunked run is at or above `pcc_lower_bound` on every layer and reaches
the same per-layer PCC as P1's one-shot run.

---

## 7. Logging

The bring-up writes an append-only log so the *process* can be measured, not just the result: which
stages cost the most reiterations, which test in a Testing table is the real bottleneck, and where
the agent had to leave the script.

**File:** `models/demos/<model>/bringup_log.jsonl` — one JSON object per line, **appended, never
rewritten**. No line is edited or deleted after it is written, including lines that record a wrong
turn. The log is committed with the model.

### 7.1 When to write

Five triggers, all mechanical — there is no "worth logging?" judgment to make:

| Trigger | Event |
|---|---|
| Entering a stage | `enter` |
| Every run of the stage's Testing table, pass or fail | `verify` |
| Going off-script: something the recipe, the spec, or the code you borrowed from did not cover, **and you solved it** | `judgment` |
| Falling back to torch CPU for a block (D3/M3 step 4) | `fallback` |
| Dropping a Testing-table row the model does not have | `skip` |
| Choosing (or rejecting) an existing implementation to borrow from, in E | `source` |

**Log successful resolutions only.** Unsuccessful attempts are not logged — the `verify` failures
already record that the stage was fighting back, and how many times.

### 7.2 Records

`t` is ISO-8601 UTC to the minute. `stage` is one of `E D1 D2 D3 M1 M2 M3 P1 P2`.

| Event | Fields | Notes |
|---|---|---|
| `start` | `model`, `mesh`, `recipe_sha` | Once, first line. `recipe_sha` is the git sha of this file, so a run is attributable to the version of the recipe it followed. |
| `enter` | `t`, `stage` | Written before any work in the stage. Its timestamp is the clock start. |
| `verify` | `t`, `stage`, `result`: `pass`\|`fail`, `failed`: `[]` | `failed` holds pytest node ids and must be non-empty when `result` is `fail`. |
| `judgment` | `t`, `stage`, `kind`, `issue`, `fix`, `failed`: `[]` | `failed` is the tests that were failing, if any — a judgment call need not have started as a test failure. |
| `fallback` | `t`, `stage`, `block`, `why` | `block` is the module path that went to CPU, e.g. `rope.indexed_cache`. |
| `skip` | `t`, `stage`, `row`, `why` | `row` is the Testing-table reference being dropped. |
| `source` | `t`, `stage`, `part`, `chosen`, `envelope`, `rejected`: `[]` | One per part in §2.2. `chosen` is the file or package (or `""` for write-fresh); `envelope` is the `(hidden, head_dim, chunk, sp × tp)` it was measured at; `rejected` names candidates passed over, so a later `reference_gap` shows whether a better one was already on the table. |

```jsonc
{"ev":"start","model":"minimax_m3","mesh":"8x4","recipe_sha":"883d2d9"}
{"t":"2026-09-02T09:00Z","ev":"enter","stage":"D3"}
{"t":"2026-09-02T12:00Z","ev":"verify","stage":"D3","result":"fail",
 "failed":["tests/unit/test_ring_joint_cache_read_sp_vs_ref.py::test_sp8"]}
{"t":"2026-09-02T15:00Z","ev":"judgment","stage":"D3","kind":"reference_gap",
 "issue":"borrowed program config assumed q_chunk 256, PCC stalled at 0.91 at sp=8",
 "fix":"set q_chunk_size to seq_local per SP row (128)",
 "failed":["tests/unit/test_ring_joint_cache_read_sp_vs_ref.py::test_sp8"]}
{"t":"2026-09-02T17:00Z","ev":"verify","stage":"D3","result":"pass","failed":[]}
{"t":"2026-09-02T17:05Z","ev":"fallback","stage":"D3","block":"rope.indexed_cache",
 "why":"no ttnn gather on tile layout; built the whole-cache index on host once"}
{"t":"2026-09-02T08:20Z","ev":"source","stage":"E","part":"kv_cache",
 "chosen":"models/demos/gpt_oss_d_p/tt/attention/","envelope":"hidden 2880, head_dim 64, chunk 1024, sp4xtp8",
 "rejected":["minimax_m3: 3 caches (index_k), wrong count for GQA"]}
```

Wrapped above for reading only — in the file each object is a single line.

### 7.3 `kind` — a closed vocabulary

Free text does not aggregate, so every `judgment` is filed under exactly one of these. Pick the
cause, not the symptom.

| `kind` | Means | Reading it |
|---|---|---|
| `spec_gap` | A value the work needed was absent or ambiguous in the prefill spec. | The spec template (§1) is missing a field. |
| `recipe_gap` | This document was silent, ambiguous, or wrong. | Fix the doc. |
| `reference_gap` | Code borrowed from another package did not transfer: it conflicted with the spec, or it did not hold at this model's shape. | Exploration (§2) found the wrong source, or the envelope (§2.4) was not checked. |
| `ttnn_gap` | No ttnn op existed; the math had to be composed, or could not be. | Op-coverage backlog. Pairs with a `fallback` when it could not be. |
| `model_quirk` | The model itself does something the reference implementations do not. | Genuine per-model cost, not a process defect. |
| `env` | Build, mesh bring-up, fabric, or tooling. | Infra, not bring-up. |

### 7.4 Length cap

`issue`, `fix`, and `why` are each **one sentence, on one line, at most 160 characters**. The cap is
enforced, not advisory — `bringup_digest.py --lint` fails a longer field.

It exists because these fields are read side by side across a dozen bring-ups: entries have to be
comparable at a glance, and a field that can hold a paragraph will hold a paragraph. If 160
characters cannot carry the point, the detail belongs in the module's docstring or `README.md` —
`fix` says *what changed*, and the code says why in full.

### 7.5 What is never logged

Derived by the digest, so the agent does not track it:

- **Reiterations** — the count of failing `verify` events in a stage. A Testing table that goes
  green on its first run scores **0**. This is the headline number: it is what separates a stage
  that is understood from one that is being guessed at.
- **Elapsed time** — `enter` to first passing `verify`.
- **Bottleneck test** — the node id appearing most often across all `failed` lists in a stage.
- **Exploration cost** — E's elapsed time, and how many `source` records it produced. A long E with
  few sources is searching, not deciding.
- **Source miss rate** — `reference_gap` judgments against the number of `source` records. This is
  the number that says whether exploration is picking well; nothing measured it before, because the
  choice used to be made outside the bring-up.

### 7.6 Reading the log

```
models/demos/common/prefill/tools/bringup_digest.py            # all models
models/demos/common/prefill/tools/bringup_digest.py --lint     # malformed records only
```

```
stage  models  green  reiters  worst  hours  top failing test
-------------------------------------------------------------
E           4      4      0.0      0    0.9  -
D1          4      4      0.2      1    1.8  -
D2          4      4      0.5      2    3.6  test_ep_moe_vs_ref.py::test_ep8 (2)
D3          4      3      9.8     17    7.8  test_ring_joint_cache_read_sp_vs_ref.py::test_sp8 (14)
```

A stage with a high `reiters` average across models is a stage this recipe does not yet explain well
enough — that is the point of the log. A high `spec_gap` or `recipe_gap` count is the same signal
aimed at the inputs instead of the stages.

---

## 8. Definition of done in terms of this model bringup

- [ ] Every module has a `*_vs_ref` test asserting at the spec's `pcc_lower_bound`; every module below
  `pcc_target` has its measured PCC and the reason in the `README.md` PCC status table
- [ ] Weights resolved per §4 step 0, and `README.md` says whether they are real or synthetic
- [ ] Full model runs at target mesh shape with real weights, at **full depth and full width**; per-layer KV PCC and the e2e PCC recorded in `README.md`, each labelled with the fabric topology it was measured on
- [ ] Any reduced run is recorded in `README.md`, logged, and labelled as reduced wherever quoted
- [ ] Mesh smoke test passes on the pod's wiring (linear always; torus too where the pod offers it)
- [ ] Runtime asserts on out-of-contract chunk ranges
- [ ] `README.md` records architecture, reuse-vs-fresh, PCC status, run commands, and known gaps
- [ ] `bringup_log.jsonl` is committed and `bringup_digest.py --lint` is clean

Explicitly **not** required to call bring-up done: serving and migration (a separate follow-on),
perf numbers, registration in a CI tier, and top-1 / logits agreement with HF. KV-cache PCC is a proxy for correctness, not a substitute for that
last one — track it as follow-on work, not as a blocker.
