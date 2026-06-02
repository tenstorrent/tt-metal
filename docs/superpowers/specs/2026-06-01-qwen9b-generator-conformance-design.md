# Qwen3.5‑9B — Generator‑compatible interface + framework reuse (design)

**Date:** 2026-06-01
**Model:** `models/demos/blackhole/qwen3_5_9b` (branch `qwen9b-p150`, Blackhole P150, single device)
**Status:** design approved; pending implementation plan

> Local‑only: nothing in this effort is committed or pushed unless explicitly requested.

## 1. Goal & scope

Finish the two deferred conformance items from the structural reorg: a **Generator‑compatible interface** and **framework‑building‑block reuse**, so Qwen3.5‑9B is driven by the stock `tt_transformers.Generator` like the other models, and its custom surface shrinks to only the genuinely Qwen‑specific (hybrid Gated‑DeltaNet) parts.

**In scope** (one coherent spec, **interface first**, then reuse):
- The model implements the `Generator` contract methods natively for the parts Generator owns (decode + short prefill).
- The vLLM wrapper becomes a thin `Generator` subclass, kept **local** in the qwen dir.
- The demo is driven by `Generator` (decode + short prefill) plus a shared dispatch for long prefill.
- High‑value building‑block reuse: MLP, LM head, embedding, RMSNorm, and (optional) RoPE for the attention layers.

**Out of scope (deferred, documented as a framework gap — "Tier‑B"):** paged/evictable Gated‑DeltaNet (GDN) recurrent‑state management for vLLM continuous batching. `tt_transformers` has no recurrent/SSM state concept (no precedent for Mamba/linear‑attention state), so this is net‑new framework infrastructure, not a port.

## 2. Decisions (locked)

| # | Decision | Choice |
|---|---|---|
| D1 | Scope | Interface **+** high‑value reuse, **interface first** |
| D2 | Acceptance bar | Validate **vs the HuggingFace reference, keep quality** (end‑to‑end PCC stays in the validated >0.98 band; 61‑test suite + e2e demo pass). Not required to be bit‑identical to current TT ops. |
| D3 | Definition of done | Stock `Generator` drives Qwen9b **decode + short prefill**; **long‑ISL prefill keeps working** via the model's chunk‑outer trace; vLLM wrapper is a thin `Generator` subclass. Tier‑B deferred. |
| D4 | Long‑prefill seam | **Option (b):** one dispatching `prefill_forward` entry point hides the short/long seam from callers. |
| D5 | Wrapper location | **Local** in the qwen dir (`tt/qwen35_vllm.py`); `tt_transformers` imported but **unmodified**. |
| D6 | Interface structuring | **Hybrid:** native contract methods for decode + short prefill; **keep** chunk‑outer long‑prefill model‑owned; **delete** the model's decode‑trace methods *only after* the Generator decode path is proven. |
| D7 | KV / state ownership | **Both model‑bound.** GDN state is *forced* model‑bound (no `kv_cache` representation for recurrent state). Attention KV is *chosen* model‑bound for consistency + minimal diff; safe because per‑request routing is by `page_table` (passed per call), not cache identity. The `kv_cache` param is accepted but vestigial. |
| D8 | RoPE threading | Carry Qwen's `(cos, sin)` through the contract's rot slot rather than refit to `rot_mat_idxs`. Both ends are ours; no framework change. |

## 3. Architecture & layering

Three layers; only the bottom two live in the qwen dir. `tt_transformers` is imported, never modified.

```
 tt_transformers (imported only)
   Generator  (UNMODIFIED)
     • decode_forward / _easy_trace / _capture_trace
     • prefill_forward_text  (single-trace, short only)
     • generate(...)   ← demo decode loop
     calls the contract on each model[i]
        ▲ subclasses                         │ calls contract
 qwen dir (tt/qwen35_vllm.py)                │
   Qwen35ForCausalLM(Generator)             │
     • initialize_vllm_model                 │
     • prefill_forward → DISPATCH:           │
         short(≤2048) → super().prefill_forward_text
         long (>2048) → model.prefill_traced_chunked / prefill_paged
     • decode_forward → super().decode_forward
     • allocate_kv_cache → model.allocate_kv_caches
     • warmup_model_prefill / warmup_model_decode
        │                                    │
 qwen dir (tt/model.py)                      ▼
   Qwen35Model  (speaks the contract natively)
     NEW contract surface (§4)
     KEPT model-owned (long ISL): capture_prefill_trace_chunked,
       prefill_traced_chunked, _forward_prefill_chunk(_eager), prefill_paged
     DELETED after P2 proven: capture_decode_trace_paged, decode_traced_paged
     state: paged KV (8 attn) + external GDN recurrent/conv (24)  — both model-bound
```

- The wrapper wraps the model as 1‑element DP lists `[model]` / `[args]` (`data_parallel=1` on single‑device P150).
- The prefill seam (D4) lives entirely in the wrapper's `prefill_forward`. The demo and wrapper share one qwen‑dir helper so the seam is defined once.
- Decode rides entirely on stock `Generator`.

## 4. Contract‑method mapping

Text‑generation subset only (no vision methods). Each method is defined **on `Qwen35Model`** and **called by `Generator`**.

| Contract method | Returns | Built from |
|---|---|---|
| `prepare_decode_inputs_host(tokens, current_pos, page_table)` | host ttnn: `(tokens_u32, cos, sin, cur_pos_i32[B], page_table_i32)` | host‑prep half of `decode_paged` (`model.py:1074‑1087`) + `rope.get_cos_sin_host` |
| `prepare_inputs_decode(...)` | same, on device | `prepare_decode_inputs_host` + `tt_transformers.tt.common.copy_host_to_device` |
| `ttnn_decode_forward(tokens, current_pos, rot_mats, page_table=None, kv_cache=None, sampling_on_device=False, ...)` | `(logits, None)` | **rename of `_forward_decode`** (`model.py:326`) — already device‑only/trace‑safe; norm+lm_head inside |
| `prepare_inputs_prefill(tokens, start_pos=0, page_table=None, ..., trace_enabled=False)` | tokens + `(cos,sin)` + page tables | short‑prefill prep in `prefill_paged` T≤1024 branch (`model.py:1010‑1015`) |
| `ttnn_prefill_forward(x, rot_mats_*, user_id=0, page_table=None, get_last_token=-1, kv_cache=None, ...)` | logits | short‑prefill body in `prefill_paged` (`model.py:1017‑1022`) |
| `process_output_prefill(tt_out, last_token_idx)` | torch `[vocab]` | `ttnn.to_torch` + last‑token slice (today in demo) |
| `process_output_decode(tt_out, B, S=1, is_tokens=False, is_log_probs=False)` | torch `[B,S,vocab]` | `ttnn.to_torch` (today in demo) |
| `switch_mode(mode)` | — | no‑op (no prefetcher) |

**Attributes to add** (cheap): `self.mesh_device` (alias of `self.device`), `self.configuration` (→ `self.args`, exposes `.max_seq_len`), and the optional‑sampling trio `sampling=None`, `sampling_dp=1`, `_supports_on_device_sampling=False` (so Generator takes the host‑sampling branch).

**`ttnn_prefill_forward` handles a single chunk only** (≤2048). Multi‑chunk long prefill is *not* a contract method — it stays model‑owned (§5), because the GDN multi‑chunk recurrence is orchestration Generator's prefill loop does not perform.

**Deliberate divergences (documented):** D7 (model‑bound KV/state; `kv_cache` param vestigial) and D8 (carry `(cos,sin)`).

## 5. Prefill dispatch & trace ownership

Threshold is one GDN chunk (2048 tokens) — the hard GDN ≤16‑sub‑chunk correctness boundary.

The dispatch decision lives in **one shared qwen‑dir helper** so the seam is defined once. It takes the `generator` and `model` handles (it cannot use `super()`, since the demo calls it without a wrapper). The wrapper's `prefill_forward` calls it with `self` as the generator; the demo calls it with its `Generator` instance.

```
qwen_prefill_dispatch(generator, model, tokens, page_table, prompt_lens, use_trace, **kw):
    T = prompt length
    if T <= 2048:                                  # single chunk; GDN correct; small trace
        return normalize(generator.prefill_forward_text(...))  # Generator owns short prefill + its trace
    # T > 2048 → model-owned (both modes share the GDN multi-chunk recurrence):
    if use_trace:
        return normalize(model.prefill_traced_chunked(padded_tokens, page_table, actual_len=T))
    return normalize(model.prefill_paged(tokens, page_table))   # existing non-traced chunked path

# wrapper:  def prefill_forward(self, *a, **k):  return qwen_prefill_dispatch(self, self.model[0], *a, **k)
# demo:     out = qwen_prefill_dispatch(generator, model, tokens, page_table, ..., use_trace=use_trace)
```

`normalize(...)` reshapes either branch's output to vLLM's expected `(logits, rope_deltas)` return (the demo ignores `rope_deltas`).

### Trace ownership

| Path | Owner | Mechanism |
|---|---|---|
| Decode | **Generator** (new) | `decode_forward(enable_trace=True)` → `_easy_trace` → `ttnn_decode_forward` inside capture; `prepare_inputs_decode` refreshes inputs between replays |
| Short prefill (≤2048) | **Generator** | `prefill_forward_text` → `_easy_trace_prefill`, lazily keyed by seq‑len |
| Long prefill (>2048) | **Model** (kept) | `capture_prefill_trace_chunked` (warmup, once) + `prefill_traced_chunked` (per call); non‑traced → `prefill_paged` |

### Why long prefill (both modes) is model‑owned

Two blockers, not one:
1. **Trace ceiling (traced only):** a single whole‑sequence prefill trace at long context overflows tt‑metal's 4 GiB uint32 `total_trace_size` → FATAL.
2. **GDN multi‑chunk recurrence (both modes):** GDN state must be (a) zeroed once before chunk 0, (b) carried *in place* across chunk calls, (c) the final partial chunk zero‑padded to the next multiple of 128 (not bucket‑padded), or the recurrent state decode continues from is corrupted. Generator's chunk loop knows none of this — it assumes each chunk only appends to the KV cache. `prefill_paged`/`prefill_layer_chunked`/`_forward_prefill_chunk_eager` already encapsulate all three correctly.

Invariant: **T ≤ 2048 → Generator (its own trace toggle); T > 2048 → model‑owned, both modes.** The single dividing line is the GDN multi‑chunk recurrence. (Routing long non‑traced through Generator was rejected: it would re‑derive the GDN orchestration as a framework‑side hook and create an ugly split where two paths differing only by a trace toggle live in different layers.)

### GDN state under Generator's decode trace — the one real risk

Decode is a *running* recurrence: GDN state updates in place every step and is carried forward (never reset between steps). The state buffers are persistent (`allocate_kv_caches`, `use_inplace_state=True`), so their addresses bake into Generator's captured trace like the KV pool — replay *should* carry state correctly. But Generator's decode‑trace machinery was written assuming the only mutable device state is the KV cache; it has never driven a model with extra hidden recurrent state. This is the item proven before any deletion (P2). If it fails, the old methods remain as fallback and we add a minimal GDN‑state hook (the one place a tiny `tt_transformers` touch could surface — flagged, not expected).

## 6. Local vLLM wrapper + demo migration

### Wrapper (`tt/qwen35_vllm.py`)

`Qwen35ForCausalLM(Generator)` replaces today's `TTQwen35ForCausalLM(nn.Module)`:
- `model_capabilities = {"supports_prefix_caching": False, "supports_async_decode": False}`
- `initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, **kw)`: resolve `HF_MODEL` from `hf_config._name_or_path` (unchanged), build via `create_tt_model`, bind caches via `model.allocate_kv_caches`, return `cls([model], [args], mesh_device)`.
- `prefill_forward` — the §5 dispatch.
- `decode_forward` → `super().decode_forward(...)`.
- `allocate_kv_cache` → `self.model[0].allocate_kv_caches(...)` (binds KV + GDN state; returns the 8 KV pairs for vLLM bookkeeping).
- `warmup_model_prefill` → `model.capture_prefill_trace_chunked(...)` (short is lazy); `warmup_model_decode` → dummy `decode_forward(enable_trace=True)`.

**Dropped:** the three vLLM protocol stubs (`embed_input_ids` / `forward` / `compute_logits`) — only needed because the old wrapper was an `nn.Module` vLLM probed via Protocol. Generator‑based wrappers (`QwenForCausalLM`, `GptOssForCausalLM`) don't carry them.

### Demo (`demo/text_demo.py`)

Keep all scaffolding (prompt loading, Frankenstein configs, perf measurement, `_assert_results`). Swap only the generation core:
- Build `Generator([model], [args], device)` directly (demo isn't vLLM).
- Decode loop → `generator.decode_forward(...)` (traced/non‑traced via `enable_trace`); host‑argmax on returned logits stays (deterministic for the perf test).
- Prefill → the shared qwen‑dir dispatch helper honoring traced vs non‑traced:
  - short (≤2048) → `generator.prefill_forward_text`
  - long traced → `model.prefill_traced_chunked`
  - long non‑traced → `model.prefill_paged` (kept; the path `_run_paged_generation` already uses)

The demo's `traced_*` / `paged_*` parametrization (`text_demo.py:176‑201`) keeps working; no long‑context coverage regresses (chunk‑outer path untouched).

## 7. Building‑block reuse

Sequenced **after** the interface. Each swap is independent and follows add‑new → validate → replace, judged against D2 (HF‑reference / keep‑quality), not bit‑identity.

| Order | Swap | Reuse target | Risk |
|---|---|---|---|
| 1 | `Qwen35MLP` → framework MLP | `tt_transformers.tt.mlp` | low — plain SwiGLU; deltas (bf4 gate/up, bf8 down, decode/prefill memcfg) as config |
| 2 | inline LM‑head `ttnn.linear` → `LMHead` | `tt_transformers.tt.lm_head` | low — gains vocab sharding (~151k) |
| 3 | inline `ttnn.embedding` → `Embedding` | `tt_transformers.tt.embedding` | low |
| 4 | `rms_norm_ttnn` → framework RMSNorm | `tt_transformers.tt.rms_norm` | low — `+1` offset stays weight‑prep |
| 5 | `Qwen35RoPESetup` → `RotarySetup` (attn layers only) | `tt_transformers.tt.rope` | medium — verify `partial_rotary_factor`; GDN layers use no RoPE; **optional, last** |

**Not reused** (the GDN‑specific "build custom" set, mirroring how gpt_oss/gemma4 hand‑build only their MoE/sliding parts): `Qwen35DecoderLayer` (hybrid dispatch), all of `tt/gdn` + `tt/gdn_kernel`, the `load_state_dict` remap (`weight_mapping.py`). `ModelArgs` is already reused.

## 8. Testing strategy

- **Interface plumbing (behavior‑preserving):** new contract unit test — build `Generator([model],[args],device)`, run decode + short prefill, assert logits match the pre‑refactor model methods on a fixed prompt.
- **Decode‑trace equivalence (the prove‑then‑delete gate, P2):** Generator‑driven traced decode produces the same token stream as today's `decode_traced_paged` before the old methods are deleted.
- **Reuse swaps:** per‑block component PCC vs HF (`tests/unit/test_component_pcc.py`) + e2e demo + the 61‑test suite as regression.
- **Long‑context (no‑regression):** the demo's `traced_8k/16k/32k/64k/128k` cases keep passing.

## 9. Phases

Honoring new‑before‑delete, interface‑first, Tier‑B deferred:

- **P0 — Baseline.** Logits oracle for the contract‑method inputs (proves P1 plumbing is behavior‑preserving).
- **P1 — Add contract surface.** Contract methods + attributes (pure additions); add the local `Qwen35ForCausalLM(Generator)` wrapper. No deletions.
- **P2 — Generator decode + short prefill.** Wire and validate Generator decode (traced + non‑traced) and short prefill against the old paths → **then** delete `capture_decode_trace_paged` / `decode_traced_paged`.
- **P3 — Prefill dispatch + demo.** `prefill_forward` dispatch (option b) + shared helper; migrate the demo to `Generator` + helper; confirm long‑context cases unchanged.
- **P4 — Reuse swaps.** MLP → LM head → embedding → RMSNorm → (optional) RoPE, each add‑new → validate → replace, vs the D2 bar.
- **P5 — Optional cleanup.** Retire the now‑unused legacy bucket trace (`capture_prefill_trace_paged` / `prefill_traced_paged`) once everything above is green.
- **Deferred — Tier‑B.** Paged GDN recurrent‑state management for vLLM continuous batching: documented framework gap, not built this cycle.

## 10. Risks & open items

- **R1 (primary):** Generator's decode‑trace machinery correctly carrying the model‑bound GDN recurrent/conv state across replays. Mitigation: P2 prove‑then‑delete gate; old methods stay as fallback; minimal GDN‑state hook only if needed.
- **R2:** short‑prefill impedance between `super().prefill_forward_text` and Qwen's `ttnn_prefill_forward` signature/return. Mitigation: adapt on the qwen side (shim); framework edit avoided.
- **R3:** RoPE reuse (`RotarySetup`) supporting `partial_rotary_factor`. Mitigation: optional/last swap; skip if it doesn't fit cleanly.
- **R4:** reuse swaps shifting numerics. Mitigation: D2 bar (quality vs HF), per‑block PCC + e2e regression.
```
