# DeepSeek-V3.2 — TT (ttnn) implementation of the MLA layer with DSA (Indexer)

## 1. Goal

Port the DeepSeek-V3.2 attention stack — **MLA** (Multi-Head Latent Attention) and the
nested **Indexer** (the "lightning indexer" of DeepSeek Sparse Attention, *DSA*) — from the
CPU reference to **ttnn**, running on a **single Tenstorrent device**. The TT layer must be
**functionally equivalent** to the CPU reference: same forward math, same prefill/decode
caching contract, validated numerically via PCC.

This is a **functional** port. Performance (speed) is explicitly out of scope (§9).

## 2. References

- **CPU reference (source of truth)** — `models/demos/deepseek_v32/reference_cpu/`
  - `model.py` — `ModelArgs`, `IndexerCPU`, `MLACPU`, building blocks (`Linear`, `LayerNorm`, `RMSNorm`).
  - `weights.py` — `initialize_weights` (single entry point: random or pretrained layer load + HF fp8 dequant).
  - `utils.py` — CPU kernel equivalents (`act_quant_cpu`, `fp8_index_cpu`, `rotate_activation_cpu`) + RoPE (`precompute_freqs_cis`, `apply_rotary_emb`).
  - `test_model.py` — equivalence / determinism / pretrained harness (the contract our TT tests mirror).
  - `MLA_LAYER.md`, `README.md` — full architecture, shapes, dataflow diagrams, and the prefill/decode/cache invariants. **Read these; this spec does not restate the math.**
- **TT MLA from DeepSeek v3 (pattern donor, do not modify)** — `models/demos/deepseek_v3_d_p/tt/mla/`
  - Used only as a **reference for conventions**: ttnn weight conversion + `as_tensor` caching, RoPE tensor setup, op/compute-kernel-config choices. We do **not** depend on or edit it.

## 3. Scope decisions (resolved)

| # | Decision | Choice |
|---|----------|--------|
| 1 | **Indexer (DSA)** | Implement the **full** layer (MLA **and** Indexer) in ttnn. Drop the orthogonal Hadamard transform and fp8 quantization functionally (they are identity / precision-sim — same simplification the CPU reference makes); fall back to CPU only for ops with no ttnn equivalent (§6). |
| 2 | **Device scope** | **Single device only** (`world_size = 1`). No CCL, no SP/TP sharding, replicated caches — mirrors the CPU reference. Multi-device distribution is a **documented follow-up**, not built here. Keep the head dimension explicit in tensor layouts so it can be sharded later, but add no distribution machinery now. |
| 3 | **Validation** | Compare TT output against the **live CPU reference** via PCC, using **(a) seeded random weights** (the equivalence + determinism contract) and **(b) pretrained layer-0 weights** (real-weight PCC, skipped if the ~5 GB shard isn't cached). |
| 4 | **Code reuse** | Write a **fresh** v3.2 ttMLA/ttIndexer; **borrow patterns** from v3 ttMLA by reimplementing locally. **Hard constraint: all new/changed files live under `models/demos/deepseek_v32/`** (both `reference_tt_single_chip/` and `reference_cpu/` are fair game; **nothing outside `deepseek_v32/` is edited**). Borrowed v3 helpers (rope, weight-cache, op configs) are adapted into `deepseek_v32/`, not modified upstream. |
| 5 | **Target arch** | **Blackhole.** Compute-kernel-config defaults target BH; correctness is arch-independent. |
| 6 | **Engineering** | Best practices: **no duplication, keep it simple.** Factor logic shared between MLA and the Indexer (rope, weight conversion/caching, op configs) into one place within `deepseek_v32/`; don't copy-paste. |

## 4. What to implement

A `ttMLA` and a `ttIndexer` (names TBD) under `models/demos/deepseek_v32/reference_tt_single_chip/`, each with:

- **`__init__`** — takes `ModelArgs` (or equivalent config), a `state_dict`, and a `mesh_device`
  (single-device mesh). Converts weights to ttnn tensors and caches them via `ttnn.as_tensor`
  (`cache_file_name`) so conversion happens once. Allocates the ttnn KV caches.
- **`forward`** — mirrors the CPU `forward` signatures and semantics:
  - **MLA**: `forward(hidden_states, start_pos, rope_tensors, mask)` → `out [B,S,dim]`.
    Internally calls the indexer, builds the additive index mask, runs the prefill (MHA) or
    decode (MQA-absorbed) path selected by `mask is None`.
  - **Indexer**: `forward(x, qr, start_pos, rope_tensors, mask)` → `topk_indices` (and
    `index_score` for inspection, as the CPU ref returns).

### 4.1 Front-end (both paths)
Q-latent stem (`wq_a`→`q_norm`→`wq_b`, interleaved RoPE on `q_pe`) and KV-latent stem
(`wkv_a`→split→`kv_norm`, interleaved RoPE on `k_pe`), then write latent `kv`/`k_pe` to the
caches. See `MLA_LAYER.md §2`. `qr` (post-`q_norm`) is also fed to the indexer.

### 4.2 Indexer (DSA)
Per `MLA_LAYER.md §A` and `README.md` Indexer section: `wq_b` (on `qr`), single-head `wk`,
**LayerNorm** `k_norm`, fp32 `weights_proj`; **non-interleaved** RoPE, `[pe, nope]` order;
score = `sum_h relu(q·k) * weights`; `topk(min(2048, T))`. Output is `topk_indices`.
Hadamard + fp8 dropped (orthogonal / precision-only).

### 4.3 Prefill (MHA) / Decode (MQA) paths
Exactly as `MLA_LAYER.md §3–§4` / `README.md`: prefill materializes per-head K/V via `wkv_b`
and runs dense SDPA; decode absorbs `wkv_b` into Q and the output and attends against the
latent caches. `index_mask` (`-inf` except `0` at `topk_indices`) is added to scores before
softmax; prefill also adds the causal mask. `softmax_scale = 192**-0.5 × mscale²` (YaRN, since
`max_seq_len(16384) > original_seq_len(4096)`).

### 4.4 Caches
`kv_cache [max_batch, max_seq, 512]`, `pe_cache [max_batch, max_seq, 64]`, indexer
`k_cache [.,.,128]` + `k_scale_cache [.,.,1]`, written at `[:, start:end]`. The
**cache-correctness invariant** (`MLA_LAYER.md §B.6`) must hold: prefill(P) ≡ prefill(P-1)+decode(1)
at position P-1.

## 5. Weights

- **Single conversion+cache path** for both random and pretrained, keyed by layer index, using
  `ttnn.as_tensor(..., cache_file_name=...)` (mirror v3 `_convert_and_cache_weights`).
- Source `state_dict` comes from the **CPU reference's `weights.py`**: random (`initialize_weights(module)`)
  or pretrained (`initialize_weights(module, layer=0, ...)`, which handles HF fp8→bf16 dequant).
  We consume the already-dequantized bf16/fp32 CPU module weights and convert to ttnn — we do
  **not** re-implement fp8 dequant. This reuse requires `reference_cpu/` to be **importable from
  `reference_tt_single_chip/`** — make it a proper package (add `__init__.py`, switch its intra-module `from model import …`
  to package-relative imports). That refactor is in scope (it lives under `deepseek_v32/`) and must
  keep `reference_cpu`'s own tests passing.
- Bias-free linears throughout. Norm weights: `q_norm`/`kv_norm` RMSNorm; indexer `k_norm` LayerNorm
  (weight+bias); `weights_proj` fp32.

## 6. ttnn op mapping & CPU-fallback policy

Default to ttnn ops; **fall back to host/PyTorch only where no ttnn op exists, and document each
fallback inline** (`# CPU FALLBACK: <reason>`). As-built mapping (implemented and validated):

| Operation | Implementation |
|---|---|
| Linears / matmuls | `ttnn.linear` |
| RMSNorm / LayerNorm | `ttnn.rms_norm` / `ttnn.layer_norm` |
| Interleaved RoPE (MLA) | `ttnn.experimental.rotary_embedding_llama`; cos/sin built Meta-style from the reference `precompute_freqs_cis` (YaRN match proven in `utils._meta_style_cos_sin`) |
| Non-interleaved RoPE (Indexer) | **CPU FALLBACK** — no ttnn op for the half-split convention; reuse the reference `apply_rotary_emb(interleaved=False)` (lossless bf16 round-trip) |
| Attention (prefill MHA) | **manual** `ttnn.matmul`→scale→mask→`ttnn.softmax`→`ttnn.matmul` (not SDPA: maps 1:1 to the reference einsums and adds the DSA index mask before softmax; the qk=192/v=128 dims are fine for SDPA via `head_dim_v`) |
| Decode absorbed einsums | `ttnn.matmul` over per-head `wkv_b1`/`wkv_b2`; absorption weights + head-shared latent cache materialized to `[B,H,..]` (ttnn.matmul broadcasts neither batch nor heads here) |
| Indexer `relu(q·k)`, head-sum | `ttnn.matmul` + `ttnn.relu` + `ttnn.multiply` + `ttnn.sum(dim=1)` |
| `topk` + scatter-to-`{0,-inf}` mask | **CPU FALLBACK** — `index_score` computed on device; `torch.topk` + scatter on host (exact selection match) |
| KV-cache storage | **host-managed** (CPU FALLBACK): latent computed on device, stored/indexed on host — ttnn cache ops require tile-aligned (mult-of-32) write offsets that the `start_pos=7` equivalence test violates; bf16 round-trip is lossless |
| Head merge | transpose+reshape (`nlp_concat_heads` overflows L1 at 128 heads) |
| Hadamard, fp8 act-quant (indexer), fp8 KV sim (MLA) | **dropped** — reference run with `use_fp8_path=False` / `simulate_fp8=False` for exact parity |

Softmax computed via `ttnn.softmax` (fp32 accumulation), matching the CPU ref's fp32 softmax.

**Precision deviations** (bf16 functional port): the indexer's `weights_proj` is fp32 in the
reference but converted to bf16 here (and `weights_proj(x)` runs in bf16 on device); norms run
in HiFi4/fp32-accumulate rather than pure fp32. These are within the 0.99 PCC floor and noted in
`convert_indexer_weights` / the module docstrings.

## 7. Tests (start here)

All under `models/demos/deepseek_v32/reference_tt_single_chip/`, pytest-discoverable, mirroring `reference_cpu/test_model.py`.
Use the same config (`ModelArgs`), seeds, and the small `seq_len` regime where `S ≤ index_topk`
(so the index mask is all-zero and attention math is isolated from discrete top-k jitter), plus at
least one shape exercising real top-k if feasible.

1. **TT-vs-CPU PCC (random weights)** — build CPU module (seeded), run live; convert the *same*
   weights to ttnn, run TT; assert output PCC. Covers both prefill and decode.
2. **Prefill-vs-decode equivalence (TT)** — port `test_mla_layer`: TT prefill(P) vs TT prefill(P-1)+decode(1)
   at position P-1.
3. **Determinism** — port `test_mla_pretrained_determinism`: N identical runs, output + caches unchanged.
4. **Indexer PCC** — TT `topk_indices` / `index_score` vs CPU indexer (shapes + score PCC; index-set match where unambiguous).
5. **Pretrained layer-0 PCC** — same as (1) with real layer-0 weights; **skipped if shard uncached**.

**PCC thresholds** (proposed, tune during bring-up): bf16 TT-vs-CPU ≥ **0.99** per op-path; prefill/decode
equivalence ≥ **0.999**; determinism ≥ **0.9999**. (CPU ref uses 0.999 / 0.9999 for the latter two.)

## 8. Suggested implementation order

1. Test scaffold + weight conversion/caching (random) — get a TT MLA prefill forward producing
   *some* output of the right shape.
2. MLA prefill PCC vs CPU (random) → fix op mapping until PCC passes.
3. MLA decode path + prefill-vs-decode equivalence + determinism.
4. Indexer in ttnn + indexer PCC; wire its `topk_indices` into MLA's index mask (replace the
   all-zero-mask shortcut).
5. Pretrained layer-0 PCC.

## 9. Out of scope

- Implementing ttnn ops that don't exist (fall back to CPU + document instead).
- Performance / speed optimization, tuned program configs, ring/fused-sparse attention.
- Multi-device distribution (SP/TP/CCL) — structure for it, don't build it.
- True fp8 storage / Hadamard / sparse-skip attention (functional sim only).
- Any change outside `models/demos/deepseek_v32/` (both `reference_tt_single_chip/` and `reference_cpu/` are in scope).

## 10. Status / resolved

Implemented and validated on a single Blackhole device — `reference_tt_single_chip/{utils,mla,indexer,test_model}.py`,
with `reference_cpu/` packaged for import and given `simulate_fp8` / `use_fp8_path` parity flags.
**All 12 TT tests + 10 reference tests pass.** TT tests (PCC vs the live CPU reference):

- `test_mla_prefill_pcc` (b1s8/b2s8/b1s100), `test_mla_prefill_decode_equivalence`,
  `test_mla_determinism`, `test_indexer_index_score_pcc`, `test_mla_with_indexer_sparse_pcc`
  (small `index_topk` exercises the real sparse mask), `test_mla_prefill_pretrained_pcc`.

Resolved during bring-up:
- **Non-interleaved RoPE / top-k / KV-cache** → host fallbacks (see §6), each documented inline.
- **bf16 PCC floor** — 0.99 op-path / 0.999 equivalence / 0.9999 determinism all hold comfortably.

Remaining follow-ups (not built, per scope): multi-device SP/TP distribution; migrating the
host-managed KV cache + top-k to on-device ttnn ops; a >`index_topk` (2048+) sequence test to
exercise the sparse mask at production scale.

## 11. Issues & learning points

Single-place log of every CPU/host fallback, ttnn workaround, and deviation hit during the port.
Each is also documented inline at the cited location; this section is the index.

### 11.1 CPU / host fallbacks (no clean ttnn equivalent)

| # | Fallback | Why it was needed | Notes | Location |
|---|---|---|---|---|
| F1 | **Non-interleaved RoPE** (Indexer) | `ttnn.experimental.rotary_embedding_llama` implements only the *interleaved* convention; the Indexer uses the half-split (non-interleaved) one. | Apply the reference `apply_rotary_emb(interleaved=False)` on host, re-upload. bf16 round-trip is lossless → numerically exact. | `reference_tt_single_chip/utils.py::apply_noninterleaved_rope_host` |
| F2 | **top-k selection** | Matching `torch.topk`'s exact selection/tie-breaking on device is fragile; the discrete indices feed an equality-sensitive mask. | The continuous `index_score` is computed **on device**; only the final `topk` runs on host. | `reference_tt_single_chip/indexer.py::forward` |
| F3 | **top-k → `{0,-inf}` scatter mask** (MLA) | No clean ttnn scatter over the key axis. | Mask built on host from the host top-k indices, then uploaded; added to scores on device. Inert (all-zero) when `S ≤ index_topk`. | `reference_tt_single_chip/mla.py::_index_mask` |
| F4 | **KV-cache storage/indexing** | ttnn cache ops (`fill_cache_for_user_`, `update_cache_for_token_`) require **tile-aligned (mult-of-32) write offsets**; the prefill(P-1)+decode(1) equivalence test writes at `start_pos=7`. | Latent is **computed on device**; only the cache *storage/indexing* is host-side. bf16 round-trip lossless → numerically exact. | `reference_tt_single_chip/mla.py::__init__/_write_cache/_read_cache` |

All four are functional-only; perf is out of scope (§9). Migrating F2–F4 to on-device ttnn ops is a follow-up (§10).

### 11.2 ttnn workarounds (op exists, but needed adapting)

| # | Issue | Symptom | Resolution | Location |
|---|---|---|---|---|
| W1 | `nlp_concat_heads` overflows L1 at 128 heads | `TT_THROW`: static CBs grow to ~2.2 MB on core `[0-0]` > 1.5 MB L1 (single-core op config). | Merge heads with `transpose(1,2)` + `reshape` instead. | `reference_tt_single_chip/mla.py::forward` (head merge) |
| W2 | `ttnn.matmul` won't broadcast **batch (dim0)** | Decode absorption `q[B,H,..] @ wkv_b[1,H,..]` → `TT_FATAL` batch-0 mismatch. | `ttnn.repeat` the absorption weights to batch `B`. | `reference_tt_single_chip/mla.py::_decode_attn` |
| W3 | `ttnn.matmul` won't broadcast **heads (dim1)** when `a[1]≠1` | Decode `q[B,H,..] @ kv_cache[B,1,..]` → `TT_FATAL` dim-1 mismatch for `B>1`. | `ttnn.repeat` the head-shared latent cache (and indexer key) to all `H` heads. | `reference_tt_single_chip/mla.py::_decode_attn`, `reference_tt_single_chip/indexer.py::forward` |
| W4 | Manual attention instead of a fused SDPA op | The DSA additive index mask must be added to the scores before softmax; mapping the reference einsums 1:1 is simplest for a functional port. (The qk=192/v=128 dims are *not* a blocker — SDPA supports a separate `head_dim_v`.) | **Manual** attention: `matmul → scale → +mask → softmax → matmul`, matching the reference einsums 1:1. | `reference_tt_single_chip/mla.py::_prefill_attn` |

W1–W3 are L1/perf-shaped and acceptable for a functional port (§9); each is the natural sharding axis under a future multi-device version.

### 11.3 Deviations & parity choices

| # | Item | Choice | Rationale | Location |
|---|---|---|---|---|
| D1 | MLA fp8 KV-cache simulation | **Dropped**; reference run with `simulate_fp8=False`. | e4m3-grid rounding has no clean ttnn equivalent; flag gives exact parity (both store bf16). Reference default unchanged. | `reference_cpu/model.py::MLACPU`, `tt` tests |
| D2 | Indexer Hadamard + fp8 path | **Dropped**; reference run with `use_fp8_path=False`. | Hadamard is orthogonal (does not change `q·k`); fp8 is precision-only (MLA_LAYER.md §A.6). Flag gives exact parity. | `reference_cpu/model.py::IndexerCPU`, `reference_tt_single_chip/indexer.py` |
| D3 | `weights_proj` (fp32 in ref) | Converted to **bf16**; runs in bf16 on device. | Functional bf16 port; within the 0.99 PCC floor. | `reference_tt_single_chip/utils.py::convert_indexer_weights` |
| D4 | Norms (`q_norm`/`kv_norm` fp32 in ref) | `ttnn.rms_norm`/`layer_norm` with **HiFi4 / fp32-accumulate**. | Closest device equivalent; within PCC floor. | `reference_tt_single_chip/utils.py::default_compute_kernel_config` |
| D5 | Interleaved RoPE cos/sin source | Built **from the reference `precompute_freqs_cis`** (not HF YaRN). | Guarantees bit-for-bit YaRN match; equivalence to interleaved `apply_rotary_emb` proven in code. | `reference_tt_single_chip/utils.py::_meta_style_cos_sin` |
| D6 | `reference_cpu` packaging | Bare imports → **absolute imports + repo-root bootstrap**. | Needed so `reference_tt_single_chip/` tests can import it; still runs as script, `pytest`, and `python -m pytest`. | `reference_cpu/{model,weights,test_model,conftest}.py` |

### 11.4 Things that worked without special handling (non-issues)

- Interleaved MLA RoPE via `rotary_embedding_llama` (Meta-style cos/sin, proven equivalent — D5).
- Batch > 1: fold batch into the head axis for RoPE so a single position table broadcasts.
- Decode RoPE at `seq == 1` (no decode-mode gather needed at the tested scale).
- Norm weight tile layout `[1, 1, dim/32, 32]` (v3 convention).
