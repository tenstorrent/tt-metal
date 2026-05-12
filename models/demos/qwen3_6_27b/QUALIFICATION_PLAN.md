# Qwen3.6-27B on BH Galaxy — DeltaNet Base Qualification Plan

We have **four in-flight branches** with overlapping coverage of the Qwen3.5 / Qwen3-Next / Qwen3.6 family. None is a complete starting point for Qwen3.6-27B on BH GLX, but the user's prior shelved branch supplies the reference + weight-remap glue, and the qualification reduces to one decisive test (kernel PCC at our shapes).

---

## 1. Branch landscape at a glance

| Branch | What it has | What it lacks | Verdict |
|---|---|---|---|
| **`ssinghal/qwen3.5-27B`** (user's own, shelved) | Standalone PyTorch reference of the full hybrid stack (`reference/gated_delta_net.py`, `reference/model.py`). HF→TT weight remap: `convert_hf_to_meta_qwen3_5`, `split_qwen3_5_attn_gate`, `map_hf_to_meta_keys_qwen3_5` — **handles `wq_gate` split** for `attn_output_gate`. Config plumbing in `tt_transformers/tt/model_config.py` (`linear_attention_pattern`, `partial_rotary_factor`, `rope_dim`, linear_*). TTNN gated attention with on-device QK-norm + partial-RoPE + paged_update_cache + sdpa_decode in `models/demos/qwen3_5/tt/attention.py`. | **DeltaNet block runs on CPU** (explicit comment: "no corresponding TTNN primitives"). Single-device, replicated weights, `tt_ccl=None` everywhere. No MRoPE despite config exposing it. No vision, no MTP, no trace, no batching, no real-weight PCC numbers. WH compute configs hardcoded. | **Reference + weight loader = direct use. TTNN gated-attention = lift with mesh sharding added. TTNN DeltaNet = replace entirely.** |
| `ign/deltnet_kernel_fusion` | Standalone DeltaNet TTNN kernels: recurrent (decode) + chunked (prefill). FLA-faithful math. PCC ≥ 0.999 vs torch up to T=32K on N300. Trace-safe constants precompute. Native `ttnn.conv1d` ≤512 + FIR fallback. | **Single device only.** No mesh/TP/CCL. WH compute configs hardcoded. "fused" chunked is a Python op-loop (not a real fused kernel). Test shapes (4 heads, K=128/V=256) don't match Qwen3.6 (16/48 heads, K=V=128). No model wrapper, no HF weight loader. | **lift with row-axis TP sharding added** — algorithmic foundation for the DeltaNet kernel |
| `ign/tt_qwen3nextcoder` | Mesh-aware (BHGLX 8×4 parametrized) gated-attention with output-gate + partial RoPE. Recurrent/chunked DeltaNet routing. TTNN MoE softmax routing. Trace-safe pattern. CCL via experimental async + DeepSeek CCL infra. | No mesh-sharded weights (opportunistic AG only — projection weights loaded full on each chip). No MRoPE. HF module-swap approach: PyTorch retains embedding/residual/RMSNorm/decoder; every layer round-trips through Python. No trace, no KV cache in TTNN, no recurrent state persistence. "Functional" only — no validated PCC. | **reference only** — algorithmic correctness; superseded by user's branch + `deltnet_kernel_fusion` for our path |
| `changh95/Qwen3.6-35B-A3B_bh_lb` | `is_qwen35` flag + `linear_num_key_heads` plumbing in `tt_transformers`. **Fused all_gather_matmul fix for `n_heads*head_dim != dim`** (exactly our 27B case). BH GLX CCL retuning. KV-head replication when `n_kv < n_devices`. Batch=64 decode (Python + C++ kernel limits). Custom-RoPE hook + non-fused paged_update_cache. SDPA LOFI + per-layer BFP4. | DeltaNet kernel absent. `TransformerBlock` is unconditional → linear-attn layers would run as standard attn. Targets P150x8 in num-splits dict. No 27B model_params or end-to-end demo. | **cherry-pick BH GLX scaffolding into tt_transformers** |

---

## 2. Locked composition

Decision locked. Composition by piece:

| Piece | Source | Status |
|---|---|---|
| PyTorch golden reference (hybrid text stack) | `ssinghal/qwen3.5-27B`: `models/demos/qwen3_5/reference/` | Direct cherry-pick; add vision ref + MTP ref later |
| HF→TT weight remap (incl. `wq_gate` split, linear_attn keys) | `ssinghal/qwen3.5-27B`: `tt_transformers/tt/load_checkpoints.py` | Direct cherry-pick; config keys identical for 3.5 ↔ 3.6 |
| Config plumbing (`is_qwen35`, `linear_attention_pattern`, `attn_output_gate`, `partial_rotary_factor`, `rope_dim`, linear_*) | `ssinghal/qwen3.5-27B` model_config + `changh95/Qwen3.6-35B-A3B_bh_lb` BH tunings | Merge: user-branch keys as base + changh95 BH num-splits + KV-replication + batch=64 |
| TTNN gated-attention (Q+gate, QK-norm, partial-RoPE, paged_update_cache + sdpa_decode) | `ssinghal/qwen3.5-27B`: `models/demos/qwen3_5/tt/attention.py` | Lift verbatim, then **add mesh sharding** (cols=4 head split, rows=8 dim split). Replace WH compute configs with `is_blackhole()`-conditional. |
| TTNN DeltaNet kernels (recurrent + chunked, conv1d native/FIR, RMSNormGated) | `ign/deltnet_kernel_fusion`: `models/experimental/gated_attention_gated_deltanet/tt/*.py` | Wrap in `tt/linear_attention.py` with row-axis TP sharding (48 V-heads/8=6, 16 K-heads/8=2; K_dim, V_dim=128 unsharded). BH compute configs. Trace-safe constants from same branch's `fused_chunked_delta_rule_placeholder.py`. |
| Layer-type dispatch in decoder | NEW (no branch handles correctly) | New code, ~30 lines: `if layer_types[i] == "linear_attention": linear_attn else: gated_attn`. |
| Fused all-gather matmul for `n_heads*head_dim != dim` | `changh95/Qwen3.6-35B-A3B_bh_lb` commit `fc90c3ac` | Direct cherry-pick |
| BH GLX CCL retuning | `changh95/Qwen3.6-35B-A3B_bh_lb` commit `43c75a2aae6` (chunks_per_sync=1, num_workers_per_link=1, AG links 2→1) | Direct cherry-pick |
| DeltaNet recurrent state cache | NEW (no branch has it) | Per-layer FP32 state `[B, 48, 128, 128]` sharded across rows → `[B, 6, 128, 128]` per chip ≈ 375 KB. DRAM-resident, paged into L1 each layer. |
| MRoPE [11,11,10] | `qwen3_vl/tt/rope.py` | Port into `tt_transformers/tt/rope.py`, gated by `mrope_interleaved` flag |
| Vision encoder (ViT, merger, patch/pos embed) | `qwen3_vl/tt/vision_*.py` + `patch_merger.py` | Direct wiring (shapes match: depth=27, hidden=1152, merger 4608→5120) |
| LM head | `llama3_70b_galaxy/tt/lm_head.py` | **Vocab-parallel** across 32 chips (40 MB/chip vs 5 GB if replicated) |
| MTP | `deepseek_v3/tt/mtp.py` | Phase 2 |
| Embedding | `llama3_70b_galaxy/tt/llama_embedding.py` | Direct use |
| CCL (AG/RS, prefetcher) | `tt_transformers/tt/ccl.py` + `changh95` retunings + `llama3_70b_galaxy` prefetcher | Direct use |

Working dir target: `models/demos/qwen3_6_27b/` (this directory) with `reference/`, `tt/`, `tests/`, `demo/` subdirs mirroring the qwen3_5 layout from the user's branch.

---

## 3. Qualification protocol (decide-by-data)

Run all three tests below at our target shapes, on BH GLX where possible (single BH for kernel-level, N300 surrogate for CCL-free sanity). **PCC threshold: > 0.99 vs HF reference.**

### Test A — DeltaNet kernel PCC at Qwen3.6 shapes (single chip, BH)
**What:** lift `ign/deltnet_kernel_fusion`'s recurrent and chunked TTNN ops; run at:
- `H_QK = 16, H_V = 48, K = V = 128, conv_k = 4, dt_bias[48], A_log[48]`
- Batch = 1, T ∈ {1, 32, 256, 4096, 32K}
- Real Qwen3.6 layer-0 `linear_attn` weights from HF, loaded via `convert_hf_to_meta_qwen3_5` from `ssinghal/qwen3.5-27B`.
- Golden = the PyTorch `gated_delta_net.py` from `ssinghal/qwen3.5-27B/models/demos/qwen3_5/reference/` (already validated self-consistent: prefill == decode).

**Pass:** PCC > 0.99 vs reference at all T, both recurrent and chunked paths.
**Fail action:** fall back to `ign/tt_qwen3nextcoder`'s `tt_symbiote/modules/recurrent_deltanet.py` (different matmul program configs); if both fail, investigate FLA numerical drift via element-wise p99 vs reference.
**Owner:** ttnn phase.
**Time budget:** 2 days.

### Test B — DeltaNet decode latency (single chip)
**What:** measure ms/step for one DeltaNet layer in recurrent mode at our shapes.

**Targets to beat:**
- < 1.5 ms / step (decode budget = total per-token target / 64 layers; if we want 25 tok/s ≈ 40 ms/tok, that's ~0.6 ms/layer; allow 2× for safety = 1.2 ms; round to 1.5).

**Pass:** ≤ 1.5 ms/step on a single BH chip with L1-resident activations.
**Fail action:** revisit kernel — either custom matmul program configs (`recurrent_deltanet.py` from tt_qwen3nextcoder), or push more state to L1, or reduce FP32 accumulation scope.
**Owner:** optimization phase.

### Test C — Mesh CCL on the gated-attention path
**What:** `changh95` branch's fused AGM K-dim fix on a 2-chip BH subset (use 4x_bh_quietbox if available), then on full 8×4 BH GLX if available, exercising `n_q=24, n_kv=4, head_dim=256`.

**Pass:** numerical equality vs single-chip replicated attention (PCC > 0.9999); no hangs; CCL overhead in line with the branch's profile from the commit "global CCL tuning + SDPA LOFI for Qwen3-32B decode".
**Fail action:** retune `chunks_per_sync`/`num_workers_per_link` per the changh95 commit, file a CCL issue if it's the BH 1-link fabric being the limiter.
**Owner:** debug + optimization.

### Test D (only if A+B pass) — End-to-end 4-layer hybrid slice
**What:** assemble (3 DeltaNet + 1 Gated-Attention + 4 MLPs + norms) on BH GLX with real Qwen3.6-27B layer-0…3 weights. Compare against HF reference run on CPU/GPU.

**Pass:** PCC > 0.99 on hidden state after layer 3.
**Fail action:** layer-by-layer bisect using the skill `/debug` — most likely culprits: GroupRMSNorm dtype, RoPE freq table, mask construction, recurrent state initialization.

---

## 4. Decision tree

```
                ┌── A pass ──┐
                │            │
                │      ┌── B pass ──┐
                │      │            │
                │      │      ┌── C pass ──┐
                │      │      │            │
                │      │      │      ┌── D pass ──> SHIP composition above
                │      │      │      │
                │      │      │      D fail ─────> debug layer-by-layer; recheck PCC + masks
                │      │      C fail ─────────────> CCL retune; if persistent, escalate fabric BW
                │      B fail ────────────────────> switch to tt_qwen3nextcoder recurrent kernel
                A fail ──────────────────────────> hold; investigate FLA math drift; numerical analysis
```

---

## 5. What we are NOT doing

- Writing a new DeltaNet kernel from scratch. At least one of the two branches has working math — qualify, don't reinvent.
- Treating any branch as a black-box drop-in. Every component is a candidate to be re-validated at Qwen3.6 shapes.
- Lifting `tt_qwen3nextcoder`'s HF-module-swap wrapper. That's a research scaffold, not a production model — the scaffold from `changh95` is the right host because it gives us proper TP-sharded weights, trace, and TTNN-resident KV cache.

---

## 6. Tracking

| Task | Owner | Status |
|---|---|---|
| Test A — kernel PCC | (ttnn phase) | not started |
| Test B — decode latency | (opt phase) | blocked on A |
| Test C — mesh CCL | (debug phase) | parallel with A |
| Test D — 4-layer hybrid PCC | (ttnn phase) | blocked on A+C |
| Update `ARCHITECTURE.md` §6 + §10 with concrete file refs once Test A picks a base | — | blocked on A |

Anchor file: this document. Update it as qualification evidence comes in.
