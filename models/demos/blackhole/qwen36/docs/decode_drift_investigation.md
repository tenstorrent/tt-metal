# Qwen3.6-27B (TT Blackhole) — Decode-Drift Root-Cause Investigation

**Date:** 2026-07-12
**Scope:** `models/demos/blackhole/qwen36/` on P150x4 (P300×2, 4 chips), mesh (1,4), TP=4.
**Status:** Root cause identified. Not a single fixable bug; a systematic prefill-vs-decode
algorithmic divergence that compounds across the 64-layer hybrid stack.

---

## 1. Symptom

The TT-served Qwen3.6-27B produces **fluent but factually-drifting** long generations:

- Short recall (capital of France, one-line biography) → **correct**.
- Multi-step arithmetic / long CoT (GSM8K) → **~0–5%**, vs GPU/HF reference ~90%.
- Failure mode: the model starts coherently, tracks the prompt numbers correctly for the
  first ~100–150 tokens, then **confabulates prompt numbers** ("16 eggs" → "12/14 eggs")
  and **loops**. Happens in greedy and sampling, demo path and vLLM, B=1 and B=8.

This is **pre-existing** and **independent of the fused-GDN-decode optimization** — `fused == python`
at every measurement, so PRs #49587 / #49588 are accuracy-neutral and healthy.

## 2. Method

All experiments are teacher-forced or single-op PCC vs a trusted reference:

| Tool | What it measures |
|---|---|
| `score_tp` (added to `model.py`) | prefill/all-position logits → **WikiText PPL 9.03** (prefill healthy) |
| `test_prefill_gen.py` | autoregressive generation via the **prefill path only** → **reasons correctly** (GSM8K solved) |
| `test_decode_vs_prefill.py` | incremental **decode** logits vs `score_tp` (prefill) reference, teacher-forced |
| `test_gdn_drift.py` | GDN **recurrent** kernel vs a torch fp32 recurrence |
| `test_attn_capture.py` | per-layer q-post-RoPE / attn-output, decode vs prefill |
| `test_sdpa_pcc.py` | `scaled_dot_product_attention_decode` op vs exact torch softmax attention |
| `test_gdn_chunk_vs_rec.py` | GDN **chunk (prefill)** kernel vs **recurrent (decode)** kernel, same input |
| `test_decode_gen_probe.py` | real decode-path generation, FLAT vs SHARP q/k-norm |

## 3. What was eliminated

| Suspect | Verdict | Evidence |
|---|---|---|
| Prefill path / weights | **healthy** | PPL 9.03; prefill-path autoregressive gen solves GSM8K |
| GDN recurrence (vs torch) | clean | recurrent kernel vs torch fp32 recurrence PCC **0.9998** (stable over 256 steps) |
| RoPE (decode vs prefill) | identical | `rot_mats_decode` / `rot_mats_prefill` freq + apply logic byte-identical (code inspection) |
| Compute precision | not it | forcing decode matmul+SDPA to HiFi4 → decode-vs-prefill unchanged (0.896 ↔ 0.897) |
| DistributedNorm decode-mode | not it | forcing PREFILL-mode norm at decode → unchanged (0.8973 ↔ 0.8972) |
| **SDPA-decode op** | **clean** | `scaled_dot_product_attention_decode` vs exact torch = **PCC 0.9998** (MHA/GQA, full/partial cache, HiFi2/HiFi4) |
| q/k-norm scheme | flat is best | see §5 |
| conv / l2norm (in GDN) | clean | GDN cross-path pos0 PCC 0.998 (state empty ⇒ ~conv only) |
| Fused GDN decode optimization | neutral | `fused == python` everywhere |

## 4. Root cause — GDN chunk (prefill) vs recurrent (decode) divergence

`test_gdn_chunk_vs_rec.py` feeds the **same** 128-token hidden sequence to one GDN layer via both
paths and compares per-position outputs (device-0 shard):

| position | mean PCC | min | first |
|---|---|---|---|
| 0–31 | 0.989 | 0.959 | **0.998** |
| 32–63 | 0.981 | 0.943 | |
| 64–95 | 0.972 | 0.904 | |
| 96–127 | 0.968 | 0.906 | |

Key observations:
- The two kernels are **not numerically equivalent**, and the gap **grows monotonically with
  position** — unlike the SDPA-decode op which is a flat 0.9998.
- **pos0 = 0.998** (recurrent state empty ⇒ essentially conv-only) localizes the divergence to the
  **recurrence / state accumulation**, not the FIR conv or the norms.
- It is the chunk-parallel associative scan vs the sequential recurrence: an algebraic
  reformulation that is only exactly equal in infinite precision; in fp32 they diverge and the
  divergence accrues with recurrence length.

**Mechanism of the symptom:** real decode builds the GDN state with the *chunk* kernel during
prefill, then continues with the *recurrent* kernel. As generation lengthens, the recurrent state
drifts away from the chunk reference the rest of the network effectively "expects." Hence: short
recall is fine, long multi-step reasoning drifts — exactly the observed GSM8K pattern.

`test_gdn_drift` (recurrent ≈ torch, 0.9998) tells us the **recurrent path is the more faithful one**;
the chunk kernel is what departs from the true recurrence. But the model reasons correctly *under the
chunk path* (prefill), so the operative problem is the **inconsistency between the two paths**, not
which one is "more correct."

**Scale / compounding:** one GDN layer's real-flow divergence is small (3 GDN layers →
decode-vs-prefill 0.9983). Across the full 64-layer hybrid stack (48 GDN + 16 full-attn) the small
per-layer divergences compound to **logit PCC ~0.90** (argmax agreement ~66%). Enough to keep text
fluent, fatal for exact-number multi-step reasoning.

## 5. The q/k-norm finding (branch reference)

The user recalled the branch (`yito/qwen36_27b_p300x2_tp`) being more accurate and asked to reference
it. Two branch differences were tested:

- **SDPA config** (branch: `q_chunk=32, k_chunk=32, max_cores_per_head_batch=16`): scores 0.41–0.51
  in the unit harness (does not transfer to main's kernel/shape) — **not** the source of any accuracy
  advantage; main's config is already at the 0.9998 ceiling.
- **q/k-norm**: HF `Qwen3_5RMSNorm` is zero-centered (`output*(1+weight)`), so the HF-correct and
  branch-consistent choice is **sharp (+1)**. main's decode uses **flat (no +1)**. But adopting sharp
  at decode makes real generation **much worse** (immediate confabulation + garbage loop), confirmed
  in both the proxy (sharp 0.78 < flat 0.90) **and** real decode generation. main's flat-norm is a
  HF-incorrect but empirically superior mitigation that partially masks the §4 divergence.

**Conclusion:** the branch's accuracy advantage does not come from its SDPA config or its norm scheme;
neither transfers to main. Do **not** switch main to sharp-decode or the branch SDPA config.

## 6. Bottom line

- The decode drift is a **systematic prefill(parallel)-vs-decode(incremental) algorithmic divergence**
  — hard-demonstrated in the **GDN chunk-vs-recurrent kernels** (growing with position) and propagated
  through the attention layers — that **compounds across 64 layers**. Not one fixable bug.
- `flat` q/k-norm at decode is the best available mitigation and should stay.
- The fused-GDN-decode work is accuracy-neutral; the shipped PRs are unaffected.
- Fix directions require kernel-level or path-level changes (see the fix-design note).

## 7. Reproduce

Overlay recipe (no rebuild; Python-only): mount `work/tt-metal` + the `qwen36-port` worktree's
`qwen36/` + flux `python_env` + HF cache; env `MESH_DEVICE=P150x4`,
`TT_MESH_GRAPH_DESC_PATH=.../p300_x2_mesh_graph_descriptor.textproto`,
`TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1`. Then:

```
pytest models/demos/blackhole/qwen36/tests/test_gdn_chunk_vs_rec.py -s     # the key finding
pytest models/demos/blackhole/qwen36/tests/test_sdpa_pcc.py -s             # SDPA op clean
pytest models/demos/blackhole/qwen36/tests/test_decode_gen_probe.py -s     # FLAT vs SHARP
pytest models/demos/blackhole/qwen36/tests/test_decode_vs_prefill.py -s    # end-to-end proxy
```

Diagnostic envs (all gated-off by default): `QWEN_ATTN_SHARP_DECODE`, `QWEN_DECODE_HIFI4`,
`QWEN_DECODE_NORM_PREFILL`, `QWEN_ATTN_CAPTURE`, `QWEN_GDN_FUSED_DECODE`, `N_LAYERS`.
