# W2b loop plan — long-prompt masked attention, prompt + canvas > 32768

**Standalone plan for the W2b blocker (issue #47462).** Read [`DEVICE_LOOP.md`](./DEVICE_LOOP.md) first for the overall roadmap; this file is the deep, spike-first plan for the one device-attention path that does **not** yet scale past 32768. W2a (prompt+canvas ≤ 32768, single non-chunked masked SDPA) is **done & validated**; W2b is its long-context sibling and is **not** on the critical path to the short/medium-prompt functional milestone — it gates only the full-256K criterion.

**Audience:** an agent running this as its own `/loop`. **Definition of done:** a DiffusionGemma denoise step whose canvas attention is correct (PCC vs the all-attend torch reference) when `prompt_len + 256 > 32768`, up to the 256K context criterion, on QB2.

---

## TL;DR — the reframe (read this before assuming "new kernel")

The original framing ("non-causal masked chunked attention is new kernel work") is **probably wrong**, or at least far heavier than the actual gap. Source investigation (2026-06-26) found:

1. **Denoise attention is a `[256 × (P+256)]` rectangular ALL-ATTEND region**, not a triangular masked one. The canonical denoise mask is **all-zeros** for *both* sliding and full layers (`reference/attention_mask.py:71-76`; `tt/denoise_forward.py:36-42`, `NEG=-1e9` never actually applied in the all-attend case). The only load-bearing geometry is the **canvas RoPE offset = `prompt_len`** (`tt/denoise_forward.py:96,123`), which is already correct at any length.
2. **The non-causal, maskless SDPA path already exists in the kernel** and visits *every* K-chunk: `reader_interleaved.cpp:367-369` sets `q_high_idx = Skt` for `!is_causal`; `compute_streaming.hpp:1308-1311` documents "the only config that stamps nothing is plain non-causal attention with no mask at all." So an all-attend `[256 × longK]` SDPA needs **no mask and no causal logic**.
3. **The 32768 limit is not a hard constant** — there is no `32768`/`2^15` in the SDPA device-op index path; page/tile ids are `uint32_t` (`dataflow_common.hpp:167,229,288`), and 256K/32 = 8192 tiles/row stays well inside `uint32`. The "garbage > 32768" is an **empirically-observed** wrong-result cliff documented only in a gemma4 Python comment (`operations.py:25-29`), seen on the **causal large-Sq prefill** shape — the opposite extreme from W2b's tiny `Sq=256` / large `Sk`.

**⇒ The entire effort hinges on one cheap experiment (S1): does the existing maskless non-causal op return correct results at `[256 × Sk]` for `Sk` up to 262144?**
- **If S1 PASSES:** W2b collapses to **near-zero kernel work** — lift the `prefill.py:180-181` guard, re-derive `long_seq` against **K length** (not Q seq_len), pad `P+C` to a tile, and drop the mask. That is **D1**.
- **If S1 FAILS:** the cliff is most likely bf16 online-softmax numeric (running max/sum are `Float16_b`, `sdpa_program_factory.cpp:637-642`, accumulated over hundreds of K-chunks) → fall to **D4** (host K-chunking) or **D3** (kernel numerics), or **D5** (paged) if contiguous long-K DRAM is the ceiling.

Do **not** start any kernel work before S1.

---

## The problem (precise, source-grounded)

DiffusionGemma's denoise step needs the 256-token canvas to attend to `[prompt_prefix ; canvas]` **bidirectionally** (non-causal), and this must work when `prompt_len + 256 > 32768`. Today three facts block it:

- **Non-chunked SDPA is the only masked/non-causal path, and it's gated off above 32768.** `prefill.py:176-205` dispatches: `long_seq = seq_len > PREFILL_SDPA_MAX_SEQ` (32768); the masked branch (`prefill.py:179-190`) **hard-raises** for `long_seq` (`prefill.py:180-181`). That guard keys off **Q `seq_len`** — which for denoise is always 256 and never trips — so the guard is currently mis-keyed *and* the long-K case is untested.
- **The "chunked" long-context op is causal-only and refuses masks.** Both `chunked_scaled_dot_product_attention` overloads hardcode `is_causal=true`, `attn_mask=std::nullopt` (`sdpa.cpp:117,153,299,427`), and `validate_chunked_mode` FATALs on any mask (`sdpa_device_operation.cpp:216-218`). The causal assumption is the **K-chunk loop bound** (`compute_common.hpp:1947-1949`, `compute_streaming.hpp:1910-1913`, `reader_interleaved.cpp:361-369`), not just a triangular stamp.
- **`attn_mask` ⊥ `sliding_window_size`.** They share one L1-accumulate slot and are `static_assert`'d exclusive (`sdpa_device_operation.cpp:67-72`; `compute_streaming.hpp:1859-1860`). So sliding layers in denoise must bake any window into the dense mask — which the canonical all-attend mask already does (i.e. no window at all in denoise).

The masked non-causal SDPA **mechanism** is fully implemented (`compute_streaming.hpp:1312-1330` provided-mask streaming; `reader_interleaved.cpp:521-559` per-chunk mask reader; mask shape `[b|1, h|1, Sq, Sk]`, TILE, BF16/BFP8/BFP4, DRAM, `sdpa_device_operation.cpp:74-106`). W2a is literally this path at `Sk ≤ 32768`. **W2b is the same op (or its maskless twin) above 32768.**

---

## Loop protocol (spike-first)

1. **Run the spikes in order (S1 → …).** Each spike is a pure-op or single-layer device experiment vs a torch oracle — *no model build needed for S1/S2*. The spike outcomes route the decision tree below; do not pick a direction before its gating spike passes.
2. **One increment per iteration**, validated on QB2 (recipe in `DEVICE_LOOP.md` §1). Oracle = `torch.softmax(QK^T·scale) @ V` (all-attend) / `reference/attention_mask.py`.
3. **Record every spike result** (Sk, head_dim, PCC, pass/fail) in the Status section at the bottom of this file.
4. **Escalate, don't guess:** if S1 fails, the root-cause bisection (S3) decides between a localized numeric fix and a paged rewrite — these have very different cost; surface the bisection result before committing.
5. **Never mark W2b ✅ without a device PCC test at `[256 × >32768]`** (and at the 131072 / 262144 milestones) — this regime is currently completely untested (`test_sdpa_prefill.py` masked tests max ~2–8K, square `Sq==Sk`).

---

## Spike sequence (the heart of the plan)

### S1 — does the cliff even bite the W2b shape?  *(highest leverage, do first)*
- **Method:** pure `ttnn.transformer.scaled_dot_product_attention(Q=[1,nqh,256,DH], K=V=[1,nkv,Sk,DH], is_causal=False, NO attn_mask)` vs `torch.softmax(QK^T·scale)@V`. Sweep `Sk ∈ {8K, 32K, 33K (non-tile-aligned tail), 64K, 131072, 262144}`, `head_dim ∈ {512 (global, L1-tightest), 256 (sliding)}`.
- **Pass:** PCC ≥ 0.99 at every `Sk`, both head_dims (include the non-tile-aligned `Sk` to exercise the `use_padded_mask` writer, `sdpa_program_factory.cpp:240-245`).
- **Routes:** PASS → **D1** (collapse to guard-lift). FAIL → **S3**.

### S2 — masked A/B control  *(parallel with S1, cheap)*
- **Method:** identical sweep but **with** the explicit `[1,1,256,Sk]` all-zeros bf16 mask (the W2a call, `is_causal=False` + `attn_mask`).
- **Why:** if maskless (S1) passes but masked (S2) fails at the same `Sk` (or vice-versa), the cliff is **path-specific** (provided-mask streaming reader vs padded-mask writer) and is localized. If both pass, choose **D1** and drop **D2**.

### S3 — bisect the cliff  *(only if S1/S2 fail)*
- Force the **legacy** (non-streaming) compute kernel via `fp32_dest_acc_en` (`sdpa_program_factory.cpp:78,361` sets `use_streaming_compute=false`) and re-run the failing `Sk`. Legacy passes where streaming fails ⇒ bug is in `compute_streaming.hpp` online-softmax.
- Sweep `GEMMA4_PREFILL_SDPA_KCHUNK` at fixed `Sk`: if PCC tracks **chunk count** (not `Sk`), it confirms bf16 running-stat accumulation (`Float16_b` stats, `sdpa_program_factory.cpp:637-642`) ⇒ **D3/D4**. *(Caveat: confirm legacy compute handles non-causal maskless identically before trusting it as a control.)*

### S4 — RoPE cache reachability  *(independent, cheap, do early)*
- `model.py:124` default `max_seq_len=131072` < `config.py:196` `max_context=262144`; `model.py:475-477` slices `cos[:,:,:seq_len,:]`. Build/pass caches sized 262144 and assert `q_rope_offset=prompt_len` up to ~256K returns a **full-length** slice (and a hard error, not a silent short slice, if exceeded). **This gates any direction above 131072 regardless of SDPA.**

### S5 — paged non-causal corner  *(only if S1 passes but contiguous long-K DRAM is infeasible at 256K)*
- Drive `ttnn::prim::sdpa` directly (bypass the `is_causal=true` public wrappers) with `is_causal=false` + `chunk_start_idx=0` + a `page_table` covering `P+C > 32768`, `Q=[1,H,256,DH]` single chunk, **no mask**, PCC vs all-attend reference. Tests whether the existing reader/compute `is_chunked` + `!is_causal` branches **compose** before committing to **D5**.

---

## Candidate directions (cheapest / most-reuse first)

| Dir | Mechanism | Effort / Risk | Feasibility (critic) |
|---|---|---|---|
| **D1 — maskless non-causal regular SDPA (the reframe)** | Drop the all-zeros mask entirely; call non-causal SDPA over `[256 × (P+C)]` with `P+C` tile-padded. Reuses the existing zero-stamp non-causal streaming K-loop. Re-derive the `long_seq` guard against **K length**. | small / high | **MAYBE → the target.** Plumbing verified correct; only risk is the (unproven) cliff at this shape. **S1 decides.** |
| **D2 — keep the all-zeros mask, just lift the guard** | The exact W2a call at `Sk > 32768`. Legal shape (`sdpa_device_operation.cpp:104-105,118-121`). | small / high | **MAYBE, but strictly worse than D1** — materializes a full `[1,1,256,P+C]` DRAM mask (~134 MB/chip bf16 at 256K) for zero numerical benefit. Keep only as the **S2 A/B control** / a carrier for a non-canonical local-window op-test. |
| **D3 — fp32 running stats in streaming compute** | Carry online-softmax max/sum in fp32. | **large / medium** | **NO as described.** `sdpa_program_factory.cpp:637-642` asserts `im_df==stats_df`; the SALAD rescale binds out/sum/exp to one format (`compute_streaming.hpp:1596-1611`). fp32 stats ⇒ `fp32_dest_acc_en` ⇒ drops the streaming kernel. Not a localized edit. Use only if S3 proves a numeric cliff *and* D4 is rejected. |
| **D4 — Python K-chunking + host online-softmax** | Slice K/V into ≤32768 column blocks, run maskless non-causal SDPA per block, recombine with a host fp32 online-softmax (running max + exp-rescale), mirroring `operations.py:262-298` but iterating K with `Q=256` fixed. | medium / medium | **MAYBE (numeric fallback).** Fatal practical snag: the public SDPA op **does not return per-slice logsumexp/max** (`sdpa.cpp`), so recombination needs a second pass or a small op extension. Keeps every sub-op in the proven ≤32K regime. Dead weight if S1 passes outright. |
| **D5 — paged non-causal (maskless) chunked extension** | Relax `validate_chunked_mode` for `!is_causal`; add a public non-causal chunked wrapper forwarding `page_table` (+ optionally mask). Reads K from the paged cache in-kernel (`dataflow_common.hpp:256-315`) — long K never materializes contiguously. | large / very-high | **MAYBE, last resort.** Reader has independent `is_chunked`/`is_causal` branches that *appear* to compose, but compute has causal-coupled invariants (`compute_streaming.hpp:988` "KV-pad rotation mask is causal-only"). The **all-attend maskless** variant dodges the mask-page-id reconciliation. `ring_joint`'s `is_cross` proves non-causal short-Q/long-K chunked flash *runs* on BH (`ring_joint_sdpa_device_operation.cpp:319-331`) — existence proof, **not** forkable (CCL + structural mask). Pursue only if S1 passes but contiguous `[1,nkv,262144,DH]` K/V DRAM is infeasible per chip at TP=4. |

---

## Decision tree

```
S1 (maskless non-causal at [256 × Sk], Sk→262144)
├─ PASS ───────────────► D1: lift prefill.py:180-181 guard, re-key long_seq on K length,
│                            tile-pad P+C, delete mask.  (near-zero kernel work)
│                            └─ also run S4 (RoPE cache ≥262144) in parallel — independent gate.
│                            └─ if contiguous long-K DRAM infeasible at 256K → S5 → D5 (paged, maskless).
└─ FAIL ──► S3 (bisect: legacy vs streaming; chunk-count sweep)
            ├─ numeric (bf16 online-softmax) ──► D4 (host K-chunking, fallback)  or  D3 (kernel, heavy)
            └─ structural / DRAM ceiling ──────► D5 (paged non-causal, net-new kernel)
```

---

## Acceptance criteria (W2b done)

- **S1 device PCC:** non-causal maskless regular SDPA at `Sk ∈ {32K, 33K, 64K, 131072, 262144}`, `Sq=256`, PCC ≥ 0.99 vs torch `softmax(QK^T·scale)@V` on QB2 (mesh (1,4), TP=4), for `head_dim` 512 **and** 256, including a non-tile-aligned `Sk`.
- **End-to-end denoise step:** a single DiffusionGemma denoise step at `prompt_len` pushing `P+C` past 32768 (and at 131072, 262144) produces canvas hidden states matching the all-attend reference (`q_rope_offset=prompt_len`) at the gemma4 PCC convention (threshold = `floor(measured − 0.005)`, ratchet up only). Both sliding and full layers run all-attend (`sliding_window_size` forced `None`).
- **Guard re-derivation:** the `prefill.py:180-181` ValueError is re-keyed against **K length** (`P+C = tt_k.shape[-2]`), not Q `seq_len`; the W2a (`P+C ≤ 32768`) suite still passes unchanged; the long path is selected only when `P+C > PREFILL_SDPA_MAX_SEQ`.
- **Memory:** at `P+C=262144`, per-chip SDPA-input DRAM is documented and within budget — D1 materializes **no** mask (saves ~134 MB/chip bf16); the contiguous long-K `[1,nkv,P+C,DH]` footprint is measured and fits (else D5).
- **RoPE (S4):** caches sized to 262144 (or chunked slicing) verified so `q_rope_offset=prompt_len` up to ~256K returns a full-length slice, with a hard error (not a silent short slice) past the configured max.
- **If a kernel change is made (D3/D5):** the streaming path stays selected (`fp32_dest_acc_en` stays false); existing causal/sliding/joint/ring SDPA unit tests still pass; a new unit test covers the `[256, >32768]` non-causal regime.

---

## Risks & open questions

1. **Biggest unknown:** is the >32768 cliff **numeric** (bf16 `Float16_b` online-softmax running max/sum over hundreds of K-chunks) or **structural**, and does it bite the W2b shape at all? It was observed on the **causal large-Sq** prefill; W2b is the opposite extreme (`Sq=256` = one Q-chunk, one set of running stats vs very large `Sk`). These may behave completely differently. **S1 is the only way to know and it bifurcates the whole effort.**
2. **Contiguous long-K DRAM:** is `[1,nkv,262144,DH]` K/V allocatable per chip at TP=4 alongside weights/activations, or is paged KV (D5) mandatory at the top end? Decides whether the cheap D1 path scales all the way to 256K.
3. **RoPE cache** is only sized to 131072 by default — independent gate above that (S4).
4. **256K may break elsewhere:** the attention-output concat, the TP=4 allreduce/CCL, and activation residency may independently break above 32768/131072, separate from SDPA. W2b's SDPA fix is necessary but may not be sufficient for the full 256K criterion — scope a separate end-to-end-at-length check.
5. **Non-tile-aligned `P+C`:** the maskless `use_padded_mask` writer path (`sdpa_program_factory.cpp:240-245`) is unproven at >32768; S1 must include a non-aligned `Sk`. Fallback: tile-pad `P+C` in the `prefix_kv` layout.

---

## Key source map

| What | Where |
|---|---|
| 32768 cliff comment (empirical, causal large-Sq) | `models/demos/gemma4/tt/attention/operations.py:25-29` |
| Dispatch + **the guard to lift** | `models/demos/gemma4/tt/attention/prefill.py:176-205` (guard `:180-181`; W2a masked call `:182-190`; RoPE offset `:99-107`; prefix_kv concat `:116-128`) |
| Python Q-chunking (causal) / sliding chunker | `operations.py:220-298` / `:301-364`; program config `:185-217` |
| SDPA op: causal hardcode / mask reject / shape rules | `sdpa.cpp:117,153,299,427`; `sdpa_device_operation.cpp:56-60,67-72,74-106,118-121,216-218` |
| Non-causal full-K loop / maskless zero-stamp | `reader_interleaved.cpp:367-369`; `compute_streaming.hpp:1308-1311` |
| Provided-mask streaming reader / compute | `reader_interleaved.cpp:521-559`; `compute_streaming.hpp:1312-1330` |
| Online-softmax (running max / SALAD rescale) + stat format | `compute_common.hpp:2131-2177`; `compute_streaming.hpp:1596-1611`; `sdpa_program_factory.cpp:637-642` |
| Streaming vs legacy selection (`fp32_dest_acc_en`) | `sdpa_program_factory.cpp:78,361` |
| `use_padded_mask` writer | `sdpa_program_factory.cpp:240-245` |
| Paged in-kernel reader (for D5) | `dataflow_common.hpp:256-315`; uint32 ids `:167,229,288` |
| Causal-only KV-pad invariant (D5 threat) | `compute_streaming.hpp:988` |
| Non-causal short-Q/long-K chunked flash exists (not forkable) | `ring_joint_sdpa_device_operation.cpp:319-331` |
| Canonical all-attend denoise mask (all-zeros, both layer types) | `reference/attention_mask.py:71-76`; `tt/denoise_forward.py:28-49,36-42` |
| Canvas RoPE offset = prompt_len | `tt/denoise_forward.py:96,123` |
| RoPE cache size gate | `models/demos/gemma4/tt/model.py:124,475-477`; `config.py:192-197` |

---

## Out of scope / non-goals

- W2b does **not** gate the short/medium-prompt functional milestone (W2a covers `≤ 32768`). Flag it to the manager as a **standalone effort + likely-but-unconfirmed kernel risk** — the risk is fully resolved (up or down) by S1.
- This plan is attention-only. The decision-fidelity bar (#48291) and the rest of the e2e glue (#47464) are separate and tracked in `DEVICE_LOOP.md`.
- top-k/top-p, batching, and serving are out of scope here.

---

## Status log

| Date | Spike / step | Result |
|---|---|---|
| 2026-06-26 | Plan authored (source-grounded design + adversarial feasibility) | S1 is the gating experiment; D1 is the target pending S1; D3 ruled out as a localized edit. |
| 2026-06-26 | S1/S2 harness + smoke | Added `tests/test_device_long_sdpa_w2b.py` with a memory-bounded fp32 online-softmax oracle and opt-in full sweep. QB2 smoke passed: S1 maskless `Sk=8192,DH=256`; S1 maskless non-tile `Sk=33000,DH=256`; S2 masked `Sk=8192,DH=256`. |
| 2026-06-26 | S1 maskless non-causal `[256 × Sk]` PCC sweep | ✅ PASS on QB2: `Sk ∈ {8192, 32768, 33000, 65536, 131072, 262144}`, `head_dim ∈ {256,512}`, PCC ≥ 0.99 vs fp32 online-softmax oracle (`DG_RUN_DEVICE=1 DG_W2B_SDPA_SWEEP=full pytest models/experimental/diffusion_gemma/tests/test_device_long_sdpa_w2b.py -x -q`). Routes W2b to D1; no kernel work needed for this spike. |
| 2026-06-26 | S2 masked A/B control | ✅ PASS on QB2 for the same `Sk × head_dim` sweep with explicit all-zero bf16 mask; no path-specific cliff observed between maskless and masked regular SDPA. |
| 2026-06-26 | D1 maskless non-causal denoise path | ✅ Wired: `denoise_forward.py` defaults canonical all-attend denoise to `attn_mask=None` and passes `is_causal=False`; Gemma4 prefill attention now exposes an explicit `is_causal` knob while preserving the default causal path. Validated with CPU guards, `test_device_bidirectional_attention_integration.py` (7 passed), and the full W2b SDPA sweep (24 passed). |
| 2026-06-26 | S4 RoPE cache ≥ 262144 | ✅ PASS: `create_rope_caches` no longer allocates a hidden-width dummy for 256K caches; `_get_rope_mats` and `_slice_rope_cache` now hard-error on overrun. CPU guards passed (4/4), and QB2 `test_w2b_rope_slice_reaches_256k` verified `q_rope_offset=261888`, `canvas_len=256` returns a full 256-token slice from a 262144 cache. |
