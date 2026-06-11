# TT multichip deepseek v3.2 MLA

## Goal
Implement deepseek v3.2 MLA layer for multichip TT setup. Start from existing v3 and introduce changes.
Identify missing ttnn operations.
Identify problems and what needs to be updated/changed.

## Scope
Prefill only.
MLA layer.

## Agreements
1. **Start from the existing v3 op and modify it**.
2. Keep track of issues and learning points.
3. K cache stays in the same format so that it doesn't affect decode downstream.
4. ~~TopK needs Row-major input.~~ CORRECTED 2026-06-10: ttnn.topk asserts TILE input (topk_device_operation.cpp:152), verified on Blackhole.
5. PCC truth is the CPU reference (reference_cpu), not v3 — v3.2 output is not assumed to match v3.
6. Weights: start from random initialization, move to pretrained later. Pretrained MLA loading already exists in reference_cpu; torch→ttnn conversion + sharding exists in v3 (_convert_and_cache_weights / build_ttnn_cache) and is reused.
7. Always follow tensor shapes — document the shape (and sharding) of every input/output/intermediate; CPU reference shapes are normative.
8. APIs for missing ops are derived from the fused-op reports in context/ (DeepGEMM fp8_mqa_logits for indexer, FlashMLA sparse attention) + composing existing ttnn ops, per "Approach to missing ops".
9. Every decision is documented here before/while implementing — keeps implementation unblocked and reviewable async.
10. Test-first whenever possible. For a missing op the first test checks shapes (op API runs end-to-end with the agreed input/output shapes); numerics vs CPU reference come after.
11. Target hardware: QuietBox, 4 Blackhole devices. Tests parametrized by mesh shape so Galaxy works later — no hardcoded 4-device assumptions.
12. Parallelism and sharding follow v3 decisions exactly. Any deviation forced by v3.2 is documented here. CPU reference is single-device truth only — it says nothing about distribution.
13. Mesh bring-up order: start 1x4 (pure TP, matches spec-multichip TP-only plan); add 2x2 (SP x TP, exercises chunked path) later. Both stay parametrized.
14. Scale bring-up order: small single-shot first (~4-8k tokens, cheap CPU reference); 50k cache + 5k chunk later as gating milestone with cached CPU outputs.
15. Chunk size is a configurable test parameter, 1k for the dev loop (CPU truth fast enough for iteration); 5k once proper kernels (fused sparse attention) land. Same code path either way.

## Status (updated 2026-06-11)

### ✅ Foundations (pre-port)
- [x] CPU reference (reference_cpu) — MLA + Indexer, matches DeepSeek reference
- [x] Single-chip ttnn port (reference_tt_single_chip) — MLA + Indexer w/ CPU fallbacks; decisions in spec.md, multichip plan in spec-multichip.md
- [x] Multichip scaffold (tt/ + tests/) reusing deepseek_v3_d_p as a library
  - 🧩 tt/mla/mla.py — ttMLA subclass of v3 ttMLA; all DSA changes live here
  - 🧩 tt_prefill_block.py / tt_prefill_transformer.py — v3 copies, only MLA/block import changed (→ backlog 13: upstream injection params, delete copies)

### ✅ Step 1 — test rewire to CPU reference (2026-06-10, 1x4: seq256 0.9991 / seq2k 0.9986, KVPE 0.9999)
- 🧩 Weight map MLACPU→v3 dict is a pure rename (wq_a→q_a_proj, …, wo→o_proj); same [out,in] layout, bf16
- 🧩 Bring-up seq = index_topk(2048): DSA mask is 0 over causal region → MLACPU dense-equivalent, passthrough must match; seq>2048 diverges by design
- 🧩 MLACPU simulate_fp8=False (device KVPE bf16); HF config shape-asserted vs ModelArgs; rope interleaved↔reference proven (spec D5)
- 💾 CPU outputs cached at $DEEPSEEK_V32_MLA_REF_CACHE, keyed tag+seq+seed; seq2k cold 431s
- 🐞 Fixed: ModelArgs.max_seq_len=seq_len disabled YaRN in the CPU ref (mscale² 1.87x drift, PCC 0.78) → keep max_seq_len=16384

### ✅ Step 2 — missing-op APIs (tt/ops.py), shapes + numerics on 1x4
- [x] Shape contracts (test_ops_shapes.py, 5 tests, 8.5s): indexer_logits composed on device; ttnn.topk on device at k=2048 (TILE in); sparse_mla CPU fallback
- [x] Numerics (test_ops_numerics.py vs functional CPU path): indexer_logits PCC ≥0.99, sparse_mla 0.999
- [x] topk index-set overlap vs torch ≥1−max(2/k,1%) per row — bf16 ties swap a small boundary band (11/2048 measured; not a bug)

### ✅ Step 3 — single-shot DSA in v32 ttMLA (2026-06-10, suite 12/12; seq4k 0.9975, sparse rows 0.9948)
- 🧩 `seq ≤ index_topk` → `super().forward` (dense == sparse); `seq > index_topk` → `_dsa_forward` = v3 forward with ring SDPA replaced by `ops.sparse_mla` over top-k latents, `wkv_b2` after attention
- 🖥️ Indexer stems on host (qr from MLA q_a host copy; F1 non-interleaved rope); logits+topk on device. Limit: sp=1 single-shot only (asserted)
- 🧪 e2e seq4k case; indexer weights flow via WEIGHT_NAME_MAP (popped before v3 sees them); band PCC diagnostics kept in test_mla.py
- 🐞 Bugs found & fixed:
  - [x] hidden is TP-sharded → indexer host stems concat shards
  - [x] epilogue is RS-only (RS+AG gave replicated 28672 output)
  - [x] sparse_mla must re-impose causality for rows with <k causal keys (future indices from topk's −inf band) — now op contract
  - [x] CPU truth ran fp8/Hadamard indexer (selection-divergent) → `use_fp8_path=False` per spec §104
  - [x] q head-sharded but sparse_mla read shard 0 + replicated → 3/4 chips wrong heads; now per-shard (q/out TP-sharded, kvpe+indices replicated)
- ⚠️ Gotcha: don't pipe pytest through `tail` — swallows exit code; log to file, check `$?`

### ✅ Step 4 — chunked DSA prefill (2026-06-11, 1x4: 4k cache + 1k chunks, PCC 0.9982, per-chunk ≥0.996; suite 15/15)
- [x] (1) `sparse_mla(start_pos)` causal offset
- [x] (2) host indexer K-cache + chunked `_indexer_topk`
- [x] (3) chunked `_dsa_forward` (v3 `update_padded_kv_cache` write, rope offset, sparse_mla over populated prefix)
- [x] (4) chunked e2e harness (chunk loop + `get_rope_tensors_indexed`; MLACPU decode-branch truth w/ chunk mask, cached per (seq,chunk,seed))
- 📐 Chunk size is a test param (agreement 15): 1k dev; 50k gate postponed (backlog 3); 5k deferred to fused kernels
- 🐞 Bugs found & fixed:
  - [x] dense-passthrough chunks skipped indexer K-cache write → DSA chunks scored against zeros (0.855/0.725 → ≥0.996). **Lesson (3rd occurrence): write EVERY per-chunk cache on every chunk, dense or sparse**
  - [x] ring buffer sized from constructor seq_len = full cache, not chunk
  - [x] MLACPU dense branch out-of-bounds for chunked truth (start_pos>0 → decode branch + chunk mask)
  - [x] bf8 cache quantization ruled out (mirror test); KVPE prefix 0.9999 isolated the fault to selection

### 🔨 Step 5 — 2x2 SP×TP (backlog 4, in progress 2026-06-11)

**Premise:** the 1x4 code already uses v3's **hidden-sharded** residual + TP-per-head stems (RS/AG via `_tp_rs_ag`), i.e. spec-multichip §3.6.1's end-state TP layout — *not* the "replicated sequence" Phase-0. So at sp=1 the TP scheme already matches v3 exactly. 2x2 only **adds the SP (sequence) axis**; authority for the layout is spec-multichip §3.6.1. Mesh stays parametrized (agreement 11/13).

**Distribution vs v3** (per-block; `=` follows v3 exactly, `Δ` v3.2-specific):

| Block | v3 | v32 1x4 (done) | v32 2x2 plan |
|---|---|---|---|
| Stems wq_a/wkv_a + norms | input-sharded TP, RS/AG | = | = (SP just means fewer tokens/chip; TP RS/AG unchanged) |
| wq_b/wkv_b heads | TP per-head, H/tp local | = | = |
| MLA kvpe cache | SP-sharded seq, TP-replicated | sp=1 (no shard) | **= v3**: reuse init_kvpe_cache/fill/update, SP-shards at sp=2 |
| Attention core | `ring_joint_sdpa` (ring over SP) | `ops.sparse_mla` (host) — Δ DSA needs index mask, no ring-SDPA mask hook (§3.3) | Δ host gather of full-T KVPE across SP, then local sparse attn |
| Indexer stems | n/a (no indexer in v3) | device, TP input-sharded + AG-reduce (backlog 6) | = TP; + SP |
| Indexer K cache | n/a | host `_index_k_cache` flat | Δ keys are SP-local; gather across SP for scoring |
| indexer_score+topk | n/a | device, full seq (sp=1) | Δ local-Q × full-T keys (after SP gather) → topk |
| o_proj | row-parallel + RS | = | = |

**The only thing 2x2 adds is SP communication on the key axis**, and v3 solves it with `ring_joint_sdpa`/`ring_mla` — which v3.2 can't use (no additive-mask hook for DSA, §3.3). So v3.2 substitutes **host SP-gathers** (functional, per "no ttnn op → CPU fallback"); device ring_sparse_attention is the documented follow-up (backlog 8/12).

**DECISION (2026-06-11, confirmed): replicate the indexer key cache, keep the MLA KVPE SP-sharded (v3).** Index key is tiny (single head, 128-wide) so full-T replication is cheap (~T·128·2B) and turns the read-time SP gather into a one-shot gather-at-write into the (host) cache — removes distributed-topk entirely. The big MLA KVPE latent stays SP-sharded per v3; sparse_mla gathers selected latents. Deviation from §3.6.1 (which SP-shards the index cache), justified for a functional port; documented per agreement 12.

**Key implementation note — global positions.** Under SP each chip's local tokens map to *non-contiguous* global positions (contiguous sharding: chip sp_i holds global [sp_i·S/sp, …)). So per SP shard the host RoPE freqs offset and the causal-mask triu offset must use the **global** query start (sp_i·local + start_pos), not the local/chunk offset. q and out stay SP-sharded; only keys/latents are gathered full-T.

**Slices (test-first):**
- [x] 5.1 lifted `sp_factor==1` assert; added 2x2 (mesh (2,2)) to single-shot test param; 1x4 regression green (0.9966 unchanged).
- [x] 5.2 indexer: SP all-gather of the stem outputs (k/q/wts) to full-S (device `all_gather_async` over sp_axis) → existing global-contiguous logic runs unchanged on full seq; index cache replicated, full indices replicated. (Simpler than per-shard global-pos: gather makes positions contiguous.)
- [x] 5.3 sparse_mla SP×TP-aware: KVPE SP-gathered full-T in `_dsa_forward`; per (sp_i,tp_j) shard attends local queries (global-pos causality) → reassembled via ShardTensor2dMesh (heads on tp, seq on sp). sp=1 collapses to prior behavior (regression-safe).
- [x] **5.4 e2e PCC vs CPU reference on 2x2 single-shot: seq4k 0.9974 (sparse rows 0.9925, dense 0.9987), KVPE 0.9999 — matches 1x4.** Same cached truth (distribution-agnostic).
- [ ] 5.5 **chunked 2x2** (guarded with NotImplementedError now): the chunked `_dsa_forward` cache-slot prefix read must SP-gather; indexer write already SP-aware. Follow-up.

### CPU fallbacks (multichip) — running list
| id | fallback | where | status / SP behavior |
|---|---|---|---|
| F-rope | non-interleaved RoPE on host (issue #4) | indexer pe slices | host; SP-agnostic (per-token) |
| F-sparse | sparse_mla gather+SDPA on host (backlog 8) | `ops.sparse_mla` | host; **5.3 makes it SP-gather full-T KVPE** |
| F-mla-prefix | MLA cache-slot prefix readback (backlog 9) | chunked `_dsa_forward` | host; reads slot — **5.3 gathers across SP** |
| F-idx-key | indexer key SP-gather (new, 2x2) | `_indexer_topk` | **5.2**: host AG of index keys across SP for full-T scoring |

### ⏭️ Next
Open work tracked in the Backlog section below.

## Backlog (execution order; numbers are stable cross-refs)

Legend: `[x]` done · `[ ]` open · ⏸️ postponed · 📌 resolved as decision (no code).

### Recommended implementation order (open items, 2026-06-11)

`18 → 14 → 13 → 4 → 9 → 8 → 19 → 16 → 3 → 15 → 11/12`

| # | Item | Why here |
|---|---|---|
| **4** | **2x2 SP×TP** | top production blocker; **rewrites the sparse/cache read paths** → do before the perf-debt items below (else 1x4 rework). Folds in **(17)** mask dedup |
| 9 | MLA cache-slot readback → device | done inside (4)'s SP-aware read path |
| 8 | sparse_mla gather+SDPA → device | biggest perf debt; SP-aware after (4); stand-in for fused (12) |
| 19 | indexer key cache → device | couples with (4); prereq = device non-interleaved RoPE (**issue #4**) |
| 3 ⏸️ | 50k scale gate | hardware-time gated; cheaper once 8/9 speed the path |
| 16 | multi-layer / multi-user cache | broadens to the full model |
| 18 | determinism tests | cheap, no deps — guards every change below |
| 14 | v32 tests in CI | small after 18; locks regressions (long CPU-truths gated) |
| 13 | upstream injection → v3, delete copies | independent hygiene; kills drift from copied files |
| 11/12 | fused C++ ops | out-of-scope follow-ups (Approach §4) |
| 15 | decode path | beyond current prefill-only scope; largest expansion |

**Step 4 — chunked prefill**
- [x] **(1)** MLACPU decode branch accepts intra-chunk causal mask (was mask=None → no within-chunk causality)
- [x] **(2)** chunked e2e harness (chunk loop, get_rope_tensors_indexed, chunked ttMLA); slice-3 wiring tested at 4k+1k
- [ ] ⏸️ **(3)** scale gate 50k cache + 1k chunks overnight (cached truth); 5k once fused kernels land (agreement 15)

**Functional gaps (blocking production)**
- [ ] **(4)** 2x2 SP×TP mesh — indexer needs full sequence per chip (seq AG or distributed topk); sparse_mla prefix read needs SP-aware gathering
- [x] **(5)** pretrained weights — test knobs (conftest `--ds-layer` / `--ds-checkpoint` / `--ds-repo` / `--ds-input`); `build_cpu_reference(layer, checkpoint_path, repo)` loads a specific MLA+indexer layer via reference_cpu `initialize_weights`; `make_hidden(--ds-input)` injects file-driven input (chunked + indexer tests; single-shot uses v3 harness input). ref-cache keyed by weight source. **Validated on real layer-0 weights: seq256 (dense) output PCC 0.9997; seq4k (DSA active) output 0.9996, sparse rows 0.9994, KVPE 0.9999** — full path HF download → fp8 dequant → weight map → PCC. DSA on trained weights ≈ random or slightly better (sharper top-k selection).
- [x] **(6)** device-side indexer stems — wq_b/wk/weights_proj GEMMs + k_norm (LayerNorm) + TP all-reduce on device, replicated across TP; wk/weights_proj sharded on the `dim` contraction axis (per-chip partials → `_tp_rs_ag`); qr reuses the v3 q_a stem. Only non-interleaved RoPE stays on host (F1, pe slices read back per chunk). Eliminates the per-chunk full-hidden readback + host GEMMs. Validated 1x4: indexer chunked==single-shot selection green; seq4k e2e PCC 0.9966 (was 0.9975 host — `weights_proj` fp32→bf16 cost, within 0.98 threshold). Test needs FABRIC_1D now (device CCL). Follow-ups → backlog 9/10 (device K-cache, drop pe readback), 4 (2x2)
- [x] 📌 **(7)** fp8/Hadamard parity — follow v3 cache format (kvpe bfloat8_b); ttnn has no matching fp8, so the functional path is the contract; truth stays use_fp8_path=False/simulate_fp8=False

**Host fallbacks → device ops (perf debt; contracts in tt/ops.py + Missing op APIs)**
- [ ] **(8)** sparse_mla gather+SDPA — full host fallback per layer (biggest copy/compute hit); needs sparse gather + SDPA-with-indices, per-row valid length + start_pos
- [ ] **(9)** MLA cache-slot host readback in chunked _dsa_forward / sparse_mla (whole KVPE prefix read every chunk)
- [x] **(10)** ~~indexer host stems readback (full hidden concat per chunk)~~ — resolved by (6); only the pe-slice RoPE readback remains, folded into (6)'s F1 host-rope note (coupled to issue #4)
- [ ] **(19)** indexer key cache → device — reuse v3's cache-*creation*/write **logic** (init_kvpe_cache-style allocator + fill_cache_for_user_/update_padded_kv_cache) to stand up a parallel, index-shaped cache ([max_seq, 128]); not the physical KVPE tensor. Gated on device-side RoPE (#4, so keys are written device-resident) and the dense-full-prefix-read vs SP-shard question (couple with (4) 2x2). Today: host `_index_k_cache` torch buffer.

**Fused C++ ops (out of scope per Approach §4, documented follow-ups)**
- [ ] **(11)** indexer_logits (DeepGEMM fp8_mqa_logits-style), causal windows, fp8
- [ ] **(12)** sparse FlashMLA-style attention, per-row topk_length

**Hygiene**
- [ ] **(13)** upstream mla_class/block_class injection to v3 → delete the two copied files
- [ ] **(14)** v32 tests in CI
- [ ] **(15)** decode path
- [ ] **(16)** multi-layer / multi-user cache
- [ ] **(17)** replicated-vs-sharded mask dedup
- [ ] **(18)** determinism tests (Testing section)

## References
1. models/demos/deepseek_v32/reference_cpu - deepseek's reference implementation running on CPU w/o fused ops and sparse attention
2. models/demos/deepseek_v32/reference_tt_single_chip - reference implementation using ttnn that runs on single chip and w/ CPU fallbacks
3. models/demos/deepseek_v3_d_p - tt multichip implementation for deepseek v3

## Issues
1. No fused indexing op in ttnn (fp8_index + causal mask) — composed-op workaround in tt/ops.py::indexer_logits (device, bf16, no fp8/Hadamard); fused C++ op is follow-up
2. ~~ttnn.topk k=2048 untested~~ RESOLVED 2026-06-10: works on Blackhole at k=2048 (TILE input, shape test green)
3. No sparse attention in ttnn — CPU fallback in tt/ops.py::sparse_mla carrying the agreed contract (TP-shard distribution, causality, start_pos); fused C++ op is follow-up
4. Missing non-interleaved RoPE op — host fallback (F1); after (6) only the pe-slice readback remains on host.
   **Proposed device fix (all on device, no readback):** the two conventions differ only by a fixed head-dim permutation `P` (interleave the two halves), same per-pair freqs → `RoPE_ni(x) = P⁻¹·RoPE_il(P·x)`.
   - Bake `P` into v3's rope transformation matrix (`get_rot_transformation_mat`) so a single device matmul yields the non-interleaved result — tile-friendly, reuses v3's rope hook (the naive permute crosses 32×32 tile boundaries sub-tile, so avoid explicit transpose/slice/concat in TILE layout).
   - Indexer optimization: only `q·k → topk` matters and `P` is orthonormal (preserves dot products), so q and k can stay in permuted space — **no inverse permute needed** as long as every chunk is consistent and the index K-cache stores permuted-space keys (it's internal to v32). Inverse permute only where a consumer needs non-interleaved layout (MLA KVPE — already interleaved, matches v3).
   - TODO before impl: confirm `rotary_embedding_llama` accepts a custom trans_mat and the cos/sin freq ordering lines up. Unblocks (19) + full device residency.
5. v3 composition files hardcode ttMLA/TtPrefillBlock — forced copies in v32; fix by upstreaming injection params (tt/README.md)
6. V3.2 checkpoints (indexer weights) not wired into test conftest — tests run with v3 weights

### Missing op APIs (proposed 2026-06-10, step 2 — review async)
ttnn-shaped equivalents of the fused references (DeepGEMM fp8_mqa_logits, FlashMLA sparse fwd). All activations [1, B, S_local, ·] TILE bf16 like v3; indexer replicated across TP, S sharded on SP (spec-multichip §3.6). B=1 prefill.

1. `indexer_logits(q, k, w) -> logits` — q [1,B,Sq,H_idx*D_idx] (H=64, D=128), k [1,B,Skv,D_idx], w [1,B,Sq,H_idx] (fp32 weights_proj out). Out [1,B,Sq,Skv] bf16 (fp8 inputs later). Causal window per row (DeepGEMM ks/ke), no materialized mask. Workaround: per-head matmul + ReLU + weighted head-sum + causal mask add. CPU fallback for non-interleaved rope (F1).
2. `topk_indices(logits, k=2048) -> indices` — TILE in (corrected agreement 4), out [1,B,Sq,k] uint32, padded with last valid where Skv<k. Workaround: ttnn.topk; host fallback. K cache format untouched (agreement 3).
3. `sparse_mla(q, kvpe_cache, indices, scale) -> out` — q [1,H,Sq,576] absorbed; kvpe [1,1,Skv,576]; indices [1,B,Sq,2048]; out [1,H,Sq,512]; indices replace causal mask (FlashMLA contract). Workaround: gather (embedding-style) + chunked dense SDPA; stub: dense ring MLA (valid Sq≤2048). **API learning (e2e seq4k, PCC 0.20→fixed):** rows with <k causal keys receive arbitrary future indices from topk's -inf band and scores are recomputed from latents, so the op itself must drop index > row_pos (FlashMLA solves via per-row topk_length; fused op must too). Chunked prefill needs the start_pos offset here.

Shape tests are the first deliverable per op; numerics vs reference_cpu after.

### Approach to missing ops
When no op exist try to **0. define an API (inputs/outputs)** and
1. create a workaround by composing existing ops
2. fallback to CPU implementation
3. implement stub op that emits a warning and returns random/zeroes/ones tensor in the expected format.
4. Proper implementation of c++ ops is out of scope. That's follow-up that should be documented.

## Long-running tasks
Track every step that takes minutes — each is either a bug risk (silent hangs, stale state) or a caching/optimization opportunity. Add measured times as we collect them.

| Task | When | Time | Mitigation / caching |
|---|---|---|---|
| First e2e MLA test run (mesh init + fabric + weight upload, no output until end) | every fresh pytest | measured: 472s seq2k (cold CPU ref incl.), ~40s seq256 | pytest -s for live progress; track time per stage; weight cache reuses v3 build_ttnn_cache |
| CPU reference forward (uncached) | per (tag, seq, seed); 128 heads + 64-head indexer | measured: ~7 min seq2k, 48 min seq4k — quadratic; 50k+5k infeasible cold | disk cache /tmp/deepseek_v32_mla_ref_cache (env DEEPSEEK_V32_MLA_REF_CACHE) — keep tag/seed stable; 50k+5k truth must be cached once on a big box |
| ttnn incremental rebuild after .so staleness | after rebase/pull | ~3 min observed | rebuild ninja -C build ttnn (target is "ttnn", not "_ttnn"); fresh lib lands in build/ttnn/_ttnn.so |
| HF config-only download | first run / new variant | seconds-min, network | already cached by v3 conftest |
| Pre-commit hooks (isort/black, EOF fixer) | every commit | tens of sec | don't partial-stage — keep index clean or hooks loop on fix-rollback |

## Testing
- Primary goal: prefilled 50k cache, 5k chunk
- add determinism tests
- add accuracy tests that should match CPU reference (CPU reference outputs should be cached somewhere to speed-up testing)
