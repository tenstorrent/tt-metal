# Migration log — v3.2 / GLM-5.1 onto v3.1 infra

Running log of what was **done**, what was **learned**, and the **decisions** made while
executing the plan in [`migration.md`](migration.md). Newest entries on top. This is for
later inspection — keep it append-only and dated.

Branch: `mvasilijevic/dsa_w_ops`.

---

## Backlog (next steps — highest priority first)

The MLA-layer test migration (P0 + P1) is **done and device-verified** (see Progress / Success criteria
below). Everything here is follow-on, ordered by what to do next. Items B1–B3 came out of the PR #47832
review (`context/pr_47832_review_findings.html`) + this session's verification.

1. **B1 — Triage the sparse-regime chunked-indexer divergence (NEW, cheap, possible bug).** Extending
   `test_indexer_chunked` from `seq=2048` (== index_topk, trivial all-select) to `seq=4096` (real
   2048-of-N selection) FAILS: `row 2833: overlap 2025/2048` (~1.1% chunked-vs-single-shot top-k
   divergence, just over the 1% tolerance). The seq=2048 test passes only because it never enters the
   sparse regime. Check whether the chunked vs single-shot indexer K-cache is bit-identical (→ near-tie
   numeric noise, safe to widen seq + tolerance) or drifts (→ a real chunked indexer-cache bug). Do this
   before extending the test. (Reverted the seq bump to keep the suite green.)
2. **B2 — Cache-only load silently disables the indexer (review finding 1; highest severity).** A
   cache-only run of a V3.2/GLM layer (`ttMLA(config, {}, weight_cache_path=...)` via the prefill runner)
   gets `_has_indexer=False` (it's `bool(idx_host)` popped from the live `state_dict`) → silently runs
   dense for `end_pos > index_topk`. Worse, `check_cache_complete` (`mla.py:34`) and the offline builder
   (`transformer_helpers.py:590`) omit `INDEXER_WEIGHT_NAMES`, so the cache is reported complete without
   indexer files. Fix = (a) add `INDEXER_WEIGHT_NAMES` to the completeness check + builder, (b) a
   cache-only `TtIndexer` construction branch, (c) a config-level indexer signal — note DSV32 reuses R1's
   config (no indexer marker; dims via getattr defaults), so "detect from config" needs that marker added.
   Cache/runner subsystem — do BEFORE any V3.2/GLM e2e/serving work, NOT in the test-migration PR.
4. **B4 — Per-user indexer slot isolation (review finding 2 + KISS-3 cache half).** `_index_kbuf` is one
   layer-global buffer grown by a private `ttnn.concat`; `write_k`/`forward` take no `cache_user_id` while
   the KVPE write path does. Interleaved multi-user requests cross-contaminate. Needs a per-slot indexer key
   cache reusing the slot-aware cache machinery (the block-cyclic redesign in the `indexer.py` FOLLOW-UP
   comment) — the review correctly notes this shared cache primitive also underlies B2/B5. Before multi-user serving.
5. **B5 — Honor `actual_end` / absolute positions in the indexer (review finding 3).** `write_k` concats
   the full physical chunk (incl. pad); routing uses physical length, not `actual_end`. Partial / rotated
   chunks append pad rows. This is the unresolved "how padding works" question. Before partial-chunk serving.
6. **B6 — CI-wire `test_vs_gpu_ref`** (the gold-standard official-GPU/vLLM gate). Needs the trace bundles
   staged on the CI runners (they live on exabox). Add a skip-gated job that flips on once staged.
7. **B7 — DSA block / full-model e2e parity** for V3.2/GLM (today only the MLA layer; v3.1 has block + e2e).
8. **B8 — CI perf gate for the DSA path** (promote a trimmed `test_mla_perf` off `@skipif(CI)` with a
   regression threshold; review finding T3: the perf harness stubs `_index_kbuf` directly → add a
   real-prefix-warmed profile mode so it can't look healthy while the prefix path is broken/slower).
9. **B9 — Push the branch** to `origin/mvasilijevic/dsa_pr` (not pushed; review is happening off local commits).
10. **B10 — Unify the CPU-reference HF shard resolver (review KISS-2; low/med, pre-existing).**
    `cpu_deepseek_v32/weights.py` calls `hf_hub_download` directly, a second local-vs-HF resolution policy
    beside the test conftest's resolver. Fold into one shard resolver that can return *attention-only*
    per-layer shard paths for a variant/repo/layer (keep CPU weight *mapping* in `weights.py`). Note: this
    is pre-existing reference code (relocated, not introduced here) and the existing resolver doesn't do
    attention-only per-layer loading, so it's a real refactor, not a trivial dedup.


*(Completed cleanups B3 + B11 moved to Progress below.)*

---

## Success criteria (test_spec.md) — status

All three **MET** (the criteria are scoped to the MLA layer; the backlog items are out-of-scope follow-ons).

- **P0 — onboarded; MLA-layer coverage matches v3.1 (accuracy/determinism/feature/mesh), DSA checks bespoke.**
  ✅ `test_dsa_mla` (accuracy+mesh), `test_dsa_mla_determinism`, `test_dsa_mla_chunked` (feature) over
  [deepseek_v32, glm_5_1], device-verified (DS/GLM output ~0.997–0.999, KVPE ~0.9999, determinism pcc=1.0).
  `test_vs_gpu_ref` (official-GPU parity) preserved as the bespoke suite.
- **P0 — onboarding a new model = TestVariant + capabilities, no new per-model test code.** ✅ The DSA test
  bodies branch on capabilities (`has_indexer`/`tp_cap`/`config_builder`/`cpu_model_args`/`reference_kind`),
  never on name; GLM rode in as a second entry in the `variant` parametrize. A 3rd sparse model needs no new
  test code. (The `test_dsa_*` functions are per-*capability*, shared across all DSA models — not per-model.)
- **P1 — `test_coverage_overview.md` carries the analysis (groups, real budget, dev-vs-gate, sparsification,
  un-portable + plan).** ✅ Rewritten to the delivered state; `.html` companion rewritten succinct + self-sufficient.

**Caveat (honest):** the criteria measure the *MLA-layer* migration, which is fully met. They do NOT cover the
production-serving paths the review flagged (cache-only, multi-user, partial-chunk) — those are real gaps
(B2/B4/B5), correctly outside P0 scope and documented in the backlog. B1 also shows the chunked-equivalence
coverage has a hole (it doesn't test real sparse selection) — current tests pass, but the coverage is thinner
than "matches v3.1" implies for the sparse regime until B1 is triaged.

---

## Decisions

### D1 — CCL: keep collectives inlined, do NOT factor into helpers (2026-06-22)
The team prefers inline collective code over helper functions, matching v3.1's style.
- **Implication:** the convergence direction is *v3.2 adopts v3.1's inline pattern*, NOT
  "lift v3.2's helpers into the v3.1 base class" (the originally-proposed refactor is
  rejected).
- **Action item (G1.2):** inline v3.2's `_tp_rs_ag` / `_tp_ag_reduce` / `_sp_all_gather`
  at their call sites so v3.2 reads like v3.1. Accept the resulting inline duplication —
  it is the preferred convention here.
- Note the one nuance when inlining: the AG+reduce block (`all_gather_async` dim=1 +
  `fast_reduce_nc` dims=[1]) must stay guarded by `if self.tp_factor > 1:` (no internal
  tp==1 short-circuit), whereas the RS+AG / RS-only blocks no-op naturally at tp==1.

### D2 — RoPE: defer the freq-table unification as its own task (2026-06-22)
Freq tables are unifiable (see L1), and for the *current* configs the scale handling is
numerically equivalent — so the risk is lower than first thought. Still tracking it as a
separate G1.1 follow-up rather than folding it into the low-risk CCL pass, because it
touches a correctness-sensitive convention and needs a PCC re-validation + a decision on
the `mscale != mscale_all_dim` case (GLM-5.1, resolved in L3).

### D3 — G1.1-indexer plan: parameterize v3.1 rope tables, keep the indexer apply (2026-06-22)
How to put the DS-V3.2 / GLM indexer RoPE on v3.1 infra. The rope splits into a
**shareable table/op part** (fold into v3.1 behind one flag) and an **indexer-specific
apply part** (stays). mscale needs nothing — confirmed identical (L1/L3); it lives in
`self.scale`, which the indexer already reuses.

**Share — parameterize v3.1's rope helpers (small, mechanical):**
1. `get_cos_sin_matrix(hf_config, interleave=True)` (`deepseek_v3_d_p/tt/mla/rope.py:25`):
   `interleave` toggles the cos/sin arrangement — `stack`→duplicated pairs `[c0,c0,c1,c1,…]`
   (`rope.py:51`) vs `cat`→concatenated halves `[c0,…,c31,c0,…]`. Note the HF
   `DeepseekV3YarnRotaryEmbedding` already produces the rotate-half (`cat`) form internally
   and `rope.py:50-51` converts it to interleaved — so the flag just *skips that conversion*.
   No new math. (Also add `scale_handling` per the pending decision; for current configs
   `_mscale==1` so tables are pure either way.)
2. `RotarySetup.get_rope_tensors(…, interleave=…)` (`rope.py:78`): build `trans_mat` ONLY
   when `interleave` (rotate-half passes no trans_mat).
3. Apply-op selection: `rotary_embedding_hf` (rotate-half, no trans_mat) vs
   `rotary_embedding_llama` (interleaved, + trans_mat). `rotary_embedding_hf` is NEW to v3.1
   (v3.1 only calls `_llama` today) but is a stock ttnn op.
   → these three flip *together*; that triple is exactly what `index_rope_interleave`
   gates today (`mla.py:105-110` stack/cat, `:121` trans_mat, `:182-187` op).

**Keep indexer-specific — `_device_rope_pe` (`deepseek_v32/tt/mla/mla.py:168`):** NOT a flag.
It ropes only the first 64 of a 128-wide `index_head_dim`, on `[1, H_idx, S, D_idx]` index
tensors, SP-mesh-partitions cos/sin for the SP-sharded queries (query-sharded / key-replicated
asymmetry), slices cos/sin to global `[start_pos, start_pos+glob)`, and concats the nope back.
v3.1's MLA apply (`_apply_rope_one_shot`/`_apply_rope_padded`) doesn't cover these shapes/sharding,
so this apply wrapper stays.

**Reference-parity constraint (why the flag must exist, from the indexer-cache finding):**
DS indexer must stay rotate-half (`rotary_embedding_hf`) — that is what makes the stored
indexer k match the GPU/vLLM reference element-wise (PCC ~0.99999). Forcing v3.1's
interleaved path on the DS indexer would store k in the interleaved frame and drop raw k_pe
PCC to ~0.43 (the harmless-but-mismatching MLA k_pe frame artifact, reborn) — output stays
correct (q·k is layout-invariant) but element-wise reference parity breaks. GLM uses
interleaved, so it matches v3.1 natively. Hence `interleave` is load-bearing, not cosmetic.

**Verify:** `test_vs_gpu_ref.py` covers both models; assert k_pe via per-row L2-norm (not raw
PCC) per the kpe-rope-frame note, and check indexer logits/topk + MLA output for correctness.

### D4 — G1.3 plan: unify the v3.2 ttMLA into v3.1's ttMLA (chosen: single class) (2026-06-22)
User picked **unify into v3.1's ttMLA** (eliminate the v3.2 subclass). Inventory of what the
v3.2 subclass adds (14 methods): 6 indexer (`_build_index_rope_tables` [done, shared],
`_upload_indexer_weights`, `_device_rope_pe`, `_chunk_offset`, `_indexer_write_k`,
`_indexer_topk`), 3 DSA forward (`_dsa_forward`, `_gather_kvpe_prefix`, `_unrotate_blockcyclic`),
2 overrides (`__init__`, `forward`), 3 CCL helpers (`_tp_rs_ag`, `_tp_ag_reduce`, `_sp_all_gather`).

**Dependency analysis (no circular import).** `deepseek_v32/{__init__,tt/__init__}.py` are empty;
`tt/ops.py` and `reference_cpu/model.py` import only torch/ttnn. So v3.1 can import/own the DSA
ops and `ModelArgs` with no cycle. The end-goal (v3.1 owns everything, v32 folder retired) means
the ops + index config should live under `deepseek_v3_d_p`.

**Phasing (each phase its own verified commit):**
- **P1 — relocate `ops.py` → `deepseek_v3_d_p/tt/ops.py`** with a re-export shim at the old path
  (incl. private `_to_host`/`_chunk_offset_tensor`). v3.1 now owns the DSA ops. Verify: ops tests.
- **P2 — merge methods into v3.1 ttMLA**, gated by `_has_indexer` (set when indexer weights are
  present). Dense v3.1 path provably untouched (all new code behind the gate). CCL stays inline
  per D1 (the 3 helpers move as-is for now; inline cleanup optional).
- **P3 — forward dispatch:** v3.1 `forward` routes to `_dsa_forward` when
  `_has_indexer and end_pos > index_topk`, else the existing dense body. Reconcile the
  `forward` signatures (v3.2 `actual_start/actual_end/**kwargs` vs v3.1 `cache_layer_idx/...`).
  THIS rewrites v3.1's production forward — highest-risk phase, review before merge.
- **P4 — retire the subclass:** `deepseek_v32/tt/mla/mla.py` → thin re-export of v3.1's ttMLA
  (+ keep `interleaved_to_halfsplit_perm`); `__init__.py` re-exports. No call-site churn.
- **P5 — config cleanup (ties to the line-19 TODO):** express the indexer config via the HF
  `PretrainedConfig` instead of `ModelArgs`, so v3.1 no longer imports `reference_cpu`. Then the
  inverted dep is fully gone.

**Verify each phase:** `test_vs_gpu_ref.py` (DS + GLM) must stay green; add a v3.1 dense smoke
test (e.g. `deepseek_v3_d_p/tests/test_mla.py`) to confirm the dense path is unchanged.

### D5 — DSA op wrappers folded into call sites; op-wrapper tests removed (2026-06-23)
`tt/ops.py` (`indexer_program_config` / `indexer_logits` / `topk_indices` / `sparse_mla`) was a thin
wrapper layer. Decision: inline it — indexer ops into `TtIndexer.forward` (direct
`ttnn.experimental.indexer_score` / `topk_large_indices`), `sparse_mla` into `ttMLA._sparse_mla`.
`test_ops_shapes.py` / `test_ops_numerics.py` deleted as redundant: the raw ttnn kernels have their own
unit tests (`test_indexer_score`, `test_sparse_sdpa`, `test_topk_large_indices`) and the wrappers are
exercised e2e by `test_mla` / `test_vs_gpu_ref`. `ops.py` no longer exists; `test_vs_gpu_ref`'s
logits-capture monkeypatch was repointed to `ttnn.experimental.topk_large_indices`.

### D6 — TtIndexer decoupled from ttMLA (constructor DI + inlined collectives) (2026-06-23)
The indexer was extracted from ttMLA into `tt/mla/indexer.py` as `TtIndexer`, then fully decoupled: no
back-reference — ttMLA injects everything it reuses through the constructor (config, mesh + sp/tp axes,
scale, the q_a stem weights + a `q_a_mm_kwargs(seq_len)` callable, compute configs, weight-cache
path/layer, tt_ccl handles). The indexer runs its **own** inlined TP/SP collectives (consistent with
D1 — no collectives shared between dense and indexer). Entry renamed `topk` → `forward`; `write_k` stays
public (ttMLA calls it on dense chunks to keep the key-cache warm).

### D7 — Unify dense + sparse MLA forward into one `forward(mode)` (2026-06-23)
`_dense_forward` + `_dsa_forward` collapsed into a single `forward` with shared helpers (`_q_stem`,
`_kv_stem`, `_apply_wkv_b2`, `_write_kvpe`, `_o_proj_epilogue`) and an explicit 4-mode branch
(`dense_single` / `dense_chunked` / `sparse_single` / `sparse_chunked`) + `else: raise`. Caveat
resolutions are recorded in `context/mla_forward_unification.html`: CCL inlined into the shared stems
(`_tp_rs_ag` / `_tp_ag_reduce` deleted), per-mode q-norm `memory_config` (DRAM for sparse, tuned for
dense — TODO to converge), one `_apply_wkv_b2` placed per-mode, `kv_only` + `return_kv_intermediates`
preserved. Net −125 lines in `mla.py`.

### D8 — Onboard models by capability flags on `TestVariant`, not by `name` branches (2026-06-24)
Per the signed-off `context/test_spec.md`, a model is data on `TestVariant`, and test bodies branch on
capabilities. Kept the capability set **minimal** — only what is NOT already on the resolved config:
- `has_indexer` (bool) — runs the DSA indexer + sparse SDPA. (On device the ttMLA still auto-detects the
  indexer from the weights via `_has_indexer`; this flag is for *test-body* branching.)
- `tp_cap` (int|None) — GLM's 64 q-heads force sparse_sdpa's per-chip `H/tp >= 32`, i.e. tp ≤ 2. DS = None.
- `config_builder` (callable|None) — GLM's `glm_moe_dsa` is not registered with transformers, so AutoConfig
  cannot load it; when set this overrides conftest's disk/HF config resolution. DS = None (rides AutoConfig).
- `reference_kind` ("hf_attn"|"mlacpu") — the dense `run_reference_mla` HF path is only correct for
  `seq_len <= index_topk`; above that the sparse regime needs the `deepseek_v32.reference_cpu` `MLACPU`
  truth. Both DSA variants are "mlacpu".

Index dims (`index_n_heads/head_dim/topk`, `index_rope_interleave`) are **read off the resolved config**, not
duplicated onto the variant — single source of truth (the unified indexer already reads them via getattr).

Two structural facts that shaped this (from the integration map):
- **DSV32 ≠ new config class.** No `DeepSeekV32Config` exists and V3.2-Exp shares MLA dims with R1, so DSV32
  resolves config from the R1 checkout; the indexer's getattr-defaults already match DeepSeek. hf_repo points
  at R1, sidestepping the V3.2-Exp remote-code AutoConfig uncertainty.
- **MLACPU does not fit `run_reference_mla`.** Different ctor (`ModelArgs`, not HF config + layer_idx),
  reference weight keys, a `(x, start_pos, freqs_cis, mask)` forward, and a single-tensor return. So the
  sparse-regime truth is wired via `reference_kind="mlacpu"` (an adapter / direct MLACPU driver), NOT by
  forcing MLACPU through the HF attention interface. The dense regime (`seq <= index_topk`) can still use the
  config-driven `create_mla_reference` that already serves Kimi.

### D9 — DSA test substrate: shared CPU reference, shared plugin, reference relocated (2026-06-24)
How the v3.2/GLM MLA tests reuse the v3.1 infra without per-model test code or backwards deps.
- **Reference truth = MLACPU, not the dense reference.** `random_weights`/`pretrained` are dense (7 keys,
  no indexer) and `create_mla_reference`/`cpu_mla_reference` attend to *all* causal positions — so they
  are WRONG once context exceeds `index_topk` (device attends to top-k only). The sparse truth is the
  `reference_cpu` MLACPU (indexer + {0,-inf} index mask). Extracted the working v32 recipe
  (`build_cpu_reference`/`run_cpu_reference[_chunked]`/`WEIGHT_NAME_MAP`/`assert_config_matches`) into
  `tests/dsa_reference.py`, made variant-driven (ModelArgs via `variant.cpu_model_args`). Device weights
  and CPU truth come from the SAME MLACPU instance (remapped via `WEIGHT_NAME_MAP`) → PCC is meaningful.
  The DeepSeek-only YaRN assert is dropped (GLM sets max==original to disable YaRN by design).
- **Shared options/markers via a plugin, not a conftest.** v3.1 and v3.2 `tests/` are SIBLINGS, so neither
  conftest auto-applies to the other. Putting the `--ds-*` options + dev/gate/nightly markers in a conftest
  and loading it from the other via `pytest_plugins` errors ("already registered under a different name")
  because pytest also auto-loads it under its path name. Fix: a standalone `tests/dsa_plugin.py` (NOT a
  conftest), loaded by both conftests via `pytest_plugins` under its dotted name → registered exactly once.
  This is also the unification of v3.1 (`-k`) and v3.2 (dev/gate/nightly) onto one scheme.
- **config_builder hook.** `_resolve_config_only` returns `variant.config_builder()` first — GLM's
  `glm_moe_dsa` can't load via AutoConfig, so GLM resolves through the hand-built `glm_hf_config`.
- **Reference relocated (user request).** `reference_cpu/` moved from `deepseek_v32` to
  `deepseek_v3_d_p/reference/reference_cpu/`. It is self-contained (imports neither package), and the
  unified model already lives in v3_d_p — so the v3.1 test infra no longer reaches back into v32. The v3.1
  `reference/` dir is now the single home for the family's references; v32 keeps only its bespoke trace tests.

---

## Learnings

### L1 — RoPE Details: freq tables unifiable; scale handling reconciled (2026-06-22)

**Question:** the freq tables look unifiable — is the difference only in how scale (mscale)
is applied? How do the v3.1 and v3.2 *reference* implementations handle scale?

**Frequencies — identical.** Both compute the same YaRN-rescaled base frequencies:
- v3.1: `DeepseekV3YarnRotaryEmbedding` (`deepseek_v3/reference/modeling_deepseek.py`),
  `inv_freq = freq_inter*(1-mask) + freq_extra*mask`.
- v3.2: `precompute_freqs_cis` (`reference_cpu/utils.py:215`),
  `freqs = freqs/factor*(1-smooth) + freqs*smooth`.
- The ramp is the same function (`mask == smooth`), same blend. Given equal
  `beta_fast`/`beta_slow`/`factor`/`original_seq_len`, the frequencies match.

**Amplitude / mscale — this is the only real difference, and it cancels for current configs.**

`yarn_get_mscale(scale, mscale) = 0.1 * mscale * log(scale) + 1.0` (same formula both sides).

- **v3.1 (HF convention)** splits mscale between the RoPE tables and the softmax scale:
  - cos/sin scaled by `_mscale = yarn_get_mscale(s, mscale) / yarn_get_mscale(s, mscale_all_dim)`
    (`modeling_deepseek.py:302-303`).
  - `softmax_scale *= yarn_get_mscale(s, mscale_all_dim)**2`, only when `mscale_all_dim` is set
    (`:704-708`).
- **v3.2** keeps cos/sin **pure** (unit-magnitude `torch.polar(ones, freqs)`) and folds *all*
  of mscale into the softmax scale:
  - `softmax_scale = qk_head_dim**-0.5`; if `max_seq_len > original_seq_len`,
    `softmax_scale *= mscale_term**2` where `mscale_term = 0.1*args.mscale*log(rope_factor)+1`
    (`reference_cpu/model.py:354-358`). The indexer uses the same `softmax_scale`
    (`reference_cpu/model.py:189`, applied at `:270`/`:284`).

**The two DeepSeek references differ structurally (verified against genuine configs).**
- HF `DeepseekV3` (`deepseek_v3/reference/config.json`): has BOTH `mscale` and
  `mscale_all_dim` (both `1.0`, factor `40`) → splits mscale across cos/sin and softmax.
- Official DeepSeek inference style (v3.2 `reference_cpu`, `ModelArgs`): has NO
  `mscale_all_dim` field at all — single `mscale = 1.0`, factor `40`, pure cos/sin,
  `softmax *= m(mscale)²`.
- (Earlier I'd grounded this on `kimi_k2_6/config.json` — Kimi-K2, not DeepSeek; the genuine
  DeepSeek-V3 config confirms the same `mscale == mscale_all_dim == 1.0`.)

**Why they coincide for the configs in play.** With `mscale == mscale_all_dim == 1.0`:
- v3.1's `_mscale = m/m = 1.0` → cos/sin **pure** (unscaled), and the full term goes to
  `softmax_scale *= m**2`.
- That is **exactly** v3.2's behaviour. → numerically identical.

**Where they would diverge.** Only if a config sets `mscale != mscale_all_dim`. Then v3.1
puts a nonzero `_mscale` ratio into cos/sin (scaling the rope component of q·k by
`_mscale**2`) while v3.2 cannot represent that split (cos/sin always pure). The current DS
configs don't exercise this; **GLM-5.1 needs to be checked** before relying on it.

**Consequence for unification (G1.1).** A single freq-table builder is viable. Pick one of:
1. Adopt v3.2's "pure cos/sin + all-mscale-in-softmax" and assert `mscale == mscale_all_dim`
   (simplest; correct for DS-V3.2; verify for GLM).
2. Carry HF's general split (handles `mscale != mscale_all_dim`).
Either way, re-validate MLA + indexer PCC after the switch.

**Already-deduped RoPE (not pending):** v3.2 already imports and reuses
`get_rot_transformation_mat` and `RotarySetup` from v3.1. The remaining v3.2 RoPE code
(`_build_index_rope_tables`, `_device_rope_pe`) is the *indexer* RoPE — genuinely new
functionality, not a v3.1 duplicate.

---

### L2 — v3.1's device scale already assumes mscale == mscale_all_dim (2026-06-22)
Tracing the *device* (TT) path, not just the HF reference:
- **Table build** (`tt/mla/rope.py:84` → `get_cos_sin_matrix` `:25` → `DeepseekV3YarnRotaryEmbedding`)
  bakes `_mscale = m(mscale)/m(mscale_all_dim)` into cos/sin (`modeling_deepseek.py:302-303`).
- **Softmax scale** (`tt/mla/mla.py:279-282`) sets `self.scale = qk_head_dim**-0.5`, then
  `*= mscale**2` using `rope_scaling["mscale"]` and **ignoring `mscale_all_dim`**.
- These two would double-count mscale unless `_mscale == 1`, i.e. `mscale == mscale_all_dim`
  (true for the real config — both `1.0`). So v3.1's device path is *effectively Mode A*
  (pure tables + full mscale² in softmax); the `_mscale` baking in `get_cos_sin_matrix` is
  dead weight that only equals 1 by config coincidence.
- **Consequence:** switching `get_cos_sin_matrix` to pure tables (Mode A) is *more*
  self-consistent with the existing device `self.scale`, not less. v3.2's `precompute_freqs_cis`
  is already pure; v3.2's reference softmax (`reference_cpu/model.py:354-358`) uses the
  identical `scale *= m²` formula. The only unmerged table builder is the indexer's
  `_build_index_rope_tables`.

### L3 — GLM-5.1 config resolves the scale_handling risk; fixes the Mode-A guard (2026-06-22)
Checked `tests/test_vs_gpu_ref.py` (`_glm_hf_config`, `_glm_model_args`) and
`context/GLM_5_1_TRACE.md`. GLM-5.1 rope_scaling: `factor=1.0, mscale=1.0,
mscale_all_dim=0.0`, `rope_theta=1e6`, `max_seq_len==original_seq_len` → **YaRN OFF**.
- GLM sets `mscale (1.0) != mscale_all_dim (0.0)` — but `factor=1.0` makes
  `yarn_get_mscale(1.0, ·) = 0.1·mscale·ln(1)+1 = 1.0` for ANY mscale. So `_mscale` ratio
  = 1 (pure cos/sin) and the softmax mscale² term is skipped on both sides → plain RoPE,
  `scale = qk_head_dim**-0.5`. **Mode A is correct for DS-V3, V3.2, AND GLM-5.1.** No
  config in play needs Mode B.
- **Bug caught:** the Mode-A guard must be `yarn_get_mscale(factor, mscale) ==
  yarn_get_mscale(factor, mscale_all_dim)` (i.e. `_mscale == 1`), NOT
  `mscale == mscale_all_dim` — the naive form false-fires on GLM (1.0 vs 0.0).
- Latent inconsistency to keep in mind: device guards scale on `rope_factor > 1.0`, HF
  guards on `if mscale_all_dim:`. Agree for DS + GLM; a `factor>1 & mscale_all_dim=0`
  config would diverge. The `_mscale == 1` assertion catches it.
- Other GLM specifics (not scale-related): `qk_nope/qk_rope/v = 192/64/256`, indexer RoPE
  **interleaved** (`index_rope_interleave=True`), `tp <= 2` (H=64/tp ≥ 32), hand-built
  config (transformers can't load `glm_moe_dsa`).

### Pending decision — `get_cos_sin_matrix` mscale param  → IMPLEMENTED + DEVICE-VERIFIED
Shipped as `bake_mscale: bool = False` (local-context name: "fold mscale into cos/sin?").
- `False` (default) → pure rotations; mscale applied in the attention softmax scale (`self.scale`).
- `True` → cos/sin × `_mscale = m(mscale)/m(mscale_all_dim)` (HF split; only needed if
  `mscale != mscale_all_dim`).
(Earlier iterations named this `scale_handling="softmax"/"baked"` then `mscale_in="softmax"/"tables"`;
renamed to a local boolean per review — the old names referenced the *external* softmax, not what
this function does to the tables.) Default `False` forces `mscale_all_dim = mscale` so `_mscale = 1`
— can't double-count; no assert needed.

## Progress

### 2026-06-24 (cont.) — backlog cleanup: B3 (retire dup) + B11 (stale README) DONE
Low-risk removal cleanups from the PR #47832 review.
- **B3 (review KISS-1) — deleted `deepseek_v32/tests/test_mla.py` entirely.** Its three tests
  (`test_v32_mla_vs_cpu_reference` / `_determinism` / `_chunked_vs_cpu_reference`) are fully subsumed by the
  variant-driven `test_dsa_mla*` in v3.1 (which cover deepseek_v32 + glm_5_1), and its DeepSeek-only helper
  copies duplicated `dsa_reference.py`. Repointed the three importers (`test_vs_gpu_ref`,
  `test_indexer_chunked`, `test_mla_perf`) to `dsa_reference` — they now pass the `deepseek_v32` variant to
  the variant-driven `build_cpu_reference` (identical default ModelArgs → behavior-preserving). The autouse
  `use_v32_mla` monkeypatch went with the file (it was a no-op: `ttMLAv32 is ttMLA` post-unification).
- **B11 (review KISS-4) — rewrote `deepseek_v32/tt/README.md`.** It described a v32 ttMLA *subclass* + copied
  `tt_prefill_block`/`tt_prefill_transformer`, none of which exist post-unification. Now a short accurate note
  (thin re-export overlay; bespoke trace tests here, MLA-layer tests in v3.1, reference in cpu_deepseek_v32).
- Verified: all suites collect (1121 items, v32 test_mla.py gone); `test_indexer_chunked` passes on device
  through the repointed helpers. Rejected backlog item (KISS-3 collective helper, conflicts with D1/D6) removed.

### 2026-06-24 (cont.) — test migration COMPLETE: group markers (#3) + coverage doc (P1)
All signed-off `test_spec.md` deliverables landed. Commits `afef5ba861d` (#3), `2ba4510719b` (P1).
- **#3 group markers:** registered accuracy/determinism/feature_chunking/feature_cache/mesh/perf in
  `dsa_plugin` (orthogonal to dev/gate/nightly); applied to the v3.1 DSA tests + the bespoke v32 suites.
  Verified `-m accuracy` / `-m feature_chunking` select correctly; strict-marker collection clean (1136).
- **P1 coverage doc:** rewrote `test_coverage_overview.md` to the delivered state — group→parameter table,
  tier map, live CI jobs, measured per-test budget; pruned the stale plan/projection. `.html` status → DONE.

**P0 (#1 variants, #2 test wiring, #3 markers, #4 CI) + P1 all done and device-verified.** Open follow-ups
(non-P0, see coverage doc): retire the now-redundant v32 `test_mla.py` CPU-ref tests; sparse partial-chunk
padding; DSA block/e2e; CI-wire `test_vs_gpu_ref` (needs exabox traces); CI perf gate. Branch not yet pushed.

### 2026-06-24 (cont.) — test migration 2c/2d: DSA tests wired into v3.1 + CI jobs
The actual deliverable: variant-driven DSA tests live in v3.1 `test_mla.py`, device-verified, with CI jobs.
Commits `ec47ec6df7c` (2c), `c8b84233a15` (2d), CI yaml (this entry).
- **2c — accuracy + determinism** (first TODO anchor): `test_dsa_mla` (output + KVPE vs MLACPU over
  {seq256(dev),2k,4k} × box-adaptive SP×TP) and `test_dsa_mla_determinism`, both over [deepseek_v32, glm_5_1].
  `mesh_utils` moved into v3.1 + `skip_if_tp_exceeds_cap` (GLM tp<=2). Device-verified on the 8-chip box:
  DS seq256 0.999 / seq4k 0.997 (bands 0.999/0.992); GLM seq256/2k/4k 0.999/0.998/0.997; KVPE ~0.9999;
  GLM tp=4 skipped by tp_cap; determinism exact=True pcc=1.0.
- **2d — chunked** (second TODO anchor): `test_dsa_mla_chunked` (per-chunk + full + un-rotated KV vs the
  MLACPU *chunked* truth). The 50k-production `test_mla_chunked_prefill` is deliberately NOT extended to
  sparse (its dense refs are wrong above topk; its scenarios exceed the MLACPU window) — documented inline;
  the sparse partial-chunk padding stays an open follow-up. Verified: DS/GLM per-chunk 0.998→0.988 (the drop
  past index_topk is the expected sparse signature), full 0.997, kvpe 0.9999.
- **CI (deliverable #4):** the DSA tests use random weights + on-the-fly MLACPU truth + R1 config (already on
  CI) / hand-built GLM config → **no staged V3.2/GLM weights or traces needed**, so the jobs actually RUN
  (not skip-gated). Hardware mapping falls out of `tp_cap`: Galaxy (8x4, tp=4) runs deepseek_v32 (GLM skips);
  GLM (tp<=2) is covered on the LoudBox `sp4xtp2`; DS also gets the asymmetric `sp2xtp4` there. Added
  `(Galaxy) DeepSeek Prefill - DSA MLA` + LoudBox `bh-lb-deepseek-dsa`. `-m gate` selects accuracy+chunked,
  drops dev determinism. yaml parses; LB selections collect 4 tests each and pass on device.

Remaining: deliverable #3 (group markers: accuracy/determinism/feature/mesh) + P1 (`test_coverage_overview.md`
update). Then the v32 `test_mla.py` CPU-reference tests are redundant with the v3.1 ones and can be retired.

### 2026-06-24 (cont.) — test migration increments 2a/2b + DSA reference relocation (D9)
Substrate landed; no v3.1 DSA test bodies yet (those + on-device verify come next). Commits
`1ecc830f507` (2a), `0ad375ea458` (2b), `3a3e182c0b1` (reference move).
- **2a:** `tests/dsa_reference.py` — variant-driven MLACPU CPU-reference substrate (see D9). `glm_model_args()`
  joins `glm_hf_config()` in `glm_5_1_config.py`; `test_vs_gpu_ref.py` delegates to both. `TestVariant` gains
  `cpu_model_args`. CPU-verified: builds all 5 indexer keys with correct shapes for DS (idx 64 heads) + GLM
  (32 heads, dims 6144/256).
- **2b:** `_resolve_config_only` config_builder hook; `tests/dsa_plugin.py` shares the `--ds-*` options +
  dev/gate/nightly markers across both sibling suites (see D9). v32 conftest reduced to star-import + plugin.
  Verified: v32-only (15) and both-dirs-together (1103) collect, no `--ds-layer` collision; `-m dev` honored.
- **reference move (D9):** `reference_cpu` → `deepseek_v3_d_p/reference/reference_cpu`, all importers repointed.
  1113 tests collect across reference_cpu + v3.2 + v3.1.

Next (2c): vendor `mesh_utils.py` into v3.1; add variant-driven DSA test functions (DS + GLM) at the
`test_mla.py` TODO anchors using the substrate; verify deepseek_v32 on the 8-chip box. Then 2d: dispatch
`_run_chunked_prefill`'s `reference=="cpu"` to the MLACPU chunked truth for `has_indexer` variants.

### 2026-06-24 — test migration increment 1: register DSV32 + GLM51 variants (D8)
Signed-off `test_spec.md` P0 deliverable #1 landed. Committed `b03fa1acdf6`.
- `TestVariant` gains the DSA capability set (D8): `has_indexer`, `tp_cap`, `config_builder`, `reference_kind`.
- Registered `DSV32` (R1-config-backed, `has_indexer`, `reference_kind=mlacpu`) and `GLM51`
  (`config_builder=glm_hf_config`, `tp_cap=2`, `supports_pretrained=False`, `reference_kind=mlacpu`).
- DRY: `glm_hf_config()` moved to `reference/glm_5_1_config.py` (single source of truth); `test_vs_gpu_ref.py`
  now aliases it (`_glm_hf_config = glm_hf_config`) instead of carrying a copy — dropped the local `types` import.
- Import-verified: `TEST_VARIANTS` resolves all four; GLM builder + test_vs_gpu_ref delegation load clean.
- **No behavior change yet** — variants are not wired into `test_mla.py` until increment 2, so the DSV3/Kimi
  gates are untouched by this commit.

Next (increment 2): wire the variants into `test_mla.py` at the two TODO anchors with a capability-gated DSA
branch + the `reference_kind="mlacpu"` adapter for the sparse regime; vendor `mesh_utils.py` into v3.1.

### 2026-06-23 (cont.) — unified-forward follow-ups (perf, TODOs, q_a reuse) + step-6 plan
Post-unification cleanup, each committed + device-verified:
- **Perf-check (step 3):** the unified forward runs through the tracy harness at **27.2 ms critical-path
  device-kernel time / 113 op calls** (DSv3.2 MLA chunked, 5120-tok chunk @ 51200 cache, SP4×TP2) — PASSED.
  The unification is op-graph-identical (pure restructuring), so this matches the pre-unify perf.
- **q-norm memory_config (step 4a):** experiment confirmed the sparse path runs fine with the dense tuned
  `_get_act_mem_config` (no L1-OOM, no PCC change) → dropped the per-mode DRAM/tuned split; tuned in all modes.
- **indexer key-cache (step 4b):** reworded the "reuse MLA cache ops" TODO to a scoped follow-up — it is a
  layout redesign (replicated-natural concat → block-cyclic-SP cache + gather/un-rotate read), not a drop-in.
- **q_a latent shared (step 5):** extracted `ttMLA._q_a_latent`; `forward` computes qr once and threads it
  into both the indexer and `_q_stem`. The indexer now holds **zero MLA weights** (qr flows in as data).
  Verified: v32 test_mla 12/3 skipped, indexer_chunked 1, vs_gpu_ref indexer+mla_output 34/8 skipped, 0
  accuracy/dispatch/dealloc failures. (v31 dense failures observed are all CCL-topology env on the 8-chip
  host; dense_single is verified via v32 and dense_chunked shares the same _q_a_latent/_q_stem stem.)

**Step 6 — test-suite migration onto v3.1 (G3.1): PLANNED, deliberately NOT landed.** It modifies the shared
DSV3/Kimi test gates and needs multi-GB v3.2 weights to verify end-to-end (can't satisfy verify-then-commit
here), so per the original G3.1 note ("reviewed follow-up … shouldn't land unattended") it stays a plan:
1. Register `DSV32` + `GLM51` `TestVariant`s in `model_variants.py` — `model_config` = the v3.2/GLM HF config
   (DS via `_resolve_config_only`; GLM via `glm_5_1_config.GLM51Config`), `reference_model_cls`/`reference_attention_cls`
   = `MLACPU` / `IndexerCPU` from `deepseek_v32.reference_cpu` (import-only, no cycle per D4), MLA-PCC thresholds
   from `test_vs_gpu_ref` (LOGITS 0.95, OUTPUT 0.98, KV_LATENT 0.99); GLM caps tp ≤ 2.
2. Wire `test_mla.py` (the two `# TODO: modify this file for v32, GLM` spots): add the variants to the shared
   `@parametrize("variant", …)` and add a DSA-aware branch (gated on `_has_indexer`) for the indexer
   logits/topk + sparse-output + k_pe-frame assertions.
3. Vendor `mesh_utils.py` into v3.1 tests for box-adaptive SP×TP parametrization.
Risk: steps 1–2 touch the DSV3/Kimi gates → must re-run those gates after. Prereq: pre-stage the
DeepSeek-V3.2-Exp / GLM-5.1 HF shards + CPU/GPU truths on the runner.

### 2026-06-23 — refactor wave + forward unification (merge → cleanup → extract → decouple → fold → unify)
Merged latest `main`, which had independently landed the DSA ops: resolved by keeping **ours** for
`indexer_score` (it carries the `chunk_offset` feature; main = bare PR #47223) and taking **main's** newer
`sparse_sdpa`; one latent bug surfaced and was fixed (the no-offset `fill_cb_offset` instantiated an
out-of-range `TensorAccessorArgs` — now templated/dependent so the discarded branch isn't checked). Then a
sequence of behaviour-preserving refactors, each device-verified:
- relocated MLA helpers to their domain homes: `interleaved_to_halfsplit_perm` → `rope.py`,
  `_unrotate_blockcyclic` → `utils.py` as `blockcyclic_to_natural`.
- extracted the DSA indexer into `tt/mla/indexer.py` (`TtIndexer`) and decoupled it from ttMLA (D6);
  renamed its entry `topk` → `forward`.
- folded `tt/ops.py` into the call sites and removed the redundant op-wrapper tests (D5).
- stripped plan/log references from code comments (kept the runtime "migration worker" mentions).
- restored `deepseek_v3_d_p/tests/test_mla.py` to origin/main + only the WIP TODOs.
- unified dense + sparse forward into one `forward(mode)` (D7) + tidy follow-ups (`_write_kvpe`, `is_sparse`).

**Device-verified (8× Blackhole), all four modes, 0 accuracy/dispatch regressions:**
v32 `test_mla` 12 passed / 3 skipped (dense_single + sparse_single + sparse_chunked vs MLACPU);
`test_vs_gpu_ref` mla_output 17 passed / 4 skipped (e2e vs official GPU); `test_indexer_chunked` passed;
v31 `test_mla` dense 14 passed (remaining failures are `fabric2d` topology on the 8-chip host — env, not
regressions; the L1-OOM `seq100k/128k` and 32×4 configs are likewise hardware-bound).

**Next (in progress, this session):** perf-check the unified hot path; resolve the two in-code TODOs
(q-norm `memory_config` convergence; indexer key-cache → MLA cache ops); reuse the `q_a` latent across
MLA + indexer (drops the last indexer→MLA coupling); then **G3.1** test migration — register v3.2/GLM
`TestVariant`s and wire the `test_mla.py` TODOs.

### 2026-06-22 — G3.1–G3.4 ANALYZED: test coverage overview + migration plan
Surveyed both test suites (two Explore passes) and wrote `context/test_coverage_overview.md` covering
all four G3 goals: coverage matrix + gaps (G3.4), migration plan onto v3.1's `TestVariant` registry
(G3.1), what stays bespoke — `test_vs_gpu_ref` trace bundles + `test_mla_perf` hardcoded mesh (G3.2),
CI time budget from measured runtimes (G3.3), and speed-up / coverage ideas.
- **Key finding:** v3.1's `model_variants.py` `TEST_VARIANTS` registry + variant-agnostic conftest is
  the clean extension point; post-G1.3 the v3.2/GLM MLA tests already run against v3.1's unified
  `ttMLA`, so "migration" = register v3.2/GLM as variants + extend `@parametrize("variant", …)`.
- **Decision:** delivered G3 as analysis/plan, NOT a blind code-migration. Registering variants and
  rewiring the shared `@parametrize` lists touches the DSV3/Kimi gates (shared v3.1 infra) — too risky
  to land unattended. Flagged as a reviewed follow-up with concrete steps in the overview doc.
- **Top actionable ideas:** replace the `sparse_mla` host fallback with device `sparse_sdpa` in the
  kv/output device tests (the 4–7-min runtime is the fallback, not the kernel — biggest CI speedup);
  add GLM determinism + chunked tests (biggest coverage gap); tier @dev/@gate/@nightly in the pipelines.

### 2026-06-22 — G2.1 DONE: indexer weights ride the on-disk weight cache
`_upload_indexer_weights` switched from `ttnn.from_torch` to `ttnn.as_tensor(cache_file_name=…)`,
using `weight_cache_path / layer_{idx}.mla.indexer_{name}` — same dir + naming scheme as the MLA
weights in `_convert_and_cache_weights`. Same shardings/dtypes (bf16; wq_b col-parallel via
ShardTensor2dMesh dim=1, wk/weights_proj contraction-sharded dim=0, k_norm/bias replicated), so
numerics are unchanged; it just adds disk caching keyed by layer + cache_path.
- **Decision (scope):** kept this surgical (cache the indexer upload) rather than folding into
  `_convert_and_cache_weights`. Reason: the cache-only LOAD path (state_dict=None) can't detect
  indexer presence — DS v32's HF config carries no index_* attrs (getattr defaults), so the only
  signal is indexer weights in the state_dict. Full integration would need an explicit has-indexer
  flag through the placeholder path (bigger API change). Surgical caching gets the win (CI weight
  caching) at low risk.
- **Follow-up:** `build_ttnn_cache` (device=None pre-build) won't emit indexer files; they cache on
  first real load instead. Documented; revisit if pre-build of indexer weights is needed.
- **Verified:** `test_vs_gpu_ref` indexer device (DS + GLM) PCC unchanged (as_tensor == from_torch
  at cache_file_name=None, the test path); cache round-trip (tmp weight_cache_path) wrote & reused
  5 `layer_0.mla.indexer_*` files; full L0 device suite 15 passed / 3 skipped.

### 2026-06-22 — G1.3 P5 DONE: indexer config from HF config; inverted dep removed
v3.1's `ttMLA.__init__` now reads the 4 indexer constants
(`index_n_heads/index_head_dim/index_topk/index_rope_interleave`) off the HF `config` via
`getattr` with DS defaults (64/128/2048/False), stored in a `types.SimpleNamespace` as
`self.index_args` (keeps the existing `self.index_args.X` accesses unchanged). Removed the
`index_args` __init__ param and the lazy `from ...deepseek_v32.reference_cpu import ModelArgs`.
**v3.1 mla.py now has zero references to the deepseek_v32 package** (verified by source scan) —
the last inverted dependency is gone; closes the `config: PretrainedConfig` + `extend mla config`
TODOs. GLM: added `index_n_heads/head_dim/topk` + `index_rope_interleave=True` to `_glm_hf_config`
and dropped the `index_args=_glm_model_args()` override from `_make_mla` (the CPU reference still
uses `_glm_model_args`). Device-verified: `test_vs_gpu_ref -k "device and L0"` → 15 passed, 3
skipped (GLM tp>2), GLM interleaved rope now sourced from its config (indexer PCC 0.9995, DS 0.959,
MLA output 0.9998). **G1.3 complete.**

### 2026-06-22 — G1.3 P2–P4 DONE: unify the MLA into one v3.1 class
Folded the v3.2 indexer + DSA forward into v3.1's `ttMLA`; the v3.2 subclass is gone.
- **`deepseek_v3_d_p/tt/mla/mla.py`:** added module-level `INDEXER_WEIGHT_NAMES` +
  `interleaved_to_halfsplit_perm` (moved from v32); imports `ops` / `get_cos_sin_matrix` /
  `get_rot_transformation_mat`. `__init__` gains `index_args=None`, pops indexer weights from the
  state_dict at the top (absent → dense), and at the end sets `_has_indexer` / `_index_kbuf` and
  (when present) builds the indexer (lazy `ModelArgs` import — removed in P5). Renamed the old
  `forward` → `_dense_forward` (byte-for-byte) and added a dispatching `forward`: indexer present &
  `end_pos > index_topk` → `_dsa_forward`; else `_dense_forward` (writing indexer K first when an
  indexer is present in the dense regime). Appended all 12 indexer/DSA/CCL-helper methods.
- **Dispatch guard fix:** condition checks `self._has_indexer` FIRST, then `index_args.index_topk`
  (v3.2 relied on `index_args` always existing; dense v3.1 has none).
- **P4 — subclass retired.** `deepseek_v32/tt/mla/__init__.py` re-exports v3.1's `ttMLA`. Per user
  (v3.2 is new code on this branch; only tests must pass) the `deepseek_v32/tt/{mla/mla.py,ops.py}`
  back-compat shims were **deleted**, not kept; the 3 `deepseek_v32.tt.ops` test imports repointed
  to `deepseek_v3_d_p.tt`.

**Verified (8× Blackhole):**
- DSA: `test_vs_gpu_ref` indexer/kv/mla_output, DS + GLM, all meshes — PASS (indexer PCC DS 0.959,
  GLM 0.9995; MLA output + KV latent ~0.9995). One test-only fix: the indexer test monkeypatches
  `ops.topk_indices`; since the unified code calls `deepseek_v3_d_p.tt.ops`, the test's `ops` import
  was repointed there so the patch hits the same module object.
- Dense v3.1: `test_mla_chunked_prefill` 2x4-line — all completed cases PASS, KV cache PCC
  k_nope/k_pe ~0.99988 (one trailing 56k-token CPU-reference case hit the 580s wall-clock timeout,
  not a failure). The dense forward is unchanged (`_dense_forward` is the old body verbatim).
- Collection: 152 tests across the 6 affected v32 test files collect with no import errors.

**P5 next:** express indexer config via the HF `PretrainedConfig` to drop the lazy
`deepseek_v32.reference_cpu.ModelArgs` import — removes the last inverted dep; closes the
`config: PretrainedConfig` + `extend mla config` TODOs.

### 2026-06-22 — G1.3 P1 DONE: relocate DSA ops to v3.1
Moved `deepseek_v32/tt/ops.py` → `deepseek_v3_d_p/tt/ops.py` verbatim (self-contained: torch +
ttnn only). Left a re-export shim at the old path (public API + private `_to_host` /
`_chunk_offset_tensor` that tests use). v3.1 now owns the DSA ops; no circular import (v32 pkg
`__init__`s are empty). Verified: `test_ops_shapes` + `test_ops_numerics` pass through the shim;
`test_vs_gpu_ref` indexer device tests (DS + GLM, L0) pass with PCC unchanged (DS 0.959, GLM 0.9995).
Next: P2 (merge methods into v3.1 ttMLA, gated by `_has_indexer`).

### 2026-06-22 — G1.1-indexer IMPLEMENTED (per D3)
Put the indexer RoPE table build on v3.1 infra. Files touched:
- `deepseek_v3_d_p/tt/mla/rope.py`: `get_cos_sin_matrix(hf_config, interleave=True,
  scale_handling="softmax")` — `interleave` → `stack` (duplicated pairs) vs `cat` (half-split);
  `scale_handling="softmax"` forces `mscale_all_dim=mscale` so `_mscale=1` (pure tables),
  `"baked"` keeps the HF split. `RotarySetup.get_rope_tensors(seq_len, interleave=True)` →
  threads the flag and builds `trans_matrix` only when interleaved (else `None`). Both default
  to the prior behaviour → **v3.1 MLA path unchanged**.
- `deepseek_v32/tt/mla/mla.py`: `_build_index_rope_tables` now calls
  `get_cos_sin_matrix(self.config, interleave=index_rope_interleave, scale_handling="softmax")`
  instead of the inline `precompute_freqs_cis` + stack/cat; removed that import. `_device_rope_pe`
  (the op selection `rotary_embedding_hf`/`_llama` + SP slicing + nope concat) **kept as-is**.
  Frequencies now come from the HF `config` (single source of truth, == the MLA's own rope).

**Verification (CPU probe, `/tmp/rope_probe.py`):**
- New shared builder vs old `precompute_freqs_cis` path: **cos/sin PCC 1.0000000, max|Δ|=5.96e-08**
  (fp32 rounding) for BOTH DS (interleave=False, θ=1e4, factor=40) and GLM (interleave=True,
  θ=1e6, factor=1). → freq-source switch is numerically identical.
- MLA invariance: `interleave=True` `softmax` vs `baked` tables identical (max|Δ|=0.0) for DS & GLM
  → defaulting to `"softmax"` does not change the MLA.
- Both edited modules import cleanly; `precompute_freqs_cis` fully removed from `mla.py`.

**Device verified (`test_vs_gpu_ref.py`, 8× Blackhole, L0, all meshes):** all PASSED — DS + GLM,
`test_indexer_device_vs_reference` / `test_kv_cache_device_vs_reference` / `test_mla_output_device_vs_reference`
(GLM `sp2xtp4` skipped by design, tp>2). DS indexer logits PCC 0.959, topk 0.985; GLM indexer
0.9995, topk 0.9954, MLA output 0.9998, KV latent 0.9995, k_pe L2 rel-err 0.0068.
- **PCC unchanged by this edit (proven):** ran the DS indexer test on the pre-change code via
  `git stash` → PCC `0.9591557697114468`, identical to post-change to the last digit. The ~0.96 is a
  *pre-existing fp8 baseline* — the GPU reference computes the DS indexer in fp8 (+ Hadamard), our
  port is bf16 (`LOGITS_PCC=0.95` threshold, header documents "Observed layer 0: logits 0.96").
  GLM's reference has no fp8 there → 0.9995 through the same code, confirming rope is exact and the
  gap is quantization, not layout.
- Final naming: `bake_mscale: bool = False` (see updated pending-decision note).

### 2026-06-22 — G1.1 / G1.2 diagnosis (execution-order item 1)
- Mapped RoPE and CCL across v3.1 and v3.2 (two Explore passes), verified key claims by
  reading source directly.
- **CCL (G1.2):** v3.2's `_tp_rs_ag` (mla.py:529) and `_tp_ag_reduce` (:558) are
  byte-for-byte the inline blocks in v3.1 (`deepseek_v3_d_p/tt/mla/mla.py:718-738`,
  `797-809`, RS-only `911-923`, and `_forward_kv_only` `954-967`). Decision D1: converge by
  inlining v3.2 to match v3.1 (no shared helpers). Inlining not yet applied.
- **RoPE (G1.1):** see L1. Frequencies unifiable; scale handling equivalent for current
  configs. Decision D2: track unification as its own task.
- Hardware available locally: 8× Blackhole p150b → can run `test_v32_mla_vs_cpu_reference`
  on a 2×4 mesh to verify changes (no CI dependency for this step).
