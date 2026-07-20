# KDA (Kimi Delta Attention) bringup log

> Living ledger for the tt-metal bringup of **Kimi Delta Attention (KDA)**, the linear-attention
> operator of MoonshotAI's Kimi-Linear architecture (arXiv 2510.26692). Follows
> `~/.claude/references/model-bringup.md`. Timestamps in UTC.

---

## Goals

- Bring up the **KDA attention layer** (the genuinely-new delta) on Blackhole, as a variant of the
  existing in-tree Gated DeltaNet (GDN), through Phase 6 (correctness PCC ≥ 0.98).
- **Out of scope** (this pass): the full Kimi-Linear-48B-A3B model, the MoE, the hybrid 3:1 KDA:MLA
  stacking, tokenizer/demo. KDA layer only.
- Ground truth = torch reference vs ttnn on identical inputs (random-init weights). No large
  checkpoint download needed for correctness (single-layer HF weights optional, deferred).

---

## Nomenclature

| Symbol | Shape | Meaning |
|---|---|---|
| `B, T` | | batch, sequence length |
| `H` | | number of q/k heads (`num_heads`, 32 in 48B config) |
| `HV` | | number of v heads (`num_v_heads`; GVA if `HV>H`) |
| `K` | | key/query head dim (`head_k_dim`, **128**) |
| `V` | | value head dim (`head_v_dim`, 128; `expand_v=1`) |
| `q, k` | `[B,T,H,K]` | queries/keys (L2-normed after short-conv+SiLU) |
| `v` | `[B,T,HV,V]` | values (short-conv+SiLU) |
| `g` | `[B,T,HV,K]` | **per-channel** decay gate in **log space** (this is the KDA delta) |
| `beta` | `[B,T,HV]` | scalar per-head learning rate (sigmoid) |
| `S` (state) | `[B,HV,K,V]` | recurrent state matrix |

**KDA state recurrence** (paper Eq., FLA `naive_recurrent_kda`):
```
S_t = Diag(exp(g_t)) · S_{t-1} + (β_t k_t) ⊗ (v_t − k_tᵀ S_{t-1})     # delta rule w/ diagonal decay
o_t = q_tᵀ S_t                                                          # scale q by K^-0.5
```
**Gate transform** (FLA `naive_kda_gate`, applied before the recurrence, from a low-rank proj
`f_proj: hidden→head_v_dim→HV·K`):
```
g = -exp(A_log)[:,None] · softplus(f_proj(x) + dt_bias)     # A_log:[HV], dt_bias:[HV·K]
```
(Optional lower-bound variant: `g = lower_bound · sigmoid(exp(A_log)·(f+dt_bias))`.)

**Full layer pipeline** (FLA `fla/layers/kda.py::KimiDeltaAttention`):
`q/k/v_proj → short-conv(k=4)+SiLU → [L2norm q,k] → gate g, β → KDA recurrence/chunk →
head-wise gated RMSNorm( · , sigmoid(g_proj(x)) ) → o_proj`. **NoPE** (no RoPE anywhere).

---

## Decisions

- **D1 (layer-vs-greenfield): layer.** KDA is a *variant* of Gated DeltaNet, which already exists
  in-tree twice. Reuse, don't reinvent. Per playbook "variant bringup collapses the spine" — no
  throwaway spike; Phase 2 is a delta analysis against the existing GDN. *(2026-07-20)*
- **D2 (reference scaffold): `models/experimental/gated_attention_gated_deltanet/`.** It is the
  self-contained torch-golden + ttnn + PCC-harness GDN; the qwen36 `tp.py` production path is a
  full TP model (more than a layer bringup needs) and its real math *delegates to this same
  experimental module's ttnn ops* + a fused C++ op. Mirror the experimental module's layout. *(2026-07-20)*
- **D3 (the delta): diagonal gate.** The one substantive change vs the reference is the forget gate:
  scalar `g[B,T,H]` → per-channel `g[B,T,HV,K]`. Everything else (delta rule, β, L2-norm, short-conv,
  gated-RMSNorm output, NoPE) is GDN-shaped. *(2026-07-20)*
- **D4 (device path for Phase 5): recurrent.** The ttnn *recurrent* delta-rule op extends to the
  diagonal gate naturally (decay becomes `[K,1]` broadcast over V instead of a per-head scalar). The
  *chunked* path uses the fused C++ op `ttnn.transformer.chunk_gated_delta_rule` which bakes scalar-gate
  cumsum internally → diagonal-gate chunk kernel is a **Phase 7 requirement** (see Backlog). For
  phases 5–6, chunked prefill uses a CPU fallback (allowed at this gate). *(2026-07-20)*
- **D5 (weights): random-init, PCC vs torch ref.** Correctness proves the *algorithm* (CPU-ref vs
  ttnn on identical inputs), not real-weight numerics; deferred single-layer HF weight pull optional. *(2026-07-20)*
- **D6 (module location): `models/experimental/kimi_delta_attention/`** with `torch_functional/`,
  `tt/`, `tests/` mirroring the reference module. *(2026-07-20)*

---

## Learnings

### Phase 1 — infra map (the terrain KDA plugs into)

**Two in-tree GDN implementations:**

1. **`models/experimental/gated_attention_gated_deltanet/`** — the reference scaffold (D2).
   - `torch_functional/delta_rule_ops.py` — torch golden: `recurrent_gated_delta_rule` (:31-111,
     token loop) and `chunk_gated_delta_rule` (:114-245, WY/forward-substitution chunk). Layout
     `q,k[B,T,H,K] v[B,T,H,V] beta,g[B,T,H]`. **Scalar gate**: decay broadcast `[...,None,None]`
     (:93), chunk cumsum over time only (:189).
   - `torch_functional/gated_deltanet.py::gated_deltanet_forward` (:73-217) — full layer:
     proj→conv1d+SiLU→heads/GVA→`beta=sigmoid`,`g=-exp(A_log)*softplus(a+dt_bias)`→delta→
     `rms_norm_gated`(=`rmsnorm(o)*silu(gate)`)→o_proj.
   - `tt/ttnn_delta_rule_ops.py` — ttnn recurrent path: `recurrent_gated_delta_rule_decode_ttnn`
     (:383-479, T=1, `decay=exp(g_t)` :425, `h=h*decay` :456, outer-product write), general-T loop
     `recurrent_gated_delta_rule_ttnn` (:511-576). Ops: `ttnn.matmul/multiply/add/subtract/exp`,
     `ttnn.rms_norm`(L2), `transpose/typecast/reshape`.
   - `tt/ttnn_delta_rule_seq.py` — chunked prefill; **no native `ttnn.cumsum`** → cumsum via matmul
     with triu-ones (:528); calls C++ `ttnn.transformer.gated_delta_attn_seq` (:718).
   - `tests/` — torch-only equivalence (recurrent≡chunk `<1e-2`), and `test_ttnn_validation.py`
     (needs HW): `assert_with_pcc` (:24-62), thresholds **0.95 recurrent / 0.98 attn**.
   - **Self-contained** (only torch/ttnn/math imports), ~660 LOC torch + ~2600 ttnn. Ideal golden.

2. **`models/demos/blackhole/qwen36/tt/gdn/`** — production. `Qwen36GatedDeltaNet` (single-dev,
   delegates to the experimental ttnn op) and `TPGatedDeltaNet` (`tp.py`, tensor-parallel, the one
   that runs Qwen3.6-27B). Mesh `(1,tp)`, cluster_axis=0, per-v-head recurrence with no cross-device
   comms inside the delta rule; col-parallel in-proj, row-parallel out-proj + all-reduce/RS. Gate
   assembly at `tp.py:500-503`/`:640-644`. Fused chunk op `ttnn.transformer.chunk_gated_delta_rule`
   at chunk_size 32 (`fused_chunk.py`). Config via `GDNConfig.from_args` + `model_config.py` derived
   `gdn_*` fields. **Both paths use scalar gate.**

**Authoritative external reference (downloaded to scratchpad `kda_ref/`):**
- FLA `fla/ops/kda/naive.py` — `naive_recurrent_kda` / `naive_chunk_kda` (the true torch golden;
  diagonal gate already, `g[B,T,HV,K]` log-space). **This is what the Phase-4 reference is validated against.**
- FLA `fla/ops/kda/gate.py` — `naive_kda_gate` (the `-exp(A_log)*softplus(g+dt_bias)` transform).
- FLA `fla/layers/kda.py::KimiDeltaAttention` — the nn.Module wiring (projections/conv/norms/gates).
- HF `moonshotai/Kimi-Linear-48B-A3B-Instruct`: `modeling_kimi.py`, `configuration_kimi.py`, `config.json`.

**48B-A3B config facts** (`config.json`): `linear_attn_config = {num_heads:32, head_dim:128,
short_conv_kernel_size:4}`, hidden_size 2304, 27 layers, KDA layers 1-indexed
`{1,2,3,5,6,7,...,25,26}`, full-attn (MLA) layers `{4,8,12,16,20,24,27}` → the 3:1 ratio.
MLA + KDA both `use_nope=true`.

### Phase 2 — delta analysis (feasibility)

**The delta, precisely, vs the scalar-gate GDN reference** (3 sites, per the reference map):
1. Recurrent decay: `h *= exp(g)` where `g` becomes `[.,K]` → decay is `[K,1]` broadcast over `V`
   (was a per-head scalar broadcast over the whole `[K,V]` state).
2. Chunk cumsum: `g.cumsum` over time now yields a per-channel `[.,K]` cumulative log-decay; the
   intra-chunk `L_mask[i,j]=exp(cumsum[i]-cumsum[j])` and `A` matrix become per-channel (FLA
   `naive_chunk_kda` :120-163 shows the exact form).
3. State update: `S = S*exp(g_last) + ((exp(g_last-g)*k)ᵀ @ v_new)` with per-channel `g`.

**Feasibility verdict: high.** The recurrent form is a mechanical extension of an op already on
device; only the decay broadcast shape changes. No new *recurrent* ttnn op needed. Gate transform
(`-exp(A_log)*softplus`), β-sigmoid, L2-norm, short-conv, gated-RMSNorm all already exist as ttnn
ops in the reference module. First feasibility PCC to be recorded in Progress (Phase 6).

**Missing/blocked ops (Phase 7 inputs):**
- Diagonal-gate **chunked** delta-rule kernel. The fused C++ op `ttnn.transformer.chunk_gated_delta_rule`
  assumes scalar per-head gate (cumsum baked in). A KDA chunked prefill needs either a new C++ kernel
  or a ttnn-composed per-channel chunk scan. **Deferred to Phase 7**; phases 5–6 use recurrent + CPU
  fallback for long prefill.
- No native `ttnn.cumsum` (documented in the reference) — relevant only to the chunked path.

---

## Backlog

- [ ] Phase 7: diagonal-gate chunked delta-rule kernel (C++ or ttnn-composed per-channel chunk scan).
- [ ] Phase 7: replace any CPU fallback (chunked prefill) with on-device ops; re-validate PCC.
- [ ] Optional: pull a single KDA layer's real weights from HF to sanity-check real-weight numerics.
- [ ] Later phases: hybrid 3:1 block wiring, MoE, full model, perf.

---

## Progress

- **2026-07-20 16:04Z** — Env recon: 8× Blackhole p150b (LoudBox), tt-metal main built, worktrees
  present, internet OK. Found the two GDN analogs. Scope + weights decisions confirmed.
- **2026-07-20 16:08Z** — Created worktree `mvasilijevic/kda-bringup` off `origin/main`
  (`3218270556c`); launched submodule-init + `build_metal.sh` + `create_venv.sh` in background.
- **2026-07-20 16:09Z** — Downloaded authoritative KDA reference (FLA naive/chunk/gate/layer + HF
  modeling/config) to scratchpad. Read them; algorithm fully specified above.
- **2026-07-20 16:12Z** — Both analog maps complete (Explore agents). Phase 0/1/2 written.
- **2026-07-20 16:16Z** — Worktree build DONE; `ttnn.__file__` resolves under the worktree (import
  trap avoided). Phase 3 (`API_SPEC.md`) + Phase 4 torch reference written; **Phase 4 PASS** —
  port is bit-exact vs FLA author ref (max|Δ|=0.0, PCC=1.0, recurrent & chunk, GVA + non-GVA);
  recurrent≡chunk within 5.95e-4 (<1e-2 precedent); gate matches to fp32 eps.
- **2026-07-20 16:24Z** — Phase 5 ttnn ops + end-to-end layer written; **Phase 6 PASS** on
  LoudBox (device 0, Blackhole p150b). Full suite **12/12 green**. Diagonal-gate recurrence
  correct on device. See validation ledger below.

### Phase 5/6 — device validation ledger (LoudBox, 1× Blackhole p150b, fp32)

| Test | Config | PCC | Gate | Status |
|---|---|---|---|---|
| `recurrent_kda_op` | T∈{1,4,16}, HV=4, K=V=64 | 0.99999970–0.99999982 | ≥0.98 | ✅ |
| `recurrent_kda_op` | T∈{1,4,16}, HV=8, K=V=128 (prod dims) | 0.99999966–0.99999978 | ≥0.98 | ✅ |
| `kda_gate_op` | HV=8, K=64 | pass | ≥0.99 | ✅ |
| `l2norm_op` | K=128 | 0.99999258 | ≥0.99 | ✅ |
| `kda_layer` (end-to-end) | T∈{1,8,16}, conv on | 0.99997689–0.99998241 | ≥0.98 | ✅ |
| `kda_layer` (end-to-end) | T=8, conv off | 0.99998557 | ≥0.98 | ✅ |

**Command:** `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_kda_ttnn.py` → `12 passed`, `SAFE_PYTEST_RESULT: PASS`.

**VALIDATED:** diagonal-gate KDA recurrence (op), KDA gate op, L2-norm op, and the full KDA layer
(projections + gate + recurrence + gated-RMSNorm + o_proj on device), single Blackhole, fp32,
random-init weights, PCC vs torch reference which is itself bit-exact vs the FLA author reference.

**NOT VALIDATED (out of scope / Phase 7+):** chunked-prefill *on device* (torch `naive_chunk_kda`
covers prefill correctness on CPU; device chunk kernel is a Phase 7 requirement); short-conv on
device (CPU fallback used — allowed at this gate, logged for Phase 7); bf16 numerics (fp32 only);
multi-device / TP mesh orientations (single-device layer bringup); real-weight numerics (random-init);
GVA HV>H on device (config uses HV==H; torch ref validates GVA). Determinism across runs and
performance are not part of Phases 0–6.
