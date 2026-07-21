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

**Short-conv on device — native `ttnn.conv1d` vs FIR (resolved).** First tried native
`ttnn.conv1d` (Conv1dConfig fused SiLU) mirroring the GDN reference: it **hung the device NOCs at
T=1** (degenerate `input_length=4`; the reference routes decode through FIR, not native conv1d) and
gave **PCC 0.0 at T≥2** (output layout/format mismatch). More decisively, native conv1d **overflows
L1 at D>2048**, and the real KDA `key_dim = num_heads*head_dim = 32*128 = 4096` — so native conv1d
could not run the production layer regardless. Switched to the reference's **FIR path**
(`_causal_conv1d_fir`): left-pad by K−1, K shifted multiply/`addcmul` taps, then SiLU — fp32, any T,
any D, no L1-scratch. Result: `conv1d_op` PCC 1.0; layer PCC byte-identical to the host conv.
*Lesson: "the existing op" was FIR, not `ttnn.conv1d`; verify an op's shape/dim envelope against the
**target model dims** before adopting it.*

**Missing/blocked ops (Phase 7 inputs):**
- Diagonal-gate **chunked** delta-rule kernel. The fused C++ op `ttnn.transformer.chunk_gated_delta_rule`
  assumes scalar per-head gate (cumsum baked in). A KDA chunked prefill needs either a new C++ kernel
  or a ttnn-composed per-channel chunk scan. **Deferred to Phase 7**; phases 5–6 use recurrent + CPU
  fallback for long prefill.
- No native `ttnn.cumsum` (documented in the reference) — relevant only to the chunked path.

---

### Phase 8 — distribution implementation (in progress)

- **Option B approved** (SP=8×TP=4 Galaxy; measured on LoudBox SP2×TP4 proxy). Roofline in `ROOFLINE.md`
  (CCL-bound, esp. Galaxy 0.27:1). Rebased onto origin/main `b8012569c86` (realtime profiler available).
- **TP distribution done + validated.** `tt/ttnn_kda_dist.py::TtKimiDeltaAttentionMesh`: TP shards heads
  (col-parallel q/k/v/gate projections, row-parallel o_proj + TP all-reduce via all-gather+local-sum);
  recurrence/gate/conv/norm are per-head-local, reused unchanged. **PCC 0.99998 on the real (2,4)
  LoudBox board** (`test_kda_dist.py`, conv on/off), sequence replicated over SP.
- **HW findings:** (1,4) is NOT a trainable mesh on the 8-device LoudBox — supported shapes are
  {(8,1),(4,2),(2,4)} (`sparse_mla_mesh.py`); use (2,4). Fabric config = canonical **FABRIC_2D**
  (fabric_router_config + RELAXED_INIT); a bare `open_mesh_device(fabric_config=…)` fails — the mesh_device
  fixture applies fabric via `set_fabric`. `reduce_scatter_minimal_async` needs 3 semaphores (own set);
  `all_gather_async` needs 2 — used all-gather+sum for the all-reduce. `test_fabric_probe.py` is a kept
  board-health probe.
- **Fused-op trigger — investigated and RETRACTED (measurement error).** A first probe reported
  "4.7 s/forward at T=640" and I called it dev-loop-breaking. That was a **single un-warmed forward
  including per-shape kernel JIT compile**. Re-measured with warmup: **cold 1.0 s, warm 637 ms/forward
  at T=640** (~1 ms/token, host-dispatch-bound; the profiler-reported *device* time is far smaller).
  A full perf pass (warm + cold-11 + long × before/after) is ~20–30 s — **not** a dev-loop blocker
  (user: "a couple of minutes is not a blocker at this stage"). Lesson: never conclude perf from a cold,
  single-sample forward — warm up and take min-of-N (my own warm/cold rule).
- **Plan (original ordering restored):** (1) perf harness — realtime profiler, warm/cold/long, box-scaled,
  before (single-device) vs after (TP `(2,4)`), per-op-group breakdown vs ROOFLINE targets; (2) real SP
  sequence-sharding + state-scan; (3) fused chunk op as a **perf-phase optimization** (removes dispatch
  overhead + efficient kernels) — a real win but not a blocker.

## Backlog

- [ ] Phase 7: diagonal-gate chunked delta-rule kernel (C++ or ttnn-composed per-channel chunk scan).
- [x] ~~short-conv on device~~ DONE — FIR multiply-accumulate (see Learnings; native `ttnn.conv1d`
      rejected: overflows L1 at D>2048 = the real KDA key_dim=4096, and hung at T=1).
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
- **2026-07-20 17:48Z** — Ported short-conv off the CPU fallback to **on-device FIR** (after native
  `ttnn.conv1d` hung at T=1 and can't handle D=4096 — see Learnings). Forward path now **fully on
  device, no CPU fallback**. Full suite **16/16 green**.
- **2026-07-21** — Phase 8 (Model distribution) analysis → `DISTRIBUTION.md`. Scope: KDA layer on
  Galaxy 8×4, prefill, + full-model sketch. Recommend **SP=8 (axis0) × TP=4 (axis1)** = production MLA
  mesh; KDA honors MLA's boundary contract, its only new collective a fixed-size SP-axis **state-scan**
  (~9µs/layer) replacing MLA's KV all-gather. First-order estimate: prefill compute-bound (~8–18:1
  C:DM). **At the judgment-call gate — awaiting decision before any distribution code.**

### Phase 5/6 — device validation ledger (LoudBox, 1× Blackhole p150b, fp32)

| Test | Config | PCC | Gate | Status |
|---|---|---|---|---|
| `recurrent_kda_op` | T∈{1,4,16}, HV=4, K=V=64 | 0.99999970–0.99999982 | ≥0.98 | ✅ |
| `recurrent_kda_op` | T∈{1,4,16}, HV=8, K=V=128 (prod dims) | 0.99999966–0.99999978 | ≥0.98 | ✅ |
| `kda_gate_op` | HV=8, K=64 | pass | ≥0.99 | ✅ |
| `l2norm_op` | K=128 | 0.99999258 | ≥0.99 | ✅ |
| `conv1d_op` (FIR short-conv) | T∈{1,2,8,16}, D=256 | 0.9999999997–1.0 | ≥0.98 | ✅ |
| `kda_layer` (end-to-end) | T∈{1,8,16}, conv on device | 0.99997689–0.99998241 | ≥0.98 | ✅ |
| `kda_layer` (end-to-end) | T=8, conv off | 0.99998557 | ≥0.98 | ✅ |

**Command:** `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_kda_ttnn.py` → `16 passed`, `SAFE_PYTEST_RESULT: PASS`.

**VALIDATED:** diagonal-gate KDA recurrence (op), KDA gate op, L2-norm op, depthwise causal
short-conv+SiLU (FIR, on device), and the full KDA layer **entirely on device — no CPU fallback in
the forward path** (projections + short-conv + gate + recurrence + gated-RMSNorm + o_proj), single
Blackhole, fp32, random-init weights, PCC vs torch reference which is itself bit-exact vs the FLA
author reference. The on-device FIR conv is numerically identical to the host conv (layer PCC
byte-identical before/after the port).

**NOT VALIDATED (out of scope / Phase 7+):** chunked-prefill *on device* (torch `naive_chunk_kda`
covers prefill correctness on CPU; device chunk kernel is a Phase 7 requirement); bf16 numerics (fp32 only);
multi-device / TP mesh orientations (single-device layer bringup); real-weight numerics (random-init);
GVA HV>H on device (config uses HV==H; torch ref validates GVA). Determinism across runs and
performance are not part of Phases 0–6.
