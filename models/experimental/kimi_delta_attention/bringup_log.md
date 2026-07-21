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

### Phase 8 — perf: before vs after distribution (realtime profiler, LoudBox)

`test_kda_perf.py` — per-forward device-kernel time (max-across-chips), real KDA head dims (32 heads,
head_dim 128), FABRIC_2D, realtime profiler (not tracy). before = TP=1 `(8,1)` (all heads/chip, no
all-reduce); after = TP=4 `(2,4)`. Per-chunk cost = **warm ≡ long** for KDA (fixed-size state); cold =
N chunks × per-chunk.

**T=640 (box-scaled local chunk), device µs:**

| group | before TP=1 | after TP=4 | Δ |
|---|---|---|---|
| matmul | 112 987 | 30 756 | **3.7× sharded** (ideal 4×) |
| collective | 0 | 26 354 | TP all-reduce added |
| other (recur/gate/norm) | 104 289 | 59 427 | 1.8× |
| **total** | **217 276** | **116 538** | **1.87×** |

(T=256: 87 871 → 47 709 µs, 1.84×.) **Distribution validated as a win** (TP shards matmul ~4×).

**Gap to roofline, quantified (after TP=4, T=640, per chip):**
| | measured | ideal | gap | util (target) |
|---|---|---|---|---|
| compute (matmul+other) | 90.2 ms | 88 µs HiFi4 / 44 µs HiFi2 | **~1030×** | **0.10%** (60%) |
| collective | 26.4 ms | 49 µs (LoudBox roofline) | **~540×** | **0.19%** (40%) |

Achieved 0.147 GFLOP/s vs 152 TFLOP/s HiFi4 peak. **Root cause: launch-bound, not FLOP-bound** —
the token-loop issues **~9000 device kernels/forward** (640 steps × ~14 ops + projections) at ~10 µs
fixed launch overhead each → ~90 ms is almost entirely launch overhead. The FLOP roofline is ~1000×
away *because* of un-fusion. Fused chunk op (task 14) collapses ~9000 kernels → a handful, removing
that overhead; it's the precondition for reaching the 60%/40% targets. TP-sharding benefit (relative)
is the trustworthy signal now; absolute-vs-roofline waits on fusion.

### Phase 8→9 — perf loop iteration 1: chunked prefill op (fuse the token-loop)

**Analysis (prior):** recurrence launch-bound — ~14 700 tiny kernels/forward at T=640, 0.10% compute util,
~1030× off roofline. **Improvement:** `chunk_kda_ttnn` (matmul-based chunked, intra-chunk batched over
chunks + Neumann-doubling inverse; only the NT≈10 cross-chunk state scan loops). PCC vs torch
`naive_chunk_kda` = **0.9999993** (HV=8, K=V=128, T=256, NT=4). **Re-analysis** (recurrence op alone,
production dims, realtime profiler on (8,1)):

| T | recurrent (token-loop) | chunked | gain |
|---|---|---|---|
| 640 | 78.7 ms / 14 695 kernels | 5.25 ms / **371 kernels** | **15.0× faster, 39.6× fewer kernels** |
| 256 | 31.4 ms / 5 879 | 2.13 ms / 173 | 14.7× / 34× |

Kernel count collapsed 14 695→371 as intended. Still ~14 µs/kernel × 371 → chunk is *less* launch-bound
but not yet fused into one kernel; next loop iterations: wire into the layer + re-measure full before/after;
then reduce the 371 (fuse NT loop / bigger chunk / C++ kernel) toward the ~44 µs compute roofline.

### Phase 9 — perf loop iteration 2: chunk wired into the layer → collective now dominates

Wired `chunk_kda_ttnn` into both layers (prefill when `T % 64 == 0`, else recurrent). Layer PCC still
green incl. T=128 chunk path. **Full-layer before/after re-measured (T=640, device µs):**

| group | before TP=1 | after TP=4 |
|---|---|---|
| matmul | 14 341 | 3 923 |
| collective | 0 | **26 355** |
| other | 18 687 | 12 804 |
| **total** | **33 028** (was 217 276) | **43 082** (was 116 538) |

- **Chunking cut the un-distributed layer 217→33 ms (6.6×).** matmul in the distributed case 30.8→3.9 ms.
- **TP=4 is now SLOWER than TP=1** (43 vs 33 ms): once compute is fused, the **un-optimized all-reduce
  (26.4 ms, all-gather+local-sum) dominates** and swamps the compute savings. This realizes the roofline's
  **CCL-bound** prediction — distribution only pays once the collective is optimized.
- **Next target: the collective.** 26.4 ms vs 49 µs roofline (~540× off) — the all-gather+sum path is
  inefficient; switch to `reduce_scatter` + `all_gather` (needs the 3-semaphore RS set), or reduce
  collective volume. This is the next loop iteration.

### Phase 9 — perf loop iteration 3: optimize the collective (RS+AG)

**Improvement:** replaced the all-reduce's all-gather+local-sum with `reduce_scatter_minimal_async` +
`all_gather_async` (3-semaphore RS set). PCC still 0.99998 on (2,4). **Re-analysis (T=640, after TP=4):**

| group | prev (all-gather+sum) | now (RS+AG) |
|---|---|---|
| collective | 26 355 µs | **477 µs** (**55× faster**) |
| **total (after TP=4)** | 43 082 µs | **10 898 µs** |

**TP=4 vs TP=1 now: 10.9 ms vs 33.0 ms = 3.0× win** — distribution pays once the collective is efficient.
Collective now ~10× off the 49 µs roofline (was ~540×). **New dominant term: "other" 6.5 ms** (the chunk
op's elementwise: exp/mul/mask/transpose), then matmul 3.9 ms. Next loop targets: reduce chunk elementwise
op count / fuse; then matmul fidelity/config.

**Distribution + fusion scorecard (T=640, per-chip device time):**
- token-loop, TP=1: 217 ms → chunk, TP=1: 33 ms → chunk, TP=4, RS+AG: **10.9 ms** (**~20× from the start**).

### Phase 9 — perf loop iteration 4: matmul fidelity (HiFi4→HiFi2) — a no-op

**Hypothesis:** the 3.9 ms matmul group is FLOP-bound (big projections) → lower fidelity helps.
**Result:** HiFi2 everywhere (chunk + projections) → matmul 3923→3878 µs (~1%), chunk op 5249→5209 µs
(~1%), PCC 0.9999 (unchanged). **Refuted — everything is launch-bound.** At TP=4 the projections are only
~70–140 µs of actual compute; the matmul group is dominated by the chunk op's ~70 *small* matmuls at
per-kernel launch overhead. **Conclusion: kernel COUNT is the only lever, not fidelity; the composed
chunk op is at its structural floor (~371 kernels / ~5 ms).** Kept HiFi2 (production-realistic, PCC fine).

**Where this leaves the loop.** Cumulative T=640: 217 → 10.9 ms (~20×) via chunking + RS/AG collective.
Both the chunk op (~5 ms) and the layer (10.9 ms) are launch-bound at hundreds of kernels — the composed
ttnn approach cannot reach the ~44 µs compute roofline (which assumes ~1 fused kernel). **The next real
lever is a fused C++ diagonal-gate chunk kernel** (collapses hundreds of kernels → O(1)); composed
micro-opts (hoist redundant ops, fewer reshapes) are ≤10–20% and hit the same floor. Documented as the
major remaining perf item; SP sequence-sharding (task 15) is the other (Galaxy-relevant).

### Phase 9 — perf loop iteration 5: hoist redundant NT-loop ops (composed floor confirmed)

Hoisted `q*eg` (used 2×/iter) and `k*eng` out of the NT loop (batched over chunks). Kernels 371→**333**
(T=640), time 5209→5132 µs (**~1.5%**), PCC 0.99998. **Confirms the composed floor:** even −38 kernels
buys ~1.5%; composed micro-opts are ~1–2% each. The composed chunk op is done at ~333 kernels / ~5 ms.

**Perf loop summary (T=640, per-chip device time):**
| stage | total | note |
|---|---|---|
| token-loop TP=1 | 217 ms | launch-bound, ~14.7k kernels, 0.1% util |
| + chunked op | 33 ms | 15× recurrence |
| + TP=4 (all-gather+sum) | 43 ms | collective-bound |
| + RS/AG collective | 10.9 ms | collective 55× → TP=4 a 3× win |
| + HiFi2 / hoist | ~10.7 ms | ~no-op (launch-bound floor) |

**~20× achieved.** Remaining gap to the ~93 µs (compute+CCL) roofline is **launch overhead** — the
composed op's hundreds of small kernels. CCL util is now ~10% (target 40%, within 4×); compute is still
launch-bound (~0.4% util). **The only lever left for device time is a fused C++ diagonal-gate chunk
kernel** (collapse hundreds of kernels → O(1)); it's the major remaining perf item. Trace capture would
cut *wall-clock* (host dispatch) but not the profiler device-time sum. SP sequence-sharding (task 15,
Galaxy) is the other open item.

### Phase 9 — C++ fused chunk kernel: decision + plan

**Decision: FORK** `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/` → new `chunk_kda/`
(not greenfield, not a `diagonal_gate` flag on the live op). Evidence (recon):
- ~80% reusable verbatim: op scaffolding, phased prep/scan (grid fan-out + value-parallel scan), and the
  **WY block-inverse** (`invert16`/`invert_block`/`asm4` in `compute/chunk_gdn_prep.cpp`) — numerically-hard,
  already-debugged fp32. The op's comments (`chunk_gated_delta_rule.cpp:302-307`) say Neumann-doubling
  (my prototype's inverse) goes **all-NaN at NC≥16** → greenfield would repeat that. **Keep the op's WY
  inverse; take the decay math from the prototype.**
- Scalar→diagonal delta is **~14 localized sites** (all in `compute/chunk_gdn_prep.cpp` + `chunk_gdn_scan.cpp`):
  gate CBs resize (`cb_g/decay*` `Ct→ck`, `cb_dl` `1→Kt`); 5 row-broadcast decay muls → `[C,K]` elementwise;
  scalar state-decay `bcast_scalar_mul` → per-K column broadcast (copy `bcast_cols_mul` into scan); host g
  head-split `[BH,NC,C,1]→[BH,NC,C,K]`; **delete** the `[C,C]` L_mask block (`prep.cpp:450-468`) and its two
  mask-multiplies, replace by pre-scaling matmul operands `(k⊙eg)@(k⊙eng)ᵀ` — net *simpler*. Prototype
  already validated this factored form at PCC 0.9999993.
- Not a flag: the diverging CB tile-sizes + L_mask branch are compile-time; a flag risks regressing the
  qwen36 production GDN (`fused_chunk.py`, `tp.py`).
- Build surface: mirror the dir (host op, device op, phased prims, 2 program factories, compute prep/scan,
  dataflow readers/writers, nanobind), + `sources.cmake`/`CMakeLists.txt`/`transformer_nanobind.cpp`.

**Plan:** (A) fork verbatim + rename + build + verify identical to the original (scalar gate) — de-risk the
fork mechanics; (B) apply the ~14 gate edits, build, PCC vs `naive_chunk_kda`; (C) wire into the layer,
measure vs the composed ~10.9 ms. C++ builds are multi-minute; expect kernel-debug iterations.

### Phase 9 — C++ fused kernel Phase A: fork compiles

Forked `chunk_gated_delta_rule` → `ttnn.transformer.chunk_kda` (20 files, symbol-renamed). Build wired
(sources.cmake, CMakeLists kernel glob, transformer_nanobind). **Unity-build gotcha:** the fork's
file-local anonymous-namespace helpers (`pad_time_tile`, `check`, `compute_cfg`, `PrepWorkDist`,
`distribute_prep`, …) clash with the original's in a merged unity TU → `SKIP_UNITY_BUILD_INCLUSION` on the
5 forked host TUs (cleaner than renaming every symbol; a `using namespace` wrap *reintroduced* ambiguity).
**Builds clean; `ttnn.transformer.chunk_kda` loads** — still scalar-gate (identical to source). Commit `3d820fc`.

**Phase B — the gate edits (scalar→per-channel), sites confirmed by reading the kernels:**
`compute/chunk_kda_prep.cpp`: cumsum `mm(cb_tril,cb_g,cb_decay,Ct,Ct,1)` → `Nt=Kt` (:446); `expc` Ct→ck
(:448); **delete** the `[C,C]` L_mask block (:453-468) + its two consumers (:491 A, :555 intra) — instead
pre-scale operands `A=(k⊙eg)@(k⊙eng)ᵀ`, `intra=(q⊙eg)@(k⊙eng)ᵀ`; `decayfac` → `[C,K]` (:473-482); `w`
`bcast_cols_mul`→elementwise (:547); `q_decay` (:559), `k_dec` (:563) elementwise; `dl` 1→Kt tiles
`=exp(g_last)` (:587). `compute/chunk_kda_scan.cpp`: state decay `bcast_scalar_mul(cur_S,cb_dl,…)` →
`bcast_cols_mul(cur_S[K,V], dl[K,1])` per-K over V (:172, copy the helper from prep). Readers/writers: g
`Ct→ck`, dl `1→Kt`. Program factories: cb_g/decay*/dl tile-sizes. Host `chunk_kda.cpp`: g head-split
`[BH,NC,C,1]→[BH,NC,C,K]`. **Keep the WY block-inverse untouched** (gate-independent). Validate PCC vs
`naive_chunk_kda` (prototype's factored form = 0.9999993); C++ builds ~2-3 min each, iterative.

### Phase 9 — C++ kernel Phase B: implementation approach (32-CB constraint solved)

Target **C=32 (Ct=1)** — simplest WY path (`invert_block`), numerically safe for `exp(±decay)`. All 32
CBs are in use, so: **delete the L_mask → free `cb_lmask` (c_12), repurpose as `k_keng`=k⊙exp(−decay)**
(resize cc→ck). **Key reuse:** the factored operands already exist as outputs — `w=k_beta⊙eg` (`prep:547`)
and `q_decay=q⊙eg` (`prep:559`) — so `A=w@k_kengᵀ`, `intra=q_decay@k_kengᵀ`; only `k_keng` is new
(+ transient `eng` in scratch, scr=16 tiles holds ck=4). Reorder prep to compute `eng/k_keng/w/q_decay`
before A. Gate CBs `g/decay/decay_exp/decayfac` Ct→ck; `cb_dl` 1→Kt (dl=exp(g_sum) per-K = last row of
decay_exp, transposed [1,K]→[K,1]). Scan state decay `bcast_scalar_mul`→per-K `bcast_cols_mul`. KDA op
forces phased mode (monolithic left scalar). Host g head-split →[BH,NC,C,K].

### Phase 9 — C++ fused kernel: DONE + validated + measured

**Correctness:** `ttnn.transformer.chunk_kda` vs torch `naive_chunk_kda` **PCC 0.9999906** (HV=4, K=V=128,
C=32, T∈{32,64,128}). Correct essentially first-run — only a host reshape `[BH,NC,C,1]→[BH,NC,C,K]` and a
test-unpack fix needed; the diagonal-gate tile math + reused WY inverse were right. Commit `1e9edbe`.

**Perf (recurrence op alone, production dims, realtime profiler on (8,1)):**
| T | C++ fused | composed ttnn | gain |
|---|---|---|---|
| 640 | **610 µs / 14 kernels** | 5133 µs / 333 kernels | **8.4× faster, 24× fewer kernels** |
| 256 | 279 µs / 14 | 2089 µs / 159 | 7.5× |

Kernel count 333→14 (near the ~6-program phased floor). The fusion goal is met: the recurrence is no
longer launch-bound. Next: wire `chunk_kda` into the layer (prefill, C=32) and re-measure the full
before/after — the layer's "other" term (6.5 ms, dominated by the composed chunk) should collapse.

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
