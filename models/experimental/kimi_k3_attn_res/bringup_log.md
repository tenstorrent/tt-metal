# AttnRes (Kimi K3 Attention Residuals) — bringup log

Living ledger. Append-only below §Decisions; nothing above §Learnings is rewritten
once a phase has closed.

Methodology mirrors `models/experimental/kimi_delta_attention/` (branch
`mvasilijevic/kda-bringup`), adapted for an op with a different cost profile and a
residual-path blast radius.

---

## Goals

**In scope**

- The AttnRes read op: `attn_res(prefix_sum, block_residual, q) -> hidden`.
- The `block_residual` lifecycle: seal at `layer_idx % 12 == 0`, reset `prefix_sum`,
  and the one model-level output read.
- A synthetic 93-layer depth harness (random module outputs) proving error does not
  compound across 186 chained softmax mixtures.
- TP distribution on LoudBox `(2,4)`, designed for Galaxy `(8,4)` and a multi-Galaxy
  pipeline.

**Out of scope this pass** — stated first, deliberately

- KDA, gated MLA, LatentMoE, the vision tower, the tokenizer, any end-to-end demo.
- Decode (`T=1`). State is `[B, 8, 7168]`; near-trivial. Follow-up.
- Real K3 weight loading beyond a name-mapping table (1.6 TB across 96 shards).
- Actually running on a Galaxy. LoudBox `(2,4)` is the TP proxy; `(8,4)` SP behaviour
  is modelled, not measured.

**Ground truth** is the HF reference `_apply_attn_res`
(`modeling_kimi_linear.py:1075-1088`, Kimi-K3 repo), vendored verbatim into
`reference/hf_attn_res.py` and exercised by a committed test.

---

## Nomenclature

The shape contract, fixed before any code.

| Symbol | Shape | Meaning |
|---|---|---|
| `d` | 7168 | `hidden_size` |
| `L` | 93 | `num_hidden_layers` |
| `Bk` | 12 | `attn_res_block_size` — counts **transformer layers**, not modules |
| `N` | `B*T` | flattened token axis |
| `S` | 0…8 | sealed snapshot count, grows monotonically |
| `eps` | 1e-5 | `rms_norm_eps` |
| `prefix_sum` | `[N, d]` | the single live residual stream |
| `block_residual` | `[N, S, d]` | append-only sealed snapshots, write-once |
| `v` | `[N, S+1, d]` | `cat(block_residual, prefix_sum)` — the candidate set |
| `q_l` | `[d]` | folded pseudo-query, `res_norm.weight * res_proj.weight` |
| `α` | `[N, S+1]` | softmax weights, row-stochastic |

Governing equations:

```
ŝ_i = rsqrt(mean(v_i²) + eps) · ⟨q_l, v_i⟩      # RMS is a per-(token, candidate) SCALAR
α   = softmax_i(ŝ)
out = Σ_i α_i · v_i                             # weighted sum of UN-normalized v
```

Keys are normalized; **values are not**. The mixture is over raw `v`.

Layer pipeline (`_forward_attn_residual`, `modeling_kimi_linear.py:973-1046`):

```
1.  h ← AttnRes(prefix_sum, block_residual; self_attention_res_*)   [skipped iff S==0, i.e. l==0]
2.  if l % 12 == 0:  block_residual ← cat(block_residual, prefix_sum);  prefix_sum ← None
3.  h ← input_layernorm(h)
4.  h ← self_attn(h)                                               # KDA (69 layers) or gated MLA (24)
5.  prefix_sum ← (prefix_sum + h) if prefix_sum is not None else h  # PLAIN ADD, weight 1
6.  h ← AttnRes(prefix_sum, block_residual; mlp_res_*)             [always]
7.  h ← post_attention_layernorm(h)
8.  h ← block_sparse_moe(h)  or  mlp(h) at l==0
9.  prefix_sum ← prefix_sum + h
10. return prefix_sum, block_residual
```

Then once at model level: `h ← AttnRes(prefix_sum, block_residual; output_attn_res_*)`
→ `model.norm` → `lm_head`.

Step 1 reads the **old** `block_residual`, step 6 the **new** one. Step 6 is
unconditional (at `l=0` it mixes 2 candidates). Seals land at
`l ∈ {0,12,24,36,48,60,72,84}` → 8 snapshots. Executed reads: 92 pre-attn + 93
pre-MLP + 1 output = **186**; 187 parameter sets.

Writes into `prefix_sum` are plain `+=` with weight 1. AttnRes rewrites only the
**read**.

Weights — 374 tensors, all bf16, ~2.68 M params:

```
language_model.model.layers.{0..92}.self_attention_res_norm.weight   [7168]
language_model.model.layers.{0..92}.self_attention_res_proj.weight   [1, 7168]
language_model.model.layers.{0..92}.mlp_res_norm.weight              [7168]
language_model.model.layers.{0..92}.mlp_res_proj.weight              [1, 7168]
language_model.model.output_attn_res_norm.weight                     [7168]
language_model.model.output_attn_res_proj.weight                     [1, 7168]
```

---

## Decisions

Numbered, dated, never renumbered. Each names what it rejected.

- **D1 — 2026-07-30 — scope: op + lifecycle + depth harness.** Not primitive-only
  (leaves the seal/reset and depth-compounding risks unproven), not full-layer
  (couples to an unmerged KDA branch and to MLA/MoE work that does not exist yet).
- **D2 — 2026-07-30 — home: `models/experimental/kimi_k3_attn_res/`, off `main`
  @ `6d526e8d61d`.** Not off `mvasilijevic/kda-bringup` — 32 unmerged commits of
  rebase cost for composition we do not need this pass. Not inside
  `deepseek_v3_d_p/` — wrong model, and it eats that suite's CI budget.
- **D3 — 2026-07-30 — prefill first.** Decode is `T=1` and near-free; prefill carries
  every real problem.
- **D4 — 2026-07-30 — composed-first, structured as inter-block-batched + live
  merge.** The split is algebraically exact and published (AttnRes paper §2.2, §5.4.2);
  pretending not to know it costs a perf-loop iteration. C++ only after the composed
  floor is measured.
- **D5 — 2026-07-30 — fold the query at load: `q_l = res_norm.weight ⊙
  res_proj.weight`.** Two `[d]` tensors collapse to one, the RMSNorm gain multiply
  disappears, and the normalized tensor `k` is never materialized (RMS is a
  per-(token, candidate) scalar). Load-time transform, no runtime cost.
- **D6 — 2026-07-30 — vendor the 13-line HF reference as a committed fixture.** The
  KDA branch claimed Phase-4 bit-exactness against an out-of-tree reference and could
  not commit the check. AttnRes is plain PyTorch with no Triton dependency; there is
  no excuse.
- **D7 — 2026-07-30 — random-init weights for bringup, driven from the torch
  reference's own `state_dict()`.** One source of truth for golden and device paths.
  A real-weight loader is a follow-up.
- **D8 — 2026-07-30 — base class `LightweightModule`**
  (`models/common/lightweightmodule.py`). Not `AbstractModule`/`WeightConfig`/
  `convert_weights` — verified: those live only under `models/demos/deepseek_v3/**`
  and `tt-train/**`, zero hits in `deepseek_v3_d_p`.
- **D9 — 2026-07-30 — three-rung Phase-4 gate, not one bit-exact gate.** The plan
  called for `max|Δ| = 0` against the vendored oracle for the folded form. That is
  unachievable and was wrong: folding reassociates the score product from
  `Σ(v·rms_inv·q)` to `rms_inv·Σ(v·q)`, which changes fp32 rounding. The gate splits:
  the HF-order transliteration is bit-exact, the fold and the online-softmax split are
  held to tight fp32 tolerances against the rung below. Rejected alternative — relax
  the whole ladder to PCC — which would have let a real transliteration error hide
  under the fold's rounding noise.
- **D10 — 2026-07-30 — candidates on tensor dim 1, not the last dim.** `S+1 ≤ 9`, so a
  last-dim candidate axis tile-pads 9 → 32 (3.5× waste) and, worse, a last-dim softmax
  over the padded region admits `exp(0) = 1` from padding zeros. Reductions over `d`
  stay last-dim; the softmax over candidates is hand-rolled on dim 1. Rejected
  alternative — a Python list of `S+1` separate `[1,1,N,d]` tensors — costs ~90 op
  launches per read site (~16.7 k per forward) versus ~12 (~2.2 k).
- **D11 — 2026-07-30 — amends D9: no rung is bit-exact; drop the transliteration
  rung.** D9 kept a rung that reproduced the reference's operation order verbatim so it
  could be gated at `max|Δ| = 0`. That rung is a copy of the oracle and therefore tests
  nothing. The ladder now starts at the folded form versus the oracle under the fp32
  dot-product noise floor, plus an fp64 arm proving the fold does not *degrade* accuracy
  — which is the claim D5 actually makes. Rejected alternative: keep the copy for the
  comfort of one bit-exact number in the log.
- **D12 — 2026-07-30 — hand-roll the candidate softmax; do not call
  `ttnn.softmax(dim=1)`.** `ttnn.softmax` reaches its attention-optimized kernel only
  when reducing the last dim; the dim-1 fallback loses **~4 % of the softmax mass even
  in fp32** (rel err 1.4e-2, row sums to 0.962), and neither `numeric_stable` nor
  `fp32_dest_acc_en` moves it. The `max`/`exp`/`sum`/`div` chain measures 15–27× closer
  and is needed anyway by `inter_block`, which must expose `m` and `Z`. One code path,
  better numerics. Rejected alternative — the fused op for its lower launch count —
  trades a measured 4 % magnitude error for 3 op launches on a bandwidth-bound op.
- **D13 — 2026-07-30 — every device gate carries a magnitude check, never PCC alone.**
  PCC is scale-invariant: weights summing to 0.96 scale the output by 0.96 and PCC stays
  above 0.9999. D12's defect is exactly that shape, and the first cut of the rung-4 gate
  passed all 8 configs while carrying it. Each device test now asserts a relative error
  alongside PCC, and `Σα = 1` is promoted to a primary gate rather than a nicety.
  Rejected alternative — inherit the analog's PCC-only gate unmodified — which is what
  let the defect through.
- **D14 — 2026-07-30 — one shared `_walk` drives both backends in the depth harness.**
  `TtAttnResStream` is deliberately interface-compatible with the torch `AttnResStream`
  (`prefix_sum` / `num_sealed` / `read` / `seal` / `accumulate` / `block_size`), so the
  93-layer walk is written once and parametrized by `apply_module` and `free`. The
  read/seal/write order then *provably* cannot diverge between reference and device.
  Rejected alternative — a separate device walk mirroring the torch one — where a seal
  fired one layer late shows up as a precision-looking PCC dip, which is the single
  hardest class of bug to attribute in a 186-read chain.
- **D15 — 2026-07-30 — `TtAttnResStream` owns its tensors.** Construction takes
  ownership of `hidden_states` and `accumulate` takes ownership of `module_out`, which
  makes the first `seal` a zero-copy ownership move into `block_residual` and lets every
  later `seal` free what it superseded. Rejected alternative — caller-owned tensors with
  clone-on-seal — costs an `[1,1,N,d]` copy at each of the 8 seals and, worse, leaves
  the layer-0 aliasing hazard unresolved: at `S = 0` the read is skipped, so the caller's
  `h` *is* the stream's `prefix_sum`, and freeing it after the seal frees
  `block_residual`.
- **D16 — 2026-07-30 — gate the saturated-score case against torch-bf16, not against
  `PCC_GATE`.** `PCC_GATE = 0.9999` is calibrated for scores of order 1, which is where
  the folded query puts them; the saturated test drives `max|score|` to 120 on purpose
  and the device lands at 0.99985/0.99987. Rejected alternatives — declare it a defect
  and chase it (the fix is an fp32 `[1,C,N,d]` intermediate, 2× the op's largest tensor,
  for a regime the model never enters), or loosen `PCC_GATE` globally (blinds the eight
  order-1 configs where the gate does real work). The finiteness assert is what actually
  pins the max-subtraction, and it is absolute.

---

## The two facts that shape everything

### It is bandwidth-bound, not launch-bound

Per read site per token: ~`4·S·d` FLOP against `(S+1)·d·2` bytes → **AI ≈ 1.8
FLOP/byte**. KDA's central finding — "kernel COUNT is the only lever, not fidelity" —
**does not transfer**. Expect the composed floor to be a *traffic* floor, and expect
matmul-fidelity and kernel-count micro-opts to be near-no-ops for a different reason
than on KDA.

Average candidates per read = **5.39**, not 9 (`S` ramps 0→8):

| block | sealed `S` | candidates | reads |
|---|---|---|---|
| l 0–11 | 1 | 2 | 24 |
| l 12–23 | 2 | 3 | 24 |
| … | … | … | … |
| l 84–92 | 8 | 9 | 18 + output |

Σ candidate-vectors touched per token per forward = **1002**.

Naive form (2 passes over `v`, bf16 storage): `2 × 1002 × 7168 × 2 B` =
**28.7 MB/token**. Vanilla residual-stream traffic for comparison: `4d × 93` =
**5.33 MB/token** → AttnRes adds **~5.4× the entire residual-stream traffic of a
vanilla model**.

### The inter/intra-block split is exact and worth ~2.5×

Within a 12-layer block every one of the 24 read sites sees the **identical** sealed
set, and the queries `q_l` are static parameters. So the inter-block contribution for
all 24 reads is computable in **one** pass over `[N, S, d]` at the block boundary; each
read site then merges its precomputed `[N, d]` partial against the live `prefix_sum`
via an online-softmax rescale (2 candidates).

Per block: `(S_n + 24 + 72)·d ≈ (96 + S_n)·d` versus naive `48(S_n+1)·d`. Σ over 8
blocks: **804·d batched vs 2004·d naive → 2.49×**.

The batched form's dominant term is the `24d` write + `24d` read of partials (48 of
96), so the group size is a perf-loop knob, not a fixed choice.

**Phase 5 revises the 2.49× down for the composed form.** That figure assumes the
weighted sum `Σ_s e[r,n,s]·v[n,s,:]` is computed once for all 24 read sites. It cannot
be: the contraction is over the candidate axis *per token*, which is not a broadcast —
it is an `N`-batched matmul `[N,R,S] × [N,S,d]`, and at `R=24, S≤8` both operands
tile-pad to 32 for a **~19 % tile efficiency**. What *does* amortize with plain
broadcasts is the reciprocal-RMS pass, one of the two passes over the sealed set. So the
composed split buys roughly **1.3×**, not 2.49×; the remaining 1.9× is gated on whether
that batched matmul beats 24 broadcast passes, which is a Phase-9 measurement. This
strengthens the Phase-10 case: a fused kernel can hold `e` in L1 and contract over `s`
without ever paying tile padding.

A sealed snapshot's RMS never changes after sealing, so fold `rms_inv` for sealed
candidates at **seal time**: the per-block pass then needs only dot products, and the
live stream's statistics are the only per-read reduction.

### TP is nearly free; PP is not

Both reductions are over `d`. With `d` TP-sharded, AttnRes needs an all-reduce of
`2(S+1)` **scalars** per token per read — the `tt_distributed_rms_norm.py:236-290`
pattern (communicate statistics, never the stream). Combine the sum-of-squares and the
dot into one payload. With the block batching above, that is one small all-reduce per
block for 24 reads plus one 2-scalar reduce per live-stream read.

`block_residual` residency at `S=8`, bf16: **112 KB/token**.

- 5120-token chunk, replicated: 573 MB.
- TP=4 on `d`: 143 MB/chip.
- SP=8 (640 tok/chip) × TP=4: **17.9 MB/chip**. Comfortable.

It does **not** persist across prefill chunks — unlike a KV cache, `block_residual` is
per-forward-pass state. Long-context prefill does not grow it.

**The PP boundary is the real new risk.** At a pipeline boundary you must ship
`prefix_sum` **and** the sealed snapshots: `(1+S)·d`.

| split | boundary layer | `S` | payload multiplier |
|---|---|---|---|
| 2 Galaxies | ~47 | 4 | **5×** |
| 4 Galaxies | ~23 / 46 / 69 | 2 / 4 / 6 | 3× / 5× / **7×** |

At `S=4` that is 70 KB/token across the socket. Socket buffers are **L1-only**
(`d2d_stream_service.cpp:260`) and `outbound_socket_service_sync` **TT_FATALs on
per-shard spec mismatch** — no host relayout, no implicit reshard. A canonical
cross-rank layout must be committed before any distribution code, and produced
natively by the op.

---

## Phase ladder

KDA's numbering, so the parallel is legible. Phase 11 is new.

| # | Phase | Artifact | Exit gate | Machine |
|---|---|---|---|---|
| 0 | Scope + decisions | this file, §Goals/§Nomenclature/§Decisions | worktree builds; `ttnn.__file__` resolves under the worktree | LoudBox host |
| 1 | Infra map | §Learnings 1 | every in-tree analog at `file:line` **plus its own test thresholds** — we inherit them | — |
| 2 | Delta analysis | §Learnings 2 | AttnRes as a countable delta against the analogs; feasibility verdict; `Missing/blocked ops` list → becomes the backlog | — |
| 3 | API spec | `API_SPEC.md` | tensor contract; torch API; ttnn API mirroring it param-for-param; HF weight-name map; numeric validation plan. Written before code, never rewritten after | — |
| 4 | Torch reference | `torch_functional/` | three-rung gate per D9 | CPU |
| 5 | TTNN composite | `tt/` | forward runs single-device | LoudBox dev 0 |
| 6 | Device correctness **+ depth** | `tests/` | per-read PCC ≥ analog threshold; 93-layer depth harness under the **relative** gate | LoudBox dev 0 |
| 7 | Remove fallbacks | §Backlog | no host fallback; production `T=5120` runs | LoudBox |
| 8 | Distribution | `DISTRIBUTION.md` → **judgment gate** → TP impl → `ROOFLINE.md` | memo ends at the gate, no code before a mapping is chosen; then PCC on real `(2,4)` | LoudBox `(2,4)` |
| 9 | Perf harness + perf loop | `tests/perf/` | numbered hypothesis→measure→keep-or-refute iterations; refutations recorded | `(8,1)` and `(2,4)` |
| 10 | Fused C++ op — **only if** the floor demands | `ttnn/cpp/.../attn_res/` | composed floor measured against the pre-committed roofline first | LoudBox |
| 11 | PP boundary contract | `PIPELINE.md` + socket round-trip test | `(1+S)·d` canonical layout defined and round-tripped through a `MeshSocket` pair | LoudBox, 2 submeshes |

---

## Gating discipline

Thresholds are inherited from the analog, not invented. Analog: the `_d_p` rmsnorm
tests use `assert_with_pcc(expected, actual, pcc=0.9999)`
(`tests/ttnn/utils_for_testing.py:94`).

| Test | Config | Gate |
|---|---|---|
| `test_folded_matches_hf_oracle` | `S ∈ {0,1,4,8}`, `d ∈ {256, 7168}`, 2 score scales | rel err ≤ 1e-5, PCC ≥ 1 − 1e-9 |
| `test_fold_does_not_degrade_accuracy` | `d = 7168` | `err(folded) ≤ max(4·err(reference), 1e-5)` vs fp64 |
| `test_block_split_matches_direct_form` | same, `R ∈ {1, 24}` | rel err ≤ 1e-5 vs the direct form |
| `test_lifecycle_seal_schedule` | 93-layer walk, `Bk=12` | seals exactly at `{0,12,…,84}`; `S` ramps 0→8; 185 reads + output |
| `test_forward_matches_torch_reference` | `S ∈ {0,1,4,8}`, `d ∈ {256, 7168}` | PCC ≥ 0.9999 **and** rel err ≤ 2e-2 (D13) |
| `test_split_matches_forward_on_device` | same, `R ∈ {1, 24}` | PCC ≥ 0.9999 against the direct TT form |
| `test_split_statistics_match_torch` | `S ∈ {1,8}` | rel err ≤ 2e-2 on `partial`, `shift`, `mass` |
| `test_mixture_weights_are_row_stochastic` | `C ∈ {1,5,9}` | `\|Σα − 1\| ≤ 1e-2` |
| `test_hand_rolled_softmax_beats_fused` | `C = 9`, fp32 | rel err no worse than `ttnn.softmax(dim=1)` (D12) |
| `test_saturated_scores_do_not_overflow` | `max\|score\| = 120`, `d ∈ {256, 7168}` | output finite (absolute); PCC ≥ torch-bf16 − 1e-3 (D16) |
| `test_values_are_not_normalized` | `S = 4`, `d = 7168` | output moves when a candidate is rescaled |
| `test_sharded_stream_is_rejected` | last dim ≠ `hidden_size` | raises, via the repo's `expect_error` fixture |
| `test_depth_fidelity` | 93 layers, random module outputs | **relative** gate, below, **plus** norm ratio within 2e-2 |
| `test_device_lifecycle_matches_torch` | 93-layer device walk, `Bk=12` | 186 reads; seals at `{0,12,…,84}`; `S` ramps 0→8 monotonically |
| `test_attn_res_dist_tp` | `(2,4)`, `S ∈ {4,8}` | PCC ≥ 0.9999 |
| `test_attn_res_perf` | `T ∈ {640, 5120}`, `(8,1)` and `(2,4)` | no assertion; numbers transcribed here |
| `test_pp_roundtrip` | 2 submeshes, `S=4` | bit-exact through the socket |

**The depth gate must be relative.** AttnRes sits on the residual highway for all 93
layers — 186 chained softmax mixtures. bf16 compounds through that on its own, so an
absolute PCC number would either be vacuous or fail for reasons unrelated to our
kernels. The gate is

> `PCC(TT, torch_fp32) ≥ PCC(torch_bf16, torch_fp32) − ε`

i.e. no worse than a bf16 torch implementation of the same math. Report the whole
per-layer PCC curve, not just the final number: a mid-stack trough can be a pure
precision artifact with end-to-end behaviour still correct. If a real checkpoint ever
lands, greedy-token agreement becomes the authoritative metric.

**But the relative depth gate is not a scale-defect detector.** Phase 6 mutation-tested
it: depth *dilutes* a per-read scale error instead of compounding it, so the D12 defect
walks all 93 layers and passes. The depth harness catches order-changing and
algebra-changing bugs; D13's op-level magnitude gate is what catches scale. Both are
required, and neither substitutes for the other — see §Learnings Phase 6.

Test hygiene — two defects found in the mHC test wiring, do not repeat:

1. Parametrize ids must follow `deepseek_v3_d_p/tests/pcc/mesh_configs.py`
   (`mesh-RxC`, `fabric2d-torus-y-RxC`, …). `ids=["mesh1x4","mesh2x4"]` matches no CI
   `-k` clause.
2. Carry `pytest.mark.requires_mesh_topology(mesh_shape=(R,C), topology="mesh-RxC")`.
   Without it the arch guard cannot skip a mismatched shape. Note also:
   **`(1,4)` is not a valid mesh on the 8-device LoudBox — only `{(8,1),(4,2),(2,4)}`.**
3. Run correctness under a hang-safe wrapper (`scripts/run_safe_pytest.sh --run-all`;
   `--dev` adds watcher + NoC sanitizer + auto-triage).
4. Keep a board-health probe whose only job is "does `(2,4)` FABRIC_2D open and an
   all-gather run" — isolates HW from our op in one command.

---

## Learnings

### Phase 0 — environment

`import ttnn` from the repo root resolves to the source directory as an implicit
namespace package: `ttnn.__file__ is None`, zero ops registered. The built extension
is only reachable through `python_env/`. Separately, checking out a different commit
without rebuilding leaves `ttnn/ttnn/*.py` ahead of `build/lib/_ttnn.so`, which
surfaces as `ModuleNotFoundError: No module named 'ttnn._ttnn.layer_completion';
'ttnn._ttnn' is not a package` — a stale-build symptom wearing an import error's
clothes.

*Lesson: "op not found" and "module not found" are both, until proven otherwise,
build-state claims rather than API claims. Confirm `ttnn.__file__` resolves under the
worktree before believing anything an import probe says.*

### Phase 4 — the oracle is fp32-internal, and so is the metric

Two things had to be fixed before the ladder measured the op rather than the harness.

**The vendored oracle cannot be its own high-precision reference.** It widens with
`v.float()`, which in torch means "cast to fp32", not "cast to floating point". Handed
fp64 inputs it still computes every reduction in fp32 and merely widens on return — so
an "fp64 ground truth" built by calling it with `.double()` arguments bit-matched the
fp32 call exactly, and the fold-degradation ratio became a division by zero. The fp64
ground truth has to be written independently. Our reference now promotes with
`torch.promote_types(dtype, float32)` instead, which is identical on every dtype the
model uses and can actually be evaluated in fp64. Corollary for Phase 6: upstream's own
numerics *are* fp32, so `torch_fp32` is the reference, not an approximation to something
better.

**`torch.corrcoef` in fp32 caps near 0.99999988** on `64 × 7168` = 458 k elements, even
for bit-identical inputs. A `PCC ≥ 1 − 1e-9` gate therefore failed at `d = 7168` and
passed at `d = 256` — a pure element-count effect that reads exactly like a
dimension-dependent op bug. PCC is computed in fp64 from here on.

**Score scale decides whether the tests test anything.** With a unit-variance query,
`⟨q, v⟩ ~ ±√d` (±85 at `d = 7168`), the softmax saturates to one-hot and every gate
below goes vacuous — the output is just the argmax candidate. Test queries are built the
way the model builds them (an RMSNorm gain near one times a `std ≈ 0.02` projection),
giving scores of order 1, and a deliberately saturated case is carried as a second
parametrization.

Measured rung-1 error, `S ∈ {1,4,8}` × `d ∈ {256, 7168}`: **1.5e-7 … 4.0e-7** against a
1e-5 gate. `S = 0` is bit-exact in both forms, as it must be — a one-candidate softmax
is exactly 1.0.

*Lesson: before believing a numeric gate, check that the metric and the ground truth are
more precise than the thing being gated. Both failed that check here, and both failure
modes impersonated a real op bug — one dimension-dependent, one dtype-dependent.*

### Phase 5 — a fused op that is worse than its parts, and a gate that could not see it

**`ttnn.softmax(dim=1)` loses ~4 % of the softmax mass.** Measured on `[1,9,64,1]`
against an fp64 reference:

| path | dtype | rel err | row sums |
|---|---|---|---|
| `ttnn.softmax(dim=1)` | fp32 | 1.4e-2 | 0.962 … 1.0003 |
| `ttnn.softmax(dim=1)` | bf16 | 1.9e-2 | 0.958 … 0.9999 |
| `max`/`exp`/`sum`/`div` | fp32 | 5.2e-4 | 1.0002 … 1.0011 |
| `max`/`exp`/`sum`/`div` | bf16 | 3.1e-3 | 0.9966 … 1.0034 |

Unchanged by `numeric_stable=False` and by `fp32_dest_acc_en=True`. `ttnn.exp` alone is
accurate to **6e-8** in fp32, so the deficit is the fallback reduction, not the
exponential — the docstring's "attention-optimized kernels require … reducing on the
last dimension" is the tell. Hence D12.

**PCC could not see it.** The first cut of `forward` used the fused op and passed
`PCC ≥ 0.9999` on all 8 `S × d` configs while carrying a 4 % magnitude error, because
PCC is scale-invariant and lost softmax mass is *exactly* a scale error. What caught it
was a side test comparing the two softmax paths to each other — not the gate. Hence D13:
magnitude alongside PCC everywhere, and `Σα = 1` promoted to a primary gate.

**PCC also cannot gate a constant reference.** At `S = 1` the `inter_block` mass is
exactly 1.0 for every token, and `corrcoef` of a constant vector is `nan`. The
`inter_block` statistics are gated on relative error instead. Measured, bf16, `d = 256`:
`partial` 2.0e-3 … 6.9e-3, `shift` 5.4e-3 … 6.3e-3, `mass` 0 … 5.0e-3 — all consistent
with bf16 storage at `2⁻⁸ = 3.9e-3` after a `d`-length reduction, against a 2e-2 gate.

**The merge algebra is self-correcting about `Z`.** A biased mass appears in both the
numerator and the denominator of `merge`, so it largely cancels — which is why the split
form clears PCC 0.9999 even where `mass` carries 5e-3. Good for robustness, bad for
diagnosis: gate the `inter_block` intermediates directly or a wrong `m` will hide inside
a compensating rescale.

**This box is Blackhole, not Wormhole** — pytest ids come back `blackhole-*`. Nothing in
the op depends on it yet, but `WormholeComputeKernelConfig` is the wrong name to reach
for reflexively, and the roofline constants in Phase 8 must be Blackhole's.

*Lesson: a fused op is not automatically the more accurate one, and "fewer launches" is
not a numerics argument. Measure the fused op against its own decomposition before
preferring it — and never gate a normalized quantity with a scale-invariant metric.*

### Phase 6 — depth dilutes a scale defect, so the depth gate cannot be the safety net

**The 93-layer harness is blind to D12.** Injecting each defect into the shipped op and
running the full depth walk, `d = 7168`, against the fp32 reference (torch-bf16 baseline
0.9999741, gate = baseline − 1e-3):

| injected defect | final PCC | verdict |
|---|---|---|
| none (as shipped) | 0.9999408 | PASS |
| `ttnn.softmax(dim=1)` — loses ~4 % of the mass per read | 0.9999205 | **PASS** |
| unshifted softmax (no max-subtraction) | 0.9999540 | **PASS** |
| values normalized before mixing | 0.5014967 | FAIL |

Two of the three real defects survive 186 chained reads. The reason is that a per-read
scale error does not compound: the mixture renormalizes, so a 4 % weight deficit shows up
as a 0.5 % shift in the final output norm, not as `1.04^186`. The norm-ratio check added
here (`|1 − ratio| ≤ 2e-2`, as-shipped 1.0037 at `d = 7168`) catches a *gross* scale
error and cannot resolve the 0.5 % case, so it is documented as exactly that. The
op-level gate is the primary detector for this defect class, and it does fire — the same
injection fails 6 of 8 `forward` configs plus two row-stochastic configs.

*Lesson: pick the instrument for the defect class. Depth compounds order and algebra
errors and dilutes scale errors, so a depth harness cannot be the safety net under an
op-level gate — it is a different measurement, not a stronger one.*

**The device trails torch-bf16 more as scores saturate, and it is the `v ⊙ q` product.**
Sweeping the query scale, PCC against the fp32 reference, both arms finite everywhere:

| `max\|score\|` | device `d=256` | torch-bf16 `d=256` | device `d=7168` | torch-bf16 `d=7168` |
|---|---|---|---|---|
| 1.0 | 0.9999910 | 0.9999969 | 0.9999798 | 0.9999945 |
| 8.0 | 0.9999687 | 0.9999916 | 0.9999770 | 0.9999932 |
| 120.0 | 0.9998480 | 0.9999775 | 0.9998749 | 0.9999570 |

The gap widens from ~1.5e-5 to ~1.3e-4. "Inherent bf16" does not explain it: the torch-bf16
arm stores bf16 too. What differs is *where* the rounding lands — `attn_res` promotes with
`_at_least_fp32` and computes the score reduction in fp32, while on device the `v ⊙ q`
product is a bf16 `[1,C,N,d]` tensor before it is summed. At `|score| = 120` bf16's
mantissa step is 0.5, so candidate score *differences* quantize to 0.5 and `exp(−Δ)`
moves by 1.65×. Closing it means an fp32 `[1,C,N,d]` intermediate — 2× traffic on the
op's largest tensor, on a bandwidth-bound op, for a regime the folded query
(`|score| ≈ 5` over 9 candidates) never reaches. Not fixed; gated relatively per D16 and
recorded here so Phase 9 does not rediscover it as a mystery.

**186 reads, 187 parameter sets.** These are different numbers and conflating them cost a
false assertion failure. The walk executes 185 in-layer reads plus the model-level read;
the query count is `2·93 + 1 = 187` because `q_pre[0]` is loaded and never used — the
layer-0 pre-attention read is skipped at `S = 0`. The test now asserts
`parameter_sets == executed_reads + 1` so the relationship is stated rather than assumed.

**The layer-0 read skip is an ownership hazard on device only.** Because that read is
skipped, `h` aliases the stream's own `prefix_sum`, which `seal` then takes ownership of
(D15) — so freeing `h` after the module call frees `block_residual` out from under the
next 92 layers. In torch this is invisible (GC), which is exactly why the shared walk
carries an explicit `borrowed` flag instead of an unconditional `free`.

*Lesson: interface-compatible backends make lifecycle bugs impossible to hide, but they
do not make ownership uniform. The reference path can afford to be sloppy about who frees
what; the device path cannot, and the shared walk has to encode the difference.*

---

## Backlog

- [ ] Phase 1 — infra map with `file:line` citations and inherited thresholds.
- [ ] Phase 2 — delta analysis; `Missing/blocked ops` list.
- [x] Phase 3 — `API_SPEC.md`.
- [x] Phase 4 — `torch_functional/`, numeric ladder (D9, amended by D11).
- [x] Phase 5 — `tt/` composite, single device.
- [ ] Measure the `N`-batched matmul `[N,R,S] × [N,S,d]` against 24 broadcast passes;
      decides whether the split form's remaining ~1.9× is reachable in composed ops.
- [x] Phase 6 — device correctness + 93-layer depth harness.
- [ ] Phase 7 — remove host fallbacks; `T=5120`.
- [ ] Phase 8 — `DISTRIBUTION.md`, TP on `(2,4)`, `ROOFLINE.md`.
- [ ] Phase 9 — perf harness + numbered perf loop.
- [ ] Phase 10 — fused C++ op, only on measured evidence.
- [ ] Phase 11 — `PIPELINE.md` + socket round-trip.
- [ ] Fold `rms_inv` at seal time (needs the batched form first).
- [ ] Decode path (`T=1`).
- [ ] Real K3 weight loader.
- [ ] Flag to mvasilijevic: shipped `modeling_kimi_linear.py:520-521` allocates
      `A_log` as `[num_heads] = [96]`, but the checkpoint stores `F32 [128]` =
      `head_dim` (verified in two independent shards). Trust the checkpoint —
      per-channel decay, consistent with KDA's channel-wise gate.

---

## Progress

Append-only. UTC timestamps. `PASS` / `FAIL` bolded.

- **2026-07-30** — Branched `nmilicevic/bringup/kimi-k3-attnres-2026-07-30` off `main`
  @ `6d526e8d61d`. Phase 0 scope, nomenclature and decisions D1–D10 recorded.
- **2026-07-30** — Phase 0 environment gate **FAIL**: `import ttnn` under the system
  interpreter gives `__file__ = None`; under `python_env/` it reaches the extension but
  `build/lib/_ttnn.so` predates the checked-out `main`. See §Learnings Phase 0.
- **2026-07-30** — Phase 0 **PASS** after `git submodule update --init --recursive` +
  `bash build_metal.sh`. `ttnn.__file__` = `<worktree>/ttnn/ttnn/__init__.py`; every op
  the composite form needs is present on the module.
- **2026-07-30** — Phase 3 **PASS**: `API_SPEC.md` written. Amended once inside the
  phase, before any code depended on it, per D11.
- **2026-07-30** — Phase 4 **PASS**: `reference/hf_attn_res.py` (vendored oracle) and
  `torch_functional/attn_res.py`. **41/41** in
  `tests/test_torch_attn_res.py`. Two harness defects found and fixed first — see
  §Learnings Phase 4. Rung-1 error 1.5e-7 … 4.0e-7 against a 1e-5 gate.
  **Command:** `PYTHONPATH=$TT_METAL_HOME python_env/bin/python -m pytest
  models/experimental/kimi_k3_attn_res/tests/test_torch_attn_res.py -q`

  **VALIDATED:** The folded reference reproduces the upstream read across
  `S ∈ {0,1,4,8}` × `d ∈ {256,7168}` × two score regimes, without degrading accuracy
  against fp64. The inter-block/merge split reproduces the direct form for every read
  site at `R ∈ {1,24}`. The lifecycle seals exactly at `{0,12,…,84}`, `S` ramps 0→8, and
  the walk performs 185 in-layer reads plus the model-level read. Values are confirmed
  un-normalized by a scale-invariance probe.

  **NOT VALIDATED:** Anything on device — no `tt/` code exists yet. Depth compounding
  over 93 layers. Distribution. Real K3 weights. Decode.
- **2026-07-30** — Phase 5 op-surface probe **PASS**: all 14 ttnn ops the D10 dim-1 form
  needs exist with the right shapes and broadcast semantics on device 0, including all
  three `ttnn.mul` broadcast patterns (`[1,C,N,d]` against `[1,1,1,d]`, `[1,C,N,1]`,
  `[1,1,N,1]`) and `sum`/`max` on both dim 1 and dim 3 with `keepdim`. The flat dim-1
  chain reproduced the torch reference at PCC 0.9999903, `max|Δ|` 7.8e-3, all-bf16.
- **2026-07-30** — Phase 5 divergences from the frozen `API_SPEC.md` §5, all recorded
  rather than edited into the spec:
  1. **`block_residual = None` replaces `[1, 0, N, d]`.** ttnn has no zero-extent
     dimension, so `S = 0` cannot be a real tensor the way it is in torch. `forward`
     short-circuits to a clone — a fresh tensor, not the input, so the caller's
     deallocation is uniform across both paths.
  2. **`inter_block` / `merge` take and return Python lists**, one entry per read site,
     not the stacked `[R, N, d]` of the torch signature. Stacking read sites on a tensor
     axis buys nothing while the weighted sum is still a per-read broadcast pass.
  3. **`input_memcfg`, `stats_memcfg`, `weight_cache_path`, `cache_name_prefix` are not
     accepted yet.** They are Phase 8/9 knobs with nothing to configure while everything
     is DRAM-interleaved and the only weights are 187 `[d]` vectors. `cluster_axis`,
     `num_links` and `topology` *are* accepted and stored, so the distribution shape is
     visible in the signature from the start.
- **2026-07-30** — Phase 5 **PASS**: `tt/attn_res.py` — `forward` (direct form) and
  `inter_block` / `merge` (the block-batched split). **28/28** in
  `tests/test_tt_attn_res.py`, single device.
  **Command:** `PYTHONPATH=$TT_METAL_HOME python_env/bin/python -m pytest
  models/experimental/kimi_k3_attn_res/tests/test_tt_attn_res.py -q -p no:randomly`

  **VALIDATED:** `forward` matches `torch_functional.attn_res` across
  `S ∈ {0,1,4,8}` × `d ∈ {256,7168}` at PCC ≥ 0.9999 **and** rel err ≤ 2e-2.
  `inter_block` + `merge` reproduces `forward` for every read site at
  `R ∈ {1,24}` × `S ∈ {1,4,8}` × `d ∈ {256,7168}` — 24 reads is the real per-block count.
  The `inter_block` statistics `partial`/`shift`/`mass` are gated individually so a wrong
  `m` cannot hide in a compensating rescale. Mixture weights sum to 1 within 1e-2 at
  `C ∈ {1,5,9}`. Values confirmed un-normalized on device by the scale-invariance probe.
  A sharded stream is rejected rather than silently reduced per-shard.

  **NOT VALIDATED:** Depth compounding over 93 layers — the `AttnResStream` lifecycle has
  no device counterpart yet, so nothing has run more than one read deep. Production `T`;
  every test here is `N = 64`. Any performance claim — no perf run, and the split form is
  written for correctness, not traffic (see §The inter/intra-block split, Phase 5
  revision). Distribution, real K3 weights, decode.
- **2026-07-30** — Phase 6 **PASS**: `tt/attn_res_stream.py` (device `block_residual`
  lifecycle, D15) and `tests/test_tt_attn_res_depth.py` (the 93-layer harness, D14).
  **74/74** across the three test modules, whole suite in ~41 s.
  **Command:** `PYTHONPATH=$TT_METAL_HOME python_env/bin/python -m pytest
  models/experimental/kimi_k3_attn_res/tests/ -q`

  **VALIDATED:** A full 93-layer walk runs on device — 186 chained reads, 8 seals,
  `S` ramping 0→8 — and holds the relative gate at both `d = 256` (device 0.9999791 vs
  torch-bf16 0.9999922, worst layer 47: 0.9999545 vs 0.9999724) and `d = 7168` (device
  0.9999408 vs 0.9999741, worst layer 92: 0.9999105 vs 0.9999501), with output norm ratios
  1.000531 and 1.003655. The reference and device walks are the *same* walk, parametrized
  only by `apply_module` and `free`, so the seal schedule cannot silently diverge; it is
  separately asserted to fire at `{0,12,…,84}` with a monotonic snapshot count. Stream
  norms stay in regime (14.0→7.8 and 84.9→65.1), so the PCCs are not measuring a decayed
  or overflowed stream. The max-subtraction in the candidate softmax is gated by a
  saturated case at `max|score| = 120`; removing the shift makes the output non-finite in
  both configs, which the test catches.

  **NOT VALIDATED:** Scale defects *at depth* — mutation testing shows the depth gate
  passes both `ttnn.softmax(dim=1)` and an unshifted softmax; the op-level D13 gate is the
  detector for that class. Production `T` — still `N = 64` everywhere. Real module
  outputs: `apply_module` is an elementwise `h ⊙ w`, which exercises the residual
  bookkeeping but not KDA/MLA/MoE traffic patterns. Memory headroom at production shape —
  `block_residual` reaches `[1, 8, T, d]` and nothing has measured that against DRAM at
  `T = 5120`. Any performance claim. Distribution, real K3 weights, decode.
