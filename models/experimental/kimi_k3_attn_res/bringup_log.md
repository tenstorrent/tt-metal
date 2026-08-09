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

**Ground truth** is `reference/attn_res_reference.py` — the unfolded read written from
the published definition, computed in fp64, and pinned by closed forms rather than by
comparison with another implementation. It shares no algebra with the folded form that
`torch_functional/` and `tt/` implement, which is what lets rung 1 prove the fold.

**The external anchor** is `reference/hf_attn_res.py` — upstream's `_apply_attn_res`
verbatim from `modeling_kimi_linear.py:1075-1088`. It is the only gate in the module that
compares against something we did not write, so it is the only one that can catch the whole
ladder agreeing on the wrong equation. It cannot be the root: it widens with `.float()`, so
it computes in fp32 whatever it is handed. Anchor for the algebra, `ref` for the precision.
It carries the Kimi K3 License rather than Apache-2.0 — see `GALAXY.md` §8.

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
- **D17 — 2026-07-30 — the op owns the distributed layout, including `sp_axis`.**
  `TtAttnRes` exposes `stream_mapper`, `vector_mapper` and `stream_composer`, and
  `forward` checks its input's last dim against `hidden_size // tp_factor`. It never
  communicates on `sp_axis` — that axis is placement only — but it still names it, so the
  layout has exactly one definition. Rejected alternative — leave placement to callers and
  take only `tp_axis` — which means the op's `hidden_size` (global) and the caller's shard
  width are agreed by convention rather than by assertion, and that agreement is the one
  place a sharded AttnRes returns quietly wrong numbers instead of failing: divide by the
  local width and every score is off by `tp_factor`, with no shape error anywhere.
- **D18 — 2026-07-30 — the statistics reduction is one `ttnn.all_reduce` of a dim-1 stack,
  in fp32.** `all_reduce` over `all_gather` + post-op because its output shape equals its
  input shape — no strided "sum every `C`-th column" in composed ops — and because
  `all_gather` silently takes a slow composite path when the gather dim has padded tiles
  (`all_gather_nanobind.cpp:39`), which a 1- or 2-wide stats dim always does. Stacked on
  **dim 1** rather than the last dim so the halves come back apart on a tile-plane
  boundary instead of a sub-tile read; that doubles a payload which is 0.65 % of the op's
  traffic. fp32 because `all_reduce` otherwise reduces in bf16 (`all_reduce_nanobind.cpp:48`)
  and fp32 measures 0.9999500 against bf16's 0.9999401 over 186 chained reads, for 1.5 MB
  per read on a 900 MB budget. Rejected alternative — the analog's `all_gather` + `strided
  sum`, which is both slower here and needs an op we do not have.
- **D19 — 2026-08-06 — amends D9 and D11 again: the root of the ladder is an unfolded fp64
  reference of our own, and the vendored oracle sits beside it, not under it.** D11 left the
  ladder starting at "the folded form versus the oracle", which was circular for the algebra
  in a way I did not see at the time: every TTNN test rooted on `torch_functional/attn_res.py`,
  and that file *is* the folded form — same fold, same rsqrt pull-out as the device op. It gates
  numerics and plumbing and can never gate the algebra. The only unfolded check was a
  test-local `_exact` helper. So rung 0 is now `reference/attn_res_reference.py`, deliberately
  naive, fp64, pinned by three closed forms where the answer is known outright rather than by
  agreement with anything. Rung 0b keeps the vendored oracle as the *external* anchor — the one
  thing no reference of ours can supply, evidence that upstream computes the equation we believe
  it does. Two roles, because the oracle cannot fill both: it widens with `.float()`, so it
  cannot be a precision reference, and it is ours-by-copying, so it cannot be pinned
  intrinsically. Rejected alternative — keep the oracle as the sole root, which is what D11 did
  and what left the algebra ungated. What proved the new root works is mutation testing, not its
  own suite passing: seven injected porting errors, each caught, and the run found two real holes
  (a `sum`-for-`mean` softmax-temperature slip and a dropped `res_norm` gain) that only agreement
  with the implementation under test had been catching. Both are now closed by the
  constant-along-`d` closed form.

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

> **Amended 2026-07-30 (Phase 8, `ROOFLINE.md`).** The heading is half right, and which
> half depends on tracing. Two corrections. First, "2 passes over `v`" understates the
> composed form by 6×: counting DRAM touches op by op gives **12V**, not 2V — the concat,
> and a full-`V` intermediate for each of the three `mul`s. So 172 MB/token, and a
> 215.5 ms DRAM floor per forward at `T = 5120` on `(2,4)`. Second, arithmetic intensity
> is **0.25 flop/byte**, not 1.8 — the 1.8 assumed the naive pass count and counted the
> `S`-fold reuse of `q` as work. Bandwidth-bound is therefore *more* true than Phase 0
> thought. But the launch term was never priced: measured, one ttnn call costs ~130 µs
> untraced against an 88 µs-per-launch break-even, so **untraced the composed op is
> launch-bound even at production shape**, and traced it is DRAM-bound by 69×. KDA's
> "kernel count is the only lever" transfers exactly in the regime we develop in and not
> at all in the regime we ship in. `ROOFLINE.md` §2, §3, §6.

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

**Phase 7 measures 1.50×, above that estimate.** Wall clock, warm, `T=5120`, `S=8`,
24 read sites, single device: direct 516.2 ms (21.5 ms/read) versus split 343.4 ms
(14.3 ms/read). The 1.3× estimate undercounted because the split form also drops the
`concat` entirely — it never materializes `v` — and its mixture pass covers 8 candidates
plus a 1-candidate merge instead of 9. Not a Phase-9 result: wall clock, no profiler, one
shape, and the live stream is held constant across the 24 merges (timing-neutral, since
`merge` recomputes from `prefix_sum` every call, but not a realistic value pattern).

A sealed snapshot's RMS never changes after sealing, so fold `rms_inv` for sealed
candidates at **seal time**: the per-block pass then needs only dot products, and the
live stream's statistics are the only per-read reduction.

### TP is nearly free; PP is not

Both reductions are over `d`. With `d` TP-sharded, AttnRes needs an all-reduce of
`2(S+1)` **scalars** per token per read — the `tt_distributed_rms_norm.py:236-290`
pattern (communicate statistics, never the stream). Combine the sum-of-squares and the
dot into one payload.

> **Amended by Phase 8.** The sentence that used to follow — "with the block batching
> above, that is one small all-reduce per block for 24 reads plus one 2-scalar reduce per
> live-stream read" — is wrong, and wrong in the unflattering direction. Each read site
> still needs its own dot against the sealed set, so `inter_block` amortizes the sealed
> **RMS** and nothing else: 49 collectives for 24 read sites against the direct form's 24.
> See `DISTRIBUTION.md` §4 and the Phase 8 learnings.

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

Thresholds are not invented — but they are **not** inherited from the analog either, and
this section said they were until Phase 1 was back-filled. The `_d_p` rmsnorm tests pass
`pcc=0.99` explicitly (`models/demos/deepseek_v3_d_p/tests/pcc/test_rmsnorm.py:137` for the
distributed form, `:205` for single-chip). `0.9999` is `assert_with_pcc`'s *signature
default* (`tests/ttnn/utils_for_testing.py:94`). We hold the repo-wide default, which is
100× stricter than the nearest analog's deliberate loosening; the analog's 0.99 is a floor
we must not fall below, not the target. See §Learnings Phase 1.

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
| `test_unexpected_shard_width_is_rejected` | last dim ≠ `hidden_size / tp_factor` | raises, via the repo's `expect_error` fixture |
| `test_depth_fidelity` | 93 layers, random module outputs | **relative** gate, below, **plus** norm ratio within 2e-2 |
| `test_device_lifecycle_matches_torch` | 93-layer device walk, `Bk=12` | 186 reads; seals at `{0,12,…,84}`; `S` ramps 0→8 monotonically |
| `test_production_forward_matches_torch` | `T = 5120`, `S ∈ {0,8}` | PCC ≥ 0.9999 **and** rel err ≤ 2e-2 |
| `test_production_split_matches_forward` | `T = 5120`, `S = 8`, 24 read sites | PCC ≥ 0.9999 against the direct form |
| `test_ragged_token_count_matches_torch` | `T ∈ {1000, 5119}`, `S = 8` | PCC ≥ 0.9999 **and** rel err ≤ 2e-2 across a tile-padded `T` |
| `test_token_axis_is_pure_batch` | `T ∈ {64, 1000, 5120}`, `S = 8` | `max\|Δ\| == 0` on the shared token slice |
| `test_production_depth_walk` | 93 layers at `T = 5120` | `max\|Δ\| == 0` vs the `T = 64` walk; seal schedule; finite |
| `test_tp_forward_matches_torch` | `(2,4)`, `S ∈ {0,1,8}`, `d ∈ {256,7168}` | PCC ≥ 0.9999 **and** rel err ≤ 2e-2 |
| `test_tp_split_matches_forward` | `(2,4)`, `d = 7168`, 24 read sites | PCC ≥ 0.9999 against the direct form |
| `test_sequence_axis_communicates_nothing` | `(2,4)`, sharded vs replicated sequence | `max\|Δ\| == 0` on the shared tokens, **and** the two SP rows agree to 0 |
| `test_statistics_reduction_is_load_bearing` | `(2,4)`, `_reduce_stats` stubbed out | PCC **< 0.9999** — the mutation must fail the gate |
| `test_tp_depth_walk` | `(2,4)`, 93 layers, 186 collectives | **relative** to torch-bf16 (−1e-3) **plus** norm ratio within 2e-2 |
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

## Phase 9 perf loop

`tests/perf/test_attn_res_perf.py`. No timing assertions anywhere — the file measures and
logs, the ledger holds the verdicts. Every row is `T = 5120`, 20 iterations after 3
warmups, on an otherwise idle LoudBox. **Traced** rows replay a captured trace and are
device time; **untraced** rows are Python-driven and include host.

The method, and its one trap: time the host enqueue of R iterations with no readback,
then time to `synchronize_device`. `enqueue ≈ total` is **ambiguous** — it means the
device is idle *or* host and device are running alongside each other at the same rate.
Only a traced run, where enqueue collapses to ~10 µs, reports device time on its own.
P1 fell into this trap and P2 pulled it back out.

`tests/perf/` sits under `tests/`, per the ladder and `deepseek_v3_d_p`'s layout, so the
correctness sweep now has to exclude it — 30 device-heavy parametrizations with no gate:

```
correctness  pytest models/experimental/kimi_k3_attn_res/tests/ --ignore=models/experimental/kimi_k3_attn_res/tests/perf -q
perf         pytest models/experimental/kimi_k3_attn_res/tests/perf/ -q -p no:randomly
```

The perf file carries no pytest marker on purpose. `models_device_performance_bare_metal`
is what CI selects on, and claiming it for a harness with no assertions would put 30
ungated tests into a regression job.

### P1 — the launch term, untraced (`S = 8`, peak shape)

*Hypothesis (`ROOFLINE.md` §6):* ~130 µs per launch × 22 launches = 2.86 ms of host
against a 1.94 ms DRAM floor, so the composed op is launch-bound even at production shape.

| placement | `num_links` | enqueue µs | total µs | µs per ttnn call |
|---|---|---|---|---|
| `(1, 1)` | — | 1 678 | **21 511** | 105 |
| `(8, 1)` | 1 | 2 415 | 2 765 | 151 |
| `(2, 4)` | 1 | 3 348 | 3 378 | 152 |
| `(2, 4)` | 2 | 3 385 | 3 414 | 154 |

The `(1, 1)` row is the control that validates the method: host finishes 12.8× early, and
21.5 ms reproduces Phase 7's 21.6 ms independently. A launch costs **105 µs on one device
and ~152 µs on eight** — so 8-device fan-out is ~46 µs of it, which `ROOFLINE.md` §8
listed as unknown.

*Verdict:* **refuted** — and I read my own table wrong first. On `(2, 4)` enqueue was 99%
of total, which I labelled "host-bound". P2 shows the device needs 3 282 µs of that
3 378 µs — the enqueue was **overlapped, not blocking**. Dispatch is pipelined: the cost
is `max(host, device)`, never the sum. §6's per-read verdict is wrong at production shape,
where the two terms are within 2% of each other.

### P2 — traced, so device time alone (µs per read)

| `S` | `(1, 1)` | `(8, 1)` | `(2, 4)` |
|---|---|---|---|
| 1 | 5 010 | 780 | 886 |
| 4 | 12 218 | 1 571 | 1 934 |
| 8 | 21 515 | 2 766 | 3 282 |

Traced enqueue is 2.6–10 µs, a **345× reduction** in host time — and total moves ≤3% at
`S = 8`. Fits: `(2, 4)` device ≈ **201 + 342·(S+1)** µs, `(8, 1)` ≈ 213 + 284·(S+1).
Sequence sharding scales 8.3× on the slope — free, as M1 claimed.

Against §3's 1 935 µs DRAM floor: `(8, 1)` runs at **70% of DRAM peak**, `(2, 4)` at 59%.
So "DRAM-bound traced" is confirmed and now has a number — there is 1.43× of headroom in
the composed form before Phase 10's rewrite is the only lever left.

**TP costs 516 µs per read** at `S = 8` (18.7%), of which the standalone all-reduce at
that exact payload is 348 µs (P4) and the remaining ~168 µs is the fp32 typecast pair,
the candidate concat and the two slices.

### P3 — the same sweep untraced, which is where the launch term actually lives

`(1, 1)` untraced is 5 002 / 12 212 / 21 518 µs — within 0.2% of traced, because 16
launches at 105 µs never approach 21 ms of device time. On eight devices the untraced
totals **pin flat at a 2.2–3.6 ms host floor regardless of `S`**, so at `S = 1` the op
burns 3.6× its device time waiting on Python.

*Verdict:* **kept**, with the shape corrected. The launch term binds at the **schedule**,
not at the peak shape. `S` ramps 0→8 across the 93 layers, mean `(S+1) = 5.39`, so most
of the 186 reads sit in the small-`S` regime where host wins outright. Over the real
schedule on `(2, 4)`:

```
traced    186·201 µs + 1002·342 µs  =  380 ms per forward   (device)
untraced  186 · 22 · 152 µs         =  622 ms per forward   (host, and it wins)
```

**Tracing is worth 1.64× per forward** and ~1.00× at the peak shape. Measuring only the
peak shape would have priced tracing at zero.

### Profiler attribution — `(2, 4)`, `S = 8`, untraced

650 device programs over 23 reads = **28.3 programs per read** (D10's decision text says
~12; `ROOFLINE.md` §6 guessed ~25). 4 425 µs per read of device FW time, ~35% above the
traced 3 282 µs, which is the profiler's own instrumentation cost.

| what | µs | share |
|---|---|---|
| 7 big-tensor ops — concat 521, `mul(v,v)` 722, `sum` 319, `mul(v,q)` 604, `sum` 326, `mul(v,w)` 464, `FastReduceNC` mix 407 | 3 363 | **76%** |
| ReduceScatter 263 (17 cores) + AllGather 342 (**2 cores**) | 605 | 13% |
| fp32 typecast pair | 141 | 3% |

The whole statistics path is **23% of device time for 0.6% of the bytes**
(`ROOFLINE.md` §4). And `ttnn.max(dim=1)` is not native: FillPad 18.3 + Transpose 43.4 +
Reduce 46.7 + Transpose 19.2 = **128 µs to take a max over 9 elements** — the price D12's
hand-rolled softmax already pays and D18 should watch.

### P4 — what the collective actually charges for (traced, `(2, 4)`)

| shape | padded | useful | `links = 1` | `links = 2` |
|---|---|---|---|---|
| `[1, 18, 2560, 1]` — today's stats | 5 760 KiB | 184 KiB | **348.1** | **235.9** |
| `[1, 18, 2560, 32]` | 5 760 KiB | 5 760 KiB | 348.2 | 235.5 |
| `[1, 1, 2560, 18]` — folded stats | 320 KiB | 180 KiB | **46.8** | 50.2 |
| `[1, 2, 2560, 1]` | 640 KiB | 20 KiB | 63.0 | 62.4 |

Three readings, and the first two are the phase's most useful facts:

1. **Padded bytes cost exactly what useful bytes cost.** 184 KiB of payload in a
   5 760 KiB envelope costs the same 348 µs as 5 760 KiB of real data. The collective is
   payload-bound at **~18 KiB/µs above a ~29 µs floor** — 18–25% of fabric peak, and
   core-limited, since AllGather runs on **2 worker cores**.
2. So **folding the candidate axis into the last dim is worth 7.4× on the collective** —
   348 → 47 µs, ~300 µs per read, against two permutes worth ~40–120 µs traced.
3. `num_links = 2` is worth **1.48×, but only at the big payload**, and is
   neutral-to-negative at ≤640 KiB. The fold and `links = 2` are **alternatives, not
   additive**: after the fold there is no payload left for the second link to help.

*Verdict:* `ROOFLINE.md` §4's fabric model is **off by 2.7×** (88.5 µs predicted at
`links = 2`, 235.9 measured) and its deferral of the padding fix is **refuted on device
time** — correctly reasoned untraced, where two extra launches at 152 µs cancel the
saving exactly, and wrong traced, where the fix is a clean 5–8% of the read.

### P5 — does the split form survive 2× the collectives?

The last open question from Phase 8. Traced, a full 24-site block on `(2, 4)`, per read
site: direct **3 274.6 µs** vs split **2 228.3 µs**.

*Verdict:* **kept. 1.47× on a TP mesh**, against 1.50× measured on one device in Phase 7,
while issuing 49 collectives per block against the direct form's 24. Amortizing the sealed
half's RMS pass across 12 layers is worth ~1 047 µs per site; the extra collective costs
~350 µs. The split form is not a single-device artifact.

*Re-measured after P6* (the fold is now the default): direct **3 127.8** vs split
**2 186.6**, so **1.43×**. The ratio moved because the fold is worth 146.8 µs per site to
the direct form and only 41.7 to the split one — the direct form's collective carries all
`2(S+1) = 18` stats planes per site, the split form's two carry ~10 between them. The split
form was already spending less on the collective, which is part of *why* it was faster, so
the fold takes back some of its edge. The conclusion is unchanged; the number is 1.43×.
(P5's first run also leaked its trace between the two forms — but the direct number lands
within 0.3% of P6's independent unfolded row, so the fold explains the shift and the leak
cost nothing measurable.)

### P6 — the statistics fold, priced on the real op

P4's 7.4× was measured on a bare `all_reduce`. This is the same fold inside the op, behind
`fold_stats`, paying for its own two `ttnn.permute` calls. Hypothesis going in: worth
~300 µs per read at the peak shape, and **widening as `S` falls**, since at small `S` the
unfolded payload is nearly all padding while the folded one is a single tile row either way.

Traced, `(2, 4)`, µs per read:

| `S` | links | unfolded | folded | fold Δ |
|---|---|---|---|---|
| 1 | 1 | 885.7 | 867.4 | −18.3 |
| 1 | 2 | **858.9** | 864.1 | +5.2 |
| 8 | 1 | 3 283.9 | 3 136.3 | **−147.6** |
| 8 | 2 | 3 174.2 | **3 131.6** | −42.6 |

1. **The fold is worth 147.6 µs (4.5%) at the peak shape — half of what P4 priced it at.**
   The permute pair costs ~153 µs, above the ~40–120 µs I guessed. P4 timed the collective
   in isolation and never charged for getting into and out of the layout.
2. **P6's own hypothesis is refuted.** At `S = 1` the fold is worth 18.3 µs, and
   `links = 2` unfolded is the *best* row there. The permutes track the padded tensor
   exactly as the collective does, so both terms shrink together and the ratio never moves.
   There is no small-`S` regime where the fold wins bigger. Net saving fits
   **18.6·(S+1) − 18 µs** (149 predicted vs 147.6 at `S=8`; 19.2 vs 18.3 at `S=1`), so over
   the real schedule — 186 reads, Σ(S+1) = 1 002 — it is **15.3 ms of the 380 ms forward,
   4.0%**.
3. **P4's "alternatives, not additive" is confirmed on the real op.** `links = 2` alone
   buys 109.7 µs; the fold alone buys 147.6; both together buy 152.3. The second link adds
   **4.7 µs** on top of the fold.

Correctness gate — the 186-read depth walk, both layouts at both `T` (at `T = 64` the two
layouts can take *different* `all_reduce` algorithms, not just reassociate, so both are
parametrized):

| `T` | folded | unfolded | torch-bf16 analog |
|---|---|---|---|
| 64 | 0.9999545 | 0.9999500 | 0.9999741 |
| 256 | 0.9999207 | 0.9999223 | 0.9999638 |

±5e-6 **in both directions** — reassociation noise, not a precision cost. A probe against a
replicated 4× fp32 reference showed *neither* layout is bit-exact through `ttnn.all_reduce`
(unfolded 7.8e-3, folded 1.6e-2 at `C = 18`, both ~50× inside one bf16 ULP), so exactness
was never the right gate; 186-read depth PCC is.

*Verdict:* **`fold_stats=True` by default, `num_links=1` by default.** The fold is a strict
improvement at every measured point but `(S=1, links=2)`, where it costs 5.2 µs. That
settles the `num_links` question in the direction Phase 8 did not expect — leave it at 1,
because the fold makes the second link worthless, and on Galaxy a link not taken is a link
another op can have.

### P7 — the `d`-wide reductions, where the time actually was

P4 through P6 spent three iterations on the collective and won 4.0% of the forward. The
tracy profile had already said the weight was elsewhere: seven ops touch the full
`[1, C, N, d/tp]` tensor and account for 76% of the read's device time, and both of the
op's `d`-reductions were written as `mul` then `sum` — an elementwise pass that writes a
second copy of the largest tensor in the op to DRAM, then a reduce that reads it back.
**Three passes over 79 MiB to produce 0.6% of the op's bytes.**

Two composed primitives do it in one. `ttnn.rms_norm_pre_all_gather` is the
distributed-RMSNorm statistics kernel: it squares inside the reduce, returns `Σx²` (not the
mean) in column 0 of a 32-wide output, batched over leading dims — exactly the shape
`_local_sum_squares` has. And `_local_dots` is a matvec, so `ttnn.matmul` against `q` as a
column needs no intermediate at all.

Traced on `(2, 4)`, one variant per trace, `[1, 9, 2560, 1792]` bf16 = 79 MiB in:

| form | µs | ×floor | admissible? |
|---|---|---|---|
| **floor**: `sum(v)` — one pass, no intermediate | 229.0 | 1.00 | control |
| sumsq: `mul` + `sum` (what the op did) | 781.6 | 3.41 | — |
| sumsq: `rms_norm_pre_all_gather` | 229.5 | 1.00 | **no** — 4.78e-2 |
| sumsq: `rms_norm_pre_all_gather` **HiFi4** | **232.3** | 1.01 | **yes** — 2.54e-3 |
| dots: `mul` + `sum` (what the op did) | 792.5 | 3.46 | — |
| dots: `matmul` | 439.4 | 1.92 | **no** — 1.28e-2 |
| dots: `matmul` **HiFi4** | **450.1** | 1.97 | **yes** — 3.155e-3 |
| mix: `mul` + `sum` over the candidate axis | 791.1 | 3.45 | — |
| **floor**: `sum(v, dim=1)` | 228.2 | 1.00 | control |

**Amended by P9:** the mix row above sits **15% (102 µs) above** the shape the op runs. It
multiplies by `q`, reusing the matvec's `[1, 1, 1, d/tp]` operand, where `_mix` multiplies
by a `[1, C, N, 1]` weight. On that shape the mixture is **688.3 µs at 3.01× floor** (mean
of two runs). Readings 1–3 below are unaffected; they concern the two `d`-reductions, whose
rows are correct.

1. **Fidelity is part of the candidate, not a knob.** At default (LoFi) fidelity both
   one-pass forms lose an order of magnitude — the statistics kernel goes to 4.78e-2 against
   the 2.44e-3 that `mul` + `sum` achieves, the matvec to 1.28e-2. HiFi4 with
   `fp32_dest_acc_en` restores both (2.54e-3 and 3.155e-3, against today's 2.44e-3 and
   3.189e-3) and is **free on device**:
   232.3 against 229.5 µs, 450.1 against 439.4. These reductions are bandwidth-bound, so the
   extra math passes hide under the reads. Had the accuracy been checked after the timing,
   the honest version of this row would have been 2% slower and the fast one wrong.

   **Corrected after P9 (this row called both defaults "LoFi"; only one of them is).**
   `rms_norm_pre_all_gather` *already defaults to HiFi4* — with `fp32_dest_acc_en=false` and
   `math_approx_mode=true` (`rmsnorm_pre_all_gather.cpp:24`). So the squares' 4.78e-2 →
   2.54e-3 is **not a fidelity effect**; it is `fp32_dest_acc_en`, and possibly approx mode.
   `ttnn.matmul` on bf16 inputs does default to LoFi (`matmul_device_operation.cpp`:
   `increase_fidelity ? HiFi2 : LoFi`), so the matvec's 1.28e-2 → 3.155e-3 is fidelity plus
   dest accumulation. The reading survives with its mechanism swapped for one op: the knob
   that buys the squares' accuracy is the accumulator's width, not the multiplier's.
2. **The floor is one pass, and the squares reach it.** 232.3 µs against a 229.0 µs
   `sum(v)` control is 1.4% off a pure single read of `v`. The matvec does not: 1.97×,
   because `N = 1` wastes 31 of 32 output columns and gives the matmul no reuse.
3. **Two reads of `v` is the composed-op floor.** 232.3 + 450.1 = 682.4 µs against a true
   one-read floor of 229. No composed op can produce two different reductions from one pass
   over the tensor — which is precisely the gap a fused kernel closes. This is the sharpest
   case Phase 10 has.

Priced in the op, where P6's lesson says it has to be — the one-pass forms are now charged
for slicing column 0 out of the 32-wide output and transposing `q` into a column:

| `C = S+1` | three-pass | one-pass | Δ | ratio |
|---|---|---|---|---|
| 2 | 867.4 | 657.5 | −209.9 | 1.32× |
| 5 (**the schedule's mean**) | 1 868.9 | 1 367.4 | −501.5 | **1.37×** |
| 9 | 3 136.4 | 2 240.6 | −896.2 | 1.40× |

4. **This time the conversion is free.** 896.2 µs saved in the op against the 891.7 µs the
   standalone variants predicted — the slice and the transpose cost nothing measurable, and
   the per-candidate slope falls 323.8 → 225.7 µs, i.e. 98.1 µs per candidate against a
   predicted 99.1. The contrast with P6 is the point: P6's permutes reshaped a
   `2(S+1) × N`-row tensor and cost half the win, while P7's slice and transpose touch a
   32-wide strip and a single row. **A layout conversion costs what it moves** — P6's moved
   megabytes, P7's moves kilobytes.
5. Fit over the real schedule (186 reads, Σ(S+1) = 1 002), linear to ~1% at three points:
   **367 → 267 ms, 1.38×.** Against the 380 ms the ledger quotes from P3-era coefficients,
   ~276 ms. Twenty-five times the fold's 15.3 ms, from one iteration.

**The primitive does not fit a full-`d` row.** `rms_norm_pre_all_gather` keeps the row in one
core's L1, sized at `4·Wt` tiles — input double-buffered plus a double-buffered `x²` — and it
throws at program build past that. Measured against Blackhole's 1 572 864 B: 1 590 144 B
asked for at `W = 5 760`, 1 688 448 at 6 144, 1 950 592 at 7 168, the steps between them
exactly 8 192 B per tile of width. So the ceiling is 177 tiles, `W ≤ 5 664` — and
`use_2d_core_grid=True` is **not** the escape hatch, asking for *more* (1 971 072 B at 7 168)
because it splits tokens, not the row. Any `tp_factor ≥ 2` on a 7 168-wide model fits;
`tp_factor == 1` does not, so `_local_sum_squares` gates on width and falls back to
`mul` + `sum` there. The matvec has no such limit, so single-device still gets that half.

Correctness gate — the 186-read depth walk, against P6's numbers on the same tests:

| `T` | layout | P6 | P7 | Δ |
|---|---|---|---|---|
| 64 | folded | 0.9999545 | 0.9999530 | −1.5e-6 |
| 64 | unfolded | 0.9999500 | 0.9999379 | −1.2e-5 |
| 256 | folded | 0.9999207 | 0.9999273 | +6.6e-6 |
| 256 | unfolded | 0.9999223 | 0.9999232 | +0.9e-6 |

In both directions and inside the band P6 already characterized as reassociation noise. All
95 correctness tests pass, including the `d = 7168` single-device rows that exercise the
fallback (0.9999381 against 0.9999408 — moved because that path still takes the matvec).

*Verdict:* **`one_pass_stats=True` by default**, width-gated on the squares half. The whole
knob is one argument; the constraint lives in the op, not in the caller.

### P8 — `inter_block` reads the sealed set 49 times to serve 24 sites

The split form hoists the reciprocal-RMS pass on the argument that a sealed snapshot is
write-once. The dots and the mixture sat inside the loop anyway, so a 12-layer block read a
70 MiB tensor 49 times to answer 24 read sites. Both loops are one contraction batched over
sites. Priced per block, traced on `(2, 4)`, `S = 8`, against a floor of one pass over the
sealed set (219.1 µs):

| variant | µs/block | ×floor |
|---|---|---|
| floor: one pass over sealed | 219.1 | 1.00 |
| dots: ×24 | 9 268.7 | 42.3 |
| dots: **batched matmul** | **395.3** | **1.80** |
| mix: ×24 | 14 867.7 | 67.9 |
| mix: batched matmul, both conversions charged | 13 597.7 | 62.1 |

**The two halves come apart completely, and the reason is which operand has to move.** The
dots contract over `d`, already the last axis, so batching them reshapes the *queries* —
24 × 3.5 KiB — and stacking them into `[1, 1, d/tp, 24]` turns 24 matvecs into one matmul:
**23.4×**, and the 24-wide output idles 8 of 32 columns where the lone matvec idled 31
(1.80× floor against P7's 1.97×). The mixture contracts over the candidate axis, which a
matmul can only reach as a tile axis, so batching it reshapes the *sealed tensor* — 70 MiB,
twice, and `S = 8` tile-pads to 32 on the way. The matmul itself is 3.2× the loop; after the
permute in and the permute back out it is **1.09×**. Slicing 24 partials out of the padded
output instead of permuting is worse still, 1.7 ms each against 14.9 ms for the whole loop.

In-op A/B, per read site, traced on `(2, 4)`, swept over `S` because the sealed half is the
part that grows with it. `direct` is untouched by this change and is the control:

| `S` | direct (control) | split, before | split, after | |
|---|---|---|---|---|
| 1 | 649.6 | 767.3 | 710.9 | 1.08× |
| 4 | 1 356.7 | 1 165.4 | 971.2 | 1.20× |
| 8 | 2 231.6 | 1 741.4 | 1 386.5 | **1.26×** |

1. **The control holds to 0.04%** — 2 232.2, 2 231.7, 2 231.3 µs at `S = 8` across three
   independent runs on either side of the change. The direct form's fitted slope, 225.6 µs
   per candidate, lands on the 225.7 P7 measured from a different test.
2. **96% of the standalone win landed in the op**: 355.4 µs per site measured against 369.7
   predicted, so the query concat costs ~14 µs per site. P6's lesson applied cleanly for the
   second time — a layout conversion costs what it moves, and this one moves 84 KiB.
3. Fits over the real schedule (186 reads, Σ(S+1) = 1 002), linear to 2%: direct
   `209.4 + 225.6·(S+1)` → 265.0 ms; split **`481.1 + 139.4·(S+1)` → 229.2 ms** before,
   **`506.0 + 96.9·(S+1)` → 191.2 ms** after. **1.20× on the forward, 38 ms.** The slope
   falls 30% and the intercept rises 25 µs, which is the concat amortized over 24 sites.
4. **The split form loses at `S = 1`** — 710.9 against direct's 649.6 — and the sweep is the
   only reason that is visible. Its fixed cost is the second collective and `merge`'s own
   stats pass; below ~2 sealed snapshots there is not enough sealed work to amortize them.
   The crossover moves `S+1 = 3.15 → 2.30`, so P8 pulls one more block onto the split form.
   The schedule seals at layer `12k`, which puts exactly 24 reads at `S = 1`: taking the
   direct form for that one block is worth 1.5 ms of 191.2, **0.8%**. Documented as a
   threshold, not built as a mechanism.
5. The collective count is the part of P5 that this retires. The split form issued 49 per
   block against the direct form's 24 — one for the sealed RMS, one *per site* for the
   sealed dots, one per site in `merge`. Batching the dots makes it **26**, so the form that
   was 1.43× faster while paying twice the collectives now pays 8% more of them.

Correctness gate — 95 tests pass, and the 186-read depth walk moves by less than the noise
band P6 characterized. The batched form needs the same two broadcasts the loop needed, one
of them new: `[1, S, N, R] * [1, S, N, 1]` on the last dim, measured at one bf16 ULP.

*Verdict:* **batched dots kept, batched mixture rejected.** The mixture is now 96% of the
sealed half's traffic and 24 of its 26 passes, which is the whole of what Phase 10 is for.

### P9 — a second dim-1 reduce kernel, and a 102 µs error in P7's mix row

Not found by profiling. Found by writing Phase 2's delta table, which asks what else in
the tree could do delta 4 and turned up `ttnn.experimental.fast_reduce_nc` — a reduce over
dim 0, 1 or [0,1], which is the mixture's axis, exposed in Python and never A/B'd. P7 had
recorded "no one-op form" for the mix, which is true of *fusion* and says nothing about
whether `ttnn.sum` is the best available dim-1 reduce.

Same trace methodology, same 79 MiB input, `(2, 4)`. Two runs, because the whole verdict
turns on a sub-percent difference and one sample cannot tell that from noise:

| form | run 1 | run 2 | mean | ×floor |
|---|---|---|---|---|
| **floor**: `sum(v)` — reduce dim 3 | 228.8 | 228.6 | 228.7 | 1.00 |
| **floor**: `sum(v, dim=1)` | 229.0 | 228.4 | 228.7 | 1.00 |
| **floor**: `fast_reduce_nc(v, dims=[1])` | 228.4 | 229.2 | 228.8 | 1.00 |
| **floor**: `fast_reduce_nc` **+fp32 dest acc** | 228.5 | 228.5 | 228.5 | 1.00 |
| mix, `[1,1,1,d]` broadcast (P7's row) | 789.8 | 790.5 | 790.2 | 3.46 |
| mix, real `[1,C,N,1]` weight: `mul` + `sum` | 688.4 | 688.2 | **688.3** | **3.01** |
| mix, real weight: `mul` + `fast_reduce_nc` | 687.5 | 688.0 | 687.8 | 3.01 |

1. **Refuted, and the second run is what makes it a refutation.** `fast_reduce_nc` is
   0.08% faster on the mean (687.8 against 688.3 µs) against a run-to-run band that reaches
   0.35% on the floor rows — so the difference is not resolvable, let alone useful. The
   reduce half already runs at the memory floor, so a different kernel over the same bytes
   has nothing to win. Delta 4 stays composed until Phase 10, now on evidence rather than on
   the absence of a fused op. Read from run 1 alone this row said 0.13% and would have
   invited the same conclusion for the wrong reason.
2. **Four different reductions cost exactly one read of `v`** — 228.4 to 229.0 µs, a 0.26%
   spread, across two axes, two kernels, and fp32 dest accumulation on and off. **Not two
   fidelities, which is what this row claimed before the defaults were read:** every row here
   is HiFi4. `ttnn.sum` defaults to HiFi4 + `fp32_dest_acc_en` on Blackhole
   (`reduce_op.cpp:109`, the LoFi branch is Wormhole-only) and `fast_reduce_nc` defaults to
   HiFi4 (`fast_reduce_nc.cpp:31`), so the bare-vs-`HIFI` pair varies the accumulator and the
   packer, not the multiplier. The conclusion is unchanged and the support is narrower.
   The dim-1 output is **8.75 MiB**
   against the dim-3 output's **1.41 MiB** (`[1, 9, N, 1]` tile-pads to 32 wide, so the
   padding is 97% of it), a 6× difference in bytes written that does not register at all
   against a 79 MiB read. fp32 dest accumulation is free here, as HiFi4 was in P7. Every
   floor row across P7 and P9 lands in 228.2–229.5 µs; the number is the bandwidth, not the
   kernel and not the compute config.
3. **P7's mix row overstated the op, and this is the load-bearing finding.** That row
   multiplied by `q`, a `[1, 1, 1, d/tp]` broadcast, because it reused the matvec's operand.
   `_mix` multiplies by a `[1, C, N, 1]` weight. Same bytes read, same bytes written, same
   reduction after — **102 µs apart**, the P7 row sitting 15% above the shape the op runs.
   Broadcasting a scalar along the last dim is cheaper than broadcasting a row across the
   outer dims. The op's mixture is **688.3 µs at 3.01× its floor**, not 791.1 at 3.45×.
4. Phase 10's headline is unaffected — the ~3.8× came from the whole-op 191.2 ms, not from
   this row — but its component attribution moves, and the ~229 µs floor it has to beat is
   now confirmed by four independent kernels rather than one.

*Verdict:* **refuted, and a recorded number corrected.** The cost of the refutation was two
15-second device runs; the cost of not asking was a 102 µs error standing in the log since P7.

---

## Phase 10 — the fused mixture

Nine phases of composed measurement to earn one kernel. The mandate is narrow and P8 wrote
it: of the split form's 26 passes over the sealed set per block, **24 are the mixture**, 96%
of that half's traffic, and it is the one contraction composed primitives provably cannot
batch — its contracted axis is the candidate axis, and reaching it with a matmul costs more
in padding than the arithmetic saves (P8: 1.09× measured, all of a 3.2× win spent on
conversions). So Phase 10 is **not** the whole-op fused kernel §7 of `ROOFLINE.md` sketched.
It is one op: a weighted sum that MACs into its accumulator instead of materializing a
product.

### The op

`ttnn.experimental.fast_weighted_reduce_nc` —
`out[b][0][h][w] = Σ_c input[b][c][h][w] · weight[b][c][h][0]`, at
`ttnn/cpp/ttnn/operations/experimental/reduction/fast_weighted_reduce_nc/`. Built as a
sibling of `fast_reduce_nc` rather than an AttnRes-specific kernel: nothing in it knows about
attention residuals, which is why it could be gated against torch instead of against our own
reference.

The technique is `deepseek_moe_fast_reduce_nc_fused`'s, which P8 identified as the in-tree
existence proof. `init_bcast<ELWMUL, BroadcastType::COL>` configures PACK and UNPACK, then
MATH is re-initialized behind its back:

```cpp
MATH((llk_math_eltwise_binary_init<ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
    cb0, cb1, 1 /*acc_to_dest*/)));
```

so every `mul_tiles_bcast_cols` performs `dst0 += act·weight` in one pass. `tile_regs_acquire`
zero-initializes `dst0`, so candidate 0 seeds the accumulator with no special case, and
because the running sum lives in a dst register rather than an intermediate CB, **`C` is
unbounded** — there is no dst-register budget being consumed per candidate.

Five things were decided against the analog rather than copied from it:

1. **`BroadcastType::COL` is free at our layout, and this is why the op is 300 lines and not
   700.** A `[B, C, H, 1]` TILE tensor is physically `[B, C, H, 32]` tiles with the value in
   column 0 — *exactly* what COL broadcast consumes. The MoE op needs an extra ROW_MAJOR
   staging CB and face-offset arithmetic only because its scores arrive ROW_MAJOR from the
   routing convention. Ours arrive already correct at zero cost. The weight is the score
   chain's `[1, C, N, 1]` output, unchanged.
2. **Contiguous work split, not `fast_reduce_nc`'s round-robin.** `fast_reduce_nc` hands core
   `i` tiles `i, i+num_cores, ...`. That is fine when there is one input, and wrong here: the
   weight set is keyed by the *token row*, so striding by `num_cores` scatters the row and
   forces a weight refetch on nearly every output tile. Contiguous ranges give each core
   ~`num_tiles/Wt` distinct rows, so the weight is read about once — ~3% over the input read
   instead of a second 79 MiB.
3. **No semaphore between reader and compute.** `i % Wt == 0` is exactly the token-row
   boundary, because `inner_tile_size = Ht·Wt` is a whole number of rows. Reader and compute
   derive it from the same formula over the same `start_id`, so they agree by construction.
   The weight CB is `2·C` pages deep and the worst case is two outstanding sets, which is
   capacity; compute always holds enough pushed input to reach the next boundary and free one.
4. **Granularity divisibility is a correctness requirement, not a tuning preference.** The
   granule is the largest factor of `C` at most 8 (`C = 9` → 3). Compute derives a candidate's
   weight index as `j·granularity + k`, which is only the right index when the granule tiles a
   whole reduction.
5. **`Wt` and the strides are compile-time args.** RISC-V has no divide instruction; passing
   them as runtime args puts a libcall on the per-tile path instead of a multiply-shift.

Two deliberate asymmetries in the contract. The **input is bf16 only** — the acc-into-dst
path is the one numeric configuration gated against a reference, and widening it here would
ship an untested path. The **weight also takes fp32**, because `_scores` runs its whole chain
in fp32 on purpose (P-series: fp32 stats buy back rounding the single-device path also takes)
and requiring bf16 would make the call site pay a typecast to throw that away. The CBs are
sized from each tensor's own dtype, so the mixed pair costs the program factory nothing.

Defaults are HiFi4 + `fp32_dest_acc_en`, matching `deepseek_moe_fast_reduce_nc_fused` — the
other op that MACs into dst. Two reasons, and the second is the one that matters: the
accumulator is read and written once per candidate, so a bf16 dest would round the running
sum `C` times; and matching `fast_reduce_nc`'s HiFi4 default keeps the A/B below a
measurement of the *fusion* rather than of the fidelity.

**One inherited defect not reproduced.** `fast_reduce_nc` builds its output spec from
`padded_shape`, which hands a caller with 100 tokens an output claiming 128 *logical* rows —
the 28 rows of tile padding become part of the tensor and whatever the input's padding held
is readable data. This op takes `logical_shape`. Found by the `[1, 9, 100, 128]` test failing
against a 100-row reference, which is the only failure the suite produced.

### P10 — the isolated row

Same methodology as P9: traced, `(2, 4)`, one 79 MiB `[1, 9, 2560, 1792]` input, two runs.

| form | run 1 | run 2 | mean | ×floor |
|---|---|---|---|---|
| **floor**: `fast_reduce_nc(v, dim=1)` | 229.1 | 228.3 | 228.7 | 1.00 |
| **floor**: `fast_reduce_nc` +fp32 acc | 228.1 | 228.2 | 228.2 | 1.00 |
| **fused**: `fast_weighted_reduce_nc` | 257.4 | 257.2 | **257.3** | **1.13** |
| **fused**: fp32 weight | 267.9 | 265.8 | 266.9 | 1.17 |
| composed: `mul` + `sum` (P9's baseline) | 686.7 | 687.4 | 687.1 | 3.01 |

1. **2.67× on the mixture** — 687.1 → 257.3 µs. The composed form moves 3V to perform a
   reduction whose floor is ~1.13V, and the op removes exactly that: read `v`, MAC, write
   `V/C`.
2. **The weighting costs 29 µs, and that is the interesting number.** The fused op sits 13%
   above an *unweighted* reduce over the same bytes. Everything above the floor is the
   per-candidate multiply and the weight fetch; there is no third pass hiding in it. This is
   the row that says the kernel is done rather than merely faster.
3. **fp32 weight costs 3.7%** (266.9 against 257.3), against a predicted ~3% from doubling
   traffic that is 3% of the read. So the accuracy the call site keeps is priced, small, and
   the arithmetic explains it.
4. P9's baseline reproduced to 0.06% seven iterations later, on a different binary. The
   harness is stable enough that 3.7% is a signal.

### P10 — the whole block, both read forms

The isolated row is one op on one tensor. What the model pays is a block of 24 read sites,
where the mixture is one pass among many, so the `fused_mix` axis was added to P5's
block-level test rather than measured on a new one. Traced, `(2, 4)`, per read site, two runs
— every row reproduced within 0.1%, so the means are given alone:

| S | C at `_mix` | form | composed | fused | ratio |
|---|---|---|---|---|---|
| 1 | 1 | split | 710.9 | 704.6 | 1.01 |
| 4 | 4 | split | 971.0 | 780.3 | 1.24 |
| 8 | 8 | split | 1 386.3 | **991.2** | **1.40** |
| 1 | 2 | direct | 649.7 | 552.9 | 1.18 |
| 4 | 5 | direct | 1 356.8 | 1 105.2 | 1.23 |
| 8 | 9 | direct | 2 231.8 | 1 800.0 | 1.24 |

1. **The isolated row predicts the block saving to 3%.** Scale P10's 429.8 µs saving by the
   candidate count the split form's mixture actually runs — `C = S`, not `S+1`, because
   `merge` has no candidate axis — and it predicts 191.0 µs at `S = 4` against 190.7
   measured, 382.0 at `S = 8` against 395.1. The fusion is additive and nothing else moved:
   this is one op getting faster, not a schedule reshuffling.
2. **The split form gains nothing at `S = 1` (1.01×), and that is the same fact.** Its only
   `_mix` is `inter_block`'s over the sealed set, at `C = S`, so `S = 1` is the degenerate
   one-candidate reduction — `mul` then `sum` over a 1-deep axis is already a single pass with
   no intermediate worth removing. The direct form at the same `S` runs `C = 2` and gets
   1.18×. Predicted 47.8 µs of saving there, measured 6.3; the 0.13 ratio is the only place
   the model above breaks, and it breaks exactly where `C = 1` makes the fusion vacuous.
3. **`merge` never called `_mix`.** It folds two `[1, 1, N, d]` tensors with online-softmax
   scalars — an `add` of two `mul`s, not a candidate reduction. Worth recording because the
   obvious reading of "24 of 26 passes are the mixture" is that the mixture is everywhere in
   the split form, and it is in exactly one place.
4. **Per forward, using P8's fit method** (`186a + 1002b`, which reproduces P8's 191.2 ms
   exactly from its own coefficients): the direct form goes **265.0 → 216.2 ms** on a fit whose
   intercept moves 209.5 → 203.8, i.e. barely, which is what a mixture-only change should do.
   The split form's fit is degenerate — its intercept *rises* 506.0 → 603.6 while the slope
   halves, which cannot be physical — so its forward is taken from the measured ratio at the
   schedule's mean `C` of 5.39 instead: **191.2 → 153.6 ms, 1.24×**. The bad fit agrees at
   153.9 ms, which is why the number is quoted at all.

### What this does to §7's ~3.8×

`ROOFLINE.md` §7 put Phase 10's realizable win at ~3.8× against the split form's 191.2 ms.
The mixture-only kernel delivers **1.24× of it**, and there is no contradiction: fusing the
mixture removes the composed form's *extra* pass, not the pass itself. 3V → 1.13V on one op
out of a read that still runs a stats pass, a collective, a softmax chain and a divide. The
remaining ~3× is the rest of §7's kernel — the two `d`-reductions and the cross-candidate
softmax folded into the same pass over `v` — and it is a materially harder op, because it
owns the collective and the numerics that P7 spent four iterations getting right in composed
form.

That is the honest read of a 10.8× floor-to-floor estimate meeting a real kernel: **the
cheapest 1.24× of it cost one op with a 300-line program factory and no numerics risk**,
because it is the one piece that is a pure traffic problem.

### Correctness

`tests/ttnn/unit_tests/operations/reduce/test_fast_weighted_reduce_nc.py` — **19 passed**,
14.7 s, gated at PCC 0.9999 against torch in fp32. bf16 in and out means one rounding at the
pack; anything looser would hide a defect.

Coverage is by kernel path, not by shape variety: `C ∈ {1, 5, 8, 9, 12, 13}` to take the
granularity cap (8), a clean factor (6), two primes that fall back to granularity 1, and the
degenerate `C = 1`; `Wt = 1` so the weight set turns over on *every* output tile, and `Ht = 1`
so it never does; `B = 2` for the batch stride in both index chains; a token count of 100 for
the padding path; the fp32-weight pair; the full `[1, 9, 2560, 1792]` production shape. Plus a
program-cache test that holds both tensors past the assertion so a stale buffer binding cannot
pass, an equivalence check against `mul` + `sum` at matched precision, and five rejection
cases pinning the contract (`dim ≠ 1`, wide weight, mismatched leading dims, rank 3, fp32
input).

The module suite gates the wired call site: **161 passed** with `fused_mix=True` as the
default, so every PCC threshold in Phases 5–8 now holds through the fused path.

*Verdict:* **kept.** 2.67× on the op, 1.40× on the block at the peak shape, 1.24× on the
forward, at 13% above an unweighted reduce over the same bytes. The composed form stays as
the knob's `False` branch because it is the oracle the op is gated against.

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

### Phase 1 — the infra map, and the analog that is not in this tree

Written after phases 3–9, which is the wrong order and shows: both findings below are
corrections to claims the earlier phases had already acted on. The map itself is short
because AttnRes has fewer in-tree analogs than the ladder assumed.

| What we need | In-tree analog | `file:line` | Its own gate |
|---|---|---|---|
| Distributed RMS statistics | `TtDistributedRmsNorm.forward` | `models/demos/deepseek_v3_d_p/tt/tt_distributed_rms_norm.py:236` | `pcc=0.99` |
| — its pre-reduce kernel | `ttnn.rms_norm_pre_all_gather` | `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.cpp` | — |
| — its stats collective | `ttnn.all_gather(dim=3, cluster_axis=…)` | `tt_distributed_rms_norm.py:274` | — |
| PCC gate helper | `assert_with_pcc` | `tests/ttnn/utils_for_testing.py:94` | default `pcc=0.9999` |
| Distributed norm test | `test_rmsnorm_distributed` | `models/demos/deepseek_v3_d_p/tests/pcc/test_rmsnorm.py:134` | `pcc=0.99` |
| Single-chip norm test | `test_rmsnorm_single_chip` | `models/demos/deepseek_v3_d_p/tests/pcc/test_rmsnorm.py:202` | `pcc=0.99` |
| CCL semaphore hoisting | `create_global_semaphores` | `models/demos/deepseek_v3_d_p/tt/tt_ccl.py:54` | — |
| Mesh topology per axis | `per_axis_topology`, `get_num_links` | `tt_ccl.py:308`, `tt_ccl.py:376` | — |
| Mul-then-reduce over dim 1 | `deepseek_moe_fast_reduce_nc_fused` | `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/deepseek_moe_fast_reduce_nc_fused.hpp:30` | — |
| Reduce over dim 0/1 | `ttnn.experimental.fast_reduce_nc` | `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc.hpp:15` | — |

**Our thresholds are not inherited from the analog — they are stricter, and §Gating
discipline said otherwise for nine phases.** The `_d_p` rmsnorm tests pass `pcc=0.99`
explicitly, at both `file:line` above. `0.9999` is `assert_with_pcc`'s *signature
default*. The earlier text cited line 94 as if it were the analog's choice, which
conflated the repo-wide default with the nearest analog's deliberate loosening. The gate
stays at 0.9999 — we measure there and have no reason to give it up — but the provenance
is the repo default, and the analog's 0.99 is a floor, not a target.

**mHC — the nearest prior art, and the reason this task exists — is not in this tree.**
`models/demos/deepseek_v3_d_p/tt/mhc/` exists on disk holding nothing but a
`__pycache__` from another branch's checkout; `git ls-files` matches no path containing
`mhc` on this branch. So nothing was inheritable from the previous residual bringup, and
every threshold, layout and test shape here was derived from AttnRes's own algebra. That
is defensible, but it was an accident rather than a decision, and the ladder's Phase-1
gate exists precisely to make it a decision.

*Lesson: an analog you cannot cite at `file:line` is a memory, not an analog. Both errors
here — a threshold attributed to the wrong line and a module assumed present because its
directory was — survived nine phases because no one made the citation.*

### Phase 2 — six deltas, and the one still open is exactly Phase 10

AttnRes against the distributed-RMSNorm analog, as a countable delta:

| # | Delta | Status |
|---|---|---|
| 1 | Statistics over `S+1` candidate planes, not one | composed: `[1, C, N, ·]` throughout |
| 2 | A `⟨q, v_i⟩` matvec per candidate alongside the norm | one-pass via `ttnn.matmul` (P7), batched over sites (P8) |
| 3 | Softmax over the **candidate** axis (dim 1) | hand-rolled; `ttnn.softmax(dim=1)` loses 4% of the mass (D12) |
| 4 | Weighted sum over the candidate axis | `mul` + `sum(dim=1)`, still 3.01× its floor (P9) |
| 5 | Stats reduced with `all_reduce`, not `all_gather` | every rank needs the same scalar, not the concatenation |
| 6 | Values enter the mixture **un-normalized** | gated by `test_values_are_not_normalized` |

Deltas 1, 2, 5 and 6 are settled. Delta 3 is settled *against* `ttnn.softmax` — the fused
op exists and loses, so the hand-rolled chain stays. **Delta 4 is the whole of Phase 10's
mandate** (P8: 24 of the sealed half's 26 passes, 96% of its traffic).

`Missing/blocked ops`, as the gate demands:

- **Blocked, technique liftable:** `deepseek_moe_fast_reduce_nc_fused` does exactly
  delta 4's arithmetic in one pass, but its signature requires `expert_indices_tensor`
  and `expert_mapping_tensor` — the MoE routing convention — and an L1-resident input
  (verified at `…_fused.hpp:30-33`). Not callable; the kernel technique is.
- **Missing:** no fused elementwise-then-reduce over a batch dim with a plain
  `[1, C, N, 1]` weight. This is the op Phase 10 writes.
- **Present but never tried, found only by writing this table:**
  `ttnn.experimental.fast_reduce_nc` reduces dim 0, 1 or [0,1] — delta 4's axis — and is
  exposed in Python. It is *not* the fused op (the elementwise pass stays), so it cannot
  reach the floor, but it is a different dim-1 reduce kernel than `ttnn.sum`, which P7
  never A/B'd. Numerics land one bf16 ulp apart. Priced in P9.

*Lesson: the delta analysis is not paperwork. Enumerating what the op needs against what
the tree has turned up a candidate primitive nine phases of profiling had walked past,
because profiling asks "where is the time" and the delta asks "what else could do this".*

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

### Phase 7 — a bit-exact gate that needs no reference at all

**`T` is a pure batch axis, and that buys a sharper instrument than PCC.** Every
reduction in the op is over `d` or over the candidate axis; none is over `T`. So the same
read at `T = 64` and at `T = 5120`, sharing token rows, must agree **bit for bit** on the
shared slice. Measured: `max|Δ| = 0` over 458 752 elements for a single read, and still
`max|Δ| = 0` after the full 186-read 93-layer walk. That gate costs one extra device run
instead of a ~10-minute fp32 host walk at production shape, and it fails on exactly the
class of bug production shape introduces — a padding leak, a reduction on the wrong axis,
a `T`-dependent work split. Equality gates are usually a mistake on device; here the
invariant genuinely is exact, so the gate can be exact too.

**Do not use `ttnn.CONFIG.throw_exception_on_fallback` as a no-host-fallback gate.** It
reads like one. It is declared at `ttnn/api/ttnn/config.hpp:29` and **nothing in the tree
reads it** — the only consumers are Python setters in three model tests. Four in-tree
models set it to `True` and are getting no guarantee from it.

**What does establish device-residency is bandwidth arithmetic.** One production read at
`T = 5120, d = 7168, S = 8` moves a counted **7.33 GB** of DRAM traffic (concat 1.32,
squares 1.32, its reduction 0.66, the `v ⊙ q` product 1.32, its reduction 0.66, the
weighted product 1.32, its reduction 0.73) in a warm 21.6 ms → **~339 GB/s effective**.
PCIe on this box cannot sustain a tenth of that, so nothing round-trips to host. The
percentage-of-peak claim waits for Phase 8, where the constant has to be cited. Source
inspection agrees: the only `from_torch` in `tt/` is the load-time `to_query`, and there
is no `to_torch` at all.

`ttnn.graph` was the wrong tool for this. `extract_calltrace` returns buffer-lifecycle
nodes (`create_device_tensor`, `Tensor::deallocate`), not op names, so it cannot verify
D10's "~12 launches per read" claim — that is a Phase-9 profiler measurement. Worse, the
capture only populates with `enable_fast_runtime_mode = False`, and *that* mode's
decorator layer introduces its own `Tensor::cpu` calls which do not exist on the
production path. A tool that changes the thing it measures is not evidence.

**The split form's memory cost shows up only at production shape.** `inter_block` returns
all 24 partials before any `merge` runs, so 24 × `[1, 1, T, d]` = **1.7 GiB** coexists on
top of the 560 MiB of snapshots. Invisible at `T = 64` (21 MiB). It fits on Blackhole and
the 1.50× is real, but the group size is now a memory knob as well as a traffic knob,
and under TP the partials shard with `d` while the peak does not move.

*Lesson: look for an invariant the op satisfies exactly before reaching for a tolerance.
An exact invariant gives a gate with no threshold to calibrate and no reference to
compute — and it is the only kind of gate that gets sharper, not blunter, as the shape
grows.*

### Phase 8 — the sharded path's bugs are invisible on one device

`DISTRIBUTION.md` holds the mapping argument and the arithmetic. What the implementation
taught, beyond it:

**Every distribution bug in this op is a no-op at `tp_factor == 1`.** Divide `mean(v²)` by
the local shard width instead of the global `d`; point the collective at the sequence axis;
skip the collective entirely — on a single device all three are indistinguishable from
correct, because the reduction *is* the identity there. That is not a reason to be careful;
it is a reason to build gates that only exist on the mesh. Two of them earn their keep:

| gate | as shipped | mutated |
|---|---|---|
| sequence-sharded vs sequence-replicated, same tokens | `max\|Δ\| = 0` | any axis mistake moves it off zero |
| `_reduce_stats` stubbed to the identity | PCC 0.9999778 | PCC **0.5757407** |

The first is exact, and exactness is what makes it able to tell "reduced on the TP axis"
from "reduced on *an* axis". Shard 64 tokens over the two SP rows and separately replicate
32 tokens on both rows: the SP axis carries no traffic under this mapping, so the shared
tokens must agree bit for bit. A collective aimed at the SP axis mixes different tokens in
the first placement and doubles the statistics in the second; one spanning both axes does
the same. Neither shows up in a PCC test, because a PCC test only asks whether one
placement is self-consistent.

The second is a mutation test that had to be shipped rather than run once, for the same
reason: there is no single-device configuration in which it fails.

**fp32 statistics are free and slightly better than free.** `ttnn.all_reduce` reduces in
bf16 unless handed fp32 (`all_reduce_nanobind.cpp:48`). Over 186 chained reads at
`d = 7168`: fp32 stats 0.9999500, bf16 stats 0.9999401, single-device baseline 0.9999408.
The bf16-sharded number landing on the single-device number is the informative part — the
extra cross-shard summation costs nothing measurable, and fp32 is buying back rounding the
single-device path *also* takes. Which suggests fp32 statistics are worth a look on the
single-device path too, at 1.5 MB against 900 MB.

**The split form doubles the collectives.** 49 for 24 read sites against the direct form's
24: `inter_block` amortizes the sealed RMS across read sites, but each site still needs its
own dot against the sealed set, and `merge` still pays a full paired reduction on the live
stream. Phase 7 measured the split form 1.50× faster on one device; that number now has a
second term nobody has measured. The Phase-0 estimate in "TP is nearly free" said one
all-reduce per block plus one 2-scalar reduce per read, and has been amended in place.

**A collective needs `device_params={"fabric_config": FABRIC_1D}`.** Without it the first
`all_reduce` dies on `control_plane.cpp:2186` — an uninitialized fabric context, not a
numeric failure. Cheap to hit and cheap to fix, but it means every distributed test file
carries the fixture parametrization, not just `mesh_device`.

*Lesson: when a code path is a no-op in the configuration you develop in, the tests that
cover it have to run in the configuration where it isn't. Neither of this phase's real
gates can be written on one device, and both of them would have passed a `(1,1)` suite
while the op computed garbage on a mesh.*

### Phase 8 — `ttnn.all_reduce` is two algorithms, and the shape picks one

`ROOFLINE.md` set out to be a table of Blackhole constants. Two of its numbers had to be
measured instead, and both changed a conclusion.

**The collective's algorithm depends on the shape, silently.**
`all_reduce_async.cpp:359` takes a **composite all-gather + `local_sum` + two reshapes**
whenever `finding_scatter_dim` (`:33-62`) finds no dim divisible by the participant count,
and reduce-scatter + all-gather otherwise. That scan runs in **tile units** — the last two
dims divided by 32 — from the last dim backwards. For `[1, 2(S+1), T/R, 1]` at `R = 4`:

| `T/R` | tile units | divisible | path |
|---|---|---|---|
| 32 (`T = 64`, the suite) | `[1, 2(S+1), 1, 1]` | only when `S` is odd | composite for even `S` |
| 2560 (`T = 5120`, production) | `[1, 2(S+1), 80, 1]` | dim 2, always | always RS+AG |

Measured back to back on `(2,4)`: 769 µs for an **8 KiB** reduction against 481 µs for a
**5760 KiB** one — the small one 1.9× slower while moving 1/720th the bytes, because it is
on the composite path. So the distributed suite proves both algorithms correct (better
coverage than intended) and says nothing about production timing (worse than intended). It
also means an `S` sweep at small `T` produces a parity sawtooth that is an artifact of
`finding_scatter_dim`, not of AttnRes.

**A launch costs ~130 µs, flat.** The control that makes the number above meaningful: a
plain `ttnn.mul` in the same loop costs 174/137/137/130 µs across a 720× size range. Flat
means host — Python, ttnn op infrastructure, 8-device fan-out — not device. Against the
88 µs-per-launch break-even implied by the production DRAM floor, the composed op is
launch-bound untraced and DRAM-bound traced, 100× apart. Every perf number this module
ever reports has to name its regime.

The rest of the memo is arithmetic, and one line of it is the Phase-10 argument: `v` is
resident in aggregate L1 at every shape that matters (47.7% on LoudBox, 11.9% on Galaxy),
so a fused kernel's floor is `V(1 + 1/(S+1))` against the composed `12V` — **10.8×**, or
215.5 ms → 20 ms per forward.

*Lesson: a roofline is not a table of peak bandwidths. Two of the three terms here — which
algorithm a library op picks, and what a launch costs — are properties of this shape on
this box, and both were off by enough to invert the verdict. The measurement that mattered
most was the control, not the subject: 481 µs for a reduction means nothing until a local
`mul` on the same tensor has been timed in the same loop.*

---

### Phase 9 — "the host finished first" is not "the device was idle"

P1 timed 22 enqueues on `(2, 4)` at 3 348 µs and the synchronize at 3 378 µs, and I wrote
down *host-bound*. P2 traced the same read and the device alone needed 3 282 µs. Both
numbers are right; the inference was wrong. **Dispatch is pipelined**, so a host that
finishes at the same moment as the device is a host that was keeping pace with it, not a
host that was the bottleneck — the cost is `max(host, device)`, and `enqueue ≈ total` is
exactly the case where a single untraced measurement cannot tell the two apart.

The fix is structural, not analytical: only a **traced** run separates the terms, because
tracing drops host time by 345× and leaves device time untouched. The harness now labels
untraced verdicts "enqueue-limited" / "device-limited" rather than host / device, and
`_report`'s docstring carries the ambiguity so the next reader does not repeat it.

*Lesson: a perf harness that can only produce ambiguous readings will get read
unambiguously anyway. Build the disambiguating configuration — here, tracing — into the
same file as the ambiguous one, and never report the untraced number alone.*

### Phase 9 — the launch term binds at the schedule, not at the shape

`ROOFLINE.md` §6 asked whether dispatch or DRAM binds and answered it at the peak shape,
`S = 8`. At the peak shape the two terms are within 2%, so the answer there is "neither,
and it does not matter". But `S` ramps 0→8 across 93 layers and `mean(S+1) = 5.39`, and
untraced device totals on eight devices **pin flat at a 2.2–3.6 ms host floor no matter
what `S` is** — at `S = 1` that is 3.6× the device time, spent waiting on Python. Summed
over the real 186-read schedule: 622 ms untraced against 380 ms traced, so **tracing is
worth 1.64× per forward and 1.00× at the shape I would have measured**.

Same trap on the fabric side, in the other direction. The collective is payload-bound at
~18 KiB/µs and charges for tile padding at full price, so `[1, 18, 2560, 1]` — 184 KiB of
statistics in a 5 760 KiB envelope — costs 348 µs, 7.4× what the folded layout costs.
`ROOFLINE.md` §4 deferred that fix because two extra `permute` launches would cost 260 µs
against 83 µs of modelled fabric saving. Untraced, that arithmetic is right and the
deferral holds. Traced, the *launches* cost nothing and the saving is 148 µs net of what
the permutes cost on device (P6). **The same decision inverts between the two regimes**,
which is the strongest possible argument for §6's rule that no number gets reported
without naming its regime.

*Lesson: pick the measurement point from the workload's distribution, not from its
maximum. Peak shape is the shape where fixed costs matter least, so it is the one shape
guaranteed to hide the launch term — and a per-read table hides a ramp entirely.*

### Phase 9 — a layout is not free to enter

P4 priced the statistics fold by timing `all_reduce` on both layouts: 348 µs against 47.
A 7.4× ratio, ~300 µs per read. The op-level measurement gave **147.6 µs** — the two
`ttnn.permute` calls that reach the layout and come back cost ~153 µs, and I had guessed
40–120. Not a modelling error in the usual sense; the model priced the right op and simply
did not include the ops that must exist for it to see that input at all.

The corollary is what actually mattered. I expected the fold to *widen* its lead as `S`
falls, because the unfolded payload becomes almost entirely padding while the folded one
stays one tile row. Wrong: `ttnn.permute` is billed on the padded tensor by the same
mechanism the collective is, so both terms shrink at the same rate and the ratio is flat.
At `S = 1` the fold is worth 18 µs and plain `num_links = 2` beats it. **Two costs driven
by the same quantity do not trade off against each other at any scale** — which is also
why the fold and the second link turned out to be alternatives rather than additive, a
prediction P4 got right for exactly this reason and I re-derived the hard way.

*Lesson: a data-layout optimization has three prices — the win, the conversion in, and the
conversion out. Measure the composite in place. And when a saving and its cost are both
proportional to the same padded size, no sweep over that size will separate them.*

### Phase 9 — three iterations on the 5%, one on the 76%

The tracy attribution that opened this phase said seven full-`d` ops were 76% of the read
and the collective was ~5%. P4, P5 and P6 then went after the collective, because the
collective was the part I had a mental model of — a payload, a topology, a link count, a
layout to fold. Three iterations, 15.3 ms of a 380 ms forward. P7 read the same profile and
went after `mul(v, v)`, and got 100 ms in one.

Nothing about the collective work was wrong; the fold is real and it stayed. But the
profile had ranked the candidates before P4 started, and I ranked them by tractability
instead. The `d`-reductions looked less tractable precisely because "elementwise then
reduce" is the obvious way to write them — obvious enough that it reads as a given rather
than as a choice with a 3.4× price on it.

Two things made P7 cheap once started. The reductions had **one-pass primitives already in
the tree**, written for a different op (distributed RMSNorm's statistics pass) and for a
shape that happens to match; and a *floor* — `sum(v)`, one pass, no intermediate — made
"how much is left" answerable before any candidate was implemented. The floor is what turned
3.41× into a bounded question, and it is what says the matvec's remaining 1.97× is real
headroom while the squares' 1.01× is done.

*Lesson: profile, then attack in the profile's order, not in order of which part you already
have a model for. And measure the floor first — a one-pass control turns "is this fast?"
into "how much is left?", which is the question that decides whether to keep going.*

### Phase 9 — batching a loop is decided by which operand has to move

P8 batched two loops of 24 iterations over the same 70 MiB tensor. One came out 23.4× faster
and one 1.09×, and the arithmetic was not the difference: the bare matmul beat its loop 3.2×
in both cases. What separated them is which operand needed reshaping to reach the matmul.

The dots contract over `d`. `d` is already the last axis, so the batch axis had somewhere
free to go and the only thing that moved was the queries — 84 KiB of them. The mixture
contracts over the candidate axis, which a matmul can only reach as a tile axis, so batching
it moves the 70 MiB sealed tensor twice and tile-pads `S = 8` to 32 while it is there. The
same 3.2× of arithmetic saving was there to collect and the conversions took all of it.

This is P6's lesson (*a layout conversion costs what it moves*) turned into something you can
check before writing any code: **ask which side of the contraction the batch axis has to
displace.** If the contracted axis is already last, batching is free and you should just do
it. If reaching it means promoting a short axis to a tile axis, the padding tax lands on the
largest tensor in the op and composed primitives will not win — that work belongs in a kernel
that never materializes the layout at all.

*Lesson: before batching a loop, find the contracted axis. Batching along an axis that is
already last is free; batching along one that has to become a tile axis costs a padded pass
over the big operand in each direction, which is the whole win.*

### Phase 9 — one shape is not a sweep, even at the peak

`inter_block`'s A/B was measured at `S = 8`, the peak shape, where the split form is 1.61×
the direct one. Sweeping `S` to fit the schedule turned up something the peak could not show:
at `S = 1` the split form is 9% **slower** than the form it replaced. Its fixed costs — a
second collective, `merge`'s own statistics pass — need about two sealed snapshots of work to
amortize, and 24 of the schedule's 186 reads sit below that line.

The peak shape is the right place to look for a bottleneck and the wrong place to conclude
from. Every phase before this one quoted `S = 8` numbers because that is where the bytes are;
what it hid is a *sign change* at the other end of the same axis, in a regime the model
actually spends 13% of its reads in.

*Lesson: a candidate measured at one shape is a candidate whose shape dependence you have
assumed. Sweep the axis the schedule sweeps — the interesting failures are at the small end,
where fixed costs stop being rounding error.*

### Phase 9 — a full-extent `ttnn.slice` aliases its input

Batching produced `[1, S, N, R]` intermediates that get sliced back into per-site columns, and
the module frees the batch afterwards. At `R == 1` that slice spans its input, and `ttnn.slice`
short-circuits: it returns a **new Python object pointing at the same device buffer**
(`buffer_address()` matches; a narrower slice's does not). Freeing the batch therefore freed
the column, and the next read of it segfaulted the process — inside `ttnn.to_torch`, several
ops downstream of the actual mistake.

Worth knowing generally: in ttnn, "this op returned a different tensor" is not evidence that
it copied. A no-op slice, and plausibly other shape ops with no-op cases, hand back a view.
Any code that slices a tensor and then deallocates the parent has a degenerate case where the
two are the same buffer, and it will present as a crash far from its cause.

*Lesson: when a helper hands out pieces of a tensor whose parent it then frees, check the
degenerate case where the piece is the whole. `buffer_address()` answers it in one line, which
is cheaper than reading a segfault traceback in a foreign frame.*

### Phase 9 — broadcast direction is worth 15% on the same bytes

P7 priced `_mix` by reusing the matvec's `q` as the multiplier: `[1, C, N, d/tp] * [1, 1, 1,
d/tp]`, a row broadcast across the outer dims. The op multiplies by `[1, C, N, 1]`, a scalar
broadcast along the last dim. Identical bytes read, identical bytes written, identical
reduction afterwards — **790.2 against 688.3 µs** over two runs each, and the version the op
actually runs is the fast one.

The mechanism is **not established** — the two shapes select different broadcast kernels
(`[1, C, N, 1]` broadcasts along a tile's columns; `[1, 1, 1, W]` broadcasts along its rows
*and* over both outer dims), and either the kernel or the path that reaches it could hold the
102 µs. It is not the operand's size: 3.5 KiB is L1-resident either way. Settling it needs a
tracy pass on the two `ttnn.mul` calls alone, which nothing yet depends on. What is
established is the direction and the magnitude, and that the shape the op runs is the faster
one.

*Lesson: a perf row that substitutes one broadcast shape for another is measuring a different
op, even when the byte counts match. Reusing a neighbouring row's operand is exactly the kind
of convenience that makes a table internally consistent and externally wrong — and it stood
for two phases because both numbers looked plausible.*

### Phase 9 — the profiler cannot surface a primitive you did not know existed

Seven perf iterations profiled this op, attributed 76% of its device time to seven big-tensor
ops, and never once asked whether `ttnn.sum(dim=1)` was the best available dim-1 reduce. The
question came from Phase 2's delta table — the paperwork phase — which asks a structurally
different question: not "where is the time" but "what else in the tree could perform this
contraction". One `grep` of the reduction ops directory turned up
`ttnn.experimental.fast_reduce_nc`, exposed in Python, reducing exactly the axis the mixture
reduces.

It lost, by 0.08% on the mean of two runs — unresolvable against a 0.35% band. That is the
point: the refutation cost two 15-second device runs, and now the mixture's floor is confirmed
by four independent kernels instead of assumed from one. An unasked question of that price is
not a saving.

*Lesson: profiling and delta analysis are not substitutes. A profiler ranks what you already
call; it is silent about the call you never made. Run the delta table before the perf loop —
and if the order slips, run it anyway rather than declaring it redundant.*

### Phase 9 — an A/B against a default you have not read is not an A/B

Two rows in this log compared an op "at default fidelity" against the same op with an explicit
HiFi4 config, and both labelled the default LoFi. One was right by luck. On Blackhole:

| op | default fidelity | default `fp32_dest_acc_en` |
|---|---|---|
| `ttnn.sum` (`reduce_op.cpp:109`) | **HiFi4** (LoFi branch is Wormhole-only) | **true** |
| `ttnn.experimental.fast_reduce_nc` (`fast_reduce_nc.cpp:31`) | **HiFi4** | false |
| `ttnn.rms_norm_pre_all_gather` (`rmsnorm_pre_all_gather.cpp:24`) | **HiFi4** | false |
| `ttnn.matmul`, bf16 in (`matmul_device_operation.cpp`) | LoFi (HiFi2 if `increase_fidelity`) | `false` unless fp32 out |

So P9's "two fidelities" varied no fidelity at all — every row was HiFi4 — and P7's claim that
LoFi mantissa truncation cost the squares an order of magnitude was attributing to the
multiplier what `fp32_dest_acc_en` did in the accumulator. Both timing conclusions survive
(the config is free either way; ~229 µs is bandwidth). Both *mechanisms* were wrong, and a
reader who trusted them would have reached for the wrong knob on the next op.

The tell was available for free: `init_device_compute_kernel_config(arch, cfg, default)` takes
the default fidelity as its **third positional argument**, so every op states its own default
in one grep-able line. Nothing had to be measured to catch this.

*Lesson: "at default" is not a measurement, it is a citation — and an uncited one. When a row
compares default against explicit, read the default's `file:line` first, or the row measures a
difference you cannot name. An accuracy A/B that varies three knobs at once and gets attributed
to one of them is worse than no A/B, because it reads as mechanism.*

---

### Phase 10 — a fused op is a traffic argument, and the traffic argument has a ceiling

Phase 5 built a fused op that was *slower* than its parts and a gate too weak to see it.
Phase 10 built one that is 2.67× its parts on the first device run, with no numerics iteration
at all. The difference is not skill and it is not luck — it is that this op was specified by
subtraction rather than by ambition:

| | Phase 5's fused op | Phase 10's |
|---|---|---|
| what it fused | several steps that looked adjacent | one pass that P8 *proved* composed ops cannot batch |
| justified against | the composed form's existence | 687.1 µs measured, against a 228.2 µs floor measured four ways |
| what it owned | numerics, layout and reduction at once | traffic only — same arithmetic, one fewer DRAM round-trip |
| result | slower than its parts | 2.67×, landing 13% above the floor |

The generalizable part is the shape of the claim. "Fuse the mixture" is a *bytes* claim: the
composed form moves 3V to do a reduction whose floor is 1.13V, and the op's whole job is to
delete the 1.87V. That is checkable before writing a line of C++ and it is checkable again
afterwards — the 29 µs the op sits above an unweighted reduce over the same bytes is the
entire remaining budget, and it is accounted for (per-candidate multiply plus a weight fetch
that is 3% of the read). Nothing about the numerics had to be right for the estimate to hold;
the numerics only had to not be *wrong*, which a torch gate settles in 15 seconds.

Contrast the ~3× still on the table. That one owns two `d`-reductions, a cross-candidate
softmax, a collective and a divide — it is a *numerics and scheduling* claim wearing a traffic
claim's clothes, and P7 spent four iterations getting those numerics right in composed form
where they were easy to inspect. Same 10.8× roofline, two completely different risk profiles
inside it.

Two smaller things worth carrying forward. **Reading the analog is not the same as copying
it**: `fast_reduce_nc`'s round-robin work split is correct for one input and actively wrong for
two, because the second operand is keyed by a row the stride scatters — a copied split would
have added a silent second 79 MiB read and still passed every PCC gate. And **an inherited
output-spec bug is still a bug**: building the spec from `padded_shape` publishes tile padding
as logical data, which one 100-row test catches and which the op it was copied from still does.

*Lesson: price a fused op in bytes before writing it, and check the residual in bytes after.
An op whose entire justification is "one fewer pass over V" can be estimated, gated and
believed in an afternoon. An op that also owns numerics cannot, and the roofline that lumps
them together will read as one number when it is two projects.*

---

## Backlog

- [x] Phase 1 — infra map with `file:line` citations and inherited thresholds. Back-filled
      after Phase 9, which is the wrong order: it found that our gate is *stricter* than the
      analog's rather than inherited from it, and that mHC is not in this tree at all.
- [x] Phase 2 — delta analysis; `Missing/blocked ops` list. Six-row delta, five settled;
      delta 4 is Phase 10's whole mandate. Surfaced the `fast_reduce_nc` candidate that P9
      then refuted.
- [x] Phase 3 — `API_SPEC.md`.
- [x] Phase 4 — `torch_functional/`, numeric ladder (D9, amended by D11).
- [x] Phase 5 — `tt/` composite, single device.
- [x] Measure the `N`-batched matmul `[N,R,S] × [N,S,d]` against 24 broadcast passes;
      decides whether the split form's remaining ~1.9× is reachable in composed ops.
      **It is not.** P8: the matmul is 3.2× the loop and the two permutes that reach its
      layout give back all but **1.09×** — 13 597.7 µs against 14 867.7 per block, on a
      70 MiB tensor that `S = 8` tile-pads to 32 planes each way. Per-site slices out of the
      padded output are worse (1.7 ms each). The mixture stays composed until Phase 10.
- [x] Batch `inter_block`'s dots across read sites — **23.4×** on that half, 49 sealed-set
      passes per block down to 26 and 49 collectives down to 26. **1.26× on the read at
      `S = 8`, 1.20× on the forward** (229.2 → 191.2 ms). The other half of the same
      backlog item, and the opposite answer, because the dots' contracted axis is already
      last.
- [ ] Let the caller pick the read form per block: the split form is 9% slower at `S = 1`
      (crossover at `S+1 = 2.30`), which is 24 of the schedule's 186 reads and 0.8% of the
      forward. Threshold documented in P8; no mechanism yet.
- [x] Phase 6 — device correctness + 93-layer depth harness.
- [x] Phase 7 — remove host fallbacks; `T=5120`.
- [x] Verify D10's launch count with the Phase-9 profiler — **28.3 device programs per
      read** on `(2,4)`, not the ~12 D10's text claims and not §6's ~25. D10's
      conclusion stands (its rejected alternative was ~90 launches); its count does not.
- [x] Phase 8 — `DISTRIBUTION.md`, TP on `(2,4)`, `ROOFLINE.md`.
- [x] Re-measure the split form's 1.50× on a mesh — **1.43×** traced on `(2,4)` (P5),
      3 127.8 → 2 186.6 µs per read site, with 49 collectives per block against 24.
      (1.47× before the fold landed; the fold is worth 3.5× more to the direct form,
      which carries all 18 stats planes per site.) It survives; batching `inter_block`'s
      24 dot tensors (49 → 26) is now an optimization, not a rescue.
- [ ] Try fp32 statistics on the **single-device** path — the sharded measurement says
      they buy ~1e-5 of depth PCC for 1.5 MB per read.
- [x] Phase 9 — perf harness + numbered perf loop. `tests/perf/test_attn_res_perf.py`,
      nine numbered iterations P1–P9 plus a tracy attribution, eight refutations recorded
      in §Phase 9 perf loop. Launch term measured first, as §6 demanded — and it refuted
      §6.
- [x] Fold the statistics into `[1, 1, T/R, 2(S+1)]` — landed as `fold_stats`, default on.
      P4's 7.4× on the bare collective is **147.6 µs per read (4.5%)** once the two
      `ttnn.permute` calls are charged for, ~15.3 ms of the 380 ms forward (P6). Depth PCC
      moves ±5e-6 in both directions.
- [x] `num_links` — **stays 1**. P6: with the fold in place the second link buys 4.7 µs
      per read. They are alternatives, not additive, so `ROOFLINE.md` §4's "the default
      should follow production's 2" is answered by taking the layout instead of the link —
      which is the better trade on Galaxy, where the fabric is contended.
- [ ] Hoist the collective's global semaphores out of the per-call path — still
      unmeasured, but P1 narrows the target: a plain launch is 105 µs on one device and
      152 µs on eight, while Phase 8 measured an `all_reduce` enqueue at 481 µs. That
      ~3× excess is where per-call semaphore creation would live. The analog hoists it
      with `create_global_semaphores` (`tt_ccl.py`).
- [x] Close the composed form's **1.43× gap to its own DRAM floor** (2 766 µs measured
      on `(8,1)` against §3's 1 935 µs) before treating Phase 10 as the only lever. The
      profiler says 76% of device time is 7 big-tensor ops; `mul(v,v)` + `sum` at 1 041 µs
      is the first candidate. **P7 took it**: both `d`-reductions are one-pass composed ops
      behind `one_pass_stats`, 1.37× on the read at the schedule's mean shape (~367 → 267 ms
      per forward). The squares land 1.4% off a one-pass floor; the matvec is still 1.97×
      off it, and *two* reads of `v` is the composed-op floor either way.
- [x] `_mix` is the same three-pass shape with no one-op form — **688.3 µs against a
      228.7 µs floor, 3.01×** (P9 corrected P7's 791.1/3.45×, which had measured the wrong
      broadcast) — and it reduces over the candidate axis, so reaching the floor needs a
      `[1, N, C, d/tp]` layout where the mix is a batched matmul. P8 priced that layout and
      it loses: the two permutes over the sealed set give back all but 1.09×.
      **P10 wrote the op instead**: 687.1 → **257.3 µs**, 2.67×, landing 1.13× off the
      floor. The `[1, N, C, d/tp]` layout was never needed — a `[1, C, N, 1]` weight is
      already what `BroadcastType::COL` consumes.
- [x] A/B `ttnn.experimental.fast_reduce_nc` against `ttnn.sum` on the mixture's dim-1
      reduce — surfaced by Phase 2's delta table, not by the profiler. **Refuted:** 687.8
      against 688.3 µs, 0.08% on the mean of two runs, against a band reaching 0.35%. The
      reduce half runs at the floor, which four kernels now confirm at ~229 µs.
- [x] Phase 10 — fused C++ op, only on measured evidence. `ROOFLINE.md` §7 puts the
      ceiling at 10.8× DRAM (215.5 ms → 20 ms per forward) with `v` resident in L1 at
      every shape that matters. Phase 9 sharpens it three times: the fused kernel is
      measured against 380 ms traced, not 622 ms untraced — after P7 against ~267 ms — and
      after P8 against **~191 ms** on the split form, so the realizable win is **~3.8×**,
      not 7.6×. The mandate narrows with it. P8 leaves the mixture as **24 of the sealed
      half's 26 passes and 96% of its traffic**, and it is the one contraction composed
      primitives provably cannot batch, because reaching it needs the candidate axis as a
      tile axis. `deepseek_moe_fast_reduce_nc_fused` already does this arithmetic —
      `init_bcast<ELWMUL, COL>` with MATH's `acc_to_dest=1`, so `mul_tiles_bcast_cols`
      MACs into `dst0` in one pass — but requires an L1-resident input and gathers its
      scores through the MoE routing convention, so the technique is liftable and the op
      is not. **Done, for the mixture only** — `ttnn.experimental.fast_weighted_reduce_nc`,
      19/19 gated at PCC 0.9999, 2.67× on the op and 1.24× on the forward. The remaining
      ~3× of §7 is the two `d`-reductions plus the softmax in the same pass, which is a
      different project: it owns the collective and P7's numerics, not just traffic.
- [ ] The rest of §7's kernel — fold the two `d`-reductions and the cross-candidate softmax
      into the mixture's pass over `v`. ~3× still on the table against P10's 153.6 ms. Not
      started, and it is deliberately not a continuation of P10: that op is 300 lines with
      no numerics risk because it changes no arithmetic, while this one owns an all-reduce
      inside a kernel and the fp32 stats chain P7 spent four iterations on. Price it before
      writing it, per §Learnings Phase 10.
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

  **NOT VALIDATED (Phase 6):** Scale defects *at depth* — mutation testing shows the depth gate
  passes both `ttnn.softmax(dim=1)` and an unshifted softmax; the op-level D13 gate is the
  detector for that class. Production `T` — still `N = 64` everywhere. Real module
  outputs: `apply_module` is an elementwise `h ⊙ w`, which exercises the residual
  bookkeeping but not KDA/MLA/MoE traffic patterns. Memory headroom at production shape —
  `block_residual` reaches `[1, 8, T, d]` and nothing has measured that against DRAM at
  `T = 5120`. Any performance claim. Distribution, real K3 weights, decode.
- **2026-07-30** — Phase 7 **PASS**: `tests/test_tt_attn_res_production.py`. **79/79**
  across the module, ~87 s.
  **Command:** `PYTHONPATH=$TT_METAL_HOME python_env/bin/python -m pytest
  models/experimental/kimi_k3_attn_res/tests/ -q`

  **VALIDATED:** Production prefill shape runs on one Blackhole device — `T = 5120`,
  `d = 7168`, `S = 8`, `block_residual` 560 MiB, 660 MiB of concatenated candidates.
  `forward` holds PCC 0.9999804 / rel err 1.54e-2 at `S = 8` and 0.9999986 / 2.80e-3 at
  `S = 0`; the split form reproduces it for all 24 read sites with 1.7 GiB of partials
  co-resident. `T` is confirmed a pure batch axis: the shared token slice is
  **bit-identical** between `T = 64` and `T = 5120` both for a single read and after the
  full 93-layer, 186-read walk, with the seal schedule intact at production shape. No host
  fallback — the only `from_torch` in `tt/` is the load-time `to_query`, there is no
  `to_torch`, and one warm read moves a counted 7.33 GB in 21.6 ms (~339 GB/s), which
  PCIe cannot do. Token counts that are not multiples of 32 hold too — `T = 1000`
  (pads to 1024) at PCC 0.9999801 and `T = 5119` (pads to 5120) at 0.9999804, with the
  slice equality intact across the padded boundary. First warm timing of the split form at
  production shape: **1.50×** (516.2 → 343.4 ms per 24-read block).

  **NOT VALIDATED (Phase 7):** That 1.50× as a *perf* result — wall clock, no profiler, one
  shape, constant live stream; Phase 9 owns it. D10's op-launch count, which `ttnn.graph`
  cannot see. `T` beyond 5120. Multi-device anything — every run above is `(1, 1)`. Real
  module outputs, real K3 weights, decode.
- **2026-07-30** — Phase 8 **PASS**: `DISTRIBUTION.md` + `ROOFLINE.md` +
  `tests/test_tt_attn_res_distributed.py` on a real `(2, 4)` mesh. **92/92** across the
  module, ~113 s.
  **Command:** `PYTHONPATH=$TT_METAL_HOME python_env/bin/python -m pytest
  models/experimental/kimi_k3_attn_res/tests/ -q`

  **VALIDATED:** The mapping the analog forces — sequence on mesh axis 0 (free), hidden on
  mesh axis 1 with one `ttnn.all_reduce` of `[1, 2(S+1), T/R, 1]` fp32 statistics per read —
  runs correctly on 8 Blackhole chips. `forward` holds PCC 0.9999778 / rel err 1.28e-2 at
  `d = 7168`, `S = 8` (single device: 0.9999804) and 0.9999986 on the `S = 0` identity path,
  across `d ∈ {256, 7168}` and `S ∈ {0, 1, 8}`. The split form reproduces the direct form at
  all 24 read sites. The 93-layer walk survives 186 chained collectives at device PCC
  0.9999500 vs torch-bf16 0.9999741, norm ratio 1.000183 — better than the single-device
  0.9999408, because the statistics reduce in fp32. Two gates that can only exist on a mesh:
  sequence-sharded and sequence-replicated runs of the same 32 tokens agree at
  **max|Δ| = 0** (and the two SP rows agree to 0), and stubbing `_reduce_stats` to the
  identity drops PCC to **0.5757407**. `hidden_size` is now the global `d` with an explicit
  `tp_factor`; `mean(v²)` divides by the global `d`; the op owns `stream_mapper` /
  `vector_mapper` / `stream_composer` and `topology` is one entry per mesh axis. Both
  reduction helpers are exact identities at `tp_factor == 1`, so the 82 single-device tests
  and every measurement above them are unchanged.

  `ROOFLINE.md` prices the mapping: every Blackhole constant cited `file:line`, the
  Gbps-vs-GB/s conversion written out once, 12 DRAM touches per read derived op by op
  (215.5 ms floor per forward at `T = 5120` on `(2,4)`), the collective at 4.6% of it, and
  two things that had to be measured rather than derived — `ttnn.all_reduce` picks a
  composite all-gather or reduce-scatter+all-gather depending on whether the shape's tile
  units divide by the participant count (`all_reduce_async.cpp:359`; an 8 KiB reduction
  measured 1.9× slower than a 5760 KiB one), and one ttnn call costs ~130 µs flat across a
  720× size range against an 88 µs break-even, so the composed op is launch-bound untraced
  and DRAM-bound traced. Phase 10's ceiling is 10.8× with `v` resident in L1.

  **NOT VALIDATED (Phase 8):** Any device-time performance claim — no profiler has run.
  Every µs in `ROOFLINE.md` §3 and §4 is a floor from a cited peak bandwidth, and its two
  measured numbers are host wall clock including Python. The split form's 1.50× was measured
  on one device and now carries 2× the collectives. Production `T` on the mesh — the
  distributed suite is `T = 64`; `T/R` at `(2,4)` is 32, one tile row, which
  `ROOFLINE.md` §5 shows is a *different collective algorithm* than production runs, so
  nothing here times what ships. Galaxy `(8,4)` and `[LINE, RING]` — untested, and the ring
  axis is exactly where a scalar topology deadlocks. `num_links` still defaults to 1 against
  production's 2. The PP boundary, real module outputs, real K3 weights, decode.

- **2026-07-30** — Phase 8 committed locally as `871bbdf968b`, 7 module files, nothing
  pushed. `DISTRIBUTION.md` reconciled with `ROOFLINE.md` (§4's 0.65% → the exact 0.595%,
  §4's 50 µs collective guess → the modelled 9.85 ms per forward, §5's two open questions
  closed). Black's pre-commit hook reformatted an assert in
  `tests/test_tt_attn_res_distributed.py`, so the committed tree differs from the measured
  one — re-ran the whole suite on it: **92 passed** in 117.70 s. **PASS**.

- **2026-07-30** — Phase 9 **PASS**: `tests/perf/test_attn_res_perf.py`, five numbered
  iterations on hardware plus a tracy attribution, all at `T = 5120` on `(1,1)` / `(8,1)` /
  `(2,4)`. No timing assertions in the file; the verdicts live in §Phase 9 perf loop.

  **VALIDATED:** Device time, at last. Traced, one read costs **201 + 342·(S+1) µs** on
  `(2,4)` and 213 + 284·(S+1) on `(8,1)` — 2 766 µs at `S = 8` against `ROOFLINE.md` §3's
  1 935 µs floor, so the composed form runs at **70% of DRAM peak**, 59% with TP. Sequence
  sharding scales 8.3× on the slope; TP costs 516 µs per read, 19%. Over the real 186-read
  schedule: **380 ms traced against 622 ms untraced**, so tracing is worth 1.64×. The split
  form holds at **1.47×** on a TP mesh (P5 — 1.43× once P6's fold landed) while issuing 49 collectives per block against
  24 — Phase 8's last open question, answered in the split form's favour. The collective is
  payload-bound at ~18 KiB/µs above a ~29 µs floor and charges tile padding at full price,
  which makes the deferred statistics fold worth **7.4×** on the collective (348 → 47 µs).
  A launch is 105 µs on one device and 152 µs on eight, so 8-device fan-out is ~46 µs of it.

  **REFUTED, four claims, three of them mine:** §6's "launch-bound even at production
  shape" — dispatch is pipelined, the criterion is `max(host, device)` not the sum, and at
  production shape the two terms are within 2%. §4's fabric model — 88.5 µs predicted at
  `num_links = 2`, **235.9 measured**, off 2.7× because the collective reaches 18–25% of
  fabric peak and AllGather runs on **2 worker cores**. §4's deferral of the 32× padding
  fix — correct untraced, wrong traced, and the decision inverts between the regimes.
  D10's "~12 ops per read" — it is **28.3**.

  **NOT VALIDATED (Phase 9):** Nothing above is an assertion — the harness logs, so none of
  it is regression-guarded. The `N`-batched matmul against 24 broadcast passes is still
  unmeasured, so the split form's remaining headroom is unknown. The fold, `num_links = 2`
  and the semaphore hoist are all priced and none are implemented. `(4,2)` skipped
  deliberately; `(8,4)`, `[LINE, RING]`, decode, PP and real weights untouched. The
  profiler ran untraced on `(2,4)` only, and its own overhead is ~35% of what it reports.

- **2026-07-30** — Phase 9 **P6**: the statistics fold implemented and shipped on by
  default (`fold_stats`), the first perf-loop iteration to change the op rather than
  measure it. **95 passed** in 145.31 s with the fold as default, including a depth walk
  parametrized over both layouts at both `T`.

  **VALIDATED:** The fold is worth **147.6 µs per read, 4.5%**, at the peak shape and
  **15.3 ms of the 380 ms forward** over the real schedule, fitting 18.6·(S+1) − 18 µs. It
  is numerically free — depth PCC moves ±5e-6 in *both* directions across 186 chained
  reads. P4's "the fold and `num_links = 2` are alternatives, not additive" holds on the
  real op: the second link adds 4.7 µs on top of the fold, so **`num_links` stays at 1** and
  the Phase-8 open question closes by taking the layout instead of the link.

  **REFUTED, both mine:** P4's ~300 µs — it is 147.6, because P4 timed a bare collective
  and never charged for the two permutes, which cost ~153 µs, above the ~40–120 µs I
  guessed. And P6's own hypothesis that the fold would *widen* its lead at small `S`: the
  permutes track the padded tensor exactly as the collective does, so both shrink together
  and at `S = 1` the fold is worth 18.3 µs while plain `links = 2` is the best row.

  **NOT VALIDATED:** Still logged, not asserted. The `~153 µs` permute cost is inferred
  from the difference of two totals, not attributed by the profiler. Depth PCC is measured
  at `T = 64` and `T = 256`, both below the tile-row threshold where `all_reduce` switches
  algorithms on the *production* shape, so the fold's numerics at `T/R = 2560` rest on the
  fp32 probe rather than on a depth walk.

- **2026-07-30** — Phase 9 **P7**: both `d`-wide reductions taken in one pass over `v`
  (`one_pass_stats`, default on) — `rms_norm_pre_all_gather` for the sum of squares,
  `matmul` for the dot. **95 passed** in 157.86 s.

  **VALIDATED:** **1.37× on the read at the schedule's mean shape** (1 868.9 → 1 367.4 µs),
  1.40× at the peak, ~**367 → 267 ms per forward** — twenty-five times the fold's 15.3 ms,
  from one iteration. `mul` + `sum` was 3.41× and 3.46× a one-pass `sum(v)` floor; the
  squares now land **1.4% off it**. The fidelity finding is load-bearing: at default LoFi
  both one-pass forms lose an order of magnitude (4.78e-2 for the statistics kernel against
  today's 2.44e-3), and HiFi4 + `fp32_dest_acc_en` restores both while costing **nothing on
  device** — 232.3 µs against 229.5 — because these reductions are bandwidth-bound. Depth
  PCC over 186 chained reads moves ≤1.2e-5, in both directions, inside P6's noise band.
  Unlike P6, the conversions are free: 896.2 µs saved in the op against 891.7 predicted
  standalone, because the slice and the `q` transpose move kilobytes where P6's permutes
  moved megabytes.

  **REFUTED:** `use_2d_core_grid=True` as the fix for the full-`d` row — it asks for *more*
  L1 (1 971 072 B against the 1D factory's 1 950 592 at `W = 7 168`), because it splits
  tokens, not the row. And my read of the factory's CB sizing as pure `4·Wt`: there is a
  fixed ~113 KiB on top, so the ceiling is `W ≤ 5 664`, not the 6 144 the clean formula
  gives. Both were found by measuring three widths rather than trusting the arithmetic.

  **NOT VALIDATED:** The forward figure is a three-point fit (linear to ~1%), not a measured
  93-layer walk — the same convention the 380 ms uses, and it inherits that convention's
  weakness. The `W ≤ 5 664` bound is measured on Blackhole's L1 only; Wormhole's smaller L1
  moves it, which is why the gate falls back rather than asserting. The matvec is still
  **1.97×** off the floor (`N = 1` wastes 31 of 32 output columns) and `_mix` is untouched at
  3.47×. Nothing here is asserted — the perf harness logs.

- **2026-07-30** — Phase 9 **P8**: `inter_block`'s 24 dot products batched into one matmul
  and one collective (`_dots_by_site`), the score chain batched with them. **95 passed** in
  165.79 s.

  **VALIDATED:** **1.26× on the split read at `S = 8`** (1 741.4 → 1 386.5 µs per site),
  1.20× on the forward by the schedule fit (**229.2 → 191.2 ms**), from 49 passes over the
  sealed set per 12-layer block down to 26 and 49 collectives down to 26 — which retires
  P5's standing worry that the split form pays twice the collectives to buy its 1.43×. The
  standalone row said 23.4× on that half (9 268.7 → 395.3 µs per block) and **96% of it
  landed in the op**, because the only thing the layout change moves is 84 KiB of queries.
  The direct form, untouched, reproduces to **0.04%** across three runs either side of the
  change, and its fitted 225.6 µs per candidate lands on the 225.7 P7 measured elsewhere.

  **REFUTED:** the batched **mixture**, and this is the more useful half of the result. The
  matmul alone is 3.2× its loop; charged for the permute in and the permute back out it is
  **1.09×** (13 597.7 against 14 867.7 µs per block), because its contracted axis is the
  candidate axis and reaching it drags 70 MiB through a 4× tile pad twice. Per-site slices
  out of the padded output are worse still (1.7 ms each against 14.9 for the whole loop). The
  backlog item that asked whether the split form's remaining ~1.9× is reachable in composed
  ops is now answered: no, and Phase 10's mandate is exactly that mixture.

  Also refuted, by the sweep rather than by a candidate: that `S = 8` is a safe place to
  conclude from. **The split form is 9% slower at `S = 1`** (710.9 against direct's 649.6);
  the crossover sits at `S+1 = 2.30` and 24 of the schedule's 186 reads are below it.

  **NOT VALIDATED:** the forward figures are three-point fits (linear to 2%), the same
  convention as P7's — 191.2 ms assumes the split form at *every* read, including the 24
  where direct is faster, so a form-selecting caller would see ~189.7. The 1.09× mixture row
  uses a `ttnn.ones` stand-in for the real weights, which is the right traffic and not the
  right numerics. Nothing here is asserted — the perf harness logs.

- **2026-07-30** — Phases **1 and 2** back-filled, nine phases late, and both found errors
  the earlier phases had already acted on. Phase 1: every analog cited at `file:line` with
  its own gate. Phase 2: AttnRes as a six-row delta against distributed RMSNorm, plus the
  `Missing/blocked ops` list.

  **VALIDATED:** the delta is four unsettled ops at the start and one now — delta 4, the
  weighted sum over candidates, which is exactly Phase 10's mandate. The blocked-op claim
  about `deepseek_moe_fast_reduce_nc_fused` is confirmed at its header: it requires
  `expert_indices_tensor` and `expert_mapping_tensor` and an L1-resident input.

  **CORRECTED:** §Gating discipline claimed our thresholds were inherited from the analog.
  They are not — the `_d_p` rmsnorm tests pass `pcc=0.99` at
  `tests/pcc/test_rmsnorm.py:137` and `:205`, while `0.9999` is `assert_with_pcc`'s
  signature default at `tests/ttnn/utils_for_testing.py:94`. Our gate is 100× stricter than
  the nearest analog's and its provenance is the repo default. Also: **mHC, the prior
  residual bringup this task is modelled on, is not tracked on this branch at all** — the
  `tt/mhc/` directory holds another branch's `__pycache__` and nothing else, so nothing was
  inheritable from it.

- **2026-07-30** — Phase 9 **P9**: A/B'd `ttnn.experimental.fast_reduce_nc` against
  `ttnn.sum` on the mixture's dim-1 reduce — a candidate Phase 2's delta table surfaced and
  seven phases of profiling had walked past. **7 passed** twice, 15.23 s and 15.72 s.

  **REFUTED:** 687.8 against 688.3 µs — 0.08% on the mean of two runs, against a band that
  reaches 0.35%, so unresolvable. The reduce half is at the memory floor, which four kernels
  now agree on to 0.26% (228.4–229.0 µs across two axes, two kernels, two fidelities). HiFi4
  is free here as it was in P7.

  **CORRECTED:** P7's mix row sits 15% high. It multiplied by the matvec's `[1, 1, 1, d/tp]`
  operand; `_mix` multiplies by a `[1, C, N, 1]` weight. Same bytes, **102 µs** apart —
  broadcasting a scalar along the last dim beats broadcasting a row across the outer dims.
  The op's mixture is **688.3 µs at 3.01× floor**, not 791.1 at 3.45×. Phase 10's ~3.8×
  headline is unaffected (it came from the whole-op 191.2 ms), its component attribution is
  not.

  **NOT VALIDATED:** the two `fast_reduce_nc` rows are timing-only — the kernel's numerics
  were probed once off-mesh at one bf16 ulp from `ttnn.sum`
  (`scratchpad/probe_fast_reduce_nc.py`) and never gated, which is enough for a refutation
  and would not be enough for adoption.

- **2026-07-30** — **Correction pass, source-only, no device run.** Read the compute-kernel
  defaults that P7 and P9 had A/B'd against. Both rows that compared "default fidelity"
  against explicit HiFi4 were mislabelled.

  **CORRECTED (P9):** its four floor rows were claimed to span "two fidelities". They span
  none — `ttnn.sum` defaults to HiFi4 + `fp32_dest_acc_en` on Blackhole (`reduce_op.cpp:109`;
  the LoFi branch is `is_wormhole`) and `fast_reduce_nc` defaults to HiFi4
  (`fast_reduce_nc.cpp:31`). The bare-vs-`HIFI` pair varies `fp32_dest_acc_en` and
  `packer_l1_acc`. Table row and test id renamed from `hifi` to `fp32acc`.

  **CORRECTED (P7):** the squares' 4.78e-2 → 2.54e-3 was attributed to LoFi truncating the
  bf16 mantissa. `rms_norm_pre_all_gather` already defaults to HiFi4
  (`rmsnorm_pre_all_gather.cpp:24`) with `fp32_dest_acc_en=false` and `approx_mode=true`, so
  the knob was the accumulator, not the multiplier. The matvec row's LoFi label **is** right —
  `ttnn.matmul` on bf16 defaults to LoFi.

  **UNCHANGED:** every timing conclusion. The compute config is free on these bandwidth-bound
  reductions either way, and ~229 µs is still the one-pass floor.

- **2026-07-30** — Phase 10 **PASS**: `ttnn.experimental.fast_weighted_reduce_nc` at
  `ttnn/cpp/ttnn/operations/experimental/reduction/fast_weighted_reduce_nc/`, wired into
  `_mix` behind `fused_mix` (default on). **19/19** in the op's own unit suite, 14.7 s;
  **161/161** in the module suite through the fused path.
  **Commands:** `PYTHONPATH=$TT_METAL_HOME python_env/bin/python -m pytest
  tests/ttnn/unit_tests/operations/reduce/test_fast_weighted_reduce_nc.py -q` and
  `... models/experimental/kimi_k3_attn_res/tests/ -q`

  **VALIDATED:** `Σ_c input[b][c][h][w] · weight[b][c][h][0]` against torch in fp32 at PCC
  0.9999, over `C ∈ {1,5,8,9,12,13}` (every granularity path including both primes and the
  degenerate `C = 1`), `Wt = 1` and `Ht = 1` (weight set turning over on every output tile and
  on none), `B = 2`, an unaligned 100-row token count, a bf16×fp32 operand pair, and the full
  `[1, 9, 2560, 1792]` production shape. Program-cache reuse with both tensors held past the
  assertion. Equivalence with `mul` + `sum` at matched precision. Five rejection cases.

  **MEASURED (traced, `(2,4)`, two runs each, all rows within 0.1%):** the mixture
  **687.1 → 257.3 µs, 2.67×**, sitting **1.13×** above an unweighted reduce over the same
  bytes; fp32 weight +3.7%. Per read site over a whole 12-layer block: split form
  1 386.3 → 991.2 µs at `S = 8` (**1.40×**), 1.24× at `S = 4`, **1.01× at `S = 1`** where the
  sealed mixture is one candidate and there is nothing to fuse. Per forward
  **191.2 → 153.6 ms, 1.24×** on the split form; 265.0 → 216.2 ms on the direct form. The
  isolated 79 MiB row predicts the block-level saving to 3% once scaled by `C = S`.

  **NOT VALIDATED:** the remaining ~3× of `ROOFLINE.md` §7 — the two `d`-reductions and the
  cross-candidate softmax in the same pass — is not started. `dim = 0` is rejected rather than
  implemented. Non-bf16 *input* is rejected, so the op has no fp32 or bfp8 path. Nothing here
  is measured on `(8,4)`, in decode, or on real K3 weights.

- **2026-08-06** — **Reference rework, CPU only, no device run.** Per D19: added
  `reference/attn_res_reference.py` as the unfolded fp64 root, restored `reference/hf_attn_res.py`
  as the external anchor with its real upstream header and `reference/LICENSE-Kimi-K3` alongside
  it. **109/109** CPU cases (36 in `tests/test_attn_res_reference.py`, 73 in
  `tests/test_torch_attn_res.py`), 58.8 s; 239 collected across the module.
  **Command:** `PYTHONPATH=$TT_METAL_HOME python_env/bin/python -m pytest
  models/experimental/kimi_k3_attn_res/tests/test_attn_res_reference.py
  models/experimental/kimi_k3_attn_res/tests/test_torch_attn_res.py -q`

  **VALIDATED:** the new root against three closed forms, scale invariance in two arms (exact at
  `eps = 0`, and an analytic `(eps/2·mean(v²))·(1−c⁻²)` bound at `eps = 1e-5` — measured 3.684e-6
  against a predicted 4.44e-6, because `eps` makes the invariance approximate by construction),
  row-stochasticity, the convex-hull bracket, non-narrowing internal precision, and no overflow at
  saturation. Rung 0b: upstream's read at its own fp32 against our fp64 root, `S ∈ {0,1,4,8}` ×
  `d ∈ {256,7168}` × two score regimes. Rung 1 now has an fp64 arm at 1e-13, which is what
  actually proves the fold rather than failing to detect an error in it.

  **VALIDATED BY MUTATION, not by the suite passing:** eight injected errors, each caught —
  mixing `k` instead of `v` (8 gates), `sum` for `mean` in the RMS (4), `eps` outside the `rsqrt`
  (3), a dropped `res_norm` gain (4), an unshifted softmax (2), a drifted vendored extract (1 —
  the anchor, and only the anchor), and on the folded side an additive `fold_query` (2) and a
  dropped rsqrt pull-out (2). Two of those rows were the point: temperature and the dropped gain
  had been caught *only* by agreement with the implementation under test until the
  constant-along-`d` closed form was added.

  **NOT VALIDATED:** every device rung. The last full LoudBox run predates this, and the device
  tests still root on `torch_functional/attn_res.py` — which shares the fold with the op, so they
  gate numerics and plumbing, not algebra. Re-rooting them on `ref` needs a device run. The fp32
  arms cannot see the `eps`-placement error (only the fp64 arm can), and the anchor structurally
  cannot either, since upstream computes in fp32 whatever it is handed.
