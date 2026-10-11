# Kimi-K3 single-layer transformer tests (LB + Galaxy) — plan

Owner: pjosipovic.

Base: `origin/pjosipovic/kda-device-request-initialization` (PR #60185, head `5aa5dee81f7`) — the
request-lifecycle scenarios depend on its device-side request-start reset
(`zero_initial_state_on_start`) and `select_request_history`. Rebase onto `main` once #60185 merges.

Paths are relative to `models/demos/deepseek_v3_d_p/` unless they start with `tests/`, `.github/`,
`tt_metal/` or `ttnn/`.

## 1. Goal

One parametrized test file that runs **every distinct Kimi-K3 layer code path as a single layer**
(plus one two-layer tail case), at **real K3 dimensions**, through the **production module stack**
(`TtKimiK3Transformer` → `TtKimiK3Block` → KDA / `ttMLA` / `TtMoe` / `TtFfn` / `TtAttnRes`), with
**synthetic weights and inputs** (no checkpoint, no golden trace), on **both**:

- **LB** — Blackhole LoudBox, 8×P150, mesh `(2,4)`, SP2×TP4, FABRIC_2D;
- **GH** — Blackhole Galaxy SC1, mesh `(8,4)`, SP8×TP4, torus_xy.

Constraints: fast (target set by the Phase 0 gate, aim ≤ 3–4 min per case cold), no `/mnt/weka` or
checkpoint dependence (LB runners mount none), same test ids on both boxes.

**CI-critical:** whether this can run in CI is decided mainly by **host reference runtime** (§5.5).
The device side of a single layer takes seconds. The CPU reference for a full layer at real dims is
tens of TFLOP per case on GH, it runs cold on every CI run because there is no persistent cache, and
it runs on runner CPUs we have not measured. Phase 0 measures it first, on the actual CI runners.

Non-goals: replacing the Galaxy checkpoint/golden-trace depth tests (they remain the only real-weight
accuracy evidence); decode; MTP; multi-slot *traced* KDA (refused by
`TtKimiK3Transformer.set_trace_controller`); perf gating (device time may be logged, not gated);
arbitrary mid-block entry through the runtime/D2D pipeline (production splits stay block-aligned via
`KimiK3Adapter.layer_split_boundaries`).

## 2. Background: what varies per layer

K3 = 93 layers. Per layer:

- attention: KDA (69) or MLA (24: ids 3,7,…,87 and 91,92 — `reference/kimi_k3_config.py:114-150`);
- FFN: dense SiTU-GLU (layer 0 only, width 33792) or LatentMoE (layers 1–92: 896 routed experts,
  top-16, sigmoid + `e_score_correction_bias`, latent 3584 / intermediate 3072, shared MLP 6144);
- AttnRes position (block 12): seals at 0,12,…,84; snapshots before layer L: `S = ceil(L/12)`
  (0..8). Two reads per layer, `pre_L` (skipped at L0, which borrows the live stream) and `post_L`;
  the seal at 12k fires between `pre_12k` and `post_12k`; the model ends with the `q_out` read +
  final RMSNorm (on the last rank, unless it runs `kv_only_last_layer`).

**Fold** (`tt/attn_res/attn_res.py:72,347-376`): `_collective` folds any `[1,C,N,1]` TP all-reduce
with `4 ≤ C ≤ 32`. Two call sites reach it, both inside `inter_block`:
- the sealed-set RMS stats (`_reciprocal_rms`, C = S) — whenever `inter_block` runs with S ≥ 4;
- the query-dot reduce (`_dots_by_site`, `[1,S,N,R]`) — only when the query batch has R = 1 and S ≥ 4.

`inter_block` runs (a) at every seal over the new sealed set and (b) **at pipeline entry** over the
inherited set for the head group. So in an unsplit walk fold runs only at seals 36,48,…,84; a split
(or single-layer) entry at any L with S ≥ 4 folds at entry, and an aligned entry (head `[pre_F]`,
R = 1) at F ≥ 48 also folds the dots. Production aligned pipeline entries at 48/60/72/84 take that
R=1 dot-fold path.

```
layer:  0   1  2  3   4 … 11 │12 … 23 │24 … 35 │36 … 47 │ … │84 … 90 91 92
attn:  KDA KDA KDA MLA       │        │        │        │   │       MLA MLA
ffn:  dense ─────────── MoE everywhere else ─────────────────────────────
seal:   ●                     ●        ●        ●            ●
S:      0→1  ── 1 ──          2        3        4  …         8
fold:   none ─────────────────────────────── RMS fold at seals S≥4 (36,48,…,84);
        (split/pipeline entry with S≥4 also folds at entry; R=1 entry also folds the dots)
extra: embed                                       last rank: kv_only (prod) or q_out+norm
```

Sealed planes are **block-local sums** (sealing clears the live stream; the next block accumulates
from zero — `reference/kimi_k3/attn_res/attn_res.py:163`), not cumulative prefixes.

## 3. Test cases

| id | Layers | Attn | FFN | Paths exercised | S in → out | Priority |
|---|---|---|---|---|---|---|
| `L0` | 0 | KDA | dense | embedding, borrowed read, first seal, `inter_block` S=1 | 0 → 1 | P0 |
| `L12` | 12 | KDA | MoE | aligned entry (R=1, S=1, unfolded), seal, `inter_block` S=2 | 1 → 2 | P0 |
| `L27` | 27 | MLA | MoE | mid-block entry, S=3 (max unfolded), MLA mid-block | 3 → 3 | P0 |
| `L36` | 36 | KDA | MoE | aligned entry S=3, seal → **first RMS fold** S=4 | 3 → 4 | P0 |
| `L84` | 84 | KDA | MoE | aligned entry R=1 at S=7 (**dot fold**), seal 7→8 (max-width RMS fold, max concat) | 7 → 8 | P0 |
| `L87` | 87 | MLA | MoE | mid-block entry S=8 (RMS fold at entry), MLA at deepest shape | 8 → 8 | P0 |
| `L92kv` | 92 | MLA | – | **production last rank**: `kv_only_last_layer` — KV write only, residual discarded, no FFN/tail | 8 → – | P0 |
| `L92full` | 92 | MLA | MoE | `q_out` model-level read + final norm (non-pipelined full model tail) | 8 → – | P1 |
| `T91_92kv` | 91–92 | MLA,MLA | MoE,– | two-layer tail: adjacent MLA, distinct local KV slots 0/1, global layer ids, kv_only last | 8 → – | P1 |
| `L89` | 89 | KDA | MoE | KDA mid-block at S=8 | 8 → 8 | P2 (drop if budget-bound; L87 covers S=8 shape, L12/L36/L84 cover KDA) |
| `L1` | 1 | KDA | MoE | mid-block shallow KDA+MoE (S=1) | 1 → 1 | P2 |

P0 runs in CI on both boxes; P1 on both boxes if the Phase 0 budget allows, otherwise nightly-only;
P2 local/manual.

## 4. Current state (why this needs code, not just a test)

Single-layer construction works today only at block-aligned L (0, 12, 24, 36, …) with
`is_last_rank=True`. Blockers:

1. `tt/kimi_k3/transformer.py:149-153` — non-first rank must start at `first_layer_idx % 12 == 0`.
2. Seal decision uses the **local** index: `TtKimiK3Block.forward` → `residual.open(self.local_idx)`
   (`tt/kimi_k3/block.py:202`) → `TtAttnResWalk.open_layer` seals on `layer_idx % block_size == 0`
   (`tt/attn_res/attn_res_stream.py:226-244`).
3. `TtAttnResWalk` doesn't know `first_layer_idx`; `_block_sites` (`attn_res_stream.py:141-152`)
   assumes the stack starts on a block boundary and always appends `q_out`; the inherited head is
   just `[q_pre[0]]` (`:206-215`).
4. Plane formulas use floor: `inbound = 1 + F//12`, `outbound = 1 + (F+n)//12` in
   `tt/kimi_k3/transformer.py:362-375`, `tt/kimi_k3/runtime.py:45`,
   `tt/runners/adapters/kimi_k3.py:251` (`pipeline_activation_planes`, sizes the D2D socket), and the
   assert in `tests/attn_res/model/test_attn_res_pipeline_split.py:141`. Wrong whenever F or F+n is
   unaligned — e.g. F=12, n=1 produces 3 planes but `outbound_planes` says 2, so any non-last
   single layer trips `_pack_handoff`'s assert. Correct: `1 + ceil(F/12)`, `1 + ceil((F+n)/12)`
   (identical on aligned values).

Also missing: a single-layer K3 reference, a synthetic `attn_res_weights` builder, and a fast
synthetic MoE path (today's `test_kimi_k3_moe` builds 896 fp32 experts ≈ 118 GB host RAM and takes
~15 min of test time on GH).

## 5. Design

### 5.1 Phase 1 — production change: unaligned entry in `TtAttnResWalk` / `TtKimiK3Transformer`

- **Walk owns the global position.** `TtAttnResWalk.__init__(..., first_layer_idx=0, is_last=True)`:
  - build the *global* site order once (`walk_sites` over global layers, `q_out` last), slice it to
    this rank's window `[first site of F, first site of F+n)` (+ `q_out` iff `is_last`), and cut at the
    **global** 24-site group boundaries. The first slice element is `pre_F` when the rank has an
    inherited set (F > 0), so the head group = sites up to and including the next seal's `pre`
    (`1 + 2*((-F) % 12)` sites; = `[pre_F]` when aligned — no special case). Host test property: the
    rank's groups are a refinement of the global groups.
  - `open_layer(local)` computes `global = first_layer_idx + local` from its own counter, asserts the
    caller's index agrees, and seals on `global % 12 == 0`. **Do not** change `block.py:202` — three
    test residual factories (`tests/kimi_k3/test_chunked_prefill.py:157`,
    `tests/kimi_k3/test_prefill_perf.py:100`, `tests/kimi_k3/test_transformer_depth.py:182`) build
    walks without F and must keep working (default `first_layer_idx=0`).
  - Validate the inherited plane count against `ceil(F/12)`; keep `block_size` parameterized.
  - Keep `_block_sites`' existing signature working (used by
    `tests/attn_res/model/test_attn_res_pipeline_split.py`); add the new behaviour behind new kwargs.
- **`q_out` on non-last ranks:** drop it from the site order (it is never read there), but note this
  changes the trailing group's R for aligned non-last ranks — e.g. a rank ending right after a seal
  can drop to R=1 and start folding the dot reduce. This is a numerics change on the *production*
  aligned pipeline path → gate it: verify `tests/kimi_k3/test_transformer_pipeline_split.py` (GH, real
  weights) before/after; if it moves, keep `q_out` in the non-last trailing group (unread) to preserve
  R, and document. `fold_queries` still requires the `output_attn_res_*` weights on every rank
  (`reference/kimi_k3/attn_res/weights.py:106`).
- **Transformer:** pass `first_layer_idx` and `is_last_rank` into the walk in the default
  `residual_factory`; replace the hard block-boundary `ValueError` with opt-in
  `allow_unaligned_entry: bool = False` (production splitter unchanged); fix the plane formulas in all
  four places (§4.4) to the ceil forms (no-op for aligned splits; the runtime/adapter remain
  aligned-only by policy — add an assert there that F is aligned, so the socket sizing claim stays
  true).
- **Trace ownership unchanged:** `_unpack_handoff` keeps slicing copies (address-stable input);
  packed inputs stay caller-owned.
- **Tests:**
  - extend `tests/attn_res/model/test_attn_res_pipeline_split.py` (pure host, already in GH
    `k3_contracts`): for every F ∈ 0..92 and n ∈ {1, 2, 12, 93−F}, groups refine the global grouping,
    each group reads against the right S, seals land exactly on global multiples of 12, and plane
    counts match the ceil formulas;
  - device split-walk test (LB 2×4, AttnRes only, synthetic linear "modules"), 96 layers deep with
    splits at F ∈ {1, 5, 11, 12, 13, 47, 48, 49, 84, 85, 92}:
    - **PCC ≥ 0.99999** (not bitwise) for the live stream and outputs vs the unsplit walk — split
      execution legitimately changes rounding: `handoff()` flushes the deferred residual with
      `ttnn.add` (`attn_res_stream.py:108,127`) where the unsplit fused read weights the addends
      separately (`attn_res_gather_softmax.cpp:328`), and entry heads with R=1 at S≥4 take the
      non-bit-neutral dot fold (`attn_res.py:356`);
    - **bitwise** for sealed planes that are copied, not recomputed, across the handoff;
    - independent fp32 torch reference walk at the same depth as a third arm.
  - `tests/kimi_k3/test_transformer_pipeline_split.py` and `tests/attn_res/` green, unchanged.

### 5.2 Phase 2 — test building blocks (`tests/kimi_k3/single_layer/`)

**`synthetic_weights.py`**
- `make_layer_state(L, *, seed)` → per-layer dict exactly as `TtKimiK3Block` / `build_attention`
  consume it (keys in §10): `attn_norm_weight`, `ffn_norm_weight` (`1 + 0.02·randn`, never ones —
  a missing norm becomes `torch.empty`, #54841), plus
  - KDA: `kda_weights` at `reference/kimi_k3_config.py::kimi_k3_kda_config()` — generalize
    `tests/kda/utils.py::random_weights` to take a seed (it hardcodes `20260723`);
  - MLA: `mla_weights` incl. `g_proj` (distribution of the `random_weights` fixture,
    `tests/conftest.py:911`);
  - L0: `ffn_weights` (dense 33792);
  - MoE: `gate_weights`, `shared_expert_weights`, `latent_weights` via `tt/moe/init_helpers.py`
    (`create_gate_weights` / `create_shared_expert_weights` / `create_latent_weights`), and routed
    experts per §5.4.
- `make_attn_res_weights(L_range, *, seed, query_scale)` → checkpoint-key dict for
  `load_attn_res_weights` (`{prefix}layers.{L}.{self_attention,mlp}_res_{norm,proj}.weight` +
  `output_attn_res_{norm,proj}.weight`, required on every rank). **Calibrate `query_scale`** so the
  candidate softmax has a target mean entropy (e.g. ≥ 0.5·log(S+1)) on the synthetic state; assert it
  in the test — near-one-hot softmax would hide sealed-set ordering bugs.
- Embedding `[163840, 7168]` bf16 generated directly in bf16 (L0 only, ~2.3 GB host);
  `norm_weight` (L92full only).
- All seeded per (L, component); identical on LB and GH.

**`synthetic_state.py`**
- Inherited AttnRes state for L > 0: `S = ceil(L/12)` sealed planes + 1 live plane `[N, d]`; planes
  are block-local sums, generated with distinct per-plane RMS (stress distribution, documented as
  such) and per-token variation, plus a small fraction of heavy-tailed rows. Packed `[1, S+1, N, d]`
  via `TtAttnRes.stream_mapper` (SP→dim 2, TP→dim 3), plane 0 = live.
- L0 input: token ids uniform over the vocab via `tt/runners/input_prep.py::prepare_prefill_input_tensor`
  (block-cyclic SP order).
- MLA continuation prefix: produced by running chunk 1 on device (tests the real KV write path).

**`reference.py`** — single-layer K3 reference from component references (runtime budget,
sampling and overlap rules in §5.5 — they shape this module's API)
- AttnRes: `reference/kimi_k3/attn_res/attn_res.py` (`AttnResStream`, `attn_res_layer`,
  `fold_query`); set `stream.block_residual = sealed` (device `[1,S,N,d]` → ref `[N,S,d]`), pass the
  global L; `q_out` + RMSNorm for `L92full`.
- KDA: `reference/kda/layer.py::kda_forward_reference(hidden, weights, cfg, state)` with explicit carry
  in/out. The existing cache `tests/kda/reference_cache.py` is hard-wired to `KimiK3TestCase` /
  `KIMI_K3_FIRST_KDA_LAYER`, takes no initial state and writes under `ttnn.CONFIG.model_cache_path`
  → generalize it (key: content hash of weights + inputs + initial state + config + source
  fingerprint) or add a sibling cache for this suite.
- MLA: `single_layer/reference_ops.py::mla_forward` (fp32, NoPE, output gate; pinned against
  `MLAReference` by `test_reference_ops.py`). It takes a `prefix_kvpe` and offsets the causal mask
  for it, so chunk 2 runs against chunk 1's KV without recomputing the prefix; the prefix KV comes
  from the reference's own chunk-1 pass (and the device's chunk-1 KV is compared to it separately).
  Do **not** use `reference/mla_reference.py::MLAReference.forward` incrementally (causal mask not
  offset for a prefix, `:188`); `cpu_mla_reference` stays usable as a whole-request cross-check.
- Dense FFN: `KimiMLP` + `SituAndMul` (betas 4/25, `kimi_k3_hf_config()`), `reference/kimi_k3/modeling_kimi_moe.py`.
- MoE: compact pooled reference (§5.4) **driven by the device's selected expert ids and weights**
  (exported from an eager diagnostic forward, outside capture) to remove top-k tie flips between
  the fp32 device gate on bf16 activations and the reference gate; a separate gate check compares
  device vs reference top-k sets and bounds the flip rate (e.g. ≤ 0.5% of token-slots).
- Norms in fp32 with bf16 rounding where the device rounds (as
  `tests/kimi_k3/test_transformer_kda_loudbox.py::reference`). Config: override
  `kimi_k3_hf_config()` max positions ≥ request length (default 8192 < GH 2-chunk 10240).

**`harness.py`**
- Build `TtKimiK3Transformer(num_layers=n, first_layer_idx=L, is_first_rank=(L==0),
  is_last_rank=(case in {L92kv, L92full, T91_92kv}), kv_only_last_layer=(case endswith "kv"),
  allow_unaligned_entry=True, is_chunked=True, slot_num=1|2, max_seq_len ≥ 2 chunks (+1 chunk for S3),
  weight_cache_path=None, build_tail=(case == L92full), num_links=2,
  topology=per_axis_topology(), routing_use_l1_small_for_semaphores=True,
  dispatch_buffer_capacity_factor=2)` — i.e. the production K3 block config
  (`tt/runners/adapters/kimi_k3.py:52`, `tt_prefill_runtime.py:57`).
- `weight_cache_path=None` (no tensorbin writes; ~16–17 GB/MoE layer otherwise; `mark_layer_cached`
  is a no-op on None — `tt/kimi_k3/weights.py:260-263`).
- KDA: allocate `KdaStates` slabs and `bind_slabs` as production does; read carries with
  `kda_states.read(L, slot=…)` (**global** layer index — `tt/kimi_k3/kda_state.py:41-50,104`).
- MLA: `utils/kv_cache_utils.py::allocate_mla_kvpe_cache`; readback via `gather_cache_tp0` /
  `unrotate_cache_layer`, with `mla_row_permutation` for chunk offsets.
- Outputs: non-last → packed handoff `[1, S_out+1, N, d]` (every plane compared); `L92full` →
  `norm(finish())`; kv_only → `None` + KV slot.
- Branch isolation: compare the **layer delta** (live-stream change = attention out + FFN out, or the
  post-seal partial) and per-branch tensors where exposed, not only the whole hidden state — a large
  inherited residual must not hide a broken branch. Add magnitude (rel-L2) bounds next to PCC.
- `finally`: copy outputs out, release traces, `release_sub_device_managers()`,
  `kda_states.deallocate()`, free slabs.
- Mesh lifecycle: one test function per (case, mesh) running all scenarios in order (the existing
  `mesh_device` / `device_params` fixtures are function-scoped — root `conftest.py:312,553`); use
  `DS_PREFILL_REUSE_MODEL=1` reuse if it applies, otherwise one mesh open per case. Device params via
  **keyword-only** helpers: `fabric2d_device_params(fabric_payload_size=…, l1_small_size=…,
  trace_region_size=…)` / `torus_xy_device_params(...)` (`tests/fabric_profiles.py:16,90`).

### 5.3 Scenarios (run in order inside each case)

Per-chip rows: **640** on both boxes ⇒ LB chunk 1280 (SP2), GH chunk 5120 (SP8) — 640 is a legal
KDA local T (`tt/kda/config.py:118`); starts 32-aligned (`attention.py:34`). If Phase 0 shows GH
reference time is prohibitive, GH reference-backed scenarios (S1/S2/S4) drop to 320 rows/chip
(chunk 2560) while S3/S5/S6 stay at 640.

| # | Scenario | Cases | Oracle |
|---|---|---|---|
| S1 | one chunk, start 0 | all | reference: layer delta + outputs + KDA carries / MLA KV (PCC + rel-L2); finite valid rows |
| S2 | continuation: chunk 1 at 0, chunk 2 at `chunk`; plus rotated starts `32` and `local_rows+32` | all | full-prefix reference (KDA carry flows; MLA chunk 2 attends to chunk-1 KV via `mla_row_permutation`) |
| S3 | dirty restart: **B (2 chunks) → A (2 chunks) → B (2 chunks)** on one model; also short dirty restarts (1- and 2-token B after A) | KDA, MLA | **bitwise** B-first vs B-after-A for outputs, carries, valid KV rows; continuing B's second chunk exposes wrong final carries. Stale KV rows beyond B are intentionally untouched — excluded |
| S4 | unaligned end (`actual_end = start + chunk − 37`), 1- and 2-token requests | all | reference on valid rows; padding ignored; valid rows finite (catches NaN leaking via routing/collectives) |
| S5 | determinism: S1 ×3 | all | bitwise |
| S6 | traced: compile pass, then capture with on-device `(slot,start,end)` metadata (`test_prefill_transformer_chunked.py:2154-2266`); replay S2's chunks, S3's restart and S4's unaligned/1-/2-token metadata | all (single slot) | **PCC ≥ 0.999 vs eager** (eager KDA uses host bounds `K3KdaChunk`, trace uses device metadata — different programs; prior art `test_prefill_chunk_traced_vs_eager.py:48-50`); **bitwise replay-vs-replay** |
| S7 | eager multi-slot: `slot_num=2`, interleave two requests across slots (A0 chunk1, B1 chunk1, A0 chunk2, B1 chunk2) | KDA, MLA | bitwise vs each request run alone |
| S8 | KDA slab lifecycle (ported from `test_transformer_kda_loudbox.py`): carry buffer addresses stable across requests, `import_layer` round-trip bit-identical, one-shot vs chunked at two capacities | KDA cases | bitwise / PCC as in the old test |
| S9 | MoE routing identity + negative controls: (a) expert-id permutation / per-chip perturbation must fail the oracle; (b) sealed-set plane swap must fail the AttnRes oracle; (c) MoE dispatch counts/offsets from an eager `return_intermediates=True` diagnostic forward (outside capture) — no overflow/dropped tokens (`tt/moe/tt_moe.py:1030`) | MoE cases (a, c), S≥2 cases (b) | oracle sensitivity proven |

Thresholds (initial; set from Phase 3 measurements with ≥ 2× margin):
- KDA carries vs reference 0.9995 (component bar, `tests/kda/layer/test_acceptance.py`);
- MLA vs `cpu_mla_reference` per `test_mla` K3 bars; KV ≥ 0.999;
- MoE branch ≥ 0.965 (`tests/pcc/test_ttnn_moe.py:1280`), dense FFN ≥ 0.97 (`tests/pcc/test_ffn.py:205`)
  — or tighter where the device-routed reference removes routing noise;
- layer delta / handoff / `L92full` output: measured − margin, plus rel-L2 bound;
- pass-through sealed planes (copied, not resealed): bitwise.

### 5.4 Synthetic MoE weights (Phase 0 decides)

Routed experts dominate everything: 896 experts × 3 × (3584×3072) ≈ 29.6 G params per MoE layer.
Pooling host weights alone does **not** fix cost — `_convert_and_cache_expert_weights`
(`tt/moe/tt_routed_expert.py:163-286`) still tilizes and bf4-packs all 896 experts (112/chip LB,
28/chip GH; ≈16.6 GB bf4 across the mesh per case), and the reference still does T·16 expert GEMMs
(~1e13 FLOP at GH 10240 tokens).

Design constraints:
- per-expert identity must survive bf4 (`tt_metal/impl/data_format/blockfloat_common.cpp:245`:
  16-element shared exponent, 3-bit magnitude): small continuous scales (`1 + 0.25·e/896`) collapse
  (only ~33 distinct bf16 values in [1, 1.25]; pool neighbours become byte-identical after bf4);
- `Q_bf4(s·W) ≠ s·Q_bf4(W)` in general → the reference must use the **dequantized device weights**
  (or perturbations that commute exactly with bf4);
- latent RMSNorm suppresses common scaling → use sign/permutation, not magnitude, as identity.

Options (Phase 0 measures, picks one):
- **(c, preferred) device-side expansion:** convert a pool of P (16) experts once per mesh position,
  then build each local expert's device buffers by an exact transform per (chip, local index):
  sign flip of whole `down^T` K-rows (16-element blocks — commutes with bf4 exactly) and/or a
  power-of-two scale (exponent shift, exact while in range), or a whole-row permutation. Needs a
  `TtRoutedExpert` hook to accept pre-built device weights (or a "weight provider" producing
  per-(chip,local) tensors on device). Cross-chip mis-dispatch then changes signs → highly visible.
  Reference: dequantize pool once, apply the same exact transform per expert, group GEMMs by pool id.
- **(b) host-side pack once:** pack each pool entry to bf4 once, apply the exact transform on packed
  bytes (sign bit flip per element / exponent add) per expert, upload. Needs a converter hook for
  pre-packed tiles.
- **(a) persistent synthetic TTNN cache:** rejected as default — LB runners have only read-only
  `/mnt/MLPerf`, and caches cost ~16–17 GB/layer.

Validation: a CPU unit test with **more experts than pools** (e.g. 64 experts, P=8, repeated pool ids,
distinct transforms, shared expert + latent norm on) comparing the compact reference to `TorchMoe` /
`run_reference_moe`; plus the S9(a) negative control on device. Keep pool weights in bf16 (P=16 ≈
1 GB) — fp32 P=32 was ~4.2 GB plus stacking temporaries (`tt_routed_expert.py:225`).

### 5.5 Reference runtime budget (CI gate)

The reference, not the device, decides CI feasibility. Rough cost at real K3 dims (fp32, 2 FLOP per
weight per token; to be replaced by measurements):

| Component | Per token | Notes |
|---|---:|---|
| KDA projections (q,k,v,g,o + f/b) | ~0.9 GFLOP | plus a **per-token Python recurrence loop** (`reference/kda/ops.py::kda_recurrent_reference`) — T iterations, overhead-bound |
| MLA projections (q_a,q_b,kv_a,kv_b,o,g) | ~0.46 GFLOP | plus causal attention in the absorbed (latent) form ≈ 96 heads × T²/2 × (576+512) × 2 → **~11 TFLOP at T=10240**; `cpu_mla_reference` runs the module in **bf16 on CPU** (measured end to end below) |
| LatentMoE (16 routed + shared + latent + gate) | ~1.4 GFLOP | routed GEMMs dominate |
| Dense SiTU-GLU (L0) | ~1.45 GFLOP | |
| AttnRes reads / seal | small | token-wise, S+1 ≤ 9 planes |

| Box | Tokens (1 / 2 chunks) | KDA+MoE layer, one 2-chunk pass | MLA+MoE layer, one 2-chunk pass |
|---|---|---:|---:|
| LB | 1280 / 2560 | ~6 TFLOP | ~5 TFLOP + 0.7 TFLOP attn |
| GH | 5120 / 10240 | ~24 TFLOP | ~19 TFLOP + 11 TFLOP attn |

At a realistic 0.5–2 TFLOP/s of host fp32, one GH pass takes roughly 15 s to 1 min. Several passes
per case × 7–9 cases × bf16 MLA attention × the KDA loop can turn that into tens of minutes. That
would not fit the 18 min free in `bh_loudbox` or the 0 min free in `bh_sc1`.

**Measurement (Phase 0a, first deliverable):** `tests/kimi_k3/single_layer/bench_reference.py`, host-only
and with no device, runs each component reference at the real dims for T ∈ {640, 1280, 2560, 5120,
10240}. It runs cold, with `CI=true` (MLA cache off), an explicit `torch.set_num_threads`, and both
fp32 and bf16 for MLA. It reports wall time, peak RSS and thread count. Run it **on the CI runner
hosts themselves** (a dispatch-only job on an LB runner and a GH runner), because their CPUs may differ
from dev boxes. The result is a table in this file: seconds per component per T per box.

**Measured (Phase 0a, 2026-10-10), dev LB host `bh-lb-19`, AMD EPYC 9124 (16 cores, no AMX),
torch 2.11 CPU, 16 threads, cold, fp32 functional references (`reference_ops.py`), 512 sampled rows:**

| Component | variant | T=1280 | T=2560 | T=5120 | T=10240 |
|---|---|---:|---:|---:|---:|
| KDA | full | 5.9 s | 7.1 s | 15.1 s | 29.8 s |
| KDA | sampled | 5.8 s | 6.6 s | 13.7 s | 27.1 s |
| MLA | full | 1.2 s | 2.9 s | 8.1 s | 25.0 s |
| MLA | sampled | 0.6 s | 0.8 s | 1.3 s | 2.3 s |
| MLA module (`cpu_mla_reference`, bf16) | full | 2.5 s | 4.5 s | 10.1 s | 26.3 s |
| MoE (P=16 pool, 896 experts) | full | 1.5 s | 3.1 s | 8.2 s | 16.5 s |
| MoE | sampled | 0.7 s | 0.7 s | 0.7 s | 0.7 s |
| Dense FFN (L0) | full | 1.7 s | 3.4 s | 6.5 s | 12.3 s |
| Dense FFN | sampled | 0.8 s | 0.8 s | 0.8 s | 0.8 s |

Raw: `bench_reference.py --tokens 1280 2560 5120 10240 --rows 512`; one repeat, so every number is the
first (cold) call; the box also ran an 80 s host pytest partway through (slightly pessimistic).

Reading:
- **The reference is not the CI blocker on this host.** One 2-chunk reference pass per case costs
  at worst ~28 s on GH (KDA+MoE, sampled) and ~7 s on LB; ~46 s on GH unsampled. With lever 1 at
  one pass per case, the 7 P0 cases add ≤ ~3.5 min on GH and < 1 min on LB. Keeping S2's two
  rotated starts in CI makes it ~3 passes per case (~10 min GH); they are the first thing to move to
  nightly. P1 `T91_92kv` cannot sample layer 91's MoE (it feeds layer 92's token mixing), so it pays
  the full ~16.5 s MoE term per pass.
- **KDA is bound by the per-token loop** (Python overhead and/or streaming the ~6 MB fp32 state
  per step; not separated yet): sampling barely helps, so lever 5 (a chunked/WY recurrence) is the
  one that would move it. Not needed at these numbers. The T=1280 KDA point (5.9 s) is above the
  ~2.7 ms/token trend of T ≥ 2560 and is likely noise from the concurrent pytest; T=640 measured
  1.8 s in the smoke run (now in the default `--tokens`).
- **MLA and MoE: sampling is the lever** (MLA 25 → 2.3 s, MoE 16.5 → 0.7 s at T=10240).
- End to end, the bf16 module reference (`cpu_mla_reference`: module construction, cache-key hashing,
  chunked SDPA) took 2.5 / 4.5 / 10.1 / 26.3 s against 1.2 / 2.9 / 8.1 / 25.0 s for the fp32 functional
  path — a different implementation, so this does not isolate dtype. Either way both lose ~10× to the
  sampled fp32 path at T=10240, which is what the suite uses.
- Not yet measured: synthetic weight generation (the 163840×7168 embedding for L0, MoE pools), and
  the **CI runner hosts** (LB and GH runners via dispatch) — still required before the CI budget
  request, because the KDA loop is single-thread-latency bound and runner CPUs differ.

**Re-measured 2026-10-11, same host, 32 threads, after the chunked KDA recurrence (16-token blocks
in matmul form instead of the per-token loop):**

| Component | variant | T=1280 | T=2560 | T=5120 | T=10240 |
|---|---|---:|---:|---:|---:|
| KDA | full | 1.4 s | 2.6 s | 5.0 s | 9.9 s |
| KDA | sampled | 1.3 s | 2.3 s | 3.9 s | 7.8 s |
| MLA | full | 0.9 s | 2.3 s | 6.1 s | 18.8 s |
| MLA | sampled | 0.4 s | 0.7 s | 1.0 s | 1.7 s |
| MoE | full | 1.0 s | 2.2 s | 6.2 s | 12.0 s |
| MoE | sampled | 0.5 s | 0.5 s | 0.5 s | 0.5 s |

KDA is now linear in T (~0.8 ms/token sampled, dominated by the fp32 projections); sampled MLA and
MoE stay near constant. GH (4x the tokens per chunk) therefore costs ~4x LB on KDA layers'
reference and about the same on the sampled MoE/MLA terms. Estimate for a GH KDA+MoE case: reference
~55 s (LB 18 s), plus ~4x on synthetic request inputs and host readback/placement (~25 s more) --
roughly 2 min a case, ~15-18 min for P0 -- unmeasured until a GH run.

**Budget model:** for each case, wall time = synthetic weight gen + device weight build/upload + device
scenarios + **reference on the critical path**. CI time = Σ over the cases in the leg + mesh open.
Go/no-go: the reference on the critical path is ≤ ~25% of per-case wall, and the leg fits the bucket
headroom (or the approved raise) at timeout = measured + 50%.

**Levers (apply in order until the budget fits; each is designed in from the start):**
1. **One pass serves many scenarios.** Every path is causal, so the 2-chunk S2 pass at start 0
   covers S1 (first chunk), S4's unaligned end (a row slice), and S9's controls. Scenarios S3, S5,
   S6, S7 and S8 are device-vs-device and need **no reference**. Target: **1 reference pass per case**
   (plus 1 per rotated start, which is the first thing to move to nightly).
2. **Token-sampled downstream.** AttnRes, FFN/MoE and the output projections are token-wise. Compute
   the full T only where it is mathematically required: the inputs to the KDA recurrence, the KDA
   carries, and MLA K/V. Compute everything after the token-mixing step only for a stratified row
   sample, e.g. 256–512 rows covering each SP shard, the chunk boundaries, the first and last valid
   rows, and the heavy-tailed rows. For MLA, compute sampled query rows against the full K/V. This
   cuts the dominant MoE term about 20–40× and the MLA attention term by T/rows.
3. **fp32 on host**, with bf16 rounding emulated only where the device rounds. Never run CPU GEMMs
   in bf16: move `cpu_mla_reference` to fp32 or use a local fp32 path. Pin the thread count.
4. **Overlap with device work.** Inputs are seeded, so the reference can start in a worker process
   at test start and run while the device converts and uploads weights (the host CPU is otherwise
   idle then). MoE routing: the reference uses its own gate. Only the rows whose top-k set differs
   from the device's are recomputed with the device's ids once those are available, which is cheap.
5. **Vectorize the KDA recurrence** in chunks (WY/chunked form) if the per-token loop dominates.
   Keep the loop as a slow oracle in a unit test.
6. **GH rows per chip 640 → 320** for reference-backed scenarios.
7. **Last resort:** reference-backed scenarios go nightly only. CI runs S1 (sampled) plus the
   device-vs-device scenarios S3/S5/S6/S7, which still catch most lifecycle bugs.

## 6. CI wiring (Phase 4)

**Budget first** (aggregated from the current matrices). The request is sized from the Phase 0
reference measurements (§5.5); without them it is a guess.

| Bucket (`verify_time_budget.py`) | Declared / budget | Free |
|---|---:|---:|
| models / e2e / bh_loudbox | 182 / 200 min | 18 min |
| models / demo / bh_sc1 | 1836 / 1836 min | **0** min |

→ Raise `.github/time_budget.yaml` (needs budget owner approval) or reallocate before adding entries.
Deleting a command without lowering its entry's timeout frees nothing.

- **LB:** new entry in `tests/pipeline_reorg/blackhole_e2e_tests.yaml`, `model-name: kimi_k3`
  (selected by `blackhole-e2e-tests.yaml` `model=kimi_k3` / `all`; no checkpoint mounts needed), sku
  `bh_loudbox`, cmd `scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_single_layer.py -k "LB-2x4 and (P0 or P1)" -vvv --tb=short`.
  Timeout = measured + 50%. Keep `test_transformer_kda_loudbox.py` in
  `bh-lb-disaggregated-prefill-accuracy` until S8 is green, then remove it and lower that entry's
  timeout accordingly.
- **GH:** a **separate** entry in `tests/pipeline_reorg/blaze_models_prefill_tests.yaml` (the 46-min
  `k3_accuracy_suite` can't absorb 7–9 MoE builds), `fabric_profile: torus_xy` (the impl sets the
  descriptor and `PREFILL_TORUS_XY_CERTIFIED` — `.github/workflows/blaze-models-prefill-tests-impl.yaml:208`),
  `test_type: k3_single_layer`, tags `[kimi_k3, kda, mla, moe, attn_res, k3_single_layer]`,
  `sp_config: sc1`, mpirun wrapper as in the K3 sections.
- **Non-skip guard:** each CI cmd asserts the expected number of executed (not skipped) tests — a
  missing mesh-topology mark or fabric gate must not turn the leg green with zero coverage.
- Cmd blocks run under `bash -eo pipefail`; the lint (`.github/scripts/utils/lint_pipeline_cmds.py`)
  rejects `|| var=…`, `|| true`/`|| :`, `set +e`; multi-command blocks use fail-fast
  (`|| exit $?` style, as the K3 sections do).
- Run locally before pushing: `verify_time_budget.py`, `validate_test_type_selection.py`,
  `verify_test_owner.py` (needs the active-owner input list), `lint_pipeline_cmds.py`.
- Independent quick win: switch the GH traced K3 leg from `iters1 and no_determinism` to
  `two_iters and with_determinism` (real-weight L24 same-prompt restart, bitwise KV). Complements S3
  (different-request restart), does not replace it.

## 7. Phases, deliverables, exit criteria

| Phase | Work | Exit criteria |
|---|---|---|
| 0a Reference timing (1 d, first) | `bench_reference.py` (§5.5) on a dev box, then via dispatch on an LB and a GH CI runner | per-component s/T table per box in this file; per-case reference estimate after levers 1–3; decision on which levers are mandatory and whether reference-backed scenarios can run in CI or must be nightly |
| 0 Feasibility gate (2 d) | Cold measurements on LB and GH for `L12` and `L87`, `weight_cache_path=None`: MoE weight build (options b/c prototype), upload, peak host RAM, peak device DRAM/L1, disk temp; KDA / MLA / MoE reference time at 640 and 320 rows/chip, 1 and 2 chunks; trace capture fits `trace_region_size` for one MoE layer on LB (112 experts/chip) | table in this file; MoE option chosen; per-case cold runtime estimate for S1–S9 **including reference on the critical path**; go/no-go on GH row count; CI budget request drafted |
| 1 Unaligned entry (2–3 d) | §5.1 code; extended host test; device split-walk test | host test green for all F/n; split-walk within PCC bound, copied planes bitwise; `tests/attn_res/` (LB) and `test_transformer_pipeline_split.py` (GH) green; aligned-pipeline numerics unchanged or documented |
| 2 Building blocks (3–4 d) | §5.2 modules; MoE option from Phase 0 incl. `TtRoutedExpert` hook; pooled-reference CPU unit test; seedable KDA `random_weights`; reference cache | S1 green for `L0`, `L12`, `L36` on LB |
| 3 Scenarios (3–4 d) | S1–S9 for P0 cases on LB, then GH; P1 cases | all P0 green on both boxes; thresholds from measurements with margin; negative controls fail as expected; runtime per case recorded |
| 4 CI (1–2 d) | budget approval; LB + GH entries; non-skip guards; retire old LB test after S8; GH `two_iters` switch | dispatched runs green: `blackhole-e2e-tests.yaml` (`model=kimi_k3`, LoudBox) and `blaze-models-prefill-tests.yaml` (`k3_single_layer`) |

### 7.1 Status (2026-10-10, LB 2x4 `bh-lb-19`)

| Phase | State | Evidence / notes |
|---|---|---|
| 0a | done on the dev LB host | §5.5 table. Still owed: the same bench on an LB and a GH CI runner. |
| 0 | MoE option **(c) chosen and built**; rest partly measured | `PooledRoutedExperts`: 16 pools replicated, each (chip, slot) expert made distinct on device by row-sign flips of `down^T`, exact under bf4 (all 112 slots x 8 chips checked; only signed zeros differ). Build: 112 slots/chip in **7.5 s** on LB. Per-case S1 wall on LB (warm JIT, incl. weight gen + build + reference): **23-100 s**, all 11 cases 4 min. Not yet measured: GH, trace region, peak DRAM/L1. |
| 1 | **done** (commit `cfdee245060`) | 93-layer split walk bitwise-equal to a settled undivided walk at every tested split on LB; tests/attn_res 813 pass. Owed: GH-only `test_transformer_pipeline_split.py`. Found and fixed an op bug on the way: `attn_res_weighted_reduce_nc` overran its CBs for R = 7, 11, ..., 31 (commit `1bbcb98ef8a`). |
| 2 | S1 green for **all 11 cases** on LB | Synthetic MLA weights recalibrated so the attention branch is visible (default 0.02 std hid it under the FFN); every comparison asserts PCC and relative L2; branch-visibility and AttnRes-entropy guards. Device MLA alone measures PCC ~0.998 vs the reference, so MLA cases carry a looser layer bar. Deferred to Phase 3: device-routed MoE reference + top-k flip rate (needs an eager routing export), rotation-aware KV unrotate for rotated starts. |
| 3 | **S1-S9 implemented; all green on LB** (one test per case; S6 traced and S8 one-shot reuse the case's model, S7 builds a two-slot one) | Per-case model runs S1 (+ guards, MoE checks) -> S5 -> S2 -> S4 -> S3. MoE: device gate vs reference gate on identical bf16 input disagrees on 0.63-0.86% of picks; MoE alone with the reference driven by the DEVICE's routing (eager `moe_diagnostic`) matches at PCC 0.99913; S9(a) one-chip mis-dispatch caught (1-PCC 0.022 vs bar 0.0017); dispatch never near capacity. S6: traced replays are **bitwise equal to eager** here (better than the planned 0.999), replay-vs-replay bitwise, traced restart bitwise. S7 interleaved slots bitwise. S8 chunked vs one-shot carries PCC 1.000000. S9(b): live<->sealed swap is a no-op at sealing and kv_only layers (set semantics) -- used only where it is a real perturbation; plane duplication otherwise. LB wall: see §7.2. GH not run (needs push + Blaze dispatch). |
| 4 | YAML drafted locally, not committed | needs: budget approval (LB e2e has 18 min free vs 14.9 min measured for the whole directory, ~10 min for P0 + MoE exactness; GH bh_sc1 has 0), and moving the device split-walk test out of the 10-min `bh-lb-kimi-k3-attn-res` entry. |

### 7.2 LB wall time (2x4, warm JIT, 2026-10-11)

`test_single_layer` is ONE test per case: the one-shot model (P0 KDA cases, built first, see the
open issue below), the case's model for S1+guards+MoE, S5, S2, S4, S3, S8 and S6, then a two-slot
model for S7/S8 (traced K3 is single-slot). The large synthetic matrices (attention, FFN, MoE)
are shared by every layer of a kind and memoized (norms and AttnRes queries stay per layer; the two
layers of a multi-layer case get different variants). Whole directory on LB: **25 passed in
14.9 min**; the 11 cases 803 s of calls -- P0 51-113 s (L92kv 51, L27 59, L12 63, L0 69, L36 83,
L87 96, L84 113; P0 total ~8.9 min), P1 58-83 s, P2 38-91 s; `test_synthetic_moe` 74 s.
History: 33.8 min for the whole directory (36 tests) on 2026-10-10, before the reference speed-ups
(chunked KDA recurrence, row sampling, shard-concat readback) and the per-case merge.

Per L12 case (profiled, 47 s before the merge): host reference 18 s (KDA projections ~8, MoE on the
sampled rows 6, KDA recurrence 4), model build 16 s (synthetic weights 6, upload conversion 5,
expert placement 3), request inputs 5 s, device runs + readback ~4 s.

**Open issue found in Phase 3 (not a production path, reproduced):** on one mesh, after a 2-slot
`TtKimiK3Transformer` (L36) has run a forward and been closed, a NEW model with a different chunk size
computes a wrong MoE (MoE alone PCC 0.954 vs 0.999; layer 0.977 vs 0.998). Not reproduced by: a
2-slot model that never ran, a 1-slot first model, a same-size second model. Clearing the program
cache does not cure it, and filling DRAM with garbage does not reproduce it. Hypothesis (unproven):
per-mesh state (semaphores / sub-device managers / CCL) rather than programs; stale DRAM is not
ruled out. The suite avoids the pattern: in each case's test the one-shot model is built first, and
`mesh_device` is function-scoped, so no two-slot model precedes it on the same mesh. Worth a ticket.

**Commit gate (every phase, every commit).** No code is committed until it has been reviewed by
**both** of these:
- a **separate Claude Code instance** (fresh session, not the one that wrote the code; e.g.
  `claude -p` with the staged diff and this plan);
- **Codex Astra** (`codex` CLI).

Procedure for each unit of work that is meant to become a commit:
1. Stage exactly the intended change (`git add …`). The review covers `git diff --cached` plus the
   relevant plan section and the files the diff touches.
2. Run both reviews independently. Neither reviewer sees the other's output. Each is asked for
   correctness bugs, deviations from this plan, numerics and threshold soundness, and test gaps.
3. Address every finding: fix it, or record why not. Re-stage, then re-run both reviews on the new
   diff. Repeat until neither reviewer has open blocking findings.
4. Commit. The commit message body has a short `Reviewed-by:` note for each reviewer that
   summarizes their findings and resolutions. Larger review notes go in a scratch location and are
   not committed (like this plan, they stay out of the PR).

This applies equally to production changes (§5.1), test code, CI yaml (§6) and plan-driven
refactors. A commit that skips the gate is reverted and redone.

## 8. How to run

```
./build_metal.sh --release
# LB, all P0 cases
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_single_layer.py -k "LB-2x4 and P0" -vvv --tb=short
# one case
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_single_layer.py -k "LB-2x4 and L36" -vvv
# host-only walk tests
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/attn_res/model/test_attn_res_pipeline_split.py -vvv
# GH: dispatch blaze-models-prefill-tests.yaml with test-type k3_single_layer
```

Blackhole boxes are flaky on repeated full-mesh reopen within one pytest session (ethernet-init
timeouts): reset with `tt-smi -r` between runs; prefer one mesh open per pytest process.

## 9. Risks and open questions

1. **Production change to the walk** — default-off kwargs, aligned behaviour unchanged except the
   possible `q_out`/R change (§5.1, gated). Fallback if rejected: a test-only layer loop composing
   `KimiK3LayerSchedule` / `build_attention` / `TtKimiK3Block` / `KdaStateCache` with a test-side walk
   (doesn't test the production transformer wiring).
2. **MoE cost** — the Phase 0 gate; needs a `TtRoutedExpert` hook either way (b or c).
3. **Reference runtime — the CI-deciding risk.** At GH's 5120 tokens/chunk the cost comes from the
   KDA per-token Python loop, bf16 MLA attention on CPU, and the MoE GEMMs. CI runners have no
   reference cache and their CPUs are unmeasured. Measured first (Phase 0a, on the runners); levers
   in §5.5. If the levers still don't fit, reference-backed scenarios go nightly.
4. **Synthetic ≠ real activations** — mitigated by calibrated AttnRes entropy, heavy-tailed rows,
   branch-delta checks; real-weight coverage remains on GH depth tests.
5. **CI budget** — `bh_sc1` demo bucket is full; needs approval.
6. **Trace on LB** with 112 experts/chip — trace region / L1 small sizing (Phase 0).
7. **S6 eager-vs-trace** is PCC, not bitwise (different KDA bounds programs); bitwise only replay-vs-replay.

## 10. Weight dict keys (reference for implementers)

Top level: `embed_weight [163840,7168]` (first rank), `norm_weight [7168]`, `layers` (list per local
layer), `attn_res_weights` (flat checkpoint-key mapping), `attn_res_prefix` (default
`"language_model.model."`).

Per layer: `attn_norm_weight`, `ffn_norm_weight` `[7168]`;
- `mla_weights`: `q_a_proj.weight [1536,7168]`, `q_a_layernorm.weight [1536]`,
  `q_b_proj.weight [18432,1536]`, `kv_a_proj_with_mqa.weight [576,7168]`,
  `kv_a_layernorm.weight [512]`, `kv_b_proj.weight [24576,512]`, `o_proj.weight [7168,12288]`,
  `g_proj.weight [12288,7168]`;
- `kda_weights`: `{q,k,v}_proj.weight [12288,7168]`, `{q,k,v}_conv1d.weight [12288,1,4]`,
  `A_log [1,1,96,1]`, `f_a_proj.weight [128,7168]`, `f_b_proj.weight [12288,128]`, `dt_bias [12288]`,
  `b_proj.weight [96,7168]`, `o_norm.weight [128]`, `o_proj.weight [7168,12288]`,
  `g_proj.weight [12288,7168]`;
- L0 `ffn_weights`: `gate_proj, up_proj [33792,7168]`, `down_proj [7168,33792]`;
- MoE: `gate_weights {weight [896,7168], e_score_correction_bias [896]}`,
  `shared_expert_weights {gate_proj, up_proj [6144,7168], down_proj [7168,6144]}`,
  `latent_weights {down_proj [3584,7168], up_proj [7168,3584], norm [3584]}`,
  `routed_expert_weights`: list of 896 `{gate_proj, up_proj [3072,3584], down_proj [3584,3072]}`.
