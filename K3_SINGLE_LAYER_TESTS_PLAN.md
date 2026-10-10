# Kimi-K3 single-layer transformer tests (LB + Galaxy) — plan v2

Status: v2, revised after independent reviews by Claude Code (Opus 5.5) and Codex Astra (gpt-6-astra)
of v1 (summary of findings and resolutions in §10). Owner: pjosipovic.

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
`is_last_rank=True`. Blockers (all refs verified on the base commit by both reviewers):

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
  consume it (keys in §10.3): `attn_norm_weight`, `ffn_norm_weight` (`1 + 0.02·randn`, never ones —
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

**`reference.py`** — single-layer K3 reference from component references
- AttnRes: `reference/kimi_k3/attn_res/attn_res.py` (`AttnResStream`, `attn_res_layer`,
  `fold_query`); set `stream.block_residual = sealed` (device `[1,S,N,d]` → ref `[N,S,d]`), pass the
  global L; `q_out` + RMSNorm for `L92full`.
- KDA: `reference/kda/layer.py::kda_forward_reference(hidden, weights, cfg, state)` with explicit carry
  in/out. The existing cache `tests/kda/reference_cache.py` is hard-wired to `KimiK3TestCase` /
  `KIMI_K3_FIRST_KDA_LAYER`, takes no initial state and writes under `ttnn.CONFIG.model_cache_path`
  → generalize it (key: content hash of weights + inputs + initial state + config + source
  fingerprint) or add a sibling cache for this suite.
- MLA: `utils/chunked_prefill_utils.py::cpu_mla_reference` as a **full-prefix** oracle (run on the
  whole request, compare the chunk's slice). Do **not** use `reference/mla_reference.py::MLAReference.forward`
  incrementally (causal mask not offset for a prefix, `:188`). Its disk cache is disabled under
  `CI=true` and defaults to `/tmp` (`chunked_prefill_utils.py:29-30,159`) — acceptable, but budget
  the time.
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

Design constraints from review:
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

## 6. CI wiring (Phase 4)

**Budget first** (read-only aggregation of the current matrices by Codex review):

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
| 0 Feasibility gate (2 d) | Cold measurements on LB and GH for `L12` and `L87`, `weight_cache_path=None`: MoE weight build (options b/c prototype), upload, peak host RAM, peak device DRAM/L1, disk temp; KDA / MLA / MoE reference time at 640 and 320 rows/chip, 1 and 2 chunks; trace capture fits `trace_region_size` for one MoE layer on LB (112 experts/chip) | table in this file; MoE option chosen; per-case cold runtime estimate for S1–S9; go/no-go on GH row count; CI budget request drafted |
| 1 Unaligned entry (2–3 d) | §5.1 code; extended host test; device split-walk test | host test green for all F/n; split-walk within PCC bound, copied planes bitwise; `tests/attn_res/` (LB) and `test_transformer_pipeline_split.py` (GH) green; aligned-pipeline numerics unchanged or documented |
| 2 Building blocks (3–4 d) | §5.2 modules; MoE option from Phase 0 incl. `TtRoutedExpert` hook; pooled-reference CPU unit test; seedable KDA `random_weights`; reference cache | S1 green for `L0`, `L12`, `L36` on LB |
| 3 Scenarios (3–4 d) | S1–S9 for P0 cases on LB, then GH; P1 cases | all P0 green on both boxes; thresholds from measurements with margin; negative controls fail as expected; runtime per case recorded |
| 4 CI (1–2 d) | budget approval; LB + GH entries; non-skip guards; retire old LB test after S8; GH `two_iters` switch | dispatched runs green: `blackhole-e2e-tests.yaml` (`model=kimi_k3`, LoudBox) and `blaze-models-prefill-tests.yaml` (`k3_single_layer`) |

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
3. **GH reference time** at 5120 tokens/chunk (KDA per-token Python loop, MLA host attention, MoE
   GEMMs) — Phase 0; fallback 320 rows/chip for reference-backed scenarios.
4. **Synthetic ≠ real activations** — mitigated by calibrated AttnRes entropy, heavy-tailed rows,
   branch-delta checks; real-weight coverage remains on GH depth tests.
5. **CI budget** — `bh_sc1` demo bucket is full; needs approval.
6. **Trace on LB** with 112 experts/chip — trace region / L1 small sizing (Phase 0).
7. **S6 eager-vs-trace** is PCC, not bitwise (different KDA bounds programs); bitwise only replay-vs-replay.

## 10. Review log (v1 → v2)

Reviewers: Claude Code (Opus 5.5, independent instance) and Codex Astra; both read the base commit,
neither ran devices. Both: direction sound; head-group and ceil formulas correct; request changes.

| Finding | Source | Resolution |
|---|---|---|
| Fold also runs at split/pipeline entry over the inherited set (S≥4), and on the dot reduce at R=1; v1 said mid-block never folds | both | §2 rewritten; `L84` added (dot fold + max-width seal); L87 noted as entry-fold |
| Production last rank is `kv_only_last_layer`; v1's L92 tested the non-serving tail | Claude | `L92kv` P0, `L92full` P1 |
| 91→92 adjacency / distinct local KV slots untested | Codex | `T91_92kv` P1 |
| Split vs unsplit not bitwise (handoff flush, R=1 dot fold) | both | PCC + bitwise only for copied planes + fp32 reference arm |
| Floor plane formulas also in runtime/adapter/test assert | both | fix all four; runtime/adapter assert aligned |
| Host site-order test already exists | Claude | extend `test_attn_res_pipeline_split.py` |
| Walk should own global counter; don't change `block.py:202` | Claude | adopted |
| `q_out` drop changes R on aligned non-last ranks | Codex | gated by GH pipeline-split test; fallback keep unread `q_out` |
| Pooled + small scales indistinct in bf16/bf4; `Q(sW)≠sQ(W)`; latent norm hides scale | both | exact bf4-commuting transforms (sign/pow2/row perm); reference on dequantized weights; validation with experts > pools; negative control |
| Pooling doesn't cut conversion/upload/tensorbin/reference cost | both | `weight_cache_path=None`; Phase 0 picks device-side (c) or packed (b) expansion |
| Gate top-k tie flips vs reference | Claude | reference driven by device-selected experts + flip-rate bound |
| AttnRes synthetic softmax may be ~one-hot | Claude | entropy-calibrated query scale, asserted |
| `kda_states.read(0)` wrong — global index | both | `read(L, slot)` |
| Slabs not bound; old LB test has slab/import/address/two-capacity checks | both | bind slabs; S8 ports them; keep old test until then |
| `random_weights` unseeded; KDA ref cache not reusable; MLA ref cache off in CI; `MLAReference` prefix mask | both | §5.2 adjusted |
| Production block config: `routing_use_l1_small_for_semaphores=True` | Claude | harness |
| `max_seq_len` ≥ 2 chunks; GH ≥ 10240 > hf default 8192 | both | harness + config override |
| `*_device_params` keyword-only; module-scoped fixture impossible on function-scoped mesh | both | §5.2 |
| S3 rebuild cost; stronger oracle | both | B→A→B + continue + short restarts |
| S6 eager vs trace different programs | Claude | PCC 0.999 + bitwise replay |
| Eager multi-slot untested | Claude | S7 |
| Rotated starts, traced unaligned/short metadata | Codex | S2 starts, S6 replays S4 metadata |
| Overflow check must be concrete | Codex | S9(c) via eager diagnostic forward |
| Whole-layer threshold can hide a broken branch; KDA 0.9995 ≠ layer bar | Codex | branch deltas + rel-L2; component bars cited |
| CI budget: `bh_sc1` demo 0 min free, `bh_loudbox` e2e 18 min | Codex | §6 budget first; separate GH entry |
| `|| exit $?` not literally required | both | §6 wording |
| Sealed planes are block-local, not cumulative | Codex | §2, §5.2 |
| `two_iters` GH switch valid but same-prompt | Codex | kept as complement |

### 10.3 Weight dict keys (reference for implementers)

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

## 11. Handoff checklist (moving to another machine)

- Branch: `pjosipovic/k3-single-layer-tests` from `origin/pjosipovic/kda-device-request-initialization`
  (`5aa5dee81f7`); this file at repo root as `K3_SINGLE_LAYER_TESTS_PLAN.md`, removed before the PR.
- Needs: LB (8×P150) for Phases 0–3; Galaxy SC1 via Blaze dispatch for GH.
- `./build_metal.sh --release`; device tests only via `scripts/run_safe_pytest.sh`.
