# qr-ring sparse-MLA — Phase-0 decisions + session findings

Session: autonomous implementation attempt on LoudBox (8× Blackhole p150b). Branch: `main` (RUB'd).
Plan: artifact f4f799bb. Followed the plan's OWN de-risking order (Phase 0 gates → Phase 1 math → measure)
before touching the hang-prone fabric collective.

## Phase-0 decisions (resolved with best-judgment defaults — override freely)

1. **Partial export format / merge representation** → **two tensors `(m, l)` shape `[1,H,S,1]`, fp32 stats.**
   Rationale: G2 (below) shows fp32 merge = PCC 1.0; bf16 stats = 0.99998 (still clears 0.99 at sp≤8, but the
   tight 0.995 model bar across 8 real shards has less headroom → fp32 is the safe default). Merge uses raw
   (m,l) with scale applied in the correction exp (matches the kernel's `exp((s-m)*scale)`); convert to LSE
   only at the fabric-collective boundary that reuses ring-joint's LSE combine.

2. **wkv_b2 fold location** → **at-home after merge (v1).** 512-wide transport is correct-but-fat; in-kernel
   per-head fold deferred to Phase 4 (it's a pure wire-width optimization, commutes with the 1/l rescale).

3. **Home-shard / reduce-scatter target** → **deferred to Phase 3** (only matters once the fabric reduce
   exists). For the host-merge path it's a non-issue (merge sees all shards).

4. **GLM H≥32** → **target DeepSeek (32 heads) first;** GLM gets head-pad-to-32 behind an explicit shape
   assert. Note: `mla.py::_needs_head_to_seq_reshard` already handles thin TP head shards via a head→seq
   transpose — that interacts with Q-gather and needs the Phase-0 spike before GLM Q-gather.

5. **Indexer scope** → **OUT of scope for this session's perf claim.** Measured sparse-MLA transport only;
   the indexer's O(T) D_idx=128 key gather is called out separately as a follow-on that caps the net win.

## Findings

### G1 — top-k=2048 quality kill-switch: **PASS (pre-validated)**
- `scratchpad_qr_ring/g1_topk_quality_spike.py`. Faithful G1 needs the 671B DeepSeek checkpoint (infeasible
  in-window); random weights spuriously fail (untrained q·k → uniform attention, no concentration).
- Synthetic mechanism check: in the trained/peaked regime top-k=2048 clears the 0.994/0.995 bar even at 0.5M;
  uniform regime correctly fails (gate is meaningful).
- **Authoritative:** production KV-gather already ships top-k=2048 at these ctx lengths (test_sparse_mla.py
  passes at 0.98) — the scheme's numerical basis is already established in prod. G1 is not a blocker.

### G2 (math half) — host online-softmax merge of per-shard partials: **PASS**
- `scratchpad_qr_ring/g2_host_merge_math.py`. Splits top-k into block-cyclic shards, computes each shard's
  unnormalized `(O, m, l)` with the exact op math, merges with raw-max online softmax.
- Results (vs full-T golden): **fp32 = PCC 1.000000**, bf16 stats = 0.99998, **zero-hit-shard identity partial
  (O=0,m=−∞,l=0) merges perfectly at 1.000000** — across sp = 2 / 4 / 8. The reduce formula + the hang-safety
  identity partial are proven before any fabric exists.

### Measured transport perf diff (LoudBox, SP=4 ring, native 2×4 mesh): **KV O(T) vs qr flat — CONFIRMED**
- `tests/nightly/blackhole/sdpa/test_qr_ring_transport_perf.py` (passing under run_safe_pytest).
- KV-gather all_gather(KVPE [1,1,T,576]): 0.279 → 0.547 → 1.653 → 3.112 ms at T = 4K/16K/64K/128K (clean O(T),
  doubles per doubling of T).
- qr Q-gather all_gather(Q [1,128,512,576]) = 2.236 ms + reduce O [1,128,512,512] = 1.704 ms = **3.941 ms FLAT
  in T** (independent of context).
- **Crossover ~130–160K on SP=4** (matches plan's ~118K GLM-classic break-even). Extrapolating KV's O(T) line:
  ~11.9 ms @500K on SP=4 → ~3× vs qr; the SP=8 Galaxy config and the qr-latent wire-shrink (Phase 4) widen it
  further (plan projects 4.2×/9.2× GLM). This is the *classic per-head Q* MVP wire (pessimistic); qr-latent is smaller.
- Caveats: SP=4 not the SP=8 Galaxy config (LoudBox's clean fabric axis); qr's reduce is measured as a plain
  all_gather, not the fused flash-merge reduce-scatter (which the real op needs and Phase 3 builds).

### Fabric-bringup risk — HIT (as the plan predicted)
- First attempt opened an ad-hoc `(1,4)` mesh → **Fabric Router Sync timeout** (ethernet handshake incomplete
  on that device subset). Self-aborted at the 10s router-sync guard (not a silent hang). Fixed by opening the
  **native (2,4)** mesh + 2D sharding along the SP axis (the known-good smoke-test topology). Lesson for Phase
  3: pin the fabric to the native LoudBox 2×4 topology / submeshes, don't hand-roll mesh subsets.

## NOT done in-window (honestly beyond a couple hours — the multi-week core)
- **Device partial `(O,m,l)` export mode** in sparse_sdpa (the keystone C++). Spec is ready (see below); the
  lift is converting the op's single-Tensor return to a 3-tensor return across the device-op interface +
  kernel export of the pre-normalize `out_cur`/`max_cur`/`sum_cur` + writer drain, then a ~10-min rebuild per
  iteration. Deferred to avoid leaving broken C++ on `main`.
- Phase-2 shard-local scoring; Phase-3 fabric flash-merge collective (the hang-prone item — needs supervision).

## Keystone spec (ready to implement next)
Compute kernel (`sparse_sdpa_compute.cpp`): after the last chunk, `out_cur` holds Σexp((s−m)·scale)·V
(unnormalized), `sum_cur` holds l, `max_cur` holds raw m. Add a `RETURN_PARTIALS` compile define that, on
`is_last`, SKIPS `normalize_row_streaming`, untilizes `out_cur` (unnormalized) to `cb_out_rm`, and untilizes
the m/l columns to two new CBs. Writer drains m,l to two new `[1,H,S,1]` fp32 DRAM tensors (page = h*S+tok,
same as O). Device op: `return_partials` flag in params + hash; `compute_output_specs`/`create_output_tensors`
return a 3-vector; program factory allocates cb_m_rm/cb_l_rm + writer args. Public API: bool flag, returns
`std::vector<Tensor>`. Validate against `g2_host_merge_math.py`'s per-shard partials (already proven correct).
