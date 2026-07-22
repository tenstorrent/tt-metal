# LTX/FLUX Mt<=8 deep-kernel optimization — final report

Branch `cglagovich/regime-a-ltxflux-opt` (off single-chip head 323ae42b161; AGMM excluded). Two phases.

## Phase 1 — picker (banked, on the M-scaling LTX/FLUX shapes): 4 wins, regression-clean
Root cause: cost-model fallback over-splits K. Fix = per-shape lookup-table entries (shallower Pk + larger kb):
- 64x6144x1536 (1,12,1,2,1)->(1,3,1,8,2): -3.0%
- 64x4608x6144 (1,6,1,1,8)->(2,3,1,2,3): -2.8%
- 32x6144x2304 (1,4,1,2,9)->(1,3,1,4,5): -2.1%
- 64x15360x1536 (1,12,1,1,3)->(1,6,1,2,3): -2.2%
All config=None, PCC 1.0000, 0 corpus regression.

## Phase 2 — deep kernel (this goal's 9-shape corpus): all closed with causal evidence, corpus at limit
Tooling: DIAG_ZONES (compile-gated DeviceZoneScopedN; compute in0/in1-wait, writer ring + reduction sub-zones,
reader in1-read) + zone_parse.py + ltxflux_corpus_decomp.py. Perturbation +0.6-2.1%; mask-0 byte-identical
(171 unit+corpus tests; 0 corpus perf drift).

Full decomposition table + per-shape closures: LTX_FLUX_OPT_LOG.md. Summary:
- **Deep-K (32x6144x1536/2304/4608/6144):** in1-wait dominant, 476-498 GB/s at the ~500 SP1 read ceiling =>
  DRAM-read-bound floor. Root RECVWAIT (80-91% of wall) OVERLAPS the read (NO_REDUCE 0.7-4.6%) — the
  RECVWAIT-fraction is a red herring.
- **Shallow low-AI (32x256x6144 Kt8, 32x2048x512/1536/2048):** in1-wait dominant, below ceiling for geometry
  (short reads / tiny shapes); Pk<=2 reduction minimal. Overhead/geometry floor.
- **256x2048x1024 (worst, w/id 1.99):** sole in0-ring-exposed shape. Wall = matmul 9us (floor) + in0-ring
  4.9us exposed (FORECLOSED: scatter/exchange/repl/chunk/direct-read all rejected here) + reduction 3.5us
  exposed. Reduction tree = only non-foreclosed lever (~5%/1.2us) but DEFERRED (negligible absolute + negative
  config signal [shorter-chain Pk2 same-cores +29%] + rejected direction + high multi-child-rewrite risk;
  design documented for future higher-Pk use).

**Conclusion:** the corpus is at its practical limit under the current (heavily-optimized) kernel; no
lossless lever with favorable risk/reward remains. Every shape improved (phase 1) or closed with fine-grained
causal evidence (phase 2). Deferred lever (reduction tree) fully documented with numerical ceiling + design.
