# REVISION-4 — silicon validation + attribution + system-context pass (2026-08-17)

Owner-directed ("execute using safe pytest + validate wins + update the paper").
All measurements fresh on the final landed tree, flock-serialized, per-cell
subprocess Tracy, correctness before timing. Evidence: §8 rows B8–B10 in
`evidence/paper/evidence.md`; raw dirs `generated/canonical_sweep/glmval_*`.

## Measurements

1. **Anchor replication (B8)**: scenario DSA cell 712.2 µs op / 893.9 µs routed
   (printed 712.3/895.3 — −0.01%/−0.16%); competition op 33.96 µs @32c
   (printed 34.3); proxy 356.43 µs/row (printed 357.0). Every printed anchor
   reproduces on today's tree.
2. **SFPLOADMACRO attribution A/B (B9)**: `TT_METAL_DISABLE_SFPLOADMACRO=1`
   costs the op 33.96→35.19 µs and the proxy 356.43→363.35 µs → the micro-op
   win is banked in both printed anchors at **3.6% (op) / 1.9% (proxy)**,
   correcting the header-derived "worth 5–9%" claim; the row-parallel arm
   inherits it via the shared header (porting headroom 0); true as-shipped
   per-row ≈ 363 µs (Tab. II's proxy slightly flatters the incumbent).
3. **System context for the DSA rows (B10)**: top-k ≈ 0.3% of GLM-5.2
   end-to-end prefill (external baseline ~2.6 s/chunk, GH #51331); one TP
   collective (~1 ms/layer/chunk, GH #47803) outweighs the whole cell. The
   integration lever (regather/re-partition inverse-pair elision for
   thin-head-shard variants) landed in the model repo (fe1930d50c2), pending
   8×4 mesh validation.

## Text changes (page-neutral net)

- §6-B arms: "worth 5–9%" → measured "3.6% by a disable-flag A/B"; proxy
  no longer claimed "byte-identical" — it inherits the shared-header micro-op
  paths (1.9%).
- §6-E scenarios: 4-line system-context addition (0.3% share; collective
  outweighs the cell; inverse-pair elision as the live lever) + evidence
  comment (GH #51331/#47803, commit fe1930d50c2).
- §2.3 GPU recap compressed per BUDGET overflow rule 1 (all cites kept) to pay
  for the additions. Body ends on page 10 exactly; 0 errors, 0 overfull.
