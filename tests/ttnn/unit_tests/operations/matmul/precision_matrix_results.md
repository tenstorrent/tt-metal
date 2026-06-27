# matmul — precision matrix results

Source test: `test_matmul_precision_matrix.py` (Refinement 1 — numerical configurability).
Reference: `torch.matmul` in fp32. Metrics in fp64. **Last run: 2026-06-27** —
**120 passed, 24 skipped** (the `{dtype=float32, fp32_dest_acc_en=False}` EXCLUSION).
Device: wormhole_b0.

Assert is on **PCC** only; relRMS / max_abs / median_abs printed for observability.
PCC floors (matrix context): fp32 ≥ 0.999, bf16 ≥ 0.99, bf8b ≥ 0.98 (keyed on the
**effective** = coarser of activation/weight dtype).

## Key findings

- **HiFi4 ≡ HiFi2 for bf16/bf8b inputs.** Every low-precision row is bit-identical
  across the two fidelities — the op clamps HiFi4→HiFi2 for bf16/bf8b inputs
  (issue #38306 guard) and bf16's ≤7 mantissa bits gain nothing from HiFi4. So the
  clamp costs zero precision. Only fp32 inputs actually use HiFi4.
- **fp32 keeps HiFi4 and benefits**: fp32/fp32 @128×256×512 — HiFi4 PCC 0.99999980
  / relRMS 0.0013 vs HiFi2 PCC 0.99999379 / relRMS 0.0073.
- **fp32_dest_acc_en=True is materially better at depth.** At K=4096 the acc=True
  rows sit at relRMS ~0.01; the acc=False (16-bit DEST) rows climb to relRMS ~0.07
  (bf16) — the ~O(√K) accumulation rounding. PCC stays ≥ 0.999 throughout (PCC is
  insensitive to the magnitude error that RMS captures).
- **bf8b output + acc=False uses hardware packer-L1 accumulation** (Lever B) so the
  K-sum is held in a bf16 interm instead of re-quantizing to bf8b each K-block —
  this is what keeps bf8b acc=False relRMS ~0.018–0.02 instead of blowing up at depth.

## Representative table (HiFi4 column; bf16/bf8b identical under HiFi2)

| shape (A@B) | act | wt | eff | fp32_acc | PCC | relRMS |
|---|---|---|---|:--:|---|---|
| 128×256×512 | fp32 | fp32 | fp32 | True  | 0.99999980 | 0.00134 |
| 128×256×512 | fp32 | bf16 | bf16 | True  | 0.99999380 | 0.00671 |
| 128×256×512 | fp32 | bf8b | bf8b | True  | 0.99996501 | 0.01012 |
| 128×256×512 | bf16 | fp32 | bf16 | True  | 0.99999412 | 0.00498 |
| 128×256×512 | bf16 | bf16 | bf16 | True  | 0.99999431 | 0.00444 |
| 128×256×512 | bf16 | bf16 | bf16 | False | 0.99996652 | 0.00867 |
| 128×256×512 | bf16 | bf8b | bf8b | True  | 0.99996568 | 0.00875 |
| 128×256×512 | bf8b | bf8b | bf8b | True  | 0.99991209 | 0.01351 |
| 128×256×512 | bf8b | bf8b | bf8b | False | 0.99988532 | 0.01844 |
| 128×256×512 | bf8b | fp32 | bf8b | False | 0.99991138 | 0.01570 |
| 256×4096×512 (deep K) | bf16 | bf16 | bf16 | True  | 0.99998869 | 0.00943 |
| 256×4096×512 (deep K) | bf16 | bf16 | bf16 | False | 0.99901225 | 0.06960 |
| 256×4096×512 (deep K) | bf8b | bf8b | bf8b | True  | 0.99990712 | 0.01389 |
| 256×4096×512 (deep K) | bf8b | bf8b | bf8b | False | 0.99986197 | 0.02049 |
| 256×4096×512 (deep K) | fp32 | fp32 | fp32 | True  | 0.99999416 | 0.00728 |

(Full 120-row trace is printed by the test under `-s`.)

## Skipped / known-limit combinations

- **`{dtype=float32, fp32_dest_acc_en=False}` (24 cases)** — op-side EXCLUSION
  (maxed input demands a maxed accumulator). `pytest.skip`.
- **bf16-OUTPUT + acc=False at K≥8192** — NOT in this matrix's shapes (deepest here
  is K=4096, which passes). The golden suite shows `A256x8192` bf16/bf16 and
  bf16/fp32 acc=False at relRMS 0.125–0.128 vs the golden band 0.10 — the
  fundamental 16-bit-DEST floor (~O(√K)), fixable only by `fp32_dest_acc_en=True`.
  Tracked as Refinement 1b (see `op_requirements.md` / `changelog.md`). PCC there is
  still ~0.997, so a PCC-only assert would pass; the miss is purely the RMS band.
