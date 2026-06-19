# SDPA Precision Matrix Results

Authoritative precision characterization for `scaled_dot_product_attention`
across the R1 numeric surface. Produced by
`test_scaled_dot_product_attention_numerical.py::test_scaled_dot_product_attention_precision_matrix`.

- **Last run**: 2026-06-19 (Refinement 1)
- **Device**: Blackhole
- **Distribution**: `randn` (the registry-matrix distribution). Adversarial
  uniform / negative distributions are characterized separately in
  `eval/golden_tests/scaled_dot_product_attention/test_regression.py` — they
  expose a known, out-of-R1-scope bf16-accumulator limitation (running softmax
  stats stored in bf16 CBs; fp32 CBs hang this LLK, Issue #13364).
- **Defaults reproduced when `compute_kernel_config=None`**: HiFi2,
  `fp32_dest_acc_en=True`, `math_approx_mode=False`, `dst_full_sync_en=False`.
- **Assertion**: permissive PCC ≥ 0.95 floor (catastrophe guard only — the
  golden suite is the tight gate). All 90 cases passed.

## Worst-case PCC per (dtype × fidelity), across all shapes / fp32_acc

| dtype | HiFi4 | HiFi2 | LoFi |
|-------|-------|-------|------|
| bfloat16  | 0.99992 | 0.99992 | 0.99896 |
| float32   | 0.99991 | 0.99989 | 0.99887 |
| bfloat8_b | 0.99981 | 0.99980 | 0.99905 |

(min over shapes ∈ {1×1×32×32, 1×1×128×64, 1×4×256×64, 1×8×512×64,
1×2×128×128} and fp32_acc ∈ {True, False}.)

## Observations

- **fp32 input is not more accurate than bf16 input** at matched fidelity. SDPA
  unpacks every input through srcA/srcB (→ TF32, 10-bit mantissa) for the QKᵀ
  and PV matmuls, so the fp32 L1 storage buys nothing past TF32 — exactly the
  production SDPA behavior, and why the intermediate CBs stay bf16. The win from
  fp32 input is range/contract conformance, not precision.
- **`math_fidelity` modulates precision monotonically** HiFi4 ≳ HiFi2 ≫ LoFi —
  confirming `compute_kernel_config` is wired through. HiFi4/HiFi2 are
  near-indistinguishable here (matmul dims are small); LoFi drops rel-RMS to
  ~6–8% (PCC still ≥ 0.9987).
- **`fp32_dest_acc_en=False` costs a little precision** (PCC drops ~5e-5 to
  ~1e-4, rel-RMS ~+0.5%) but never breaks — the default stays `True`.
- **bfloat8_b is solid on randn**: PCC ≥ 0.9998 at HiFi2/HiFi4 across all shapes,
  rel-RMS ~1.7–2.2%. The block-float additive `−inf` causal mask also passes
  (see `test_dtype_mask_scale`), so **no EXCLUSIONS were needed** for the
  tile-aligned bf8b cells.

## Skipped combinations

None. The full matrix (5 shapes × 3 dtypes × 3 fidelities × 2 fp32_acc × randn)
ran clean — 90/90 passed. The `bfloat8_b × non_tile_aligned` cell that the R1
plan flagged as an EXCLUSIONS candidate is not reachable yet (alignment is still
SUPPORTED=[tile_aligned]); it will be re-evaluated under Refinement 3.
