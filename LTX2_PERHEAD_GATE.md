# LTX-2 Per-Head Gate Precision Investigation

## Goal
Achieve exact match (PCC > 0.999 per layer) between TTNN LTX-2 transformer and the official LTX-2.3 22B reference pipeline, including per-head attention gating.

## Decisions
| Decision | Reason | Rejected Alternative |
|----------|--------|----------------------|
| Gate logits computed on HOST (torch fp32) | TT bf16 matmul over K=4096 with near-zero gate weights produces accumulation errors | On-device `to_gate_logits` Linear in fp32 — would work but gate weights are tiny (32×4096), host F.linear is negligible latency |
| Gate multiply on device in BHNE space | Mathematically equivalent to reference (B,T,H,D) multiply; avoids reshape overhead | Apply gate after to_out — requires unfusing addcmul; apply gate on host — same PCC since SDPA output is the bottleneck |
| `to_gate_logits` weights NOT loaded to device | Saves device memory; gate computed on host only | Keep device copy for potential future on-device gate compute |

## Constraints & Workarounds
- Hardware: Wormhole B0, 8-chip loudbox (tested on 1×1 mesh for PCC validation)
- Per-layer PCC with gates: **0.999954** (with correctly gated reference model)
- **RESOLVED**: The 0.989 PCC was a TEST BUG — the reference `LTXModel` was constructed with `apply_gated_attention=False` (default), so it computed ungated attention while our TT model computed gated attention. They were mathematically different operations. Passing `apply_gated_attention=True` to the reference model gives PCC 0.999954.

## Surprises & Discoveries
- **ROOT CAUSE of 0.989 PCC**: Test bug! `LTXModel(apply_gated_attention=False)` is the default, so the reference model had no gates while our TT model applied gates. Fix: pass `apply_gated_attention=True`.
- Gate values for this checkpoint are highly non-trivial: range [0.04, 1.98] for attn1, [0.06, 1.88] for attn2. Not near-identity.
- `2*sigmoid(0) = 1.0` means zero-initialized gate weights produce identity gates
- PyTorch bf16 × bf16 → bf16 (stays bf16, no implicit promotion)
- The reference applies gate BEFORE `to_out` (not after): `sdpa → concat_heads → view(B,T,H,D) * gates → view(B,T,H*D) → to_out`

## Open Questions
- [ ] Full 48-layer end-to-end video generation matching official pipeline output
- [ ] Performance impact of host-side gate computation (device→host readback per attention layer)
- [ ] Is there a way to batch the host gate computation to hide latency (async readback)?

## State
- [x] Identified gate weights in LTX-2.3 checkpoint (`to_gate_logits.weight/bias` per attention layer)
- [x] Added `query_input_dim` and `output_dim` params to LTXAttention for cross-attention flexibility
- [x] Implemented host-side gate computation (F.linear on host, push bf16 to device)
- [x] Implemented on-device gate multiply in BHNE space (before concatenate_heads)
- [x] Verified gate values are correct (PCC 1.0 vs reference gate computation)
- [x] Verified on-device broadcast multiply precision (PCC 0.999991 in isolation)
- [x] Tested HiFi4 for to_out — no improvement
- [x] Tested full host-side gate+to_out — no improvement (same 0.989)
- [x] Confirmed root cause: SDPA precision × gate amplification
- [ ] Try `fp32_dest_acc_en=True` on SDPA compute kernel config
- [ ] Full 48-layer end-to-end PCC measurement
- [ ] Video quality assessment with gates enabled vs disabled

## Key Measurements

### Per-layer PCC (1 transformer block, 1×1 mesh, N=192)
| Configuration | PCC | Notes |
|---|---|---|
| Gated ref + gated TT (correct comparison) | **0.999954** | `apply_gated_attention=True` on reference |
| Ungated ref + gated TT (WRONG comparison) | 0.989 | Bug: reference had no gates! |
| Zeroed gate weights (identity gates) | 0.999973 | Both models effectively ungated |

### Gate value statistics (checkpoint: ltx-2.3-22b-dev)
| Attention | Mean | Std | Range |
|---|---|---|---|
| attn1 (self) | 1.0037 | 0.2297 | [0.0445, 1.9753] |
| attn2 (cross) | 1.0031 | 0.3890 | [0.0612, 1.8809] |
