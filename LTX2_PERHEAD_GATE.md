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
- Per-layer PCC with gates: **0.989** (vs 0.9994 without gates / zeroed weights)
- Per-layer PCC without gates: **0.9994** (baseline — inherent SDPA + matmul precision)
- **Root cause of 0.989**: The TT SDPA output differs from reference SDPA by PCC 0.9994. When per-head gates (range [0.04, 1.98]) are applied, the gate-weighted error propagates through the `to_out` matmul differently than unweighted error. This amplifies the SDPA precision gap. Even computing gate+to_out entirely on HOST in bf16 gives the SAME 0.989 PCC — confirming the bottleneck is SDPA output precision, not gate/to_out computation.
- Permanent fix: Improve TT SDPA precision (fp32 accumulation, HiFi4, or larger chunk sizes)
- **Workaround**: Currently using on-device gate multiply with host-computed gate values. The 0.989 PCC per layer compounds over 48 layers. Video quality may degrade at high step counts.

## Surprises & Discoveries
- HiFi4 for `to_out` matmul did NOT improve PCC (0.989 → 0.9896) — the fused `dit_minimal_matmul_addcmul_fused` kernel may not respect `compute_kernel_config` for math fidelity
- Host-side gate+to_out computation in bf16 gives IDENTICAL PCC (0.989640) to device computation — proves the error is in the SDPA output, not the gate or to_out path
- Gate values for this checkpoint are highly non-trivial: range [0.04, 1.98] for attn1, [0.06, 1.88] for attn2. Not near-identity.
- `2*sigmoid(0) = 1.0` means zero-initialized gate weights produce identity gates (PCC 0.9994 = baseline)
- PyTorch bf16 × bf16 → bf16 (stays bf16, no implicit promotion)
- The reference applies gate BEFORE `to_out` (not after): `sdpa → concat_heads → view(B,T,H,D) * gates → view(B,T,H*D) → to_out`

## Open Questions
- [ ] Can SDPA precision be improved with `fp32_dest_acc_en=True` or HiFi4 without OOM?
- [ ] Does the 0.989 per-layer PCC produce acceptable video at 20+ diffusion steps?
- [ ] Would fp32 SDPA accumulation (`fp32_dest_acc_en=True` on `sdpa_compute_kernel_config`) help? Currently `fp32_dest_acc_en=False`.
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
| Configuration | PCC | Command |
|---|---|---|
| Zeroed gate weights (identity gates) | 0.999973 | Test with `filt[k] = torch.zeros_like(filt[k])` for `to_gate_logits` keys |
| Real gate weights, on-device multiply | 0.989 | Default attention_ltx.py |
| Real gate weights, host gate+to_out (bf16) | 0.989640 | Host path in attention forward |
| Real gate weights, HiFi4 to_out | 0.989636 | `hifi4_mm_config` for to_out |
| Isolated broadcast multiply (1,32,192,1)×(1,32,192,128) | 0.999991 | Direct ttnn.multiply test |

### Gate value statistics (checkpoint: ltx-2.3-22b-dev)
| Attention | Mean | Std | Range |
|---|---|---|---|
| attn1 (self) | 1.0037 | 0.2297 | [0.0445, 1.9753] |
| attn2 (cross) | 1.0031 | 0.3890 | [0.0612, 1.8809] |
