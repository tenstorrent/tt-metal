# Round 2 proposals (G2-r2: automatic per skill §7; none is an exception: all are model-local Python / config changes)

Round 1 took paper-accuracy from 1056.8 to 541.0 ms (−48.8 %), so the diminishing-returns exit does not apply. Evidence is in analysis.md.

| id | candidate | expected | risk | measurement |
|---|---|---|---|---|
| R2-1 | Group attention without permutes: qkv viewed as (B, 1, G·S_pad, 3d) on the time-attention path (heads, q/k norm, SDPA) with an additive same-slot + pad mask (R1-10 revisited) | multivariate −550 to −650 ms, paper 0 | medium (mask size, SDPA at 87 % masked) | Micro-benchmark one group layer first (≤ 40 ms, PCC ≥ current). Tests: V = 8 masked, PAD, G = 4. D2 guard-only rule: multivariate ≥ 10 % better, target within ±1 % |
| R2-2 | ff1 fused SwiGLU block / subblock sweep at M = 98304 (now 42.9 %); re-sweep QKV / wo against the current tree | −25 to −40 ms | low | One sweep job (sweep_swiglu.py) |
| R2-3 | q and k through one RMSNorm and one RoPE (concatenated heads), then fuse the head split (minimal_matmul_split / nlp_create_qkv_heads variants) | −40 to −90 ms | medium | Reduced bench and the module PCC tests (q/k after QK-norm + RoPE) |
| R2-4 | Remove nlp_create_qkv_heads by writing q / k / v directly from the QKV matmul (minimal_matmul_split), if its head layout fits | about −30 ms (overlaps R2-3) | medium | as R2-3 |
| R2-5 | Embed fidelity HiFi4 -> HiFi2 on the K = 96 matmuls only | −2 to −3 ms | low; long_context drifted at HiFi2 in job 417, so likely rejected | bench long_context |
| R2-6 | Pack rows 66 -> fewer padded tiles (block-diagonal masks) | up to −100 to −150 ms | high | Matmul-only micro-benchmark first; experiment only |

Order: R2-1, R2-2, R2-3, R2-4, then R2-5 and R2-6 if the round has not met the exit. Host prepare (R1-9) is not proposed: the profile shows no hot spot (candidates.md).
