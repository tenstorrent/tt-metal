# Track 2 — MoE: router, experts (EP=32, 8 per chip), dispatch/combine

Branch/worktree: `ssinghal/mimo-moe` / `/mnt/tt-data/ssinghal/mimo/wt/moe`. Host: .33 only. Skills: `tt-moe-correctness`, `tt-lab-ops`.
Package: `models/demos/blackhole/mimo_v26/tt/moe*.py`, `router*.py`, tests in `models/demos/blackhole/mimo_v26/tests/`. Must not touch `reference/` or `tt/attention*`.
IMPORTANT: this clone's main lacks the DSV4 package and the one-copy ring MoE ops: first check which ops exist in this build (`ttnn.experimental.deepseek_prefill.*`, `moe_compute`, `TTMoEGate`); if ops are missing, STOP and report
(a C++ rebuild/cherry-pick needs the user's approval). Read DSV4 files via `git show old/ssinghal/dsv4p1:models/demos/blackhole/deepseek_v41_flash/tt/...` (moe_block, moe_weights, prefill_unified_moe, uni_policy, router, router_select).

Phase A (3 h box, single card / small mesh): op limits at K=4096, N=2048, 8 experts/chip, 256 experts top-8: single-card moe_compute test at these dims (study 4.1 expects ring tile counts to fit: verify by RUNNING it, PCC > 0.999 vs torch SwiGLU
with no clamp), ring 8 banks, bfp8 weights from dequantised fp4 (bfp8 must reproduce checkpoint values exactly: measure max abs diff on a real expert). Router: MiMo biases ~1.7-2.2 (L1), ~10.4 (L20), ~11.7 (L46) with std 0.02-0.05 vs bf16 ulp 0.0625:
reproduce expert-set agreement on real tokens (need track 0 dumps/real hidden states; until then use real embedding+L0 outputs from a quick CPU run or ask track 0) with plain bf16 gate vs hi/lo bias split + fp32 score (DSV41Gate technique, sigmoid instead of sqrtsoftplus, norm_topk_prob over 8, no scale).
Phase B: 4x8 layer: router + dispatch + experts + combine, decode layout (moe_compute) and prefill (unified op, one-copy ring if it works; else document the fallback and DRAM cost with the memory table from the study). Capacity check that fails loudly, per-mesh-row checks,
real-token skew test, determinism (5 runs bit-identical), 32-row corruption test at N=256..4096.
Gate: router expert-set match >= 99% on real tokens (flip rate per layer logged); layer PCC >= 0.999 vs dump (layers 1, 20, 46 weights); no dropped tokens; determinism passes.
