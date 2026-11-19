# Operation topology audit

Baseline source 082c82fa, before any source edits or candidate measurements.
Evidence: ../multichip_decoder/profile_accounting.md and corresponding
advice-enabled tables; tt/multichip_decoder.py and inherited optimized_decoder.py.

| Boundary | Current sequence/cost | Candidate family/action planned | Constraints and evidence |
|---|---|---|---|
| Input/post-attention norm | Hidden shard -> stats AG -> distributed norm -> hidden AG; gather-first selected in some phase/kind cases | Compare gather-first whole layer and stats family under same precision; preserve sharded inter-layer residual | BF16; two norms. Explicit decode CCL 76–96 us; prefill 0.4–12.5 ms |
| Repeated input projections | GDN packed QKV/Z/B/A plus narrow correction; full attention packed QKV and separate gate | Compare fully packed QKV/gate or packed/split families; setup weight transforms | BFP8 default; GDN correction accuracy retained |
| Row attention/MLP outputs | DRAM-sharded decode MM -> RS; intermediate prefill native MMRS in 128-row pieces -> shard | Compare fused paths through residual/next norm, never immediate old-contract restoration alone | Current public residual BF16 [B,S,1280]/rank; no inter-layer collective |
| Packed MLP | Packed gate/up decode; tile-pair native SwiGLU prefill | Compare separate, packed, native and compatible fused AG-MM family | GDN HiFi2, full LoFi; BF16 activations |
| Batched prefill flattening | Repeated [B,S,H] -> [1,B*S,H] -> [B,S,H] physical reshapes | Keep flat representation through token-independent regions; isolate attention/GDN temporal boundaries | B32/S33 reshape 4.043ms GDN / 3.686ms full; internal padding must preserve logical lengths |
| Decode placement | Compact [1,B,H], eight activation storage cores, DRAM bank-sharded weights, 2 readers/K2 | Preserve residual L1 longer; precision-locked per-role geometry incl legal larger K divisors | Final inherited matmul bandwidth 293–443GB/s of 512GB/s |
| Collective allocation | Async CCL ping-pong semaphores; trace-captured outputs | Shape/role keyed persistent buffers with exact lifetime analysis and replay stress | No cross-layer alias, no allocation tracking exemptions |
| Precision | BFP8 weight/KV, BF16 activation/CCL; LoFi attention; GDN MLP HiFi2 | Test BF8 CCL/activation under compatible family; fidelity by role and real activations | BFP4 candidates only, never default; unchanged PCC |

For each family, record axis 1, dtype, persistent buffer plan, actual next
consumer and measured cost. Replicated whole-layer controls are comparisons,
not an acceptable final fallback. Hidden shard communicates 1/4 residual;
RS produces 1280 values/rank from 5120 local partial sums, AG consumes 1280
and produces 5120. At BF16 a padded 32-row decode output is 327680 bytes/rank
before RS and 81920 bytes/rank after; stats AG is much smaller. These are
tensor bytes, not wire-traffic claims. No layer-boundary gather is timed or
required by the current sharded contract.
