# Cumulative optimized decoder contract

These contracts are selected together. Final per-device row times and shapes
are in `tracy/review_profile_l0` and `tracy/review_profile_l3`; the headline
latencies come from the unprofiled `after_review_*` default runs.

| Component | Selected contract | Measured evidence |
| --- | --- | --- |
| Mesh/runtime | Four local Blackhole chips, TP axis 1, Ring with two links; host pass-through thread pool selected before mesh creation | Final environment/mesh records; `AUTOFIX_profile_gap.md` |
| Linear attention projection | One packed Q/K/V/Z/B/A projection, per-rank `[5120,4160]`; local key heads 4, value heads 12, head dimension 128 | Attention matmul row and projection geometry CSV |
| Full attention projection | One packed Q/K/V/gate projection, per-rank `[5120,3584]`; local Q heads 6, KV heads 1, head dimension 256; on-device head split, Q/K norm and RoPE | Matmul, head-creation, norm/RoPE, cache update and SDPA rows |
| Projection numerics | BF16 input/output, BFP4 weights, LoFi, FP32 destination accumulation | Actual runtime dtype/fidelity columns in `final_matmul_rows.csv` |
| Linear recurrence | Native phased GDN, FP32 recurrent state, BF16 convolution history | GDN rows; adapted monolithic and serial controls |
| Full KV cache | BFP8 TILE, per rank `[physical_pages,1,32,256]`; page block 32; replicated logical page tables | Strict cache PCC, changed page-table/position and write-ownership checks |
| Logical users | Headline B1 is one active user, with 32 internal tile rows; tests preserve B2–32 as actual users | Per-user PCC and B32 stress, no tile-padding substitution for logical batch |
| S128 decode metadata | Current position tensor `[1]` contains 128; shuffled page table `[1,5]`; cache has 8 physical pages including 3 unused guards; only selected user's current row changes | Runner allocation formula and cache write-ownership gate |
| Decode SDPA | Explicit short grid 8×2, Q chunk 32, K chunk 32 for S128; longer grid 11×10, K up to 128 halved to divide mapped capacity | Final short and long SDPA rows, K64/K128 paired controls |
| Both residual norms | L1 WIDTH_SHARDED 40 cores, grid 10×4, shard `[32,128]`; sharded RMSNorm program block H1/W4, subblock W4, non-inplace | Both `LayerNormDeviceOperation` rows |
| Residual adds | BF16 output retained in the same 40-core L1 layout for B1; no collective or reshard inserted between layers | `inter_layer_profile_audit.json` and `inter_layer_contract.md` |
| MLP | Packed gate/up `[5120,8704]`, device split and SiLU/multiply, down `[4352,5120]`; BFP4/LoFi throughout | Packed/separate/fused comparisons and final matmul rows |
| Decode geometry | Attention/output/gate/down input cores 10/8/40/8, K blocks 16/6/4/17; DRAM readers per bank 2/2/3/2 | Precision-locked geometry CSV with program, M/N, subblocks, layouts, latency and PCC |
| Local layout transitions | Input/norm activation to each selected projection grid; attention/cache helper packing; output projection to 40-core AR/residual; packed MLP split | Every `Reshard`, interleaved/sharded, reshape and slice row remains in the complete per-device tables |
| Row reductions | Two async native ARs per layer, BF16 axis 1, persistent shared 1.25 MiB workspace; no extra reduction at the stack boundary | Runtime AR rows and coherent sharded-residual alternatives |
| Prefill | DRAM public input/output; internal chunk 4096, logical tails preserved; 1D short 64–256 with identity casts removed, long minimal/2D | Before/after prefill profiles and `review_cast_audit.json` |
| Trace | Caller-owned stable input/state/page-table storage; whole device-only decode captured; optional prefill trace validated separately | Trace bitwise gates, 100-step queued stress and watcher |

`performance_accounting.json` reports kernel time and op-to-op gaps separately
for every device, the critical device span, and the same signposted host
interval. It also gives the stored-weight/KV read lower bound. Larger-batch
public residual storage and the shared-workspace lifetime are specified in the
inter-layer contract; they are part of the supported path.
