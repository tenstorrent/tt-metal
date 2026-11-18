# Roofline per op group: one 5120-token chunk at 142k history, galaxy 8x4 (X.7, 2026-11-18)

Measured: X.7 (`results/X.7_ops.json.gz`, `X.7_profile.json`), the committed default at b4e05edc584 (4-bit history,
DECODER_SKIP=decode = layers 0-19 + layer 20's KV head), steady chunk [145408, 150528). Device 82.2 ms a chunk (slowest
chip; per-op sum 87.3 ms). Clean wall 106.5 ms (P.77's A/B at 142k; X.6 before P.77: 111.0) -> **48.1k tok/s**;
device-bound ceiling 62.3k tok/s. Comparison bar (owner): ~67k new tok/s a Galaxy on layers 0-20.

Recompute: `python3 X.7_roofline.py` (shapes and constants in the script; per chip, 20 layers).

## Hardware (per chip, Blackhole, 8x4 torus) and the achievable fraction used

| Resource | Peak | Planned at | Source |
|---|---|---|---|
| Matmul | 4096 FLOP/cycle a core at LoFi, 1.35 GHz = 5.4 TFLOPS a core; HiFi2 / 2. 110 cores (PSU rule) = 594 TF LoFi, 297 TF HiFi2 | 70 % (large M); sparse sdpa 50 % | tech_reports/GEMM_FLOPS/GEMM_FLOPS.md (BH 580 TF measured on the full grid) |
| DRAM | 8 GDDR6 channels x 64 GB/s = 512 GB/s | 80 % = 410 GB/s | tech_reports/CCLs/CCL_bandwidth.md (64 GB/s a bank measured); Saturating_DRAM_bandwidth.md (>92 % on WH); kimi_k27_4x4_layer.md (80 % of 512) |
| Ethernet | 25 GB/s a link a direction (BH Galaxy links run at half the 50 GB/s rate); 2 links a neighbour (single_bh_galaxy_torus_xy: channels 2), 4 neighbours | 66 % for all_to_all-like traffic; 70 % gathers | CCL_bandwidth.md: BH lines 66-96 %, rings 68-95 % of line rate |
| Torus hops (uniform traffic) | 8-ring average 2.0, 4-ring 1.0 | | |

Precision is unchanged everywhere (HiFi2 matmuls, fp4 experts, the reference's own formats): the floors below keep it.

## Floors and gaps (ms a chunk, per chip)

| Group | Measured | Floor | Bound by | Gap | What the floor assumes |
|---|---|---|---|---|---|
| **Host not hidden** (wall - device) | 24.3 | ~1 | host | **~23** | engram rows and the next dispatch ready before the device needs them; 836 op launches a chunk overlapped (or traced) |
| **MoE experts + combine** (`routed_experts_combine`) | 26.4 | 13 | DRAM | **~13** | expert weights 4.50 GB a chunk a chip (12 of 384 experts x 3 x 5120 x 2304 x 0.53 B, 20 layers) = 11.0 ms; compute 1.36 TF = 6.5 ms HiFi2 (3.3 LoFi) overlapped; combine's non-overlapped share ~2 ms. Achieved today: 170 GB/s = 33 % of peak |
| **Attention** (projections, sparse sdpa, RoPE, glue) | 21.9 | 8.3 | DRAM | **~13.6** | wq_b + wo_b + wo_a weights 2.5 GB a chunk (every chip holds them whole: 160 rows a chip) = 6.1 ms vs 10.3 measured (linear 6.8 + matmul 3.5); sparse sdpa 13.4 GFLOP a layer = 1.7 ms vs 6.5; q RoPE fused into wq_b's output = 0 vs 2.4; glue (slices, untilize, concat) ~0.5 vs ~2.7 |
| **MoE head** (`moe_head`: router, shared expert, owners) | 8.9 | 1.6 | compute | **~7.3** | shared expert 0.23 TF = 1.1 ms + weights 0.5 ms; router 5120 x 384 negligible |
| **MoE dispatch** (`sparse_dispatch_2d`) | 12.0 | 5.8 (bf16) / 3.1 (fp8) | links | **~6** (bf16) / ~9 (fp8) | 160 tokens x 6 picks x 10 KB = 9.5 MB a chip a layer; X ring 2 hops over 4 links a direction = 0.19 ms a layer at line rate; 66 % |
| **Residual mHC** (`mhc_post` x 2 + collapse) | 7.8 | 2.7 (DRAM) / ~0.5 (L1) | DRAM | **~5** | 4 fp32 streams = 13.1 MB a chip, read + write a sublayer; resident in L1 the floor is ~0.5 |
| Indexer (score ring, top-k; 3 sources) | 3.1 | 1.9 | compute | ~1.2 | score 0.30 TF = 1.45 ms HiFi2; top-k ~0.4 |
| q_a, kv, norms, typecasts | 2.3 | 0.8 | DRAM | ~1.5 | q_a weights 0.19 GB = 0.5 ms |
| Compressor, engram, history gather | 3.8 | ~2 | mixed | ~1.8 | 4-bit gather: 3 x 76800 x 320 B in over 8 links at 70 % = 0.5 ms |
| **Device total** | 82.2 | **~40** | | **~42** | |

Ranked gaps (ms a chunk): host 23 > attention 13.6 ≈ experts + combine 13 > MoE head 7.3 > dispatch 6 (9 with fp8) >
mHC 5 > the rest < 2 each. A realistic target (floors x 1.5) is a ~60 ms device chunk and a wall within ~2 ms of it:
**~85k tok/s** at 5120-token chunks.

## Chunk size (owner pointer 4)

18 ms of the 40 ms floor is weights read once a chunk whatever its length (experts 11.0, attention projections 6.1,
q_a and shared 1.0). At 160 rows a chip, every projection and expert matmul is DRAM-bound (intensity ~300 FLOP/B vs
the chip's ~720 FLOP/B balance at HiFi2). An 8192-token chunk (256 rows a chip) spreads those 18 ms over 1.6x the
tokens: floor ~6.5 us a token vs 7.8 (+20 %), and more where today's kernels are far from their floor (small-M
expert tiles: 80 rows an expert today, 128 at 8192). 1536 / 2560 go the other way. The sweep (1536 / 2560 / 5120 / 8192
at ~142k) measures it before anything is built on it.

## Notes

- Under the profiler the gaps between ops (143.6 ms summed) are host dispatch; the clean wall's host share is 24.3 ms.
  The engram wait (host n-gram hash and lookup before layer 1) is the largest single idle window in the timeline.
- The kernel sums are of each op's slowest chip; the chips agree within 1.7 % (78.0-79.6 ms timeline busy).
