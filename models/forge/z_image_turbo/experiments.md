# DiT Single Pass Performance Optimization — Final Report

## Results
- **Baseline: 1176.6 ms**
- **Final: 1004.7 ms** (-171.9 ms, **-14.6%**)
- **PCC: 0.998** (threshold: 0.98)
- **44 experiments** conducted, 17 keeps

## Active Optimizations (in order of impact)

| Optimization | Savings | Category |
|---|---|---|
| BF16 RoPE (was F32 + typecast) | -83 ms | Data format |
| Cached RoPE freqs (was recomputed per call) | -48 ms | Redundant computation |
| CCL 4 ethernet links (was 1) | -15 ms | Communication |
| HiFi2 + packer_l1_acc matmuls | -7 ms | Compute config |
| fp32_dest_acc disabled for norms | -3 ms | Compute config |
| Metal trace capture/replay | -3 ms | Dispatch overhead |
| BFP8 MLP weights | -3 ms | Data format |
| nlp_create_qkv_heads for RoPE transpose | -3 ms | Op fusion |
| BFP8 adaLN + embedder weights | -1 ms | Data format |
| HiFi2 + math_approx for norms | -1 ms | Compute config |
| Default matmul blocking (custom was worse) | -3 ms | Compute config |
| Fused w1+w3 MLP via matmul_split | -0.3 ms | Op fusion |
| all_gather use_hyperparams=False | -0.3 ms | API tuning |

## Key Insights

1. **RoPE was the #1 hidden bottleneck** — F32 complex rotations + recomputed frequency tables consumed 131 ms (11% of baseline), purely unnecessary overhead

2. **CCL links were under-provisioned** — going from 1 to 4 ethernet links saved 15 ms. Hardware supports exactly 4 links per hop direction.

3. **Model is DRAM-bandwidth-bound** — HiFi4 → HiFi2 → LoFi all give identical latency; BFP8 attention weights give zero speedup; fused silu gives zero speedup. The matmul kernel is idle waiting for DRAM reads.

4. **BFP8 helps MLP but not attention** — MLP weights benefit from BFP8 (-3 ms); attention weights (QKV fused, to_out) show zero or negative improvement regardless of conversion method.

5. **BFP4 destroys accuracy** — PCC drops to 0.963 with BFP4 MLP weights.

6. **Fused MMRS doesn't help at this grid size** — requires sacrificing 2 rows of cores for CCL workers, netting zero gain.

7. **Custom matmul blocking counterproductive with trace** — the default 8x8x8 works better than custom (4,K,4) when combined with metal trace.

## Experiment categories exhausted
- Compute configs: HiFi2/LoFi, packer_l1_acc, fp32_dest_acc, math_approx (all combinations)
- Data formats: BFP8/BFP4 for all weight types, BFP8 activation output
- CCL: 1-4 links, Ring/Linear topology, persistent/non-persistent buffers, hyperparams
- RoPE: F32→BF16, cached freqs, pre-sliced (crash), nlp_create_qkv_heads
- Op fusion: fused silu, fused MMRS (default + custom config), fused w1+w3
- API variants: matmul vs minimal_matmul, different fabric configs
- Layout: interleaved refiners, l1_small_size, trace_region_size

## Next steps requiring architectural changes
1. **L1 sharding** — would keep weight tiles in L1, dramatically reducing DRAM reads
2. **Fused MMRS with tuned 13x9 config** — needs custom core grid partitioning
3. **Weight compression** — INT8/INT4 with dequant, or learned quantization
4. **Sequence parallelism** — distribute seq_len across devices alongside TP

## Plot Data

```csv
experiment,time_ms
0,1176.6
2,1170.1
4,1165.6
7,1162.7
13,1160.2
14,1077.1
15,1029.1
19,1028.8
22,1026.0
23,1022.7
24,1022.1
25,1009.8
26,1006.9
27,1005.0
36,1004.7
```
