# Attention DRAM-sharded projection comparison

Four real-weight paired TT1/TT4 runs:4096 input,128 decode,batch1,Linear
1x4 mesh. Shared geometry1, optimized shared weights, grouped MoE reduction,
hybrid EP-prefill/indexed-TP-decode, fused tail, LoFi QKV/WO. Attention CCL
is BF16 sliding and BFP8 full, matching their passing controls.

The candidate uses setup-time bank-sharded BFP8 weight copies and
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, eight banks with one
worker per bank. Inputs/outputs have L1 width shards; measured layer time
includes layout conversion. Original weights and projection remain for
prefill. No public layout, cache or logical sequence contract was changed.

| Changed projection | Layer | Min output PCC | Min cache PCC | TP4 prefill host us | TP4 decode host us | Control decode host us | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| qkv | sliding | 0.998923837 | 0.999996637 | 93825.473 | 736.702 | 735.835 | attention_dram_qkv_sliding.json |
| qkv | full | 0.995221383 | 0.999971440 | 79017.007 | 730.119 | 723.983 | attention_dram_qkv_full.json |
| output | sliding | 0.998884506 | 0.999996658 | 93700.124 | 743.375 | 735.835 | attention_dram_output_sliding.json |
| output | full | 0.999438167 | 0.999971502 | 78984.603 | 731.682 | 723.983 | attention_dram_output_full.json |

All four completed normally,exit0,with minimumPCC>=.995, all-rank KV checks,
exact repeated trace output for every step and runtime fallback guard. No
API/config fix, reset, or lock cleanup was needed. Hardware was released
after all four meshes closed. No default policy, runtime or runner edit was
made by the experiment executor.

None improves the observed complete-layer median versus its matching control.
The full QKV variant additionally approaches the accuracy threshold. This
rejects the tested bank-sharded attention projections for the selected path;
it is not a blanket claim that DRAM-sharded matmul is slow. The QKV sliding
difference is small enough that it should be described as no observed win,
not a statistically established regression. Timings are host-wall, not device
measurements; no profiler or Watcher ran in these comparisons.

Exact commands, runtime hashes, per-step PCC/timings and configuration labels
are embedded in each JSON. Matching `.log` files preserve execution and clean
device close. Runtime source SHA256 is
`49cd4b3e47aeca24389fd58c29262556d05ec35cf0f0d80c8077c02af51f88e5`.
Runner SHA256 at completion is `acf7c7991775248983d779bc6e782d6f803516528c998213d01aeb4a05138876`.
