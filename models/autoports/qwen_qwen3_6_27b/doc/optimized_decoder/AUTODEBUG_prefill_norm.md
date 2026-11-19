# Prefill precise norm prototype and measured movement opportunities

Source/CPU work only; no main decoder or shared-kernel edits, no hardware run. New helper snapshots:

- `candidates/hidden_rmsnorm_prefill.py`: generalized precise BF16-staged SFPU hidden RMSNorm.
- `candidates/prefill_data_movement.py`: corrected-projection and fused-SwiGLU movement reductions.
- `candidates/check_prefill_norm_mapping.py`, `prefill_norm_cpu_checks.json`: address/ownership checks.
- `prefill55_breakdown_cpu.json`: grouped device durations from the actual profiler CSV.

## Generalized exact norm

`hidden_rmsnorm_sfpu(x,gamma,epsilon=...,output=None,cores=None)` accepts DRAM-interleaved BF16 TILE `[B,S,5120]` for every positive logical B/S. Gamma is the existing folded FP32 `[1,1,5120]`. No `S%chunk==0` requirement is introduced. It returns the same logical/padded shape and BF16 dtype; a supplied output buffer retains its address.

The host distributes logical rows over available cores; each core runs its contiguous row range. Reader maps logical row `r` to physical row `(r//S)*round_up(S,32)+(r%S)`. This is essential for B32/S33: flattening `r` directly as a physical index would enter each user's31 padding rows. The output uses the identical mapping. Only a user's last logical token writes zeros into that user's remaining padding rows; there are no overlapping output writes.

Each core stages gamma once and retains it across all its rows. It uses the existing five-dense-tile BF16 staging and the **source-identical** square/reduction/FP32 reciprocal/epsilon/rsqrt/affine/BF16-rounding sequence from `hidden_rmsnorm_bf16_staged.py`. The only compute changes are the row loop, reset to BF16 input unpack at its start, reset from BF16 output pack back to FP32 inverse pack after the first row, and retaining gamma/zero until loop completion. Each row's raw input/output/inverse CB transactions remain balanced. Total CB allocation is70KiB/core, independent of S; no tensor host conversion or FP32 full-input intermediate exists.

Suggested integration: retain the existing compact decode norm branch first; before `explicit_sum_norm`, dispatch other BF16 rank3 width5120 inputs to this helper under a prefill-norm policy. The source handles B1 small lengths too. Parent should first validate B1/S1,17,33,65; B3/S33; B32/S33 and an aligned large prefix, then the existing output/state/sequence gates and warmed prefill. This prototype is not yet device-compiled or measured.

CPU verification: syntax passed; per-row arithmetic source block exactly matches the existing staged helper; all32 physical row positions ×5120 channel mappings passed. Sixty B/S cases (B1,3,7,32; S1,15,16,17,31,32,33,63,64,65,127,128,511,512,513), each with1/8/110-core partitions, verified every logical row is owned once, padding is zeroed once, and logical/padding writes are disjoint. Reproduce:

```bash
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_prefill_norm_mapping.py
```

## Profile evidence: norm is material, but reshapes are larger

`perf/profile55_bfp8_0/prefill_perf_report.csv` was available during this investigation. Its metadata identifies **norm50**, B32/S33, not the later passing packer53. Use it to locate topology cost, not claim final-candidate correctness. It reports31.283ms kernels +2.671ms gaps =33.954ms. Grouped kernel costs:

| Operation category | Device time |
|---|---:|
| ReshapeView,12 ops | **9.482ms** |
| Matmul,15 ops | 7.065ms |
| Binary,17 ops | 2.842ms |
| Chunk GDN,64 ops | 2.493ms |
| Minimal fused SwiGLU,1 op | 2.072ms |
| Slice,89 ops | 1.828ms |

The two explicit hidden norm sequences, IDs1022–1031 and1267–1276, take1.317ms and1.305ms respectively. The new norm targets these2.622ms; norm alone cannot explain or erase the full prefill regression. ID1021 is a separate399.5us typecast preceding the norm and is not included in this attribution.

Two larger, immediately actionable movement reductions are implemented as separate optional helpers:

1. **Corrected projections: flatten once, add FP32 partials to BF16 while flat, then unflatten once.** Existing `_linear` calls `_linear_raw` twice; each independently flattens x and restores padded `[B,S,N]` FP32 output before addition. Packed-GDN FP32 restore IDs1045/1059 cost1.693/1.688ms; output-projection restores1253/1264 cost.523/.523ms. `corrected_projection_flat_prefill` retains the exact same two matmul calls, configs, FP32 partials, and BF16-add boundary, but calls `_linear_raw` on `[1,B*S,K]` and restores only the final BF16 result. It also avoids one duplicated input flatten per pair. New final BF16 restore still costs time: these gross removed costs are not predicted net savings.
2. **Keep fused-SwiGLU output flat through down projection.** `_finish` currently reshapes activated `[1,B*S,17408]` to `[B,S,17408]`; `_linear` immediately flattens it again. IDs1279/1280 cost1.711+1.048=**2.759ms**. `minimal_mlp_flat_prefill` passes the flat activation directly to the same down projection, then restores `[B,S,5120]` once before residual addition. Its matmul/activation precision policy is unchanged.

Both helpers use native logical reshape semantics and preserve arbitrary nonaligned lengths. Validate accuracy and latency separately and together. They are preferable next experiments to weakening prefix precision. The CSV also advises increasing88-core matmuls toward110 cores, but current row-chunk safety limits and prior 2D output-block addressing findings still apply; do not blindly reintroduce partial output-width blocks.

## Independent full-attention prefill packing proposal

Use a phase-specific mode such as `full_pack_all='prefill'`, retaining both existing separate `packed_qkv`/`gate` and the added `packed_qkvg` weight. Setup currently deletes the separate weights whenever `full_pack_all` is truthy; change that cleanup only for true all-phase mode. Add an explicit `prefill` parameter to `_qkv`, set by `_attention_prefill`, and select packed-QKVG only for that phase. Compact decode should use packed-QKVG only when policy is literally true/all-phase; otherwise keep its best legal separate projections. Do not infer phase from sequence length, because prefill S1 is valid.

This is separate from GDN profile55 and is not yet an implementation or measurement. It must beat tuned separate full-attention prefill. Retaining both representations adds about74.375MiB of BFP8 tile payload for K5120×N14336 before any bank padding; check actual DRAM/context headroom before retaining it. No context-contract reduction is justified without physical-capacity evidence. Decode behavior/configs should remain on the already measured winner.
