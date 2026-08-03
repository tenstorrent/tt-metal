# Operation-topology audit — full-model token-out decode path (optimized_full_model)

Derived from the reduced one-of-each-kind tt-perf-report (`tracy/tokenout/decode_perf_report.txt`,
eager 1-step so per-op device times are attributable; op-to-op GAP columns are eager-dispatch
artifacts that vanish under trace). Batch-1 decode, M=32 tile-padded single token, 1×4 Blackhole mesh.

The 40-layer decoder body is inherited UNCHANGED from stage-04 `OptimizedMultichipDecoder` (TP=4
attn/dense + EP=4 MoE, replicated BF16 residual, 2 ring all_reduce/layer, packed QKV + packed
gate/up, BFP8 KV, BFP8 attn/dense/shared weights LoFi, BFP4 routed-expert weights LoFi, HiFi2
router, fp32/HiFi4 SDPA). This stage's remit is the FULL-MODEL-ONLY (terminal + orchestration) path.

## Terminal op sequence per decode step (full-model-only cost, one device shown; all 4 run in parallel)

| # | op | device time | cores | dtype / fidelity | bound | notes / action |
|---|----|-------------|-------|------------------|-------|----------------|
| 1 | `EmbeddingsDeviceOperation` (token embed, replicated) | ~1 µs | 1 | UINT32,BF16=>BF16 | trivial | replicated embed so token feedback stays on device; free. Keep. |
| 2 | (40× decoder layers) | — | — | — | — | stage-04 territory, preserved; see `doc/optimized_multichip_decoder/tracy/`. |
| 3 | `LayerNormDeviceOperation` (final RMSNorm) | ~30 µs | 1 | BF16,BF16=>BF16 | — | exact on replicated hidden, HiFi4 ck. Single-core norm on H=2048; same as the decoder norms. Keep. |
| 4 | `MatmulDeviceOperation 32 x 2048 x 25088` (**LM head**, column-sharded) | **277→163 µs** | 98 | LoFi BF16 x **BF16→BFP8** | **DRAM 73.8%** | **OPTIMIZED THIS STAGE**: weight BF16→BFP8 halves DRAM bytes of this DRAM-bound op → measured 41.7% faster (279.5→163.0 µs), greedy token preserved, logits PCC 0.99976. Plain tiled matmul fanning N=25088 across 98 cores (DRAM-width-shard helper overflows L1 at this N — recorded stage-05); 73.8% DRAM util is already bandwidth-efficient. |
| 5 | prep (`Typecast`,`FillPad`,`Pad`) | ~35 µs | 110 | — | — | Sampling1D pads each 25088 vocab shard → 32768 (power-of-2) so local TopK uses the fast multi-core path, and masks invalid vocab ids. Keep (this is the anti-slow-TopK fix). |
| 6 | `TopKDeviceOperation` (local per-shard top-32) | 125 µs | **65** | BF16,UINT16=>BF16 | — | **multi-core (65) fast path**, NOT the slow single-core fallback. Per-device 32768-wide shard, k=32. Local top-k on each vocab shard — NOT a full-vocab TopK. |
| 7 | `AllGatherDeviceOperation` (candidate values) | 13 µs | 5 | BF16 | — | gathers only the **4×32 candidate set**, NOT the 100352-wide vocab. |
| 8 | `AllGatherDeviceOperation` (candidate indices) | 13 µs | 5 | UINT16 | — | 4×32 indices. |
| 9 | `ManualSeedDeviceOperation` | 17 µs | 110 | — | — | Sampling1D seed; greedy params k=1/p=1/temp=1. |
| 10 | `SamplingDeviceOperation` | 25 µs | 1 | BF16,INT32=>UINT32 | — | writes sampled id into `tt_out_tok` persistent buffer (device feedback). |
| 11 | `PlusOneDeviceOperation` ×2 (cur_pos, rope_idx) | ~1 µs ea | 1 | INT32/UINT32 | — | on-device position/RoPE advance; no host position refresh. |

## Anti-pattern check (goal + $optimize LM-head/sampling contract)
- **No `ArgMaxDeviceOperation`** anywhere in the measured path. ✓
- **No full-vocab (100352) all-gather**: the only all-gathers are the 4×32 candidate value+index sets (5 cores). ✓
- **No generic single-core full-vocab `TopKDeviceOperation`**: terminal TopK is per-shard (32768) on 65 cores (fast path); the 45-µs single-core TopKs in the table are the MoE **router** top-8 inside layers, not terminal. ✓
- **No host argmax / full-logits readback** on the measured token-out path (host-sampling compat mode is separate). ✓
- Force-argmax rejected in stage-05 (full-vocab all-gather + wrong ids on this mesh); greedy = top-k(k=1) split sampling, semantically exact. ✓

## Sampler+feedback cost reconciliation
Terminal sampler ops (5–11) sum ≈ 35+125+13+13+17+25+2 ≈ **230 µs ≈ 0.23 ms**, matching the
directly-measured full-model `token-out − logits-only = 0.257 ms/tok (0.78%)`. The sampler provably
does not dominate. The LM head (op 4) is in BOTH logits-only and token-out so it does not appear in
that delta; the BFP8 change reduces both.

## Collectives / residual (inherited, verified unchanged)
Per-layer: 1 `ReduceScatter` (~22 µs) + 1 `AllGather` (~13 µs) per all_reduce, 2/layer, on the
replicated BF16 residual. No inter-layer collective added by the wrapper (residual stacks directly).
Terminal adds only the 2 tiny 4×32 candidate all-gathers. Sharded/fused-CCL/persistent-buffer
families were measured and rejected in stage-04 (`doc/optimized_multichip_decoder/ccl_family_evidence.md`);
preserved here.

## Actions taken
1. LM-head weight BF16 → **BFP8** (measured 41.7% faster, token-preserving) — applied via
   `LagunaModel.from_pretrained(lm_head_dtype=ttnn.bfloat8_b)` default.
2. Everything else already optimal: logits-only decode sits AT the decoder-layer-stack floor;
   terminal norm/embed are trivial; sampler is 0.78%; no avoidable gather/argmax/reshard. No further
   full-model-only op is a material decode cost.
