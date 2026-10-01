# t51: what to do next on LTX-2.5 denoise (S1 ~2.14 s, S2 ~2.42 s)

Off-device review, 2026-10-01. Sources: research/ltx_denoise_t14.md (per-op block profile, 2.3 DiT,
same shapes as 2.5), research/denoise_plan.md (t22), worktrees/t32/NOTES.md (ring SDPA sweep),
~/.tt-buddy/notes/ltx-1080high-rt.md (fidelity study). No raw profiler CSVs survive; numbers are
the saved summaries. Excluded per spec: ring SDPA chunking (t32: shipped configs fastest, V2A split-K
2x slower), sparse S2 attention (t41: rejected).

## S2 block (16.4 ms, median per-op sum) by op type, minus excluded items
| op | us/block | note |
|---|---|---|
| ring SDPA (excluded) | 5366 | |
| all-gather-matmul (8) | 2609 | comm-bound, 5% FLOP util |
| plain matmul (14) | 1759 | incl. QKV/Q run as plain matmul after the dedup gather |
| all-gather (9) | 1321 | ~1150 of it = the 3 hoisted gate gathers (384 each) |
| matmul-reduce-scatter (2) | 1302 | ff2 |
| strided AG-matmul (2) | 1060 | |
| distributed RMSNorm (20) | 1001 | |
| addcmul (12) | 600 | |
| RoPE (8) | 381 | |
| eltwise (21) | 281 | |
| concat/heads/permute/slice | 510 | |

## Ranked (est. e2e saving, S1+S2)
1. Gate fold (LTX_FUSE_GATE): -0.20 s block-measured (S1 -3.7%, S2 -4.9%), older e2e
   2.33/2.60 -> 2.24/2.47 s. Removes the 3 hoisted gate all-gathers and 3 skinny N=8 matmuls per
   block. Not bit-exact (block PCC >= 0.99999); needs the 5-seed VBench + visual. Was blocked on a
   second ~37 GB weight cache: the prototype below removes that blocker.
2. Gate fold + RMSNorm-AdaLN fusion together: block -12.2% / -11.1% -> about -0.5 s est.
   AdaLN alone measured -0.10 s e2e (job 582). Not bit-exact; one shared 5-seed eval for both.
3. Audio branch without TP: -0.15 to -0.25 s est. (audio is 18% of the S1 block, 32 tokens/device,
   latency-bound CCL matmuls). Not bitwise exact (to_out/ff2 reduction order changes). High effort.
4. num_links 4 on the ring: unknown, maybe -0.2 s (AGMM/MMRS/AG are ~6.3 of 16.4 ms at S2).
   Needs a full 4x8 run; hang risk on a shaky box. Cheapest big unknown once full-mesh runs return.
5. bf8 activations on the AGMM gathers only (halves fabric payload; weights stay bf16): unmeasured.
   Prior bf8 tier corrupted video at "medium"; SDPA-input bf8 lost PCC (0.876). Low odds.
6. Fabric-AGMM entries for A2V to_out (K=2048) and V2A to_kv: -0.03 to -0.05 s, exact for M/N-only
   block changes. Needs a device sweep.
7. Small exact items, ~10-30 ms each: drop the V2A video pad-mask multiply if the ring's
   logical_n already excludes those keys (one full-seq eltwise/block); batch the 6 per-block AdaLN
   table adds per step (~280 tiny programs/step).
Rejected/known dead: LoFi/fidelity (no gain), V2A split-K (2x slower), ring chunk re-sweep (shipped
optimal), sparse S2 attention (quality).

Pick: #1 and #2 as one 5-seed eval. Only #1/#2/#4 can reach 0.2 s; none of them is bit-exact.

## Prototype on this branch: LTX_FUSE_GATE_ON_DEVICE=1
Post-load hook folds each attention's gate into Q/QKV on device: fused shard d =
concat(qkv_d, pad(gate_d to 32 cols)). Loads from the existing unfused `transformer/` cache, so no
`transformer_fusedgate/` cache is written; the unfused params' device data is freed after the fold.
Forward then takes the same code path as LTX_FUSE_GATE=1 (fused AGMM, no hoisted gather).
- CPU: models/tt_dit/tests/models/ltx/test_fold_gate_layout.py, 10 pass (video self/cross, audio
  self, A2V, V2A at TP 4 and 2). Runs the real _prepare_torch_state for both layouts; fused shards
  are torch.equal to the folded unfused shards. Mutation check: reversing the fold order fails all 10.
- Not run on device. Unverified there: ttnn.pad 8->32 columns on a TILE weight, concat of the padded
  gate, Parameter._check_data accepting the result, DRAM peak during the fold (one block's extra
  copy at a time).
- Device check when allowed (one short job): block A/B `test_ltx_transformer_block_trace_perf` with
  LTX_FUSE_GATE=1 (fused cache) vs LTX_FUSE_GATE_ON_DEVICE=1; block outputs must be bit-identical,
  timings equal. Then the 5-seed eval for the fold (+ AdaLN).
