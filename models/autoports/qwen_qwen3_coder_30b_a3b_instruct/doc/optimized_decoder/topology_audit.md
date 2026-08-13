# Operation-topology audit — measured path

Required by `$optimize` before any local knob tuning. Derived from the stage-01
profile (`doc/functional_decoder/ops_perf_*.csv`) read together with
`tt/functional_decoder.py`, not from comparison to another model.

Baseline: prefill ~536 µs/token (flat, S=128..2048); traced decode 1.5655 ms
@ ctx128 / 1.9942 ms @ ctx4096 (`../functional_decoder/perf_decode.csv`).
Single p300c die, batch 1, bf16 weights.

## Where the time goes

| | prefill S=512 | decode (paged) |
|---|---|---|
| `SparseMatmulDeviceOperation` | **96.79%** — 265.45 ms / 48 calls | **92.6%** — 17.64 ms / 6 calls |
| `UnaryDeviceOperation` (SiLU) | 1.21% | 1.4% |
| `BinaryNgDeviceOperation` (mul) | 0.84% | 0.9% |
| `PermuteDeviceOperation` | 0.52% | 0.5% |
| `FastReduceNCDeviceOperation` | 0.29% | 0.3% |
| `MatmulDeviceOperation` (qkv/o/router) | 0.11% | 1.6% |
| `SDPAOperation` | 0.08% | — |
| everything else | <0.2% | <2% |

The "decode (paged)" column is the whole `ops_perf_decode_paged32.csv` capture —
that profile runs a 32-token prefill and then one decode step, so its 6
`sparse_matmul` calls are 3 of each and the 17.64 ms is dominated by the prefill
half. Over the decode region alone (rows 47–104, 58 ops, 1513.9 µs) the same two
lines read 69.0% for `SparseMatmul` and 10.2% for `Matmul`. The finding below
does not turn on which of the two you use; the table at the bottom of this file
uses the decode region throughout.

Attention is 0.08% of prefill. **All meaningful optimisation is in the experts**
— true of prefill, and true of decode only until the experts got fast. See
finding G and the post-optimization table at the bottom.

## Current op sequence (one layer)

```
rms_norm -> linear(wqkv) -> nlp_create_qkv_heads -> per-head rms_norm x2
         -> rotary_embedding x2 -> SDPA -> nlp_concat_heads -> linear(wo) -> add
rms_norm -> linear(router) -> topk -> max/exp/sum/div -> scatter
         -> [per 32-token chunk] sparse_matmul(gate), sparse_matmul(up),
            silu, mul, sparse_matmul(down), mul(routing), fast_reduce_nc
         -> add
```

That is the audited (stage-01) sequence. After finding H the router half reads
`linear(router) -> topk -> slice(max) -> sub -> exp -> scatter -> matmul(ones)
-> div`: the same arithmetic with both keepdim reductions, and the two
``FillPad``s they pulled in, gone.

## Findings and candidate replacements

| # | finding | evidence | candidate | action |
|---|---|---|---|---|
| A | **Dense all-expert prefill.** Prefill passes sparsity as `[1,1,group_size,E]` — granularity is per 32-token *tile*, not per token. Across 32 tokens × top-8 = 256 selections over 128 experts, effectively every expert is hit, so tile-granular sparsity can never prune anything. | profile row: `active=128/128`; `nnz = num_experts * group_size` | Use per-token sparsity as decode already does (`[1,1,tokens,E]`), giving `nnz = 32×8 = 256` instead of `32×128 = 4096` | **do first** — the checklist forbids a dense all-expert runtime path outright |
| B | **gate and up are separate matmuls over the same activation.** Two `sparse_matmul` calls read the identical `hidden_grouped`. | 2 of every 3 sparse_matmul calls | Pack `[gate;up]` into one weight and split the output — the checkpoint already stores them fused as `[E, 2I, H]`, so packing is *undoing* work stage 01 did | measure packed vs well-tuned separate |
| C | **~5.4% of peak FLOPs, DRAM 12.5–24.9%.** Neither compute- nor bandwidth-bound; the kernel is under-fed. | `tt-perf-report` stacked report | program-config sweep: core grid, `in0_block_w`, output subblocks | sweep after A |
| D | **24 of 110 worker cores** on gate/up (64 on down). `_sparse_matmul_config` only spreads N across cores and N=768 for gate/up gives 24 tiles. | profile `Cores` column | larger/2D grids; reconsider M-dimension blocking | sweep after A |
| E | **bf16 expert weights at HiFi4.** Deliberate stage-01 choice for bringup clarity; expert matmuls dominate, so this is the canonical precision lever. | 96.8% / 92.6% of time | BFP8 then BFP4 on gate/up, then down; fidelity LoFi/HiFi2/HiFi4 swept per group | required by checklist; real weights only |
| F | **Decode activations are DRAM interleaved** between norm / attention / MLP boundaries. | `in0:dram_interleaved` on every row | width-shard decode activations in L1 | after A–E |
| G | Attention already uses fused QKV + SDPA + `nlp_concat_heads`. | 0.08% of prefill | none | **revised — see below.** The original call was "no action", correct for prefill and wrong for decode once A–E landed. |
| H | **The router and its routing prep are 111.6 µs of decode device time and had no finding.** Rows 68–88 of `../functional_decoder/ops_perf_decode_paged32.csv`: `TopK` 26.27, router `Matmul` 24.57, `FillPad` ×3 25.03, hand-written softmax 15.81, a tile↔row-major round trip around `scatter` 12.45, `Scatter` 3.17, `Slice`/`Typecast` 4.32. A–G left every one of them untouched, so the block went from 7.4% of stage-01 decode to **20.9%** — larger than `SparseMatmul` (17.1%) and `Matmul` (13.8%), and comparable to the `ReshapeView` compaction this file calls the dominant remaining gap. | the row range above, in a profile that is still on disk | delete both keepdim reductions (each drags a 10.4 µs `FillPad`); move the sum after the scatter where the reduction length is tile-aligned; try dropping the layout round trip | **done — 111.6 → 87.8 µs**, whole traced layer 0.5866 → 0.5615 ms, routing bit-comparable (`work_log.md` §7); the divisor guard §7 also adopted put the block back to 88.9 µs for +1.6 µs on the layer. The round trip is *not* removable — `sparse_matmul` requires ROW_MAJOR sparsity while every scale consumer requires TILE — and the residue is `TopK` + the router matmul, neither of which this layer controls. |

## Finding G, revised after A–E

The audit measured attention against the *stage-01* decode, where the expert
matmuls were 69.0% of decode device time and attention was 10.2%. Once
A–E cut the experts by ~5×, the same absolute attention cost became a large
relative one, and "no action" stopped being justified:

| decode device time | stage 01 | after A–E | after G revised | after H |
|---|---|---|---|---|
| `SparseMatmulDeviceOperation` (experts) | 69.0% | 18.5% | 17.1% | 18.0% |
| `MatmulDeviceOperation` (qkv + o_proj + router + the §7 sum) | 10.2% | 24.7% | 13.8% | 14.7% |
| `ReshapeViewDeviceOperation` (expert-path compaction, 3 of the 4) | 0.2% | 18.4% | 21.4% | 21.7% |
| router + routing prep (rows 69–88, not an op-code line) | 7.4% | — | 20.9% | 17.3% |
| total decode device time | 1513.9 µs | 620.8 µs | 534.8 µs | **512.7 µs** |

The first and last columns are the *decode region* of the two archived
profiles: rows 47–104 of `../functional_decoder/ops_perf_decode_paged32.csv`
(58 ops, 1513.9 µs) and rows 45–103 of `ops_perf_optimized_decode.csv` (59 ops,
512.7 µs). The "after G revised" column was the same file before finding H was
acted on; its 534.8 µs profile has been replaced by the re-profile and is not
on disk, which is why every H-column figure below is quoted from the file that
is.
An earlier revision put 92.6% / 1.6% in the stage-01 column, which are that
profile's *whole-capture* shares — prefill included, where the dense
all-expert `sparse_matmul` dominates — and so were not comparable with the other
two columns. The middle column is an intermediate profile taken after A–E and is
not archived; it is kept because it is what motivated revisiting G.

Action taken: the two decode projections now run
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` over 8 cores, one per
DRAM bank, with L1 width-sharded activation and output, at bfloat8_b — qkv
57.06 → 27.33 µs and o_proj 72.80 → 21.91 µs, both legs read off the archived
decode regions above (rows 48 and 65 of the functional CSV, rows **47 and 65**
of the optimized one; `tt-perf-report` prints the same two ops as its rows 49
and 67). The router matmul (24.45 µs, 4 cores) is left alone: its N=128 is 4
tiles, so it cannot use more cores, and its fp32 logit-space top-k is
load-bearing for routing correctness. Everything *around* that matmul is
finding H. See `work_log.md` §5–7.

The generalisable point: a percentage-of-total finding has a shelf life of
exactly one optimization. Findings A–F were all expert-path items, so acting on
them invalidated the evidence behind G — and G was an attention item, so acting
on *it* is what finally left the router as the largest unaudited block in the
profile (finding H). The audit was re-read against the profile once per round of
optimization, and it needed to be.

## Where the time goes now (post-optimization)

From `ops_perf_optimized_decode.csv` (rows 45–103, 59 decode ops, 512.7 µs) and
`ops_perf_optimized_prefill_s512.csv` (256 ops, 35.23 ms). Every op code at or
above 1% of either column is listed, so the columns are read the same way:

| op code | prefill S=512 | decode |
|---|---|---|
| `SparseMatmulDeviceOperation` | 71.8% | 18.0% |
| `ReshapeViewDeviceOperation` | — | 22.3% |
| `MatmulDeviceOperation` | 0.8% | 14.7% |
| `UnaryDeviceOperation` | 9.6% | 10.3% |
| `LayerNormDeviceOperation` | 0.2% | 9.4% |
| `BinaryNgDeviceOperation` | 6.5% | 3.4% |
| `PermuteDeviceOperation` | 4.0% | — |
| `SliceDeviceOperation` | 3.4% | 1.4% |
| `TopKDeviceOperation` | 0.1% | 5.1% |
| `FastReduceNCDeviceOperation` | 2.2% | — |
| `NLPCreateQKVHeadsDecodeDeviceOperation` | — | 2.8% |
| `SdpaDecodeDeviceOperation` / `SDPAOperation` | 0.6% | 2.3% |
| `PagedUpdateCacheDeviceOperation` | — | 2.0% |
| `RotaryEmbeddingDeviceOperation` | 0.2% | 1.8% |
| `UntilizeWithUnpaddingDeviceOperation` | 0.1% | 1.3% |
| `TilizeWithValPaddingDeviceOperation` | <0.1% | 1.1% |
| `ShardedToInterleavedDeviceOperation` | <0.1% | 0.9% |
| `FillPadDeviceOperation` | — | 0.8% |
| `InterleavedToShardedDeviceOperation` | <0.1% | 0.7% |
| `ScatterDeviceOperation` | <0.1% | 0.6% |

An earlier revision of this table listed sub-0.5% prefill entries while omitting
`Permute` at 3.9% and `Slice` at 3.4%, which are the third and fourth largest
things prefill does.

The top decode line is no longer a matmul. `ReshapeView` is the cost of
compacting `sparse_matmul`'s 32× M padding at M=1; both ways of removing it were
measured and are recorded as rejected in `work_log.md`. The 22.3% is the op-code
total over all four reshapes; the **three** that do the compaction are 111.12 µs
= 21.7%, and the fourth (3.39 µs) is in the layer tail. `FillPad` is down from
4.7% to 0.8% because finding H removed two of its three calls; the survivor is
inside `ttnn.topk`.

## Constraints carried from stage 01

- `fp32_dest_acc_en` must stay **off** for expert matmuls (halves matmul dest,
  corrupts output on Blackhole — tt-metal #49068).
- Router top-k must stay in logit space with fp32 accumulation; dropping to a
  bf16 128-wide softmax misroutes 83/128 tokens.
- Non-aligned sequence lengths must keep working; padding stays internal.
- Context contract: full 262144, must not regress.
