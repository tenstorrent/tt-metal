# Separate BFP8 P in FP32 streaming LoFi attention

Private copy of the streaming header and an isolated full-chip driver. No
shared streaming/fullchip sources were changed. The reader/writer wrappers
include the existing full-chip dataflow, including its optional chain reader.

`p8_streaming.py --p-format fp32` is the existing FP32 streaming LoFi/BFP8-KV
control. `--p-format b8` changes probability storage to a separate native BFP8
CB7, with scalar BFP8 packing and a cheap unbiased exp fit. Both use BF16 Q
pre-rounded to seven significant bits, K/V pre-rounded to five and packed BFP8,
LoFi QK/PV, FP32 DST and FP32 recurrent state. Maxima stay BF16 as before.

P8 selects `SDPA_DIAG_EXP_MODE=1` to choose the relative exp fit, removes
`SDPA_MATCH_HIFI2`, and sets `SDPA_LOFI_DENOM`. Merely removing MATCH would not
change the cheap `calculate_sdpa_exp_stream_effective` coefficients. There is
no added per-value SFPU P-rounding pass. Native BFP8 quantization is part of the
experiment; it is not claimed to be RNE/unbiased. LoFi PV's logical left input
maps to SrcB and consumes all seven significant bits decoded from BFP8. The
denominator multiplies **the same stored P** by ones with LoFi, keeping the
effective weights matched rather than reducing pre-quantized DST values.

## Lifecycle and configuration

- CB6: 128 FP32 score pages. All QK writes, masks, max reduction and subtract
  inputs remain CB6. Reserve128 at K-chunk entry, publish16 per score row with
  held write pointer, pop128 at chunk end.
- CB7: 128 separate BFP8 P pages. Reserve128 at entry. Each phase1 Q-row loop
  after the first computes the preceding P row; publish16 only after its last
  column-subblock is packed. Phase2 drains/publishes the final P row. Every PV
  wait/load, denominator load and final P pop uses CB7. Both independent FIFOs
  wrap exactly once per chunk; no manual pointer repair is required.
- P packing is explicitly one tile per close/header/exponent section. Existing
  BFP width4 blocked packing has a separately observed framing failure. Setting
  only the MOP width to one is insufficient: the generic width4 row-pack helper
  would still issue only one pack call. This prototype explicitly loops tiles.
- Every sub/exp call restores FP32 score format **and width4 MOP** after P pack,
  because the next QK matmul skips pack initialization. QK and output packing
  retain their original width4 behavior. P/Scores are never L1-accumulated;
  existing FP32 PV output/recurrence accumulation is unchanged.
- The final-row drain retains the pack-to-unpack barrier before denominator
  and later PV work. Early phase2 PV targets the already-published first P row,
  not the still-being-generated last row.

Initial compile guards restrict P8 to materialized V, noncausal/unpadded square
attention, D128, K512, Q128 or Q256, one Q tile row per subblock and PV height1.
No ring, hybrid, residual, paired-FP32 score-alias, attention sink or pipeline
experiments are enabled. The driver also excludes centering/Q-prescale/B8-RNE
input experiments initially so the first comparison isolates this change.

## Memory and alias assessment

At Q256/K512 the original FP32 LoFi/BFP8-KV CB footprint is 1,212,416 bytes.
P8 adds 128*1088=139,264 bytes, yielding 1,351,680 bytes (1.289 MiB). No Q/K/V
chunk sizes or buffering slots change. This leaves 221,184 bytes of physical
1.5MiB before firmware/code/stack reservations; actual allocation still needs
device validation.

A compact shared-allocation alias is not a one-line datatype change:
`CircularBufferConfig` requires total bytes divisible by every page size.
524288 score bytes do not divide 1088. Rounding to the common 69632-byte quantum
gives unequal FIFO capacities, breaking current pop128-to-base wrap assumptions.
In-place write ordering also needs proof when processing multiple Q rows per
exp subblock; the BF16 schedule can overwrite unread scores. A padded BFP8 view
with 4096-byte page strides is source-plausible and may retain smaller native
payload traffic, but needs an isolated pack/unpack qualification before use.
The separate buffer avoids all these alias questions for the first datapoint.

## Commands and qualification

The card owner runs JIT/device tests; prototype preparation ran neither.

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/p8_streaming.py --label p8-separate-smoke-v1 --p-format b8 --length 1024 --heads 1 --cores 1 --sample-rows 32 --iters 1 --warmup 0 --trace-repeats 1 --max-l2 10
python experiments/sdpa-l2/bfp4-lofi-v2/p8_streaming.py --label p32-control-smoke-v1 --p-format fp32 --length 1024 --heads 1 --cores 1 --sample-rows 32 --iters 1 --warmup 0 --trace-repeats 1 --max-l2 10
python experiments/sdpa-l2/bfp4-lofi-v2/p8_streaming.py --label p8-h10-32k-v1 --p-format b8 --length 32768 --heads 10 --cores 110
```

First cover multiple K chunks and multiple Q jobs to test both FIFO wraps.
Then normal/uniform/scaled-QK/constant-V and several seeds, comparing PCC/L2,
gain and trace equality. Report attention-only and preprocess+attention time
separately. This first driver is full-chip real-DM; a resident wrapper can use
the same private header once correctness passes. Reduced P traffic is not a
performance claim: scalar pack and extra format/MOP switches can outweigh it.
