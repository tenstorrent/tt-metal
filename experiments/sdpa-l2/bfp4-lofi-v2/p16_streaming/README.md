# Private P16 streaming probe

Question: does reducing probability storage from FP32 to BF16 accelerate the
streaming kernel when BF16 can retain properly framed width-4 blocked packing?
BFP8 probability packing required scalar tile framing, so its smaller payload
did not isolate the benefit of reducing L1 bytes from pack-control overhead.

Both drivers expose `--p-format fp32|bf16 --p-pack-width 1|4` (defaults FP32/4).
Both choices use identical unbiased cheap exp coefficients, LoFi QK/PV, FP32
DST/recurrent state, and a LoFi P-times-ones denominator. The denominator reads
the same stored P as PV; this is a matched *effective LoFi* sum, not an exact
sum of every stored mantissa bit. Neither choice adds SFPU P pre-rounding.
Thus the FP32 control is not the older biased-exp `SDPA_MATCH_HIFI2` control.

Q256/K512/D128 and Q7/BF16 plus K/V RNE5/native-BFP8 preprocessing are fixed.
Q has two slots, K and V one slot each. No input-slot, chunk-size, or alias
optimization is made. BF16 P uses a separate 128-tile CB7. The existing FP32
control keeps its original in-place FP32 CB6/7. QK scores remain FP32 until exp;
BF16 P is packed only after exp. The FP32 QK pack format and width are restored
after each P pack, including after the scalar FP32 control.

## L1 precheck

| P storage | Total CB bytes/core | Raw 1.5 MiB minus CBs |
| --- | ---: | ---: |
| FP32 in-place control | 1,212,416 | 360,448 |
| Separate BF16 P | 1,474,560 | 98,304 |

Raw headroom excludes firmware, program, allocator reservations and semaphores.
The BF16 candidate may fail device allocation. Both drivers print this warning
before attention launch. Do not silently shrink inputs or alias BF16 P to make
it fit: an allocation failure is a result of this experiment.

## Suggested qualification order

First try all four format/width combinations with distinct K/V and all query
rows. Use fresh labels:

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/p16_streaming.py \
  --label p16_bf16_w4_4k_smoke --p-format bf16 --p-pack-width 4 \
  --length 4096 --heads 2 --cores 4 --sample-rows 4096 \
  --check-preprocess --iters 0
```

Then compare scalar versus blocked output hashes for the same P format before
accepting timings. Cross-format outputs need not match. Default streaming
timing is N8192/H10/110 cores using the existing per-head KV chain. It records
attention and preprocessing separately, plus combined useful TFLOPs.

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/p16_resident.py \
  --label p16_bf16_w4_resident --p-format bf16 --p-pack-width 4
```

Resident mode loads input slots once, repeats the same K/V chunk, and saves only
the last Q block. It reports useful TFLOPs/core; preprocessing is excluded.
Its repeated inputs do not test changing online maxima or distinct long-context
keys, so passing the resident reference is insufficient by itself.

Only this directory and the two P16 drivers are new. The private header is a
copy of the P8 header; readers/writers include existing code unchanged. No
production, frozen, existing P8, or other active headers are modified.

Status: host syntax/descriptor checks only; JIT, allocation, scalar/blocked
equality, accuracy, and timing require device qualification.
