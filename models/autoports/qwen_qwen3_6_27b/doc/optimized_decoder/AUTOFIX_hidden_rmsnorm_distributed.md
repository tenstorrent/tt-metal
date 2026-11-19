# Distributed accurate hidden RMSNorm candidate

Source-only AutoFix hypothesis: the accurate per-user norm spends too much time
issuing 320 small x reads and 320 small gamma reads per user. Partition channels
across 20 cores, retain the same FP32 SFPU arithmetic and BF16 output boundary,
and use full-page NoC transfers instead. No measured speedup is claimed.

## Artifacts and integration

`candidates/hidden_rmsnorm_distributed.py` exports
`hidden_rmsnorm_sfpu(x, gamma, epsilon=1e-6, output=None, scratch=None)` with the
same compact x `[1,B,5120]` BF16 / folded gamma `[1,1,5120]` FP32 contract as the
passing helper, for B=1..32. It additionally accepts reusable scratch
`[1,32,640]`, FP32 TILE interleaved DRAM. Supplying both scratch and output avoids
tensor allocation. Scratch must not alias x, gamma, or output. Its 20 tiles hold
one core's 32 row sums in column zero; other columns are unspecified.

Callers may import this helper instead of the incumbent for a candidate run;
there are no main decoder, shared TTNN, or device modifications in this work.
The existing `tests/run_norm_precision_probe.py` accepts this file unchanged.

## Implementation and source proof

Pass 1 assigns eight of the 160 input tiles to each core, covering all 32 token
rows and 256 channels per core. It issues eight complete 2048-byte BF16 reads,
performs eight FP32 SFPU squares and seven elementwise FP32 folds, then uses
`sfpu_reduce<SUM,Float32,REDUCE_ROW>` to produce a 32-entry column vector. The
whole tile is packed Float32 and written to the corresponding scratch page.

Pass 2 reads all 20 scratch pages on each core, folds them in FP32 SFPU, applies
an FP32 reciprocal of 5120, epsilon, and precise rsqrt. It loads its eight full
x/gamma tiles, multiplies x by the inverse vector with `sfpu_mul_bcast_col`,
then by row-zero FP32 gamma with `sfpu_mul_bcast_row`. An explicit SFPU typecast
rounds to BF16 before packing. The writer zeroes all nonlogical token rows and
writes eight complete output pages, with no overlapping writers.

`tt_metal/hw/inc/api/compute/sfpu_binary_bcast.h` explicitly documents FP32
column-zero and row-zero operands for these broadcasts. The Blackhole
`ckernel_sfpu_reduce.h` implements row SUM output in column zero. Thus neither
gamma padding nor nonzero unused partial columns contributes to logical values.
Arbitrary x padding cannot affect another row's statistic and is zeroed in the
final output. This differs from the old scalar whole-tile reduction: only the
channel axis is reduced.

All consumed Float32 CBs use `UnpackToDestFp32`; BF16 CB0 uses Default unpack
into FP32 destination. Every square, add, normalization multiply and affine
multiply stays in SFPU FP32, avoiding the native sharded norm's packed-scaler
and TF32 precision issues. There is no fitted scaling, host tensor conversion,
or replacement of the effective FP32 gamma with BF16.

Pass 1 uses 20 KiB CB storage/core; pass 2 uses 148 KiB/core. All NoC transactions
have 2048- or 4096-byte lengths and offsets, meeting 64-byte DRAM read alignment.
Two successive generic ops on the same default queue establish the scratch
producer/consumer dependency; no cross-core semaphore is needed. Each op
registers every referenced tensor and reconstructs runtime arguments from its
current addresses.

## Validation performed

Commands from the checkout:

```sh
python3 -m py_compile models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/hidden_rmsnorm_distributed.py models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_hidden_rmsnorm_distributed.py
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_hidden_rmsnorm_distributed.py > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/hidden_rmsnorm_distributed_cpu.json
```

Both pass. The CPU check validates complete nonoverlapping channel/page
ownership, read alignment, exact physical BF16 padding addresses, Python syntax,
inline kernel brace balance, and 24 same-input arithmetic cases. Cases cover
B=1,3,16,17,31,32 and x scales 0,.001,1,16, using BF16 x and FP32 gamma.
Worst modeled relative RMS versus a direct HF-order reference is 2.71e-5;
minimum BF16 bit equality is 99.99874%. The CPU reduction uses Torch's row SUM,
so it establishes the decomposition and an illustrative rounding envelope, not
the device SFPU reduction tree. An initial system-python attempt lacked Torch;
the recorded passing result uses the checkout's `python_env`.

## Required device proof

The parent owns all device use. First run:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_norm_precision_probe --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/hidden_rmsnorm_distributed.py --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/hidden_rmsnorm_distributed_device.json
```

Device compilation, same-input accuracy, changed-input reuse, persistent-buffer
trace replay, padding, watcher, warmed latency and the unchanged 512-step decoder
gate remain unverified. The core count is fixed at 20; the different positive-sum
association can change BF16 ties and must pass the whole decoder gate. B1 still
processes padded rows, so measured B1 benefit is especially uncertain. Full-page
statistics gather adds 1.56 MiB total reads and 19 SFPU tile adds/core; two program
dispatches add overhead. These are explicit risks to the performance hypothesis,
not reasons to claim an improvement without measurement.
