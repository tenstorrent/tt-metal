# Direct preparation fix and real-weight fusion, October 10

The earlier physical prototype passed L1 but failed the first DRAM case,
B32/token-row1. Q, K and V differed on all four ranks; gates matched.
The reader compacted32-byte BF16 face rows into adjacent32-byte scratch slots.
Blackhole DRAM reads require matching source/destination low six address bits
(`NOC_DRAM_READ_ALIGNMENT_BYTES=64`); the second face violated that contract.
L1 requires only16-byte matching alignment, explaining the placement difference.

The fixed reader uses64-byte face-row scratch slots and indexes around their
padding. All staging fits in the existing4096-byte private buffer. The nine-case
physical rerun passed, including both live allocations, all four ranks,
changed-input trace replay and bit-identical Q/K/V/gates. Native exp and FP32
normalization/recurrence arithmetic are unchanged. This is a Qwen preparation
prototype bug; it does not establish a cause for historical Kimi transport stalls.

## Real-weight layer comparison

Opt-in policy `single_step_flat_prepare_epilogue` preallocates value/gate scratch
and combines the direct reader with the existing fused epilogue atB16/B32.
B1/B8 retain their previous paths. Prefill retains native chunked scan.
All twelve control/candidate/control layer cases passed64FP32-reference updates
and bit-identical projected output, raw output and state on all four ranks.

| Batch | Shared-QK block us | Combined block us | Block speedup | Projected48-layer saving |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1016.70 | 805.58 | 1.262x | 10.134ms |
| 16 | 710.27 | 568.64 | 1.249x | 6.798ms |
| 8 | 559.00 | 558.89 | unchanged fallback | timing noise |
| 1 | 363.24 | 362.86 | unchanged fallback | timing noise |

Those blocks include projections, convolution, recurrence and output norm;
they exclude MLP. Applied to the measured shared-QK full-model times, the
linear-saving projection is16.54/11.88TSU at32K B16/B32 and18.38/13.83TSU
at16K. The projected useful-byte roofline fraction at32K rises to41.6%/43.3%.
These are predictions, not full-model measurements. At32K/B16,20TSU requires
another10.45ms saved after this projection; B32 requires another34.19ms.

## Persistent full-model follow-up

The v1 queue passed481CPU tests plus40subtests (one unrelated skip), then
repeated the layer gate on its exact frozen source. It reproduced the block
gain and output/state equality. The user then requested profiling first.
v1 was intentionally stopped during the first control's model loading; retain
its interrupted receipt without labeling it a model failure.

Replacement `qwen38-gdn-fusion-full-v2-20261010.service`, invocation
`a257a3e921da46a3a9c987347c196190`, repeats the short layer gate, profiles
current/fused paths at32K B32/B16, runs three full-model sweeps at32K/16K
B32/B16, then performs fresh eight-replica G0 and complete198-question GPQA.
The profiles use real-weight representative GDN/attention layers and synthetic
caches. They are not a complete traced-model critical-path measurement.

Every hardware stage uses `/tmp/tt-device.lock`; source manifests are checked
before stages. Model comparisons require identical output hashes and unchanged
precision/source/workload. The service has16h/256GiB/16CPU limits, bounded
children and output budgets, survives disconnect, and does not auto-resume
after reboot. No default or serving configuration is promoted by this queue.

## Next optimization priorities

1. Measure the combined full-model gain and the changed stage balance.
2. Eliminate packed-convolution layout work and retain useful activation
   layouts through convolution, preparation, recurrence and output projection.
3. Tune recurrence unpack/pack synchronization and math. Existing phase data
   shows about10us input wait in a162us B32 recurrence, so reader starvation
   alone is not the leading explanation. Two input buffers are already enabled.
4. Evaluate bank-local bulk KV delivery including redistribution, page-table
   traversal and compute backpressure. Raw reads achieved499-508GB/s/chip;
   this excludes attention compute and is not an attainable-model guarantee.
5. Calibrate the full traced-model critical path, CCL and matmul/layout costs
   before promising a twofold whole-model gain or75-78% useful-byte efficiency.

Precision stays BFP8 for weights/KV and FP32 for recurrent state. Primary context
is32K, secondary16K, with128K/256K tradeoffs explicitly reported. Blaze work is
out of scope following the user's direction to resume tt-metal optimization.
