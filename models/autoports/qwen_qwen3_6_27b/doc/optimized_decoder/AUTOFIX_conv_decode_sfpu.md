# Single-launch decode convolution and carry update

This is a source-only AutoFix performance candidate. No device compilation,
correctness, watcher, trace, or latency pass is claimed here.

## Measured reason to try it

The accurate BFP8 profile in
`perf/profile55_bfp8_0/decode_perf_report.txt` reports 61 decoder operations and
2447.962 microseconds of kernel time. IDs 1548–1562 cover convolution's untilize,
three history slices, four products, three sums, SiLU, and history
slice/concat/copy. Their displayed times sum to approximately 190 microseconds,
excluding gaps. The eight arithmetic operations alone, IDs 1552–1559, total
approximately 120 microseconds. The recurrence remains larger: ID 1571 is
approximately 730 microseconds, 29.4% of the reported decoder time. Thus conv is
a material secondary target, while recurrence overlap/partition tuning remains
necessary for a large end-to-end improvement.

## Candidate and integration

- `candidates/conv_decode_sfpu.py` provides
  `conv_decode_sfpu(token, history, taps, output=None, cores=None)`.
- `candidates/make_conv_decode_candidate.py` inserts a policy-controlled call
  into a copy of an existing frozen decoder.
- `candidates/conv_decode_on57.py` is the generated frozen candidate based on
  `integrated57.py`; neither that baseline nor the main decoder was changed.
- `tests/run_conv_precision_probe.py` compares the helper with the actual frozen
  primitive `_batched_conv` method and a CPU reference using real conv weights.
- `candidates/check_conv_decode_cpu.py` and `conv_decode_cpu.json` are CPU-only
  source, address, arithmetic-order, and carry-update checks.

Enable `sfpu_decode_conv=True` on the generated decoder. The helper requires the
validated policy `conv_round_output=True`, `conv_packed_rm=True`,
`conv_functional=False`, and `conv_round_products=False`. `sfpu_conv_cores`
optionally caps its core count. The policy branch asserts these requirements;
it does not silently use a fallback when selected. The existing recurrence,
its FP32 Q/K normalization, BF16 persistent state, and readout arithmetic remain
unchanged. Prefill also remains unchanged.

To generate against a later frozen decoder without overwriting any candidate:

```sh
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_conv_decode_candidate.py --base pipeline62.py --output conv_on_pipeline62.py
```

The input token is compact TILE BF16 `[1,B,10240]`, for B1 through B32. Carry is
ROW_MAJOR BF16 `[B,3,10240]`; each of four taps is ROW_MAJOR BF16
`[1,1,10240]`. Output is packed ROW_MAJOR BF16 `[B,1,10240]`, so the existing raw
packed recurrence accepts it directly. All buffers are interleaved DRAM. A
separate caller output can be supplied for stable allocation. Current addresses
of the token, carry, four taps, and output are registered in every generic-op
descriptor; there is no host tensor operation in the runtime helper.

## Layout, arithmetic, and synchronization

Each work item handles 1024 consecutive channels of one user. There are ten
items per user, spread across up to 110 cores. The reader loads each carry/tap
segment directly into a BF16 CB. It gathers the corresponding TILE token's
face rows using legal aligned 64-byte reads, then copies the selected 32-byte
halves into contiguous channel order. Every operand has the same physical
permutation within its dense tile. Because all conv operations are pointwise,
that common permutation is harmless; packed output is already in contiguous
channel order. There is no need to tilize or untilize the dense local vectors.

The compute kernel performs four separate FP32 SFPU multiplications and a
left-associated FP32 sum. It explicitly rounds the sum to BF16 before SiLU,
matching the validated primitive boundary. Native TTNN BF16 SiLU uses
`calculate_silu<false>` and rounds its result to BF16. The candidate calls that
same arithmetic through an SFPU wrapper that retains FP32 DST addressing;
calling `silu_tile<false>` directly would incorrectly select BF16 destination
addressing in this kernel. This distinction follows
`api/compute/compute_kernel_api.h`, the Blackhole `ckernel_sfpu_silu.h`, and
`ttnn/.../eltwise/unary/unary.cpp`. Device own-input comparison is still needed
to verify the combined implementation.

The writer commits output and shifts the already-loaded raw carry/token values
back to the caller's existing carry allocation. Different work items own
disjoint channel ranges. There is no overlap between any writer's range and a
different reader's carry range, regardless of scheduling. The reader cannot
reuse a chunk's input CBs until its carry writes finish.

Compute is the sole consumer that pops input CBs. The writer signals completion
through CB10 after its NoC write barrier, then compute pops inputs and the
completion token. This avoids a subtle invalid scheme where the writer pops
compute's inputs: Blackhole `llk_wait_tiles` uses compute-local `tiles_acked`,
which a dataflow-side pop would not update. All input CBs are one entry deep;
the writer's carry read pointer therefore remains at the sole physical entry.
CB allocation is 24 KiB/core, including token gather scratch and completion CB.

A direct conv+recurrence fusion remains possible, but shared Q/K carry appears
in three value heads. Updating it independently in each head would create a
cross-core read/write race. An implementation must assign complete head groups
to one owner or explicitly coordinate those consumers. The standalone candidate
removes the current multi-op overhead without introducing that race or changing
the recurrence's numerical behavior. Its measured result should determine
whether a more invasive combined kernel is justified.

## Validation and commands

CPU checks passed for B1/B3/B32, three continuation steps, arbitrary task order,
FP32 sum and BF16 boundary bit equality, and exact carry updates. Every tiled
token gather position was checked for all 32 possible users and ten chunks.
Python AST and inline-kernel brace checks passed. These tests do not emulate
device SiLU, compile TT kernels, or measure device performance.

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_conv_decode_cpu.py
```

The following is a device command for the parent stage's serialized device queue;
it has not been run by the source-only helper:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_conv_precision_probe --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/conv_precision_probe.json
```

The probe uses only actual checkpoint conv tap weights, actual TILE device
inputs, B1/B3/B32, random scales 0/.01/1/8, and capture41 TT/HF pre-conv values and
histories. Captured users repeat to fill a batch; random cases have distinct
users. It reports candidate and primitive output versus CPU, candidate versus
primitive, bit mismatch counts, exact carry, and allocation stability. Two
sequential changed-input steps reuse token, carry, and candidate output buffers.
`--require-baseline-exact` optionally asserts bit equality to the primitive
output; normal diagnostic mode records numerical differences without replacing
the existing decoder acceptance gates. Follow with short decoder correctness,
the unchanged 512-step sequence, traced warmed latency, and watcher before any
candidate can replace the incumbent. Compile/API errors are repair work, not
evidence against the optimization.

## V2 API repair after first compile attempt

The parent's `conv63_probe.log` records compilation failure before device
execution: five reader source sites passed only four arguments to
`Noc::async_read`. The header `tt_metal/hw/inc/api/dataflow/noc.h` requires both
source and destination coordinate arguments; only the sixth `NocOptVals` has a
default. `CircularBuffer` destination coordinates accept `offset_bytes`, which
must be zero for these direct full-tile reads.

New immutable files `candidates/conv_decode_sfpu_v2.py` and
`candidates/conv_decode_on57_v2.py` add `{.offset_bytes=0}` as the fifth argument
at the history read and four tap read sites. V1 helper, frozen decoder, and log
remain unchanged. No arithmetic, layout, or synchronization changed. The source
audit `conv_decode_v2_api_audit.json` verifies all six reader transfer sites and
both writer transfer sites have five required arguments. Token gather and
writer source/destination arguments were already correct. Python AST and kernel
brace checks pass; this does not establish device compilation or correctness.

Use the existing probe's `--candidate` flag for the retry:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_conv_precision_probe --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/conv_decode_sfpu_v2.py --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/conv_precision_probe_v2.json
```

`candidates/make_conv_decode_candidate_v2.py` inserts the repaired helper into a
new copy of a later frozen baseline without altering that baseline. The V2
helper SHA256 is
`e8f0a15adb50289637ca0e7d5d9d41f4efdb2edf846f9db6054426589457cb88`.
