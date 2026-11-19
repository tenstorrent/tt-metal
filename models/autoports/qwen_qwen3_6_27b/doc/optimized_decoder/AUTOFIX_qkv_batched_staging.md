# Compact full-attention QKV split: batch aligned staging across heads

Source-only candidate, based on the passing norm153 topology frozen as
`candidates/norm_l1_handoff_v3_on134.py`. Parent profile evidence places native
NLPCreateQKVHeadsDecode near 31 us at B32. No device execution, speedup, or
device bit-equivalence is claimed by this preparation.

## Native operation audit

Native interleaved factory:
`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/`
`device/nlp_create_qkv_heads_decode_interleaved_program_factory.cpp`.

Its kernel is `device/kernels/`
`reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp`.
For BF16 DRAM on Blackhole, a 16-element face row is 32 bytes, smaller than the
64-byte DRAM alignment. The aligned path reads 64 bytes into scratch and copies
the selected 32 bytes into the output. It does **not** read whole input tiles.
Two data-movement RISCs handle the two horizontal faces independently on each
logical user's output core.

Scratch holds only one head's eight channel-tile slices per RISC. For each of
24 Q + 4 K + 4 V heads, the kernel:

1. Issues eight aligned DRAM reads, then waits at a read barrier.
2. Copies eight 32-byte local slices through `tt_memmove`'s NoC read path,
   then waits at a second read barrier before reusing scratch.

That is 64 inner barriers plus a final read barrier per RISC. There is no
public program config for enlarging staging or grouping heads. The native
splitter's output shape logic is in
`nlp_create_qkv_heads_decode_device_operation.cpp:113-195`: Q has logical
24 heads, K/V four; each is padded to32 heads with one `[32,256]` HEIGHT_SHARDED
L1 shard per user. Full-attention callers use the first B cores in an
**eight-column row-major** grid. This ordering matters independently of the
device's 11-column compute grid.

## Candidate topology

`candidates/qkv_split_batched_staging.py` is one decoder-local generic operation
with two data-movement kernels and no compute kernel:

- Each RISC stages all 256 channel tiles (32 heads * 8 tiles/head) into unique
  64-byte scratch slots before one read barrier.
- It then scatters all 256 selected 32-byte slices before a second read barrier.
- Source/destination bit patterns, channel/head ordering, two-face ownership
  and native output MemoryConfig remain unchanged.
- Odd users read the upper 32-byte half of their aligned scratch slot; even
  users read the lower half. No illegal 32-byte DRAM reads are introduced.

Scratch increases from native 576 to 16,448 bytes per RISC, including 64 bytes
for explicit alignment. Each user core uses 32,896 bytes total scratch plus
the existing three 16 KiB output shards. All input and output tensors are
registered with generic_op; runtime addresses come from those tensors, not
process-global constants. No torch/from_torch/to_torch or arithmetic appears
in the measured helper.

At B32, DRAM read volume remains 1,048,576 bytes (twofold row overfetch), and
local scatter volume remains 524,288 bytes. The hypothesis is less serialized
barrier/scratch reuse, not fewer DRAM bytes. Two-RISC NoC scheduling, scratch
pressure and dispatch cost can still negate the benefit; whole-path timing
decides selection.

## Padding contract

The native kernel writes only logical Q/K/V heads into newly allocated empty
buffers; it does not initialize padded head rows. Those physical native
padding bits are unspecified and cannot be required to match fresh candidate
allocations. Parent agreed to strict **logical** bit equality, exact native
MemoryConfig/core order, and deterministic zero/finite candidate padding.

The candidate explicitly zeros only padded output rows. The two RISCs own
disjoint horizontal faces. Every zero operation is completed through
`write_zeros_l1_barrier` before ordinary NoC operations; no zero-mode writes
overlap logical scatter destinations. Padding zero traffic is 32 KiB/user and
must be included in actual measured candidate latency. No downstream norm,
rotary, paged-cache update or SDPA operation changes.

## Artifacts and source validation

- Helper: `candidates/qkv_split_batched_staging.py`.
- Frozen decoder: `candidates/qkv_batched_staging_on_norm153.py`.
- Generator: `candidates/make_qkv_batched_staging_candidate.py`; replaces only
  the compact full-attention native splitter call, then appends the helper.
- CPU checker: `candidates/check_qkv_batched_staging_cpu.py`.
- CPU results: `candidates/qkv_batched_staging_cpu.json`.
- Device probe: `tests/run_qkv_batched_staging_probe.py` (**not run here**).

CPU/source and syntax checks passed. The CPU checker uses arbitrary 16-bit
element patterns to verify source aligned reads, selected half rows, native
tile/face addressing and output scatter. For B1/3/31/32, every physical output
element is written exactly once, logical Q/K/V match the input bitwise, and
all padded rows are zero. CoreRangeSet uses the native rectangle-plus-tail
decomposition, because set equality compares ranges rather than only covered
coordinates. This also avoids the previously observed 8x4 core-order issue.

## Recommended device checks

Use the parent's authorized serialized device queue:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_qkv_batched_staging_probe \
  --batch 1 3 31 32 --repeats 3 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/qkv_staging_probe.json
```

The probe checks strict logical equality to both native output and direct CPU
splitting, native MemoryConfig/logical/padded shapes, finite zero candidate
padding, and three changed-input/allocation runs. It records harness, helper
and native-reader SHAs. Native padding nonzero counts are reported without an
equality requirement. Kernel/API errors require diagnosis and repair, not
automatic rejection of the topology.

After this passes, run the frozen full decoder through short correctness,
long/repeated decode, changed-input traced replay, watcher and warmed traced
B32 whole-path latency. A native/candidate kernel-only profile helps explain
results but does not substitute for a whole-decoder win. Integration into a
newer combined candidate is a one-method-call replacement through the generator;
the existing frozen base and shared TTNN sources remain untouched.
