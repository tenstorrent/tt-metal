# Native SDPA head concat + sigmoid gate candidate

2026-11-13. CPU-only review/fixes and new standalone probe. No device run,
Torch/TTNN import, native build, main integration or accepted-performance claim.
Owned files are `tt/attention_gate_candidate.py`,
`tests/probe_attention_gate.py`, and this audit. Packed precision is unchanged.

## Native contract and corrected helper

The actual non-MLA SDPA output is logical **[1,B,6,256]**, physically
**[1,B,32,256]**, BF16 TILE DRAM. This follows
`SdpaDecodeDeviceOperation::compute_output_specs` in
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:588`:
`num_q_heads = q_shape_unpadded[2]`, then `output_shape[2] = num_q_heads`.
It preserves the six logical Q heads; padding is provided by the tile layout.
The helper now admits this native shape, as well as an explicit logical32
physical view with identical physical contract. It still reads only six heads.
Gate is logical[1,B,1536], physical[1,32,1536], BF16 TILE DRAM.

For each output tile `t` in[0,48), source head is `t//8`, column tile `t%8`.
User `u` addresses source tile `u*8 + t%8`. Its source head row is selected
within a64-byte aligned two-row face read:

```text
aligned = (head//16)*1024 + face*512 + (head%16//2)*64
selected_row_skew = (head%2)*32
output_element_byte = (user//16)*1024 + face*512 + (user%16)*32 + j*2
```

The parent sketch's scratch CB pointer was not explicitly64-byte aligned.
The corrected reader allocates4160 bytes, aligns its private scratch pointer
up to64, and uses `CoreLocalMem` destinations. The64 live source fragments
at maximum B32 require4096 bytes; the64-byte slack covers alignment. Scratch
has65 pages of64 bytes and is owned by the reader only. Integer row copies
assemble output tiles without BF16 arithmetic or a material concat tensor.
Unused head rows6..31 are never read: even head5's two-row read covers4/5.
The reader overwrites output and gate-CB rowsB..31 with zero before compute;
poisoned public-input padding cannot enter live math. Inputs are never modified.

Output is BF16[1,B,1536], physical[1,32,1536], either DRAM or **the exact
canonical eight-storage-core width-sharded L1 `[32,192]`** memory from the
current `o_proj` contract (`multichip_decoder._prepare_matmuls`). The helper
checks equality with `canonical_output_memory(device)`; arbitrary divisors of
48 or a displaced eight-core set no longer silently qualify. Compute workers
are independent of output storage cores. The supplied-output memory is used
when no explicit memory config is passed; aliases and incompatible outputs
are rejected. Signature:

```python
attention_gate(heads, gate, *, output=None, memory_config=None, cores=48)
```

## Native arithmetic and pack boundary

The current baseline in `multichip_decoder._local_compact_attention_decode`
after SDPA is:

1. DRAM heads to per-user height-sharded L1 `[32,256]`.
2. `nlp_concat_heads_decode(...num_heads=6, sub_core_grids=full_grid)`.
3. Width-sharded concat result to DRAM, logical user slice and compact reshape.
4. `multiply(gate, out, input_tensor_a_activations=[SIGMOID])`.
5. Conversion to the canonical width-sharded L1 activation for `o_proj`.

Native binary SFPU code
`operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`
normally packs the activated input to a BF16 CB, reloads it, multiplies and
packs the output. The helper does not select a higher precision approximation:

- `compute_kernel_api.h` binds `sigmoid_tile<VectorMode::RC,false>` to the
  nonapproximate path; the omitted destination template follows
  `fp32_dest_acc_en=False` in the descriptor.
- Blackhole `ckernel_sfpu_sigmoid.h:71–87` explicitly applies
  `convert<vFloat16b>(..., RoundMode::Nearest)` before storing into DST when
  FP32 destination accumulation is off. Thus the native intervening BF16
  pack/reload should preserve the same value already held in the register.
- `ckernel_sfpu_binary.h:221–243` implements `mul_binary_tile<false>` with
  software BF16 RNE and native zero substitution. The helper uses this exact
  native function, then normal `pack_tile(0,2)` with BF16 output.
- Sigmoid init immediately precedes sigmoid evaluation; multiplication init
  occurs afterward so it cannot overwrite sigmoid constants prematurely.

This is a source-based equivalence argument, not hardware proof. Exact native
comparison is required. In particular, preserve the native zero behavior;
Torch sigmoid/multiply is not an interchangeable bitwise oracle.

## Probe and timing boundary

`tests/probe_attention_gate.py` provides:

- Default batches1/3/17/32; `--all-batches` covers every public B1..32.
  Native logical6 is default; `--logical-heads 6 32` also checks the physical
  view accepted by the helper.
- Per-rank/user distinct heads and gates, changed inputs, gate saturation
  extremes and signed-zero/value controls. Exact IEEE NaNs are placed in all
  unused head rows6..31 and gate user rowsB..31.
- Host upload uses `preserve_nan_values=True`; raw complete-tile copies write
  the logical buffers, avoiding padding-clearing transforms. Full physical
  readback must match every input bit, including NaN counts. Input immutability
  and direct-output zero padding are checked before and after queued traces.
- Complete native baseline, fused DRAM+shard and direct canonical-L1 families;
  default worker choices8/48. Every path ends at the same o_proj input memory.
  Native intermediate buffers, all candidate outputs, live/rebound inputs and
  every input bank are allocated before any trace capture.
- Native APIs use supported `output_tensor` arguments. B32's logical slice is
  skipped as its original operation is a no-op; an artificial copy into a
  separate slice buffer would unfairly penalize the baseline. B<32 preserves
  the real native slice operation. No timing includes an extra output copy.
- Eager comparison against native, per-rank/user bit equality and PCC>=0.995,
  rebound inputs, changed-input traces, 50 queued replay sets, persistent output
  addresses, and one deliberate allocation-tracker positive control. Native
  and candidates are all warmed before the first trace.
- Timings only with watcher/profiler unset and allocation tracking off. Each
  sample rotates path order; `cases[].trace_us[path]` contains samples and median
  microseconds for the complete boundary. Input-bank copies, SDPA, gate
  projection and o_proj matmul are outside this isolated boundary. Final
  whole-layer selection/timing must include them and live L1 effects.

The oracle is the current TTNN native boundary, not HF or Torch math. This
probe supplies synthetic SDPA-shaped data; the native-output-shape assertion
is source-anchored. Actual SDPA→candidate integration must still pass the
whole-layer/default gates before this can be selected.

## CPU evidence and resource accounting

`python3 -m models.autoports.qwen_qwen3_6_27b.tests.probe_attention_gate --cpu-check`
passes **811008 address/element checks**, **6144 tile-ownership checks** and
five negative geometry cases. It exhaustively covers B1..32, all live
head/column/face elements, worker partitions1/8/17/48, output storage mapping,
all possible scratch base alignments, and the source-level SDPA/rounding facts.
Python AST parsing and Black checks pass. None of these is device execution.

Each worker has three double-buffered2048-byte tile CBs plus4160-byte scratch:
**16448 bytes/core** (131584 aggregate bytes at8 workers,789504 at48), excluding
kernel metadata/allocator alignment. Canonical output is98304 bytes/device,
12288 bytes on each of eight storage cores. Direct L1 removes the intermediate
98304-byte DRAM output and its following reshard from the candidate family;
its live L1 placement must still be measured with the next matmul/collective.
Input source reads are6144×B bytes across workers (the paired face row is read
twice across adjacent heads); gate and output are each98304 bytes. These are
algorithmic byte counts, not measured throughput or a context-capacity gain.

## Parent-serialized commands

Preserve failures under their original names. No device command below has
been executed by this audit task.

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
TT_METAL_WATCHER=10 bash "$D/run_job.sh" attention_gate_smoke_v1 \
  env -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=1 \
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m \
  models.autoports.qwen_qwen3_6_27b.tests.probe_attention_gate \
  --batches 1 3 --cores 8 48 --no-timing \
  --output "$D/attention_gate_smoke_v1.json"

TT_METAL_WATCHER=10 bash "$D/run_job.sh" attention_gate_wide_v1 \
  env -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=1 \
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m \
  models.autoports.qwen_qwen3_6_27b.tests.probe_attention_gate \
  --all-batches --logical-heads 6 32 --cores 8 48 --no-timing \
  --output "$D/attention_gate_wide_v1.json"

bash "$D/run_job.sh" attention_gate_perf_v1 env \
  -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_attention_gate \
  --batches 1 3 17 32 --cores 8 48 --samples 7 --replays 100 \
  --output "$D/attention_gate_perf_v1.json"
```

Any first descriptor/API/layout error requires adaptation and retry, not a
material-family rejection. Preserve exact arithmetic gates; if register-only
equivalence fails, inspect native pack/reload and reproduce its local BF16 CB
round-trip before evaluating a rejection. No restore to the old height-sharded
head layout is needed by either candidate family.
