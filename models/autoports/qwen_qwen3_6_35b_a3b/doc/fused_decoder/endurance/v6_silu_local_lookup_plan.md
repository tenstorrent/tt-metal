# Local SiLU lookup and final head packing candidate

The complete post-sum tail adapter already passed all 14 tested cases and trace
determinism, but only improved 0.528467567638 ms to 0.507525331341 ms in
`v6_gated_norm_tail_timing_device.json`. This new component replaces the remaining
SiLU lookup, final multiply, and head concatenation. It does not change the frozen
normalization helper or any decoder source. No device execution or performance
claim accompanies this CPU/source hand-off.

## Exact boundary and layout

`tests/v6_silu_local_lookup_probe.py` defines `LocalSiluMultiply`. Its inputs are
the already-BF16-rounded FP32 weighted carrier `[1,32,128]` and the original BF16
Z tensor `[1,1,4096]`, both TILE. It outputs BF16 TILE logical `[1,1,4096]`, padded
`[1,32,4096]`. The output uses 32 width shards of `[32,128]`, assigned row-major
across the device worker grid. This matches the native head concat operation's
output allocation; it does not assume an 8-column physical grid.

Each worker owns one head. Its reader fetches eight 64-byte weighted row segments
and eight 32-byte Z segments, 768 input bytes per worker. Raw BF16 Z payloads
index a local FP32 table. Each core has its own immutable 65,536-entry table,
created from general BF16-domain `torch.nn.functional.silu` during setup only.
There are no captured activations or oracle values in this table. The persistent
cost is 262,144 bytes/core, 8,388,608 bytes across 32 workers. A full decoder must
still demonstrate sufficient L1 capacity with all other allocations resident.

The reader places the lookup values and weighted values in FP32 tile row zero,
initializing all padding to zero. The compute kernel uses direct FP32 unpack,
one SFPU FP32 multiply, explicit BF16 round-to-nearest-even via integer payload
rounding, and BF16 pack directly into the final output shard. This is the same
multiply/round boundary already tested inside `GatedNormTail`; the new data
movement and layout still require hardware verification. Debug mode additionally
packs the untouched FP32 gate into a separate tensor, allowing lookup fidelity
to be checked independently of the BF16 result.

Runtime uses one generic device operation, no torch, from/to-torch, gather,
tilize/untilize, reshard, or host fallback. Setup allocates the table once. The
generic accessor supports the weighted/Z buffer placement; the initial probe
uses interleaved L1 weighted values and interleaved DRAM Z, matching the prior
complete adapter probe. There are no undisclosed placement copies in the timed
component. Integration with a future weighted-only normalization producer is
separate work; the frozen helper currently also computes the final gate.

## CPU evidence and source basis

`v6_silu_local_lookup_cpu.json` is CPU-only preparation evidence:

- 30 fixtures: saved actual positions 128, 129, 130, 137, 181, 3271; eight
  randomized ordinary inputs; 16 blocks covering every BF16 code.
- Actual weighted carriers come from the saved, verified native device
  intermediates in `v6_gated_norm_tail_fusion_device.json.device_tensors.pt`.
  They are component test inputs, not values injected into a decoder run.
- All 4,096 weighted-element face addresses were independently reconstructed
  from a dense coordinate matrix and checked.
- CPU lookup equals direct SiLU for all 65,536 codes (NaN class compared
  separately from payload). Ordinary cases retain all BF16 boundaries.
- AST audit finds no runtime host conversion or tensor layout operation.
- CPU preparation asserts that TTNN has not been imported. No device opened.

Authoritative local source references:

- `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp`:
  native output shape, row-major worker assignment, and width-shard allocation.
- `tt_metal/hw/inc/api/tensor/noc_traits.h`: per-page byte offsets for
  TensorAccessor reads.
- `tests/ttnn/unit_tests/gtests/accessor/kernels/copy_all_pages_abstract_wrapper.cpp`:
  chained accessor compile/runtime offset pattern.
- `tests/v6_ordered_router_fusion_probe.py`: already exercised inline reader,
  NoC, generic-op and direct FP32 unpack API pattern.
- `tests/v6_gated_norm_tail_fusion_probe.py`: already verified SFPU multiply,
  BF16 rounding, packing, and sharded output CB pattern. SHA unchanged:
  `47d50c370b8038bc1b10f24f86c5e58f61222e21b6cc7a1bb22d6318055961bd`.

## Parent-owned hardware command

Frozen candidate SHA-256:
`ac22bfa650d716e1f8dda856c0f042be82c13d5fd024de3040e80bf2806df563`.
Source and both kernel strings are snapshotted beside the CPU report. No
previously measured file was modified.

From the checkout, after normal device ownership/environment checks:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_silu_local_lookup_probe.py --run-device --time --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_silu_local_lookup_device.json
```

`--time` requires watcher/profiler off. Omit it for a separate correctness run.
Timing uses five dispatch warmups, five trace warmups, and five samples of 100
replays, comparing the complete existing lookup/multiply/concat component with
the local-table candidate. Both include runtime input handling. Debug output is
disabled for timing. First and last replay output must match baseline and each
other exactly. This timing excludes the preceding normalization and subsequent
projection and cannot substitute for full decoder timing.

The device probe records every exhaustive block, including subnormal and
nonfinite behavior. Ordinary actual/random cases must be bitexact. Full-code
semantic exactness is a separate explicit report field; NaN payload differences
are separately counted. An unexpected full-domain difference needs inspection,
not an undocumented acceptance waiver. Full decoder gates and context capacity
remain unchanged. No speedup, hardware compilation, capacity preservation, or
stage acceptance is claimed from CPU setup alone.

## Compile correction after first device attempt

The parent-owned first attempt closed the device cleanly and failed compilation
before correctness or timing. The preserved
`v6_silu_local_lookup_device.json` and `.log` identify an ambiguous conversion
from the SFPI destination-register proxy to `sfpi::impl_::vVal` in
`sfpi::as<sfpi::vUInt>(sfpi::dst_reg[0])`. The original source snapshot still has
SHA `ac22bfa650d716e1f8dda856c0f042be82c13d5fd024de3040e80bf2806df563`.

The only script change loads that register into an explicitly typed
`sfpi::vFloat product` before calling `sfpi::as<sfpi::vUInt>(product)`. This
matches the typed-load pattern in the verified normalization helper; it does
not change the stored-product rounding boundary, table, or data movement.
Revised frozen script SHA:
`43d20b3fabec4765eb237a2a3156fc68b2b1a92a48612b155377dabf7f0faef5`.
Python compilation and CPU setup passed again, including the same 30 cases,
65,536 table codes and 4,096 face addresses. New evidence and source/kernel
snapshots are `v6_silu_local_lookup_compilefix_cpu.json*`. Device compilation
and execution still require the parent's retry; this CPU check does not claim
them. No measured helper or previous evidence was altered.

Use a fresh result path for the parent-owned retry:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_silu_local_lookup_probe.py --run-device --time --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_silu_local_lookup_compilefix_device.json
```

## DRAM half-row alignment correction

The compile-corrected device attempt ran all 30 cases and closed cleanly, then
failed the unchanged ordinary-case bitexact assertion. Preserved report
`v6_silu_local_lookup_compilefix_device.json` has exactly 2,048 gate mismatches
in every case. Every ordinary actual/random case also has exactly 2,048 final
output mismatches. Because the untouched debug gate already differs, this is
upstream of the final multiply and BF16 rounding. The invariant half-tensor
pattern points to a row-half transfer/layout problem, rather than a small
arithmetic approximation.

Source inspection identified an invalid NoC read: the second Z half-row used
DRAM source offset 512 and local scratch destination offset 32. Blackhole
`noc_parameters.h:379` sets `NOC_DRAM_READ_ALIGNMENT_BYTES` to 64.
`internal/debug/sanitize.h:746` checks that source and local destination address
bits under that alignment mask agree. The old half1 transaction disagrees by
32 bytes; half0 agrees for the observed aligned buffer bases. This is a
source-proven bug and matches the half-output failure pattern, but the device
retry must still confirm that it explains all mismatches.

The new reader reserves separate 64-byte-aligned slots for its two 32-byte
half-rows. A bias computed from the actual CB write pointer ensures matching
low six address bits even for a CB base with only 16-byte L1 alignment. The LUT
index load follows the two-slot scratch layout. Weighted addresses, table,
compute source, product rounding, and output allocation are unchanged.

Revised frozen script SHA:
`10d1206e9686bee0dd7515a27be1b305aa67e3224d0ac8fa46c53cf3cf328e24`.
`v6_silu_local_lookup_aligned_cpu.json` adds 1,024 transaction-alignment checks
and raw-code recovery checks spanning all heads/tiles and all four possible
16-byte-aligned CB-base residues modulo 64. The previous full table, 30-case
setup, and 4,096 weighted-face checks still pass; Python compilation passes.
No hardware was used for this correction.

The next device run also saves gate/output/native tensors for positions 128,
3271 and the first exhaustive block before evaluating the existing assertion,
so any remaining discrepancy can be localized without another instrumentation
round. Neither the ordinary bitexact assertion nor the exhaustive reporting
criteria changed. Parent-owned retry, with a new evidence path:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_silu_local_lookup_probe.py --run-device --time --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_silu_local_lookup_aligned_device.json
```
