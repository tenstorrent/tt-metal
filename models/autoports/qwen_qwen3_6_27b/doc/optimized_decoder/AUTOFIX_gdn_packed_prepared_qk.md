# Reduce serialized Q/K unpacking

The expanded-Q/K candidate passed 54 exact output/state cases, but B32 whole
latency improved only from 1.818283 to 1.809059 ms. This does not support a large
gain from broadcast instruction removal alone. Profile140 measures the accepted
recurrence at 498.673 us; TRISC durations include waits and cannot separate math
work from unpack/pack synchronization. No 130 us gain is asserted for this new
candidate.

## Source-supported hypothesis

The accepted recurrence performs 48 prepared-Q/K `copy_tile` calls per head:
16 prediction K loads, 16 update K loads, and 16 readout Q loads. Each source tile
is FP32. Blackhole's `llk_math_eltwise_unary_datacopy.h` synchronizes each direct
FP32 unpack with `math_unpack_to_dest_math_ready()` and
`math_unpack_to_dest_tile_ready()`, then clears four face zero flags for the
hardware workaround. `llk_unpack_common_api.h` explicitly documents that this is
a per-tile handshake rather than a section-level asynchronous unpack pipeline.
These repeated full-tile transfers also serialize with the SFPU operations that
consume their DST values.

Prepared K and Q each contain only 128 distinct values, yet their current
expanded representation uses four 1024-value FP32 tiles apiece. Each four-row
SFPU band has four separate eight-column slots at raw offsets 0, 2, 16 and 18.
The candidate packs the four 32-row segments into those slots in one tile per
vector. Each slot retains the same already-broadcast eight-lane values as the
passing expanded helper. Selection requires only an address offset, with no
new shuffle or floating operation in the recurrence.

For each value-column traversal, one K tile loads into DST5 before prediction
and remains intact through the update. After beta is consumed, one Q tile loads
into DST7 for readout. The four ordered m iterations select the appropriate
eight-column slot. This reduces Q/K direct-unpack calls from 48 to eight per
head and associated L1-to-DST bytes from 192 to 32 KiB. Preparation also packs
one tile rather than four per vector. Compulsory DRAM state traffic is unchanged;
this targets local transfers and synchronization. Actual latency impact depends
on whether those transfers are on the critical path.

## Exactness and ownership

Preparation retains the existing FP32 normalization, transpose and native
broadcast sequence. It transposes each source segment into DST0, then writes
its eight-column slot into DST1. The four slots are disjoint and cover the full
output tile before packing. `transpose_tile` writes only its requested DST0;
DST1 remains acquired until all four segments are assembled.

The original matrix arithmetic/store instruction order, two reduction sites,
and state persistence rounding are unchanged. DST5 survives delta construction;
DST7's beta is dead before Q replaces it. CB13 and CB14 shrink from four FP32
tiles each to one; reserve, push, wait and pop counts change together. Raw input
CBs remain four tiles and grouped duplicate-input consumption is unchanged.
Reader, writer, partitioning, state buffers and synchronization are unchanged.

## Frozen artifacts and checks

- `candidates/gdn_sfpu_packed_prepared_qk.py`
- `candidates/integrated_packed_prepared_qk134.py`
- `candidates/make_gdn_packed_prepared_qk_candidate.py`
- `candidates/check_gdn_packed_prepared_qk_cpu.py`
- `candidates/gdn_packed_prepared_qk_cpu.json`

Source/CPU checks passed: unchanged reader/writer, unchanged arithmetic/store
sequence, unchanged persistence suffix, host changes limited to CB13/14 sizes,
and exact lane-placement comparisons for both vectors over six finite scales,
including signed zero and subnormals. Python compilation passed. Device
compilation, timing and end-to-end accuracy remain unverified. No hardware or
main/shared source was touched.

Parent-serialized next check: run the existing
`tests/run_recurrence_exact_comparison.py` with candidate
`gdn_sfpu_packed_prepared_qk.py`, reference `gdn_sfpu_update_readout.py`, and
`--state-dtypes bf16 fp32`. Require all 54 cases to match output/state bits,
then test B32 short/512 and B1 short with the whole candidate. Measure recurrence
device latency to distinguish a real unpack benefit from whole-run noise.
