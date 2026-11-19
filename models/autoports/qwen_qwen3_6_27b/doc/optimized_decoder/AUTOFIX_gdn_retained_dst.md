# Retained FP32 state in full DST

Source-only candidate: `candidates/gdn_sfpu_retained_dst.py`, 2026-11-12. This implements the proposed four-state-tile schedule without changing state dtype. No hardware or C++ compilation was performed by this helper.

## Legality and register allocation

Blackhole `hw/inc/internal/tt-1xx/blackhole/tensix_types.h:191` defines full DEST as1024 face rows. `tt-llk/tt_llk_blackhole/common/inc/ckernel.h:859` divides capacity by two for FP32 and by64 rows per32x32 tile, yielding **eight FP32 tiles in full-sync mode**, versus four in half-sync. `ComputeConfigDescriptor` exposes `dst_full_sync_en`, its Python constructor accepts it, and `impl/program/program.cpp:584` passes it into the kernel config. The candidate sets it to true and includes a compile-time capacity assertion.

| DST index | Lifetime |
|---|---|
|0..3|Four K-row state tiles for one32-wide value-column, retained through decay/update/readout|
|4|Temporary prediction/readout product or expanded delta row|
|5|Q/K column tile|
|6|Prediction accumulator, then delta row, then output accumulator|
|7|Decay or beta scalar|

`api/compute/copy_dest_values.h` supplies `copy_dest_values<DataFormat::Float32>` for precise in-DST copies. Its Blackhole implementation copies using FP32 SFPU load/store rather than FPU math. `api/compute/pack.h` supports arbitrary source DST indices and explicit CB tile positions. The candidate reserves16 updated-state tiles and4 output tiles before the column loop, packs each column's four retained states and reduced output in one commit/wait/release, and publishes both CBs only after all four columns finish.

## Arithmetic and synchronization

Each column first decays its four FP32 state tiles. K contraction copies those retained tiles into work/accumulator registers, multiplies by K, folds the four tile products in the original order, then applies the same SFPU column reduction. V-minus-prediction and beta scaling produce the delta row in DST6. Each update copies that row into DST4, replicates row0, and performs a separate FP32 multiply and add directly into its retained state register. Q contraction uses those updated states with the original accumulation order.

Prediction and delta padding may remain unspecified inside DST; all consumers use their valid first row. Final output packing enables the reduction edge mask so public output padding remains zero. State packing occurs with the mask cleared. The existing writer waits for both state and output publication, preserving in-place state ownership and trace allocation behavior.

The normalized path drops from53 to29 tile packs/head: the16 decayed-state packs,4 prediction packs, and4 delta packs disappear. It also removes the corresponding CB reads and the updated-state reread for output contraction. Full synchronization reduces math/pack overlap; the replacement DST copy traffic and this synchronization cost must be assessed by actual latency. The candidate is a hypothesis, not a claimed improvement.

The standalone helper retains the original tile and direct row-major intake interfaces, including optional packed QKV. Its reader and writer are unchanged from the packed-row-major helper. To integrate into the selected decoder, use its compute string and set the compute descriptor's full-sync flag; keep explicit direct FP32 unpack enabled. Unused CB8/9/10 are reduced to one tile each in the standalone host descriptor.

## Validation and follow-up

Python AST, inline-brace, full-sync configuration, and source checks passed. A host NumPy float32 comparison verified bit-equal state and output between the baseline full-head schedule and retained-column schedule for random, zero-K/V, and zero-initial-state cases, with the same four-tile fold order. This validates the arithmetic schedule only; it does not validate kernel compilation, DST instructions, or device synchronization.

Required device gates are the isolated own-input probe, repeated/long sequence, trace changed-input checks, watcher, and full-decoder latency against the best correct candidate. Compile or first-run errors should be diagnosed before rejecting the schedule.

A separate future candidate may round the persisted state to BF16 **after computing the current output from unrounded FP32 state**, matching the HF storage boundary. It could pack updated state into BF16 CB19, unpack that CB and pack FP32 CB20 for the unchanged public state allocation. CB19 must use its appropriate BF16 unpack mode and CB20 direct FP32 unpack; publication must wait for the rounded-state buffer. This would remove external DRAM typecast round trips, but adds local pack/unpack work and is intentionally absent from this FP32 candidate. Direct BF16 public storage could halve recurrent-state traffic, but changes public state dtype, accuracy, and capacity contracts and needs separate evidence.

## Retained33 performance finding and targeted correction

The stage owner reports retained33 passes the isolated own-input probe at approximately1.5e-7 output and6e-8 state relative error, but short full-decoder latency regresses from boundary32's2.29294ms to2.43152ms. Fewer packs did not establish a speedup.

Source comparison identifies48 added full-tile SFPU copies/head:32 copies from retained state into K/Q contraction work registers and16 copies of delta before row expansion. Each copy traverses all1024 FP32 elements on MATH and is followed by another SFPU traversal. This replaces baseline unpack/pack traffic, which can overlap with MATH, with serialized SFPU movement. Full-sync mode also removes the original half-DST math/pack overlap. These are source-supported overheads; their individual latency shares have not been measured.

`candidates/gdn_sfpu_retained_no_copy.py` removes all48 copies without changing arithmetic or synchronization. The existing native `_calculate_sfpu_binary_bcast_full_tile_` accepts independent source and destination DST indices; its column implementation already uses separate data/output addresses. A stage-local wrapper therefore multiplies retained state directly into work/accumulator. The existing `_gdn_copy_row_` likewise accepts separate row-source/output indices, so delta6 expands directly into work4. No retained state is modified by contractions, and the product, fold, reduction, and update rounding points are unchanged.

The new helper has no `copy_dest_values` call or include. Python AST validation passed; reader, writer, config, and CB scheduling remain unchanged from the first retained candidate. Own-input correctness, repeated sequence, and latency measurements are pending. If it remains slower, half-sync column streaming or pack/math overlap should be investigated separately rather than assuming fewer CB operations imply a better schedule.
