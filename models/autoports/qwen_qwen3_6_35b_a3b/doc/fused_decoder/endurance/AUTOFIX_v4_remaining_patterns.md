# Remaining exact graph controls after serial L1 prefill

This is source/CPU-only investigation. Production is still the qualified v3 implementation. The parent measured the serial-L1 precursor `v3_prefill_serial_l1.py.snapshot` (`b8f0bb72…`) at S256 118.354758 ms prefill / 1.352486 ms decode and S4097 2278.039810 / 1.351580 ms, with unchanged PCC. These are candidate measurements, not production selection or final qualification. The new controls below use `v4_serial_l1_compact.py.snapshot`, SHA256 `deb1fa8f3fb3020bfb0f983cc04c43e45f96f245535e0e8e7b5741fe417cf4ad`, which removes the two unused legacy projection weight uploads and their CPU padding. No arithmetic relaxation, native modification or device execution occurred in this investigation.

## First candidate: move input untilization outside the token loop

The serial multirow projection currently calls its singleton path on `x[:, row:row+1, :]`. Native `slice/slice.cpp:224–235` selects the row-major path for a non-tile-aligned starting row; at line382 it converts the input tensor to RM before slicing. Thus most token slices can untilize the complete normalized input independently. The singleton then immediately requires RM coefficients for the pair32 gather anyway.

`v4_serial_input_rm.py.snapshot` converts the complete input to DRAM RM once before the loop. Every row slice then stays RM into the unchanged singleton method. Its existing `to_memory_config(...,DRAM)` and `to_layout(...,RM)` are no-ops for these operands. `core/to_layout/to_layout_op.cpp:62–82` returns the input directly when layouts match; it does not launch another conversion. This removes repeated full-input conversions without changing element order, gather indices, K1024 prefill policy, products, scans, block sums or rounding.

The conversion explicitly requests DRAM for a TILE input. Native `to_layout` intentionally ignores a requested memory change when the input is already RM; the actual measured caller supplies TILE normalized input. The singleton still explicitly places any RM input in DRAM, so this nuance does not create an implicit L1 residency requirement. Tests must include non-aligned lengths and rows, rather than timing only row0.

## Second candidate: write compact RM rows directly to DRAM

The current loop copies each padded TILE singleton result from L1 to DRAM and retains it. `concat/concat.cpp:96–143` subsequently sees H1 logical / H32 padded concat operands, untilizes each with unpadding, concatenates RM rows and tilizes the result. The TILE DRAM copy is therefore an intermediate representation, not a required consumer contract.

`v4_serial_output_rm.py.snapshot` replaces that copy with `to_layout(projected,ROW_MAJOR_LAYOUT,memory_config=DRAM)`. Native `to_layout_op.cpp:169–182` derives the unpadding endpoint from logical shape and forwards the requested memory configuration into `untilize_with_unpadding`. Both source and destination are interleaved; `untilize_with_unpadding_device_operation.cpp` accepts that geometry. The operation keeps the original dtype. The loop retains compact RM rows, performs native RM concat in DRAM, then explicitly tilizes once to the unchanged public `[1,S,6208]` TILE output.

For BF16 each retained row is 12,416 useful bytes instead of the 397,312-byte H32 TILE allocation; for FP32 it is 24,832 instead of 794,624 bytes. This does not remove the per-row untilize that native concat already required. It combines that necessary layout conversion with the DRAM placement, removing the preceding padded TILE copy and reducing retained allocation. Native interleaved concat already batches above47 inputs (`concat/device/concat_device_operation.cpp:411–483`); a model-side concat tree adds no legality benefit. The final tilize and all native concat batches remain inside the timed invocation.

`v4_serial_both_rm.py.snapshot` combines these two independent placements. The three snapshots have adjacent exact patches and provenance. Their SHA256 values are:

| Candidate | SHA256 |
|---|---|
| Input RM hoist | `459e4a6652d22220c8ec5feac84a78b5462d27fa2d07388e2f62c6e0edf47010` |
| Compact RM output rows | `346db81099eab62d7c2cd574ae214dff0b7380b530c742b0c10014f78932e230` |
| Both | `0ab8e6e99641c344782ad70ae438cb85614aa86e4f5050bd7294004766a3ae67` |

## Norm and other apparent shortcuts

The final four-group/eight-lane ordered norm reduction already has a device-exact nine-case control in `v3_norm_last_reduce_device.json`. Its complete multirow norm improves, while singleton changes are small and inconsistent. That is an existing candidate for a later combined control, not a new untried pattern or an established decoder win. Conditional native norm identity-cast removal is already present in this serial base. Whole-norm L1 placement was previously slower; repeating that unchanged control is not justified by the projection placement change alone.

The remaining exact-norm split-product correction uses high/low operands more than once. Folding a round/widen chain into a single consumer would not remove the materialized high value needed by the others. Replacing `inverse + inverse*residual` with addcmul or MAC would change separate-product/add rounding, and prior device evidence establishes that SFPU MAD does not provide the assumed IEEE residual precision. Reassociating the final `input*inverse` then `*gamma` also changes the required FP32 boundary. These are precision-contract blockers to blind dispatch folding, not claims that all mathematically equivalent formulations have been benchmarked. The existing scalar and reciprocal activation chains already fold the safe scalar operations.

A preallocated final projection matrix with row scatter would be another structural possibility, but it is not prepared here: a valid native row-write operation and its output/capture ownership must first be established. `ttnn.copy` on a sliced tensor is not proof of a writable view into the original allocation. Existing concat handles non-aligned logical rows and is the source-backed first control.

## Dedicated actual-method probe

`tests/v4_serial_rm_probe.py` preserves the previous probe unchanged. It accepts explicit baseline/candidate snapshot paths, verifies their independent hashes and the candidate's base hash, and asserts that only `linear` changed and its singleton branch is AST-identical. Setup loads only real checkpoint K/V/A/B weights and the compact scan view for both sides. There is no 776 MiB baseline allocation in this comparison.

Coverage is singleton K512 and K1024 plus S2/32/33/128/129 K1024, at both FP32 and BF16 output boundaries, all6208 columns. It calls the actual snapshot methods and checks logical shape, dtype and TILE layout, exact output codes, first/last traced replay and post-eager outputs. Complete eager and traced timings include input conversion, every row slice, gather, singleton projection, output placement/conversion, concat and final tilize. Checkpoint setup and host inspection stay outside measured execution. The probe writes source provenance, refuses accidental report overwrite, opens only physical chip0 when invoked by the hardware owner, and closes in `finally`.

CPU-only preparation validates source scope and fixture arithmetic; it does not establish native adapter correctness or speed. The combined CPU artifact is `v4_serial_both_rm_cpu.json`. No hardware was run by this agent.

Parent-only device commands, serialized with other device work:

```bash
for variant in input output both; do
  HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_serial_rm_probe \
    --baseline-snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_serial_l1_compact.py.snapshot \
    --candidate-snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_serial_${variant}_rm.py.snapshot \
    --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_serial_${variant}_rm_probe.json
done
```

Only exact, faster complete adapters should advance to source-bound whole-decoder S256/S4097 comparisons, endurance and the unchanged final gates. The Stop Rule remains unmet; driver ownership time is separate from active kernel time and does not establish a 70% roofline result.

## Parent device results for RM placement

All three parent-run reports pass all seven cases at FP32/BF16 boundaries and all first/last replay checks. These are complete actual linear-adapter measurements, not whole-decoder results; no replacement is selected here.

| Report | SHA256 | Status |
|---|---|---|
| [v4_serial_input_rm_probe.json](v4_serial_input_rm_probe.json) | `2ef9eadcc5f9aea5d169d29dca72f9fe52e91875de150cfa276cd113c0ce3554` | exact |
| [v4_serial_output_rm_probe.json](v4_serial_output_rm_probe.json) | `8f4ff91e86d05604f04d138188a197c5a6ffa09e6655f444ac860e96be25c2b1` | exact |
| [v4_serial_both_rm_probe.json](v4_serial_both_rm_probe.json) | `d1f4829c04b3c76ecfdeb2fd9d8647bdda9fbc34ff387596a41ad1c43d56a98a` | exact |

Combined RM complete eager S128 improves 32.972 → 30.828 ms and S129 improves 35.137 → 31.063 ms. Whole-decoder benchmarks remain separate pending evidence. All input slicing, singleton work, output conversion/placement and final concat/tilize are included.

## Subsequent parent whole-decoder RM measurements

These source-bound candidate benchmarks remain distinct from production qualification.

| Length | Compact serial PF ms | Both RM PF ms | Compact decode ms | Both RM decode ms |
|---|---:|---:|---:|---:|
| 256 | 124.255996 | 110.207670 | 1.351969 | 1.351809 |
| 4097 | 2284.371728 | 1888.362174 | 1.353217 | 1.353708 |

Artifacts: [compact S256](v4_serial_l1_compact_s256.json), [both RM S256](v4_serial_both_rm_s256.json), [compact S4097](v4_serial_l1_compact_s4097.json), [both RM S4097](v4_serial_both_rm_s4097.json), each with adjacent provenance. PCC is unchanged. The changes are prefill-only; small decode timing differences are not attributed to a decode rewrite.
