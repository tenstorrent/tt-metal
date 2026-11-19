# Direct prefill projection consumers and native gate stride

The smaller native gate operand is legal under the current source contract. This investigation prepared a model-local snapshot and diagnostic without running hardware or editing production/core harnesses. Base is `../selected_fused_decoder_v3.py.snapshot`, SHA256 `73ba9aabb10a2c34f3b10b5ff1fc93418470c0a624dd863e87a8c8c2af2901ba`. Candidate is `v3_direct_prefill_consumers.py.snapshot`; its adjacent patch and provenance bind the exact change.

## Native contract and index identity

`sigmoid_gated_rms_norm_device_operation.cpp:69–88` requires a rank-three interleaved tiled BF16 gate tensor `[B,T,W]`, matching sequence and batch, with tile-aligned `gate_column_offset` and `W >= offset + heads*value_dim`. It does not require the model's full Q/K/V/Z/A/B packing. For 32 heads and V128, `[1,S,6144]` with offset2048 meets `6144 = 2048 + 32*128`. Existing native physical sequence alignment is unchanged; the public planner still handles non-aligned logical lengths.

`sigmoid_gated_rms_norm_program_factory.cpp:118–119` derives `gate_row_tiles` from the actual operand's padded width and `gate_offset_tiles` from the supplied offset. The reader at `device/kernels/dataflow/reader_sigmoid_gated_rms_norm.cpp:54–55` addresses each head's gate as:

```
gate_base = (batch * sequence_tiles + token_tile) * gate_row_tiles
            + gate_offset_tiles + head * value_tiles
```

Old width/offset are 386/256 tiles; new width/offset are 192/64 tiles. Both point to the same Z coefficients for every token tile, head, row and lane. The native operation reads only those gate tiles. Core input, checkpoint gamma, epsilon, output BF16 dtype, compute config, and final Z multiplication remain unchanged. There is no nonlinear input-activation fusion or relocated BF16 arithmetic boundary.

State projection is `[K2048,V4096,A32,B32]`; output projection is `[Q2048,Z4096]`. Both decode and prefill now obtain:

- QKV from `concat(output_projected[:2048], state_projected[:6144])`.
- A/B from state columns6144:6176 and6176:6208.
- Z from output columns2048:6144.

Prefill passes `output_projected` directly to native gated norm with offset2048. Decode is unchanged. This removes three slicing calls and narrows the consumer concat from12352 to8192 columns. At S256 the concat output payload decreases by2,129,920 bytes; this is an allocation/write-size difference, not measured DRAM traffic or latency. The native gate read width remains4096 useful columns, with a different row stride. Output projection, exact state-producing projection, convolution, cache update and public planning are untouched.

## Actual-source probe

`tests/v3_direct_prefill_probe.py` extracts the consumer block and the actual native-gated-norm/final-multiply statements from each immutable snapshot's AST. It records source and extracted-AST hashes. The probe does not replace these with a second hand-written TTNN implementation.

CPU-only preparation passes exact unique-INT32 label mappings for S1/17/32/33/128/129/256, including equality of the old/new native gate-column ranges (`v3_direct_prefill_cpu.json`). It loads the real layer0 checkpoint gamma only; no full model or TTNN import is needed in `--oracle-only` mode.

The device probe uses deterministic finite BF16 synthetic state/output projection operands and FP32 core tensors at actual dimensions, plus real checkpoint gamma. These are synthetic adapter inputs, not a claim of captured model activations. It compares all BF16 output codes for QKV/A/B/Z, native gated norm and final merged output. Non-aligned rows check consumer mapping only; aligned S32/128/256 exercise native norm. Timing captures the complete extracted consumer concat/slices, norm and final multiply, including allocation/layout costs. It excludes unchanged upstream projections, recurrence, core permutation and output matmul. First and last trace outputs must equal the baseline. Code differences are recorded and skip timing rather than being waived.

The script writes provenance, protects existing output, uses physical chip0 exclusively, disables watcher/profiler/tracking for timing, uses explicit HiFi4/nonapproximate/FP32-destination compute configuration, and closes the mesh in `finally`. No device result is claimed yet.

Root-owned device command:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v3_direct_prefill_probe
```

After exactness and complete-adapter timing, use the standard whole-decoder snapshot benchmark with a fresh output:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_direct_prefill_consumers.py.snapshot --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_direct_prefill_consumers_s256.json
```

Fewer dispatches alone do not justify retention. A faster full path must pass unchanged seeded stream, arbitrary-tail, context, watcher and final-source gates. The prior decode-only memo described the old prefill operand arrangement; it did not establish an impossibility of using this smaller legal native gate tensor.

## Completed isolated device control

The serialized hardware owner produced [v3_direct_prefill_probe.json](v3_direct_prefill_probe.json). All seven logical consumer cases pass exact comparisons. Native gated normalization and merged output are also exact for aligned S32/128/256; the non-aligned cases check consumer mapping only. Timing includes the complete packing/slicing/native-normalization/final-multiply adapter, excluding projection, recurrence and output matmul. The device closed successfully. Production remains qualified v3; these microbenchmarks do not select a replacement or establish full-decoder speed. Whole-decoder comparisons are in progress.

| Case | Baseline median ms | Candidate median ms |
|---|---:|---:|
| S32 | 0.061813 | 0.048527 |
| S128 | 0.105541 | 0.080227 |
| S256 | 0.171569 | 0.124035 |
