# Exact FP32 gate lookup through BF16 embedding

Prepared `tests/gate_embedding_probe.py`; no TTNN import/device execution or decoder edits during preparation. CPU validation passed. This is a diagnostic candidate, not a measured speedup or acceptance waiver.

## Representation and runtime

Setup computes the existing canonical FP32 gate and decay tables, then splits each FP32 word into four numeric bytes. Each byte, 0–255, is exactly representable as BF16. Each encoded table is ROW_MAJOR BF16 `[1,1,65536,128]`; columns `4*h:4*h+4` hold the four bytes for head h. This is **16 MiB per table**, twice the corresponding FP32 table, or 32 MiB for both gate and decay.

The complete runtime adapter accepts BF16 A `[1,S,32]` and performs:

1. BF16→UINT16 bitcast, numeric UINT16→UINT32 cast, flatten to `[1,S*32]`, convert indices to ROW_MAJOR.
2. `ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT)`, producing `[1,S*32,128]`.
3. `ttnn.gather(...,-1,static_columns)`, with setup-uploaded TILE UINT32 `[1,S*32,4]` column indices repeating head columns for each token.
4. Numeric BF16→UINT32 cast; slice four columns, shift bytes 1/2/3 by 8/16/24, combine with three bitwise ORs.
5. UINT32→FP32 bitcast and reshape to `[1,S,32]`.

No host work, uploads, oracle values, or checkpoint reads occur inside this adapter. Gate and decay use the same adapter independently. Decode production needs only decay after g-lookup pruning; the diagnostic still validates both tables. S=128 is supported as a sequence-layout check, using a repeated mix of the nine capture inputs; this is not a full prefill recurrence test.

## Existing native contracts

All references are relative to the repository root:

- `ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp:52–67` requires ROW_MAJOR BF16 weights, UINT32 or BF16 indices, interleaved tensors, and first two weight dimensions equal to one. The adapter uses numeric UINT32 indices, so codes above 256 are never rounded through BF16.
- `embedding/embedding.cpp:43–78,89–101` folds ROW_MAJOR indices into `[B,1,1,L]`, supports TILE output when L and embedding width align to 32, and returns `[B,L,width]`. Both L=32 and L=4096 satisfy the contract.
- `data_movement/gather/device/gather_device_operation.cpp:53–106` accepts UINT16/UINT32 index tensors with matching rank/layout and non-gather dimensions no larger than input. Here both tensors are TILE rank 3; only the final width shrinks 128→4.
- `eltwise/unary/unary_nanobind.cpp:2147–2171` explicitly supports same-width bitcasts BF16↔UINT16 and FP32↔UINT32, with TILE or ROW_MAJOR layout. These are the two bitcast pairs used.
- `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:232–247,280–287` provides numeric BF16→UINT32 and UINT16→UINT32 conversion. The latter preserves all 16 input index bits.
- `eltwise/binary_ng/device/binary_ng_device_operation.cpp:85–91` accepts UINT32 left shifts and UINT32 bitwise OR; Python bindings are in `eltwise/binary/binary_nanobind.cpp:2395–2420`. Shifts are scalar integers 8,16,24, all within the supported range.

The remaining four-byte gather is over a narrow 128-column embedding result, rather than a 65536-column FP32 table. It still costs a gather plus integer reconstruction; source inspection alone cannot establish that this beats the current lookup. Decode reads 32 embedding rows, 8192 BF16 bytes, per table; S=128 reads 1 MiB per table before reconstruction. Full adapter trace timing, rather than embedding-only timing, is therefore necessary.

## Validation and command

`gate_embedding_probe_cpu.json` records **zero FP32 code differences for 65,280 finite BF16 input codes × 32 heads for each table**. Nine canonical capture lookups and the actual embedding/head-column/reassembly geometry for S=1 and S=128 also reconstruct exactly on CPU. Nonfinite outputs caused by finite extreme inputs remain included in code comparisons and are separately excluded from numeric-error metrics.

The root/device owner can run:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.gate_embedding_probe
```

Default device work validates gate and decay against all nine canonical captures, then traces each complete adapter at S=1 and S=128. It opens physical chip 0, one trace at a time, and reports eager, first replay, final replay, and replay-to-replay exact FP32 comparisons with synchronized replay-group timings. The measured region excludes setup/uploads and host conversion. JSON and provenance are written under `doc/fused_decoder/endurance/`; existing reports are protected unless `--overwrite` is explicit.

For the smallest decode-first performance control, add `--decode-only --trace-kinds decay`. Exhaustive code verification is CPU-only; device coverage is explicitly limited to the captures and trace inputs. `--oracle-only` never imports TTNN.
