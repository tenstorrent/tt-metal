# Single-dispatch corrected matmul diagnostic

`tests/v6_fused_corrected_matmul_probe.py` is a new, isolated model-local probe. The measured helpers, decoder snapshots and production source remain frozen. The builder agent performed only standard-library source/shape checks; no Torch/TTNN import, device opening, kernel compilation, measured speed, integrated correctness or context-capacity claim is made here.

Probe SHA256: `845c959110a713f808d1475eb22ded5ffa4e6ddd8209f02170fb90ed02b353dc`.
Source evidence: `v6_fused_corrected_matmul_source.json` and its `.source.py.snapshot`, `.reader.cpp.snapshot`, `.compute.cpp.snapshot`, `.writer.cpp.snapshot` siblings. Arithmetic graph SHA256: `0c68773db4d564c119477646937afa28856dd0f279f0d223feeed226d869ff6a`.

## Arithmetic and kernel boundary

The frozen `CorrectedMatmul` performs increasing-K calls to `CorrectedFma`, each after device slicing, identical-branch WHERE broadcasting and L1 sharding. The prototype expresses the same increasing-K recurrence inside one generic operation. It retains the entire frozen correction graph verbatim, including the signed-zero correction and full FP32 DST scratch allocation. The initial accumulator is positive-zero bits, matching `ttnn.zeros`.

The first candidate deliberately retains every FP32 pack/unpack boundary. Interim tile7 results are packed into a local feedback CB24; the next K iteration direct-unpacks CB24 into accumulator tile2. This avoids assuming persistent DST arithmetic is identical to the existing FP32 pack/unpack path and preserves full register-acquire/commit/wait/release synchronization per K. The final iteration packs tile7 to output CB16. Eight full FP32 destination tiles and `dst_full_sync_en=True` match the existing correction adapter. No split accumulators, reduction reassociation, native narrowed matmul, SFPMAD substitution or host arithmetic fallback is introduced.

The dataflow reader loads one A tile and one B tile per32 K positions for each output tile. A source column and B source row are expanded by copying uint32 words into FP32 CB0/CB1 tiles. The face layout is `(row/16)*512+(col/16)*256+(row%16)*16+col%16`, matching the existing tested terminal-row relocation. No conversion or floating arithmetic is performed by the reader. Leading-axis broadcasting uses right-aligned shape strides, with zero strides for singleton input axes. Host metadata planning never reads tensor values.

Output tiles are partitioned contiguously over up to64 cores by default, capped by available device grid and tile count. Each output tile has one complete increasing-K recurrence on one core. Inputs/output are interleaved TILE FP32; both DRAM and L1 input allocation are accepted, and output is DRAM. Local CB storage is40,960 bytes/core: double-buffered operands, one zero tile, two source scratch tiles, one output tile and two feedback tiles. The reader's scratch pages stay reserved for the operation and are not consumed by compute. Scalar reader expansion may become a bottleneck; its benefit is unmeasured.

## Shapes and CPU checks

The runtime derives M/N/K independently, allows positive nonaligned spatial dimensions, iterates only logical K, and uses input/output padded tile counts for addressing. Invalid output rows/columns receive zero source operands. No public prefill sequence restriction is introduced by this diagnostic helper.

Seven source-check families include actual model matrix sizes `(64,128)@(128,64)`, `(64,64)@(64,128)`, `(128,64)@(64,128)`, nonaligned `(33,65)@(65,17)`, singleton K, unequal input ranks, and multiple independently broadcast leading axes. The checks verify32x32 face-offset bijection, source address bounds, batch broadcast mapping against a separate flattening construction and complete nonoverlapping1/4/64/110-core partitions. They do not establish device CB synchronization or kernel compilation.

The source audit finds one `generic_op` in `FusedCorrectedMatmul.__call__`, no Torch/from_torch/to_torch call there, and one exact frozen arithmetic body inside the increasing-K compute loop. The pure graph class is extracted with AST at setup in this diagnostic; any eventual standalone decoder integration must embed the resulting literal source, as previous builders do.

## Parent-owned device sequence

Start with one actual-prefix head and the Gram matrix, four output tiles, then expand to state-update/output matrix kinds and leading broadcast. The probe compares the new result both to the frozen per-K adapter on the same operands and the saved CPU matmul. Upload/readback and CPU reference are outside the device-only region. It records fixture bytes/hash and new kernel/source hashes. A cold diagnostic elapsed time is explicitly not a warmed speed measurement.

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_fused_corrected_matmul_probe --run-device --heads 1 --matrices chunk0_gram --max-cores 4 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_fused_corrected_matmul_gram_device.json
```

After a passing smallest component, use `--matrices chunk0_transformed_v,chunk0_transformed_k,chunk0_state_contribution,chunk0_state_update --broadcast-case`, then all32 actual heads, repeated execution and warmed timing. Device assembly/parity failures must be preserved and repaired in a separately versioned source if the initial source has already been measured. Full output/state propagation and long-stream gates remain necessary before integration.

The hypothesis is fewer host operations and device transfers, not yet higher utilization. Each matrix removes K-level slicing, WHERE expansion, resharding and generic dispatch; reading A/B and local corrected arithmetic remain. This could materially reduce the profiler's dynamic-source count and prefill dispatch cost. It does not by itself prove feasibility at advertised context262144, or the required faster-than-best traced-decode gate. The probe is prefill matrix work and leaves decode untouched.
