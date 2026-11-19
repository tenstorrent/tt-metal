# V6 corrected-FMA adapter and ordered GEMM preparation

[The new module](../../../tests/v6_corrected_fma_matmul_probe.py) provides two device-only callables:

```python
fma = CorrectedFma(ttnn, mesh)
result = fma(accumulator, a, b)  # a*b+accumulator, full/row/column/spatial-scalar inputs
mm = CorrectedMatmul(fma)
product = mm(a_matrix, b_matrix)  # increasing-K recurrence, one corrected FMA per K
```

The arithmetic graph is imported unchanged from the bounded corrected-FMA primitive that the parent measured successfully in `v6_fma_sfpu_corrected_device_v2.json`. This preparation changes only dataflow and composition. It uses four tensor-backed CBs (multiplicands A/B, accumulator, output), FP32 direct-to-DST unpack for all three inputs, and height sharding over up to an 8×8 grid. For the actual 32-head shapes, all 64 cores participate. Each core owns 2 tiles for an M64×N64 product, 4 for M64×N128, or 8 for M128×N128. Four live CBs therefore hold at most 128 KiB per core for those shapes, excluding standard runtime allocations and any transient buffers.

Row/column expansion uses native `where(full_zero_condition, input, input)`, whose identical branches select the original FP32 bit pattern. This avoids arithmetic with zero and avoids the singleton repeat path's tilize/untilize conversions. The ternary implementation supports these row/column broadcast patterns and configures every FP32 operand for direct-to-DST unpack (`ternary_program_factory.cpp:1483–1520`). The expansion itself remains a hardware validation obligation; adapter tests independently read all three expanded inputs and compare their bits against CPU broadcasting, including negative zero. Full-shaped input tensors bypass this expansion. The output stays height-sharded between K steps, with one final DRAM conversion at the GEMM boundary.

`CorrectedMatmul` can replace a **diagnostic component instance's** `Chunk64StateGraph.mm` method:

```python
graph.mm = CorrectedMatmul(CorrectedFma(ttnn, mesh))
```

No production decoder or measured probe was changed. This is deliberately a per-K diagnostic; repeated dispatch, broadcasting, and data movement have not been optimized or accepted for final latency. Spatial matrix extents are tile aligned because the chunk graph internally pads to 64. This helper does not add a public sequence-length restriction.

## Prepared evidence and commands

Default execution emits source/CPU fixtures without importing TTNN or opening hardware. Four adapter fixtures cover full×full, column×row, row×column, and spatial-scalar×full, with signed zero injected into every operand. Their expected values use scalar CPU `fmaf`. Ten actual matrix fixtures cover both chunks' Gram matrix, transformed V/K, incoming-state contribution, and final state update across all 32 heads. Their expected values use the same CPU matrix multiplication order previously checked against scalar sequential FMA. Subnormal discrepancies remain possible and are reported as FP32 numerical/bit and BF16 code differences; this is not a claim of general FP32 FMA correctness.

Frozen preparation artifacts:

- `v6_corrected_fma_adapter_source_only.json`, `.compute.cpp.snapshot`, `.fixtures.pt`.
- `v6_corrected_fma_matrices_source_only.json`, `.compute.cpp.snapshot`, `.fixtures.pt`.

Python syntax, CPU fixture generation, source hashes, and the no-torch/no-host-conversion runtime AST audit pass. No hardware or full wrapper compilation was performed in this preparation. Python source SHA: `facef89f3f760fcd86c54e31e63ac938657f2ce02c639715699eb3c94ab90442`; emitted compute SHA: `be3b5c5c44aae30dc927e2977949508675638276bbd8e1c88a65f19f4675a7cd`.

Parent-controlled commands, using fresh output paths:

```bash
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_corrected_fma_matmul_probe --run-device --mode adapter --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_corrected_fma_adapter_device.json
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_corrected_fma_matmul_probe --run-device --mode matrices --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_corrected_fma_matrices_device.json
```

`--matrices chunk0_state_update,chunk1_state_update` selects particular actual products. `--k-limit` and `--heads` can reduce a diagnostic run; both are recorded, and such a reduced run is not full-matrix qualification. The runtime is guarded with `device_only()`. Reported wall time is explicitly cold diagnostic elapsed time, including compilation; it is not warmed latency evidence.
