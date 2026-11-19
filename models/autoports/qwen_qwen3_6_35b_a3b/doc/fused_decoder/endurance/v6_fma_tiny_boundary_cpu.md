# Corrected-FMA tiny boundary: bounded cause and consequence

CPU simulation of arithmetic flushing reproduces **every observed mismatch** in the three independently measured matrices. No upload or bitcast corruption is required to explain these fixtures. The measured tiny changes produce zero final-state code differences when injected into canonical downstream CPU arithmetic. No additional correction is proposed on this evidence; all decoder and performance gates remain unchanged.

Run from repository root:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_fma_tiny_boundary_cpu
```

The [test script](../../../tests/v6_fma_tiny_boundary_cpu.py) compiles a CPU C++ transcription generated directly from the current corrected-FMA `Graph` nodes, using `-O2 -ffp-contract=off -fno-fast-math`. Its data are the exact [saved matrix fixtures](v6_corrected_fma_matrices_device.fixtures.pt) and [saved device outputs](v6_corrected_fma_matrices_device.device.pt). The source, fixture, device tensor, metadata, generated C++, and parent full-graph hashes are recorded in [v6_fma_tiny_boundary_cpu.json](v6_fma_tiny_boundary_cpu.json). Generated C++ is preserved as [v6_fma_tiny_boundary_cpu.cpp.snapshot](v6_fma_tiny_boundary_cpu.cpp.snapshot). The script asserts TTNN was not imported; no hardware was accessed.

The simulation includes every mismatching output and 512 matching controls from each of the three affected matrices. Each output follows increasing-K accumulation. Arithmetic FTZ means that after each separately rounded FP32 addition, subtraction or multiplication in the correction graph, a subnormal result becomes signed zero. Input FTZ is tested independently. These hypotheses use actual runtime operands; no expected outcome is hardcoded.

| Independent matrix | Device mismatches | Explained by graph arithmetic FTZ | Explained by input-only FTZ | Explained by true-FMA output-only FTZ |
|---|---:|---:|---:|---:|
| Chunk0 transformed K | 2,372 | 2,372 | 98 | 676 |
| Chunk1 transformed K | 2,054 | 2,054 | 28 | 706 |
| Chunk1 state contribution | 3,768 | 3,768 | 243 | 291 |

Arithmetic FTZ preserves all 512 matched controls per matrix and produces zero code differences against the saved device results on all selected outputs. Adding input FTZ to arithmetic FTZ gives the same result. Therefore this fixture cannot establish whether subnormal input values also flush during transfer, but **input-only corruption is insufficient**, and no additional corruption hypothesis is needed. The seven matrices that were already device-exact are recorded without repeating the simulation.

The largest observed difference, `3.851859888774472e-34`, is one FP32 ULP at a normal value around `3.2458456169156174e-27`: expected bits versus actual bits are preserved in the JSON. A lost subnormal correction term can alter the final rounding decision at a larger normal magnitude. Thus the larger absolute error does not contradict the prior CPU proof: the IEEE arithmetic graph on these same selected outputs differs from true-FMA/Torch by at most `1.1479437019748901e-41`; adding FTZ to its intermediate arithmetic reproduces the larger device differences exactly. This is a measured arithmetic explanation, not a proof of universal hardware behavior. Tiny-value PCC and relative norms are not used as evidence because their calculations can underflow; exact code counts and absolute differences carry the conclusion.

Two causal downstream controls are distinct:

1. Replace transformed-K matrices with captured device matrices, then compute their state products and the remaining graph canonically. Final FP32 and BF16 state code differences are both zero.
2. Replace state-contribution matrices with captured device matrices from the independent canonical-state fixtures. Final FP32 and BF16 state code differences are both zero. The report's `both` row has the same independent state-contribution replacement; it is explicitly not a compositional second transformed-K experiment.

The parent subsequently ran the actual corrected-FMA plus scaled-MKL output/state graph at length128. [v6_chunk64_corrected_fma_output_device.json](v6_chunk64_corrected_fma_output_device.json) reports bit-exact FP32 output, BF16 output, and FP32 final state, with a clean device close. This independently supports the bounded consequence result. It does not establish every-token decoder acceptance, every logical length, or required latency. Additional scaling or a wider software accumulator would add cost without a demonstrated benefit for these fixtures and is not proposed absent a consequential counterexample.
