# V6 cumulative-decay double-single candidate

**CPU proof passed on1,193,984 prefixes with zero FP32-code differences from the installed HF/Torch cumsum.** This is finite-fixture CPU evidence; the device helper has not been run and no universal exactness or decoder acceptance claim is made.

The [probe](../../../tests/v6_double_single_cumsum_probe.py) implements the requested full TwoSum residual:

```text
s = hi + x
b = s - hi
e = (hi - (s - b)) + (x - b)
lo = lo + e
output = s + lo
hi = s
```

Every operation above rounds to FP32. `hi` and `lo` restart at each internal64-token chunk. Final chunks are zero-padded, including comparisons of the padded output prefixes used by the state update's `g_last`.

The [final CPU result](v6_double_single_cumsum_cpu_final.json) compares all32 actual layer34 prefix heads at lengths1,2,31,32,33,63,64,65,127,128. Random coverage uses seeds0,1,42,154,913; lengths1,31,33,63,64,65,127,128,129,257; and negative uniform, negative log-spread, and negative softplus-like distributions with per-head scaling. Each random case contains64 independent sequences. Every target Torch FP32 cumsum was also checked against FP64 cumsum rounded to FP32.

`ttnn_double_single_chunk_cumsum(ttnn, values, mesh_device=mesh, memory_config=...)` is a proposed pure-device helper in the same probe. It uses FP32 device zeros, optional device padding, ADD/SUB without the unsupported BF16 rounding flag, and concatenation. It returns the internally padded length, with no host tensor conversion. It requires positive logical length but imposes no alignment restriction.

The probe saves the exact source snapshot, source/input hashes, distribution parameters, counts, mismatch examples when present, and adjacent provenance. The earlier `v6_double_single_cumsum_cpu.json` preserves the first identical CPU proof; its helper obtained its device through the input tensor. The final source takes `mesh_device` explicitly, matching the component's setup-owned mesh.

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_double_single_cumsum_probe \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/FRESH_DOUBLE_SINGLE_CPU.json
```

Remaining work is a same-operand device comparison, followed by chunk64 state and integrated decoder checks. The CPU result does not establish that SFPU ADD/SUB, padding, or concatenation reproduce these prefixes on hardware. It also does not eliminate the independent exponential, triangular-sum, or GEMM rounding boundaries.
