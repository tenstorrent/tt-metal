# Direct GDN input preparation: unqualified simulator prototype

`tt/gdn_step/flat_prepare.py` and `flat_prepare_reader.cpp` prototype a direct
path from tiled model Q/K/V and gates to the persistent FP32 recurrence inputs.
They consume token row zero, expand BF16 bits exactly to FP32, pack V/gates,
and reuse the existing shared-Q/K FP32 normalization compute/writer kernels.
The native `exp(log_decay)` operation stays external. No model path selects it.

This targets the current adapter's repeated slices, layout conversions,
casts, gate concatenation/permutation/padding and memory-config conversions.
It does not change FP32 recurrent state or BFP8 weights/KV. There is no measured
performance uplift or correctness qualification for this prototype yet.

The simulator attempt was started persistently at about 22:14 UTC Oct 9 and
terminated with exit-code failure:

```text
UnsupportedFunctionality: tensix_setdvalid: interaction between SETDVALID and implied src format is ill-specified (use UNPACR_NOP instead)
```

It did not finish the first native-reference/candidate comparison. The last
probe receipt still says `running`, with zero cases and `passed:false`; the
terminal systemd receipt proves the process stopped. This is a simulator
instruction-support failure, not evidence that the new hardware kernel is
correct or incorrect. Exact instruction attribution remains to be narrowed.

The test requests B1/B16/B32, logical time rows 1/32, DRAM/L1 inputs and two live
allocations with 0/1/0 rebinding. All four prepared outputs must be bit-identical
to the existing adapter/preparation operations. Inputs and output addresses
must be preserved; nonfinite/unwritten outputs fail. The gate was not relaxed.

The run used the pinned virtual Blackhole library, explicit instruction fallback,
one CPU, 8 GiB host RAM, private JIT cache and a 45-minute bound. No physical device
was opened. Launch/log/probe/source manifests and exact compressed source bytes
are retained. This prototype is excluded from the prioritized hardware queue.
