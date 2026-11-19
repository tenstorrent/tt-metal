# Next layout control: outer-axis exact norm reduction

Source/CPU-only; no F edits or hardware execution. Excludes the recurrent HC-before-reshape candidate already owned by root.

## Profile evidence

`profile_qz_norm/tracy/layer_0/decode_op_sequence.csv` steps3–21 implement the exact input norm's mean reduction. The initial reshape `[1,1,2048] → [1,4,16,32]` costs11.019us (physical H16 pads to32). Each cumsum's last-element selection lowers to untilize→slice→tilize:

| Selection | Steps | Conversion time | Slice time |
|---|---|---:|---:|
| Last of16 rows | 5–7 | 9.579us | 1.706us |
| Last of4 blocks | 10–12 | 5.062us | 0.913us |
| Last of4 groups | 15–17 | 5.027us | 1.028us |
| Last of8 lanes | 19–21 | 4.284us | 1.178us |

Conversions total23.952us and slices4.825us (28.777us combined). The previous quick message's 27.99us estimate was imprecise; these values are exact sums of the CSV rows. The first two stages additionally contain reshape step8 (3.165us) and cumsums steps4/9 (1.871/1.197us). This is separate from the recurrent reduction movement.

## One best falsifiable geometry hypothesis

For **single-token input norm only**, replace the first two cumsum/last-row stages with existing sequential FP32 `fast_reduce_nc` over outer dimensions:

```
# Setup, before measured execution / capture:
# norm_outer_result: empty FP32 TILE DRAM tensor, logical shape (1,1,1,32).

squares = ttnn.reshape(ttnn.multiply(wide, wide), (4,16,1,32))
total = ttnn.experimental.fast_reduce_nc(
    squares, dims=[0,1], output=self.norm_outer_result,
    compute_kernel_config=self.compute,
)
# total logical shape (1,1,1,32), correct 32 CPU lanes.
groups = ttnn.reshape(total, (1,1,4,8))
# Preserve existing four-group cumsum, eight-lane cumsum and all inverse math.
```

Logical flattening maps `squares[b,r,0,l]` to original index `b*512+r*32+l`. The wrapper processes sorted dimensions descending, so dim1 sums the16 rows left-to-right, then dim0 sums the4 blocks left-to-right. This is the same validated CPU schedule. Its FP32 kernel initializes from the first tile and uses sequential SFPU `add_binary_tile` for subsequent tiles. No FPU within-tile reduction is substituted.

**Important output contract:** `fast_reduce_nc` default output specification uses the input **padded shape**, so without a preallocated logical result the intermediate H1 padding becomes logical H32. The existing installed signature supports `output=`; provide the setup-owned `(1,1,1,32)` FP32 result for the final stage. The source validates output rank/dtype and preserves its tensor spec; the factory computes whole-tile counts from the input padded geometry. A one-tile result physically fits both shapes, while only row0 is logically consumed. Padding cannot affect row0 because reduction is elementwise across outer planes. Do not use a metadata reshape to discard logical elements after a default output; that violates volume checks.

References: `experimental/reduction/fast_reduce_nc/fast_reduce_nc.cpp` (descending-dimension wrapper and final output forwarding), `device/fast_reduce_nc_device_operation.cpp` (padded output spec and preallocated spec), `device/fast_reduce_nc_program_factory.cpp` (outer-plane tile geometry), and `device/kernels/reduce_nc.cpp` (FP32 ordered SFPU reduction).

Tradeoff: moving16 into an outer dimension expands the first tiled representation from4 tiles to64 tiles: FP32 storage16KiB→256KiB. It avoids two conversion triplets and the intermediate block reshape, but the larger initial reshape and low-output-tile reduction occupancy could erase the savings. This is why it is a device control, not a proven improvement. Prefer this focused first-two-stage change over making all four reductions outer axes, which would explode one-scalar-per-tile padding to8MiB and likely lose.

Run the existing exact norm same-input corpus before endurance, requiring unchanged mean/output codes; then paired warmed traced latency and post-change op profile. The prefill norm path can remain unchanged initially. Scratch allocation is setup-only, reused serially and trace-safe; do not upload host mean values or change the public nonaligned/context contract.

## Allocation refinement and actual-method probe

Root's implementation plan improves the scratch ownership: call runtime `ttnn.empty` for the explicitly shaped final reduction output instead of adding persistent setup scratch. This replaces the output allocation that `fast_reduce_nc` would otherwise make internally; it introduces no additional shared mutable state or lifetime contract. It remains subject to the normal trace-allocation rules and stress/determinism gates. The earlier setup-owned scratch description was one possible implementation, not a required contract.

`tests/input_norm_actual_probe.py` invokes the actual `FusedDecoder.norm` unbound method on a minimal namespace with real input gamma/config, device/compute and round/widen chain. It neither duplicates the device norm arithmetic nor constructs full expert weights. Default inputs are17 sensitive/edge positions from each of the actual seed913/42/154 oracle streams, each invoked as a single-token row. `--all-stream` covers all prefix, decode and continuation rows (8353 per seed); `--snapshot` loads an explicitly frozen source. An observer captures the mean's `multiply(...,1/2048)` output when present; final BF16 codes are always checked. Reports are diagnostic, not decoder acceptance.

CPU-only validation used `--oracle-only`, confirmed all51 selected references finite and no TTNN import, and wrote `input_norm_actual_cpu.json`. Device comparison fields are null in that CPU artifact, not claimed zero-error hardware results. Root's device command:

```
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.input_norm_actual_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/input_norm_actual_device.json
```
