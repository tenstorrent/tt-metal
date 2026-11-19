# Final four-group and eight-lane norm reduction control

This is a source-supported candidate with a passing isolated device probe against selected v3 (`73ba9aabb10a2c34f3b10b5ff1fc93418470c0a624dd863e87a8c8c2af2901ba`). Production source is unchanged. The snapshot `v3_norm_last_reduce.py.snapshot` has SHA256 `aa23c1ea1f1174a2f3693e35da20426dea95335d811808710a12625e6b9c1a0a`; its adjacent patch and provenance record the exact change.

The earlier `AUTOFIX_norm_outer_layout.md` control replaced only the first sixteen-row and four-block reductions. It explicitly retained the final four-group/eight-lane cumsums. The later HC prefill layout also changed the earlier stages. This control therefore targets a different, previously unmeasured part of the norm. It rearranges only the 32 already-reduced lanes, not all 2048 input scalars.

## Order and native contract

The original final sum reshapes each row's 32 lanes into four groups of eight, sums the four groups in increasing order, then sums the eight lanes in increasing order. The candidate untilizes that 32-lane tensor, reshapes to `(rows,4,8,1)`, permutes to `(8,4,rows,1)`, tilizes, and invokes `fast_reduce_nc(..., dims=[0,1])` with an explicit logical output `(1,1,rows,1)`.

`ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc.cpp` sorts dimensions descending. It therefore reduces the four groups on axis1 first, then the eight lanes on axis0. Its `device/kernels/reduce_nc.cpp` FP32 path copies the first tile and sequentially applies SFPU `add_binary_tile`; it does not substitute a within-tile FPU reduction. The explicit output preserves logical singleton width/height instead of accepting the default padded output shape. Generic interleaved FP32 row-major permutation is supported by `operations/data_movement/permute/device/permute_device_operation.cpp`; this permutation leaves the last axis unchanged and can use its row-invariant path.

The old scans start with zero; the outer reduction starts with the first operand. These operands are sums of squares and are nonnegative. The diagnostic still requires exact consumed FP32 sum and shifted-mean codes, plus complete BF16 norm codes; it does not waive signed-zero or rounding differences. All subsequent mean scaling, epsilon, inverse refinement and gamma boundaries are unchanged.

## Cost to measure

The rewrite removes two cumsums and their last-element selections. It adds row-major conversion, reshape, permutation, tilization and two outer reductions. Its 32 scalar outer planes physically occupy 128KiB for a singleton row and 512KiB for 128 rows. Relative to the incoming compact 32-lane total this is 32-fold storage expansion for aligned multirow input, although the original subsequent spatial reshapes also pad. These adapters may erase any reduction savings. A lower op count or CPU equivalence is not a latency result.

## Prepared evidence and commands

`tests/v3_norm_last_reduce_probe.py` calls the actual baseline and candidate `FusedDecoder.norm` methods on minimal setup objects, with checkpoint gamma and saved external-stream inputs. Defaults cover six sensitive single-token positions, prefix32, prefix128 and a zero-input case. It records the exact final FP32 sum, epsilon-shifted mean and complete BF16 output, then measures the complete norm trace including every adapter and output allocation. It records provenance, preserves existing report files and closes chip0 in `finally`. It is diagnostic only and cannot satisfy decoder acceptance by itself.

CPU-only mapping proof `v3_norm_last_reduce_cpu.json` passed all nine cases, including actual zero codes. This proves the mapping and CPU addition order only. No hardware execution was performed by this source-investigation agent; the serialized hardware owner subsequently completed the device control recorded below.

Hardware owner command:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v3_norm_last_reduce_probe
```

If the exact-code checks pass and complete-norm traces improve, benchmark the full decoder with the isolated snapshot using the existing snapshot-aware candidate launcher. Retain only after complete traced decode beats the selected correct control and the usual correctness, endurance, context and watcher gates pass. No performance, capacity, or Stop Rule claim follows from the prepared CPU proof.

## Completed isolated device control

The serialized hardware owner produced [v3_norm_last_reduce_device.json](v3_norm_last_reduce_device.json). All nine actual-method cases pass exact final-sum, epsilon-shifted-mean and complete normalized-output comparisons, including zero codes. First and last traced outputs are exact. Timing covers the complete actual norm with all layout adapters, both reductions and output allocation. The device closed successfully. Production remains qualified v3; these microbenchmarks do not select a replacement or establish full-decoder speed. Whole-decoder comparisons are in progress.

| Case | Baseline median ms | Candidate median ms |
|---|---:|---:|
| position_128 | 0.169622 | 0.169366 |
| position_130 | 0.169678 | 0.169367 |
| position_430 | 0.169664 | 0.169486 |
| position_3979 | 0.169477 | 0.168441 |
| position_5991 | 0.169874 | 0.168889 |
| position_7703 | 0.169997 | 0.170283 |
| prefill_first32 | 0.211670 | 0.198100 |
| prefill_first128 | 0.242781 | 0.233815 |
| zero_input | 0.169961 | 0.169788 |

Singleton gains are small and not consistent across all rows: position7703 regresses in this probe. The prefix32/128 complete norm improves, but whole-decoder decode must still beat the qualified correct control before retention.
