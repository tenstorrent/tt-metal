# Decode expert weighted reduction: source-only candidate

The existing Blackhole op can express the decode expert mix without transposing
the large expert-output tensor. The runnable candidate is
[probe_optimized_expert_compact.py](../../tests/probe_optimized_expert_compact.py),
owned jointly with the indexed-expert investigation; no duplicate probe was
created. `--expert-backend expanded --expert-merge weighted` changes only the
mixing tail. `--expert-backend indexed --expert-merge weighted` uses the same
reduction with eight compact expert slots.

Runtime source is unchanged at
`e81018299b722aa81eae0e4e9ec3ec638520adcb3bdd85d3c2c8b429dfa0e370`.
The independently checked probe hash is
`bc0d1adf68ae940e3d00cf15155736cdfd8276056e49d080cf91f7ccea82cf41`.
This report contains no device accuracy, allocation-fit, or performance result.

## Exact contract and adaptation

The bound Python name is
`ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc`, as defined in
`attn_res_weighted_reduce_nc_nanobind.cpp:20` under
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_weighted_reduce_nc`.
The formula is `out[r,0,h,w] = sum_e input[0,e,h,w] * weight[r,e,h,0]`.

| Constraint | Source and consequence |
| --- | --- |
| Blackhole only | `device/attn_res_weighted_reduce_nc_device_operation.cpp:23-27`. The recorded `verified_headline_layer0.log:8` identifies Blackhole. No cross-architecture support claim. |
| BF16 input; BF16 or FP32 weights | Device validation lines38-39. Current sparse-down output is already BF16; preserve routing-weight dtype rather than introducing a cast. |
| TILE, interleaved input/weights/output | Device validation lines38-45; `check_tensor` enforces tiled device tensors. Weighted output cannot directly use the existing width-sharded mix memory. Produce interleaved L1, then apply the original mix memory config. |
| Rank4, reduction dim1, singleton input batch | Device validation lines51-88. Input `[1,E,1,2816]`; weight `[1,E,1,1]`; both have physical row height32. E can be128 or8; it is not rounded to32 on this outer dimension. |
| Weight logical last dimension1; matching expert/row dimensions | Device validation lines61-80. Permute routing `[1,1,1,E]` by `(0,3,2,1)`. Do not use a metadata-only reshape across its padded axes. |
| Output BF16 `[1,1,1,2816]` | Output spec lines102-108 preserves logical dimensions and input dtype. It matches the old matmul result's logical shape/dtype. |
| FP32 accumulation, HiFi4 | API `attn_res_weighted_reduce_nc.cpp:27-37` selects this tested default; kernel `device/kernels/weighted_reduce_nc.cpp:76-95` MACs all expert tiles into one destination tile before packing. |

[OptimizedExperts._chunk](../../tt/optimized_decoder.py#L170) keeps sparse gate,
GELU/up, sparse down, rounding, and logical reshape intact. Its old mixing tail
(lines202-209) transposes `[1,128,1,2816]` before matmul. The candidate permutes
only routing, calls the weighted reduction, and restores `self.mix_memory`.
Prefill returns through its original path; downstream norm/residual/FP32
handling is unchanged. Reduction order and native packing may differ, so
actual-input PCC/replay gates remain required.

The probe prepares a dedicated weighted compute configuration at setup:
HiFi4, FP32 destination, math approximation false, packer accumulation false.
It preserves the original `mix_compute` for the matmul control. This avoids
assuming that the full-attention Python mix configuration requests FP32:
`fused_decoder.py:51-52,246-251` gives full attention no `mixfp32` flag. Native
matmul normalization may change the effective setting; the independent
projection review reports FP32 destination in the final native rows. The
weighted candidate declares its requested precision explicitly in either case.

Existing unit coverage in
[the op test](../../../../../tests/ttnn/unit_tests/operations/experimental/test_attn_res_weighted_reduce_nc.py)
includes E8, unaligned logical rows (lines150-165), FP32 weights with BF16
output (lines168-185), program-cache reuse, and rejected incompatible inputs.
It does not establish E128 decoder fit or performance.

## Allocation and work hypothesis

Program factory lines48-59 derives `C=E`, `Ht=1`, `Wt=88`; lines69-71 gives
one weight set per group. Lines95-125 allocate input `2*C*2048`, weights
`C*weight_tile_bytes`, and output `2*2048` bytes **per participating worker**.

| Expert slots | BF16 weight CB total | FP32 weight CB total |
| --- | ---: | ---: |
| 128 | 790,528 bytes (772 KiB) | 1,052,672 bytes (1,028 KiB) |
| 8 | 53,248 bytes (52 KiB) | 69,632 bytes (68 KiB) |

These figures exclude live tensor storage and kernel/runtime overhead. The
E128 family is therefore materially more demanding on L1 than E8. The kernel
still reads all128 expert slots in expanded mode; it does not consult routing
sparsity. It removes the large expert-output transpose and matmul but adds a
small routing permutation and interleaved-to-sharded output conversion.
Only a complete measured candidate can establish whether this is faster.

## Probe and unapplied patch

First isolated whole-layer command, to be run only by the hardware owner:

```bash
python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_expert_compact \
  --expert-backend expanded --expert-merge weighted --defaults \
  --layer 0 --length 4096 --real \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer0_4096_128.pt \
  --decode --steps 128 --timing --prefill-timing --verify-program-cache \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_expert_expanded_weighted_layer0.json
```

Use layer5 and its matching fixture/output for full attention. For stress,
use length1025, steps512, and the corresponding `actual_text_layer{0,5}_1025_512.pt`.
The indexed family uses `--expert-backend indexed`; matmul controls keep
`--expert-merge matmul`. All original parity/runtime/trace audits stay active.

[weighted_expert_reduce.patch](weighted_expert_reduce.patch) is an **unapplied**
minimal current-E128 runtime proposal: one setup compute config and replacement
of only the decode mixing tail. It imports no tests and changes no prefill,
router, expert projection, norm, or residual implementation. `git apply --check`
passes against the source hash above. The hypothetical patched source hash is
`2edc29751d07e285accdcf05977937b32eb51dce24abc983dfbed62363048000`.
Do not apply it while a parent-owned device sweep is active or before selection.

[weighted_expert_reduce_source_checks.json](weighted_expert_reduce_source_checks.json)
records four CPU-only executions of the exact probe setup/weighted-branch AST:
E128/E8 crossed with BF16/FP32 weight descriptors. They check legal shapes,
explicit compute settings, permutation of routing only, BF16 output, restored
mix memory, and deterministic layout algebra against matrix multiplication.
These checks use stand-ins and do not import TTNN, simulate hardware rounding,
or constitute synthetic-PCC acceptance evidence.
