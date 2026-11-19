# FP32 direct all-reduce AutoFix

Status: **hardware-qualified component repair; experimental, not promoted.** This fixes a demonstrated FP32 input precision boundary. It does not establish decoder quality or restore the mandatory performance floor. Original artifacts remain unchanged.

## Cause and bounded repair

[AUTODEBUG_layer34.md](AUTODEBUG_layer34.md) identified the direct all-reduce consuming Float32 partials through `add_tiles`/SrcA/SrcB, which narrows them to TF32 even with FP32 destination accumulation. The single-chip `fast_reduce_nc` instead unpacks full FP32 into DEST and uses SFPU addition. A final BF16 pack cannot recover cancellation residuals already lost by TF32 narrowing. Rank-major scratch placement and receive completion were consistent; the diagnosis did not establish an arrival-order bug or prove the cause of any particular model token failure.

[fp32_all_reduce_repair.patch](fp32_all_reduce_repair.patch) changes exactly two C++ files under `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/`:

- `all_reduce_async_program_factory.cpp`: enable `REDUCE_FP32_IN_DEST` and `UnpackToDestFp32` for the reduction input CB only when the input is Float32 and `fp32_dest_acc` is enabled.
- `kernels/compute/reduction.cpp`: copy each rank's full FP32 tile into DEST and accumulate with `add_binary_tile`, in serial rank order 0, 1, 2, 3. Use one accumulator and one temporary DEST tile, then perform one requested output pack. Preserve the existing BF16-input branch.

The patch follows the full-FP32 branch in `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp`. Adding an unpack flag while retaining FPU `add_tiles` would not be a valid repair. Initial source identities are retained in [fp32_all_reduce_repair_patch_sources.json](fp32_all_reduce_repair_patch_sources.json).

## Build and actual runtime

The parent applied the patch, built and installed with:

```sh
cmake --build build_Release --target ttnncpp -j 2
cmake --install build_Release --component ttnn-runtime --prefix "$PWD/build_Release/fp32_ar_install"
```

The staged `lib/_ttnncpp.so` was copied to a temporary sibling of `build_Release/lib/_ttnncpp.so` and installed with `os.replace`. This atomic replacement preserved the old mapped inode in the concurrent CPU-only profile postprocessor; no device job was running during installation.

This checkout uses a unity build. Building an individual operation object is insufficient. The build produces `build_Release/ttnn/_ttnncpp.so`; the loaded runtime is the separately installed `build_Release/lib/_ttnncpp.so`, with a different inode. Installation updates that copy and its runtime search path. The successful probe records loaded binaries from `/proc/self/maps`, hashes/inodes, and both C++ source hashes. Its loaded `_ttnncpp.so` SHA256 is `cf28f221fdf91a50f429e2383ab7e343c64bef2ecec3828f25fc1805d3c5bcb0`; see the complete provenance in the artifact.

## Cancellation and trace controls

`tests/probe_mesh_fp32_all_reduce.py` uses the actual 1×4 mesh, cluster axis 1, two links, 32 width-sharded cores, input shards `[32,64]`, scratch shards `[32,256]`, FP32 input/scratch and BF16 output. Eight cancellation/association tuples are checked by exact equality. A separate BF16-input control exercises the unchanged branch. Each dtype has one eager execution and four changing-input trace replays across four ranks: 20 checks per dtype.

The probe requires `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`. All ten source/destination copy pairs are warmed before any capture; changing-input tensors remain allocated until every trace is released. Failures write their JSON before returning nonzero.

| Artifact | FP32 exact checks | BF16 controls | Interpretation |
|---|---:|---:|---|
| [fp32_ar_before.json](fp32_ar_before.json) | 0/20 | 20/20 | Original cancellation failure retained |
| [fp32_ar_after.json](fp32_ar_after.json) | 0/20 | 20/20 | Repaired kernel matches the first seven of eight tuple expectations; the final tie makes every whole-tensor record fail under the original oracle |
| [fp32_ar_after_pack_oracle.json](fp32_ar_after_pack_oracle.json) | 20/20 | 20/20 | Corrected hardware-pack oracle; **40/40 total pass** |

The original oracle used PyTorch BF16 round-to-nearest, ties-to-even. The hardware pack uses nearest, ties away from zero: the final tuple sums to FP32 `1.00390625`, which packs to `1.0078125`, while PyTorch gives `1.0`. The precise in-tree Blackhole test reference is `tests/ttnn/nightly/unit_tests/operations/experimental/nomic_embed_text_v2_moe/test_routed_combine.py:29`, `bfloat16_ties_away`: `((bits + 0x8000) & ~0xFFFF).view(torch.float32).bfloat16()`. Supporting pack-rounding comments also exist in `tests/ttnn/unit_tests/operations/data_movement/test_tilize.py:1205` and `tests/ttnn/unit_tests/operations/reduce/test_cumsum.py:395`.

The corrected finite-input oracle retains the tie tuple, all cancellation tuples, exact equality, and the original BF16 controls. First-seven tuple expectations are unchanged. Records additionally expose PyTorch's reference and tie-difference counts. Neither historical JSON was rewritten.

## Integration limits and exact recurrence

The combined native linear norm + native DRAM projection + exact SFPU recurrence candidate, with the repaired collective and safe production helpers enabled, **still fails sensitive layer 34**: HF minimum whole-output PCC `0.9879428617105454` at decode step 174. Evidence: [repair_fp32_ar_native_exact_l34.json](repair_fp32_ar_native_exact_l34.json), [HF comparison](repair_fp32_ar_native_exact_l34_hf.json), and [exact command/source provenance](repair_fp32_ar_native_exact_l34.json.provenance.json). The run uses length 128, seed 154, and 192 decode steps. The below-0.99 diagnostic anomaly is unresolved; gates and accepted baselines are unchanged. This repair therefore does not restore stage quality/performance, and full native integration qualification is not claimed.

Separately, [exact_recurrence_component.json](exact_recurrence_component.json) passes **48/48 bit-exact output/state comparisons** across four ranks, three changing steps, and rounded/unrounded state modes. `tt/mesh_exact_recurrence_probe.py` preserves the existing reduction calls and uses full-FP32 SFPU elementwise phases; it remains experimental. Component equality does not establish integrated decoder speed, real-weight quality, or watcher qualification. No integrated speed qualification is claimed for this recurrence implementation.

Subsequent whole-decoder FUNCTIONAL measurements reject the repair family: exact recurrence plus repaired CCL takes0.869184612ms/5.96287% (`repair_fp32_ar_exact_l0.json`), while native norm/DRAM projections plus repaired CCL take0.554099732ms/9.35361% (`repair_fp32_ar_native_l0.json`). Both pass layer0 accuracy/replay but fail the10% floor. The native C++ edits were then reversed and rebuilt; `native_restore_verification.json` records exact original C++ source hashes and bit-equal retained-default outputs. The unapplied patch remains available for a separate collective fix, not a promoted model configuration.
