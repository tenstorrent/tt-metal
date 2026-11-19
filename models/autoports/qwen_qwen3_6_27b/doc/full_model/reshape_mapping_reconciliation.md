# Independent reshape-map memory reconciliation

CPU-only follow-up to `b32_preparation_capacity_hypothesis.md`, 2026-11-14. No TTNN import/device calls or production edits. Parent supplied `prepare_memory_baseline.json`; the reproduction uses its actual allocation histogram, not an inferred total.

The source owner discovered by the parallel investigation is consistent with the allocation evidence: `reshape_view/device/reshape_tiled_program_factory.cpp` creates a UINT32 row-major segment-map tensor, moves its owning `MeshTensor` into `ProgramArtifacts::op_owned_tensors`, and the generic spec adapter parks that vector under `CachedMeshWorkload.shared_variables`. The exact adapter owner is `SpecSharedVariables::op_owned_tensors` at `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:899–903`; `make_shared_variables` moves the artifact's vector into that owner at lines1096–1109. This allocation is shape-dependent and remains until its program cache entry is destroyed. It is not a retained matmul activation or logits output.

`reshape_mapping_reconciliation.py` independently ports `reshape_map_output_page`'s face offsets, input-page grouping and contiguous-segment counting into plain Python. For equal tile-aligned widths it computes the column-zero map; every other column has the same segment offsets/counts and its input page IDs shifted by the column index. It then applies the existing native expression `segments += segments % 16` and row-major eight-bank/64-byte page allocation. This deliberately reproduces the existing expression rather than replacing it with a conventional alignment formula.

Command:

```bash
python models/autoports/qwen_qwen3_6_27b/doc/full_model/reshape_mapping_reconciliation.py
```

The six rows below are **exact matches of both size and count** between B32 checkpoints32 and64. Shape family is `[1,32*S,W] -> [32,S,W]`, output tile rows64. Counts match the indicated logical lengths; width is the innermost tensor dimension.

| W | S values | Native raw/aligned segments | Observed bytes/bank | Observed count | Compressed prediction bytes/bank |
| ---: | --- | --- | ---: | ---: | ---: |
| 5120 | Odd33..63 | 124/136 | 2785280 | 16 | 17408 |
| 5120 | 34,38,42,46,50,54,58,62 | 120/128 | 2621440 | 8 | 16384 |
| 2048 | Odd33..63 | 124/136 | 1114112 | 16 | 17408 |
| 1280 | Odd33..63 | 124/136 | 696320 | 16 | 17408 |
| 5120 | 36,44,52,60 | 112/112 | 2293760 | 4 | 14336 |
| 2048 | 34,38,42,46,50,54,58,62 | 120/128 | 1048576 | 8 | 16384 |

For example, odd S has136 map entries x4 UINT32 words x4 bytes =2176 bytes per output tile. W5120 has160 tile columns, hence `64 *160 *2176 /8 =2785280` bytes/bank. Keeping one column per tile row changes this to `64 *2176 /8 =17408`. The same mapping at W2048 and W1280 scales by64 and40 columns, reproducing those histogram bins exactly.

These six bins account for **896532480 bytes/device**, or **66.9033%** of the total observed **1340042240-byte** increase from `before_terminal_32` to `before_terminal_64`. They alone predict a reduction to9240576 bytes/device under same-width column compression: **887291904 bytes/device saved** for this interval. This is a source-derived prediction, not a post-fix hardware measurement. Other widths, flatten-direction maps and transient/binary allocations remain outside this six-bin reconciliation; no extrapolation claims complete attribution.

The reported 655360-byte/bank bin has observed count10. Eight entries are explained by W1280, S34 mod4, aligned128-segment maps (`64*40*2048/8`); two additional entries require distinguishing coincident sizes. This bin is excluded from the exact-count six-bin total above. The CPU reproduction does not assign those extra two owners by guess.

Relevant model call sites provide these widths without changing the selected decoder strategy:

- W1280: bounded/flat prefill returns the compact residual to `[B,S,1280]` (`tt/flat_prefill.py::_flat_chunk`, and bounded full-attention `_layer_chunk`).
- W5120: final/decoder norm expands compact gathered hidden back to `[B,S,5120]` (`tt/multichip_decoder.py::_norm`); unaligned-head and other existing paths also use this width.
- W2048: packed QKV's compact projection is restored to its logical shape (`tt/multichip_decoder.py::_qkv`).

The checkpoint at S192 reports17247420928 bytes/device after the helper returns and2503948288 after disposable pre-capture program-cache clearing. Therefore **14743472640 bytes/device** belonged to cached native workloads/resources in this diagnostic run. The six matched map bins explain why program-cache release is much larger than typical2048-byte/bank binary allocations. The clear is attribution evidence only, not a production remedy.

The same-width/tile-aligned compression covers these major owners while leaving logical shapes, kernels' output semantics, B32 native optimized prefill, dtype/fidelity and decoder CCL policy intact. It must retain general mapping for changed widths or partial tile widths. Exact page/segment equivalence and hardware comparison remain the native-fix owner's validation task; this independent check establishes the memory mechanism and size prediction, not acceptance of an untested patch.
