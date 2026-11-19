# Diagnostic: preserve physical padding while exposing only row0

Base `v4_serial_l1_compact.py.snapshot` is SHA256 `deb1fa8f3fb3020bfb0f983cc04c43e45f96f245535e0e8e7b5741fe417cf4ad`. Isolated candidate `v4_padded_views.py.snapshot` is SHA256 `a478451bdf99a1b6a7a496cc17d071166465bca1b9dcba64ee4e1429fba261f4`. Its three replacements affect only the decode concat-heads crop in linear/full attention and the decode MoE reduction crop. No production, prior probe, or native source was changed. This remains unselected and has no device result from the preparer.

The existing `pattern_assessment.json` keeps native decode concat-heads and native reduction. This candidate preserves both operators and their arithmetic. It removes their following row0 slice/repack, rather than retrying manual concat-heads or a different MoE reduction.

## Native shape and alias contract

Native concat-heads `compute_output_specs` explicitly rounds its batch dimension up to32, then exposes it as a logical dimension (`nlp_concat_heads_decode_device_operation.cpp:79–110`). Both16×256 and32×128 decode cases therefore return logical/physical `[1,1,32,4096]`, width-sharded L1. Only row0 is a model token. The candidate replaces slice-then-reshape with:

```python
merged = ttnn.reshape(merged, (1, 1, 4096), (1, 32, 4096))
```

MoE's FastReduce over eight experts exposes `[1,1,32,2048]` because its default TensorSpec derives from padded input dimensions (`fast_reduce_nc_device_operation.cpp:117–137`). The decode logical row0 is replaced with the same explicit-shape operation at width2048. The subsequent consumer is a row-wise shared-expert add. Attention's subsequent consumer is output projection; neither mutates the original producer buffer or reduces across the retained batch-padding rows.

The three-argument Python overload exists and accepts separate logical/padded shapes (`reshape_nanobind.cpp:117–138`). This is necessary: the ordinary shape-vector overload checks equal logical volume before dispatch, whereas `shape_corrector` returns explicit shapes directly when no inferred dimension is present (`reshape.cpp:343–362`). The tile-volume-change route permits a metadata view with tile-aligned padded H/W and unchanged width (`reshape.cpp:778–787`). This broad native route has a source warning about arbitrary volume-changing reshapes; this candidate uses only the narrow physically identical row0 case, not arbitrary reassociation or padding insertion.

Physical tile ordering is unchanged: rank4 `[1,1,32,W]` becomes rank3 padded `[1,32,W]`. Removing the leading unit dimension changes no tile index, face offset, row stride, or sharded placement. Logical row0 remains the first row of every width tile. Native `view_device` keeps the memory configuration when the last dimension is unchanged, builds a TensorSpec from the explicit logical/padded shapes, and aliases the existing mesh-buffer address (`ttnn/core/tensor/tensor_ops.cpp:335–475`). Standard interleaved and width-sharded paths are covered. This is not an ND-sharded claim.

Rows1..31 remain in storage and are reclassified as padding. They need not equal padding produced by the old copying slice; final logical output bits must be identical. The new view aliases its producer, but production never mutates these tensors after cropping, so no writable-alias conflict is introduced. The probe separately verifies buffer aliasing, retained padding, unchanged source contents and logical consumers; no output numeric waiver is introduced.

## Exact complete-adapter discriminator

`tests/v4_padded_views_probe.py` extracts statements from the actual baseline/candidate AST rather than duplicating the changed expressions in handwritten adapters. It runs:

- Both native head geometries16×256 and32×128, each BF16 and FP32, through actual DRAM-to-height-shard conversion, native concat-heads, and the crop adapter. It compares against the baseline and independent CPU head flattening.
- Actual MoE multiply, eight-expert FastReduce, and crop, each BF16 and FP32 at width2048. It compares against the original reduction path without replacing arithmetic by a CPU sum.
- Independent retained-padding cases for widths2048/4096, both dtypes, in DRAM and width-sharded L1. Rows1..31 contain distinct nonzero finite sentinels. The view must share the producer address; re-exposing the full physical rows must reproduce every input bit; the source must remain unchanged. Conversion to compact row-major must expose exactly row0.
- Logical downstream multiply checks on both complete adapters, and first/last traced replay checks. These verify useful outputs even when physical padding is retained. They do not substitute for the parent's full decoder/endurance validation of downstream output projection and shared-expert addition.

Synthetic finite operands are explicitly labelled; there is no claim of checkpoint captures. Every source transformation is hash-bound in the result. Timing includes the complete extracted producer/shard/crop adapter, rather than timing a host-only view in isolation. All host readbacks are outside capture/timing. Physicalchip0,32MiB trace reservation, L1-small32768, clean timing environment checks, synchronized trace release, and `finally` close are explicit.

The CPU-only artifact `v4_padded_views_cpu.json` passes row0 mapping checks at both widths. AST scope verification confirms changes only in `_linear_attention`, `_moe`, and `_full_attention`; the snapshot patch contains exactly the three crop replacements. No hardware execution was performed.

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_padded_views_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_padded_views_probe.json
```

Remaining uncertainties are native view/shard handling on the device, exact outputs with retained producer padding, and whether removing the adapters improves the complete path. These are measured gates; source legality alone earns no selection.

## Parent-owned device and whole-path results

The parent subsequently completed `v4_padded_views_probe.json` with `status=exact`,74 recorded exact checks (including the two CPU mapping checks), and clean device close. This covers both dtypes/head geometries, MoE outputs, logical consumers, source preservation, retained sentinel padding, address aliasing, compact-RM conversion, and first/last replay checks. The result's candidate/probe hashes match the frozen preparation artifacts. No output or signed-zero exception was needed.

Complete extracted-adapter traced medians, in microseconds:

| Adapter | Original | Padded view |
| --- | ---: | ---: |
| BF16 heads16×256 | 15.049 | 7.857 |
| BF16 heads32×128 | 14.987 | 8.974 |
| BF16 MoE multiply/reduce/crop | 21.143 | 19.275 |
| FP32 heads16×256 | 17.335 | 7.884 |
| FP32 heads32×128 | 16.041 | 8.996 |
| FP32 MoE multiply/reduce/crop | 33.926 | 31.492 |

Parent whole-decoder S256 results are separately source-bound to this isolated candidate:

| Artifact | Prefill ms | Decode ms | Prefill PCC | Decode eager/traced PCC |
| --- | ---: | ---: | ---: | ---: |
| `v4_padded_views_layer0_s256.json` | 118.013049 | 1.347492 | 0.999962395 | 0.999977906 |
| `v4_padded_views_layer3_s256.json` | 9.004255 | 0.587704 | 0.999890040 | 0.999923261 |

Both whole-path artifacts report deterministic execution and a passing runtime fallback audit. The source-bound compact baseline `v4_serial_l1_compact_s256.json` reports layer0 decode1.351969ms, making the isolated layer0 improvement small (~0.33%); adapter speedups are not additive whole-path predictions. The rewrites are decode-only, so prefill differences between separate runs are not attributed to them. These S256 records are explicitly diagnostic and acceptance-ineligible. The parent's newer combined candidate `82c8c629…` includes this change with PF both-RM, direct-RM convolution, and router direct slicing; its integrated benchmark/endurance work is separate and ongoing. No standalone production selection or full acceptance is inferred here.

Evidence SHA256 values, also bound in `v4_padded_views.py.provenance.json`:

- Device exact probe: `dcd65031ff0f37ae7cd6dd1342ecf2ad41f863b1a2f3a2630f8aa5e308eea3ec`.
- Layer0 S256: `7e948c9b9626e80d5c77b5703a72d502d00d93c70ac14e15ddde722ccc54d813`.
- Layer3 S256: `20200680d05e917afd4b0946a19753d1f607be7a458af6a88270b43bc041d578`.

The earlier pending statements describe preparation history; these device results resolve the isolated shape/alias/exactness/performance questions. The preparer only inspected the resulting files and updated documentation/provenance; snapshots, probes, production and core harnesses remain unchanged.
