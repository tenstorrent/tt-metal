# Program factories selected in the captured run

Source capture: `generated/ttnn/reports/qwen3_vl_demo_aug27_1509/graph_capture.json`

| op | calls | program factory | cache hits |
| --- | ---: | --- | ---: |
| `MatmulDeviceOperation` | 74412 | `ttnn::prim::MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` | 74406 (100%) |
| `MatmulDeviceOperation` | 528 | `ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory` | 520 (98%) |
| `LayerNormDeviceOperation` | 29246 | `ttnn::prim::LayerNormMultiCoreProgramFactory` | 29237 (100%) |
| `LayerNormDeviceOperation` | 29200 | `ttnn::prim::LayerNormShardedProgramFactory` | 29198 (100%) |
| `ShardedToInterleavedDeviceOperation` | 45612 | `ttnn::prim::ShardedToInterleavedProgramFactory` | 45607 (100%) |
| `InterleavedToShardedDeviceOperation` | 44402 | `ttnn::prim::InterleavedToShardedProgramFactory` | 44399 (100%) |
| `BinaryNgDeviceOperation` | 43782 | `ttnn::operations::binary_ng::BinaryNgDeviceOperation::ProgramFactory` | 43774 (100%) |
| `ReshardDeviceOperation` | 43200 | `ttnn::prim::ReshardGenericFactory` | 43197 (100%) |
| `RotaryEmbeddingLlamaDeviceOperation` | 28800 | `ttnn::experimental::prim::RotaryEmbeddingLlamaMultiCoreSharded` | 28798 (100%) |
| `RotaryEmbeddingLlamaDeviceOperation` | 288 | `ttnn::experimental::prim::RotaryEmbeddingLlamaMultiCore` | 285 (99%) |
| `PagedUpdateCacheDeviceOperation` | 28800 | `ttnn::experimental::prim::PagedUpdateCacheProgramFactory` | 28799 (100%) |
| `NLPCreateQKVHeadsDecodeDeviceOperation` | 14400 | `ttnn::experimental::prim::NLPCreateQKVHeadsDecodeInterleavedProgramFactory` | 14399 (100%) |
| `SdpaDecodeDeviceOperation` | 14400 | `ttnn::device_operation::MeshDeviceOperationAdapter<ttnn::prim::SdpaDecodeDeviceOperation>::DirectDescriptorFactory` | 14399 (100%) |
| `NLPConcatHeadsDecodeDeviceOperation` | 14400 | `ttnn::experimental::prim::NLPConcatHeadsDecodeProgramFactory` | 14399 (100%) |
| `UntilizeCodegenDeviceOperation` | 2431 | `ttnn::prim::UntilizeCodegenProgramFactory` | 2425 (100%) |
| `SliceDeviceOperation` | 1758 | `ttnn::prim::SliceTileProgramFactory` | 1751 (100%) |
| `EmbeddingsDeviceOperation` | 800 | `ttnn::prim::EmbeddingsRMProgramFactory` | 799 (100%) |
| `EmbeddingsDeviceOperation` | 400 | `ttnn::prim::EmbeddingsFusedProgramFactory` | 399 (100%) |
| `TilizeWithValPaddingDeviceOperation` | 806 | `ttnn::prim::TilizeWithValPaddingMultiCoreDefaultFactory` | 801 (99%) |
| `TilizeWithValPaddingDeviceOperation` | 14 | `ttnn::prim::TilizeWithValPaddingMultiCoreBlockInterleavedFactory` | 11 (79%) |
| `ConcatDeviceOperation` | 804 | `ttnn::prim::ConcatProgramFactory` | 801 (100%) |
| `TransposeDeviceOperation` | 800 | `ttnn::prim::TransposeHCTiledInterleavedProgramFactory` | 799 (100%) |
| `PlusOneDeviceOperation` | 800 | `ttnn::experimental::prim::PlusOneProgramFactory` | 798 (100%) |
| `TypecastDeviceOperation` | 448 | `ttnn::prim::TypecastProgramFactory` | 438 (98%) |
| `CopyDeviceOperation` | 400 | `ttnn::prim::CopyDeviceOperation::DefaultTilized` | 399 (100%) |
| `ArgMaxDeviceOperation` | 400 | `ttnn::prim::ArgMaxMultiCoreProgramFactory` | 399 (100%) |
| `NlpCreateHeadsDeviceOperation` | 144 | `ttnn::operations::experimental::transformer::NlpCreateHeadsDeviceOperation::Interleaved` | 142 (99%) |
| `SDPAOperation` | 144 | `ttnn::prim::SDPAOperation::SDPAProgramFactory` | 142 (99%) |
| `NLPConcatHeadsDeviceOperation` | 144 | `ttnn::experimental::prim::NLPConcatHeadsProgramFactory` | 142 (99%) |
| `MinimalMatmulDeviceOperation` | 144 | `ttnn::experimental::prim::MinimalMatmulProgramFactory` | 142 (99%) |
| `PagedFillCacheDeviceOperation` | 144 | `ttnn::experimental::prim::PagedFillCacheProgramFactory` | 143 (99%) |
| `UnaryDeviceOperation` | 86 | `ttnn::operations::unary::UnaryDeviceOperation::ProgramFactory` | 84 (98%) |
| `TilizeDeviceOperation` | 21 | `ttnn::prim::TilizeMultiCoreBlockProgramFactory` | 20 (95%) |
| `TilizeDeviceOperation` | 17 | `ttnn::prim::TilizeMultiCoreDefaultProgramFactory` | 12 (71%) |
| `TilizeDeviceOperation` | 1 | `ttnn::prim::TilizeMultiCoreShardedProgramFactory` | 0 (0%) |
| `PermuteDeviceOperation` | 32 | `ttnn::operations::data_movement::PermuteDeviceOperation::MultiCoreBlockedGeneric` | 28 (88%) |
| `UntilizeWithUnpaddingDeviceOperation` | 14 | `ttnn::prim::UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` | 11 (79%) |
| `UntilizeWithUnpaddingDeviceOperation` | 2 | `ttnn::prim::UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` | 1 (50%) |
| `ReshapeViewDeviceOperation` | 12 | `ttnn::prim::ReshapeViewRMProgramFactory` | 11 (92%) |
| `ScatterDeviceOperation` | 8 | `ttnn::prim::ScatterProgramFactory` | 7 (88%) |
| `FillPadDeviceOperation` | 6 | `ttnn::prim::FillPadProgramFactory` | 5 (83%) |
| `PadDeviceOperation` | 6 | `ttnn::prim::PadTileMulticoreProgramFactory` | 5 (83%) |
| `RepeatDeviceOperation` | 2 | `ttnn::prim::RepeatProgramFactoryHigherDim` | 1 (50%) |

35 op(s), 43 distinct (op, factory) pair(s), 422258 device-op launch(es).
