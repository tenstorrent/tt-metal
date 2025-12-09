#dram = #ttnn.buffer_type<dram>
#loc1 = loc("p0.2")
#loc2 = loc("p1.7")
#loc3 = loc("p2.11")
#loc4 = loc("p3.41")
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102208, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073128576, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<96x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 256 + d2, d3), <1x1>, memref<256x3x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 256 + d2, d3), <1x1>, memref<256x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 256 + d1, d2), <1x1>, memref<8x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 256 + d2, d3), <1x1>, memref<256x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 256 + d1, d2), <1x1>, memref<256x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 256 + d1, d2), <1x1>, memref<256x3x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module @SyncTensorsGraph.47 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.47 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]> loc(#loc)
      func.func @main(%arg0: tensor<3072x3072xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___o_proj_weight"} loc("p0.2"), %arg1: tensor<1x32x256x96xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_1"} loc("p1.7"), %arg2: tensor<1x32x256x256xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("p2.11"), %arg3: tensor<1x256x3072xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_2"} loc("p3.41")) -> (tensor<1x256x3072xbf16, #ttnn_layout3> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = "ttnn.typecast"(%arg2) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x32x256x256xbf16, #ttnn_layout2>) -> tensor<1x32x256x256xf32, #ttnn_layout4> loc(#loc5)
        "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<1x32x256x256xbf16, #ttnn_layout2>) -> () loc(#loc5)
        %1 = "ttnn.softmax"(%0) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x256x256xf32, #ttnn_layout4>) -> tensor<1x32x256x256xf32, #ttnn_layout4> loc(#loc6)
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<1x32x256x256xf32, #ttnn_layout4>) -> () loc(#loc6)
        %2 = "ttnn.typecast"(%1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x32x256x256xf32, #ttnn_layout4>) -> tensor<1x32x256x256xbf16, #ttnn_layout2> loc(#loc7)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x32x256x256xf32, #ttnn_layout4>) -> () loc(#loc7)
        %3 = "ttnn.reshape"(%2) <{shape = [32 : i32, 256 : i32, 256 : i32]}> : (tensor<1x32x256x256xbf16, #ttnn_layout2>) -> tensor<32x256x256xbf16, #ttnn_layout5> loc(#loc8)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x32x256x256xbf16, #ttnn_layout2>) -> () loc(#loc8)
        %4 = "ttnn.reshape"(%arg1) <{shape = [32 : i32, 256 : i32, 96 : i32]}> : (tensor<1x32x256x96xbf16, #ttnn_layout1>) -> tensor<32x256x96xbf16, #ttnn_layout6> loc(#loc9)
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<1x32x256x96xbf16, #ttnn_layout1>) -> () loc(#loc9)
        %5 = "ttnn.matmul"(%3, %4) <{transpose_a = false, transpose_b = false}> : (tensor<32x256x256xbf16, #ttnn_layout5>, tensor<32x256x96xbf16, #ttnn_layout6>) -> tensor<32x256x96xbf16, #ttnn_layout6> loc(#loc10)
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<32x256x96xbf16, #ttnn_layout6>) -> () loc(#loc10)
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<32x256x256xbf16, #ttnn_layout5>) -> () loc(#loc10)
        %6 = "ttnn.reshape"(%5) <{shape = [1 : i32, 32 : i32, 256 : i32, 96 : i32]}> : (tensor<32x256x96xbf16, #ttnn_layout6>) -> tensor<1x32x256x96xbf16, #ttnn_layout1> loc(#loc11)
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<32x256x96xbf16, #ttnn_layout6>) -> () loc(#loc11)
        %7 = "ttnn.concatenate_heads"(%6) : (tensor<1x32x256x96xbf16, #ttnn_layout1>) -> tensor<1x256x3072xbf16, #ttnn_layout3> loc(#loc12)
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x32x256x96xbf16, #ttnn_layout1>) -> () loc(#loc12)
        %8 = "ttnn.reshape"(%7) <{shape = [256 : i32, 3072 : i32]}> : (tensor<1x256x3072xbf16, #ttnn_layout3>) -> tensor<256x3072xbf16, #ttnn_layout7> loc(#loc12)
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x256x3072xbf16, #ttnn_layout3>) -> () loc(#loc12)
        %9 = "ttnn.matmul"(%8, %arg0) <{transpose_a = false, transpose_b = true}> : (tensor<256x3072xbf16, #ttnn_layout7>, tensor<3072x3072xbf16, #ttnn_layout>) -> tensor<256x3072xbf16, #ttnn_layout7> loc(#loc14)
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<256x3072xbf16, #ttnn_layout7>) -> () loc(#loc14)
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<3072x3072xbf16, #ttnn_layout>) -> () loc(#loc14)
        %10 = "ttnn.add"(%9, %arg3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<256x3072xbf16, #ttnn_layout7>, tensor<1x256x3072xbf16, #ttnn_layout3>) -> tensor<1x256x3072xbf16, #ttnn_layout3> loc(#loc15)
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<256x3072xbf16, #ttnn_layout7>) -> () loc(#loc15)
        "ttnn.deallocate"(%arg3) <{force = false}> : (tensor<1x256x3072xbf16, #ttnn_layout3>) -> () loc(#loc15)
        return %10 : tensor<1x256x3072xbf16, #ttnn_layout3> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc5 = loc("convert.13")
#loc6 = loc("divide.30")
#loc7 = loc("convert.31")
#loc8 = loc("reshape.33")
#loc9 = loc("reshape.10")
#loc10 = loc("dot.34")
#loc11 = loc("reshape.35")
#loc12 = loc("reshape.38")
#loc13 = loc("add.45")
#loc14 = loc("add.45_decomp_matmul"(#loc13))
#loc15 = loc("add.45_decomp_add"(#loc13))
