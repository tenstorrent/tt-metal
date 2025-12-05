#dram = #ttnn.buffer_type<dram>
#loc1 = loc("p0.2")
#loc2 = loc("p1.7")
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102208, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073128576, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 64 + d2, d3), <1x1>, memref<24x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 64 + d2, d3), <1x1>, memref<24x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<24x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1536 + d1 * 128 + d2, d3), <1x1>, memref<48x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<48x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<24x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module @SyncTensorsGraph.16 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.16 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]> loc(#loc)
      func.func @main_const_eval_0() -> tensor<1x12x39x39xbf16, #ttnn_layout> attributes {const_eval} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device loc(#loc)
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 0.0883789062 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<1x12x39x39>}> : (!ttnn.device) -> tensor<1x12x39x39xbf16, #ttnn_layout> loc(#loc)
        return %1 : tensor<1x12x39x39xbf16, #ttnn_layout> loc(#loc)
      } loc(#loc)
      func.func @main(%arg0: tensor<1x12x39x128xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("p0.2"), %arg1: tensor<1x12x39x128xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_1"} loc("p1.7")) -> (tensor<1x12x39x39xbf16, #ttnn_layout> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = ttcore.load_cached(@main_const_eval_0, []) : () -> tensor<1x12x39x39xbf16, #ttnn_layout> loc(#loc)
        %1 = "ttnn.reshape"(%arg1) <{shape = [12 : i32, 39 : i32, 128 : i32]}> : (tensor<1x12x39x128xbf16, #ttnn_layout1>) -> tensor<12x39x128xbf16, #ttnn_layout2> loc(#loc3)
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<1x12x39x128xbf16, #ttnn_layout1>) -> () loc(#loc3)
        %2 = "ttnn.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x39x128xbf16, #ttnn_layout1>) -> tensor<1x12x128x39xbf16, #ttnn_layout3> loc(#loc4)
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x12x39x128xbf16, #ttnn_layout1>) -> () loc(#loc4)
        %3 = "ttnn.reshape"(%2) <{shape = [12 : i32, 128 : i32, 39 : i32]}> : (tensor<1x12x128x39xbf16, #ttnn_layout3>) -> tensor<12x128x39xbf16, #ttnn_layout4> loc(#loc5)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x12x128x39xbf16, #ttnn_layout3>) -> () loc(#loc5)
        %4 = "ttnn.matmul"(%1, %3) <{transpose_a = false, transpose_b = false}> : (tensor<12x39x128xbf16, #ttnn_layout2>, tensor<12x128x39xbf16, #ttnn_layout4>) -> tensor<12x39x39xbf16, #ttnn_layout5> loc(#loc6)
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<12x128x39xbf16, #ttnn_layout4>) -> () loc(#loc6)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<12x39x128xbf16, #ttnn_layout2>) -> () loc(#loc6)
        %5 = "ttnn.reshape"(%4) <{shape = [1 : i32, 12 : i32, 39 : i32, 39 : i32]}> : (tensor<12x39x39xbf16, #ttnn_layout5>) -> tensor<1x12x39x39xbf16, #ttnn_layout> loc(#loc7)
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<12x39x39xbf16, #ttnn_layout5>) -> () loc(#loc7)
        %6 = "ttnn.multiply"(%5, %0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x12x39x39xbf16, #ttnn_layout>, tensor<1x12x39x39xbf16, #ttnn_layout>) -> tensor<1x12x39x39xbf16, #ttnn_layout> loc(#loc8)
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x12x39x39xbf16, #ttnn_layout>) -> () loc(#loc8)
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<1x12x39x39xbf16, #ttnn_layout>) -> () loc(#loc8)
        return %6 : tensor<1x12x39x39xbf16, #ttnn_layout> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("reshape.10")
#loc4 = loc("transpose.4")
#loc5 = loc("reshape.6")
#loc6 = loc("dot.11")
#loc7 = loc("reshape.12")
#loc8 = loc("multiply.14")
