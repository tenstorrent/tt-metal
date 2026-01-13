#dram = #ttnn.buffer_type<dram>
#loc = loc(unknown)
#loc2 = loc("p0.1")
#loc3 = loc("p1.15")
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102656, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073125888, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <1x1>, memref<32768x1536xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x1572864xbf16, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <1x1>, memref<32768x1536xbf16, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <1x1>, memref<1024x48x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x49152x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <1x1>, memref<32x48x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1xui32, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x49152x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module @SyncTensorsGraph.19 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.19 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]> loc(#loc)
      func.func private @main_const_eval_0() -> tensor<1xsi32, #ttnn_layout> attributes {const_eval} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device loc(#loc)
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, fill_value = 32 : i32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<1>}> : (!ttnn.device) -> tensor<1xsi32, #ttnn_layout> loc(#loc)
        return %1 : tensor<1xsi32, #ttnn_layout> loc(#loc)
      } loc(#loc)
      func.func private @main_const_eval_1() -> tensor<1xsi32, #ttnn_layout> attributes {const_eval} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device loc(#loc)
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, fill_value = 0 : i32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<1>}> : (!ttnn.device) -> tensor<1xsi32, #ttnn_layout> loc(#loc)
        return %1 : tensor<1xsi32, #ttnn_layout> loc(#loc)
      } loc(#loc)
      func.func private @main_const_eval_2(%arg0: tensor<32x1024x1536xbf16, #ttnn_layout1> loc(unknown)) -> tensor<32x1572864xbf16, #ttnn_layout2> attributes {const_eval} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device loc(#loc)
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x1024x1536xbf16, #ttnn_layout1>, !ttnn.device) -> tensor<32x1024x1536xbf16, #ttnn_layout3> loc(#loc)
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<32x1024x1536xbf16, #ttnn_layout3>) -> tensor<32x1024x1536xbf16, #ttnn_layout4> loc(#loc)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<32x1024x1536xbf16, #ttnn_layout3>) -> () loc(#loc)
        %3 = "ttnn.reshape"(%2) <{shape = [32 : i32, 1572864 : i32]}> : (tensor<32x1024x1536xbf16, #ttnn_layout4>) -> tensor<32x1572864xbf16, #ttnn_layout5> loc(#loc10)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<32x1024x1536xbf16, #ttnn_layout4>) -> () loc(#loc10)
        %4 = "ttnn.to_layout"(%3) <{layout = #ttnn.layout<row_major>}> : (tensor<32x1572864xbf16, #ttnn_layout5>) -> tensor<32x1572864xbf16, #ttnn_layout2> loc(#loc11)
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<32x1572864xbf16, #ttnn_layout5>) -> () loc(#loc11)
        return %4 : tensor<32x1572864xbf16, #ttnn_layout2> loc(#loc)
      } loc(#loc)
      func.func @main(%arg0: tensor<1xsi32, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("p0.1"), %arg1: tensor<32x1024x1536xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___w"} loc("p1.15")) -> (tensor<1x1024x1536xbf16, #ttnn_layout7> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = ttcore.load_cached(@main_const_eval_0, []) : () -> tensor<1xsi32, #ttnn_layout> loc(#loc)
        %1 = ttcore.load_cached(@main_const_eval_1, []) : () -> tensor<1xsi32, #ttnn_layout> loc(#loc)
        %2 = ttcore.load_cached(@main_const_eval_2, [%arg1]) : (tensor<32x1024x1536xbf16, #ttnn_layout1>) -> tensor<32x1572864xbf16, #ttnn_layout2> loc(#loc)
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<32x1024x1536xbf16, #ttnn_layout1>) -> () loc(#loc)
        %3 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<1xsi32, #ttnn_layout6>) -> tensor<1xsi32, #ttnn_layout> loc(#loc4)
        %4 = "ttnn.lt"(%3, %1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1xsi32, #ttnn_layout>, tensor<1xsi32, #ttnn_layout>) -> tensor<1xbf16, #ttnn_layout8> loc(#loc5)
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1xsi32, #ttnn_layout>) -> () loc(#loc5)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1xsi32, #ttnn_layout>) -> () loc(#loc5)
        %5 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<1xsi32, #ttnn_layout6>) -> tensor<1xsi32, #ttnn_layout> loc(#loc6)
        %6 = "ttnn.add"(%5, %0) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<1xsi32, #ttnn_layout>, tensor<1xsi32, #ttnn_layout>) -> tensor<1xsi32, #ttnn_layout> loc(#loc7)
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1xsi32, #ttnn_layout>) -> () loc(#loc7)
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<1xsi32, #ttnn_layout>) -> () loc(#loc7)
        %7 = "ttnn.typecast"(%4) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1xbf16, #ttnn_layout8>) -> tensor<1xf32, #ttnn_layout9> loc(#loc12)
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1xbf16, #ttnn_layout8>) -> () loc(#loc12)
        %8 = "ttnn.typecast"(%6) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1xsi32, #ttnn_layout>) -> tensor<1xf32, #ttnn_layout9> loc(#loc12)
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1xsi32, #ttnn_layout>) -> () loc(#loc12)
        %9 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<1xsi32, #ttnn_layout6>) -> tensor<1xsi32, #ttnn_layout> loc(#loc12)
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1xsi32, #ttnn_layout6>) -> () loc(#loc12)
        %10 = "ttnn.typecast"(%9) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1xsi32, #ttnn_layout>) -> tensor<1xf32, #ttnn_layout9> loc(#loc12)
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<1xsi32, #ttnn_layout>) -> () loc(#loc12)
        %11 = "ttnn.where"(%7, %8, %10) : (tensor<1xf32, #ttnn_layout9>, tensor<1xf32, #ttnn_layout9>, tensor<1xf32, #ttnn_layout9>) -> tensor<1xf32, #ttnn_layout9> loc(#loc8)
        "ttnn.deallocate"(%10) <{force = false}> : (tensor<1xf32, #ttnn_layout9>) -> () loc(#loc8)
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<1xf32, #ttnn_layout9>) -> () loc(#loc8)
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1xf32, #ttnn_layout9>) -> () loc(#loc8)
        %12 = "ttnn.typecast"(%11) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<1xf32, #ttnn_layout9>) -> tensor<1xsi32, #ttnn_layout> loc(#loc12)
        "ttnn.deallocate"(%11) <{force = false}> : (tensor<1xf32, #ttnn_layout9>) -> () loc(#loc12)
        %13 = "ttnn.reshape"(%12) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xsi32, #ttnn_layout>) -> tensor<1x1xsi32, #ttnn_layout10> loc(#loc9)
        "ttnn.deallocate"(%12) <{force = false}> : (tensor<1xsi32, #ttnn_layout>) -> () loc(#loc9)
        %14 = "ttnn.typecast"(%13) <{dtype = #ttcore.supportedDataTypes<u32>}> : (tensor<1x1xsi32, #ttnn_layout10>) -> tensor<1x1xui32, #ttnn_layout11> loc(#loc11)
        "ttnn.deallocate"(%13) <{force = false}> : (tensor<1x1xsi32, #ttnn_layout10>) -> () loc(#loc11)
        %15 = "ttnn.to_layout"(%14) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1xui32, #ttnn_layout11>) -> tensor<1x1xui32, #ttnn_layout12> loc(#loc11)
        "ttnn.deallocate"(%14) <{force = false}> : (tensor<1x1xui32, #ttnn_layout11>) -> () loc(#loc11)
        %16 = "ttnn.embedding"(%15, %2) : (tensor<1x1xui32, #ttnn_layout12>, tensor<32x1572864xbf16, #ttnn_layout2>) -> tensor<1x1x1572864xbf16, #ttnn_layout13> loc(#loc1)
        "ttnn.deallocate"(%15) <{force = false}> : (tensor<1x1xui32, #ttnn_layout12>) -> () loc(#loc1)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<32x1572864xbf16, #ttnn_layout2>) -> () loc(#loc1)
        %17 = "ttnn.reshape"(%16) <{shape = [1 : i32, 1024 : i32, 1536 : i32]}> : (tensor<1x1x1572864xbf16, #ttnn_layout13>) -> tensor<1x1024x1536xbf16, #ttnn_layout7> loc(#loc13)
        "ttnn.deallocate"(%16) <{force = false}> : (tensor<1x1x1572864xbf16, #ttnn_layout13>) -> () loc(#loc13)
        return %17 : tensor<1x1024x1536xbf16, #ttnn_layout7> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("gather.17")
#loc4 = loc("compare.11_in_0_layout")
#loc5 = loc("compare.11")
#loc6 = loc("add.8_in_0_layout")
#loc7 = loc("add.8")
#loc8 = loc("select.12")
#loc9 = loc("reshape.13")
#loc10 = loc("gather.17_reshapeInput"(#loc1))
#loc11 = loc("gather.17_workaround"(#loc1))
#loc12 = loc("select.12_workaround"(#loc8))
#loc13 = loc("gather.17_reshapeOutput"(#loc1))
