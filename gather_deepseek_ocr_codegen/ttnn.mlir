#dram = #ttnn.buffer_type<dram>
#loc2 = loc("p1.5")
#loc3 = loc("p0.1")
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073119552, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1280 + d1, d2), <1x1>, memref<36520x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x40x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1280 + d1, d2), <1x1>, memref<40x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<913x1280xsi32, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<903x1280xbf16, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<29x40x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<29x40x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<29x40x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<36120x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1280 + d1, d2), <1x1>, memref<36520x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x36520x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x36520x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1168640xui32, #dram>, <interleaved>>
#ttnn_layout14 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1155840x1xbf16, #dram>, <interleaved>>
#ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1168640 + d1, d2), <1x1>, memref<36520x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module @SyncTensorsGraph.15 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.15 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]> loc(#loc)
      func.func private @main_const_eval_0() -> tensor<913x1280x1xui32, #ttnn_layout> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device loc(#loc1)
        %1 = "ttnn.arange"(%0) <{dtype = #ttcore.supportedDataTypes<u32>, end = 1280 : i64, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, start = 0 : i64, step = 1 : i64}> : (!ttnn.device) -> tensor<1280xui32, #ttnn_layout1> loc(#loc1)
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1280 : i32, 1 : i32]}> : (tensor<1280xui32, #ttnn_layout1>) -> tensor<1x1280x1xui32, #ttnn_layout2> loc(#loc1)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1280xui32, #ttnn_layout1>) -> () loc(#loc1)
        %3 = "ttnn.repeat"(%2) <{repeat_dims = #ttnn.shape<913x1x1>}> : (tensor<1x1280x1xui32, #ttnn_layout2>) -> tensor<913x1280x1xui32, #ttnn_layout> loc(#loc1)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1280x1xui32, #ttnn_layout2>) -> () loc(#loc1)
        return %3 : tensor<913x1280x1xui32, #ttnn_layout> loc(#loc)
      } loc(#loc)
      func.func private @main_const_eval_1() -> tensor<2x1xf32, #ttnn_layout3> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device loc(#loc1)
        %1 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<[[1.280000e+03], [1.000000e+00]]> : tensor<2x1xf32>}> : (!ttnn.device) -> tensor<2x1xf32, #ttnn_layout3> loc(#loc8)
        return %1 : tensor<2x1xf32, #ttnn_layout3> loc(#loc)
      } loc(#loc)
      func.func @main(%arg0: tensor<913x1280xsi32, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<913x1280xi64>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_1"} loc("p0.1"), %arg1: tensor<903x1280xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<903x1280xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("p1.5")) -> (tensor<913x1280xbf16, #ttnn_layout6> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<913x1280xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) attributes {tt.function_type = "forward_device"} {
        %0 = ttcore.load_cached(@main_const_eval_0, []) : () -> tensor<913x1280x1xui32, #ttnn_layout> loc(#loc)
        %1 = ttcore.load_cached(@main_const_eval_1, []) : () -> tensor<2x1xf32, #ttnn_layout3> loc(#loc)
        %2 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<913x1280xsi32, #ttnn_layout4>) -> tensor<913x1280xsi32, #ttnn_layout7> loc(#loc4)
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<913x1280xsi32, #ttnn_layout4>) -> () loc(#loc4)
        %3 = "ttnn.typecast"(%2) <{dtype = #ttcore.supportedDataTypes<u32>}> : (tensor<913x1280xsi32, #ttnn_layout7>) -> tensor<913x1280xui32, #ttnn_layout8> loc(#loc5)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<913x1280xsi32, #ttnn_layout7>) -> () loc(#loc5)
        %4 = "ttnn.reshape"(%3) <{shape = [913 : i32, 1280 : i32, 1 : i32]}> : (tensor<913x1280xui32, #ttnn_layout8>) -> tensor<913x1280x1xui32, #ttnn_layout> loc(#loc5)
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<913x1280xui32, #ttnn_layout8>) -> () loc(#loc5)
        %5 = "ttnn.concat"(%4, %0) <{dim = 2 : si32}> : (tensor<913x1280x1xui32, #ttnn_layout>, tensor<913x1280x1xui32, #ttnn_layout>) -> tensor<913x1280x2xui32, #ttnn_layout> loc(#loc6)
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<913x1280x1xui32, #ttnn_layout>) -> () loc(#loc6)
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<913x1280x1xui32, #ttnn_layout>) -> () loc(#loc6)
        %6 = "ttnn.to_layout"(%arg1) <{layout = #ttnn.layout<tile>}> : (tensor<903x1280xbf16, #ttnn_layout5>) -> tensor<903x1280xbf16, #ttnn_layout6> loc(#loc9)
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<903x1280xbf16, #ttnn_layout5>) -> () loc(#loc9)
        %7 = "ttnn.reshape"(%6) <{shape = [1155840 : i32, 1 : i32]}> : (tensor<903x1280xbf16, #ttnn_layout6>) -> tensor<1155840x1xbf16, #ttnn_layout9> loc(#loc10)
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<903x1280xbf16, #ttnn_layout6>) -> () loc(#loc10)
        %8 = "ttnn.typecast"(%5) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<913x1280x2xui32, #ttnn_layout>) -> tensor<913x1280x2xf32, #ttnn_layout10> loc(#loc11)
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<913x1280x2xui32, #ttnn_layout>) -> () loc(#loc11)
        %9 = "ttnn.matmul"(%8, %1) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<913x1280x2xf32, #ttnn_layout10>, tensor<2x1xf32, #ttnn_layout3>) -> tensor<913x1280x1xf32, #ttnn_layout10> loc(#loc2)
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<913x1280x2xf32, #ttnn_layout10>) -> () loc(#loc2)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<2x1xf32, #ttnn_layout3>) -> () loc(#loc2)
        %10 = "ttnn.reshape"(%9) <{shape = [1 : i32, 1168640 : i32]}> : (tensor<913x1280x1xf32, #ttnn_layout10>) -> tensor<1x1168640xf32, #ttnn_layout11> loc(#loc12)
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<913x1280x1xf32, #ttnn_layout10>) -> () loc(#loc12)
        %11 = "ttnn.typecast"(%10) <{dtype = #ttcore.supportedDataTypes<u32>}> : (tensor<1x1168640xf32, #ttnn_layout11>) -> tensor<1x1168640xui32, #ttnn_layout12> loc(#loc13)
        "ttnn.deallocate"(%10) <{force = false}> : (tensor<1x1168640xf32, #ttnn_layout11>) -> () loc(#loc13)
        %12 = "ttnn.to_layout"(%11) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1168640xui32, #ttnn_layout12>) -> tensor<1x1168640xui32, #ttnn_layout13> loc(#loc13)
        "ttnn.deallocate"(%11) <{force = false}> : (tensor<1x1168640xui32, #ttnn_layout12>) -> () loc(#loc13)
        %13 = "ttnn.to_layout"(%7) <{layout = #ttnn.layout<row_major>}> : (tensor<1155840x1xbf16, #ttnn_layout9>) -> tensor<1155840x1xbf16, #ttnn_layout14> loc(#loc13)
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1155840x1xbf16, #ttnn_layout9>) -> () loc(#loc13)
        %14 = "ttnn.embedding"(%12, %13) : (tensor<1x1168640xui32, #ttnn_layout13>, tensor<1155840x1xbf16, #ttnn_layout14>) -> tensor<1x1168640x1xbf16, #ttnn_layout15> loc(#loc7)
        "ttnn.deallocate"(%13) <{force = false}> : (tensor<1155840x1xbf16, #ttnn_layout14>) -> () loc(#loc7)
        "ttnn.deallocate"(%12) <{force = false}> : (tensor<1x1168640xui32, #ttnn_layout13>) -> () loc(#loc7)
        %15 = "ttnn.reshape"(%14) <{shape = [913 : i32, 1280 : i32]}> : (tensor<1x1168640x1xbf16, #ttnn_layout15>) -> tensor<913x1280xbf16, #ttnn_layout6> loc(#loc14)
        "ttnn.deallocate"(%14) <{force = false}> : (tensor<1x1168640x1xbf16, #ttnn_layout15>) -> () loc(#loc14)
        return %15 : tensor<913x1280xbf16, #ttnn_layout6> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("iota.11")
#loc4 = loc("convert.9_in_0_layout")
#loc5 = loc("convert.9")
#loc6 = loc("concatenate.12")
#loc7 = loc("gather.13")
#loc8 = loc("p1.5_constant"(#loc2))
#loc9 = loc("gather.13_reshapeInput_in_0_layout"(#loc7))
#loc10 = loc("gather.13_reshapeInput"(#loc7))
#loc11 = loc("p1.5_typecast"(#loc2))
#loc12 = loc("gather.13_reshapeStartIndices"(#loc7))
#loc13 = loc("gather.13_workaround"(#loc7))
#loc14 = loc("gather.13_reshapeOutput"(#loc7))
