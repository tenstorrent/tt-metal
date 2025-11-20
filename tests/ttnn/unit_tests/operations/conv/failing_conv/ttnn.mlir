#dram = #ttnn.buffer_type<dram>
#loc1 = loc("p0.1")
#loc2 = loc("p1.3")
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101952, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073130112, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 3 + d2, d3), <1x1>, memref<98304x3xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 256 + d2, d3), <1x1>, memref<1024x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2, d3), <1x1>, memref<1024x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 65536 + d1 * 256 + d2, d3), <1x1>, memref<2048x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 65536 + d1 * 65536 + d2, d3), <1x1>, memref<2048x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 65536 + d1 * 65536 + d2, d3), <1x1>, memref<65536x128xbf16, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 16384 + d2, d3), <1x1>, memref<512x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 128 + d2, d3), <1x1>, memref<512x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module @SyncTensorsGraph.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]> loc(#loc)
      func.func @main(%arg0: tensor<256x128x3x3xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.conv2d_weight, ttir.name = "l__self___b_conv_weight"} loc("p0.1"), %arg1: tensor<1x128x256x256xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("p1.3")) -> (tensor<1x256x128x128xbf16, #ttnn_layout2> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device loc(#loc3)
        %1 = "ttnn.permute"(%arg1) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x128x256x256xbf16, #ttnn_layout1>) -> tensor<1x256x256x128xbf16, #ttnn_layout3> loc(#loc4)
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<1x128x256x256xbf16, #ttnn_layout1>) -> () loc(#loc4)
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 65536 : i32, 128 : i32]}> : (tensor<1x256x256x128xbf16, #ttnn_layout3>) -> tensor<1x1x65536x128xbf16, #ttnn_layout4> loc(#loc7)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x256x256x128xbf16, #ttnn_layout3>) -> () loc(#loc7)
        %3 = "ttnn.to_layout"(%2) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1x65536x128xbf16, #ttnn_layout4>) -> tensor<1x1x65536x128xbf16, #ttnn_layout5> loc(#loc5)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1x65536x128xbf16, #ttnn_layout4>) -> () loc(#loc5)
        %4 = "ttnn.conv2d"(%3, %arg0, %0) <{batch_size = 1 : i32, conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, deallocate_activation = false, reallocate_halo_output = false, act_block_h_override = 0, act_block_w_div = 1, reshard_if_not_optimal = false, override_sharding_config = false, transpose_shards = false, output_layout = tile, enable_act_double_buffer = false, enable_weights_double_buffer = false, in_place = false, enable_kernel_stride_folding = false>, dilation = array<i32: 1, 1>, dtype = #ttcore.supportedDataTypes<bf16>, groups = 1 : i32, in_channels = 128 : i32, input_height = 256 : i32, input_width = 256 : i32, kernel_size = array<i32: 3, 3>, out_channels = 256 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> : (tensor<1x1x65536x128xbf16, #ttnn_layout5>, tensor<256x128x3x3xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x1x16384x256xbf16, #ttnn_layout6> loc(#loc3)
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x65536x128xbf16, #ttnn_layout5>) -> () loc(#loc3)
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<256x128x3x3xbf16, #ttnn_layout>) -> () loc(#loc3)
        %5 = "ttnn.reshape"(%4) <{shape = [1 : i32, 128 : i32, 128 : i32, 256 : i32]}> : (tensor<1x1x16384x256xbf16, #ttnn_layout6>) -> tensor<1x128x128x256xbf16, #ttnn_layout7> loc(#loc6)
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x16384x256xbf16, #ttnn_layout6>) -> () loc(#loc6)
        %6 = "ttnn.permute"(%5) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x128x128x256xbf16, #ttnn_layout7>) -> tensor<1x256x128x128xbf16, #ttnn_layout2> loc(#loc3)
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x128x128x256xbf16, #ttnn_layout7>) -> () loc(#loc3)
        return %6 : tensor<1x256x128x128xbf16, #ttnn_layout2> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("convolution.5")
#loc4 = loc("convolution.5_input"(#loc3))
#loc5 = loc("convolution.5_workaround"(#loc3))
#loc6 = loc("convolution.5_reshape"(#loc3))
#loc7 = loc("convolution.5_input_reshape"(#loc4))
