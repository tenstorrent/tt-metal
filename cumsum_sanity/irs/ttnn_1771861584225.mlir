#dram = #ttnn.buffer_type<dram>
#loc1 = loc("p0.1")
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073119552, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1168640xsi32, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x36520x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1168640x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
module @SyncTensorsGraph.12 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.12 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]> loc(#loc)
      func.func @main(%arg0: tensor<1168640xsi32, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1168640xi64>>, ttir.name = "args_0"} loc("p0.1")) -> (tensor<1168640xsi32, #ttnn_layout1> {ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1168640xi64>>}) attributes {tt.function_type = "forward_device"} {
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<1168640xsi32, #ttnn_layout>) -> tensor<1168640xsi32, #ttnn_layout1> loc(#loc2)
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1168640xsi32, #ttnn_layout>) -> () loc(#loc2)
        %1 = "ttnn.reshape"(%0) <{shape = [1168640 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1168640xsi32, #ttnn_layout1>) -> tensor<1168640x1x1x1xsi32, #ttnn_layout2> loc(#loc4)
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<1168640xsi32, #ttnn_layout1>) -> () loc(#loc4)
        %2 = "ttnn.moreh_cumsum"(%1) <{dim = 0 : i64}> : (tensor<1168640x1x1x1xsi32, #ttnn_layout2>) -> tensor<1168640x1x1x1xsi32, #ttnn_layout2> loc(#loc3)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1168640x1x1x1xsi32, #ttnn_layout2>) -> () loc(#loc3)
        %3 = "ttnn.reshape"(%2) <{shape = [1168640 : i32]}> : (tensor<1168640x1x1x1xsi32, #ttnn_layout2>) -> tensor<1168640xsi32, #ttnn_layout1> loc(#loc5)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1168640x1x1x1xsi32, #ttnn_layout2>) -> () loc(#loc5)
        return %3 : tensor<1168640xsi32, #ttnn_layout1> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("reduce-window.10_in_0_layout")
#loc3 = loc("reduce-window.10")
#loc4 = loc("reduce-window.10_reshapeInput"(#loc3))
#loc5 = loc("reduce-window.10_reshapeOutput"(#loc3))
