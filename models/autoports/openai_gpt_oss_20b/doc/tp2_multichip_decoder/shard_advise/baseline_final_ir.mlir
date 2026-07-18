#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <blackhole>, grid = 10x11, coord_translation_offsets = 2x1, l1_size = 1572864, num_dram_channels = 8, dram_channel_size = 4278190080, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 64, noc_dram_address_align_bytes = 64, l1_unreserved_base = 111360, erisc_l1_unreserved_base = 87872, dram_unreserved_base = 1048704, dram_unreserved_end = 4276464000, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x8, dram_bank_to_logical_worker_noc0 = [(9, 0), (0, 0), (7, 0), (3, 0), (9, 7), (1, 7), (6, 7), (4, 7)], dram_bank_to_logical_worker_noc1 = [(9, 0), (0, 0), (7, 0), (3, 0), (9, 7), (1, 7), (6, 7), (4, 7)]}, {arch = <blackhole>, grid = 10x11, coord_translation_offsets = 2x1, l1_size = 1572864, num_dram_channels = 8, dram_channel_size = 4278190080, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 64, noc_dram_address_align_bytes = 64, l1_unreserved_base = 111360, erisc_l1_unreserved_base = 87872, dram_unreserved_base = 1048704, dram_unreserved_end = 4276464000, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x8, dram_bank_to_logical_worker_noc0 = [(9, 0), (0, 0), (7, 0), (3, 0), (9, 7), (1, 7), (6, 7), (4, 7)], dram_bank_to_logical_worker_noc1 = [(9, 0), (0, 0), (7, 0), (3, 0), (9, 7), (1, 7), (6, 7), (4, 7)]}], [0, 1], [1 : i32, 1 : i32], [ 0x0x0x0], [<[0, 8, 0], [1, 3, 0]>, <[0, 9, 0], [1, 2, 0]>]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x2880xbf16, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3), <1x1>, memref<1024x64xbf16, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 96 + d2, d3), <1x1>, memref<3x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<90x160x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x160x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<90x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<2880x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout14 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<2880x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout17 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout18 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout19 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout20 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x10>, memref<1x9x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (9,0)>]>>
#ttnn_layout21 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x90>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,7)>, #ttnn.core_range<(0,8), (1,8)>]>>
#ttnn_layout22 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x45>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,3)>, #ttnn.core_range<(0,4), (0,4)>]>>
#ttnn_layout23 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x80>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,6)>, #ttnn.core_range<(0,7), (2,7)>]>>
#ttnn_layout24 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <10x11>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout25 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <10x11>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout26 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 32 + d2, d3), <10x11>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout27 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <10x11>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout28 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <10x11>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout29 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <10x11>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout30 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout31 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x90>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,7)>, #ttnn.core_range<(0,8), (1,8)>]>>
#ttnn_layout33 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x10>, memref<1x9x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (9,0)>]>>
#ttnn_layout34 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x10>, memref<1x9x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (9,0)>]>>
#ttnn_layout35 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x90x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout36 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout37 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout38 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <10x11>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout39 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <10x11>, memref<1x1x!ttcore.tile<32x32, u16>, #l1>, <interleaved>>
#ttnn_layout40 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout41 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x10>, memref<1x9x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (9,0)>]>>
#ttnn_layout42 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x90>, memref<32x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,7)>, #ttnn.core_range<(0,8), (1,8)>]>>
#ttnn_layout43 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <10x11>, memref<1x27x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout44 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x90>, memref<32x2x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,7)>, #ttnn.core_range<(0,8), (1,8)>]>>
#ttnn_layout45 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <10x11>, memref<1x53x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout46 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <10x11>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout47 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x90>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,7)>, #ttnn.core_range<(0,8), (1,8)>]>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<10x11, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x8>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 8, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 8) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @decode(%arg0: tensor<1x1x1x2880xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<1x8x128x64xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg2: tensor<1x8x128x64xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg3: tensor<1xsi32, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg4: tensor<1x1x90x32xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg5: tensor<2880x5120xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg6: tensor<1x1x5120xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg7: tensor<1x1x128x64xbf16, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg8: tensor<1x1x128x64xbf16, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg9: tensor<64x32xbf16, #ttnn_layout7> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg10: tensor<4096x2880xbf16, #ttnn_layout8> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg11: tensor<1x1x2880xbf16, #ttnn_layout9> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg12: tensor<1x1x90x32xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg13: tensor<2880x32xbf16, #ttnn_layout10> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg14: tensor<1x32xf32, #ttnn_layout11> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg15: tensor<32x2880x5760xbf16, #ttnn_layout12> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg16: tensor<32x1x5760xbf16, #ttnn_layout13> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg17: tensor<32x2880x2880xbf16, #ttnn_layout14> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg18: tensor<32x1x2880xbf16, #ttnn_layout15> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> (tensor<1x1x1x2880xbf16, #ttnn_layout16>, tensor<1x8x128x64xbf16, #ttnn_layout17>, tensor<1x8x128x64xbf16, #ttnn_layout17>) attributes {tt.function_type = "forward_device"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{fill_value = 1.000000e+00 : f32, shape = #ttnn.shape<32x1x2880>}> : (!ttnn.device) -> tensor<32x1x2880xbf16, #ttnn_layout15>
        %2 = "ttnn.full"(%0) <{fill_value = 1.703125 : f32, shape = #ttnn.shape<32x1x2880>}> : (!ttnn.device) -> tensor<32x1x2880xbf16, #ttnn_layout15>
        %3 = "ttnn.zeros"(%0) <{shape = #ttnn.shape<1x32>}> : (!ttnn.device) -> tensor<1x32xbf16, #ttnn_layout18>
        %4 = "ttnn.reshape"(%arg4) <{shape = [2880 : i32]}> : (tensor<1x1x90x32xbf16, #ttnn_layout3>) -> tensor<2880xbf16, #ttnn_layout19>
        %5 = "ttnn.to_layout"(%arg0) : (tensor<1x1x1x2880xbf16, #ttnn_layout>) -> tensor<1x1x1x2880xbf16, #ttnn_layout16>
        %6 = "ttnn.to_memory_config"(%5) : (tensor<1x1x1x2880xbf16, #ttnn_layout16>) -> tensor<1x1x1x2880xbf16, #ttnn_layout20>
        %7 = "ttnn.to_memory_config"(%4) : (tensor<2880xbf16, #ttnn_layout19>) -> tensor<2880xbf16, #ttnn_layout21>
        %8 = "ttnn.rms_norm"(%6, %7) <{epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x1x2880xbf16, #ttnn_layout20>, tensor<2880xbf16, #ttnn_layout21>) -> tensor<1x1x1x2880xbf16, #ttnn_layout20>
        %9 = "ttnn.to_memory_config"(%8) : (tensor<1x1x1x2880xbf16, #ttnn_layout20>) -> tensor<1x1x1x2880xbf16, #ttnn_layout22>
        %10 = "ttnn.linear"(%9, %arg5, %arg6) <{matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 2, out_block_h = 1, out_block_w = 2, per_core_m = 1, per_core_n = 2, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = false}> : (tensor<1x1x1x2880xbf16, #ttnn_layout22>, tensor<2880x5120xbf16, #ttnn_layout4>, tensor<1x1x5120xbf16, #ttnn_layout5>) -> tensor<1x1x1x5120xbf16, #ttnn_layout23>
        %11 = "ttnn.to_memory_config"(%10) : (tensor<1x1x1x5120xbf16, #ttnn_layout23>) -> tensor<1x1x1x5120xbf16, #ttnn_layout24>
        %12 = "ttnn.reshape"(%11) <{shape = [1 : i32, 1 : i32, 5120 : i32]}> : (tensor<1x1x1x5120xbf16, #ttnn_layout24>) -> tensor<1x1x5120xbf16, #ttnn_layout25>
        %query, %key, %value = "ttnn.split_query_key_value_and_split_heads"(%12) <{num_heads = 64 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x1x5120xbf16, #ttnn_layout25>) -> (tensor<1x64x1x64xbf16, #ttnn_layout26>, tensor<1x8x1x64xbf16, #ttnn_layout27>, tensor<1x8x1x64xbf16, #ttnn_layout27>)
        %13 = "ttnn.reshape"(%value) <{shape = [1 : i32, 1 : i32, 8 : i32, 64 : i32]}> : (tensor<1x8x1x64xbf16, #ttnn_layout27>) -> tensor<1x1x8x64xbf16, #ttnn_layout28>
        %14 = "ttnn.reshape"(%query) <{shape = [1 : i32, 1 : i32, 64 : i32, 64 : i32]}> : (tensor<1x64x1x64xbf16, #ttnn_layout26>) -> tensor<1x1x64x64xbf16, #ttnn_layout29>
        %15 = "ttnn.reshape"(%key) <{shape = [1 : i32, 1 : i32, 8 : i32, 64 : i32]}> : (tensor<1x8x1x64xbf16, #ttnn_layout27>) -> tensor<1x1x8x64xbf16, #ttnn_layout28>
        %16 = "ttnn.to_layout"(%arg1) : (tensor<1x8x128x64xbf16, #ttnn_layout1>) -> tensor<1x8x128x64xbf16, #ttnn_layout17>
        %17 = "ttnn.to_memory_config"(%15) : (tensor<1x1x8x64xbf16, #ttnn_layout28>) -> tensor<1x1x8x64xbf16, #ttnn_layout30>
        "ttnn.paged_update_cache"(%16, %17, %arg3) <{share_cache = false}> : (tensor<1x8x128x64xbf16, #ttnn_layout17>, tensor<1x1x8x64xbf16, #ttnn_layout30>, tensor<1xsi32, #ttnn_layout2>) -> ()
        %18 = "ttnn.to_layout"(%arg2) : (tensor<1x8x128x64xbf16, #ttnn_layout1>) -> tensor<1x8x128x64xbf16, #ttnn_layout17>
        %19 = "ttnn.to_memory_config"(%13) : (tensor<1x1x8x64xbf16, #ttnn_layout28>) -> tensor<1x1x8x64xbf16, #ttnn_layout30>
        "ttnn.paged_update_cache"(%18, %19, %arg3) <{share_cache = false}> : (tensor<1x8x128x64xbf16, #ttnn_layout17>, tensor<1x1x8x64xbf16, #ttnn_layout30>, tensor<1xsi32, #ttnn_layout2>) -> ()
        %20 = "ttnn.to_layout"(%arg1) : (tensor<1x8x128x64xbf16, #ttnn_layout1>) -> tensor<1x8x128x64xbf16, #ttnn_layout17>
        %21 = "ttnn.to_layout"(%arg2) : (tensor<1x8x128x64xbf16, #ttnn_layout1>) -> tensor<1x8x128x64xbf16, #ttnn_layout17>
        %22 = "ttnn.to_memory_config"(%14) : (tensor<1x1x64x64xbf16, #ttnn_layout29>) -> tensor<1x1x64x64xbf16, #ttnn_layout31>
        %23 = "ttnn.scaled_dot_product_attention_decode"(%22, %20, %21, %arg3) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>, scale = 1.250000e-01 : f32}> : (tensor<1x1x64x64xbf16, #ttnn_layout31>, tensor<1x8x128x64xbf16, #ttnn_layout17>, tensor<1x8x128x64xbf16, #ttnn_layout17>, tensor<1xsi32, #ttnn_layout2>) -> tensor<1x1x64x64xbf16, #ttnn_layout31>
        %24 = "ttnn.reshape"(%23) <{shape = [1 : i32, 1 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x1x64x64xbf16, #ttnn_layout31>) -> tensor<1x1x1x4096xbf16, #ttnn_layout24>
        %25 = "ttnn.slice_static"(%24) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 1 : i32, 4096 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x4096xbf16, #ttnn_layout24>) -> tensor<1x1x1x4096xbf16, #ttnn_layout24>
        %26 = "ttnn.linear"(%25, %arg10, %arg11) <{matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 9>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 1, out_block_w = 1, per_core_m = 1, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = false}> : (tensor<1x1x1x4096xbf16, #ttnn_layout24>, tensor<4096x2880xbf16, #ttnn_layout8>, tensor<1x1x2880xbf16, #ttnn_layout9>) -> tensor<1x1x1x2880xbf16, #ttnn_layout32>
        %27 = "ttnn.to_layout"(%arg0) : (tensor<1x1x1x2880xbf16, #ttnn_layout>) -> tensor<1x1x1x2880xbf16, #ttnn_layout16>
        %28 = "ttnn.to_memory_config"(%27) : (tensor<1x1x1x2880xbf16, #ttnn_layout16>) -> tensor<1x1x1x2880xbf16, #ttnn_layout32>
        %29 = "ttnn.add"(%28, %26) : (tensor<1x1x1x2880xbf16, #ttnn_layout32>, tensor<1x1x1x2880xbf16, #ttnn_layout32>) -> tensor<1x1x1x2880xbf16, #ttnn_layout32>
        %30 = "ttnn.reshape"(%arg12) <{shape = [2880 : i32]}> : (tensor<1x1x90x32xbf16, #ttnn_layout3>) -> tensor<2880xbf16, #ttnn_layout19>
        %31 = "ttnn.to_memory_config"(%29) : (tensor<1x1x1x2880xbf16, #ttnn_layout32>) -> tensor<1x1x1x2880xbf16, #ttnn_layout20>
        %32 = "ttnn.to_memory_config"(%30) : (tensor<2880xbf16, #ttnn_layout19>) -> tensor<2880xbf16, #ttnn_layout21>
        %33 = "ttnn.rms_norm"(%31, %32) <{epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x1x2880xbf16, #ttnn_layout20>, tensor<2880xbf16, #ttnn_layout21>) -> tensor<1x1x1x2880xbf16, #ttnn_layout20>
        %34 = "ttnn.reshape"(%33) <{shape = [1 : i32, 2880 : i32]}> : (tensor<1x1x1x2880xbf16, #ttnn_layout20>) -> tensor<1x2880xbf16, #ttnn_layout33>
        %35 = "ttnn.typecast"(%34) : (tensor<1x2880xbf16, #ttnn_layout33>) -> tensor<1x2880xf32, #ttnn_layout34>
        %36 = "ttnn.to_memory_config"(%35) : (tensor<1x2880xf32, #ttnn_layout34>) -> tensor<1x2880xf32, #ttnn_layout35>
        %37 = "ttnn.linear"(%36, %arg13, %arg14) <{matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<1, 1>, in0_block_w = 90, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 1, out_block_w = 1, per_core_m = 1, per_core_n = 1, fuse_batch = true, mcast_in0 = false, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = false}> : (tensor<1x2880xf32, #ttnn_layout35>, tensor<2880x32xbf16, #ttnn_layout10>, tensor<1x32xf32, #ttnn_layout11>) -> tensor<1x32xf32, #ttnn_layout36>
        %38 = "ttnn.typecast"(%37) : (tensor<1x32xf32, #ttnn_layout36>) -> tensor<1x32xbf16, #ttnn_layout37>
        %39 = "ttnn.to_memory_config"(%38) : (tensor<1x32xbf16, #ttnn_layout37>) -> tensor<1x32xbf16, #ttnn_layout38>
        %values, %indices = "ttnn.topk"(%39) <{dim = 1 : i32, k = 4 : i32, largest = true, sorted = true}> : (tensor<1x32xbf16, #ttnn_layout38>) -> (tensor<1x4xbf16, #ttnn_layout38>, tensor<1x4xui16, #ttnn_layout39>)
        %40 = "ttnn.softmax"(%values) <{dimension = 1 : si32, numericStable = false}> : (tensor<1x4xbf16, #ttnn_layout38>) -> tensor<1x4xbf16, #ttnn_layout40>
        %41 = "ttnn.to_memory_config"(%40) : (tensor<1x4xbf16, #ttnn_layout40>) -> tensor<1x4xbf16, #ttnn_layout38>
        %42 = "ttnn.scatter"(%3, %indices, %41) <{dim = 1 : i32, scatter_reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x32xbf16, #ttnn_layout18>, tensor<1x4xui16, #ttnn_layout39>, tensor<1x4xbf16, #ttnn_layout38>) -> tensor<1x32xbf16, #ttnn_layout40>
        %43 = "ttnn.reshape"(%33) <{shape = [1 : i32, 1 : i32, 2880 : i32]}> : (tensor<1x1x1x2880xbf16, #ttnn_layout20>) -> tensor<1x1x2880xbf16, #ttnn_layout41>
        %44 = "ttnn.repeat"(%43) <{repeat_dims = #ttnn.shape<32x1x1>}> : (tensor<1x1x2880xbf16, #ttnn_layout41>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %45 = "ttnn.to_memory_config"(%44) : (tensor<32x1x2880xbf16, #ttnn_layout42>) -> tensor<32x1x2880xbf16, #ttnn_layout43>
        %46 = "ttnn.matmul"(%45, %arg15) <{transpose_a = false, transpose_b = false}> : (tensor<32x1x2880xbf16, #ttnn_layout43>, tensor<32x2880x5760xbf16, #ttnn_layout12>) -> tensor<32x1x5760xbf16, #ttnn_layout13>
        %47 = "ttnn.to_memory_config"(%46) : (tensor<32x1x5760xbf16, #ttnn_layout13>) -> tensor<32x1x5760xbf16, #ttnn_layout44>
        %48 = "ttnn.add"(%47, %arg16) : (tensor<32x1x5760xbf16, #ttnn_layout44>, tensor<32x1x5760xbf16, #ttnn_layout13>) -> tensor<32x1x5760xbf16, #ttnn_layout44>
        %49 = "ttnn.to_memory_config"(%48) : (tensor<32x1x5760xbf16, #ttnn_layout44>) -> tensor<32x1x5760xbf16, #ttnn_layout45>
        %50 = "ttnn.slice_static"(%49) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 1 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 2 : i32]}> : (tensor<32x1x5760xbf16, #ttnn_layout45>) -> tensor<32x1x2880xbf16, #ttnn_layout43>
        %51 = "ttnn.to_memory_config"(%48) : (tensor<32x1x5760xbf16, #ttnn_layout44>) -> tensor<32x1x5760xbf16, #ttnn_layout45>
        %52 = "ttnn.slice_static"(%51) <{begins = [0 : i32, 0 : i32, 1 : i32], ends = [32 : i32, 1 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 2 : i32]}> : (tensor<32x1x5760xbf16, #ttnn_layout45>) -> tensor<32x1x2880xbf16, #ttnn_layout43>
        %53 = "ttnn.clamp_scalar"(%50) <{max = 7.000000e+00 : f32, min = -3.40282347E+38 : f32}> : (tensor<32x1x2880xbf16, #ttnn_layout43>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %54 = "ttnn.clamp_scalar"(%52) <{max = 7.000000e+00 : f32, min = -7.000000e+00 : f32}> : (tensor<32x1x2880xbf16, #ttnn_layout43>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %55 = "ttnn.to_memory_config"(%2) : (tensor<32x1x2880xbf16, #ttnn_layout15>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %56 = "ttnn.multiply"(%53, %55) : (tensor<32x1x2880xbf16, #ttnn_layout42>, tensor<32x1x2880xbf16, #ttnn_layout42>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %57 = "ttnn.sigmoid"(%56) : (tensor<32x1x2880xbf16, #ttnn_layout42>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %58 = "ttnn.multiply"(%53, %57) : (tensor<32x1x2880xbf16, #ttnn_layout42>, tensor<32x1x2880xbf16, #ttnn_layout42>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %59 = "ttnn.to_memory_config"(%1) : (tensor<32x1x2880xbf16, #ttnn_layout15>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %60 = "ttnn.add"(%54, %59) : (tensor<32x1x2880xbf16, #ttnn_layout42>, tensor<32x1x2880xbf16, #ttnn_layout42>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %61 = "ttnn.multiply"(%60, %58) : (tensor<32x1x2880xbf16, #ttnn_layout42>, tensor<32x1x2880xbf16, #ttnn_layout42>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %62 = "ttnn.to_memory_config"(%61) : (tensor<32x1x2880xbf16, #ttnn_layout42>) -> tensor<32x1x2880xbf16, #ttnn_layout43>
        %63 = "ttnn.matmul"(%62, %arg17) <{transpose_a = false, transpose_b = false}> : (tensor<32x1x2880xbf16, #ttnn_layout43>, tensor<32x2880x2880xbf16, #ttnn_layout14>) -> tensor<32x1x2880xbf16, #ttnn_layout15>
        %64 = "ttnn.to_memory_config"(%63) : (tensor<32x1x2880xbf16, #ttnn_layout15>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %65 = "ttnn.add"(%64, %arg18) : (tensor<32x1x2880xbf16, #ttnn_layout42>, tensor<32x1x2880xbf16, #ttnn_layout15>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %66 = "ttnn.reshape"(%42) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32xbf16, #ttnn_layout40>) -> tensor<32x1x1xbf16, #ttnn_layout46>
        %67 = "ttnn.multiply"(%65, %66) : (tensor<32x1x2880xbf16, #ttnn_layout42>, tensor<32x1x1xbf16, #ttnn_layout46>) -> tensor<32x1x2880xbf16, #ttnn_layout42>
        %68 = "ttnn.to_memory_config"(%67) : (tensor<32x1x2880xbf16, #ttnn_layout42>) -> tensor<32x1x2880xbf16, #ttnn_layout43>
        %69 = "ttnn.sum"(%68) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<32x1x2880xbf16, #ttnn_layout43>) -> tensor<1x2880xbf16, #ttnn_layout47>
        %70 = "ttnn.to_memory_config"(%69) : (tensor<1x2880xbf16, #ttnn_layout47>) -> tensor<1x2880xbf16, #ttnn_layout38>
        %71 = "ttnn.reshape"(%70) <{shape = [1 : i32, 1 : i32, 1 : i32, 2880 : i32]}> : (tensor<1x2880xbf16, #ttnn_layout38>) -> tensor<1x1x1x2880xbf16, #ttnn_layout28>
        %72 = "ttnn.add"(%29, %71) : (tensor<1x1x1x2880xbf16, #ttnn_layout32>, tensor<1x1x1x2880xbf16, #ttnn_layout28>) -> tensor<1x1x1x2880xbf16, #ttnn_layout32>
        %73 = "ttnn.to_memory_config"(%72) : (tensor<1x1x1x2880xbf16, #ttnn_layout32>) -> tensor<1x1x1x2880xbf16, #ttnn_layout16>
        return %73, %16, %18 : tensor<1x1x1x2880xbf16, #ttnn_layout16>, tensor<1x8x128x64xbf16, #ttnn_layout17>, tensor<1x8x128x64xbf16, #ttnn_layout17>
      }
    }
  }
}
