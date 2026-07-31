#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <blackhole>, grid = 10x11, coord_translation_offsets = 2x1, l1_size = 1572864, num_dram_channels = 8, dram_channel_size = 4278190080, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 64, noc_dram_address_align_bytes = 64, l1_unreserved_base = 111360, erisc_l1_unreserved_base = 87872, dram_unreserved_base = 5848704, dram_unreserved_end = 4276464000, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x8, dram_bank_to_logical_worker_noc0 = [(9, 0), (0, 0), (7, 0), (3, 0), (9, 7), (1, 7), (6, 7), (4, 7)], dram_bank_to_logical_worker_noc1 = [(9, 0), (0, 0), (7, 0), (3, 0), (9, 7), (1, 7), (6, 7), (4, 7)]}], [0], [1 : i32], [ 0x0x0x0], [<[0, 2, 0], [3, 5, 0]>, <[0, 4, 0], [3, 4, 0]>, <[2, 8, 0], [3, 3, 0]>, <[2, 9, 0], [3, 2, 0]>]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x120x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3840 + d1 * 3840 + d2, d3), <1x1>, memref<120x288x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 8192 + d2, d3), <1x1>, memref<256x120x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3840 + d1 * 3840 + d2, d3), <1x1>, memref<120x480x!ttcore.tile<32x32, bfp_bf8>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 15360 + d1 * 15360 + d2, d3), <1x1>, memref<480x120x!ttcore.tile<32x32, bfp_bf8>, #dram>, <interleaved>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x120x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x11>, memref<1x11x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,0)>]>>
#ttnn_layout14 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x60>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,4)>, #ttnn.core_range<(0,5), (4,5)>]>>
#ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x8>, memref<1x15x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3840 + d1 * 3840 + d2, d3), <1x8>, memref<120x36x!ttcore.tile<32x32, bf16>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout17 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x96>, memref<1x3x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,7)>, #ttnn.core_range<(0,8), (7,8)>]>>
#ttnn_layout18 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x72>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,5)>, #ttnn.core_range<(0,6), (5,6)>]>>
#ttnn_layout19 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <32x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,1)>, #ttnn.core_range<(0,2), (9,2)>]>>
#ttnn_layout20 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <32x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(10,2), (10,2)>, #ttnn.core_range<(0,3), (10,4)>, #ttnn.core_range<(0,5), (8,5)>]>>
#ttnn_layout21 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <10x11>, memref<1x3x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout22 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout23 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <8x8>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout24 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x16>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,0)>, #ttnn.core_range<(0,1), (4,1)>]>>
#ttnn_layout25 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <10x11>, memref<1x5x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout26 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <10x11>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout27 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x8>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout28 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout29 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout30 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout31 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <16x1>, memref<2x16x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,0)>, #ttnn.core_range<(0,1), (4,1)>]>>
#ttnn_layout32 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32xsi32, #dram>, <interleaved>>
#ttnn_layout33 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x2xsi32, #dram>, <interleaved>>
#ttnn_layout34 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout35 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <7x8>, memref<5x2x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,6)>]>>
#ttnn_layout36 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <10x11>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,9)>]>>
#ttnn_layout37 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <10x11>, memref<1x3x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout38 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <10x11>, memref<1x3x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout39 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x60>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,4)>, #ttnn.core_range<(0,5), (4,5)>]>>
#ttnn_layout40 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3840 + d1 * 3840 + d2, d3), <1x8>, memref<120x60x!ttcore.tile<32x32, bfp_bf8>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout41 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x96>, memref<1x5x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,7)>, #ttnn.core_range<(0,8), (7,8)>]>>
#ttnn_layout42 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x8>, memref<1x60x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout43 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x480x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout44 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 15360 + d1 * 15360 + d2, d3), <1x8>, memref<480x15x!ttcore.tile<32x32, bfp_bf8>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<10x11, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x8>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 8, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 8) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @decode(%arg0: tensor<1x1x32x3840xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<1x1x120x32xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<1x1x3840x9216xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<1x1x16x32xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg4: tensor<1x1x16x32xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg5: tensor<1x32xui32, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg6: tensor<128x512xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg7: tensor<128x512xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg8: tensor<64x1x64x512xbf16, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg9: tensor<32xsi32, #ttnn_layout7> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg10: tensor<32x2xsi32, #ttnn_layout8> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg11: tensor<64x1x64x512xbf16, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg12: tensor<1x1x8192x3840xbf16, #ttnn_layout9> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg13: tensor<1x1x120x32xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg14: tensor<1x1x120x32xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg15: tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout10> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg16: tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout10> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg17: tensor<1x1x15360x3840x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout11> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg18: tensor<1x1x120x32xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x1x32x3840xbf16, #ttnn_layout> attributes {tt.function_type = "forward_device"} {
        %0 = "ttnn.reshape"(%arg1) <{shape = [3840 : i32]}> : (tensor<1x1x120x32xbf16, #ttnn_layout1>) -> tensor<3840xbf16, #ttnn_layout12>
        %1 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1x32x3840xbf16, #ttnn_layout>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %2 = "ttnn.to_memory_config"(%0) : (tensor<3840xbf16, #ttnn_layout12>) -> tensor<3840xbf16, #ttnn_layout14>
        %3 = "ttnn.rms_norm"(%1, %2) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x32x3840xbf16, #ttnn_layout13>, tensor<3840xbf16, #ttnn_layout14>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %4 = "ttnn.to_memory_config"(%3) : (tensor<1x1x32x3840xbf16, #ttnn_layout13>) -> tensor<1x1x32x3840xbf16, #ttnn_layout15>
        %5 = "ttnn.to_memory_config"(%4) : (tensor<1x1x32x3840xbf16, #ttnn_layout15>) -> tensor<1x1x32x3840xbf16, #ttnn_layout>
        %6 = "ttnn.to_memory_config"(%arg2) : (tensor<1x1x3840x9216xbf16, #ttnn_layout2>) -> tensor<1x1x3840x9216xbf16, #ttnn_layout16>
        %7 = "ttnn.to_memory_config"(%5) : (tensor<1x1x32x3840xbf16, #ttnn_layout>) -> tensor<1x1x32x3840xbf16, #ttnn_layout15>
        %8 = "ttnn.linear"(%7, %6) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<in0_block_w = 5, per_core_m = 1, per_core_n = 3>, transpose_a = false, transpose_b = false}> : (tensor<1x1x32x3840xbf16, #ttnn_layout15>, tensor<1x1x3840x9216xbf16, #ttnn_layout16>) -> tensor<1x1x32x9216xbf16, #ttnn_layout17>
        %9 = "ttnn.to_memory_config"(%8) : (tensor<1x1x32x9216xbf16, #ttnn_layout17>) -> tensor<1x1x32x9216xbf16, #ttnn_layout18>
        %query, %key, %value = "ttnn.nlp_create_qkv_heads_decode"(%9) <{num_heads = 16 : ui32, num_kv_heads = 1 : ui32}> : (tensor<1x1x32x9216xbf16, #ttnn_layout18>) -> (tensor<1x32x16x512xbf16, #ttnn_layout19>, tensor<1x32x1x512xbf16, #ttnn_layout20>, tensor<1x32x1x512xbf16, #ttnn_layout19>)
        %10 = "ttnn.reshape"(%query) <{shape = [1 : i32, 1 : i32, 512 : i32, 512 : i32]}> : (tensor<1x32x16x512xbf16, #ttnn_layout19>) -> tensor<1x1x512x512xbf16, #ttnn_layout21>
        %11 = "ttnn.reshape"(%arg3) <{shape = [512 : i32]}> : (tensor<1x1x16x32xbf16, #ttnn_layout3>) -> tensor<512xbf16, #ttnn_layout22>
        %12 = "ttnn.to_memory_config"(%10) : (tensor<1x1x512x512xbf16, #ttnn_layout21>) -> tensor<1x1x512x512xbf16, #ttnn_layout23>
        %13 = "ttnn.to_memory_config"(%11) : (tensor<512xbf16, #ttnn_layout22>) -> tensor<512xbf16, #ttnn_layout24>
        %14 = "ttnn.rms_norm"(%12, %13) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x512x512xbf16, #ttnn_layout23>, tensor<512xbf16, #ttnn_layout24>) -> tensor<1x1x512x512xbf16, #ttnn_layout23>
        %15 = "ttnn.reshape"(%14) <{shape = [1 : i32, 32 : i32, 16 : i32, 512 : i32]}> : (tensor<1x1x512x512xbf16, #ttnn_layout23>) -> tensor<1x32x16x512xbf16, #ttnn_layout25>
        %16 = "ttnn.reshape"(%key) <{shape = [1 : i32, 1 : i32, 32 : i32, 512 : i32]}> : (tensor<1x32x1x512xbf16, #ttnn_layout20>) -> tensor<1x1x32x512xbf16, #ttnn_layout26>
        %17 = "ttnn.reshape"(%arg4) <{shape = [512 : i32]}> : (tensor<1x1x16x32xbf16, #ttnn_layout3>) -> tensor<512xbf16, #ttnn_layout22>
        %18 = "ttnn.to_memory_config"(%16) : (tensor<1x1x32x512xbf16, #ttnn_layout26>) -> tensor<1x1x32x512xbf16, #ttnn_layout27>
        %19 = "ttnn.to_memory_config"(%17) : (tensor<512xbf16, #ttnn_layout22>) -> tensor<512xbf16, #ttnn_layout24>
        %20 = "ttnn.rms_norm"(%18, %19) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x32x512xbf16, #ttnn_layout27>, tensor<512xbf16, #ttnn_layout24>) -> tensor<1x1x32x512xbf16, #ttnn_layout27>
        %21 = "ttnn.to_memory_config"(%20) : (tensor<1x1x32x512xbf16, #ttnn_layout27>) -> tensor<1x1x32x512xbf16, #ttnn_layout26>
        %22 = "ttnn.reshape"(%21) <{shape = [1 : i32, 32 : i32, 1 : i32, 512 : i32]}> : (tensor<1x1x32x512xbf16, #ttnn_layout26>) -> tensor<1x32x1x512xbf16, #ttnn_layout25>
        %23 = "ttnn.reshape"(%value) <{shape = [1 : i32, 1 : i32, 32 : i32, 512 : i32]}> : (tensor<1x32x1x512xbf16, #ttnn_layout19>) -> tensor<1x1x32x512xbf16, #ttnn_layout26>
        %24 = "ttnn.to_memory_config"(%23) : (tensor<1x1x32x512xbf16, #ttnn_layout26>) -> tensor<1x1x32x512xbf16, #ttnn_layout27>
        %25 = "ttnn.rms_norm"(%24) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x1x32x512xbf16, #ttnn_layout27>) -> tensor<1x1x32x512xbf16, #ttnn_layout27>
        %26 = "ttnn.to_memory_config"(%25) : (tensor<1x1x32x512xbf16, #ttnn_layout27>) -> tensor<1x1x32x512xbf16, #ttnn_layout26>
        %27 = "ttnn.reshape"(%26) <{shape = [1 : i32, 32 : i32, 1 : i32, 512 : i32]}> : (tensor<1x1x32x512xbf16, #ttnn_layout26>) -> tensor<1x32x1x512xbf16, #ttnn_layout25>
        %28 = "ttnn.embedding"(%arg5, %arg6) : (tensor<1x32xui32, #ttnn_layout4>, tensor<128x512xbf16, #ttnn_layout5>) -> tensor<1x32x512xbf16, #ttnn_layout28>
        %29 = "ttnn.embedding"(%arg5, %arg7) : (tensor<1x32xui32, #ttnn_layout4>, tensor<128x512xbf16, #ttnn_layout5>) -> tensor<1x32x512xbf16, #ttnn_layout28>
        %30 = "ttnn.reshape"(%28) <{shape = [1 : i32, 1 : i32, 32 : i32, 512 : i32]}> : (tensor<1x32x512xbf16, #ttnn_layout28>) -> tensor<1x1x32x512xbf16, #ttnn_layout29>
        %31 = "ttnn.reshape"(%29) <{shape = [1 : i32, 1 : i32, 32 : i32, 512 : i32]}> : (tensor<1x32x512xbf16, #ttnn_layout28>) -> tensor<1x1x32x512xbf16, #ttnn_layout29>
        %32 = "ttnn.to_memory_config"(%15) : (tensor<1x32x16x512xbf16, #ttnn_layout25>) -> tensor<1x32x16x512xbf16, #ttnn_layout19>
        %33 = "ttnn.to_memory_config"(%31) : (tensor<1x1x32x512xbf16, #ttnn_layout29>) -> tensor<1x1x32x512xbf16, #ttnn_layout30>
        %34 = "ttnn.to_memory_config"(%30) : (tensor<1x1x32x512xbf16, #ttnn_layout29>) -> tensor<1x1x32x512xbf16, #ttnn_layout30>
        %35 = "ttnn.rotary_embedding"(%32, %34, %33) <{token_index = 0 : ui32}> : (tensor<1x32x16x512xbf16, #ttnn_layout19>, tensor<1x1x32x512xbf16, #ttnn_layout30>, tensor<1x1x32x512xbf16, #ttnn_layout30>) -> tensor<1x32x16x512xbf16, #ttnn_layout19>
        %36 = "ttnn.to_memory_config"(%22) : (tensor<1x32x1x512xbf16, #ttnn_layout25>) -> tensor<1x32x1x512xbf16, #ttnn_layout31>
        %37 = "ttnn.rotary_embedding"(%36, %34, %33) <{token_index = 0 : ui32}> : (tensor<1x32x1x512xbf16, #ttnn_layout31>, tensor<1x1x32x512xbf16, #ttnn_layout30>, tensor<1x1x32x512xbf16, #ttnn_layout30>) -> tensor<1x32x1x512xbf16, #ttnn_layout31>
        %38 = "ttnn.to_memory_config"(%37) : (tensor<1x32x1x512xbf16, #ttnn_layout31>) -> tensor<1x32x1x512xbf16, #ttnn_layout19>
        %39 = "ttnn.to_layout"(%arg9) : (tensor<32xsi32, #ttnn_layout7>) -> tensor<32xsi32, #ttnn_layout32>
        %40 = "ttnn.to_layout"(%arg10) : (tensor<32x2xsi32, #ttnn_layout8>) -> tensor<32x2xsi32, #ttnn_layout33>
        "ttnn.paged_update_cache"(%arg8, %38, %39, %40) <{share_cache = false}> : (tensor<64x1x64x512xbf16, #ttnn_layout6>, tensor<1x32x1x512xbf16, #ttnn_layout19>, tensor<32xsi32, #ttnn_layout32>, tensor<32x2xsi32, #ttnn_layout33>) -> ()
        %41 = "ttnn.to_memory_config"(%27) : (tensor<1x32x1x512xbf16, #ttnn_layout25>) -> tensor<1x32x1x512xbf16, #ttnn_layout19>
        %42 = "ttnn.to_layout"(%arg9) : (tensor<32xsi32, #ttnn_layout7>) -> tensor<32xsi32, #ttnn_layout32>
        %43 = "ttnn.to_layout"(%arg10) : (tensor<32x2xsi32, #ttnn_layout8>) -> tensor<32x2xsi32, #ttnn_layout33>
        "ttnn.paged_update_cache"(%arg11, %41, %42, %43) <{share_cache = false}> : (tensor<64x1x64x512xbf16, #ttnn_layout6>, tensor<1x32x1x512xbf16, #ttnn_layout19>, tensor<32xsi32, #ttnn_layout32>, tensor<32x2xsi32, #ttnn_layout33>) -> ()
        %44 = "ttnn.to_layout"(%arg10) : (tensor<32x2xsi32, #ttnn_layout8>) -> tensor<32x2xsi32, #ttnn_layout33>
        %45 = "ttnn.to_layout"(%arg9) : (tensor<32xsi32, #ttnn_layout7>) -> tensor<32xsi32, #ttnn_layout32>
        %46 = "ttnn.paged_scaled_dot_product_attention_decode"(%35, %arg8, %arg11, %44, %45) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>, scale = 1.000000e+00 : f32}> : (tensor<1x32x16x512xbf16, #ttnn_layout19>, tensor<64x1x64x512xbf16, #ttnn_layout6>, tensor<64x1x64x512xbf16, #ttnn_layout6>, tensor<32x2xsi32, #ttnn_layout33>, tensor<32xsi32, #ttnn_layout32>) -> tensor<1x32x16x512xbf16, #ttnn_layout34>
        %47 = "ttnn.to_memory_config"(%46) : (tensor<1x32x16x512xbf16, #ttnn_layout34>) -> tensor<1x32x16x512xbf16, #ttnn_layout35>
        %48 = "ttnn.transpose"(%47) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x16x512xbf16, #ttnn_layout35>) -> tensor<1x16x32x512xbf16, #ttnn_layout36>
        %49 = "ttnn.to_memory_config"(%48) : (tensor<1x16x32x512xbf16, #ttnn_layout36>) -> tensor<1x16x32x512xbf16, #ttnn_layout37>
        %50 = "ttnn.concatenate_heads"(%49) : (tensor<1x16x32x512xbf16, #ttnn_layout37>) -> tensor<1x32x8192xbf16, #ttnn_layout38>
        %51 = "ttnn.linear"(%50, %arg12) <{matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 6>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 2, out_block_h = 1, out_block_w = 2, per_core_m = 1, per_core_n = 2, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = false}> : (tensor<1x32x8192xbf16, #ttnn_layout38>, tensor<1x1x8192x3840xbf16, #ttnn_layout9>) -> tensor<1x1x32x3840xbf16, #ttnn_layout39>
        %52 = "ttnn.reshape"(%arg13) <{shape = [3840 : i32]}> : (tensor<1x1x120x32xbf16, #ttnn_layout1>) -> tensor<3840xbf16, #ttnn_layout12>
        %53 = "ttnn.to_memory_config"(%51) : (tensor<1x1x32x3840xbf16, #ttnn_layout39>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %54 = "ttnn.to_memory_config"(%52) : (tensor<3840xbf16, #ttnn_layout12>) -> tensor<3840xbf16, #ttnn_layout14>
        %55 = "ttnn.rms_norm"(%53, %54) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x32x3840xbf16, #ttnn_layout13>, tensor<3840xbf16, #ttnn_layout14>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %56 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1x32x3840xbf16, #ttnn_layout>) -> tensor<1x1x32x3840xbf16, #ttnn_layout39>
        %57 = "ttnn.add"(%56, %55) : (tensor<1x1x32x3840xbf16, #ttnn_layout39>, tensor<1x1x32x3840xbf16, #ttnn_layout13>) -> tensor<1x1x32x3840xbf16, #ttnn_layout39>
        %58 = "ttnn.to_memory_config"(%57) : (tensor<1x1x32x3840xbf16, #ttnn_layout39>) -> tensor<1x1x32x3840xbf16, #ttnn_layout>
        %59 = "ttnn.reshape"(%arg14) <{shape = [3840 : i32]}> : (tensor<1x1x120x32xbf16, #ttnn_layout1>) -> tensor<3840xbf16, #ttnn_layout12>
        %60 = "ttnn.to_memory_config"(%58) : (tensor<1x1x32x3840xbf16, #ttnn_layout>) -> tensor<1x1x32x3840xbf16, #ttnn_layout39>
        %61 = "ttnn.to_memory_config"(%60) : (tensor<1x1x32x3840xbf16, #ttnn_layout39>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %62 = "ttnn.to_memory_config"(%59) : (tensor<3840xbf16, #ttnn_layout12>) -> tensor<3840xbf16, #ttnn_layout14>
        %63 = "ttnn.rms_norm"(%61, %62) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x32x3840xbf16, #ttnn_layout13>, tensor<3840xbf16, #ttnn_layout14>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %64 = "ttnn.to_memory_config"(%63) : (tensor<1x1x32x3840xbf16, #ttnn_layout13>) -> tensor<1x1x32x3840xbf16, #ttnn_layout15>
        %65 = "ttnn.to_memory_config"(%arg15) : (tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout10>) -> tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout40>
        %66 = "ttnn.linear"(%64, %65) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<in0_block_w = 3, per_core_m = 1, per_core_n = 5>, transpose_a = false, transpose_b = false}> : (tensor<1x1x32x3840xbf16, #ttnn_layout15>, tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout40>) -> tensor<1x1x32x15360xbf16, #ttnn_layout41>
        %67 = "ttnn.to_memory_config"(%arg16) : (tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout10>) -> tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout40>
        %68 = "ttnn.linear"(%64, %67) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<in0_block_w = 3, per_core_m = 1, per_core_n = 5>, transpose_a = false, transpose_b = false}> : (tensor<1x1x32x3840xbf16, #ttnn_layout15>, tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout40>) -> tensor<1x1x32x15360xbf16, #ttnn_layout41>
        %69 = "ttnn.multiply"(%66, %68) : (tensor<1x1x32x15360xbf16, #ttnn_layout41>, tensor<1x1x32x15360xbf16, #ttnn_layout41>) -> tensor<1x1x32x15360xbf16, #ttnn_layout41>
        %70 = "ttnn.to_memory_config"(%69) : (tensor<1x1x32x15360xbf16, #ttnn_layout41>) -> tensor<1x1x32x15360xbf16, #ttnn_layout42>
        %71 = "ttnn.to_memory_config"(%70) : (tensor<1x1x32x15360xbf16, #ttnn_layout42>) -> tensor<1x1x32x15360xbf16, #ttnn_layout43>
        %72 = "ttnn.to_memory_config"(%arg17) : (tensor<1x1x15360x3840x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout11>) -> tensor<1x1x15360x3840x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout44>
        %73 = "ttnn.to_memory_config"(%71) : (tensor<1x1x32x15360xbf16, #ttnn_layout43>) -> tensor<1x1x32x15360xbf16, #ttnn_layout42>
        %74 = "ttnn.linear"(%73, %72) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<in0_block_w = 20, per_core_m = 1, per_core_n = 2>, transpose_a = false, transpose_b = false}> : (tensor<1x1x32x15360xbf16, #ttnn_layout42>, tensor<1x1x15360x3840x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout44>) -> tensor<1x1x32x3840xbf16, #ttnn_layout39>
        %75 = "ttnn.reshape"(%arg18) <{shape = [3840 : i32]}> : (tensor<1x1x120x32xbf16, #ttnn_layout1>) -> tensor<3840xbf16, #ttnn_layout12>
        %76 = "ttnn.to_memory_config"(%74) : (tensor<1x1x32x3840xbf16, #ttnn_layout39>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %77 = "ttnn.to_memory_config"(%75) : (tensor<3840xbf16, #ttnn_layout12>) -> tensor<3840xbf16, #ttnn_layout14>
        %78 = "ttnn.rms_norm"(%76, %77) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x32x3840xbf16, #ttnn_layout13>, tensor<3840xbf16, #ttnn_layout14>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %79 = "ttnn.add"(%58, %78) : (tensor<1x1x32x3840xbf16, #ttnn_layout>, tensor<1x1x32x3840xbf16, #ttnn_layout13>) -> tensor<1x1x32x3840xbf16, #ttnn_layout39>
        %80 = "ttnn.to_memory_config"(%79) : (tensor<1x1x32x3840xbf16, #ttnn_layout39>) -> tensor<1x1x32x3840xbf16, #ttnn_layout>
        return %80 : tensor<1x1x32x3840xbf16, #ttnn_layout>
      }
    }
  }
}
