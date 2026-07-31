#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <blackhole>, grid = 10x11, coord_translation_offsets = 2x1, l1_size = 1572864, num_dram_channels = 8, dram_channel_size = 4278190080, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 64, noc_dram_address_align_bytes = 64, l1_unreserved_base = 111360, erisc_l1_unreserved_base = 87872, dram_unreserved_base = 5848704, dram_unreserved_end = 4276464000, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x8, dram_bank_to_logical_worker_noc0 = [(9, 0), (0, 0), (7, 0), (3, 0), (9, 7), (1, 7), (6, 7), (4, 7)], dram_bank_to_logical_worker_noc1 = [(9, 0), (0, 0), (7, 0), (3, 0), (9, 7), (1, 7), (6, 7), (4, 7)]}], [0], [1 : i32], [ 0x0x0x0], [<[0, 2, 0], [3, 5, 0]>, <[0, 4, 0], [3, 4, 0]>, <[2, 8, 0], [3, 3, 0]>, <[2, 9, 0], [3, 2, 0]>]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x120x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3840 + d1 * 3840 + d2, d3), <1x1>, memref<120x256x!ttcore.tile<32x32, bfp_bf8>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2, d3), <1x1>, memref<1024x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2, d3), <1x1>, memref<128x120x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3840 + d1 * 3840 + d2, d3), <1x1>, memref<120x480x!ttcore.tile<32x32, bfp_bf8>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 15360 + d1 * 15360 + d2, d3), <1x1>, memref<480x120x!ttcore.tile<32x32, bfp_bf8>, #dram>, <interleaved>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x120x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x11>, memref<1x11x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,0)>]>>
#ttnn_layout14 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x60>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,4)>, #ttnn.core_range<(0,5), (4,5)>]>>
#ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x8>, memref<1x15x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3840 + d1 * 3840 + d2, d3), <1x8>, memref<120x32x!ttcore.tile<32x32, bfp_bf8>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout17 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x86>, memref<1x3x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,6)>, #ttnn.core_range<(0,7), (8,7)>]>>
#ttnn_layout18 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x64>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,4)>, #ttnn.core_range<(0,5), (8,5)>]>>
#ttnn_layout19 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <32x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,1)>, #ttnn.core_range<(0,2), (9,2)>]>>
#ttnn_layout20 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <32x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(10,2), (10,2)>, #ttnn.core_range<(0,3), (10,4)>, #ttnn.core_range<(0,5), (8,5)>]>>
#ttnn_layout21 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <10x11>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout22 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout23 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <8x8>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout24 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout25 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <10x11>, memref<1x3x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout26 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 256 + d2, d3), <10x11>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout27 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 256 + d2, d3), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout28 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout29 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout30 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout31 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32xsi32, #dram>, <interleaved>>
#ttnn_layout32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x2xsi32, #dram>, <interleaved>>
#ttnn_layout33 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout34 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <7x8>, memref<5x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,6)>]>>
#ttnn_layout35 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <10x11>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,9)>]>>
#ttnn_layout36 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <10x11>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout37 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <10x11>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout38 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x60>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,4)>, #ttnn.core_range<(0,5), (4,5)>]>>
#ttnn_layout39 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3840 + d1 * 3840 + d2, d3), <1x8>, memref<120x60x!ttcore.tile<32x32, bfp_bf8>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout40 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x96>, memref<1x5x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,7)>, #ttnn.core_range<(0,8), (7,8)>]>>
#ttnn_layout41 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x8>, memref<1x60x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout42 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x480x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout43 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 15360 + d1 * 15360 + d2, d3), <1x8>, memref<480x15x!ttcore.tile<32x32, bfp_bf8>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<10x11, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x8>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 8, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 8) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @decode(%arg0: tensor<1x1x32x3840xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<1x1x120x32xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<1x1x3840x8192x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<1x1x8x32xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg4: tensor<1x1x8x32xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg5: tensor<1x32xui32, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg6: tensor<128x256xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg7: tensor<128x256xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg8: tensor<64x8x64x256xbf16, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg9: tensor<32xsi32, #ttnn_layout7> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg10: tensor<32x2xsi32, #ttnn_layout8> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg11: tensor<64x8x64x256xbf16, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg12: tensor<1x1x4096x3840xbf16, #ttnn_layout9> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg13: tensor<1x1x120x32xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg14: tensor<1x1x120x32xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg15: tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout10> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg16: tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout10> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg17: tensor<1x1x15360x3840x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout11> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg18: tensor<1x1x120x32xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x1x32x3840xbf16, #ttnn_layout> attributes {tt.function_type = "forward_device"} {
        %0 = "ttnn.reshape"(%arg1) <{shape = [3840 : i32]}> : (tensor<1x1x120x32xbf16, #ttnn_layout1>) -> tensor<3840xbf16, #ttnn_layout12>
        %1 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1x32x3840xbf16, #ttnn_layout>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %2 = "ttnn.to_memory_config"(%0) : (tensor<3840xbf16, #ttnn_layout12>) -> tensor<3840xbf16, #ttnn_layout14>
        %3 = "ttnn.rms_norm"(%1, %2) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x32x3840xbf16, #ttnn_layout13>, tensor<3840xbf16, #ttnn_layout14>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %4 = "ttnn.to_memory_config"(%3) : (tensor<1x1x32x3840xbf16, #ttnn_layout13>) -> tensor<1x1x32x3840xbf16, #ttnn_layout15>
        %5 = "ttnn.to_memory_config"(%arg2) : (tensor<1x1x3840x8192x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout2>) -> tensor<1x1x3840x8192x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout16>
        %6 = "ttnn.linear"(%4, %5) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<in0_block_w = 5, per_core_m = 1, per_core_n = 3>, transpose_a = false, transpose_b = false}> : (tensor<1x1x32x3840xbf16, #ttnn_layout15>, tensor<1x1x3840x8192x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout16>) -> tensor<1x1x32x8192xbf16, #ttnn_layout17>
        %7 = "ttnn.to_memory_config"(%6) : (tensor<1x1x32x8192xbf16, #ttnn_layout17>) -> tensor<1x1x32x8192xbf16, #ttnn_layout18>
        %query, %key, %value = "ttnn.nlp_create_qkv_heads_decode"(%7) <{num_heads = 16 : ui32, num_kv_heads = 8 : ui32}> : (tensor<1x1x32x8192xbf16, #ttnn_layout18>) -> (tensor<1x32x16x256xbf16, #ttnn_layout19>, tensor<1x32x8x256xbf16, #ttnn_layout20>, tensor<1x32x8x256xbf16, #ttnn_layout19>)
        %8 = "ttnn.reshape"(%query) <{shape = [1 : i32, 1 : i32, 512 : i32, 256 : i32]}> : (tensor<1x32x16x256xbf16, #ttnn_layout19>) -> tensor<1x1x512x256xbf16, #ttnn_layout21>
        %9 = "ttnn.reshape"(%arg3) <{shape = [256 : i32]}> : (tensor<1x1x8x32xbf16, #ttnn_layout3>) -> tensor<256xbf16, #ttnn_layout22>
        %10 = "ttnn.to_memory_config"(%8) : (tensor<1x1x512x256xbf16, #ttnn_layout21>) -> tensor<1x1x512x256xbf16, #ttnn_layout23>
        %11 = "ttnn.to_memory_config"(%9) : (tensor<256xbf16, #ttnn_layout22>) -> tensor<256xbf16, #ttnn_layout24>
        %12 = "ttnn.rms_norm"(%10, %11) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x512x256xbf16, #ttnn_layout23>, tensor<256xbf16, #ttnn_layout24>) -> tensor<1x1x512x256xbf16, #ttnn_layout23>
        %13 = "ttnn.reshape"(%12) <{shape = [1 : i32, 32 : i32, 16 : i32, 256 : i32]}> : (tensor<1x1x512x256xbf16, #ttnn_layout23>) -> tensor<1x32x16x256xbf16, #ttnn_layout25>
        %14 = "ttnn.reshape"(%key) <{shape = [1 : i32, 1 : i32, 256 : i32, 256 : i32]}> : (tensor<1x32x8x256xbf16, #ttnn_layout20>) -> tensor<1x1x256x256xbf16, #ttnn_layout26>
        %15 = "ttnn.reshape"(%arg4) <{shape = [256 : i32]}> : (tensor<1x1x8x32xbf16, #ttnn_layout3>) -> tensor<256xbf16, #ttnn_layout22>
        %16 = "ttnn.to_memory_config"(%14) : (tensor<1x1x256x256xbf16, #ttnn_layout26>) -> tensor<1x1x256x256xbf16, #ttnn_layout27>
        %17 = "ttnn.to_memory_config"(%15) : (tensor<256xbf16, #ttnn_layout22>) -> tensor<256xbf16, #ttnn_layout24>
        %18 = "ttnn.rms_norm"(%16, %17) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x256x256xbf16, #ttnn_layout27>, tensor<256xbf16, #ttnn_layout24>) -> tensor<1x1x256x256xbf16, #ttnn_layout27>
        %19 = "ttnn.reshape"(%18) <{shape = [1 : i32, 32 : i32, 8 : i32, 256 : i32]}> : (tensor<1x1x256x256xbf16, #ttnn_layout27>) -> tensor<1x32x8x256xbf16, #ttnn_layout25>
        %20 = "ttnn.reshape"(%value) <{shape = [1 : i32, 1 : i32, 256 : i32, 256 : i32]}> : (tensor<1x32x8x256xbf16, #ttnn_layout19>) -> tensor<1x1x256x256xbf16, #ttnn_layout26>
        %21 = "ttnn.to_memory_config"(%20) : (tensor<1x1x256x256xbf16, #ttnn_layout26>) -> tensor<1x1x256x256xbf16, #ttnn_layout27>
        %22 = "ttnn.rms_norm"(%21) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x1x256x256xbf16, #ttnn_layout27>) -> tensor<1x1x256x256xbf16, #ttnn_layout27>
        %23 = "ttnn.reshape"(%22) <{shape = [1 : i32, 32 : i32, 8 : i32, 256 : i32]}> : (tensor<1x1x256x256xbf16, #ttnn_layout27>) -> tensor<1x32x8x256xbf16, #ttnn_layout25>
        %24 = "ttnn.embedding"(%arg5, %arg6) : (tensor<1x32xui32, #ttnn_layout4>, tensor<128x256xbf16, #ttnn_layout5>) -> tensor<1x32x256xbf16, #ttnn_layout28>
        %25 = "ttnn.embedding"(%arg5, %arg7) : (tensor<1x32xui32, #ttnn_layout4>, tensor<128x256xbf16, #ttnn_layout5>) -> tensor<1x32x256xbf16, #ttnn_layout28>
        %26 = "ttnn.reshape"(%24) <{shape = [1 : i32, 1 : i32, 32 : i32, 256 : i32]}> : (tensor<1x32x256xbf16, #ttnn_layout28>) -> tensor<1x1x32x256xbf16, #ttnn_layout29>
        %27 = "ttnn.reshape"(%25) <{shape = [1 : i32, 1 : i32, 32 : i32, 256 : i32]}> : (tensor<1x32x256xbf16, #ttnn_layout28>) -> tensor<1x1x32x256xbf16, #ttnn_layout29>
        %28 = "ttnn.to_memory_config"(%13) : (tensor<1x32x16x256xbf16, #ttnn_layout25>) -> tensor<1x32x16x256xbf16, #ttnn_layout19>
        %29 = "ttnn.to_memory_config"(%27) : (tensor<1x1x32x256xbf16, #ttnn_layout29>) -> tensor<1x1x32x256xbf16, #ttnn_layout30>
        %30 = "ttnn.to_memory_config"(%26) : (tensor<1x1x32x256xbf16, #ttnn_layout29>) -> tensor<1x1x32x256xbf16, #ttnn_layout30>
        %31 = "ttnn.rotary_embedding"(%28, %30, %29) <{token_index = 0 : ui32}> : (tensor<1x32x16x256xbf16, #ttnn_layout19>, tensor<1x1x32x256xbf16, #ttnn_layout30>, tensor<1x1x32x256xbf16, #ttnn_layout30>) -> tensor<1x32x16x256xbf16, #ttnn_layout19>
        %32 = "ttnn.to_memory_config"(%19) : (tensor<1x32x8x256xbf16, #ttnn_layout25>) -> tensor<1x32x8x256xbf16, #ttnn_layout19>
        %33 = "ttnn.rotary_embedding"(%32, %30, %29) <{token_index = 0 : ui32}> : (tensor<1x32x8x256xbf16, #ttnn_layout19>, tensor<1x1x32x256xbf16, #ttnn_layout30>, tensor<1x1x32x256xbf16, #ttnn_layout30>) -> tensor<1x32x8x256xbf16, #ttnn_layout19>
        %34 = "ttnn.to_layout"(%arg9) : (tensor<32xsi32, #ttnn_layout7>) -> tensor<32xsi32, #ttnn_layout31>
        %35 = "ttnn.to_layout"(%arg10) : (tensor<32x2xsi32, #ttnn_layout8>) -> tensor<32x2xsi32, #ttnn_layout32>
        "ttnn.paged_update_cache"(%arg8, %33, %34, %35) <{share_cache = false}> : (tensor<64x8x64x256xbf16, #ttnn_layout6>, tensor<1x32x8x256xbf16, #ttnn_layout19>, tensor<32xsi32, #ttnn_layout31>, tensor<32x2xsi32, #ttnn_layout32>) -> ()
        %36 = "ttnn.to_memory_config"(%23) : (tensor<1x32x8x256xbf16, #ttnn_layout25>) -> tensor<1x32x8x256xbf16, #ttnn_layout19>
        %37 = "ttnn.to_layout"(%arg9) : (tensor<32xsi32, #ttnn_layout7>) -> tensor<32xsi32, #ttnn_layout31>
        %38 = "ttnn.to_layout"(%arg10) : (tensor<32x2xsi32, #ttnn_layout8>) -> tensor<32x2xsi32, #ttnn_layout32>
        "ttnn.paged_update_cache"(%arg11, %36, %37, %38) <{share_cache = false}> : (tensor<64x8x64x256xbf16, #ttnn_layout6>, tensor<1x32x8x256xbf16, #ttnn_layout19>, tensor<32xsi32, #ttnn_layout31>, tensor<32x2xsi32, #ttnn_layout32>) -> ()
        %39 = "ttnn.to_layout"(%arg10) : (tensor<32x2xsi32, #ttnn_layout8>) -> tensor<32x2xsi32, #ttnn_layout32>
        %40 = "ttnn.to_layout"(%arg9) : (tensor<32xsi32, #ttnn_layout7>) -> tensor<32xsi32, #ttnn_layout31>
        %41 = "ttnn.paged_scaled_dot_product_attention_decode"(%31, %arg8, %arg11, %39, %40) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>, scale = 1.000000e+00 : f32}> : (tensor<1x32x16x256xbf16, #ttnn_layout19>, tensor<64x8x64x256xbf16, #ttnn_layout6>, tensor<64x8x64x256xbf16, #ttnn_layout6>, tensor<32x2xsi32, #ttnn_layout32>, tensor<32xsi32, #ttnn_layout31>) -> tensor<1x32x16x256xbf16, #ttnn_layout33>
        %42 = "ttnn.to_memory_config"(%41) : (tensor<1x32x16x256xbf16, #ttnn_layout33>) -> tensor<1x32x16x256xbf16, #ttnn_layout34>
        %43 = "ttnn.transpose"(%42) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x16x256xbf16, #ttnn_layout34>) -> tensor<1x16x32x256xbf16, #ttnn_layout35>
        %44 = "ttnn.to_memory_config"(%43) : (tensor<1x16x32x256xbf16, #ttnn_layout35>) -> tensor<1x16x32x256xbf16, #ttnn_layout36>
        %45 = "ttnn.concatenate_heads"(%44) : (tensor<1x16x32x256xbf16, #ttnn_layout36>) -> tensor<1x32x4096xbf16, #ttnn_layout37>
        %46 = "ttnn.linear"(%45, %arg12) <{matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 6>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 2, out_block_h = 1, out_block_w = 2, per_core_m = 1, per_core_n = 2, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = false}> : (tensor<1x32x4096xbf16, #ttnn_layout37>, tensor<1x1x4096x3840xbf16, #ttnn_layout9>) -> tensor<1x1x32x3840xbf16, #ttnn_layout38>
        %47 = "ttnn.reshape"(%arg13) <{shape = [3840 : i32]}> : (tensor<1x1x120x32xbf16, #ttnn_layout1>) -> tensor<3840xbf16, #ttnn_layout12>
        %48 = "ttnn.to_memory_config"(%46) : (tensor<1x1x32x3840xbf16, #ttnn_layout38>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %49 = "ttnn.to_memory_config"(%47) : (tensor<3840xbf16, #ttnn_layout12>) -> tensor<3840xbf16, #ttnn_layout14>
        %50 = "ttnn.rms_norm"(%48, %49) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x32x3840xbf16, #ttnn_layout13>, tensor<3840xbf16, #ttnn_layout14>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %51 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1x32x3840xbf16, #ttnn_layout>) -> tensor<1x1x32x3840xbf16, #ttnn_layout38>
        %52 = "ttnn.add"(%51, %50) : (tensor<1x1x32x3840xbf16, #ttnn_layout38>, tensor<1x1x32x3840xbf16, #ttnn_layout13>) -> tensor<1x1x32x3840xbf16, #ttnn_layout38>
        %53 = "ttnn.to_memory_config"(%52) : (tensor<1x1x32x3840xbf16, #ttnn_layout38>) -> tensor<1x1x32x3840xbf16, #ttnn_layout>
        %54 = "ttnn.reshape"(%arg14) <{shape = [3840 : i32]}> : (tensor<1x1x120x32xbf16, #ttnn_layout1>) -> tensor<3840xbf16, #ttnn_layout12>
        %55 = "ttnn.to_memory_config"(%53) : (tensor<1x1x32x3840xbf16, #ttnn_layout>) -> tensor<1x1x32x3840xbf16, #ttnn_layout38>
        %56 = "ttnn.to_memory_config"(%55) : (tensor<1x1x32x3840xbf16, #ttnn_layout38>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %57 = "ttnn.to_memory_config"(%54) : (tensor<3840xbf16, #ttnn_layout12>) -> tensor<3840xbf16, #ttnn_layout14>
        %58 = "ttnn.rms_norm"(%56, %57) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x32x3840xbf16, #ttnn_layout13>, tensor<3840xbf16, #ttnn_layout14>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %59 = "ttnn.to_memory_config"(%58) : (tensor<1x1x32x3840xbf16, #ttnn_layout13>) -> tensor<1x1x32x3840xbf16, #ttnn_layout15>
        %60 = "ttnn.to_memory_config"(%arg15) : (tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout10>) -> tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout39>
        %61 = "ttnn.linear"(%59, %60) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<in0_block_w = 3, per_core_m = 1, per_core_n = 5>, transpose_a = false, transpose_b = false}> : (tensor<1x1x32x3840xbf16, #ttnn_layout15>, tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout39>) -> tensor<1x1x32x15360xbf16, #ttnn_layout40>
        %62 = "ttnn.to_memory_config"(%arg16) : (tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout10>) -> tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout39>
        %63 = "ttnn.linear"(%59, %62) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<in0_block_w = 3, per_core_m = 1, per_core_n = 5>, transpose_a = false, transpose_b = false}> : (tensor<1x1x32x3840xbf16, #ttnn_layout15>, tensor<1x1x3840x15360x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout39>) -> tensor<1x1x32x15360xbf16, #ttnn_layout40>
        %64 = "ttnn.multiply"(%61, %63) : (tensor<1x1x32x15360xbf16, #ttnn_layout40>, tensor<1x1x32x15360xbf16, #ttnn_layout40>) -> tensor<1x1x32x15360xbf16, #ttnn_layout40>
        %65 = "ttnn.to_memory_config"(%64) : (tensor<1x1x32x15360xbf16, #ttnn_layout40>) -> tensor<1x1x32x15360xbf16, #ttnn_layout41>
        %66 = "ttnn.to_memory_config"(%65) : (tensor<1x1x32x15360xbf16, #ttnn_layout41>) -> tensor<1x1x32x15360xbf16, #ttnn_layout42>
        %67 = "ttnn.to_memory_config"(%arg17) : (tensor<1x1x15360x3840x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout11>) -> tensor<1x1x15360x3840x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout43>
        %68 = "ttnn.to_memory_config"(%66) : (tensor<1x1x32x15360xbf16, #ttnn_layout42>) -> tensor<1x1x32x15360xbf16, #ttnn_layout41>
        %69 = "ttnn.linear"(%68, %67) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<in0_block_w = 20, per_core_m = 1, per_core_n = 2>, transpose_a = false, transpose_b = false}> : (tensor<1x1x32x15360xbf16, #ttnn_layout41>, tensor<1x1x15360x3840x!ttcore.tile<32x32, bfp_bf8>, #ttnn_layout43>) -> tensor<1x1x32x3840xbf16, #ttnn_layout38>
        %70 = "ttnn.reshape"(%arg18) <{shape = [3840 : i32]}> : (tensor<1x1x120x32xbf16, #ttnn_layout1>) -> tensor<3840xbf16, #ttnn_layout12>
        %71 = "ttnn.to_memory_config"(%69) : (tensor<1x1x32x3840xbf16, #ttnn_layout38>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %72 = "ttnn.to_memory_config"(%70) : (tensor<3840xbf16, #ttnn_layout12>) -> tensor<3840xbf16, #ttnn_layout14>
        %73 = "ttnn.rms_norm"(%71, %72) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x32x3840xbf16, #ttnn_layout13>, tensor<3840xbf16, #ttnn_layout14>) -> tensor<1x1x32x3840xbf16, #ttnn_layout13>
        %74 = "ttnn.add"(%53, %73) : (tensor<1x1x32x3840xbf16, #ttnn_layout>, tensor<1x1x32x3840xbf16, #ttnn_layout13>) -> tensor<1x1x32x3840xbf16, #ttnn_layout38>
        %75 = "ttnn.to_memory_config"(%74) : (tensor<1x1x32x3840xbf16, #ttnn_layout38>) -> tensor<1x1x32x3840xbf16, #ttnn_layout>
        return %75 : tensor<1x1x32x3840xbf16, #ttnn_layout>
      }
    }
  }
}
