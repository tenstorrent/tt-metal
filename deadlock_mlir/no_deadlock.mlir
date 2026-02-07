//RUN: ttmlir-translate --ttmetal-to-flatbuffer tilize_block_not_working_with_f32_cb.mlir -o out.ttm && ttrt run out.ttm
#l1 = #ttcore.memory_space<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102208, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2560032, dram_unreserved_end = 1073171136, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102208, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2560032, dram_unreserved_end = 1073179680, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0, 1], [1 : i32, 0 : i32], [ 0x0x0x0]>
module attributes {polyblocks.cpu_target_info = {l1_cache_associativity = 8 : i8, l1_cache_size = 32768 : i32, l2_cache_associativity = 8 : i8, l2_cache_line_size = 64 : i32, l2_cache_size = 524288 : i32, num_cores = 32 : i32, omp_num_threads = 32 : i32, simd_width = 256 : i32}, polyblocks.target = "tenstorrent", polyblocks.tenstorrent_target_info = {dram_alignment_bytes = 32 : i32, l1_scratchpad_size_bytes = 1499136 : i32, l1_unreserved_base = 101152 : i32, num_cores_x = 8 : i32, num_cores_y = 8 : i32, use_full_dest_registers = true}, torch.input_to_acc_type = [[bf16, bf16]], torch.target = "tenstorrent", ttcore.system_desc = #system_desc} {
  func.func private @Compute_kernel() attributes {arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, polyblocks_tt.use_full_dest_reg = true, ttkernel.thread = #ttkernel.thread<compute>} {
    emitc.verbatim "DPRINT_UNPACK(DPRINT << \22Compute_kernel\22 << ENDL();)"
    %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    %2 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
    %3 = emitc.literal "get_compile_time_arg_val(2)" : !emitc.opaque<"::tt::CB">
    emitc.call_opaque "compute_kernel_hw_startup"(%2, %2, %3) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
    emitc.call_opaque "tilize_init"(%2, %0, %3) : (!emitc.opaque<"::tt::CB">, i32, !emitc.opaque<"::tt::CB">) -> ()
    emitc.call_opaque "tilize_block"(%2, %3, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
    emitc.call_opaque "tilize_uninit"(%2, %3) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
    return
  }
  func.func private @DataFlow1_kernel() attributes {arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 4>, <arg_type = buffer_address, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @DataFlow0_kernel() attributes {arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = buffer_address, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @forward(%arg0: memref<32xi32>) -> memref<32xf32> attributes {polyblocks.entry_function, torch.inputs = "convert_element_type"} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
    %0 = "ttmetal.create_buffer"() <{address = 102208 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1>
    %1 = "ttmetal.create_buffer"() <{address = 106304 : i64}> : () -> memref<1x1x5408x32xsi32, #ttcore.shard<128x4, 1>, #l1>
    %2 = "ttmetal.create_buffer"() <{address = 798528 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1>
    "ttmetal.enqueue_program"(%0, %1, %2) <{cb_ports = array<i64: 0, 1, 2>, kernelConfigs = [#ttmetal.noc_config<@DataFlow0_kernel, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>]>, noc0>, #ttmetal.noc_config<@DataFlow1_kernel, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>]>, noc1>, #ttmetal.compute_config<@Compute_kernel, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, hifi4, true, true, false, [default, default, default, default, default, default]>], operandSegmentSizes = array<i32: 0, 3>}> : (memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x5408x32xsi32, #ttcore.shard<128x4, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1>) -> ()
    "ttmetal.deallocate_buffer"(%0) : (memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1>) -> ()
    "ttmetal.deallocate_buffer"(%1) : (memref<1x1x5408x32xsi32, #ttcore.shard<128x4, 1>, #l1>) -> ()
    "ttmetal.deallocate_buffer"(%2) : (memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1>) -> ()
    "ttmetal.finish"() : () -> ()
    return %alloc : memref<32xf32>
  }
}
