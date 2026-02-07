#l1 = #ttcore.memory_space<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073119552, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
module attributes {polyblocks.cpu_target_info = {l1_cache_associativity = 8 : i8, l1_cache_size = 32768 : i32, l2_cache_associativity = 8 : i8, l2_cache_line_size = 64 : i32, l2_cache_size = 1048576 : i32, num_cores = 64 : i32, omp_num_threads = 64 : i32, simd_width = 512 : i32}, polyblocks.target = "tenstorrent", polyblocks.tenstorrent_target_info = {dram_alignment_bytes = 32 : i32, l1_scratchpad_size_bytes = 1499136 : i32, l1_unreserved_base = 101152 : i32, num_cores_x = 8 : i32, num_cores_y = 8 : i32, use_full_dest_registers = true}, torch.input_to_acc_type = [[bf16, bf16]], torch.target = "tenstorrent", ttcore.system_desc = #system_desc} {
  func.func private @Compute_kernel_9() attributes {arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>]>, polyblocks_tt.use_full_dest_reg = true, tt.function_type = "kernel", ttkernel.thread = #ttkernel.thread<compute>} {
    %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %2 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    %3 = emitc.literal "get_compile_time_arg_val(3)" : !emitc.opaque<"::tt::CB">
    emitc.call_opaque "cb_reserve_back"(%3, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    %4 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
    emitc.call_opaque "cb_wait_front"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_pop_front"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_reserve_back"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_push_back"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    %5 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
    emitc.call_opaque "cb_reserve_back"(%5, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "fill_tile_init"() : () -> ()
    emitc.call_opaque "unary_op_init_common"(%5, %5) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
    %6 = "emitc.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    emitc.call_opaque "tile_regs_acquire"() : () -> ()
    %7 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    emitc.call_opaque "fill_tile"(%7, %6) : (!emitc.size_t, f32) -> ()
    emitc.call_opaque "tile_regs_commit"() : () -> ()
    emitc.call_opaque "tile_regs_wait"() : () -> ()
    emitc.call_opaque "pack_tile"(%7, %5, %7) {template_args = [true]} : (!emitc.size_t, !emitc.opaque<"::tt::CB">, !emitc.size_t) -> ()
    emitc.call_opaque "tile_regs_release"() : () -> ()
    emitc.call_opaque "cb_push_back"(%5, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_wait_front"(%5, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.verbatim "DPRINT_UNPACK(DPRINT << \22cb_id \22 << get_compile_time_arg_val(1) << \22: { size: \22 << get_local_cb_interface(0).fifo_size << \22, limit: \22 << get_local_cb_interface(0).fifo_limit << \22, page_size: \22 << get_local_cb_interface(0).fifo_page_size << \22, num_pages: \22 << get_local_cb_interface(0).fifo_num_pages << \22, rd_ptr: \22 << get_local_cb_interface(0).fifo_rd_ptr << \22, wr_ptr: \22 << get_local_cb_interface(0).fifo_wr_ptr << \22, wr_tile_ptr: \22 << get_local_cb_interface(0).fifo_wr_tile_ptr << \22 }\22;)"
    emitc.verbatim "DPRINT_UNPACK(DPRINT << \22cb_idx: \22 << (uint8_t)get_compile_time_arg_val(1) << \22 tile_idx: \22 << 0 << ENDL();)"
    emitc.verbatim "DPRINT_UNPACK(DPRINT << \22======INPUT======\22 << ENDL();)"
    emitc.verbatim "DPRINT_UNPACK(for (uint16_t r = 0; r < 32; ++r) {)"
    emitc.verbatim "DPRINT_UNPACK(  DPRINT << (uint)r << \22 : \22;)"
    emitc.verbatim "DPRINT_UNPACK(  for (uint16_t c = 0; c < 32; c+=16) {)"
    emitc.verbatim "DPRINT_UNPACK(    DPRINT << \22 \22 << TileSlice((uint8_t)get_compile_time_arg_val(1), 0, SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = (uint8_t)1, .w0 = (uint8_t)(c), .w1 = (uint8_t)(c + 16), .ws = (uint8_t)1}, false, true);)"
    emitc.verbatim "DPRINT_UNPACK(  })"
    emitc.verbatim "DPRINT_UNPACK(  DPRINT << ENDL();)"
    emitc.verbatim "DPRINT_UNPACK(})"
    emitc.verbatim "DPRINT_UNPACK(DPRINT << ENDL();)"
    emitc.call_opaque "cb_wait_front"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_pop_front"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_reserve_back"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_push_back"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_wait_front"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_pop_front"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_reserve_back"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "cb_push_back"(%4, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
    emitc.call_opaque "unary_op_init_common"(%5, %3) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
    emitc.call_opaque "copy_tile_init"(%5) : (!emitc.opaque<"::tt::CB">) -> ()
    emitc.call_opaque "tile_regs_acquire"() : () -> ()
    emitc.call_opaque "copy_tile"(%5, %2, %2) : (!emitc.opaque<"::tt::CB">, !emitc.size_t, !emitc.size_t) -> ()
    emitc.call_opaque "tile_regs_commit"() : () -> ()
    emitc.call_opaque "tile_regs_wait"() : () -> ()
    emitc.call_opaque "pack_tile"(%2, %3, %2) {template_args = [true]} : (!emitc.size_t, !emitc.opaque<"::tt::CB">, !emitc.size_t) -> ()
    emitc.call_opaque "tile_regs_release"() : () -> ()
    emitc.verbatim "DPRINT_PACK(DPRINT << \22cb_id \22 << get_compile_time_arg_val(3) << \22: { size: \22 << get_local_cb_interface(2).fifo_size << \22, limit: \22 << get_local_cb_interface(2).fifo_limit << \22, page_size: \22 << get_local_cb_interface(2).fifo_page_size << \22, num_pages: \22 << get_local_cb_interface(2).fifo_num_pages << \22, rd_ptr: \22 << get_local_cb_interface(2).fifo_rd_ptr << \22, wr_ptr: \22 << get_local_cb_interface(2).fifo_wr_ptr << \22, wr_tile_ptr: \22 << get_local_cb_interface(2).fifo_wr_tile_ptr << \22 }\22;)"
    emitc.verbatim "DPRINT_PACK(DPRINT << \22cb_idx: \22 << (uint8_t)get_compile_time_arg_val(3) << \22 tile_idx: \22 << 0 << ENDL();)"
    emitc.verbatim "DPRINT_PACK(DPRINT << \22======OUTPUT======\22 << ENDL();)"
    emitc.verbatim "DPRINT_PACK(for (uint16_t r = 0; r < 32; ++r) {)"
    emitc.verbatim "DPRINT_PACK(  DPRINT << (uint)r << \22 : \22;)"
    emitc.verbatim "DPRINT_PACK(  for (uint16_t c = 0; c < 32; c+=16) {)"
    emitc.verbatim "DPRINT_PACK(    DPRINT << \22 \22 << TileSlice((uint8_t)get_compile_time_arg_val(3), 0, SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = (uint8_t)1, .w0 = (uint8_t)(c), .w1 = (uint8_t)(c + 16), .ws = (uint8_t)1}, false, true);)"
    emitc.verbatim "DPRINT_PACK(  })"
    emitc.verbatim "DPRINT_PACK(  DPRINT << ENDL();)"
    emitc.verbatim "DPRINT_PACK(})"
    emitc.verbatim "DPRINT_PACK(DPRINT << ENDL();)"
    return
  }
  func.func private @DataFlow1_kernel_9() attributes {arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 3>]>, tt.function_type = "kernel", ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @DataFlow0_kernel_9() attributes {arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 1>, <arg_type = buffer_address, operand_index = 2>]>, tt.function_type = "kernel", ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @forward() attributes {polyblocks.entry_function, tt.function_type = "forward_device"} {
    %0 = "ttmetal.create_buffer"() <{address = 103712 : i64}> : () -> memref<4x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %1 = "ttmetal.create_buffer"() <{address = 106304 : i64}> : () -> memref<4x2x3x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1>
    %2 = "ttmetal.create_buffer"() <{address = 106304 : i64}> : () -> memref<4x2x3x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %3 = "ttmetal.create_buffer"() <{address = 118592 : i64}> : () -> memref<4x2x3x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    "ttmetal.enqueue_program"(%0, %0, %1, %3, %3) <{cb_ports = array<i64: 31, 0, 1, 2, 3>, kernelConfigs = [#ttmetal.noc_config<@DataFlow0_kernel_9, #ttmetal.core_range<0x0, 4x2>, #ttmetal.kernel_args< ct_args = [<cb_port[1]>, <cb_port[2]>]>, noc0>, #ttmetal.noc_config<@DataFlow1_kernel_9, #ttmetal.core_range<0x0, 4x2>, #ttmetal.kernel_args< ct_args = [<cb_port[3]>]>, noc1>, #ttmetal.compute_config<@Compute_kernel_9, #ttmetal.core_range<0x0, 4x2>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>, <cb_port[3]>, <cb_port[4]>]>, hifi4, true, true, false, [default, default, default, default, default]>], operandSegmentSizes = array<i32: 0, 5>}> : (memref<4x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<4x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<4x2x3x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<4x2x3x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<4x2x3x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) -> ()
    "ttmetal.deallocate_buffer"(%0) : (memref<4x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) -> ()
    "ttmetal.deallocate_buffer"(%1) : (memref<4x2x3x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1>) -> ()
    "ttmetal.deallocate_buffer"(%3) : (memref<4x2x3x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) -> ()
    "ttmetal.finish"() : () -> ()
    return
  }
}
