
/tmp/perf_overhead_artifacts/wc/UNPACK_ISOLATE/unpack.elf:     file format elf32-littleriscv


Disassembly of section .init:

00006000 <_start>:
// even though -fno-asynchronous-unwind-tables -fno-exceptions flags are set
void* __gxx_personality_v0;

__attribute__((no_profile_instrument_function)) TT_ALWAYS_INLINE void do_crt0()
{
    asm volatile(
    6000:	auipc	gp,0xffafb
    6004:	addi	gp,gp,-2048 # ffb00800 <__global_pointer$>
        "la gp, __global_pointer$\n"
        ".option pop" ::
            : "memory");

    // Set stack pointer
    asm volatile("la sp, %0" : : "i"(__stack_top) : "memory");
    6008:	auipc	sp,0xffafb
    600c:	addi	sp,sp,-8 # ffb01000 <__stack_top>

    // Initialize .bss
    for (volatile std::uint32_t* p = (volatile std::uint32_t*)__ldm_bss_start; p < (volatile std::uint32_t*)__ldm_bss_end; p++)
    6010:	lui	a5,0xffb00
    6014:	lui	a4,0xffb00
    6018:	addi	a5,a5,72 # ffb00048 <llk_profiler::open_zone_cnt>
    601c:	addi	a4,a4,140 # ffb0008c <__gcov_info_end>
    6020:	bgeu	a5,a4,6044 <_start+0x44>
    6024:	addi	a4,a4,-1
    6028:	sub	a4,a4,a5
    602c:	andi	a4,a4,-4
    6030:	addi	a4,a4,4
    6034:	add	a4,a4,a5
    {
        *p = 0;
    6038:	sw	zero,0(a5)
    for (volatile std::uint32_t* p = (volatile std::uint32_t*)__ldm_bss_start; p < (volatile std::uint32_t*)__ldm_bss_end; p++)
    603c:	addi	a5,a5,4
    6040:	bne	a5,a4,6038 <_start+0x38>
    }

    // Copy .loader_init to .ldm_data
    if ((std::uint32_t)__loader_init_start != (std::uint32_t)__loader_init_end)
    6044:	lui	a5,0xa
    6048:	lui	a4,0xb
    604c:	mv	a5,a5
    6050:	mv	a4,a4
    6054:	beq	a5,a4,6094 <_start+0x94>
    {
        volatile std::uint32_t* src = (volatile std::uint32_t*)__loader_init_start;
        volatile std::uint32_t* dst = (volatile std::uint32_t*)__ldm_data_start;
        volatile std::uint32_t* end = (volatile std::uint32_t*)__ldm_data_end;
        while (dst < end)
    6058:	lui	a4,0xffb00
    605c:	lui	a3,0xffb00
    6060:	mv	a4,a4
    6064:	addi	a3,a3,72 # ffb00048 <llk_profiler::open_zone_cnt>
    6068:	bgeu	a4,a3,6094 <_start+0x94>
    606c:	addi	a3,a3,-1
    6070:	sub	a3,a3,a4
    6074:	andi	a3,a3,-4
    6078:	addi	a3,a3,4
    607c:	add	a3,a3,a5
        {
            *dst++ = *src++;
    6080:	lw	a1,0(a5) # a000 <__loader_init_start>
    6084:	addi	a5,a5,4
    6088:	sw	a1,0(a4) # ffb00000 <llk_perf::freeze_and_read_all_counters(unsigned long)::banks>
    608c:	addi	a4,a4,4
        while (dst < end)
    6090:	bne	a5,a3,6080 <_start+0x80>
        }
    }

    // Execute global constructors
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
    6094:	lui	s0,0xffb00
    6098:	lui	s1,0xffb00
    609c:	addi	s0,s0,40 # ffb00028 <llk_profiler::buffer>
    60a0:	addi	s1,s1,40 # ffb00028 <llk_profiler::buffer>
    60a4:	bgeu	s0,s1,60b8 <_start+0xb8>
    {
        (*temp_constructor)();
    60a8:	lw	a5,0(s0)
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
    60ac:	addi	s0,s0,4
        (*temp_constructor)();
    60b0:	jalr	a5
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
    60b4:	bltu	s0,s1,60a8 <_start+0xa8>

extern "C" __attribute__((section(".init"), naked, noreturn, no_profile_instrument_function)) std::uint32_t _start()
{
    do_crt0();

    main();
    60b8:	jal	60c0 <main>

#ifdef COVERAGE
    gcov_dump();
#endif

    for (;;)
    60bc:	j	60bc <_start+0xbc>

Disassembly of section .text:

000060c0 <main>:
    volatile char *dstc       = reinterpret_cast<volatile char *>(dst);
    const volatile char *srcc = reinterpret_cast<const volatile char *>(src);

    for (std::size_t i = 0; i < len; i++)
    {
        dstc[i] = srcc[i];
    60c0:	lui	a5,0x20
    60c4:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
{
    60c8:	addi	sp,sp,-48
    60cc:	sb	a5,8(sp)
    60d0:	sw	ra,44(sp)
    60d4:	sw	s0,40(sp)
    60d8:	sw	s1,36(sp)
    60dc:	sw	s2,32(sp)
    60e0:	sw	s3,28(sp)
    }

    for (std::size_t i = 0; i < len; i++)
    {
        (void)(dstc[i]);
    60e4:	lbu	a5,8(sp)
    }

    asm volatile("fence" ::: "memory");
    60e8:	fence
    std::fill(ckernel::regfile, ckernel::regfile + 64, 0);
    60ec:	lw	a5,-2000(gp) # ffb00030 <ckernel::regfile>
      // otherwise we just use another reference.
      typedef typename __gnu_cxx::__conditional_type<__load_outside_loop,
						     const _Tp,
						     const _Tp&>::__type _Up;
      _Up __val(__value);
      for (; __first != __last; ++__first)
    60f0:	addi	a4,a5,256
	*__first = __val;
    60f4:	sw	zero,0(a5)
      for (; __first != __last; ++__first)
    60f8:	addi	a5,a5,4
    60fc:	bne	a4,a5,60f4 <main+0x34>
    }
}

__attribute__((always_inline)) inline void reset()
{
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
    6100:	lui	a0,0x16b
    6104:	addi	a5,a0,-12 # 16aff4 <__runtime_args_end+0x14abf4>
    6108:	sw	a5,-2004(gp) # ffb0002c <llk_profiler::barrier_ptr>
    TTI_NOP;
}

inline void reset_cfg_state_id()
{
    cfg_state_id = 0;
    610c:	sw	zero,-1960(gp) # ffb00058 <ckernel::cfg_state_id>
    buffer        = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
    write_idx     = 0;
    open_zone_cnt = 0;

    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
    6110:	lui	a2,0x1
    6114:	li	a1,0
    buffer        = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
    6118:	sw	a0,-2008(gp) # ffb00028 <llk_profiler::buffer>
}

inline void reset_dest_offset_id()
{
    dest_offset_id = 0;
    611c:	sw	zero,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
    write_idx     = 0;
    6120:	sw	zero,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    open_zone_cnt = 0;
    6124:	sw	zero,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
    6128:	jal	6fb0 <memset>
    auto& barrier = *barrier_ptr;
    612c:	lw	a2,-2004(gp) # ffb0002c <llk_profiler::barrier_ptr>
    barrier[TRISC_ID] = 1;
    6130:	li	a5,1
    6134:	sw	a5,0(a2) # 1000 <TRISC_LOCAL_MEM_LENGTH>
    asm volatile("fence" ::: "memory");
    6138:	fence
        while (barrier[i] != 1)
    613c:	lw	a1,4(a2)
    6140:	beq	a1,a5,615c <main+0x9c>
    6144:	mv	a1,a5
    6148:	addi	a4,a2,4
    614c:	li	a3,1
            asm volatile("fence" ::: "memory");
    6150:	fence
        while (barrier[i] != 1)
    6154:	lw	a5,0(a4)
    6158:	bne	a5,a3,6150 <main+0x90>
    for (std::uint32_t i = 0; i < NUM_CORES; ++i)
    615c:	li	a5,2
    6160:	beq	a1,a5,617c <main+0xbc>
        while (barrier[i] != 1)
    6164:	lw	a3,8(a2)
    6168:	li	a4,1
    616c:	beq	a3,a4,617c <main+0xbc>
    6170:	mv	a1,a5
    6174:	addi	a4,a2,8
    6178:	j	614c <main+0x8c>
    zone_scoped(zone_scoped&&)                 = delete;
    zone_scoped& operator=(const zone_scoped&) = delete;
    zone_scoped& operator=(zone_scoped&&)      = delete;

    inline __attribute__((always_inline)) zone_scoped()
    {
    617c:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    6180:	lw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    6184:	lw	a3,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        asm volatile("" ::: "memory");
        if (!is_buffer_full())
    6188:	li	a2,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    618c:	add	a5,a4,a3
    6190:	addi	a5,a5,-1021
        if (!is_buffer_full())
    6194:	bgeu	a2,a5,61dc <main+0x11c>
// now handled by the compiler)
// workaround is needed only for GS
inline std::uint32_t reg_read(std::uint32_t addr)
{
    volatile std::uint32_t tt_reg_ptr *p_reg = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(addr);
    return p_reg[0];
    6198:	lui	a5,0xffb12
    619c:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    61a0:	lw	a5,504(a5)
    61a4:	lw	a2,-2008(gp) # ffb00028 <llk_profiler::buffer>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    61a8:	lui	a0,0xa5104
        {
            is_opened = true;
            write_entry(EntryType::ZONE_START, id16);
            ++open_zone_cnt;
    61ac:	addi	a3,a3,1
    61b0:	sh2add	a2,a4,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    61b4:	slli	a5,a5,0x14
    61b8:	srli	a5,a5,0x14
    61bc:	or	a5,a5,a0
    61c0:	sw	a5,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    61c4:	addi	a4,a4,2
            is_opened = true;
    61c8:	li	a5,1
            ++open_zone_cnt;
    61cc:	sw	a3,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    61d0:	sw	a1,4(a2)
    61d4:	sw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            is_opened = true;
    61d8:	sb	a5,12(sp)
        run_kernel(temp_args);
    61dc:	addi	a0,sp,8
    61e0:	jal	65b0 <run_kernel(RuntimeParams const&)>
    store_blocking(&pc_buf_base[1], 0);
    61e4:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
    61e8:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    61ec:	addi	a4,a4,4
    asm volatile(
    61f0:	sw	a5,0(a4)
    61f4:	lw	a5,0(a4)
    61f8:	and	zero,zero,a5
    }

    ~zone_scoped()
    {
        asm volatile("" ::: "memory");
        if (is_opened)
    61fc:	lbu	a5,12(sp)
    6200:	beqz	a5,6248 <main+0x188>
    return p_reg[0];
    6204:	lui	a5,0xffb12
    6208:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    620c:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6210:	lw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
        {
            write_entry(EntryType::ZONE_END, id16);
            --open_zone_cnt;
    6214:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    6218:	lw	a3,-2008(gp) # ffb00028 <llk_profiler::buffer>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    621c:	lui	a0,0xb5104
    6220:	slli	a5,a5,0x14
    6224:	srli	a5,a5,0x14
    6228:	or	a5,a5,a0
    622c:	sh2add	a3,a4,a3
    6230:	sw	a5,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6234:	addi	a4,a4,2
            --open_zone_cnt;
    6238:	addi	a5,a2,-1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    623c:	sw	a1,4(a3)
    6240:	sw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
    6244:	sw	a5,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    *mailbox = ckernel::KERNEL_COMPLETE;
    6248:	lui	a5,0x20
}
    624c:	lw	ra,44(sp)
    6250:	lw	s0,40(sp)
    *mailbox = ckernel::KERNEL_COMPLETE;
    6254:	li	a4,255
    6258:	sw	a4,-72(a5) # 1ffb8 <__loader_init_end+0x14fb8>
}
    625c:	lw	s1,36(sp)
    6260:	lw	s2,32(sp)
    6264:	lw	s3,28(sp)
    6268:	li	a0,0
    626c:	addi	sp,sp,48
    6270:	ret

00006274 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>:
    }

    /// @brief Get total number of faces
    constexpr std::uint8_t total_num_faces() const
    {
        return num_faces_r_dim * num_faces_c_dim;
    6274:	lbu	a5,3(a0) # b5104003 <__runtime_args_end+0xb50e3c03>
    6278:	lbu	a4,2(a0)
 *
 * @param tensor_shape: Tensor shape to validate
 * @return true if tensor shape is valid, false otherwise
 **/
__attribute__((noinline)) bool validate_tensor_shape_tile_dependent_ops_(const TensorShape &tensor_shape)
{
    627c:	mv	a3,a0
        return num_faces_r_dim * num_faces_c_dim;
    6280:	mul	a4,a4,a5
    6284:	zext.b	a4,a4
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t face_c_dim = tensor_shape.face_c_dim;
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
    6288:	addi	a5,a4,-1
    628c:	addi	a4,a4,-4
    6290:	sltiu	a5,a5,2
    6294:	seqz	a4,a4
    6298:	or	a0,a5,a4
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
    629c:	beqz	a0,62d0 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    62a0:	lbu	a4,0(a3)
    62a4:	li	a5,16
    62a8:	li	a0,0
    62ac:	bltu	a5,a4,62d0 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
    62b0:	lui	a5,0x10
    62b4:	addi	a5,a5,278 # 10116 <__loader_init_end+0x5116>
    62b8:	srl	a5,a5,a4
    62bc:	andi	a0,a5,1
    62c0:	beqz	a0,62d0 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
    62c4:	lbu	a0,1(a3)
    62c8:	addi	a0,a0,-16
    62cc:	seqz	a0,a0
}
    62d0:	ret

000062d4 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)>:
 * \return true if the conversion is supported given the FP32 accumulation setting.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_unpacker_format_conversion_supported_fp32_acc(
    const DataFormat unpack_src_format, const DataFormat unpack_dst_format, const bool is_fp32_dest_acc_en)
{
    switch (unpack_src_format)
    62d4:	li	a4,9
{
    62d8:	mv	a5,a0
    switch (unpack_src_format)
    62dc:	beq	a0,a4,63fc <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x128>
    62e0:	bltu	a4,a0,631c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x48>
    62e4:	li	a4,4
    62e8:	beq	a0,a4,63c4 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xf0>
    62ec:	bgeu	a4,a0,6370 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x9c>
        //    ISA conversions:
        //      SrcA/SrcB path: NOT possible (ISA doc explicitly states "Not possible").
        //      Dst path:       INT32 → Integer "32" (Int32): valid (32b data movement to Dst).
        //    Hence, only valid when unpack_to_dest = true (checked in _dest).
        case DataFormat::Int32:
            return unpack_dst_format == DataFormat::Int32;
    62f0:	addi	a0,a1,-8
    switch (unpack_src_format)
    62f4:	li	a3,8
            return unpack_dst_format == DataFormat::Int32;
    62f8:	seqz	a0,a0
    switch (unpack_src_format)
    62fc:	beq	a5,a3,636c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    6300:	li	a3,5
    6304:	bne	a5,a3,6408 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x134>
                    return is_fp32_dest_acc_en;
    6308:	mv	a0,a2
            switch (unpack_dst_format)
    630c:	beq	a1,a4,636c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    6310:	addi	a1,a1,-5
    6314:	seqz	a0,a1
    6318:	ret
    switch (unpack_src_format)
    631c:	li	a4,15
    6320:	beq	a0,a4,6408 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x134>
    6324:	bgeu	a4,a0,6394 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xc0>
    6328:	li	a4,26
    632c:	beq	a0,a4,637c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xa8>
    6330:	li	a4,30
    6334:	beq	a0,a4,6428 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x154>
    6338:	addi	a5,a0,-24
        // 8. UInt32 (opaque 32-bit) in L1.
        //
        //    Not explicitly listed in the ISA doc. Treated as opaque 32-bit data analogous to
        //    Int32: only valid when targeting the Dst register (unpack_to_dest = true, checked in _dest).
        case DataFormat::UInt32:
            return unpack_dst_format == DataFormat::UInt32;
    633c:	addi	a1,a1,-24
    6340:	seqz	a1,a1
    switch (unpack_src_format)
    6344:	seqz	a5,a5
    6348:	and	a0,a1,a5
    634c:	ret
        //         INT8 → Integer "8" (Int8): always valid.
        //       Dst path:
        //         INT8 → BF16  (Float16_b): always valid (BFP8+force_shared_exp).
        //         INT8 → Integer "8" (Int8): always valid.
        case DataFormat::Int8:
            switch (unpack_dst_format)
    6350:	li	a4,5
                    return true;
    6354:	li	a0,1
            switch (unpack_dst_format)
    6358:	beq	a1,a4,636c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    635c:	beq	a1,a5,636c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    6360:	addi	a1,a1,-4
    6364:	seqz	a1,a1
    6368:	and	a0,a2,a1
        // -------------------------------------------------------------------------
        // 12. Unknown or not-yet-handled formats.
        default:
            return false;
    }
}
    636c:	ret
    switch (unpack_src_format)
    6370:	beqz	a0,63e0 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x10c>
    6374:	li	a4,1
    6378:	bne	a0,a4,63ac <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xd8>
            return unpack_dst_format == DataFormat::Float16 || unpack_src_format == unpack_dst_format;
    637c:	sub	a5,a5,a1
    6380:	addi	a1,a1,-1
    6384:	seqz	a5,a5
    6388:	seqz	a1,a1
    638c:	or	a0,a1,a5
    6390:	ret
    switch (unpack_src_format)
    6394:	li	a4,14
    6398:	beq	a0,a4,6350 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x7c>
    639c:	li	a4,10
    63a0:	beq	a0,a4,637c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xa8>
    63a4:	li	a4,11
    63a8:	bne	a0,a4,6438 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x164>
            switch (unpack_dst_format)
    63ac:	li	a4,1
                    return true;
    63b0:	mv	a0,a1
            switch (unpack_dst_format)
    63b4:	beq	a1,a4,636c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
                    return unpack_src_format == unpack_dst_format;
    63b8:	sub	a5,a5,a1
    63bc:	seqz	a0,a5
    63c0:	ret
                    return is_fp32_dest_acc_en;
    63c4:	mv	a0,a2
            switch (unpack_dst_format)
    63c8:	beq	a1,a5,636c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    63cc:	bltu	a5,a1,6310 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x3c>
    63d0:	beqz	a1,636c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    63d4:	addi	a1,a1,-1
    63d8:	seqz	a0,a1
    63dc:	ret
                    return is_fp32_dest_acc_en;
    63e0:	mv	a0,a2
            switch (unpack_dst_format)
    63e4:	beq	a1,a4,636c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    63e8:	addi	a0,a1,-5
    63ec:	seqz	a0,a0
    63f0:	bltu	a4,a1,636c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    63f4:	sltiu	a0,a1,2
    63f8:	ret
            return unpack_dst_format == DataFormat::UInt16;
    63fc:	addi	a1,a1,-9
    6400:	seqz	a0,a1
    6404:	ret
            switch (unpack_dst_format)
    6408:	li	a4,4
                    return is_fp32_dest_acc_en;
    640c:	mv	a0,a2
            switch (unpack_dst_format)
    6410:	beq	a1,a4,636c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    6414:	addi	a4,a1,-5
    6418:	zext.b	a4,a4
    641c:	li	a0,1
    6420:	bltu	a0,a4,63b8 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xe4>
    6424:	ret
            return unpack_dst_format == DataFormat::Int8 || unpack_dst_format == DataFormat::UInt8;
    6428:	andi	a1,a1,239
    642c:	addi	a1,a1,-14
    6430:	seqz	a0,a1
    6434:	ret
    switch (unpack_src_format)
    6438:	li	a0,0
    643c:	ret

00006440 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)>:
 * \return true if the conversion is supported given the register destination.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_unpacker_format_conversion_supported_dest(
    const DataFormat unpack_src_format, const DataFormat unpack_dst_format, const bool unpack_to_dest)
{
    switch (unpack_src_format)
    6440:	li	a4,9
{
    6444:	mv	a5,a0
    switch (unpack_src_format)
    6448:	beq	a0,a4,65a0 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x160>
    644c:	bltu	a4,a0,647c <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x3c>
    6450:	li	a4,4
    6454:	beq	a0,a4,6544 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x104>
    6458:	bgeu	a4,a0,6520 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xe0>
    645c:	li	a3,8
    6460:	beq	a0,a3,64c0 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x80>
    6464:	li	a3,5
    6468:	bne	a0,a3,6580 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x140>
        //      Dst path:
        //        BF16 → BF16  (Float16_b): always valid (identity).
        //    Note: FP16 is NOT a valid output for BF16 input — no cross-exponent-width conversion
        //    from 8-bit exponent BF16 to 5-bit exponent FP16 is supported by the unpacker.
        case DataFormat::Float16_b:
            switch (unpack_dst_format)
    646c:	beq	a1,a4,64f0 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xb0>
    6470:	addi	a1,a1,-5
    6474:	seqz	a0,a1
    6478:	ret
    switch (unpack_src_format)
    647c:	li	a4,24
    6480:	beq	a0,a4,6568 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x128>
    6484:	bltu	a4,a0,64f8 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xb8>
    6488:	li	a4,14
    648c:	beq	a0,a4,64d0 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x90>
    6490:	bltu	a4,a0,6578 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x138>
    6494:	addi	a4,a0,-10
    6498:	zext.b	a4,a4
    649c:	li	a3,1
    64a0:	li	a0,0
    64a4:	bltu	a3,a4,65ac <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x16c>
            return unpack_dst_format == DataFormat::Float16 || unpack_src_format == unpack_dst_format;
    64a8:	sub	a5,a5,a1
    64ac:	addi	a1,a1,-1
    64b0:	seqz	a5,a5
    64b4:	seqz	a1,a1
    64b8:	or	a0,a1,a5
    64bc:	ret
        //    ISA conversions:
        //      SrcA/SrcB path: NOT possible (ISA doc explicitly states "Not possible").
        //      Dst path:       INT32 → Integer "32" (Int32): valid (32b data movement to Dst).
        //    Hence, only valid when unpack_to_dest = true.
        case DataFormat::Int32:
            return unpack_dst_format == DataFormat::Int32 && unpack_to_dest;
    64c0:	addi	a1,a1,-8
    64c4:	seqz	a1,a1
    64c8:	and	a0,a2,a1
    64cc:	ret
        //         INT8 → Integer "8" (Int8): always valid.
        //       Dst path:
        //         INT8 → BF16  (Float16_b): always valid (BFP8+force_shared_exp).
        //         INT8 → Integer "8" (Int8): always valid.
        case DataFormat::Int8:
            switch (unpack_dst_format)
    64d0:	li	a4,5
                    return true;
    64d4:	li	a0,1
            switch (unpack_dst_format)
    64d8:	beq	a1,a4,64ec <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
    64dc:	beq	a1,a5,64ec <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
    64e0:	li	a5,4
    64e4:	beq	a1,a5,64f0 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xb0>
    64e8:	li	a0,0
        // -------------------------------------------------------------------------
        // 12. Unknown or not-yet-handled formats.
        default:
            return false;
    }
}
    64ec:	ret
                    return !unpack_to_dest;
    64f0:	xori	a0,a2,1
    64f4:	ret
    switch (unpack_src_format)
    64f8:	li	a4,26
    64fc:	beq	a0,a4,64a8 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x68>
    6500:	addi	a5,a0,-30
            return unpack_dst_format == DataFormat::Int8 || unpack_dst_format == DataFormat::UInt8;
    6504:	andi	a1,a1,239
    6508:	addi	a1,a1,-14
    switch (unpack_src_format)
    650c:	seqz	a5,a5
            return unpack_dst_format == DataFormat::Int8 || unpack_dst_format == DataFormat::UInt8;
    6510:	seqz	a0,a1
    switch (unpack_src_format)
    6514:	neg	a5,a5
    6518:	and	a0,a0,a5
    651c:	ret
    6520:	bnez	a0,64a8 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x68>
            switch (unpack_dst_format)
    6524:	li	a5,1
                    return true;
    6528:	mv	a0,a1
            switch (unpack_dst_format)
    652c:	beq	a1,a5,64ec <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
                    return unpack_to_dest;
    6530:	mv	a0,a2
            switch (unpack_dst_format)
    6534:	bgeu	a5,a1,64ec <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
    6538:	addi	a1,a1,-4
    653c:	sltiu	a0,a1,2
    6540:	ret
            switch (unpack_dst_format)
    6544:	beq	a1,a0,64f0 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xb0>
    6548:	addi	a0,a1,-5
    654c:	seqz	a0,a0
    6550:	bltu	a5,a1,64ec <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
                    return unpack_to_dest;
    6554:	mv	a0,a2
            switch (unpack_dst_format)
    6558:	beqz	a1,64ec <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
    655c:	addi	a1,a1,-1
    6560:	seqz	a0,a1
    6564:	ret
            return unpack_dst_format == DataFormat::UInt32 && unpack_to_dest;
    6568:	addi	a1,a1,-24
    656c:	seqz	a1,a1
    6570:	and	a0,a2,a1
    6574:	ret
    switch (unpack_src_format)
    6578:	li	a4,15
    657c:	bne	a0,a4,64e8 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xa8>
            switch (unpack_dst_format)
    6580:	li	a4,4
    6584:	beq	a1,a4,64f0 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xb0>
                    return unpack_src_format == unpack_dst_format;
    6588:	sub	a5,a5,a1
            switch (unpack_dst_format)
    658c:	addi	a1,a1,-5
                    return unpack_src_format == unpack_dst_format;
    6590:	seqz	a5,a5
            switch (unpack_dst_format)
    6594:	sltiu	a1,a1,2
    6598:	or	a0,a1,a5
    659c:	ret
            return unpack_dst_format == DataFormat::UInt16;
    65a0:	addi	a1,a1,-9
    65a4:	seqz	a0,a1
    65a8:	ret
    65ac:	ret

000065b0 <run_kernel(RuntimeParams const&)>:

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    65b0:	addi	sp,sp,-80
    65b4:	sw	s5,52(sp)
#endif
} // namespace detail

__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    std::uint32_t n = detail::next_zone_id;
    65b8:	addi	s5,gp,-1944 # ffb00068 <llk_perf::detail::next_zone_id>
    65bc:	lw	a5,0(s5)
    65c0:	sw	ra,76(sp)
    65c4:	sw	s0,72(sp)
    65c8:	sw	s1,68(sp)
    65cc:	sw	s2,64(sp)
    65d0:	sw	s3,60(sp)
    65d4:	sw	s4,56(sp)
    65d8:	sw	s6,48(sp)
    65dc:	sw	s7,44(sp)
    for (std::uint32_t i = 0; i < n; ++i)
    65e0:	beqz	a5,6dec <run_kernel(RuntimeParams const&)+0x83c>
    {
        if (detail::zone_hashes[i] == hash_val)
    65e4:	lui	a4,0x7c867
    65e8:	lw	a3,4(s5)
    65ec:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    65f0:	beq	a3,a4,6674 <run_kernel(RuntimeParams const&)+0xc4>
    for (std::uint32_t i = 0; i < n; ++i)
    65f4:	li	a3,1
    65f8:	beq	a5,a3,6dec <run_kernel(RuntimeParams const&)+0x83c>
        if (detail::zone_hashes[i] == hash_val)
    65fc:	lw	a2,8(s5)
    6600:	beq	a2,a4,6e74 <run_kernel(RuntimeParams const&)+0x8c4>
    for (std::uint32_t i = 0; i < n; ++i)
    6604:	li	a3,2
    6608:	beq	a5,a3,6dec <run_kernel(RuntimeParams const&)+0x83c>
        if (detail::zone_hashes[i] == hash_val)
    660c:	lw	a2,12(s5)
    6610:	beq	a2,a4,6e74 <run_kernel(RuntimeParams const&)+0x8c4>
    for (std::uint32_t i = 0; i < n; ++i)
    6614:	li	a3,3
    6618:	beq	a5,a3,6dec <run_kernel(RuntimeParams const&)+0x83c>
        if (detail::zone_hashes[i] == hash_val)
    661c:	lw	a2,16(s5)
    6620:	beq	a2,a4,6e74 <run_kernel(RuntimeParams const&)+0x8c4>
    for (std::uint32_t i = 0; i < n; ++i)
    6624:	li	a3,4
    6628:	beq	a5,a3,6dec <run_kernel(RuntimeParams const&)+0x83c>
        if (detail::zone_hashes[i] == hash_val)
    662c:	lw	a3,20(s5)
    6630:	beq	a3,a4,6f70 <run_kernel(RuntimeParams const&)+0x9c0>
    for (std::uint32_t i = 0; i < n; ++i)
    6634:	li	a3,5
    6638:	beq	a5,a3,6dec <run_kernel(RuntimeParams const&)+0x83c>
        if (detail::zone_hashes[i] == hash_val)
    663c:	lui	a4,0x7c867
    6640:	lw	a2,24(s5)
    6644:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    6648:	beq	a2,a4,6e74 <run_kernel(RuntimeParams const&)+0x8c4>
    for (std::uint32_t i = 0; i < n; ++i)
    664c:	li	a3,6
    6650:	beq	a5,a3,6dec <run_kernel(RuntimeParams const&)+0x83c>
        if (detail::zone_hashes[i] == hash_val)
    6654:	lw	a2,28(s5)
    6658:	beq	a2,a4,6e74 <run_kernel(RuntimeParams const&)+0x8c4>
    for (std::uint32_t i = 0; i < n; ++i)
    665c:	li	a3,7
    6660:	beq	a5,a3,6dec <run_kernel(RuntimeParams const&)+0x83c>
        if (detail::zone_hashes[i] == hash_val)
    6664:	lw	a2,32(s5)
    6668:	beq	a2,a4,6e74 <run_kernel(RuntimeParams const&)+0x8c4>
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
    666c:	li	a4,8
    6670:	bne	a5,a4,6dec <run_kernel(RuntimeParams const&)+0x83c>
    {
        detail::zone_hashes[n] = hash_val;
        detail::next_zone_id   = n + 1;
        return n;
    }
    return 0;
    6674:	li	a5,0
    perf_counter_scoped(const perf_counter_scoped&)            = delete;
    perf_counter_scoped(perf_counter_scoped&&)                 = delete;
    perf_counter_scoped& operator=(const perf_counter_scoped&) = delete;
    perf_counter_scoped& operator=(perf_counter_scoped&&)      = delete;

    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
    6678:	sw	a5,16(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 1u; // PERF_CNT_ALL (INSTRN+FPU)
    667c:	lui	a5,0xffb12
    6680:	li	a4,1
    6684:	sw	a4,60(a5) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 1u; // TDMA_UNPACK
    6688:	sw	a4,20(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 1u; // L1
    668c:	sw	a4,56(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 1u; // TDMA_PACK
    6690:	sw	a4,248(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6694:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    6698:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    669c:	li	a3,14
    66a0:	zext.b	a4,a4
    66a4:	bltu	a3,a4,6e7c <run_kernel(RuntimeParams const&)+0x8cc>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    66a8:	sw	zero,32(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    66ac:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    66b0:	li	a3,14
    66b4:	zext.b	a4,a4
    66b8:	bltu	a3,a4,6e98 <run_kernel(RuntimeParams const&)+0x8e8>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    66bc:	sw	zero,32(a5)
    {
    66c0:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    66c4:	lw	a5,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    66c8:	lw	a3,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    66cc:	li	a2,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    66d0:	add	a4,a5,a3
    66d4:	addi	a4,a4,-1021
        if (!is_buffer_full())
    66d8:	bgeu	a2,a4,6720 <run_kernel(RuntimeParams const&)+0x170>
    return p_reg[0];
    66dc:	lui	a4,0xffb12
    66e0:	lw	a1,496(a4) # ffb121f0 <__stack_top+0x111f0>
    66e4:	lw	a4,504(a4)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    66e8:	lw	a2,-2008(gp) # ffb00028 <llk_profiler::buffer>
    66ec:	lui	a0,0xaa3a0
    66f0:	slli	a4,a4,0x14
    66f4:	srli	a4,a4,0x14
    66f8:	or	a4,a4,a0
            ++open_zone_cnt;
    66fc:	addi	a3,a3,1
    6700:	sh2add	a2,a5,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6704:	sw	a4,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6708:	addi	a5,a5,2
            is_opened = true;
    670c:	li	a4,1
            ++open_zone_cnt;
    6710:	sw	a3,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6714:	sw	a1,4(a2)
    6718:	sw	a5,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            is_opened = true;
    671c:	sb	a4,12(sp)
    const std::uint32_t unpB_num_faces  = 4)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
    6720:	li	a1,6
    6724:	mv	a0,a1
    6728:	li	a2,1
    672c:	jal	62d4 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)>
    6730:	beqz	a0,6f18 <run_kernel(RuntimeParams const&)+0x968>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6734:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    6738:	lw	a5,52(a4)
    while (semaphore_read(semaphore::UNPACK_SYNC) > 0)
    673c:	zext.b	a5,a5
    6740:	bnez	a5,6738 <run_kernel(RuntimeParams const&)+0x188>
    TTI_SETADCXY(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1011);
    6744:	ttsetadcxy	3,0,0,0,0,11
    TTI_SETADCZW(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1111);
    6748:	ttsetadczw	3,0,0,0,0,15
    if (cfg_state_id == 0)
    674c:	lw	a4,-1960(gp) # ffb00058 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    6750:	lui	a5,0xffef0
    if (cfg_state_id == 0)
    6754:	beqz	a4,675c <run_kernel(RuntimeParams const&)+0x1ac>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
    6758:	addi	a5,a5,896 # ffef0380 <__instrn_buffer+0xb0380>
    std::uint32_t unpA_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpA_ch1_x_stride;
    std::uint32_t unpB_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpB_ch1_x_stride;
    std::uint32_t exp_width         = (static_cast<std::uint32_t>(unpA_dst_format_masked) >> 2) & 0x1; // 0=5-bit, 1=8-bit

    // Strides for incrementing ch1 address to srcA and srcB
    cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
    675c:	li	a1,256
    6760:	sw	a1,228(a5)
        (0 << UNP0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
        (unpA_ch1_z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT); // Z and W(not used) stride for dest address (ch1)

    cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
    6764:	sw	a1,236(a5)
    TTI_ATGETM(index);
    6768:	ttatgetm	0
    std::uint8_t mask_b0 = Mask & 0xff;

    if (mask_b0 != 0)
    {
        std::uint8_t data_b0 = wrdata & 0xff;
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    676c:	lui	s6,0xffe40
    6770:	mv	s6,s6
    6774:	lui	a4,0xb3ff0
    6778:	sw	a4,0(s6) # ffe40000 <__instrn_buffer>
    std::uint8_t mask_b1 = (Mask >> 8) & 0xff;

    if (mask_b1 != 0)
    {
        std::uint8_t data_b1 = (wrdata) & 0xff;
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    677c:	lui	a4,0xb47f0
    6780:	sw	a4,0(s6)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    6784:	lui	a4,0xb3070
    6788:	addi	a4,a4,1 # b3070001 <__runtime_args_end+0xb304fc01>
    678c:	sw	a4,0(s6)
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    6790:	lui	a4,0xb4800
    6794:	addi	a4,a4,1 # b4800001 <__runtime_args_end+0xb47dfc01>
    6798:	sw	a4,0(s6)
    std::uint8_t mask_b2 = (Mask >> 16) & 0xff;

    if (mask_b2 != 0)
    {
        std::uint8_t data_b2 = (wrdata) & 0xff;
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
    679c:	lui	a4,0xb5010
    67a0:	addi	a4,a4,1 # b5010001 <__runtime_args_end+0xb4fefc01>
    67a4:	sw	a4,0(s6)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    67a8:	lui	a4,0xb3010
    67ac:	addi	a4,a4,2 # b3010002 <__runtime_args_end+0xb2fefc02>
    67b0:	sw	a4,0(s6)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
    67b4:	lui	a4,0xb5400
    67b8:	addi	a3,a4,71 # b5400047 <__runtime_args_end+0xb53dfc47>
    67bc:	sw	a3,0(s6)
    67c0:	addi	a4,a4,119
    67c4:	sw	a4,0(s6)
    TTI_ATRELM(index);
    67c8:	ttatrelm	0
    tile_descriptor.f.z_dim          = unpA_num_faces;
    // tile_descriptor.f.blobs_per_xy_plane = 0;
    // tile_descriptor.f.blobs_y_start = 0;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67cc:	li	a4,22
    67d0:	sw	a4,256(a5)
    67d4:	lui	a4,0x40
    67d8:	addi	a4,a4,1 # 40001 <__runtime_args_end+0x1fc01>
    tile_descriptor.f.in_data_format = row_pool ? to_underlying(DataFormat::Float32) : unpB_src_format_masked;
    tile_descriptor.f.x_dim          = unpB_face_r_dim * FACE_C_DIM;
    tile_descriptor.f.z_dim          = unpB_num_faces;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67dc:	lui	a3,0x1000
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67e0:	sw	a4,260(a5)
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67e4:	addi	a2,a3,22 # 1000016 <__runtime_args_end+0xfdfc16>
    67e8:	sw	a2,448(a5)
    67ec:	sw	a4,452(a5)
    config.f.uncompress_cntx4_7 = 0xf;
    // config.f.limit_addr = 0; // Set dynamically
    // config.f.fifo_size = 0; // Set dynamically
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    67f0:	li	a2,38
    67f4:	lui	a4,0xf0
    67f8:	sw	a2,288(a5)
    67fc:	addi	a4,a4,15 # f000f <__runtime_args_end+0xcfc0f>
    6800:	sw	a4,292(a5)
    config.f.out_data_format = row_pool ? (to_underlying(DataFormat::Float16) | (exp_width << 2)) : unpB_dst_format_masked;
    config.f.haloize_mode    = 0;

    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC1_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    6804:	sw	a2,480(a5)
    6808:	sw	a4,484(a5)
    }

    std::uint32_t unpA_x_end = (unpA_face_r_dim == 0) ? 1 : (unpA_face_r_dim << 4) - 1;
    TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    680c:	lui	a4,0x5e240
    6810:	addi	a4,a4,-1024 # 5e23fc00 <__runtime_args_end+0x5e21f800>
    6814:	sw	a4,0(s6)
    TT_SETADCXX(p_setadc::UNP_B, (unpB_face_r_dim << 4) - 1, 0x0);
    6818:	lui	a4,0x5e440
    681c:	addi	a4,a4,-1024 # 5e43fc00 <__runtime_args_end+0x5e41f800>

    // Program base address for all 2 sections (each section address is loaded to corresponding context)
    // Load dummy data to unused location if face height is 0
    const std::uint32_t Dest_cntx0_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    const std::uint32_t Dest_cntx1_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);
    6820:	lui	a2,0x400
    TT_SETADCXX(p_setadc::UNP_B, (unpB_face_r_dim << 4) - 1, 0x0);
    6824:	sw	a4,0(s6)
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);
    6828:	addi	a2,a2,64 # 400040 <__runtime_args_end+0x3dfc40>
    682c:	sw	a2,336(a5)
    // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
    const std::uint32_t face_dim                 = unpA_face_r_dim * FACE_C_DIM;
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = face_dim | (face_dim << 16);

    constexpr std::uint32_t face_dim_16x16 = FACE_R_DIM * FACE_C_DIM;
    regfile[p_gpr_unpack::FACE_DIM_16x16]  = (face_dim_16x16 / 1) | ((face_dim_16x16 / 1) << 16);
    6830:	lw	a4,-2000(gp) # ffb00030 <ckernel::regfile>
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = face_dim | (face_dim << 16);
    6834:	add	a3,a3,a1
    6838:	sw	a3,344(a5)
    regfile[p_gpr_unpack::FACE_DIM_8x16]   = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    683c:	lui	a0,0x800
    regfile[p_gpr_unpack::FACE_DIM_16x16]  = (face_dim_16x16 / 1) | ((face_dim_16x16 / 1) << 16);
    6840:	sw	a3,160(a4)
    regfile[p_gpr_unpack::FACE_DIM_8x16]   = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    6844:	addi	a3,a0,128 # 800080 <__runtime_args_end+0x7dfc80>
    6848:	sw	a3,164(a4)
    regfile[p_gpr_unpack::FACE_DIM_4x16]   = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    684c:	lui	a3,0x200
    regfile[p_gpr_unpack::FACE_DIM_4x16]   = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    6850:	sw	a2,168(a4)
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    6854:	addi	a2,a3,32 # 200020 <__runtime_args_end+0x1dfc20>
    regfile[p_gpr_unpack::FACE_DIM_1x16]   = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    6858:	lui	a3,0x100
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    685c:	sw	a2,172(a4)
    regfile[p_gpr_unpack::FACE_DIM_1x16]   = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    6860:	addi	a3,a3,16 # 100010 <__runtime_args_end+0xdfc10>
    6864:	sw	a3,176(a4)
    volatile std::uint32_t foo     = 0x0;
    6868:	sw	zero,24(sp)
    *fooptr                        = regfile[index];
    686c:	lw	a4,176(a4)
    6870:	sw	a4,24(sp)
    sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);

    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);
    6874:	ttsetc16	5,4

    // Enable address counter for unpacker ch1/dst address
    // final address is calculated as: Dest_cntx0/1_address + address_counter_ch1
    // used for face by face unpacking of entire tile into srcA
    cfg[UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32] = 0x1 << UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT;
    6878:	sw	a1,200(a5)
    unp_cfg_context = 0;
    687c:	sw	zero,-1948(gp) # ffb00064 <unp_cfg_context>
    TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    6880:	ttsetc16	41,0
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");
    configure_unpack_AB<is_fp32_dest_acc_en, false, false, false, disable_src_zero_flag>(
        unpA_src_format, unpB_src_format, unpA_dst_format, unpB_dst_format, unpA_face_r_dim, unpB_face_r_dim, 0, unpA_num_faces, unpB_num_faces);

    TT_SETDMAREG(0, LOWER_HALFWORD(unpA_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    6884:	lui	a5,0x45000
    6888:	addi	a3,a5,72 # 45000048 <__runtime_args_end+0x44fdfc48>
    688c:	sw	a3,0(s6)
    6890:	lui	a4,0x2021
    TT_SETDMAREG(0, LOWER_HALFWORD(unpB_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
    6894:	addi	a5,a5,74
    6898:	addi	a4,a4,16 # 2021010 <__runtime_args_end+0x2000c10>
    689c:	sw	a5,0(s6)
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    68a0:	addi	a0,sp,20
    68a4:	sw	a4,20(sp)
    68a8:	jal	6274 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>
    68ac:	beqz	a0,6ef4 <run_kernel(RuntimeParams const&)+0x944>
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    68b0:	lui	a5,0xb4010
    68b4:	addi	a5,a5,72 # b4010048 <__runtime_args_end+0xb3fefc48>
    68b8:	sw	a5,0(s6)
            break;
        case 8:
            TTI_SETADCXX(UNP_SEL, 8 * FACE_C_DIM - 1, 0x0);
            break;
        default:
            TTI_SETADCXX(UNP_SEL, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    68bc:	ttsetadcxx	3,255,0
    68c0:	lw	a5,20(sp)
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    68c4:	addi	a0,sp,28
    68c8:	srli	s7,a5,0x10
    68cc:	sw	a5,28(sp)
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    68d0:	zext.b	s7,s7
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
    68d4:	srli	s6,a5,0x18
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    68d8:	jal	6274 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>
    68dc:	beqz	a0,6eec <run_kernel(RuntimeParams const&)+0x93c>
    store_blocking(&pc_buf_base[2], 0);
    68e0:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
    68e4:	li	a3,0
    store_blocking(&pc_buf_base[2], 0);
    68e8:	addi	a5,a5,8
    asm volatile(
    68ec:	sw	a3,0(a5)
    68f0:	lw	a3,0(a5)
    68f4:	and	zero,zero,a3
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
    68f8:	lui	a5,0xffb80
    68fc:	sw	s7,0(a5) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
    6900:	sw	s6,4(a5)
    mop_cfg[2] = m_start_op0;
    6904:	lui	a3,0x2000
    6908:	sw	a3,8(a5)
    mop_cfg[3] = m_end_op0;
    690c:	sw	a3,12(a5)
    mop_cfg[4] = m_end_op1;
    mop_cfg[5] = m_loop_op0;
    6910:	lui	a2,0x42008
    mop_cfg[4] = m_end_op1;
    6914:	sw	a3,16(a5)
    mop_cfg[5] = m_loop_op0;
    6918:	addi	a2,a2,193 # 420080c1 <__runtime_args_end+0x41fe7cc1>
    mop_cfg[6] = m_loop_op1;
    691c:	lui	a3,0x42808
    mop_cfg[5] = m_loop_op0;
    6920:	sw	a2,20(a5)
    mop_cfg[6] = m_loop_op1;
    6924:	addi	a3,a3,193 # 428080c1 <__runtime_args_end+0x427e7cc1>
    6928:	sw	a3,24(a5)
    mop_cfg[7] = m_loop0_last_instr;
    692c:	sw	a3,28(a5)
    mop_cfg[8] = m_loop1_last_instr;
    6930:	sw	a3,32(a5)
    store_blocking(&pc_buf_base[1], 0);
    6934:	lw	a3,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
    6938:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    693c:	addi	a4,a3,4
    asm volatile(
    6940:	sw	a5,0(a4)
    6944:	lw	a5,0(a4)
    6948:	and	zero,zero,a5
        if (is_opened)
    694c:	lbu	a5,12(sp)
    6950:	beqz	a5,6998 <run_kernel(RuntimeParams const&)+0x3e8>
    return p_reg[0];
    6954:	lui	a5,0xffb12
    6958:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    695c:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6960:	lw	a3,-2008(gp) # ffb00028 <llk_profiler::buffer>
            --open_zone_cnt;
    6964:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6968:	lw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    696c:	lui	a0,0xba3a0
    6970:	slli	a5,a5,0x14
    6974:	srli	a5,a5,0x14
    6978:	or	a5,a5,a0
    697c:	sh2add	a3,a4,a3
    6980:	sw	a5,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6984:	addi	a4,a4,2
            --open_zone_cnt;
    6988:	addi	a5,a2,-1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    698c:	sw	a1,4(a3)
    6990:	sw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
    6994:	sw	a5,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        asm volatile("" ::: "memory");
        if constexpr (is_active_perf_thread<run_type>())
        {
            freeze_and_read_all_counters(zone_id);
    6998:	lw	t0,16(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u;
    699c:	lui	t6,0xffb12
    69a0:	li	a5,2
    69a4:	sw	a5,60(t6) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u;
    69a8:	sw	a5,20(t6)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u;
    69ac:	sw	a5,56(t6)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u;
    69b0:	sw	a5,248(t6)
    std::uint32_t shared_cycles            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
    69b4:	lw	a5,256(t6)
    std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
    69b8:	li	a4,860
    69bc:	mul	t0,t0,a4
    69c0:	lui	a4,0x169
    69c4:	addi	t3,a4,800 # 169320 <__runtime_args_end+0x148f20>
    69c8:	add	a7,t0,t3
    bank_cycles[0]                         = shared_cycles;
    69cc:	sw	a5,0(a7)
    bank_cycles[1]                         = shared_cycles;
    69d0:	sw	a5,4(a7)
    bank_cycles[2]                         = shared_cycles;
    69d4:	sw	a5,8(a7)
    bank_cycles[3]                         = shared_cycles;
    69d8:	sw	a5,12(a7)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    69dc:	lui	t4,0xffb00
    69e0:	lui	t1,0x20
    bank_cycles[4]                         = shared_cycles;
    69e4:	sw	a5,16(a7)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
    69e8:	mv	t2,a4
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    69ec:	mv	t4,t4
    69f0:	addi	t1,t1,-256 # 1ff00 <__loader_init_end+0x14f00>
    std::uint32_t out_idx             = 0;
    69f4:	li	a2,0
        if (bank_id == 3u)
    69f8:	li	t5,3
        std::uint32_t cw = cfg[i];
    69fc:	lw	a5,0(a4)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    6a00:	sh2add	a3,a2,a7
    6a04:	addi	a3,a3,20
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    6a08:	and	a6,a5,t1
        if (!(cw & 0x80000000u))
    6a0c:	bgez	a5,6a4c <run_kernel(RuntimeParams const&)+0x49c>
        std::uint32_t bank_id    = cw & 0xFFu;
    6a10:	zext.b	a0,a5
        ++out_idx;
    6a14:	addi	a2,a2,1
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    6a18:	sh3add	a1,a0,t4
        if (bank_id == 3u)
    6a1c:	bne	a0,t5,6a38 <run_kernel(RuntimeParams const&)+0x488>
            *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
    6a20:	lw	a0,536(t6)
    6a24:	srli	a5,a5,0xd
    6a28:	andi	a5,a5,112
    6a2c:	andi	a0,a0,-113
    6a30:	or	a5,a5,a0
    6a34:	sw	a5,536(t6)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    6a38:	lw	a0,0(a1)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    6a3c:	lw	a5,4(a1)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    6a40:	sw	a6,0(a0) # ba3a0000 <__runtime_args_end+0xba37fc00>
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    6a44:	lw	a5,4(a5)
    6a48:	sw	a5,0(a3)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
    6a4c:	addi	a4,a4,4
    6a50:	bne	a4,t3,69fc <run_kernel(RuntimeParams const&)+0x44c>
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
    6a54:	li	a5,255
    return perf_counters_zone_data_addr(zone) + PERF_COUNTERS_ZONE_DATA_BYTES;
    6a58:	add	t0,t0,t2
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
    6a5c:	sw	a5,1620(t0)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6a60:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6a64:	li	a3,14
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6a68:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6a6c:	zext.b	a4,a4
    6a70:	bltu	a3,a4,6ee0 <run_kernel(RuntimeParams const&)+0x930>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6a74:	sw	zero,40(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6a78:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6a7c:	li	a3,14
    6a80:	zext.b	a4,a4
    6a84:	bltu	a3,a4,6ed4 <run_kernel(RuntimeParams const&)+0x924>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6a88:	sw	zero,40(a5)
    std::uint32_t n = detail::next_zone_id;
    6a8c:	lw	a5,0(s5)
    for (std::uint32_t i = 0; i < n; ++i)
    6a90:	beqz	a5,6e50 <run_kernel(RuntimeParams const&)+0x8a0>
        if (detail::zone_hashes[i] == hash_val)
    6a94:	lui	a4,0xbd77
    6a98:	lw	a3,4(s5)
    6a9c:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    6aa0:	beq	a3,a4,6b24 <run_kernel(RuntimeParams const&)+0x574>
    for (std::uint32_t i = 0; i < n; ++i)
    6aa4:	li	a3,1
    6aa8:	beq	a5,a3,6e50 <run_kernel(RuntimeParams const&)+0x8a0>
        if (detail::zone_hashes[i] == hash_val)
    6aac:	lw	a2,8(s5)
    6ab0:	beq	a2,a4,6e6c <run_kernel(RuntimeParams const&)+0x8bc>
    for (std::uint32_t i = 0; i < n; ++i)
    6ab4:	li	a3,2
    6ab8:	beq	a5,a3,6e50 <run_kernel(RuntimeParams const&)+0x8a0>
        if (detail::zone_hashes[i] == hash_val)
    6abc:	lw	a2,12(s5)
    6ac0:	beq	a2,a4,6e6c <run_kernel(RuntimeParams const&)+0x8bc>
    for (std::uint32_t i = 0; i < n; ++i)
    6ac4:	li	a3,3
    6ac8:	beq	a5,a3,6e50 <run_kernel(RuntimeParams const&)+0x8a0>
        if (detail::zone_hashes[i] == hash_val)
    6acc:	lw	a2,16(s5)
    6ad0:	beq	a2,a4,6e6c <run_kernel(RuntimeParams const&)+0x8bc>
    for (std::uint32_t i = 0; i < n; ++i)
    6ad4:	li	a3,4
    6ad8:	beq	a5,a3,6e50 <run_kernel(RuntimeParams const&)+0x8a0>
        if (detail::zone_hashes[i] == hash_val)
    6adc:	lw	a3,20(s5)
    6ae0:	beq	a3,a4,6f78 <run_kernel(RuntimeParams const&)+0x9c8>
    for (std::uint32_t i = 0; i < n; ++i)
    6ae4:	li	a3,5
    6ae8:	beq	a5,a3,6e50 <run_kernel(RuntimeParams const&)+0x8a0>
        if (detail::zone_hashes[i] == hash_val)
    6aec:	lui	a4,0xbd77
    6af0:	lw	a2,24(s5)
    6af4:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    6af8:	beq	a2,a4,6e6c <run_kernel(RuntimeParams const&)+0x8bc>
    for (std::uint32_t i = 0; i < n; ++i)
    6afc:	li	a3,6
    6b00:	beq	a5,a3,6e50 <run_kernel(RuntimeParams const&)+0x8a0>
        if (detail::zone_hashes[i] == hash_val)
    6b04:	lw	a2,28(s5)
    6b08:	beq	a2,a4,6e6c <run_kernel(RuntimeParams const&)+0x8bc>
    for (std::uint32_t i = 0; i < n; ++i)
    6b0c:	li	a3,7
    6b10:	beq	a5,a3,6e50 <run_kernel(RuntimeParams const&)+0x8a0>
        if (detail::zone_hashes[i] == hash_val)
    6b14:	lw	a2,32(s5)
    6b18:	beq	a2,a4,6e6c <run_kernel(RuntimeParams const&)+0x8bc>
    if (n < PERF_COUNTERS_MAX_ZONES)
    6b1c:	li	a4,8
    6b20:	bne	a5,a4,6e50 <run_kernel(RuntimeParams const&)+0x8a0>
    return 0;
    6b24:	li	a5,0
    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
    6b28:	sw	a5,28(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 1u; // PERF_CNT_ALL (INSTRN+FPU)
    6b2c:	lui	a5,0xffb12
    6b30:	li	a4,1
    6b34:	sw	a4,60(a5) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 1u; // TDMA_UNPACK
    6b38:	sw	a4,20(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 1u; // L1
    6b3c:	sw	a4,56(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 1u; // TDMA_PACK
    6b40:	sw	a4,248(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6b44:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6b48:	li	a3,14
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6b4c:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6b50:	zext.b	a4,a4
    6b54:	bltu	a3,a4,6ec8 <run_kernel(RuntimeParams const&)+0x918>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6b58:	sw	zero,32(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6b5c:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6b60:	li	a3,14
    6b64:	zext.b	a4,a4
    6b68:	bltu	a3,a4,6ebc <run_kernel(RuntimeParams const&)+0x90c>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6b6c:	sw	zero,32(a5)
    {
    6b70:	sb	zero,20(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    6b74:	lw	a5,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    6b78:	lw	a3,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    6b7c:	li	a2,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    6b80:	add	a4,a3,a5
    6b84:	addi	a4,a4,-1021
        if (!is_buffer_full())
    6b88:	bgeu	a2,a4,6bd0 <run_kernel(RuntimeParams const&)+0x620>
    return p_reg[0];
    6b8c:	lui	a4,0xffb12
    6b90:	lw	a1,496(a4) # ffb121f0 <__stack_top+0x111f0>
    6b94:	lw	a4,504(a4)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6b98:	lw	a2,-2008(gp) # ffb00028 <llk_profiler::buffer>
            is_opened = true;
    6b9c:	li	a0,1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6ba0:	lui	a6,0xa99a8
            is_opened = true;
    6ba4:	sb	a0,20(sp)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6ba8:	slli	a4,a4,0x14
    6bac:	srli	a4,a4,0x14
    6bb0:	or	a4,a4,a6
            ++open_zone_cnt;
    6bb4:	add	a3,a3,a0
    6bb8:	sh2add	a2,a5,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6bbc:	sw	a4,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6bc0:	addi	a5,a5,2
            ++open_zone_cnt;
    6bc4:	sw	a3,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6bc8:	sw	a1,4(a2)
    6bcc:	sw	a5,-1972(gp) # ffb0004c <llk_profiler::write_idx>
constexpr std::uint32_t PERF_OUTPUT  = PERF_INPUT_C + 16 * 4096;

constexpr std::uint32_t PERF_ADDRESS(std::uint32_t buffer, std::uint32_t tile)
{
    std::uint32_t address = buffer + (tile % 16) * 4096; // Loop every 16 tiles, to prevent escaping memory
    return address / 16 - 1;                             // Correct the L1 Address for Tensix
    6bd0:	lui	t3,0x2
    6bd4:	lui	t1,0x3
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6bd8:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
inline void _llk_unpack_configure_addresses_(const std::uint32_t address_a, const std::uint32_t address_b, volatile std::uint32_t tt_reg_ptr *cfg)
{
    LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

    if (0 == unp_cfg_context)
    6bdc:	lw	a1,-1948(gp) # ffb00064 <unp_cfg_context>
    6be0:	addi	t3,t3,255 # 20ff <BRISC_LOCAL_MEM_LENGTH+0xff>
    6be4:	addi	t1,t1,255 # 30ff <BRISC_LOCAL_MEM_LENGTH+0x10ff>
    6be8:	li	a2,0
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    6bec:	lui	t0,0xffef0
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6bf0:	li	t6,14
    unp_cfg_context = 1 - unp_cfg_context;
    6bf4:	li	a6,1
            _perf_unpack_loop_set_valid<true, true>(TILE_CNT * TILE_NUM_FACES);
            return;
        }
        else
        {
            for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
    6bf8:	lui	t4,0x1
    6bfc:	lui	t5,0x10
    6c00:	srli	a5,a2,0x4
    6c04:	add	a7,a5,t3
    6c08:	add	a3,a5,t1
 */

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(const std::uint32_t address_a, const std::uint32_t address_b)
{
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111); // reset counters
    6c0c:	ttsetadczw	3,0,0,0,0,15
    if (cfg_state_id == 0)
    6c10:	lw	a5,-1960(gp) # ffb00058 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    6c14:	lui	a0,0xffef0
    if (cfg_state_id == 0)
    6c18:	beqz	a5,6c20 <run_kernel(RuntimeParams const&)+0x670>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
    6c1c:	addi	a0,t0,896 # ffef0380 <__instrn_buffer+0xb0380>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6c20:	lw	a5,52(a4)
    6c24:	andi	a5,a5,254
    while (semaphore_read(semaphore::UNPACK_SYNC) >= num_contexts)
    6c28:	bnez	a5,6c20 <run_kernel(RuntimeParams const&)+0x670>
    6c2c:	bnez	a1,6e08 <run_kernel(RuntimeParams const&)+0x858>
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
    6c30:	sw	a7,304(a0) # ffef0130 <__instrn_buffer+0xb0130>
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    6c34:	sw	a3,496(a0)
    6c38:	lw	a3,52(a4)
    6c3c:	mv	a0,a1
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6c40:	zext.b	a3,a3
    6c44:	bltu	t6,a3,6e20 <run_kernel(RuntimeParams const&)+0x870>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6c48:	sw	zero,52(a4)

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    6c4c:	ttstallwait	8,1024
    TTI_MOP(1, 0, 0); // run the double-loop template
    6c50:	ttmop	1,0,0
    TTI_SEMGET(semaphore::t6_sem(index));
    6c54:	ttsemget	32
    unp_cfg_context = 1 - unp_cfg_context;
    6c58:	sub	a1,a6,a0
    6c5c:	sw	a1,-1948(gp) # ffb00064 <unp_cfg_context>
    if (unp_cfg_context == 0)
    6c60:	beq	a0,a6,6e48 <run_kernel(RuntimeParams const&)+0x898>
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0101);
    6c64:	ttsetc16	41,257
    6c68:	add	a2,a2,t4
    6c6c:	bne	a2,t5,6c00 <run_kernel(RuntimeParams const&)+0x650>
    asm volatile(
    6c70:	li	a3,0
    store_blocking(&pc_buf_base[1], 0);
    6c74:	addi	a4,a4,4
    asm volatile(
    6c78:	sw	a3,0(a4)
    6c7c:	lw	a3,0(a4)
    6c80:	and	zero,zero,a3
        if (is_opened)
    6c84:	lbu	a4,20(sp)
    6c88:	beqz	a4,6cd0 <run_kernel(RuntimeParams const&)+0x720>
    return p_reg[0];
    6c8c:	lui	a4,0xffb12
    6c90:	lw	a0,496(a4) # ffb121f0 <__stack_top+0x111f0>
    6c94:	lw	a4,504(a4)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6c98:	addi	t4,t4,-1 # fff <__firmware_stack_size+0xdff>
    6c9c:	lw	a2,-2008(gp) # ffb00028 <llk_profiler::buffer>
            --open_zone_cnt;
    6ca0:	lw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6ca4:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    6ca8:	lui	a6,0xb99a8
    6cac:	and	a4,a4,t4
    6cb0:	or	a4,a4,a6
    6cb4:	sh2add	a2,a3,a2
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6cb8:	addi	a3,a3,2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6cbc:	sw	a4,0(a2)
            --open_zone_cnt;
    6cc0:	addi	a4,a1,-1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6cc4:	sw	a0,4(a2)
    6cc8:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
    6ccc:	sw	a4,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
            freeze_and_read_all_counters(zone_id);
    6cd0:	lw	t0,28(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u;
    6cd4:	lui	t6,0xffb12
    6cd8:	li	a4,2
    6cdc:	sw	a4,60(t6) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u;
    6ce0:	sw	a4,20(t6)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u;
    6ce4:	sw	a4,56(t6)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u;
    6ce8:	sw	a4,248(t6)
    std::uint32_t shared_cycles            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
    6cec:	lw	a4,256(t6)
    std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
    6cf0:	li	a3,860
    6cf4:	mul	t0,t0,a3
    6cf8:	lui	a3,0x169
    6cfc:	addi	t3,a3,800 # 169320 <__runtime_args_end+0x148f20>
    6d00:	add	a7,t0,t3
    bank_cycles[0]                         = shared_cycles;
    6d04:	sw	a4,0(a7)
    bank_cycles[1]                         = shared_cycles;
    6d08:	sw	a4,4(a7)
    bank_cycles[2]                         = shared_cycles;
    6d0c:	sw	a4,8(a7)
    bank_cycles[3]                         = shared_cycles;
    6d10:	sw	a4,12(a7)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    6d14:	lui	t4,0xffb00
    6d18:	lui	t1,0x20
    bank_cycles[4]                         = shared_cycles;
    6d1c:	sw	a4,16(a7)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
    6d20:	mv	t2,a3
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    6d24:	mv	t4,t4
    6d28:	addi	t1,t1,-256 # 1ff00 <__loader_init_end+0x14f00>
        if (bank_id == 3u)
    6d2c:	li	t5,3
        std::uint32_t cw = cfg[i];
    6d30:	lw	a4,0(a3)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    6d34:	sh2add	a2,a5,a7
    6d38:	addi	a2,a2,20
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    6d3c:	and	a6,a4,t1
        if (!(cw & 0x80000000u))
    6d40:	bgez	a4,6d80 <run_kernel(RuntimeParams const&)+0x7d0>
        std::uint32_t bank_id    = cw & 0xFFu;
    6d44:	zext.b	a0,a4
        ++out_idx;
    6d48:	addi	a5,a5,1
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    6d4c:	sh3add	a1,a0,t4
        if (bank_id == 3u)
    6d50:	bne	a0,t5,6d6c <run_kernel(RuntimeParams const&)+0x7bc>
            *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
    6d54:	lw	a0,536(t6)
    6d58:	srli	a4,a4,0xd
    6d5c:	andi	a4,a4,112
    6d60:	andi	a0,a0,-113
    6d64:	or	a4,a4,a0
    6d68:	sw	a4,536(t6)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    6d6c:	lw	a0,0(a1)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    6d70:	lw	a4,4(a1)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    6d74:	sw	a6,0(a0)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    6d78:	lw	a4,4(a4)
    6d7c:	sw	a4,0(a2)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
    6d80:	addi	a3,a3,4
    6d84:	bne	a3,t3,6d30 <run_kernel(RuntimeParams const&)+0x780>
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
    6d88:	li	a5,255
    return perf_counters_zone_data_addr(zone) + PERF_COUNTERS_ZONE_DATA_BYTES;
    6d8c:	add	t0,t0,t2
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
    6d90:	sw	a5,1620(t0)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6d94:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6d98:	li	a3,14
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6d9c:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6da0:	zext.b	a4,a4
    6da4:	bltu	a3,a4,6eb0 <run_kernel(RuntimeParams const&)+0x900>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6da8:	sw	zero,40(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6dac:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6db0:	li	a3,14
    6db4:	zext.b	a4,a4
    6db8:	bltu	a3,a4,6ea4 <run_kernel(RuntimeParams const&)+0x8f4>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6dbc:	sw	zero,40(a5)
                _llk_unpack_AB_<>(PERF_ADDRESS(PERF_INPUT_A, tile), PERF_ADDRESS(PERF_INPUT_B, tile));
            }
        }
        PROFILER_SYNC();
    }
}
    6dc0:	lw	ra,76(sp)
    6dc4:	lw	s0,72(sp)
    6dc8:	lw	s1,68(sp)
    6dcc:	lw	s2,64(sp)
    6dd0:	lw	s3,60(sp)
    6dd4:	lw	s4,56(sp)
    6dd8:	lw	s5,52(sp)
    6ddc:	lw	s6,48(sp)
    6de0:	lw	s7,44(sp)
    6de4:	addi	sp,sp,80
    6de8:	ret
        detail::zone_hashes[n] = hash_val;
    6dec:	lui	a3,0x7c867
    6df0:	sh2add	a4,a5,s5
    6df4:	addi	a3,a3,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
        detail::next_zone_id   = n + 1;
    6df8:	addi	a2,a5,1
        detail::zone_hashes[n] = hash_val;
    6dfc:	sw	a3,4(a4)
        detail::next_zone_id   = n + 1;
    6e00:	sw	a2,0(s5)
        return n;
    6e04:	j	6678 <run_kernel(RuntimeParams const&)+0xc8>
    }
    else
    {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
    6e08:	sw	a7,308(a0)
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    6e0c:	sw	a3,500(a0)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6e10:	lw	a3,52(a4)
    6e14:	mv	a0,a1
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6e18:	zext.b	a3,a3
    6e1c:	bgeu	t6,a3,6c48 <run_kernel(RuntimeParams const&)+0x698>
    6e20:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6e24:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    unp_cfg_context = 1 - unp_cfg_context;
    6e28:	lw	a0,-1948(gp) # ffb00064 <unp_cfg_context>
    6e2c:	sw	zero,52(a4)
    6e30:	ttstallwait	8,1024
    6e34:	ttmop	1,0,0
    TTI_SEMGET(semaphore::t6_sem(index));
    6e38:	ttsemget	32
    6e3c:	sub	a1,a6,a0
    6e40:	sw	a1,-1948(gp) # ffb00064 <unp_cfg_context>
    if (unp_cfg_context == 0)
    6e44:	bne	a0,a6,6c64 <run_kernel(RuntimeParams const&)+0x6b4>
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    6e48:	ttsetc16	41,0
    6e4c:	j	6c68 <run_kernel(RuntimeParams const&)+0x6b8>
        detail::zone_hashes[n] = hash_val;
    6e50:	lui	a4,0xbd77
    6e54:	sh2add	a2,a5,s5
    6e58:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
        detail::next_zone_id   = n + 1;
    6e5c:	addi	a3,a5,1
        detail::zone_hashes[n] = hash_val;
    6e60:	sw	a4,4(a2)
        detail::next_zone_id   = n + 1;
    6e64:	sw	a3,0(s5)
        return n;
    6e68:	j	6b28 <run_kernel(RuntimeParams const&)+0x578>
    for (std::uint32_t i = 0; i < n; ++i)
    6e6c:	mv	a5,a3
    6e70:	j	6b28 <run_kernel(RuntimeParams const&)+0x578>
    6e74:	mv	a5,a3
    6e78:	j	6678 <run_kernel(RuntimeParams const&)+0xc8>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6e7c:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6e80:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6e84:	li	a3,14
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6e88:	sw	zero,32(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6e8c:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6e90:	zext.b	a4,a4
    6e94:	bgeu	a3,a4,66bc <run_kernel(RuntimeParams const&)+0x10c>
    6e98:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6e9c:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    6ea0:	j	66bc <run_kernel(RuntimeParams const&)+0x10c>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6ea4:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6ea8:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    6eac:	j	6dbc <run_kernel(RuntimeParams const&)+0x80c>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6eb0:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6eb4:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    6eb8:	j	6da8 <run_kernel(RuntimeParams const&)+0x7f8>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6ebc:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6ec0:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    6ec4:	j	6b6c <run_kernel(RuntimeParams const&)+0x5bc>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6ec8:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6ecc:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    6ed0:	j	6b58 <run_kernel(RuntimeParams const&)+0x5a8>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6ed4:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6ed8:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    6edc:	j	6a88 <run_kernel(RuntimeParams const&)+0x4d8>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6ee0:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6ee4:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    6ee8:	j	6a74 <run_kernel(RuntimeParams const&)+0x4c4>
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    6eec:	ebreak
    6ef0:	j	68e0 <run_kernel(RuntimeParams const&)+0x330>
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    6ef4:	ebreak
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    6ef8:	lui	a4,0xb4010
    6efc:	addi	a4,a4,72 # b4010048 <__runtime_args_end+0xb3fefc48>
    config_unpacker_x_end<p_setadc::UNP_AB>(tensor_shape.face_r_dim);
    6f00:	lbu	a5,20(sp)
    6f04:	sw	a4,0(s6)
    LLK_ASSERT(
    6f08:	li	a4,16
    6f0c:	bgeu	a4,a5,6f24 <run_kernel(RuntimeParams const&)+0x974>
    6f10:	ebreak
    6f14:	j	68bc <run_kernel(RuntimeParams const&)+0x30c>
    LLK_ASSERT(
    6f18:	ebreak
    LLK_ASSERT(
    6f1c:	ebreak
    6f20:	j	6734 <run_kernel(RuntimeParams const&)+0x184>
    LLK_ASSERT(
    6f24:	lui	a4,0xffff0
    6f28:	addi	a4,a4,-279 # fffefee9 <__instrn_buffer+0x1afee9>
    6f2c:	sra	a4,a4,a5
    6f30:	andi	a4,a4,1
    6f34:	beqz	a4,6f3c <run_kernel(RuntimeParams const&)+0x98c>
    6f38:	ebreak
    switch (face_r_dim)
    6f3c:	li	a4,4
    6f40:	beq	a5,a4,6f60 <run_kernel(RuntimeParams const&)+0x9b0>
    6f44:	bltu	a4,a5,6f80 <run_kernel(RuntimeParams const&)+0x9d0>
    6f48:	li	a4,1
    6f4c:	beq	a5,a4,6f68 <run_kernel(RuntimeParams const&)+0x9b8>
    6f50:	li	a4,2
    6f54:	bne	a5,a4,68bc <run_kernel(RuntimeParams const&)+0x30c>
            TTI_SETADCXX(UNP_SEL, 2 * FACE_C_DIM - 1, 0x0);
    6f58:	ttsetadcxx	3,31,0
            break;
    6f5c:	j	68c0 <run_kernel(RuntimeParams const&)+0x310>
            TTI_SETADCXX(UNP_SEL, 4 * FACE_C_DIM - 1, 0x0);
    6f60:	ttsetadcxx	3,63,0
            break;
    6f64:	j	68c0 <run_kernel(RuntimeParams const&)+0x310>
            TTI_SETADCXX(UNP_SEL, 1 * FACE_C_DIM - 1, 0x0);
    6f68:	ttsetadcxx	3,15,0
            break;
    6f6c:	j	68c0 <run_kernel(RuntimeParams const&)+0x310>
    6f70:	li	a5,4
    6f74:	j	6678 <run_kernel(RuntimeParams const&)+0xc8>
    6f78:	li	a5,4
    6f7c:	j	6b28 <run_kernel(RuntimeParams const&)+0x578>
    switch (face_r_dim)
    6f80:	li	a4,8
    6f84:	bne	a5,a4,68bc <run_kernel(RuntimeParams const&)+0x30c>
            TTI_SETADCXX(UNP_SEL, 8 * FACE_C_DIM - 1, 0x0);
    6f88:	ttsetadcxx	3,127,0
            break;
    6f8c:	j	68c0 <run_kernel(RuntimeParams const&)+0x310>

00006f90 <_init()>:
    }
}

void _init(void)
{
}
    6f90:	ret

00006f94 <_fini()>:

void _fini(void)
    6f94:	ret

00006f98 <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
    6f98:	lui	a5,0x20
    6f9c:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
    6fa0:	sb	a5,0(a0)
        (void)(dstc[i]);
    6fa4:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
    6fa8:	fence
}
    6fac:	ret

00006fb0 <memset>:
    6fb0:	li	t1,15
    6fb4:	mv	a4,a0
    6fb8:	bgeu	t1,a2,6ff4 <memset+0x44>
    6fbc:	andi	a5,a4,15
    6fc0:	bnez	a5,7060 <memset+0xb0>
    6fc4:	bnez	a1,7048 <memset+0x98>
    6fc8:	andi	a3,a2,-16
    6fcc:	andi	a2,a2,15
    6fd0:	add	a3,a3,a4
    6fd4:	sw	a1,0(a4)
    6fd8:	sw	a1,4(a4)
    6fdc:	sw	a1,8(a4)
    6fe0:	sw	a1,12(a4)
    6fe4:	addi	a4,a4,16
    6fe8:	bltu	a4,a3,6fd4 <memset+0x24>
    6fec:	bnez	a2,6ff4 <memset+0x44>
    6ff0:	ret
    6ff4:	sub	a3,t1,a2
    6ff8:	slli	a3,a3,0x2
    6ffc:	auipc	t0,0x0
    7000:	add	a3,a3,t0
    7004:	jr	12(a3)
    7008:	sb	a1,14(a4)
    700c:	sb	a1,13(a4)
    7010:	sb	a1,12(a4)
    7014:	sb	a1,11(a4)
    7018:	sb	a1,10(a4)
    701c:	sb	a1,9(a4)
    7020:	sb	a1,8(a4)
    7024:	sb	a1,7(a4)
    7028:	sb	a1,6(a4)
    702c:	sb	a1,5(a4)
    7030:	sb	a1,4(a4)
    7034:	sb	a1,3(a4)
    7038:	sb	a1,2(a4)
    703c:	sb	a1,1(a4)
    7040:	sb	a1,0(a4)
    7044:	ret
    7048:	zext.b	a1,a1
    704c:	slli	a3,a1,0x8
    7050:	or	a1,a1,a3
    7054:	slli	a3,a1,0x10
    7058:	or	a1,a1,a3
    705c:	j	6fc8 <memset+0x18>
    7060:	slli	a3,a5,0x2
    7064:	auipc	t0,0x0
    7068:	add	a3,a3,t0
    706c:	mv	t0,ra
    7070:	jalr	-96(a3)
    7074:	mv	ra,t0
    7078:	addi	a5,a5,-16
    707c:	sub	a4,a4,a5
    7080:	add	a2,a2,a5
    7084:	bgeu	t1,a2,6ff4 <memset+0x44>
    7088:	j	6fc4 <memset+0x14>
