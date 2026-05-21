
/tmp/perf_overhead_artifacts/wc/L1_TO_L1/unpack.elf:     file format elf32-littleriscv


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
    6018:	addi	a5,a5,32 # ffb00020 <llk_profiler::open_zone_cnt>
    601c:	addi	a4,a4,100 # ffb00064 <__gcov_info_end>
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
    6064:	addi	a3,a3,32 # ffb00020 <llk_profiler::open_zone_cnt>
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
    6088:	sw	a1,0(a4) # ffb00000 <llk_profiler::buffer>
    608c:	addi	a4,a4,4
        while (dst < end)
    6090:	bne	a5,a3,6080 <_start+0x80>
        }
    }

    // Execute global constructors
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
    6094:	lui	s0,0xffb00
    6098:	lui	s1,0xffb00
    609c:	mv	s0,s0
    60a0:	mv	s1,s1
    60a4:	bgeu	s0,s1,60b8 <_start+0xb8>
    {
        (*temp_constructor)();
    60a8:	lw	a5,0(s0) # ffb00000 <llk_profiler::buffer>
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
    60ec:	lw	a5,-2040(gp) # ffb00008 <ckernel::regfile>
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
    6108:	lui	s3,0xffb00
    610c:	sw	a5,4(s3) # ffb00004 <llk_profiler::barrier_ptr>
    buffer        = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
    6110:	lui	s2,0xffb00
    TTI_NOP;
}

inline void reset_cfg_state_id()
{
    cfg_state_id = 0;
    6114:	sw	zero,-2000(gp) # ffb00030 <ckernel::cfg_state_id>
    write_idx     = 0;
    open_zone_cnt = 0;

    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
    6118:	lui	a2,0x1
    611c:	li	a1,0
    buffer        = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
    6120:	sw	a0,0(s2) # ffb00000 <llk_profiler::buffer>
}

inline void reset_dest_offset_id()
{
    dest_offset_id = 0;
    6124:	sw	zero,-2004(gp) # ffb0002c <ckernel::dest_offset_id>
    write_idx     = 0;
    6128:	sw	zero,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    open_zone_cnt = 0;
    612c:	sw	zero,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
    6130:	jal	6dc4 <memset>
    auto& barrier = *barrier_ptr;
    6134:	lw	a2,4(s3)
    barrier[TRISC_ID] = 1;
    6138:	li	a5,1
    613c:	sw	a5,0(a2) # 1000 <TRISC_LOCAL_MEM_LENGTH>
    asm volatile("fence" ::: "memory");
    6140:	fence
        while (barrier[i] != 1)
    6144:	lw	a1,4(a2)
    6148:	beq	a1,a5,6164 <main+0xa4>
    614c:	mv	a1,a5
    6150:	addi	a4,a2,4
    6154:	li	a3,1
            asm volatile("fence" ::: "memory");
    6158:	fence
        while (barrier[i] != 1)
    615c:	lw	a5,0(a4)
    6160:	bne	a5,a3,6158 <main+0x98>
    for (std::uint32_t i = 0; i < NUM_CORES; ++i)
    6164:	li	a5,2
    6168:	beq	a1,a5,6184 <main+0xc4>
        while (barrier[i] != 1)
    616c:	lw	a3,8(a2)
    6170:	li	a4,1
    6174:	beq	a3,a4,6184 <main+0xc4>
    6178:	mv	a1,a5
    617c:	addi	a4,a2,8
    6180:	j	6154 <main+0x94>
    zone_scoped(zone_scoped&&)                 = delete;
    zone_scoped& operator=(const zone_scoped&) = delete;
    zone_scoped& operator=(zone_scoped&&)      = delete;

    inline __attribute__((always_inline)) zone_scoped()
    {
    6184:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    6188:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    618c:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        asm volatile("" ::: "memory");
        if (!is_buffer_full())
    6190:	li	a2,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    6194:	add	a5,a4,a3
    6198:	addi	a5,a5,-1021
        if (!is_buffer_full())
    619c:	bgeu	a2,a5,61e4 <main+0x124>
// now handled by the compiler)
// workaround is needed only for GS
inline std::uint32_t reg_read(std::uint32_t addr)
{
    volatile std::uint32_t tt_reg_ptr *p_reg = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(addr);
    return p_reg[0];
    61a0:	lui	a5,0xffb12
    61a4:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    61a8:	lw	a5,504(a5)
    61ac:	lw	a2,0(s2)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    61b0:	lui	a0,0xa5104
        {
            is_opened = true;
            write_entry(EntryType::ZONE_START, id16);
            ++open_zone_cnt;
    61b4:	addi	a3,a3,1
    61b8:	sh2add	a2,a4,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    61bc:	slli	a5,a5,0x14
    61c0:	srli	a5,a5,0x14
    61c4:	or	a5,a5,a0
    61c8:	sw	a5,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    61cc:	addi	a4,a4,2
            is_opened = true;
    61d0:	li	a5,1
            ++open_zone_cnt;
    61d4:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    61d8:	sw	a1,4(a2)
    61dc:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            is_opened = true;
    61e0:	sb	a5,12(sp)
        run_kernel(temp_args);
    61e4:	addi	a0,sp,8
    61e8:	jal	65b8 <run_kernel(RuntimeParams const&)>
    store_blocking(&pc_buf_base[1], 0);
    61ec:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
    61f0:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    61f4:	addi	a4,a4,4
    asm volatile(
    61f8:	sw	a5,0(a4)
    61fc:	lw	a5,0(a4)
    6200:	and	zero,zero,a5
    }

    ~zone_scoped()
    {
        asm volatile("" ::: "memory");
        if (is_opened)
    6204:	lbu	a5,12(sp)
    6208:	beqz	a5,6250 <main+0x190>
    return p_reg[0];
    620c:	lui	a5,0xffb12
    6210:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    6214:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6218:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
        {
            write_entry(EntryType::ZONE_END, id16);
            --open_zone_cnt;
    621c:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    6220:	lw	a3,0(s2)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6224:	lui	a0,0xb5104
    6228:	slli	a5,a5,0x14
    622c:	srli	a5,a5,0x14
    6230:	or	a5,a5,a0
    6234:	sh2add	a3,a4,a3
    6238:	sw	a5,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    623c:	addi	a4,a4,2
            --open_zone_cnt;
    6240:	addi	a5,a2,-1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6244:	sw	a1,4(a3)
    6248:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    624c:	sw	a5,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    *mailbox = ckernel::KERNEL_COMPLETE;
    6250:	lui	a5,0x20
}
    6254:	lw	ra,44(sp)
    6258:	lw	s0,40(sp)
    *mailbox = ckernel::KERNEL_COMPLETE;
    625c:	li	a4,255
    6260:	sw	a4,-72(a5) # 1ffb8 <__loader_init_end+0x14fb8>
}
    6264:	lw	s1,36(sp)
    6268:	lw	s2,32(sp)
    626c:	lw	s3,28(sp)
    6270:	li	a0,0
    6274:	addi	sp,sp,48
    6278:	ret

0000627c <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>:
    }

    /// @brief Get total number of faces
    constexpr std::uint8_t total_num_faces() const
    {
        return num_faces_r_dim * num_faces_c_dim;
    627c:	lbu	a5,3(a0) # b5104003 <__runtime_args_end+0xb50e3c03>
    6280:	lbu	a4,2(a0)
 *
 * @param tensor_shape: Tensor shape to validate
 * @return true if tensor shape is valid, false otherwise
 **/
__attribute__((noinline)) bool validate_tensor_shape_tile_dependent_ops_(const TensorShape &tensor_shape)
{
    6284:	mv	a3,a0
        return num_faces_r_dim * num_faces_c_dim;
    6288:	mul	a4,a4,a5
    628c:	zext.b	a4,a4
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t face_c_dim = tensor_shape.face_c_dim;
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
    6290:	addi	a5,a4,-1
    6294:	addi	a4,a4,-4
    6298:	sltiu	a5,a5,2
    629c:	seqz	a4,a4
    62a0:	or	a0,a5,a4
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
    62a4:	beqz	a0,62d8 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    62a8:	lbu	a4,0(a3)
    62ac:	li	a5,16
    62b0:	li	a0,0
    62b4:	bltu	a5,a4,62d8 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
    62b8:	lui	a5,0x10
    62bc:	addi	a5,a5,278 # 10116 <__loader_init_end+0x5116>
    62c0:	srl	a5,a5,a4
    62c4:	andi	a0,a5,1
    62c8:	beqz	a0,62d8 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
    62cc:	lbu	a0,1(a3)
    62d0:	addi	a0,a0,-16
    62d4:	seqz	a0,a0
}
    62d8:	ret

000062dc <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)>:
 * \return true if the conversion is supported given the FP32 accumulation setting.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_unpacker_format_conversion_supported_fp32_acc(
    const DataFormat unpack_src_format, const DataFormat unpack_dst_format, const bool is_fp32_dest_acc_en)
{
    switch (unpack_src_format)
    62dc:	li	a4,9
{
    62e0:	mv	a5,a0
    switch (unpack_src_format)
    62e4:	beq	a0,a4,6404 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x128>
    62e8:	bltu	a4,a0,6324 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x48>
    62ec:	li	a4,4
    62f0:	beq	a0,a4,63cc <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xf0>
    62f4:	bgeu	a4,a0,6378 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x9c>
        //    ISA conversions:
        //      SrcA/SrcB path: NOT possible (ISA doc explicitly states "Not possible").
        //      Dst path:       INT32 → Integer "32" (Int32): valid (32b data movement to Dst).
        //    Hence, only valid when unpack_to_dest = true (checked in _dest).
        case DataFormat::Int32:
            return unpack_dst_format == DataFormat::Int32;
    62f8:	addi	a0,a1,-8
    switch (unpack_src_format)
    62fc:	li	a3,8
            return unpack_dst_format == DataFormat::Int32;
    6300:	seqz	a0,a0
    switch (unpack_src_format)
    6304:	beq	a5,a3,6374 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    6308:	li	a3,5
    630c:	bne	a5,a3,6410 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x134>
                    return is_fp32_dest_acc_en;
    6310:	mv	a0,a2
            switch (unpack_dst_format)
    6314:	beq	a1,a4,6374 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    6318:	addi	a1,a1,-5
    631c:	seqz	a0,a1
    6320:	ret
    switch (unpack_src_format)
    6324:	li	a4,15
    6328:	beq	a0,a4,6410 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x134>
    632c:	bgeu	a4,a0,639c <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xc0>
    6330:	li	a4,26
    6334:	beq	a0,a4,6384 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xa8>
    6338:	li	a4,30
    633c:	beq	a0,a4,6430 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x154>
    6340:	addi	a5,a0,-24
        // 8. UInt32 (opaque 32-bit) in L1.
        //
        //    Not explicitly listed in the ISA doc. Treated as opaque 32-bit data analogous to
        //    Int32: only valid when targeting the Dst register (unpack_to_dest = true, checked in _dest).
        case DataFormat::UInt32:
            return unpack_dst_format == DataFormat::UInt32;
    6344:	addi	a1,a1,-24
    6348:	seqz	a1,a1
    switch (unpack_src_format)
    634c:	seqz	a5,a5
    6350:	and	a0,a1,a5
    6354:	ret
        //         INT8 → Integer "8" (Int8): always valid.
        //       Dst path:
        //         INT8 → BF16  (Float16_b): always valid (BFP8+force_shared_exp).
        //         INT8 → Integer "8" (Int8): always valid.
        case DataFormat::Int8:
            switch (unpack_dst_format)
    6358:	li	a4,5
                    return true;
    635c:	li	a0,1
            switch (unpack_dst_format)
    6360:	beq	a1,a4,6374 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    6364:	beq	a1,a5,6374 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    6368:	addi	a1,a1,-4
    636c:	seqz	a1,a1
    6370:	and	a0,a2,a1
        // -------------------------------------------------------------------------
        // 12. Unknown or not-yet-handled formats.
        default:
            return false;
    }
}
    6374:	ret
    switch (unpack_src_format)
    6378:	beqz	a0,63e8 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x10c>
    637c:	li	a4,1
    6380:	bne	a0,a4,63b4 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xd8>
            return unpack_dst_format == DataFormat::Float16 || unpack_src_format == unpack_dst_format;
    6384:	sub	a5,a5,a1
    6388:	addi	a1,a1,-1
    638c:	seqz	a5,a5
    6390:	seqz	a1,a1
    6394:	or	a0,a1,a5
    6398:	ret
    switch (unpack_src_format)
    639c:	li	a4,14
    63a0:	beq	a0,a4,6358 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x7c>
    63a4:	li	a4,10
    63a8:	beq	a0,a4,6384 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xa8>
    63ac:	li	a4,11
    63b0:	bne	a0,a4,6440 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x164>
            switch (unpack_dst_format)
    63b4:	li	a4,1
                    return true;
    63b8:	mv	a0,a1
            switch (unpack_dst_format)
    63bc:	beq	a1,a4,6374 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
                    return unpack_src_format == unpack_dst_format;
    63c0:	sub	a5,a5,a1
    63c4:	seqz	a0,a5
    63c8:	ret
                    return is_fp32_dest_acc_en;
    63cc:	mv	a0,a2
            switch (unpack_dst_format)
    63d0:	beq	a1,a5,6374 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    63d4:	bltu	a5,a1,6318 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x3c>
    63d8:	beqz	a1,6374 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    63dc:	addi	a1,a1,-1
    63e0:	seqz	a0,a1
    63e4:	ret
                    return is_fp32_dest_acc_en;
    63e8:	mv	a0,a2
            switch (unpack_dst_format)
    63ec:	beq	a1,a4,6374 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    63f0:	addi	a0,a1,-5
    63f4:	seqz	a0,a0
    63f8:	bltu	a4,a1,6374 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    63fc:	sltiu	a0,a1,2
    6400:	ret
            return unpack_dst_format == DataFormat::UInt16;
    6404:	addi	a1,a1,-9
    6408:	seqz	a0,a1
    640c:	ret
            switch (unpack_dst_format)
    6410:	li	a4,4
                    return is_fp32_dest_acc_en;
    6414:	mv	a0,a2
            switch (unpack_dst_format)
    6418:	beq	a1,a4,6374 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0x98>
    641c:	addi	a4,a1,-5
    6420:	zext.b	a4,a4
    6424:	li	a0,1
    6428:	bltu	a0,a4,63c0 <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)+0xe4>
    642c:	ret
            return unpack_dst_format == DataFormat::Int8 || unpack_dst_format == DataFormat::UInt8;
    6430:	andi	a1,a1,239
    6434:	addi	a1,a1,-14
    6438:	seqz	a0,a1
    643c:	ret
    switch (unpack_src_format)
    6440:	li	a0,0
    6444:	ret

00006448 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)>:
 * \return true if the conversion is supported given the register destination.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_unpacker_format_conversion_supported_dest(
    const DataFormat unpack_src_format, const DataFormat unpack_dst_format, const bool unpack_to_dest)
{
    switch (unpack_src_format)
    6448:	li	a4,9
{
    644c:	mv	a5,a0
    switch (unpack_src_format)
    6450:	beq	a0,a4,65a8 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x160>
    6454:	bltu	a4,a0,6484 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x3c>
    6458:	li	a4,4
    645c:	beq	a0,a4,654c <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x104>
    6460:	bgeu	a4,a0,6528 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xe0>
    6464:	li	a3,8
    6468:	beq	a0,a3,64c8 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x80>
    646c:	li	a3,5
    6470:	bne	a0,a3,6588 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x140>
        //      Dst path:
        //        BF16 → BF16  (Float16_b): always valid (identity).
        //    Note: FP16 is NOT a valid output for BF16 input — no cross-exponent-width conversion
        //    from 8-bit exponent BF16 to 5-bit exponent FP16 is supported by the unpacker.
        case DataFormat::Float16_b:
            switch (unpack_dst_format)
    6474:	beq	a1,a4,64f8 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xb0>
    6478:	addi	a1,a1,-5
    647c:	seqz	a0,a1
    6480:	ret
    switch (unpack_src_format)
    6484:	li	a4,24
    6488:	beq	a0,a4,6570 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x128>
    648c:	bltu	a4,a0,6500 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xb8>
    6490:	li	a4,14
    6494:	beq	a0,a4,64d8 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x90>
    6498:	bltu	a4,a0,6580 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x138>
    649c:	addi	a4,a0,-10
    64a0:	zext.b	a4,a4
    64a4:	li	a3,1
    64a8:	li	a0,0
    64ac:	bltu	a3,a4,65b4 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x16c>
            return unpack_dst_format == DataFormat::Float16 || unpack_src_format == unpack_dst_format;
    64b0:	sub	a5,a5,a1
    64b4:	addi	a1,a1,-1
    64b8:	seqz	a5,a5
    64bc:	seqz	a1,a1
    64c0:	or	a0,a1,a5
    64c4:	ret
        //    ISA conversions:
        //      SrcA/SrcB path: NOT possible (ISA doc explicitly states "Not possible").
        //      Dst path:       INT32 → Integer "32" (Int32): valid (32b data movement to Dst).
        //    Hence, only valid when unpack_to_dest = true.
        case DataFormat::Int32:
            return unpack_dst_format == DataFormat::Int32 && unpack_to_dest;
    64c8:	addi	a1,a1,-8
    64cc:	seqz	a1,a1
    64d0:	and	a0,a2,a1
    64d4:	ret
        //         INT8 → Integer "8" (Int8): always valid.
        //       Dst path:
        //         INT8 → BF16  (Float16_b): always valid (BFP8+force_shared_exp).
        //         INT8 → Integer "8" (Int8): always valid.
        case DataFormat::Int8:
            switch (unpack_dst_format)
    64d8:	li	a4,5
                    return true;
    64dc:	li	a0,1
            switch (unpack_dst_format)
    64e0:	beq	a1,a4,64f4 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
    64e4:	beq	a1,a5,64f4 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
    64e8:	li	a5,4
    64ec:	beq	a1,a5,64f8 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xb0>
    64f0:	li	a0,0
        // -------------------------------------------------------------------------
        // 12. Unknown or not-yet-handled formats.
        default:
            return false;
    }
}
    64f4:	ret
                    return !unpack_to_dest;
    64f8:	xori	a0,a2,1
    64fc:	ret
    switch (unpack_src_format)
    6500:	li	a4,26
    6504:	beq	a0,a4,64b0 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x68>
    6508:	addi	a5,a0,-30
            return unpack_dst_format == DataFormat::Int8 || unpack_dst_format == DataFormat::UInt8;
    650c:	andi	a1,a1,239
    6510:	addi	a1,a1,-14
    switch (unpack_src_format)
    6514:	seqz	a5,a5
            return unpack_dst_format == DataFormat::Int8 || unpack_dst_format == DataFormat::UInt8;
    6518:	seqz	a0,a1
    switch (unpack_src_format)
    651c:	neg	a5,a5
    6520:	and	a0,a0,a5
    6524:	ret
    6528:	bnez	a0,64b0 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0x68>
            switch (unpack_dst_format)
    652c:	li	a5,1
                    return true;
    6530:	mv	a0,a1
            switch (unpack_dst_format)
    6534:	beq	a1,a5,64f4 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
                    return unpack_to_dest;
    6538:	mv	a0,a2
            switch (unpack_dst_format)
    653c:	bgeu	a5,a1,64f4 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
    6540:	addi	a1,a1,-4
    6544:	sltiu	a0,a1,2
    6548:	ret
            switch (unpack_dst_format)
    654c:	beq	a1,a0,64f8 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xb0>
    6550:	addi	a0,a1,-5
    6554:	seqz	a0,a0
    6558:	bltu	a5,a1,64f4 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
                    return unpack_to_dest;
    655c:	mv	a0,a2
            switch (unpack_dst_format)
    6560:	beqz	a1,64f4 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xac>
    6564:	addi	a1,a1,-1
    6568:	seqz	a0,a1
    656c:	ret
            return unpack_dst_format == DataFormat::UInt32 && unpack_to_dest;
    6570:	addi	a1,a1,-24
    6574:	seqz	a1,a1
    6578:	and	a0,a2,a1
    657c:	ret
    switch (unpack_src_format)
    6580:	li	a4,15
    6584:	bne	a0,a4,64f0 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xa8>
            switch (unpack_dst_format)
    6588:	li	a4,4
    658c:	beq	a1,a4,64f8 <ckernel::unpacker::is_unpacker_format_conversion_supported_dest(DataFormat, DataFormat, bool)+0xb0>
                    return unpack_src_format == unpack_dst_format;
    6590:	sub	a5,a5,a1
            switch (unpack_dst_format)
    6594:	addi	a1,a1,-5
                    return unpack_src_format == unpack_dst_format;
    6598:	seqz	a5,a5
            switch (unpack_dst_format)
    659c:	sltiu	a1,a1,2
    65a0:	or	a0,a1,a5
    65a4:	ret
            return unpack_dst_format == DataFormat::UInt16;
    65a8:	addi	a1,a1,-9
    65ac:	seqz	a0,a1
    65b0:	ret
    65b4:	ret

000065b8 <run_kernel(RuntimeParams const&)>:

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    65b8:	addi	sp,sp,-64
    65bc:	sw	s5,36(sp)
#endif
} // namespace detail

__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    std::uint32_t n = detail::next_zone_id;
    65c0:	addi	s5,gp,-1984 # ffb00040 <llk_perf::detail::next_zone_id>
    65c4:	lw	a5,0(s5)
    65c8:	sw	ra,60(sp)
    65cc:	sw	s0,56(sp)
    65d0:	sw	s1,52(sp)
    65d4:	sw	s2,48(sp)
    65d8:	sw	s3,44(sp)
    65dc:	sw	s4,40(sp)
    65e0:	sw	s6,32(sp)
    65e4:	sw	s7,28(sp)
    for (std::uint32_t i = 0; i < n; ++i)
    65e8:	beqz	a5,6c5c <run_kernel(RuntimeParams const&)+0x6a4>
    {
        if (detail::zone_hashes[i] == hash_val)
    65ec:	lui	a4,0x7c867
    65f0:	lw	a3,4(s5)
    65f4:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    65f8:	beq	a3,a4,667c <run_kernel(RuntimeParams const&)+0xc4>
    for (std::uint32_t i = 0; i < n; ++i)
    65fc:	li	a3,1
    6600:	beq	a5,a3,6c5c <run_kernel(RuntimeParams const&)+0x6a4>
        if (detail::zone_hashes[i] == hash_val)
    6604:	lw	a3,8(s5)
    6608:	beq	a3,a4,667c <run_kernel(RuntimeParams const&)+0xc4>
    for (std::uint32_t i = 0; i < n; ++i)
    660c:	li	a3,2
    6610:	beq	a5,a3,6c5c <run_kernel(RuntimeParams const&)+0x6a4>
        if (detail::zone_hashes[i] == hash_val)
    6614:	lw	a3,12(s5)
    6618:	beq	a3,a4,667c <run_kernel(RuntimeParams const&)+0xc4>
    for (std::uint32_t i = 0; i < n; ++i)
    661c:	li	a3,3
    6620:	beq	a5,a3,6c5c <run_kernel(RuntimeParams const&)+0x6a4>
        if (detail::zone_hashes[i] == hash_val)
    6624:	lw	a3,16(s5)
    6628:	beq	a3,a4,667c <run_kernel(RuntimeParams const&)+0xc4>
    for (std::uint32_t i = 0; i < n; ++i)
    662c:	li	a3,4
    6630:	beq	a5,a3,6c5c <run_kernel(RuntimeParams const&)+0x6a4>
        if (detail::zone_hashes[i] == hash_val)
    6634:	lw	a3,20(s5)
    6638:	beq	a3,a4,667c <run_kernel(RuntimeParams const&)+0xc4>
    for (std::uint32_t i = 0; i < n; ++i)
    663c:	li	a4,5
    6640:	beq	a5,a4,6c5c <run_kernel(RuntimeParams const&)+0x6a4>
        if (detail::zone_hashes[i] == hash_val)
    6644:	lui	a4,0x7c867
    6648:	lw	a3,24(s5)
    664c:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    6650:	beq	a3,a4,667c <run_kernel(RuntimeParams const&)+0xc4>
    for (std::uint32_t i = 0; i < n; ++i)
    6654:	li	a3,6
    6658:	beq	a5,a3,6c5c <run_kernel(RuntimeParams const&)+0x6a4>
        if (detail::zone_hashes[i] == hash_val)
    665c:	lw	a3,28(s5)
    6660:	beq	a3,a4,667c <run_kernel(RuntimeParams const&)+0xc4>
    for (std::uint32_t i = 0; i < n; ++i)
    6664:	li	a3,7
    6668:	beq	a5,a3,6c5c <run_kernel(RuntimeParams const&)+0x6a4>
        if (detail::zone_hashes[i] == hash_val)
    666c:	lw	a3,32(s5)
    6670:	beq	a3,a4,667c <run_kernel(RuntimeParams const&)+0xc4>
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
    6674:	li	a4,8
    6678:	bne	a5,a4,6c5c <run_kernel(RuntimeParams const&)+0x6a4>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    667c:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6680:	lw	a5,32(a4)
                ckernel::semaphore_post(PERF_ENTRY_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    6684:	zext.b	a5,a5
    6688:	bnez	a5,66a0 <run_kernel(RuntimeParams const&)+0xe8>
            {
                asm volatile("nop");
    668c:	nop
    6690:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6694:	lw	a5,32(a4)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    6698:	zext.b	a5,a5
    669c:	beqz	a5,668c <run_kernel(RuntimeParams const&)+0xd4>
    66a0:	lw	a5,32(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    66a4:	zext.b	a5,a5
    66a8:	beqz	a5,6ce4 <run_kernel(RuntimeParams const&)+0x72c>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    66ac:	li	a2,1
    66b0:	sw	a2,32(a4)
    {
    66b4:	sb	zero,0(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    66b8:	lw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    66bc:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    66c0:	li	a1,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    66c4:	add	a4,a5,a3
    66c8:	addi	a4,a4,-1021
        if (!is_buffer_full())
    66cc:	bgeu	a1,a4,6714 <run_kernel(RuntimeParams const&)+0x15c>
    return p_reg[0];
    66d0:	lui	a4,0xffb12
    66d4:	lw	a1,496(a4) # ffb121f0 <__stack_top+0x111f0>
    66d8:	lw	a4,504(a4)
            is_opened = true;
    66dc:	sb	a2,0(sp)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    66e0:	lui	a2,0xffb00
    66e4:	lw	a2,0(a2) # ffb00000 <llk_profiler::buffer>
    66e8:	lui	a0,0xaa3a0
    66ec:	slli	a4,a4,0x14
    66f0:	srli	a4,a4,0x14
    66f4:	or	a4,a4,a0
            ++open_zone_cnt;
    66f8:	addi	a3,a3,1
    66fc:	sh2add	a2,a5,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6700:	sw	a4,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6704:	addi	a5,a5,2
            ++open_zone_cnt;
    6708:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    670c:	sw	a1,4(a2)
    6710:	sw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    const std::uint32_t unpB_num_faces  = 4)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
    6714:	li	a1,6
    6718:	mv	a0,a1
    671c:	li	a2,1
    6720:	jal	62dc <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)>
    6724:	beqz	a0,6cf0 <run_kernel(RuntimeParams const&)+0x738>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6728:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    672c:	lw	a5,52(a4)
    while (semaphore_read(semaphore::UNPACK_SYNC) > 0)
    6730:	zext.b	a5,a5
    6734:	bnez	a5,672c <run_kernel(RuntimeParams const&)+0x174>
    TTI_SETADCXY(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1011);
    6738:	ttsetadcxy	3,0,0,0,0,11
    TTI_SETADCZW(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1111);
    673c:	ttsetadczw	3,0,0,0,0,15
    if (cfg_state_id == 0)
    6740:	lw	a4,-2000(gp) # ffb00030 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    6744:	lui	a5,0xffef0
    if (cfg_state_id == 0)
    6748:	beqz	a4,6750 <run_kernel(RuntimeParams const&)+0x198>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
    674c:	addi	a5,a5,896 # ffef0380 <__instrn_buffer+0xb0380>
    std::uint32_t unpA_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpA_ch1_x_stride;
    std::uint32_t unpB_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpB_ch1_x_stride;
    std::uint32_t exp_width         = (static_cast<std::uint32_t>(unpA_dst_format_masked) >> 2) & 0x1; // 0=5-bit, 1=8-bit

    // Strides for incrementing ch1 address to srcA and srcB
    cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
    6750:	li	a1,256
    6754:	sw	a1,228(a5)
        (0 << UNP0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
        (unpA_ch1_z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT); // Z and W(not used) stride for dest address (ch1)

    cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
    6758:	sw	a1,236(a5)
    TTI_ATGETM(index);
    675c:	ttatgetm	0
    std::uint8_t mask_b0 = Mask & 0xff;

    if (mask_b0 != 0)
    {
        std::uint8_t data_b0 = wrdata & 0xff;
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    6760:	lui	s6,0xffe40
    6764:	mv	s6,s6
    6768:	lui	a4,0xb3ff0
    676c:	sw	a4,0(s6) # ffe40000 <__instrn_buffer>
    std::uint8_t mask_b1 = (Mask >> 8) & 0xff;

    if (mask_b1 != 0)
    {
        std::uint8_t data_b1 = (wrdata) & 0xff;
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    6770:	lui	a4,0xb47f0
    6774:	sw	a4,0(s6)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    6778:	lui	a4,0xb3070
    677c:	addi	a4,a4,1 # b3070001 <__runtime_args_end+0xb304fc01>
    6780:	sw	a4,0(s6)
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    6784:	lui	a4,0xb4800
    6788:	addi	a4,a4,1 # b4800001 <__runtime_args_end+0xb47dfc01>
    678c:	sw	a4,0(s6)
    std::uint8_t mask_b2 = (Mask >> 16) & 0xff;

    if (mask_b2 != 0)
    {
        std::uint8_t data_b2 = (wrdata) & 0xff;
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
    6790:	lui	a4,0xb5010
    6794:	addi	a4,a4,1 # b5010001 <__runtime_args_end+0xb4fefc01>
    6798:	sw	a4,0(s6)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    679c:	lui	a4,0xb3010
    67a0:	addi	a4,a4,2 # b3010002 <__runtime_args_end+0xb2fefc02>
    67a4:	sw	a4,0(s6)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
    67a8:	lui	a4,0xb5400
    67ac:	addi	a3,a4,71 # b5400047 <__runtime_args_end+0xb53dfc47>
    67b0:	sw	a3,0(s6)
    67b4:	addi	a4,a4,119
    67b8:	sw	a4,0(s6)
    TTI_ATRELM(index);
    67bc:	ttatrelm	0
    tile_descriptor.f.z_dim          = unpA_num_faces;
    // tile_descriptor.f.blobs_per_xy_plane = 0;
    // tile_descriptor.f.blobs_y_start = 0;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67c0:	li	a4,22
    67c4:	sw	a4,256(a5)
    67c8:	lui	a4,0x40
    67cc:	addi	a4,a4,1 # 40001 <__runtime_args_end+0x1fc01>
    tile_descriptor.f.in_data_format = row_pool ? to_underlying(DataFormat::Float32) : unpB_src_format_masked;
    tile_descriptor.f.x_dim          = unpB_face_r_dim * FACE_C_DIM;
    tile_descriptor.f.z_dim          = unpB_num_faces;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67d0:	lui	a3,0x1000
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67d4:	sw	a4,260(a5)
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67d8:	addi	a2,a3,22 # 1000016 <__runtime_args_end+0xfdfc16>
    67dc:	sw	a2,448(a5)
    67e0:	sw	a4,452(a5)
    config.f.uncompress_cntx4_7 = 0xf;
    // config.f.limit_addr = 0; // Set dynamically
    // config.f.fifo_size = 0; // Set dynamically
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    67e4:	li	a2,38
    67e8:	lui	a4,0xf0
    67ec:	sw	a2,288(a5)
    67f0:	addi	a4,a4,15 # f000f <__runtime_args_end+0xcfc0f>
    67f4:	sw	a4,292(a5)
    config.f.out_data_format = row_pool ? (to_underlying(DataFormat::Float16) | (exp_width << 2)) : unpB_dst_format_masked;
    config.f.haloize_mode    = 0;

    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC1_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    67f8:	sw	a2,480(a5)
    67fc:	sw	a4,484(a5)
    }

    std::uint32_t unpA_x_end = (unpA_face_r_dim == 0) ? 1 : (unpA_face_r_dim << 4) - 1;
    TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    6800:	lui	a4,0x5e240
    6804:	addi	a4,a4,-1024 # 5e23fc00 <__runtime_args_end+0x5e21f800>
    6808:	sw	a4,0(s6)
    TT_SETADCXX(p_setadc::UNP_B, (unpB_face_r_dim << 4) - 1, 0x0);
    680c:	lui	a4,0x5e440
    6810:	addi	a4,a4,-1024 # 5e43fc00 <__runtime_args_end+0x5e41f800>

    // Program base address for all 2 sections (each section address is loaded to corresponding context)
    // Load dummy data to unused location if face height is 0
    const std::uint32_t Dest_cntx0_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    const std::uint32_t Dest_cntx1_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);
    6814:	lui	a2,0x400
    TT_SETADCXX(p_setadc::UNP_B, (unpB_face_r_dim << 4) - 1, 0x0);
    6818:	sw	a4,0(s6)
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);
    681c:	addi	a2,a2,64 # 400040 <__runtime_args_end+0x3dfc40>
    6820:	sw	a2,336(a5)
    // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
    const std::uint32_t face_dim                 = unpA_face_r_dim * FACE_C_DIM;
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = face_dim | (face_dim << 16);

    constexpr std::uint32_t face_dim_16x16 = FACE_R_DIM * FACE_C_DIM;
    regfile[p_gpr_unpack::FACE_DIM_16x16]  = (face_dim_16x16 / 1) | ((face_dim_16x16 / 1) << 16);
    6824:	lw	a4,-2040(gp) # ffb00008 <ckernel::regfile>
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = face_dim | (face_dim << 16);
    6828:	add	a3,a3,a1
    682c:	sw	a3,344(a5)
    regfile[p_gpr_unpack::FACE_DIM_8x16]   = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    6830:	lui	a0,0x800
    regfile[p_gpr_unpack::FACE_DIM_16x16]  = (face_dim_16x16 / 1) | ((face_dim_16x16 / 1) << 16);
    6834:	sw	a3,160(a4)
    regfile[p_gpr_unpack::FACE_DIM_8x16]   = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    6838:	addi	a3,a0,128 # 800080 <__runtime_args_end+0x7dfc80>
    683c:	sw	a3,164(a4)
    regfile[p_gpr_unpack::FACE_DIM_4x16]   = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    6840:	lui	a3,0x200
    regfile[p_gpr_unpack::FACE_DIM_4x16]   = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    6844:	sw	a2,168(a4)
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    6848:	addi	a2,a3,32 # 200020 <__runtime_args_end+0x1dfc20>
    regfile[p_gpr_unpack::FACE_DIM_1x16]   = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    684c:	lui	a3,0x100
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    6850:	sw	a2,172(a4)
    regfile[p_gpr_unpack::FACE_DIM_1x16]   = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    6854:	addi	a3,a3,16 # 100010 <__runtime_args_end+0xdfc10>
    6858:	sw	a3,176(a4)
    volatile std::uint32_t foo     = 0x0;
    685c:	sw	zero,8(sp)
    *fooptr                        = regfile[index];
    6860:	lw	a4,176(a4)
    6864:	sw	a4,8(sp)
    sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);

    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);
    6868:	ttsetc16	5,4

    // Enable address counter for unpacker ch1/dst address
    // final address is calculated as: Dest_cntx0/1_address + address_counter_ch1
    // used for face by face unpacking of entire tile into srcA
    cfg[UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32] = 0x1 << UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT;
    686c:	sw	a1,200(a5)
    unp_cfg_context = 0;
    6870:	sw	zero,-1988(gp) # ffb0003c <unp_cfg_context>
    TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    6874:	ttsetc16	41,0
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");
    configure_unpack_AB<is_fp32_dest_acc_en, false, false, false, disable_src_zero_flag>(
        unpA_src_format, unpB_src_format, unpA_dst_format, unpB_dst_format, unpA_face_r_dim, unpB_face_r_dim, 0, unpA_num_faces, unpB_num_faces);

    TT_SETDMAREG(0, LOWER_HALFWORD(unpA_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    6878:	lui	a5,0x45000
    687c:	addi	a3,a5,72 # 45000048 <__runtime_args_end+0x44fdfc48>
    6880:	sw	a3,0(s6)
    6884:	lui	a4,0x2021
    TT_SETDMAREG(0, LOWER_HALFWORD(unpB_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
    6888:	addi	a5,a5,74
    688c:	addi	a4,a4,16 # 2021010 <__runtime_args_end+0x2000c10>
    6890:	sw	a5,0(s6)
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    6894:	addi	a0,sp,4
    6898:	sw	a4,4(sp)
    689c:	jal	627c <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>
    68a0:	beqz	a0,6cfc <run_kernel(RuntimeParams const&)+0x744>
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    68a4:	lui	a5,0xb4010
    68a8:	addi	a5,a5,72 # b4010048 <__runtime_args_end+0xb3fefc48>
    68ac:	sw	a5,0(s6)
            break;
        case 8:
            TTI_SETADCXX(UNP_SEL, 8 * FACE_C_DIM - 1, 0x0);
            break;
        default:
            TTI_SETADCXX(UNP_SEL, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    68b0:	ttsetadcxx	3,255,0
    68b4:	lw	a5,4(sp)
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    68b8:	addi	a0,sp,12
    68bc:	srli	s7,a5,0x10
    68c0:	sw	a5,12(sp)
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    68c4:	zext.b	s7,s7
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
    68c8:	srli	s6,a5,0x18
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    68cc:	jal	627c <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>
    68d0:	beqz	a0,6cdc <run_kernel(RuntimeParams const&)+0x724>
    store_blocking(&pc_buf_base[2], 0);
    68d4:	lw	a5,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
    68d8:	li	a3,0
    store_blocking(&pc_buf_base[2], 0);
    68dc:	addi	a5,a5,8
    asm volatile(
    68e0:	sw	a3,0(a5)
    68e4:	lw	a3,0(a5)
    68e8:	and	zero,zero,a3
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
    68ec:	lui	a5,0xffb80
    68f0:	sw	s7,0(a5) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
    68f4:	sw	s6,4(a5)
    mop_cfg[2] = m_start_op0;
    68f8:	lui	a3,0x2000
    68fc:	sw	a3,8(a5)
    mop_cfg[3] = m_end_op0;
    6900:	sw	a3,12(a5)
    mop_cfg[4] = m_end_op1;
    mop_cfg[5] = m_loop_op0;
    6904:	lui	a2,0x42008
    mop_cfg[4] = m_end_op1;
    6908:	sw	a3,16(a5)
    mop_cfg[5] = m_loop_op0;
    690c:	addi	a2,a2,193 # 420080c1 <__runtime_args_end+0x41fe7cc1>
    mop_cfg[6] = m_loop_op1;
    6910:	lui	a3,0x42808
    mop_cfg[5] = m_loop_op0;
    6914:	sw	a2,20(a5)
    mop_cfg[6] = m_loop_op1;
    6918:	addi	a3,a3,193 # 428080c1 <__runtime_args_end+0x427e7cc1>
    691c:	sw	a3,24(a5)
    mop_cfg[7] = m_loop0_last_instr;
    6920:	sw	a3,28(a5)
    mop_cfg[8] = m_loop1_last_instr;
    6924:	sw	a3,32(a5)
    store_blocking(&pc_buf_base[1], 0);
    6928:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
    692c:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    6930:	addi	a4,a3,4
    asm volatile(
    6934:	sw	a5,0(a4)
    6938:	lw	a5,0(a4)
    693c:	and	zero,zero,a5
        if (is_opened)
    6940:	lbu	a5,0(sp)
    6944:	beqz	a5,6990 <run_kernel(RuntimeParams const&)+0x3d8>
    return p_reg[0];
    6948:	lui	a5,0xffb12
    694c:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    6950:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6954:	lui	a4,0xffb00
    6958:	lw	a3,0(a4) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    695c:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6960:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    6964:	lui	a0,0xba3a0
    6968:	slli	a5,a5,0x14
    696c:	srli	a5,a5,0x14
    6970:	or	a5,a5,a0
    6974:	sh2add	a3,a4,a3
    6978:	sw	a5,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    697c:	addi	a4,a4,2
            --open_zone_cnt;
    6980:	addi	a5,a2,-1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6984:	sw	a1,4(a3)
    6988:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    698c:	sw	a5,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6990:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6994:	lw	a5,40(a4)
                ckernel::semaphore_post(PERF_EXIT_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    6998:	zext.b	a5,a5
    699c:	bnez	a5,69b4 <run_kernel(RuntimeParams const&)+0x3fc>
            {
                asm volatile("nop");
    69a0:	nop
    69a4:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    69a8:	lw	a5,40(a4)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    69ac:	zext.b	a5,a5
    69b0:	beqz	a5,69a0 <run_kernel(RuntimeParams const&)+0x3e8>
    69b4:	lw	a5,40(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    69b8:	zext.b	a5,a5
    69bc:	beqz	a5,6d38 <run_kernel(RuntimeParams const&)+0x780>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    69c0:	li	a3,1
    69c4:	sw	a3,40(a4)
    std::uint32_t n = detail::next_zone_id;
    69c8:	lw	a5,0(s5)
    for (std::uint32_t i = 0; i < n; ++i)
    69cc:	beqz	a5,6cc0 <run_kernel(RuntimeParams const&)+0x708>
        if (detail::zone_hashes[i] == hash_val)
    69d0:	lui	a4,0xbd77
    69d4:	lw	a2,4(s5)
    69d8:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    69dc:	beq	a2,a4,6a5c <run_kernel(RuntimeParams const&)+0x4a4>
    for (std::uint32_t i = 0; i < n; ++i)
    69e0:	beq	a5,a3,6cc0 <run_kernel(RuntimeParams const&)+0x708>
        if (detail::zone_hashes[i] == hash_val)
    69e4:	lw	a3,8(s5)
    69e8:	beq	a3,a4,6a5c <run_kernel(RuntimeParams const&)+0x4a4>
    for (std::uint32_t i = 0; i < n; ++i)
    69ec:	li	a3,2
    69f0:	beq	a5,a3,6cc0 <run_kernel(RuntimeParams const&)+0x708>
        if (detail::zone_hashes[i] == hash_val)
    69f4:	lw	a3,12(s5)
    69f8:	beq	a3,a4,6a5c <run_kernel(RuntimeParams const&)+0x4a4>
    for (std::uint32_t i = 0; i < n; ++i)
    69fc:	li	a3,3
    6a00:	beq	a5,a3,6cc0 <run_kernel(RuntimeParams const&)+0x708>
        if (detail::zone_hashes[i] == hash_val)
    6a04:	lw	a3,16(s5)
    6a08:	beq	a3,a4,6a5c <run_kernel(RuntimeParams const&)+0x4a4>
    for (std::uint32_t i = 0; i < n; ++i)
    6a0c:	li	a3,4
    6a10:	beq	a5,a3,6cc0 <run_kernel(RuntimeParams const&)+0x708>
        if (detail::zone_hashes[i] == hash_val)
    6a14:	lw	a3,20(s5)
    6a18:	beq	a3,a4,6a5c <run_kernel(RuntimeParams const&)+0x4a4>
    for (std::uint32_t i = 0; i < n; ++i)
    6a1c:	li	a4,5
    6a20:	beq	a5,a4,6cc0 <run_kernel(RuntimeParams const&)+0x708>
        if (detail::zone_hashes[i] == hash_val)
    6a24:	lui	a4,0xbd77
    6a28:	lw	a3,24(s5)
    6a2c:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    6a30:	beq	a3,a4,6a5c <run_kernel(RuntimeParams const&)+0x4a4>
    for (std::uint32_t i = 0; i < n; ++i)
    6a34:	li	a3,6
    6a38:	beq	a5,a3,6cc0 <run_kernel(RuntimeParams const&)+0x708>
        if (detail::zone_hashes[i] == hash_val)
    6a3c:	lw	a3,28(s5)
    6a40:	beq	a3,a4,6a5c <run_kernel(RuntimeParams const&)+0x4a4>
    for (std::uint32_t i = 0; i < n; ++i)
    6a44:	li	a3,7
    6a48:	beq	a5,a3,6cc0 <run_kernel(RuntimeParams const&)+0x708>
        if (detail::zone_hashes[i] == hash_val)
    6a4c:	lw	a3,32(s5)
    6a50:	beq	a3,a4,6a5c <run_kernel(RuntimeParams const&)+0x4a4>
    if (n < PERF_COUNTERS_MAX_ZONES)
    6a54:	li	a4,8
    6a58:	bne	a5,a4,6cc0 <run_kernel(RuntimeParams const&)+0x708>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6a5c:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6a60:	lw	a5,32(a4)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    6a64:	zext.b	a5,a5
    6a68:	bnez	a5,6a80 <run_kernel(RuntimeParams const&)+0x4c8>
                asm volatile("nop");
    6a6c:	nop
    6a70:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6a74:	lw	a5,32(a4)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    6a78:	zext.b	a5,a5
    6a7c:	beqz	a5,6a6c <run_kernel(RuntimeParams const&)+0x4b4>
    6a80:	lw	a5,32(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6a84:	zext.b	a5,a5
    6a88:	beqz	a5,6d2c <run_kernel(RuntimeParams const&)+0x774>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6a8c:	li	a2,1
    6a90:	sw	a2,32(a4)
    {
    6a94:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    6a98:	lw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    6a9c:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    6aa0:	li	a1,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    6aa4:	add	a4,a3,a5
    6aa8:	addi	a4,a4,-1021
        if (!is_buffer_full())
    6aac:	bgeu	a1,a4,6af4 <run_kernel(RuntimeParams const&)+0x53c>
    return p_reg[0];
    6ab0:	lui	a4,0xffb12
    6ab4:	lw	a1,496(a4) # ffb121f0 <__stack_top+0x111f0>
    6ab8:	lw	a4,504(a4)
            is_opened = true;
    6abc:	sb	a2,12(sp)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6ac0:	lui	a2,0xffb00
    6ac4:	lw	a2,0(a2) # ffb00000 <llk_profiler::buffer>
    6ac8:	lui	a0,0xa99a8
    6acc:	slli	a4,a4,0x14
    6ad0:	srli	a4,a4,0x14
    6ad4:	or	a4,a4,a0
            ++open_zone_cnt;
    6ad8:	addi	a3,a3,1
    6adc:	sh2add	a2,a5,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6ae0:	sw	a4,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6ae4:	addi	a5,a5,2
            ++open_zone_cnt;
    6ae8:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6aec:	sw	a1,4(a2)
    6af0:	sw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
constexpr std::uint32_t PERF_OUTPUT  = PERF_INPUT_C + 16 * 4096;

constexpr std::uint32_t PERF_ADDRESS(std::uint32_t buffer, std::uint32_t tile)
{
    std::uint32_t address = buffer + (tile % 16) * 4096; // Loop every 16 tiles, to prevent escaping memory
    return address / 16 - 1;                             // Correct the L1 Address for Tensix
    6af4:	lui	t3,0x2
    6af8:	lui	t1,0x3
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6afc:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
inline void _llk_unpack_configure_addresses_(const std::uint32_t address_a, const std::uint32_t address_b, volatile std::uint32_t tt_reg_ptr *cfg)
{
    LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

    if (0 == unp_cfg_context)
    6b00:	lw	a1,-1988(gp) # ffb0003c <unp_cfg_context>
    6b04:	addi	t3,t3,255 # 20ff <BRISC_LOCAL_MEM_LENGTH+0xff>
    6b08:	addi	t1,t1,255 # 30ff <BRISC_LOCAL_MEM_LENGTH+0x10ff>
    6b0c:	li	a2,0
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    6b10:	lui	t0,0xffef0
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6b14:	li	t6,14
    unp_cfg_context = 1 - unp_cfg_context;
    6b18:	li	a6,1
            _perf_unpack_loop_set_valid<true, true>(TILE_CNT * TILE_NUM_FACES);
            return;
        }
        else
        {
            for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
    6b1c:	lui	t4,0x1
    6b20:	lui	t5,0x10
    6b24:	srli	a5,a2,0x4
    6b28:	add	a7,a5,t3
    6b2c:	add	a3,a5,t1
 */

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(const std::uint32_t address_a, const std::uint32_t address_b)
{
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111); // reset counters
    6b30:	ttsetadczw	3,0,0,0,0,15
    if (cfg_state_id == 0)
    6b34:	lw	a5,-2000(gp) # ffb00030 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    6b38:	lui	a0,0xffef0
    if (cfg_state_id == 0)
    6b3c:	beqz	a5,6b44 <run_kernel(RuntimeParams const&)+0x58c>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
    6b40:	addi	a0,t0,896 # ffef0380 <__instrn_buffer+0xb0380>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6b44:	lw	a5,52(a4)
    while (semaphore_read(semaphore::UNPACK_SYNC) >= num_contexts)
    6b48:	andi	a5,a5,254
    6b4c:	bnez	a5,6b44 <run_kernel(RuntimeParams const&)+0x58c>
    6b50:	bnez	a1,6c78 <run_kernel(RuntimeParams const&)+0x6c0>
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
    6b54:	sw	a7,304(a0) # ffef0130 <__instrn_buffer+0xb0130>
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    6b58:	sw	a3,496(a0)
    6b5c:	lw	a5,52(a4)
    6b60:	mv	a3,a1
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6b64:	zext.b	a5,a5
    6b68:	bltu	t6,a5,6c90 <run_kernel(RuntimeParams const&)+0x6d8>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6b6c:	sw	zero,52(a4)

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    6b70:	ttstallwait	8,1024
    TTI_MOP(1, 0, 0); // run the double-loop template
    6b74:	ttmop	1,0,0
    TTI_SEMGET(semaphore::t6_sem(index));
    6b78:	ttsemget	32
    unp_cfg_context = 1 - unp_cfg_context;
    6b7c:	sub	a1,a6,a3
    6b80:	sw	a1,-1988(gp) # ffb0003c <unp_cfg_context>
    if (unp_cfg_context == 0)
    6b84:	beq	a3,a6,6cb8 <run_kernel(RuntimeParams const&)+0x700>
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0101);
    6b88:	ttsetc16	41,257
    6b8c:	add	a2,a2,t4
    6b90:	bne	a2,t5,6b24 <run_kernel(RuntimeParams const&)+0x56c>
    asm volatile(
    6b94:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    6b98:	addi	a4,a4,4
    asm volatile(
    6b9c:	sw	a5,0(a4)
    6ba0:	lw	a5,0(a4)
    6ba4:	and	zero,zero,a5
        if (is_opened)
    6ba8:	lbu	a5,12(sp)
    6bac:	beqz	a5,6bf8 <run_kernel(RuntimeParams const&)+0x640>
    return p_reg[0];
    6bb0:	lui	a5,0xffb12
    6bb4:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    6bb8:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6bbc:	lui	a4,0xffb00
    6bc0:	addi	t4,t4,-1 # fff <__firmware_stack_size+0xdff>
    6bc4:	lw	a3,0(a4) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    6bc8:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6bcc:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    6bd0:	lui	a0,0xb99a8
    6bd4:	and	a5,a5,t4
    6bd8:	or	a5,a5,a0
    6bdc:	sh2add	a3,a4,a3
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6be0:	addi	a4,a4,2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6be4:	sw	a5,0(a3)
            --open_zone_cnt;
    6be8:	addi	a5,a2,-1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6bec:	sw	a1,4(a3)
    6bf0:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    6bf4:	sw	a5,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6bf8:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6bfc:	lw	a5,40(a4)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    6c00:	zext.b	a5,a5
    6c04:	bnez	a5,6c1c <run_kernel(RuntimeParams const&)+0x664>
                asm volatile("nop");
    6c08:	nop
    6c0c:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6c10:	lw	a5,40(a4)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    6c14:	zext.b	a5,a5
    6c18:	beqz	a5,6c08 <run_kernel(RuntimeParams const&)+0x650>
    6c1c:	lw	a5,40(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6c20:	zext.b	a5,a5
    6c24:	beqz	a5,6d20 <run_kernel(RuntimeParams const&)+0x768>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6c28:	li	a5,1
    6c2c:	sw	a5,40(a4)
                _llk_unpack_AB_<>(PERF_ADDRESS(PERF_INPUT_A, tile), PERF_ADDRESS(PERF_INPUT_B, tile));
            }
        }
        PROFILER_SYNC();
    }
}
    6c30:	lw	ra,60(sp)
    6c34:	lw	s0,56(sp)
    6c38:	lw	s1,52(sp)
    6c3c:	lw	s2,48(sp)
    6c40:	lw	s3,44(sp)
    6c44:	lw	s4,40(sp)
    6c48:	lw	s5,36(sp)
    6c4c:	lw	s6,32(sp)
    6c50:	lw	s7,28(sp)
    6c54:	addi	sp,sp,64
    6c58:	ret
        detail::zone_hashes[n] = hash_val;
    6c5c:	lui	a4,0x7c867
    6c60:	sh2add	a3,a5,s5
    6c64:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
        detail::next_zone_id   = n + 1;
    6c68:	addi	a5,a5,1
        detail::zone_hashes[n] = hash_val;
    6c6c:	sw	a4,4(a3)
        detail::next_zone_id   = n + 1;
    6c70:	sw	a5,0(s5)
        return n;
    6c74:	j	667c <run_kernel(RuntimeParams const&)+0xc4>
    }
    else
    {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
    6c78:	sw	a7,308(a0) # b99a8134 <__runtime_args_end+0xb9987d34>
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    6c7c:	sw	a3,500(a0)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6c80:	lw	a5,52(a4)
    6c84:	mv	a3,a1
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6c88:	zext.b	a5,a5
    6c8c:	bgeu	t6,a5,6b6c <run_kernel(RuntimeParams const&)+0x5b4>
    6c90:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6c94:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    unp_cfg_context = 1 - unp_cfg_context;
    6c98:	lw	a3,-1988(gp) # ffb0003c <unp_cfg_context>
    6c9c:	sw	zero,52(a4)
    6ca0:	ttstallwait	8,1024
    6ca4:	ttmop	1,0,0
    TTI_SEMGET(semaphore::t6_sem(index));
    6ca8:	ttsemget	32
    6cac:	sub	a1,a6,a3
    6cb0:	sw	a1,-1988(gp) # ffb0003c <unp_cfg_context>
    if (unp_cfg_context == 0)
    6cb4:	bne	a3,a6,6b88 <run_kernel(RuntimeParams const&)+0x5d0>
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    6cb8:	ttsetc16	41,0
    6cbc:	j	6b8c <run_kernel(RuntimeParams const&)+0x5d4>
        detail::zone_hashes[n] = hash_val;
    6cc0:	lui	a4,0xbd77
    6cc4:	sh2add	a3,a5,s5
    6cc8:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
        detail::next_zone_id   = n + 1;
    6ccc:	addi	a5,a5,1
        detail::zone_hashes[n] = hash_val;
    6cd0:	sw	a4,4(a3)
        detail::next_zone_id   = n + 1;
    6cd4:	sw	a5,0(s5)
        return n;
    6cd8:	j	6a5c <run_kernel(RuntimeParams const&)+0x4a4>
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    6cdc:	ebreak
    6ce0:	j	68d4 <run_kernel(RuntimeParams const&)+0x31c>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6ce4:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6ce8:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6cec:	j	66ac <run_kernel(RuntimeParams const&)+0xf4>
    LLK_ASSERT(
    6cf0:	ebreak
    LLK_ASSERT(
    6cf4:	ebreak
    6cf8:	j	6728 <run_kernel(RuntimeParams const&)+0x170>
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    6cfc:	ebreak
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    6d00:	lui	a4,0xb4010
    6d04:	addi	a4,a4,72 # b4010048 <__runtime_args_end+0xb3fefc48>
    config_unpacker_x_end<p_setadc::UNP_AB>(tensor_shape.face_r_dim);
    6d08:	lbu	a5,4(sp)
    6d0c:	sw	a4,0(s6)
    LLK_ASSERT(
    6d10:	li	a4,16
    6d14:	bgeu	a4,a5,6d44 <run_kernel(RuntimeParams const&)+0x78c>
    6d18:	ebreak
    6d1c:	j	68b0 <run_kernel(RuntimeParams const&)+0x2f8>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6d20:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6d24:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6d28:	j	6c28 <run_kernel(RuntimeParams const&)+0x670>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6d2c:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6d30:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6d34:	j	6a8c <run_kernel(RuntimeParams const&)+0x4d4>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6d38:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6d3c:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6d40:	j	69c0 <run_kernel(RuntimeParams const&)+0x408>
    6d44:	lui	a4,0xffff0
    6d48:	addi	a4,a4,-279 # fffefee9 <__instrn_buffer+0x1afee9>
    6d4c:	sra	a4,a4,a5
    6d50:	andi	a4,a4,1
    6d54:	bnez	a4,6d8c <run_kernel(RuntimeParams const&)+0x7d4>
    switch (face_r_dim)
    6d58:	li	a4,4
    6d5c:	beq	a5,a4,6d7c <run_kernel(RuntimeParams const&)+0x7c4>
    6d60:	bltu	a4,a5,6d94 <run_kernel(RuntimeParams const&)+0x7dc>
    6d64:	li	a4,1
    6d68:	beq	a5,a4,6d84 <run_kernel(RuntimeParams const&)+0x7cc>
    6d6c:	li	a4,2
    6d70:	bne	a5,a4,68b0 <run_kernel(RuntimeParams const&)+0x2f8>
            TTI_SETADCXX(UNP_SEL, 2 * FACE_C_DIM - 1, 0x0);
    6d74:	ttsetadcxx	3,31,0
            break;
    6d78:	j	68b4 <run_kernel(RuntimeParams const&)+0x2fc>
            TTI_SETADCXX(UNP_SEL, 4 * FACE_C_DIM - 1, 0x0);
    6d7c:	ttsetadcxx	3,63,0
            break;
    6d80:	j	68b4 <run_kernel(RuntimeParams const&)+0x2fc>
            TTI_SETADCXX(UNP_SEL, 1 * FACE_C_DIM - 1, 0x0);
    6d84:	ttsetadcxx	3,15,0
            break;
    6d88:	j	68b4 <run_kernel(RuntimeParams const&)+0x2fc>
    LLK_ASSERT(
    6d8c:	ebreak
    6d90:	j	6d58 <run_kernel(RuntimeParams const&)+0x7a0>
    switch (face_r_dim)
    6d94:	li	a4,8
    6d98:	bne	a5,a4,68b0 <run_kernel(RuntimeParams const&)+0x2f8>
            TTI_SETADCXX(UNP_SEL, 8 * FACE_C_DIM - 1, 0x0);
    6d9c:	ttsetadcxx	3,127,0
            break;
    6da0:	j	68b4 <run_kernel(RuntimeParams const&)+0x2fc>

00006da4 <_init()>:
    }
}

void _init(void)
{
}
    6da4:	ret

00006da8 <_fini()>:

void _fini(void)
    6da8:	ret

00006dac <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
    6dac:	lui	a5,0x20
    6db0:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
    6db4:	sb	a5,0(a0)
        (void)(dstc[i]);
    6db8:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
    6dbc:	fence
}
    6dc0:	ret

00006dc4 <memset>:
    6dc4:	li	t1,15
    6dc8:	mv	a4,a0
    6dcc:	bgeu	t1,a2,6e08 <memset+0x44>
    6dd0:	andi	a5,a4,15
    6dd4:	bnez	a5,6e74 <memset+0xb0>
    6dd8:	bnez	a1,6e5c <memset+0x98>
    6ddc:	andi	a3,a2,-16
    6de0:	andi	a2,a2,15
    6de4:	add	a3,a3,a4
    6de8:	sw	a1,0(a4)
    6dec:	sw	a1,4(a4)
    6df0:	sw	a1,8(a4)
    6df4:	sw	a1,12(a4)
    6df8:	addi	a4,a4,16
    6dfc:	bltu	a4,a3,6de8 <memset+0x24>
    6e00:	bnez	a2,6e08 <memset+0x44>
    6e04:	ret
    6e08:	sub	a3,t1,a2
    6e0c:	slli	a3,a3,0x2
    6e10:	auipc	t0,0x0
    6e14:	add	a3,a3,t0
    6e18:	jr	12(a3)
    6e1c:	sb	a1,14(a4)
    6e20:	sb	a1,13(a4)
    6e24:	sb	a1,12(a4)
    6e28:	sb	a1,11(a4)
    6e2c:	sb	a1,10(a4)
    6e30:	sb	a1,9(a4)
    6e34:	sb	a1,8(a4)
    6e38:	sb	a1,7(a4)
    6e3c:	sb	a1,6(a4)
    6e40:	sb	a1,5(a4)
    6e44:	sb	a1,4(a4)
    6e48:	sb	a1,3(a4)
    6e4c:	sb	a1,2(a4)
    6e50:	sb	a1,1(a4)
    6e54:	sb	a1,0(a4)
    6e58:	ret
    6e5c:	zext.b	a1,a1
    6e60:	slli	a3,a1,0x8
    6e64:	or	a1,a1,a3
    6e68:	slli	a3,a1,0x10
    6e6c:	or	a1,a1,a3
    6e70:	j	6ddc <memset+0x18>
    6e74:	slli	a3,a5,0x2
    6e78:	auipc	t0,0x0
    6e7c:	add	a3,a3,t0
    6e80:	mv	t0,ra
    6e84:	jalr	-96(a3)
    6e88:	mv	ra,t0
    6e8c:	addi	a5,a5,-16
    6e90:	sub	a4,a4,a5
    6e94:	add	a2,a2,a5
    6e98:	bgeu	t1,a2,6e08 <memset+0x44>
    6e9c:	j	6dd8 <memset+0x14>
