
/tmp/perf_overhead_artifacts/wc/PACK_ISOLATE/unpack.elf:     file format elf32-littleriscv


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
    6130:	jal	6cb8 <memset>
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
    65b8:	addi	sp,sp,-48
    65bc:	sw	s1,36(sp)
#endif
} // namespace detail

__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    std::uint32_t n = detail::next_zone_id;
    65c0:	addi	s1,gp,-1984 # ffb00040 <llk_perf::detail::next_zone_id>
    65c4:	lw	a5,0(s1) # ffb00000 <llk_profiler::buffer>
    65c8:	sw	ra,44(sp)
    65cc:	sw	s0,40(sp)
    65d0:	sw	s2,32(sp)
    65d4:	sw	s3,28(sp)
    65d8:	sw	s4,24(sp)
    65dc:	sw	s5,20(sp)
    for (std::uint32_t i = 0; i < n; ++i)
    65e0:	beqz	a5,6b98 <run_kernel(RuntimeParams const&)+0x5e0>
    {
        if (detail::zone_hashes[i] == hash_val)
    65e4:	lui	a4,0x7c867
    65e8:	lw	a3,4(s1)
    65ec:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    65f0:	beq	a3,a4,6674 <run_kernel(RuntimeParams const&)+0xbc>
    for (std::uint32_t i = 0; i < n; ++i)
    65f4:	li	a3,1
    65f8:	beq	a5,a3,6b98 <run_kernel(RuntimeParams const&)+0x5e0>
        if (detail::zone_hashes[i] == hash_val)
    65fc:	lw	a3,8(s1)
    6600:	beq	a3,a4,6674 <run_kernel(RuntimeParams const&)+0xbc>
    for (std::uint32_t i = 0; i < n; ++i)
    6604:	li	a3,2
    6608:	beq	a5,a3,6b98 <run_kernel(RuntimeParams const&)+0x5e0>
        if (detail::zone_hashes[i] == hash_val)
    660c:	lw	a3,12(s1)
    6610:	beq	a3,a4,6674 <run_kernel(RuntimeParams const&)+0xbc>
    for (std::uint32_t i = 0; i < n; ++i)
    6614:	li	a3,3
    6618:	beq	a5,a3,6b98 <run_kernel(RuntimeParams const&)+0x5e0>
        if (detail::zone_hashes[i] == hash_val)
    661c:	lw	a3,16(s1)
    6620:	beq	a3,a4,6674 <run_kernel(RuntimeParams const&)+0xbc>
    for (std::uint32_t i = 0; i < n; ++i)
    6624:	li	a3,4
    6628:	beq	a5,a3,6b98 <run_kernel(RuntimeParams const&)+0x5e0>
        if (detail::zone_hashes[i] == hash_val)
    662c:	lw	a3,20(s1)
    6630:	beq	a3,a4,6674 <run_kernel(RuntimeParams const&)+0xbc>
    for (std::uint32_t i = 0; i < n; ++i)
    6634:	li	a4,5
    6638:	beq	a5,a4,6b98 <run_kernel(RuntimeParams const&)+0x5e0>
        if (detail::zone_hashes[i] == hash_val)
    663c:	lui	a4,0x7c867
    6640:	lw	a3,24(s1)
    6644:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    6648:	beq	a3,a4,6674 <run_kernel(RuntimeParams const&)+0xbc>
    for (std::uint32_t i = 0; i < n; ++i)
    664c:	li	a3,6
    6650:	beq	a5,a3,6b98 <run_kernel(RuntimeParams const&)+0x5e0>
        if (detail::zone_hashes[i] == hash_val)
    6654:	lw	a3,28(s1)
    6658:	beq	a3,a4,6674 <run_kernel(RuntimeParams const&)+0xbc>
    for (std::uint32_t i = 0; i < n; ++i)
    665c:	li	a3,7
    6660:	beq	a5,a3,6b98 <run_kernel(RuntimeParams const&)+0x5e0>
        if (detail::zone_hashes[i] == hash_val)
    6664:	lw	a3,32(s1)
    6668:	beq	a3,a4,6674 <run_kernel(RuntimeParams const&)+0xbc>
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
    666c:	li	a4,8
    6670:	bne	a5,a4,6b98 <run_kernel(RuntimeParams const&)+0x5e0>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6674:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6678:	lw	a5,32(a4)
                ckernel::semaphore_post(PERF_ENTRY_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    667c:	zext.b	a5,a5
    6680:	bnez	a5,6698 <run_kernel(RuntimeParams const&)+0xe0>
            {
                asm volatile("nop");
    6684:	nop
    6688:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    668c:	lw	a5,32(a4)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    6690:	zext.b	a5,a5
    6694:	beqz	a5,6684 <run_kernel(RuntimeParams const&)+0xcc>
    6698:	lw	a5,32(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    669c:	zext.b	a5,a5
    66a0:	beqz	a5,6bd8 <run_kernel(RuntimeParams const&)+0x620>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    66a4:	li	a2,1
    66a8:	sw	a2,32(a4)
    {
    66ac:	sb	zero,0(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    66b0:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    66b4:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    66b8:	li	a1,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    66bc:	add	a5,a4,a3
    66c0:	addi	a5,a5,-1021
        if (!is_buffer_full())
    66c4:	bgeu	a1,a5,670c <run_kernel(RuntimeParams const&)+0x154>
    return p_reg[0];
    66c8:	lui	a5,0xffb12
    66cc:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    66d0:	lw	a5,504(a5)
            is_opened = true;
    66d4:	sb	a2,0(sp)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    66d8:	lui	a2,0xffb00
    66dc:	lw	a2,0(a2) # ffb00000 <llk_profiler::buffer>
    66e0:	lui	a0,0xaa3a0
    66e4:	slli	a5,a5,0x14
    66e8:	srli	a5,a5,0x14
    66ec:	or	a5,a5,a0
            ++open_zone_cnt;
    66f0:	addi	a3,a3,1
    66f4:	sh2add	a2,a4,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    66f8:	sw	a5,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    66fc:	addi	a4,a4,2
            ++open_zone_cnt;
    6700:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6704:	sw	a1,4(a2)
    6708:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    const std::uint32_t unpB_num_faces  = 4)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
    670c:	li	a1,6
    6710:	mv	a0,a1
    6714:	li	a2,1
    6718:	jal	62dc <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)>
    671c:	beqz	a0,6be4 <run_kernel(RuntimeParams const&)+0x62c>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6720:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6724:	lw	a5,52(a4)
    while (semaphore_read(semaphore::UNPACK_SYNC) > 0)
    6728:	zext.b	a5,a5
    672c:	bnez	a5,6724 <run_kernel(RuntimeParams const&)+0x16c>
    TTI_SETADCXY(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1011);
    6730:	ttsetadcxy	3,0,0,0,0,11
    TTI_SETADCZW(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1111);
    6734:	ttsetadczw	3,0,0,0,0,15
    if (cfg_state_id == 0)
    6738:	lw	a4,-2000(gp) # ffb00030 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    673c:	lui	a5,0xffef0
    if (cfg_state_id == 0)
    6740:	beqz	a4,6748 <run_kernel(RuntimeParams const&)+0x190>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
    6744:	addi	a5,a5,896 # ffef0380 <__instrn_buffer+0xb0380>
    std::uint32_t unpA_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpA_ch1_x_stride;
    std::uint32_t unpB_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpB_ch1_x_stride;
    std::uint32_t exp_width         = (static_cast<std::uint32_t>(unpA_dst_format_masked) >> 2) & 0x1; // 0=5-bit, 1=8-bit

    // Strides for incrementing ch1 address to srcA and srcB
    cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
    6748:	li	a1,256
    674c:	sw	a1,228(a5)
        (0 << UNP0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
        (unpA_ch1_z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT); // Z and W(not used) stride for dest address (ch1)

    cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
    6750:	sw	a1,236(a5)
    TTI_ATGETM(index);
    6754:	ttatgetm	0
    std::uint8_t mask_b0 = Mask & 0xff;

    if (mask_b0 != 0)
    {
        std::uint8_t data_b0 = wrdata & 0xff;
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    6758:	lui	s2,0xffe40
    675c:	mv	s2,s2
    6760:	lui	a4,0xb3ff0
    6764:	sw	a4,0(s2) # ffe40000 <__instrn_buffer>
    std::uint8_t mask_b1 = (Mask >> 8) & 0xff;

    if (mask_b1 != 0)
    {
        std::uint8_t data_b1 = (wrdata) & 0xff;
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    6768:	lui	a4,0xb47f0
    676c:	sw	a4,0(s2)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    6770:	lui	a4,0xb3070
    6774:	addi	a4,a4,1 # b3070001 <__runtime_args_end+0xb304fc01>
    6778:	sw	a4,0(s2)
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    677c:	lui	a4,0xb4800
    6780:	addi	a4,a4,1 # b4800001 <__runtime_args_end+0xb47dfc01>
    6784:	sw	a4,0(s2)
    std::uint8_t mask_b2 = (Mask >> 16) & 0xff;

    if (mask_b2 != 0)
    {
        std::uint8_t data_b2 = (wrdata) & 0xff;
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
    6788:	lui	a4,0xb5010
    678c:	addi	a4,a4,1 # b5010001 <__runtime_args_end+0xb4fefc01>
    6790:	sw	a4,0(s2)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    6794:	lui	a4,0xb3010
    6798:	addi	a4,a4,2 # b3010002 <__runtime_args_end+0xb2fefc02>
    679c:	sw	a4,0(s2)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
    67a0:	lui	a4,0xb5400
    67a4:	addi	a3,a4,71 # b5400047 <__runtime_args_end+0xb53dfc47>
    67a8:	sw	a3,0(s2)
    67ac:	addi	a4,a4,119
    67b0:	sw	a4,0(s2)
    TTI_ATRELM(index);
    67b4:	ttatrelm	0
    tile_descriptor.f.z_dim          = unpA_num_faces;
    // tile_descriptor.f.blobs_per_xy_plane = 0;
    // tile_descriptor.f.blobs_y_start = 0;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67b8:	li	a4,22
    67bc:	sw	a4,256(a5)
    67c0:	lui	a4,0x40
    67c4:	addi	a4,a4,1 # 40001 <__runtime_args_end+0x1fc01>
    tile_descriptor.f.in_data_format = row_pool ? to_underlying(DataFormat::Float32) : unpB_src_format_masked;
    tile_descriptor.f.x_dim          = unpB_face_r_dim * FACE_C_DIM;
    tile_descriptor.f.z_dim          = unpB_num_faces;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67c8:	lui	a3,0x1000
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67cc:	sw	a4,260(a5)
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    67d0:	addi	a2,a3,22 # 1000016 <__runtime_args_end+0xfdfc16>
    67d4:	sw	a2,448(a5)
    67d8:	sw	a4,452(a5)
    config.f.uncompress_cntx4_7 = 0xf;
    // config.f.limit_addr = 0; // Set dynamically
    // config.f.fifo_size = 0; // Set dynamically
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    67dc:	li	a2,38
    67e0:	lui	a4,0xf0
    67e4:	sw	a2,288(a5)
    67e8:	addi	a4,a4,15 # f000f <__runtime_args_end+0xcfc0f>
    67ec:	sw	a4,292(a5)
    config.f.out_data_format = row_pool ? (to_underlying(DataFormat::Float16) | (exp_width << 2)) : unpB_dst_format_masked;
    config.f.haloize_mode    = 0;

    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC1_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    67f0:	sw	a2,480(a5)
    67f4:	sw	a4,484(a5)
    }

    std::uint32_t unpA_x_end = (unpA_face_r_dim == 0) ? 1 : (unpA_face_r_dim << 4) - 1;
    TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    67f8:	lui	a4,0x5e240
    67fc:	addi	a4,a4,-1024 # 5e23fc00 <__runtime_args_end+0x5e21f800>
    6800:	sw	a4,0(s2)
    TT_SETADCXX(p_setadc::UNP_B, (unpB_face_r_dim << 4) - 1, 0x0);
    6804:	lui	a4,0x5e440
    6808:	addi	a4,a4,-1024 # 5e43fc00 <__runtime_args_end+0x5e41f800>

    // Program base address for all 2 sections (each section address is loaded to corresponding context)
    // Load dummy data to unused location if face height is 0
    const std::uint32_t Dest_cntx0_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    const std::uint32_t Dest_cntx1_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);
    680c:	lui	a2,0x400
    TT_SETADCXX(p_setadc::UNP_B, (unpB_face_r_dim << 4) - 1, 0x0);
    6810:	sw	a4,0(s2)
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);
    6814:	addi	a2,a2,64 # 400040 <__runtime_args_end+0x3dfc40>
    6818:	sw	a2,336(a5)
    // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
    const std::uint32_t face_dim                 = unpA_face_r_dim * FACE_C_DIM;
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = face_dim | (face_dim << 16);

    constexpr std::uint32_t face_dim_16x16 = FACE_R_DIM * FACE_C_DIM;
    regfile[p_gpr_unpack::FACE_DIM_16x16]  = (face_dim_16x16 / 1) | ((face_dim_16x16 / 1) << 16);
    681c:	lw	a4,-2040(gp) # ffb00008 <ckernel::regfile>
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = face_dim | (face_dim << 16);
    6820:	add	a3,a3,a1
    6824:	sw	a3,344(a5)
    regfile[p_gpr_unpack::FACE_DIM_8x16]   = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    6828:	lui	a0,0x800
    regfile[p_gpr_unpack::FACE_DIM_16x16]  = (face_dim_16x16 / 1) | ((face_dim_16x16 / 1) << 16);
    682c:	sw	a3,160(a4)
    regfile[p_gpr_unpack::FACE_DIM_8x16]   = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    6830:	addi	a3,a0,128 # 800080 <__runtime_args_end+0x7dfc80>
    6834:	sw	a3,164(a4)
    regfile[p_gpr_unpack::FACE_DIM_4x16]   = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    6838:	lui	a3,0x200
    regfile[p_gpr_unpack::FACE_DIM_4x16]   = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    683c:	sw	a2,168(a4)
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    6840:	addi	a2,a3,32 # 200020 <__runtime_args_end+0x1dfc20>
    regfile[p_gpr_unpack::FACE_DIM_1x16]   = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    6844:	lui	a3,0x100
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    6848:	sw	a2,172(a4)
    regfile[p_gpr_unpack::FACE_DIM_1x16]   = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    684c:	addi	a3,a3,16 # 100010 <__runtime_args_end+0xdfc10>
    6850:	sw	a3,176(a4)
    volatile std::uint32_t foo     = 0x0;
    6854:	sw	zero,8(sp)
    *fooptr                        = regfile[index];
    6858:	lw	a4,176(a4)
    685c:	sw	a4,8(sp)
    sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);

    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);
    6860:	ttsetc16	5,4

    // Enable address counter for unpacker ch1/dst address
    // final address is calculated as: Dest_cntx0/1_address + address_counter_ch1
    // used for face by face unpacking of entire tile into srcA
    cfg[UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32] = 0x1 << UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT;
    6864:	sw	a1,200(a5)
    unp_cfg_context = 0;
    6868:	sw	zero,-1988(gp) # ffb0003c <unp_cfg_context>
    TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    686c:	ttsetc16	41,0
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");
    configure_unpack_AB<is_fp32_dest_acc_en, false, false, false, disable_src_zero_flag>(
        unpA_src_format, unpB_src_format, unpA_dst_format, unpB_dst_format, unpA_face_r_dim, unpB_face_r_dim, 0, unpA_num_faces, unpB_num_faces);

    TT_SETDMAREG(0, LOWER_HALFWORD(unpA_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    6870:	lui	a5,0x45000
    6874:	addi	a3,a5,72 # 45000048 <__runtime_args_end+0x44fdfc48>
    6878:	sw	a3,0(s2)
    687c:	lui	a4,0x2021
    TT_SETDMAREG(0, LOWER_HALFWORD(unpB_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
    6880:	addi	a5,a5,74
    6884:	addi	a4,a4,16 # 2021010 <__runtime_args_end+0x2000c10>
    6888:	sw	a5,0(s2)
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    688c:	addi	a0,sp,4
    6890:	sw	a4,4(sp)
    6894:	jal	627c <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>
    6898:	beqz	a0,6bf0 <run_kernel(RuntimeParams const&)+0x638>
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    689c:	lui	a5,0xb4010
    68a0:	addi	a5,a5,72 # b4010048 <__runtime_args_end+0xb3fefc48>
    68a4:	sw	a5,0(s2)
            break;
        case 8:
            TTI_SETADCXX(UNP_SEL, 8 * FACE_C_DIM - 1, 0x0);
            break;
        default:
            TTI_SETADCXX(UNP_SEL, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    68a8:	ttsetadcxx	3,255,0
    68ac:	lw	a5,4(sp)
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    68b0:	addi	a0,sp,12
    68b4:	srli	s5,a5,0x10
    68b8:	sw	a5,12(sp)
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    68bc:	zext.b	s5,s5
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
    68c0:	srli	s2,a5,0x18
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    68c4:	jal	627c <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>
    68c8:	beqz	a0,6bd0 <run_kernel(RuntimeParams const&)+0x618>
    store_blocking(&pc_buf_base[2], 0);
    68cc:	lw	a5,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
    68d0:	li	a3,0
    store_blocking(&pc_buf_base[2], 0);
    68d4:	addi	a5,a5,8
    asm volatile(
    68d8:	sw	a3,0(a5)
    68dc:	lw	a3,0(a5)
    68e0:	and	zero,zero,a3
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
    68e4:	lui	a5,0xffb80
    68e8:	sw	s5,0(a5) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
    68ec:	sw	s2,4(a5)
    mop_cfg[2] = m_start_op0;
    68f0:	lui	a3,0x2000
    68f4:	sw	a3,8(a5)
    mop_cfg[3] = m_end_op0;
    68f8:	sw	a3,12(a5)
    mop_cfg[4] = m_end_op1;
    mop_cfg[5] = m_loop_op0;
    68fc:	lui	a2,0x42008
    mop_cfg[4] = m_end_op1;
    6900:	sw	a3,16(a5)
    mop_cfg[5] = m_loop_op0;
    6904:	addi	a2,a2,193 # 420080c1 <__runtime_args_end+0x41fe7cc1>
    mop_cfg[6] = m_loop_op1;
    6908:	lui	a3,0x42808
    mop_cfg[5] = m_loop_op0;
    690c:	sw	a2,20(a5)
    mop_cfg[6] = m_loop_op1;
    6910:	addi	a3,a3,193 # 428080c1 <__runtime_args_end+0x427e7cc1>
    6914:	sw	a3,24(a5)
    mop_cfg[7] = m_loop0_last_instr;
    6918:	sw	a3,28(a5)
    mop_cfg[8] = m_loop1_last_instr;
    691c:	sw	a3,32(a5)
    store_blocking(&pc_buf_base[1], 0);
    6920:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
    6924:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    6928:	addi	a4,a3,4
    asm volatile(
    692c:	sw	a5,0(a4)
    6930:	lw	a5,0(a4)
    6934:	and	zero,zero,a5
        if (is_opened)
    6938:	lbu	a5,0(sp)
    693c:	beqz	a5,6988 <run_kernel(RuntimeParams const&)+0x3d0>
    return p_reg[0];
    6940:	lui	a5,0xffb12
    6944:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    6948:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    694c:	lui	a4,0xffb00
    6950:	lw	a3,0(a4) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    6954:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6958:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    695c:	lui	a0,0xba3a0
    6960:	slli	a5,a5,0x14
    6964:	srli	a5,a5,0x14
    6968:	or	a5,a5,a0
    696c:	sh2add	a3,a4,a3
    6970:	sw	a5,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6974:	addi	a4,a4,2
            --open_zone_cnt;
    6978:	addi	a5,a2,-1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    697c:	sw	a1,4(a3)
    6980:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    6984:	sw	a5,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6988:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    698c:	lw	a5,40(a4)
                ckernel::semaphore_post(PERF_EXIT_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    6990:	zext.b	a5,a5
    6994:	bnez	a5,69ac <run_kernel(RuntimeParams const&)+0x3f4>
            {
                asm volatile("nop");
    6998:	nop
    699c:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    69a0:	lw	a5,40(a4)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    69a4:	zext.b	a5,a5
    69a8:	beqz	a5,6998 <run_kernel(RuntimeParams const&)+0x3e0>
    69ac:	lw	a5,40(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    69b0:	zext.b	a5,a5
    69b4:	beqz	a5,6c2c <run_kernel(RuntimeParams const&)+0x674>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    69b8:	li	a3,1
    69bc:	sw	a3,40(a4)
    std::uint32_t n = detail::next_zone_id;
    69c0:	lw	a5,0(s1)
    for (std::uint32_t i = 0; i < n; ++i)
    69c4:	beqz	a5,6bb4 <run_kernel(RuntimeParams const&)+0x5fc>
        if (detail::zone_hashes[i] == hash_val)
    69c8:	lui	a4,0xbd77
    69cc:	lw	a2,4(s1)
    69d0:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    69d4:	beq	a2,a4,6a54 <run_kernel(RuntimeParams const&)+0x49c>
    for (std::uint32_t i = 0; i < n; ++i)
    69d8:	beq	a5,a3,6bb4 <run_kernel(RuntimeParams const&)+0x5fc>
        if (detail::zone_hashes[i] == hash_val)
    69dc:	lw	a3,8(s1)
    69e0:	beq	a3,a4,6a54 <run_kernel(RuntimeParams const&)+0x49c>
    for (std::uint32_t i = 0; i < n; ++i)
    69e4:	li	a3,2
    69e8:	beq	a5,a3,6bb4 <run_kernel(RuntimeParams const&)+0x5fc>
        if (detail::zone_hashes[i] == hash_val)
    69ec:	lw	a3,12(s1)
    69f0:	beq	a3,a4,6a54 <run_kernel(RuntimeParams const&)+0x49c>
    for (std::uint32_t i = 0; i < n; ++i)
    69f4:	li	a3,3
    69f8:	beq	a5,a3,6bb4 <run_kernel(RuntimeParams const&)+0x5fc>
        if (detail::zone_hashes[i] == hash_val)
    69fc:	lw	a3,16(s1)
    6a00:	beq	a3,a4,6a54 <run_kernel(RuntimeParams const&)+0x49c>
    for (std::uint32_t i = 0; i < n; ++i)
    6a04:	li	a3,4
    6a08:	beq	a5,a3,6bb4 <run_kernel(RuntimeParams const&)+0x5fc>
        if (detail::zone_hashes[i] == hash_val)
    6a0c:	lw	a3,20(s1)
    6a10:	beq	a3,a4,6a54 <run_kernel(RuntimeParams const&)+0x49c>
    for (std::uint32_t i = 0; i < n; ++i)
    6a14:	li	a4,5
    6a18:	beq	a5,a4,6bb4 <run_kernel(RuntimeParams const&)+0x5fc>
        if (detail::zone_hashes[i] == hash_val)
    6a1c:	lui	a4,0xbd77
    6a20:	lw	a3,24(s1)
    6a24:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    6a28:	beq	a3,a4,6a54 <run_kernel(RuntimeParams const&)+0x49c>
    for (std::uint32_t i = 0; i < n; ++i)
    6a2c:	li	a3,6
    6a30:	beq	a5,a3,6bb4 <run_kernel(RuntimeParams const&)+0x5fc>
        if (detail::zone_hashes[i] == hash_val)
    6a34:	lw	a3,28(s1)
    6a38:	beq	a3,a4,6a54 <run_kernel(RuntimeParams const&)+0x49c>
    for (std::uint32_t i = 0; i < n; ++i)
    6a3c:	li	a3,7
    6a40:	beq	a5,a3,6bb4 <run_kernel(RuntimeParams const&)+0x5fc>
        if (detail::zone_hashes[i] == hash_val)
    6a44:	lw	a3,32(s1)
    6a48:	beq	a3,a4,6a54 <run_kernel(RuntimeParams const&)+0x49c>
    if (n < PERF_COUNTERS_MAX_ZONES)
    6a4c:	li	a4,8
    6a50:	bne	a5,a4,6bb4 <run_kernel(RuntimeParams const&)+0x5fc>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6a54:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6a58:	lw	a5,32(a4)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    6a5c:	zext.b	a5,a5
    6a60:	bnez	a5,6a78 <run_kernel(RuntimeParams const&)+0x4c0>
                asm volatile("nop");
    6a64:	nop
    6a68:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6a6c:	lw	a5,32(a4)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    6a70:	zext.b	a5,a5
    6a74:	beqz	a5,6a64 <run_kernel(RuntimeParams const&)+0x4ac>
    6a78:	lw	a5,32(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6a7c:	zext.b	a5,a5
    6a80:	beqz	a5,6c20 <run_kernel(RuntimeParams const&)+0x668>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6a84:	li	a2,1
    6a88:	sw	a2,32(a4)
    {
    6a8c:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    6a90:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    6a94:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    6a98:	li	a1,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    6a9c:	add	a5,a4,a3
    6aa0:	addi	a5,a5,-1021
        if (!is_buffer_full())
    6aa4:	bgeu	a1,a5,6aec <run_kernel(RuntimeParams const&)+0x534>
    return p_reg[0];
    6aa8:	lui	a5,0xffb12
    6aac:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    6ab0:	lw	a5,504(a5)
            is_opened = true;
    6ab4:	sb	a2,12(sp)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6ab8:	lui	a2,0xffb00
    6abc:	lw	a2,0(a2) # ffb00000 <llk_profiler::buffer>
    6ac0:	lui	a0,0xa99a8
    6ac4:	slli	a5,a5,0x14
    6ac8:	srli	a5,a5,0x14
    6acc:	or	a5,a5,a0
            ++open_zone_cnt;
    6ad0:	addi	a3,a3,1
    6ad4:	sh2add	a2,a4,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6ad8:	sw	a5,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6adc:	addi	a4,a4,2
            ++open_zone_cnt;
    6ae0:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6ae4:	sw	a1,4(a2)
    6ae8:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
        if (is_opened)
    6aec:	lbu	a5,12(sp)
    6af0:	beqz	a5,6b3c <run_kernel(RuntimeParams const&)+0x584>
    6af4:	lui	a5,0xffb12
    6af8:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    6afc:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6b00:	lui	a4,0xffb00
    6b04:	lw	a3,0(a4) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    6b08:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6b0c:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    6b10:	lui	a0,0xb99a8
    6b14:	slli	a5,a5,0x14
    6b18:	srli	a5,a5,0x14
    6b1c:	or	a5,a5,a0
    6b20:	sh2add	a3,a4,a3
    6b24:	sw	a5,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6b28:	addi	a4,a4,2
            --open_zone_cnt;
    6b2c:	addi	a5,a2,-1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6b30:	sw	a1,4(a3)
    6b34:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    6b38:	sw	a5,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6b3c:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6b40:	lw	a5,40(a4)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    6b44:	zext.b	a5,a5
    6b48:	bnez	a5,6b60 <run_kernel(RuntimeParams const&)+0x5a8>
                asm volatile("nop");
    6b4c:	nop
    6b50:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6b54:	lw	a5,40(a4)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    6b58:	zext.b	a5,a5
    6b5c:	beqz	a5,6b4c <run_kernel(RuntimeParams const&)+0x594>
    6b60:	lw	a5,40(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6b64:	zext.b	a5,a5
    6b68:	beqz	a5,6c14 <run_kernel(RuntimeParams const&)+0x65c>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6b6c:	li	a5,1
    6b70:	sw	a5,40(a4)
                _llk_unpack_AB_<>(PERF_ADDRESS(PERF_INPUT_A, tile), PERF_ADDRESS(PERF_INPUT_B, tile));
            }
        }
        PROFILER_SYNC();
    }
}
    6b74:	lw	ra,44(sp)
    6b78:	lw	s0,40(sp)
    6b7c:	lw	s1,36(sp)
    6b80:	lw	s2,32(sp)
    6b84:	lw	s3,28(sp)
    6b88:	lw	s4,24(sp)
    6b8c:	lw	s5,20(sp)
    6b90:	addi	sp,sp,48
    6b94:	ret
        detail::zone_hashes[n] = hash_val;
    6b98:	lui	a4,0x7c867
    6b9c:	sh2add	a3,a5,s1
    6ba0:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
        detail::next_zone_id   = n + 1;
    6ba4:	addi	a5,a5,1
        detail::zone_hashes[n] = hash_val;
    6ba8:	sw	a4,4(a3)
        detail::next_zone_id   = n + 1;
    6bac:	sw	a5,0(s1)
        return n;
    6bb0:	j	6674 <run_kernel(RuntimeParams const&)+0xbc>
        detail::zone_hashes[n] = hash_val;
    6bb4:	lui	a4,0xbd77
    6bb8:	sh2add	a3,a5,s1
    6bbc:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
        detail::next_zone_id   = n + 1;
    6bc0:	addi	a5,a5,1
        detail::zone_hashes[n] = hash_val;
    6bc4:	sw	a4,4(a3)
        detail::next_zone_id   = n + 1;
    6bc8:	sw	a5,0(s1)
        return n;
    6bcc:	j	6a54 <run_kernel(RuntimeParams const&)+0x49c>
    6bd0:	ebreak
    6bd4:	j	68cc <run_kernel(RuntimeParams const&)+0x314>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6bd8:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6bdc:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6be0:	j	66a4 <run_kernel(RuntimeParams const&)+0xec>
    LLK_ASSERT(
    6be4:	ebreak
    LLK_ASSERT(
    6be8:	ebreak
    6bec:	j	6720 <run_kernel(RuntimeParams const&)+0x168>
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    6bf0:	ebreak
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    6bf4:	lui	a4,0xb4010
    6bf8:	addi	a4,a4,72 # b4010048 <__runtime_args_end+0xb3fefc48>
    const bool within_face_16x16_transpose = transpose == ckernel::Transpose::IntraFace || transpose == ckernel::Transpose::Both;
    const bool transpose_of_faces          = transpose == ckernel::Transpose::InterFace || transpose == ckernel::Transpose::Both;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose); // transpose within the face

    config_unpacker_x_end<p_setadc::UNP_AB>(tensor_shape.face_r_dim);
    6bfc:	lbu	a5,4(sp)
    6c00:	sw	a4,0(s2)
    LLK_ASSERT(
    6c04:	li	a4,16
    6c08:	bgeu	a4,a5,6c38 <run_kernel(RuntimeParams const&)+0x680>
    6c0c:	ebreak
    6c10:	j	68a8 <run_kernel(RuntimeParams const&)+0x2f0>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6c14:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6c18:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6c1c:	j	6b6c <run_kernel(RuntimeParams const&)+0x5b4>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6c20:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6c24:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6c28:	j	6a84 <run_kernel(RuntimeParams const&)+0x4cc>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    6c2c:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    6c30:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6c34:	j	69b8 <run_kernel(RuntimeParams const&)+0x400>
    6c38:	lui	a4,0xffff0
    6c3c:	addi	a4,a4,-279 # fffefee9 <__instrn_buffer+0x1afee9>
    6c40:	sra	a4,a4,a5
    6c44:	andi	a4,a4,1
    6c48:	bnez	a4,6c80 <run_kernel(RuntimeParams const&)+0x6c8>
    switch (face_r_dim)
    6c4c:	li	a4,4
    6c50:	beq	a5,a4,6c70 <run_kernel(RuntimeParams const&)+0x6b8>
    6c54:	bltu	a4,a5,6c88 <run_kernel(RuntimeParams const&)+0x6d0>
    6c58:	li	a4,1
    6c5c:	beq	a5,a4,6c78 <run_kernel(RuntimeParams const&)+0x6c0>
    6c60:	li	a4,2
    6c64:	bne	a5,a4,68a8 <run_kernel(RuntimeParams const&)+0x2f0>
            TTI_SETADCXX(UNP_SEL, 2 * FACE_C_DIM - 1, 0x0);
    6c68:	ttsetadcxx	3,31,0
            break;
    6c6c:	j	68ac <run_kernel(RuntimeParams const&)+0x2f4>
            TTI_SETADCXX(UNP_SEL, 4 * FACE_C_DIM - 1, 0x0);
    6c70:	ttsetadcxx	3,63,0
            break;
    6c74:	j	68ac <run_kernel(RuntimeParams const&)+0x2f4>
            TTI_SETADCXX(UNP_SEL, 1 * FACE_C_DIM - 1, 0x0);
    6c78:	ttsetadcxx	3,15,0
            break;
    6c7c:	j	68ac <run_kernel(RuntimeParams const&)+0x2f4>
    LLK_ASSERT(
    6c80:	ebreak
    6c84:	j	6c4c <run_kernel(RuntimeParams const&)+0x694>
    switch (face_r_dim)
    6c88:	li	a4,8
    6c8c:	bne	a5,a4,68a8 <run_kernel(RuntimeParams const&)+0x2f0>
            TTI_SETADCXX(UNP_SEL, 8 * FACE_C_DIM - 1, 0x0);
    6c90:	ttsetadcxx	3,127,0
            break;
    6c94:	j	68ac <run_kernel(RuntimeParams const&)+0x2f4>

00006c98 <_init()>:
    }
}

void _init(void)
{
}
    6c98:	ret

00006c9c <_fini()>:

void _fini(void)
    6c9c:	ret

00006ca0 <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
    6ca0:	lui	a5,0x20
    6ca4:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
    6ca8:	sb	a5,0(a0) # b99a8000 <__runtime_args_end+0xb9987c00>
        (void)(dstc[i]);
    6cac:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
    6cb0:	fence
}
    6cb4:	ret

00006cb8 <memset>:
    6cb8:	li	t1,15
    6cbc:	mv	a4,a0
    6cc0:	bgeu	t1,a2,6cfc <memset+0x44>
    6cc4:	andi	a5,a4,15
    6cc8:	bnez	a5,6d68 <memset+0xb0>
    6ccc:	bnez	a1,6d50 <memset+0x98>
    6cd0:	andi	a3,a2,-16
    6cd4:	andi	a2,a2,15
    6cd8:	add	a3,a3,a4
    6cdc:	sw	a1,0(a4)
    6ce0:	sw	a1,4(a4)
    6ce4:	sw	a1,8(a4)
    6ce8:	sw	a1,12(a4)
    6cec:	addi	a4,a4,16
    6cf0:	bltu	a4,a3,6cdc <memset+0x24>
    6cf4:	bnez	a2,6cfc <memset+0x44>
    6cf8:	ret
    6cfc:	sub	a3,t1,a2
    6d00:	slli	a3,a3,0x2
    6d04:	auipc	t0,0x0
    6d08:	add	a3,a3,t0
    6d0c:	jr	12(a3)
    6d10:	sb	a1,14(a4)
    6d14:	sb	a1,13(a4)
    6d18:	sb	a1,12(a4)
    6d1c:	sb	a1,11(a4)
    6d20:	sb	a1,10(a4)
    6d24:	sb	a1,9(a4)
    6d28:	sb	a1,8(a4)
    6d2c:	sb	a1,7(a4)
    6d30:	sb	a1,6(a4)
    6d34:	sb	a1,5(a4)
    6d38:	sb	a1,4(a4)
    6d3c:	sb	a1,3(a4)
    6d40:	sb	a1,2(a4)
    6d44:	sb	a1,1(a4)
    6d48:	sb	a1,0(a4)
    6d4c:	ret
    6d50:	zext.b	a1,a1
    6d54:	slli	a3,a1,0x8
    6d58:	or	a1,a1,a3
    6d5c:	slli	a3,a1,0x10
    6d60:	or	a1,a1,a3
    6d64:	j	6cd0 <memset+0x18>
    6d68:	slli	a3,a5,0x2
    6d6c:	auipc	t0,0x0
    6d70:	add	a3,a3,t0
    6d74:	mv	t0,ra
    6d78:	jalr	-96(a3)
    6d7c:	mv	ra,t0
    6d80:	addi	a5,a5,-16
    6d84:	sub	a4,a4,a5
    6d88:	add	a2,a2,a5
    6d8c:	bgeu	t1,a2,6cfc <memset+0x44>
    6d90:	j	6ccc <memset+0x14>
