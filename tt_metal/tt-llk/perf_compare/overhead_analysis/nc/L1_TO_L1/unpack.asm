
/tmp/perf_overhead_artifacts/nc/L1_TO_L1/unpack.elf:     file format elf32-littleriscv


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
    601c:	addi	a4,a4,64 # ffb00040 <__fw_export_ldm_end>
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
    6130:	jal	6b4c <memset>
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
    65bc:	sw	ra,44(sp)
    65c0:	sw	s0,40(sp)
    65c4:	sw	s1,36(sp)
    65c8:	sw	s2,32(sp)
    65cc:	sw	s3,28(sp)
    65d0:	sw	s4,24(sp)
    65d4:	sw	s5,20(sp)
    65d8:	sw	s6,16(sp)
    {
    65dc:	sb	zero,0(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    65e0:	lw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    65e4:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    65e8:	li	a2,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    65ec:	add	a4,a5,a3
    65f0:	addi	a4,a4,-1021
        if (!is_buffer_full())
    65f4:	bgeu	a2,a4,6640 <run_kernel(RuntimeParams const&)+0x88>
    65f8:	lui	a4,0xffb12
    65fc:	lw	a1,496(a4) # ffb121f0 <__stack_top+0x111f0>
    6600:	lw	a4,504(a4)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6604:	lui	a2,0xffb00
    6608:	lw	a2,0(a2) # ffb00000 <llk_profiler::buffer>
    660c:	lui	a0,0xaa3a0
    6610:	slli	a4,a4,0x14
    6614:	srli	a4,a4,0x14
    6618:	or	a4,a4,a0
            ++open_zone_cnt;
    661c:	addi	a3,a3,1
    6620:	sh2add	a2,a5,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6624:	sw	a4,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6628:	addi	a5,a5,2
            is_opened = true;
    662c:	li	a4,1
            ++open_zone_cnt;
    6630:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6634:	sw	a1,4(a2)
    6638:	sw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            is_opened = true;
    663c:	sb	a4,0(sp)
    const std::uint32_t unpB_num_faces  = 4)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
    6640:	li	a1,6
    6644:	mv	a0,a1
    6648:	li	a2,1
    664c:	jal	62dc <ckernel::unpacker::is_unpacker_format_conversion_supported_fp32_acc(DataFormat, DataFormat, bool)>
    6650:	beqz	a0,6af8 <run_kernel(RuntimeParams const&)+0x540>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6654:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    6658:	lw	a5,52(a4)
    while (semaphore_read(semaphore::UNPACK_SYNC) > 0)
    665c:	zext.b	a5,a5
    6660:	bnez	a5,6658 <run_kernel(RuntimeParams const&)+0xa0>
    TTI_SETADCXY(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1011);
    6664:	ttsetadcxy	3,0,0,0,0,11
    TTI_SETADCZW(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1111);
    6668:	ttsetadczw	3,0,0,0,0,15
    if (cfg_state_id == 0)
    666c:	lw	a4,-2000(gp) # ffb00030 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    6670:	lui	a5,0xffef0
    if (cfg_state_id == 0)
    6674:	beqz	a4,667c <run_kernel(RuntimeParams const&)+0xc4>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
    6678:	addi	a5,a5,896 # ffef0380 <__instrn_buffer+0xb0380>
    std::uint32_t unpA_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpA_ch1_x_stride;
    std::uint32_t unpB_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpB_ch1_x_stride;
    std::uint32_t exp_width         = (static_cast<std::uint32_t>(unpA_dst_format_masked) >> 2) & 0x1; // 0=5-bit, 1=8-bit

    // Strides for incrementing ch1 address to srcA and srcB
    cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
    667c:	li	a1,256
    6680:	sw	a1,228(a5)
        (0 << UNP0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
        (unpA_ch1_z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT); // Z and W(not used) stride for dest address (ch1)

    cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
    6684:	sw	a1,236(a5)
    TTI_ATGETM(index);
    6688:	ttatgetm	0
    std::uint8_t mask_b0 = Mask & 0xff;

    if (mask_b0 != 0)
    {
        std::uint8_t data_b0 = wrdata & 0xff;
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    668c:	lui	s5,0xffe40
    6690:	mv	s5,s5
    6694:	lui	a4,0xb3ff0
    6698:	sw	a4,0(s5) # ffe40000 <__instrn_buffer>
    std::uint8_t mask_b1 = (Mask >> 8) & 0xff;

    if (mask_b1 != 0)
    {
        std::uint8_t data_b1 = (wrdata) & 0xff;
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    669c:	lui	a4,0xb47f0
    66a0:	sw	a4,0(s5)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    66a4:	lui	a4,0xb3070
    66a8:	addi	a4,a4,1 # b3070001 <__runtime_args_end+0xb304fc01>
    66ac:	sw	a4,0(s5)
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    66b0:	lui	a4,0xb4800
    66b4:	addi	a4,a4,1 # b4800001 <__runtime_args_end+0xb47dfc01>
    66b8:	sw	a4,0(s5)
    std::uint8_t mask_b2 = (Mask >> 16) & 0xff;

    if (mask_b2 != 0)
    {
        std::uint8_t data_b2 = (wrdata) & 0xff;
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
    66bc:	lui	a4,0xb5010
    66c0:	addi	a4,a4,1 # b5010001 <__runtime_args_end+0xb4fefc01>
    66c4:	sw	a4,0(s5)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    66c8:	lui	a4,0xb3010
    66cc:	addi	a4,a4,2 # b3010002 <__runtime_args_end+0xb2fefc02>
    66d0:	sw	a4,0(s5)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
    66d4:	lui	a4,0xb5400
    66d8:	addi	a3,a4,71 # b5400047 <__runtime_args_end+0xb53dfc47>
    66dc:	sw	a3,0(s5)
    66e0:	addi	a4,a4,119
    66e4:	sw	a4,0(s5)
    TTI_ATRELM(index);
    66e8:	ttatrelm	0
    tile_descriptor.f.z_dim          = unpA_num_faces;
    // tile_descriptor.f.blobs_per_xy_plane = 0;
    // tile_descriptor.f.blobs_y_start = 0;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    66ec:	li	a4,22
    66f0:	sw	a4,256(a5)
    66f4:	lui	a4,0x40
    66f8:	addi	a4,a4,1 # 40001 <__runtime_args_end+0x1fc01>
    tile_descriptor.f.in_data_format = row_pool ? to_underlying(DataFormat::Float32) : unpB_src_format_masked;
    tile_descriptor.f.x_dim          = unpB_face_r_dim * FACE_C_DIM;
    tile_descriptor.f.z_dim          = unpB_num_faces;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    66fc:	lui	a3,0x1000
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    6700:	sw	a4,260(a5)
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    6704:	addi	a2,a3,22 # 1000016 <__runtime_args_end+0xfdfc16>
    6708:	sw	a2,448(a5)
    670c:	sw	a4,452(a5)
    config.f.uncompress_cntx4_7 = 0xf;
    // config.f.limit_addr = 0; // Set dynamically
    // config.f.fifo_size = 0; // Set dynamically
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    6710:	li	a2,38
    6714:	lui	a4,0xf0
    6718:	sw	a2,288(a5)
    671c:	addi	a4,a4,15 # f000f <__runtime_args_end+0xcfc0f>
    6720:	sw	a4,292(a5)
    config.f.out_data_format = row_pool ? (to_underlying(DataFormat::Float16) | (exp_width << 2)) : unpB_dst_format_masked;
    config.f.haloize_mode    = 0;

    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC1_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    6724:	sw	a2,480(a5)
    6728:	sw	a4,484(a5)
    }

    std::uint32_t unpA_x_end = (unpA_face_r_dim == 0) ? 1 : (unpA_face_r_dim << 4) - 1;
    TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    672c:	lui	a4,0x5e240
    6730:	addi	a4,a4,-1024 # 5e23fc00 <__runtime_args_end+0x5e21f800>
    6734:	sw	a4,0(s5)
    TT_SETADCXX(p_setadc::UNP_B, (unpB_face_r_dim << 4) - 1, 0x0);
    6738:	lui	a4,0x5e440
    673c:	addi	a4,a4,-1024 # 5e43fc00 <__runtime_args_end+0x5e41f800>

    // Program base address for all 2 sections (each section address is loaded to corresponding context)
    // Load dummy data to unused location if face height is 0
    const std::uint32_t Dest_cntx0_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    const std::uint32_t Dest_cntx1_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);
    6740:	lui	a2,0x400
    TT_SETADCXX(p_setadc::UNP_B, (unpB_face_r_dim << 4) - 1, 0x0);
    6744:	sw	a4,0(s5)
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);
    6748:	addi	a2,a2,64 # 400040 <__runtime_args_end+0x3dfc40>
    674c:	sw	a2,336(a5)
    // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
    const std::uint32_t face_dim                 = unpA_face_r_dim * FACE_C_DIM;
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = face_dim | (face_dim << 16);

    constexpr std::uint32_t face_dim_16x16 = FACE_R_DIM * FACE_C_DIM;
    regfile[p_gpr_unpack::FACE_DIM_16x16]  = (face_dim_16x16 / 1) | ((face_dim_16x16 / 1) << 16);
    6750:	lw	a4,-2040(gp) # ffb00008 <ckernel::regfile>
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = face_dim | (face_dim << 16);
    6754:	add	a3,a3,a1
    6758:	sw	a3,344(a5)
    regfile[p_gpr_unpack::FACE_DIM_8x16]   = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    675c:	lui	a0,0x800
    regfile[p_gpr_unpack::FACE_DIM_16x16]  = (face_dim_16x16 / 1) | ((face_dim_16x16 / 1) << 16);
    6760:	sw	a3,160(a4)
    regfile[p_gpr_unpack::FACE_DIM_8x16]   = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    6764:	addi	a3,a0,128 # 800080 <__runtime_args_end+0x7dfc80>
    6768:	sw	a3,164(a4)
    regfile[p_gpr_unpack::FACE_DIM_4x16]   = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    676c:	lui	a3,0x200
    regfile[p_gpr_unpack::FACE_DIM_4x16]   = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    6770:	sw	a2,168(a4)
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    6774:	addi	a2,a3,32 # 200020 <__runtime_args_end+0x1dfc20>
    regfile[p_gpr_unpack::FACE_DIM_1x16]   = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    6778:	lui	a3,0x100
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    677c:	sw	a2,172(a4)
    regfile[p_gpr_unpack::FACE_DIM_1x16]   = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    6780:	addi	a3,a3,16 # 100010 <__runtime_args_end+0xdfc10>
    6784:	sw	a3,176(a4)
    volatile std::uint32_t foo     = 0x0;
    6788:	sw	zero,8(sp)
    *fooptr                        = regfile[index];
    678c:	lw	a4,176(a4)
    6790:	sw	a4,8(sp)
    sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);

    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);
    6794:	ttsetc16	5,4

    // Enable address counter for unpacker ch1/dst address
    // final address is calculated as: Dest_cntx0/1_address + address_counter_ch1
    // used for face by face unpacking of entire tile into srcA
    cfg[UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32] = 0x1 << UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT;
    6798:	sw	a1,200(a5)
    unp_cfg_context = 0;
    679c:	sw	zero,-1988(gp) # ffb0003c <unp_cfg_context>
    TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    67a0:	ttsetc16	41,0
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");
    configure_unpack_AB<is_fp32_dest_acc_en, false, false, false, disable_src_zero_flag>(
        unpA_src_format, unpB_src_format, unpA_dst_format, unpB_dst_format, unpA_face_r_dim, unpB_face_r_dim, 0, unpA_num_faces, unpB_num_faces);

    TT_SETDMAREG(0, LOWER_HALFWORD(unpA_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    67a4:	lui	a5,0x45000
    67a8:	addi	a3,a5,72 # 45000048 <__runtime_args_end+0x44fdfc48>
    67ac:	sw	a3,0(s5)
    67b0:	lui	a4,0x2021
    TT_SETDMAREG(0, LOWER_HALFWORD(unpB_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
    67b4:	addi	a5,a5,74
    67b8:	addi	a4,a4,16 # 2021010 <__runtime_args_end+0x2000c10>
    67bc:	sw	a5,0(s5)
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    67c0:	addi	a0,sp,4
    67c4:	sw	a4,4(sp)
    67c8:	jal	627c <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>
    67cc:	beqz	a0,6ac8 <run_kernel(RuntimeParams const&)+0x510>
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    67d0:	lui	a5,0xb4010
    67d4:	addi	a5,a5,72 # b4010048 <__runtime_args_end+0xb3fefc48>
    67d8:	sw	a5,0(s5)
    67dc:	li	a5,16
{
    static_assert(UNP_SEL == p_setadc::UNP_A || UNP_SEL == p_setadc::UNP_B || UNP_SEL == p_setadc::UNP_AB, "UNP_SEL must be UNP_A, UNP_B, or UNP_AB");
    LLK_ASSERT(
        face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == FACE_R_DIM, "face_r_dim must be 1, 2, 4, 8, or FACE_R_DIM");

    switch (face_r_dim)
    67e0:	li	a4,4
    67e4:	bltu	a4,a5,6a68 <run_kernel(RuntimeParams const&)+0x4b0>
    67e8:	li	a4,1
    67ec:	beq	a5,a4,6a78 <run_kernel(RuntimeParams const&)+0x4c0>
    67f0:	li	a4,2
    67f4:	bne	a5,a4,6ae8 <run_kernel(RuntimeParams const&)+0x530>
    {
        case 1:
            TTI_SETADCXX(UNP_SEL, 1 * FACE_C_DIM - 1, 0x0);
            break;
        case 2:
            TTI_SETADCXX(UNP_SEL, 2 * FACE_C_DIM - 1, 0x0);
    67f8:	ttsetadcxx	3,31,0
    67fc:	lw	a5,4(sp)
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    6800:	addi	a0,sp,12
    6804:	srli	s6,a5,0x10
    6808:	sw	a5,12(sp)
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    680c:	zext.b	s6,s6
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
    6810:	srli	s5,a5,0x18
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    6814:	jal	627c <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>
    6818:	beqz	a0,6af0 <run_kernel(RuntimeParams const&)+0x538>
    store_blocking(&pc_buf_base[2], 0);
    681c:	lw	a5,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
    6820:	li	a3,0
    store_blocking(&pc_buf_base[2], 0);
    6824:	addi	a5,a5,8
    asm volatile(
    6828:	sw	a3,0(a5)
    682c:	lw	a3,0(a5)
    6830:	and	zero,zero,a3
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
    6834:	lui	a5,0xffb80
    6838:	sw	s6,0(a5) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
    683c:	sw	s5,4(a5)
    mop_cfg[2] = m_start_op0;
    6840:	lui	a3,0x2000
    6844:	sw	a3,8(a5)
    mop_cfg[3] = m_end_op0;
    6848:	sw	a3,12(a5)
    mop_cfg[4] = m_end_op1;
    mop_cfg[5] = m_loop_op0;
    684c:	lui	a2,0x42008
    mop_cfg[4] = m_end_op1;
    6850:	sw	a3,16(a5)
    mop_cfg[5] = m_loop_op0;
    6854:	addi	a2,a2,193 # 420080c1 <__runtime_args_end+0x41fe7cc1>
    mop_cfg[6] = m_loop_op1;
    6858:	lui	a3,0x42808
    mop_cfg[5] = m_loop_op0;
    685c:	sw	a2,20(a5)
    mop_cfg[6] = m_loop_op1;
    6860:	addi	a3,a3,193 # 428080c1 <__runtime_args_end+0x427e7cc1>
    6864:	sw	a3,24(a5)
    mop_cfg[7] = m_loop0_last_instr;
    6868:	sw	a3,28(a5)
    mop_cfg[8] = m_loop1_last_instr;
    686c:	sw	a3,32(a5)
    store_blocking(&pc_buf_base[1], 0);
    6870:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
    6874:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    6878:	addi	a4,a3,4
    asm volatile(
    687c:	sw	a5,0(a4)
    6880:	lw	a5,0(a4)
    6884:	and	zero,zero,a5
        if (is_opened)
    6888:	lbu	a5,0(sp)
    688c:	beqz	a5,68d8 <run_kernel(RuntimeParams const&)+0x320>
    return p_reg[0];
    6890:	lui	a5,0xffb12
    6894:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    6898:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    689c:	lui	a4,0xffb00
    68a0:	lw	a3,0(a4) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    68a4:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    68a8:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    68ac:	lui	a0,0xba3a0
    68b0:	slli	a5,a5,0x14
    68b4:	srli	a5,a5,0x14
    68b8:	or	a5,a5,a0
    68bc:	sh2add	a3,a4,a3
    68c0:	sw	a5,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    68c4:	addi	a4,a4,2
            --open_zone_cnt;
    68c8:	addi	a5,a2,-1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    68cc:	sw	a1,4(a3)
    68d0:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    68d4:	sw	a5,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    {
    68d8:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    68dc:	lw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    68e0:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    68e4:	li	a2,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    68e8:	add	a4,a3,a5
    68ec:	addi	a4,a4,-1021
        if (!is_buffer_full())
    68f0:	bgeu	a2,a4,693c <run_kernel(RuntimeParams const&)+0x384>
    68f4:	lui	a4,0xffb12
    68f8:	lw	a1,496(a4) # ffb121f0 <__stack_top+0x111f0>
    68fc:	lw	a4,504(a4)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6900:	lui	a2,0xffb00
    6904:	lw	a2,0(a2) # ffb00000 <llk_profiler::buffer>
            is_opened = true;
    6908:	li	a0,1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    690c:	lui	a6,0xa99a8
            is_opened = true;
    6910:	sb	a0,12(sp)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6914:	slli	a4,a4,0x14
    6918:	srli	a4,a4,0x14
    691c:	or	a4,a4,a6
            ++open_zone_cnt;
    6920:	add	a3,a3,a0
    6924:	sh2add	a2,a5,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6928:	sw	a4,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    692c:	addi	a5,a5,2
            ++open_zone_cnt;
    6930:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6934:	sw	a1,4(a2)
    6938:	sw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
constexpr std::uint32_t PERF_OUTPUT  = PERF_INPUT_C + 16 * 4096;

constexpr std::uint32_t PERF_ADDRESS(std::uint32_t buffer, std::uint32_t tile)
{
    std::uint32_t address = buffer + (tile % 16) * 4096; // Loop every 16 tiles, to prevent escaping memory
    return address / 16 - 1;                             // Correct the L1 Address for Tensix
    693c:	lui	t3,0x2
    6940:	lui	t1,0x3
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6944:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
inline void _llk_unpack_configure_addresses_(const std::uint32_t address_a, const std::uint32_t address_b, volatile std::uint32_t tt_reg_ptr *cfg)
{
    LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

    if (0 == unp_cfg_context)
    6948:	lw	a1,-1988(gp) # ffb0003c <unp_cfg_context>
    694c:	addi	t3,t3,255 # 20ff <BRISC_LOCAL_MEM_LENGTH+0xff>
    6950:	addi	t1,t1,255 # 30ff <BRISC_LOCAL_MEM_LENGTH+0x10ff>
    6954:	li	a2,0
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    6958:	lui	t0,0xffef0
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    695c:	li	t6,14
    unp_cfg_context = 1 - unp_cfg_context;
    6960:	li	a6,1
            _perf_unpack_loop_set_valid<true, true>(TILE_CNT * TILE_NUM_FACES);
            return;
        }
        else
        {
            for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
    6964:	lui	t4,0x1
    6968:	lui	t5,0x10
    696c:	srli	a5,a2,0x4
    6970:	add	a7,a5,t3
    6974:	add	a3,a5,t1
 */

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(const std::uint32_t address_a, const std::uint32_t address_b)
{
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111); // reset counters
    6978:	ttsetadczw	3,0,0,0,0,15
    if (cfg_state_id == 0)
    697c:	lw	a5,-2000(gp) # ffb00030 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    6980:	lui	a0,0xffef0
    if (cfg_state_id == 0)
    6984:	beqz	a5,698c <run_kernel(RuntimeParams const&)+0x3d4>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
    6988:	addi	a0,t0,896 # ffef0380 <__instrn_buffer+0xb0380>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    698c:	lw	a5,52(a4)
    while (semaphore_read(semaphore::UNPACK_SYNC) >= num_contexts)
    6990:	andi	a5,a5,254
    6994:	bnez	a5,698c <run_kernel(RuntimeParams const&)+0x3d4>
    6998:	bnez	a1,6a80 <run_kernel(RuntimeParams const&)+0x4c8>
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
    699c:	sw	a7,304(a0) # ffef0130 <__instrn_buffer+0xb0130>
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    69a0:	sw	a3,496(a0)
    69a4:	lw	a5,52(a4)
    69a8:	mv	a3,a1
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    69ac:	zext.b	a5,a5
    69b0:	bltu	t6,a5,6a98 <run_kernel(RuntimeParams const&)+0x4e0>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    69b4:	sw	zero,52(a4)

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    69b8:	ttstallwait	8,1024
    TTI_MOP(1, 0, 0); // run the double-loop template
    69bc:	ttmop	1,0,0
    TTI_SEMGET(semaphore::t6_sem(index));
    69c0:	ttsemget	32
    unp_cfg_context = 1 - unp_cfg_context;
    69c4:	sub	a1,a6,a3
    69c8:	sw	a1,-1988(gp) # ffb0003c <unp_cfg_context>
    if (unp_cfg_context == 0)
    69cc:	beq	a3,a6,6ac0 <run_kernel(RuntimeParams const&)+0x508>
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0101);
    69d0:	ttsetc16	41,257
    69d4:	add	a2,a2,t4
    69d8:	bne	a2,t5,696c <run_kernel(RuntimeParams const&)+0x3b4>
    asm volatile(
    69dc:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    69e0:	addi	a4,a4,4
    asm volatile(
    69e4:	sw	a5,0(a4)
    69e8:	lw	a5,0(a4)
    69ec:	and	zero,zero,a5
        if (is_opened)
    69f0:	lbu	a5,12(sp)
    69f4:	beqz	a5,6a40 <run_kernel(RuntimeParams const&)+0x488>
    return p_reg[0];
    69f8:	lui	a5,0xffb12
    69fc:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    6a00:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6a04:	lui	a4,0xffb00
    6a08:	addi	t4,t4,-1 # fff <__firmware_stack_size+0xdff>
    6a0c:	lw	a3,0(a4) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    6a10:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6a14:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    6a18:	lui	a0,0xb99a8
    6a1c:	and	a5,a5,t4
    6a20:	or	a5,a5,a0
    6a24:	sh2add	a3,a4,a3
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6a28:	addi	a4,a4,2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    6a2c:	sw	a5,0(a3)
            --open_zone_cnt;
    6a30:	addi	a5,a2,-1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    6a34:	sw	a1,4(a3)
    6a38:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    6a3c:	sw	a5,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
                _llk_unpack_AB_<>(PERF_ADDRESS(PERF_INPUT_A, tile), PERF_ADDRESS(PERF_INPUT_B, tile));
            }
        }
        PROFILER_SYNC();
    }
}
    6a40:	lw	ra,44(sp)
    6a44:	lw	s0,40(sp)
    6a48:	lw	s1,36(sp)
    6a4c:	lw	s2,32(sp)
    6a50:	lw	s3,28(sp)
    6a54:	lw	s4,24(sp)
    6a58:	lw	s5,20(sp)
    6a5c:	lw	s6,16(sp)
    6a60:	addi	sp,sp,48
    6a64:	ret
    switch (face_r_dim)
    6a68:	li	a4,8
    6a6c:	bne	a5,a4,6ae8 <run_kernel(RuntimeParams const&)+0x530>
            break;
        case 4:
            TTI_SETADCXX(UNP_SEL, 4 * FACE_C_DIM - 1, 0x0);
            break;
        case 8:
            TTI_SETADCXX(UNP_SEL, 8 * FACE_C_DIM - 1, 0x0);
    6a70:	ttsetadcxx	3,127,0
            break;
    6a74:	j	67fc <run_kernel(RuntimeParams const&)+0x244>
            TTI_SETADCXX(UNP_SEL, 1 * FACE_C_DIM - 1, 0x0);
    6a78:	ttsetadcxx	3,15,0
            break;
    6a7c:	j	67fc <run_kernel(RuntimeParams const&)+0x244>
    }
    else
    {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
    6a80:	sw	a7,308(a0) # b99a8134 <__runtime_args_end+0xb9987d34>
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    6a84:	sw	a3,500(a0)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    6a88:	lw	a5,52(a4)
    6a8c:	mv	a3,a1
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    6a90:	zext.b	a5,a5
    6a94:	bgeu	t6,a5,69b4 <run_kernel(RuntimeParams const&)+0x3fc>
    6a98:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
    6a9c:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    unp_cfg_context = 1 - unp_cfg_context;
    6aa0:	lw	a3,-1988(gp) # ffb0003c <unp_cfg_context>
    6aa4:	sw	zero,52(a4)
    6aa8:	ttstallwait	8,1024
    6aac:	ttmop	1,0,0
    TTI_SEMGET(semaphore::t6_sem(index));
    6ab0:	ttsemget	32
    6ab4:	sub	a1,a6,a3
    6ab8:	sw	a1,-1988(gp) # ffb0003c <unp_cfg_context>
    if (unp_cfg_context == 0)
    6abc:	bne	a3,a6,69d0 <run_kernel(RuntimeParams const&)+0x418>
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    6ac0:	ttsetc16	41,0
    6ac4:	j	69d4 <run_kernel(RuntimeParams const&)+0x41c>
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    6ac8:	ebreak
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    6acc:	lui	a4,0xb4010
    6ad0:	addi	a4,a4,72 # b4010048 <__runtime_args_end+0xb3fefc48>
    config_unpacker_x_end<p_setadc::UNP_AB>(tensor_shape.face_r_dim);
    6ad4:	lbu	a5,4(sp)
    6ad8:	sw	a4,0(s5)
    LLK_ASSERT(
    6adc:	li	a4,16
    6ae0:	bgeu	a4,a5,6b04 <run_kernel(RuntimeParams const&)+0x54c>
    6ae4:	ebreak
        default:
            TTI_SETADCXX(UNP_SEL, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    6ae8:	ttsetadcxx	3,255,0
            break;
    6aec:	j	67fc <run_kernel(RuntimeParams const&)+0x244>
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    6af0:	ebreak
    6af4:	j	681c <run_kernel(RuntimeParams const&)+0x264>
    LLK_ASSERT(
    6af8:	ebreak
    LLK_ASSERT(
    6afc:	ebreak
    6b00:	j	6654 <run_kernel(RuntimeParams const&)+0x9c>
    LLK_ASSERT(
    6b04:	lui	a4,0xffff0
    6b08:	addi	a4,a4,-279 # fffefee9 <__instrn_buffer+0x1afee9>
    6b0c:	sra	a4,a4,a5
    6b10:	andi	a4,a4,1
    6b14:	beqz	a4,6b1c <run_kernel(RuntimeParams const&)+0x564>
    6b18:	ebreak
    switch (face_r_dim)
    6b1c:	li	a4,4
    6b20:	bne	a5,a4,67e0 <run_kernel(RuntimeParams const&)+0x228>
            TTI_SETADCXX(UNP_SEL, 4 * FACE_C_DIM - 1, 0x0);
    6b24:	ttsetadcxx	3,63,0
            break;
    6b28:	j	67fc <run_kernel(RuntimeParams const&)+0x244>

00006b2c <_init()>:
    }
}

void _init(void)
{
}
    6b2c:	ret

00006b30 <_fini()>:

void _fini(void)
    6b30:	ret

00006b34 <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
    6b34:	lui	a5,0x20
    6b38:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
    6b3c:	sb	a5,0(a0)
        (void)(dstc[i]);
    6b40:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
    6b44:	fence
}
    6b48:	ret

00006b4c <memset>:
    6b4c:	li	t1,15
    6b50:	mv	a4,a0
    6b54:	bgeu	t1,a2,6b90 <memset+0x44>
    6b58:	andi	a5,a4,15
    6b5c:	bnez	a5,6bfc <memset+0xb0>
    6b60:	bnez	a1,6be4 <memset+0x98>
    6b64:	andi	a3,a2,-16
    6b68:	andi	a2,a2,15
    6b6c:	add	a3,a3,a4
    6b70:	sw	a1,0(a4)
    6b74:	sw	a1,4(a4)
    6b78:	sw	a1,8(a4)
    6b7c:	sw	a1,12(a4)
    6b80:	addi	a4,a4,16
    6b84:	bltu	a4,a3,6b70 <memset+0x24>
    6b88:	bnez	a2,6b90 <memset+0x44>
    6b8c:	ret
    6b90:	sub	a3,t1,a2
    6b94:	slli	a3,a3,0x2
    6b98:	auipc	t0,0x0
    6b9c:	add	a3,a3,t0
    6ba0:	jr	12(a3)
    6ba4:	sb	a1,14(a4)
    6ba8:	sb	a1,13(a4)
    6bac:	sb	a1,12(a4)
    6bb0:	sb	a1,11(a4)
    6bb4:	sb	a1,10(a4)
    6bb8:	sb	a1,9(a4)
    6bbc:	sb	a1,8(a4)
    6bc0:	sb	a1,7(a4)
    6bc4:	sb	a1,6(a4)
    6bc8:	sb	a1,5(a4)
    6bcc:	sb	a1,4(a4)
    6bd0:	sb	a1,3(a4)
    6bd4:	sb	a1,2(a4)
    6bd8:	sb	a1,1(a4)
    6bdc:	sb	a1,0(a4)
    6be0:	ret
    6be4:	zext.b	a1,a1
    6be8:	slli	a3,a1,0x8
    6bec:	or	a1,a1,a3
    6bf0:	slli	a3,a1,0x10
    6bf4:	or	a1,a1,a3
    6bf8:	j	6b64 <memset+0x18>
    6bfc:	slli	a3,a5,0x2
    6c00:	auipc	t0,0x0
    6c04:	add	a3,a3,t0
    6c08:	mv	t0,ra
    6c0c:	jalr	-96(a3)
    6c10:	mv	ra,t0
    6c14:	addi	a5,a5,-16
    6c18:	sub	a4,a4,a5
    6c1c:	add	a2,a2,a5
    6c20:	bgeu	t1,a2,6b90 <memset+0x44>
    6c24:	j	6b60 <memset+0x14>
