
/tmp/perf_overhead_artifacts/wc/MATH_ISOLATE/pack.elf:     file format elf32-littleriscv


Disassembly of section .init:

00010000 <_start>:
// even though -fno-asynchronous-unwind-tables -fno-exceptions flags are set
void* __gxx_personality_v0;

__attribute__((no_profile_instrument_function)) TT_ALWAYS_INLINE void do_crt0()
{
    asm volatile(
   10000:	auipc	gp,0xffaf1
   10004:	addi	gp,gp,-2048 # ffb00800 <__global_pointer$>
        "la gp, __global_pointer$\n"
        ".option pop" ::
            : "memory");

    // Set stack pointer
    asm volatile("la sp, %0" : : "i"(__stack_top) : "memory");
   10008:	auipc	sp,0xffaf1
   1000c:	addi	sp,sp,-8 # ffb01000 <__stack_top>

    // Initialize .bss
    for (volatile std::uint32_t* p = (volatile std::uint32_t*)__ldm_bss_start; p < (volatile std::uint32_t*)__ldm_bss_end; p++)
   10010:	lui	a5,0xffb00
   10014:	lui	a4,0xffb00
   10018:	addi	a5,a5,32 # ffb00020 <llk_profiler::open_zone_cnt>
   1001c:	addi	a4,a4,100 # ffb00064 <__gcov_info_end>
   10020:	bgeu	a5,a4,10044 <_start+0x44>
   10024:	addi	a4,a4,-1
   10028:	sub	a4,a4,a5
   1002c:	andi	a4,a4,-4
   10030:	addi	a4,a4,4
   10034:	add	a4,a4,a5
    {
        *p = 0;
   10038:	sw	zero,0(a5)
    for (volatile std::uint32_t* p = (volatile std::uint32_t*)__ldm_bss_start; p < (volatile std::uint32_t*)__ldm_bss_end; p++)
   1003c:	addi	a5,a5,4
   10040:	bne	a5,a4,10038 <_start+0x38>
    }

    // Copy .loader_init to .ldm_data
    if ((std::uint32_t)__loader_init_start != (std::uint32_t)__loader_init_end)
   10044:	lui	a5,0x14
   10048:	lui	a4,0x15
   1004c:	mv	a5,a5
   10050:	mv	a4,a4
   10054:	beq	a5,a4,10094 <_start+0x94>
    {
        volatile std::uint32_t* src = (volatile std::uint32_t*)__loader_init_start;
        volatile std::uint32_t* dst = (volatile std::uint32_t*)__ldm_data_start;
        volatile std::uint32_t* end = (volatile std::uint32_t*)__ldm_data_end;
        while (dst < end)
   10058:	lui	a4,0xffb00
   1005c:	lui	a3,0xffb00
   10060:	mv	a4,a4
   10064:	addi	a3,a3,32 # ffb00020 <llk_profiler::open_zone_cnt>
   10068:	bgeu	a4,a3,10094 <_start+0x94>
   1006c:	addi	a3,a3,-1
   10070:	sub	a3,a3,a4
   10074:	andi	a3,a3,-4
   10078:	addi	a3,a3,4
   1007c:	add	a3,a3,a5
        {
            *dst++ = *src++;
   10080:	lw	a1,0(a5) # 14000 <__loader_init_start>
   10084:	addi	a5,a5,4
   10088:	sw	a1,0(a4) # ffb00000 <llk_profiler::buffer>
   1008c:	addi	a4,a4,4
        while (dst < end)
   10090:	bne	a5,a3,10080 <_start+0x80>
        }
    }

    // Execute global constructors
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
   10094:	lui	s0,0xffb00
   10098:	lui	s1,0xffb00
   1009c:	mv	s0,s0
   100a0:	mv	s1,s1
   100a4:	bgeu	s0,s1,100b8 <_start+0xb8>
    {
        (*temp_constructor)();
   100a8:	lw	a5,0(s0) # ffb00000 <llk_profiler::buffer>
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
   100ac:	addi	s0,s0,4
        (*temp_constructor)();
   100b0:	jalr	a5
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
   100b4:	bltu	s0,s1,100a8 <_start+0xa8>

extern "C" __attribute__((section(".init"), naked, noreturn, no_profile_instrument_function)) std::uint32_t _start()
{
    do_crt0();

    main();
   100b8:	jal	100c0 <main>

#ifdef COVERAGE
    gcov_dump();
#endif

    for (;;)
   100bc:	j	100bc <_start+0xbc>

Disassembly of section .text:

000100c0 <main>:
    volatile char *dstc       = reinterpret_cast<volatile char *>(dst);
    const volatile char *srcc = reinterpret_cast<const volatile char *>(src);

    for (std::size_t i = 0; i < len; i++)
    {
        dstc[i] = srcc[i];
   100c0:	lui	a5,0x20
   100c4:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
{
   100c8:	addi	sp,sp,-48
   100cc:	sb	a5,8(sp)
   100d0:	sw	ra,44(sp)
   100d4:	sw	s0,40(sp)
   100d8:	sw	s1,36(sp)
   100dc:	sw	s2,32(sp)
   100e0:	sw	s3,28(sp)
    }

    for (std::size_t i = 0; i < len; i++)
    {
        (void)(dstc[i]);
   100e4:	lbu	a5,8(sp)
    }

    asm volatile("fence" ::: "memory");
   100e8:	fence
    std::fill(ckernel::regfile, ckernel::regfile + 64, 0);
   100ec:	lw	a5,-2040(gp) # ffb00008 <ckernel::regfile>
      // otherwise we just use another reference.
      typedef typename __gnu_cxx::__conditional_type<__load_outside_loop,
						     const _Tp,
						     const _Tp&>::__type _Up;
      _Up __val(__value);
      for (; __first != __last; ++__first)
   100f0:	addi	a4,a5,256
	*__first = __val;
   100f4:	sw	zero,0(a5)
      for (; __first != __last; ++__first)
   100f8:	addi	a5,a5,4
   100fc:	bne	a5,a4,100f4 <main+0x34>
    }
}

__attribute__((always_inline)) inline void reset()
{
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
   10100:	lui	a5,0x16b
    buffer        = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
   10104:	lui	s2,0xffb00
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
   10108:	addi	a4,a5,-12 # 16aff4 <__runtime_args_end+0x14abf4>
    buffer        = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
   1010c:	sw	a5,0(s2) # ffb00000 <llk_profiler::buffer>
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
   10110:	lui	s3,0xffb00
    TTI_NOP;
}

inline void reset_cfg_state_id()
{
    cfg_state_id = 0;
   10114:	sw	zero,-2000(gp) # ffb00030 <ckernel::cfg_state_id>
    write_idx     = 0;
    open_zone_cnt = 0;

    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
   10118:	lui	a2,0x1
   1011c:	li	a1,0
   10120:	lui	a0,0x16d
}

inline void reset_dest_offset_id()
{
    dest_offset_id = 0;
   10124:	sw	zero,-2004(gp) # ffb0002c <ckernel::dest_offset_id>
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
   10128:	sw	a4,4(s3) # ffb00004 <llk_profiler::barrier_ptr>
    write_idx     = 0;
   1012c:	sw	zero,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    open_zone_cnt = 0;
   10130:	sw	zero,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
   10134:	jal	10b58 <memset>
    auto& barrier = *barrier_ptr;
   10138:	lw	a5,4(s3)
    barrier[TRISC_ID] = 1;
   1013c:	li	a3,1
   10140:	sw	a3,8(a5)
    asm volatile("fence" ::: "memory");
   10144:	fence
        while (barrier[i] != 1)
   10148:	lw	a4,0(a5)
   1014c:	bne	a4,a3,10144 <main+0x84>
   10150:	lw	a4,4(a5)
   10154:	li	a3,1
   10158:	beq	a4,a3,10168 <main+0xa8>
            asm volatile("fence" ::: "memory");
   1015c:	fence
        while (barrier[i] != 1)
   10160:	lw	a4,4(a5)
   10164:	bne	a4,a3,1015c <main+0x9c>
    zone_scoped(zone_scoped&&)                 = delete;
    zone_scoped& operator=(const zone_scoped&) = delete;
    zone_scoped& operator=(zone_scoped&&)      = delete;

    inline __attribute__((always_inline)) zone_scoped()
    {
   10168:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   1016c:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
   10170:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        asm volatile("" ::: "memory");
        if (!is_buffer_full())
   10174:	li	a2,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   10178:	add	a5,a4,a3
   1017c:	addi	a5,a5,-1021
        if (!is_buffer_full())
   10180:	bgeu	a2,a5,101d0 <main+0x110>
// now handled by the compiler)
// workaround is needed only for GS
inline std::uint32_t reg_read(std::uint32_t addr)
{
    volatile std::uint32_t tt_reg_ptr *p_reg = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(addr);
    return p_reg[0];
   10184:	lui	a5,0xffb12
   10188:	lw	a0,496(a5) # ffb121f0 <__stack_top+0x111f0>
   1018c:	lw	a5,504(a5)
   10190:	lw	a2,0(s2)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10194:	lui	a1,0x2
   10198:	lui	a6,0xa5104
        {
            is_opened = true;
            write_entry(EntryType::ZONE_START, id16);
            ++open_zone_cnt;
   1019c:	addi	a3,a3,1
   101a0:	sh2add	a2,a4,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   101a4:	add	a2,a2,a1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   101a8:	addi	a4,a4,2
            is_opened = true;
   101ac:	li	a1,1
            ++open_zone_cnt;
   101b0:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   101b4:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   101b8:	slli	a5,a5,0x14
   101bc:	srli	a5,a5,0x14
   101c0:	or	a5,a5,a6
   101c4:	sw	a5,0(a2) # 1000 <TRISC_LOCAL_MEM_LENGTH>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   101c8:	sw	a0,4(a2)
            is_opened = true;
   101cc:	sb	a1,12(sp)
        run_kernel(temp_args);
   101d0:	addi	a0,sp,8
   101d4:	jal	104f0 <run_kernel(RuntimeParams const&)>
    store_blocking(&pc_buf_base[1], 0);
   101d8:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
   101dc:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
   101e0:	addi	a4,a4,4
    asm volatile(
   101e4:	sw	a5,0(a4)
   101e8:	lw	a5,0(a4)
   101ec:	and	zero,zero,a5
    }

    ~zone_scoped()
    {
        asm volatile("" ::: "memory");
        if (is_opened)
   101f0:	lbu	a5,12(sp)
   101f4:	beqz	a5,10244 <main+0x184>
    return p_reg[0];
   101f8:	lui	a5,0xffb12
   101fc:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10200:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10204:	lw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
        {
            write_entry(EntryType::ZONE_END, id16);
            --open_zone_cnt;
   10208:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
   1020c:	lw	a4,0(s2)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10210:	lui	a6,0xb5104
   10214:	lui	a0,0x2
   10218:	sh2add	a4,a3,a4
   1021c:	add	a4,a4,a0
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10220:	addi	a3,a3,2
   10224:	sw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10228:	slli	a5,a5,0x14
   1022c:	srli	a5,a5,0x14
   10230:	or	a5,a5,a6
   10234:	sw	a5,0(a4)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10238:	sw	a1,4(a4)
            --open_zone_cnt;
   1023c:	addi	a2,a2,-1
   10240:	sw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    *mailbox = ckernel::KERNEL_COMPLETE;
   10244:	lui	a5,0x20
}
   10248:	lw	ra,44(sp)
   1024c:	lw	s0,40(sp)
    *mailbox = ckernel::KERNEL_COMPLETE;
   10250:	li	a4,255
   10254:	sw	a4,-64(a5) # 1ffc0 <__loader_init_end+0xafc0>
}
   10258:	lw	s1,36(sp)
   1025c:	lw	s2,32(sp)
   10260:	lw	s3,28(sp)
   10264:	li	a0,0
   10268:	addi	sp,sp,48
   1026c:	ret

00010270 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>:
    }

    /// @brief Get total number of faces
    constexpr std::uint8_t total_num_faces() const
    {
        return num_faces_r_dim * num_faces_c_dim;
   10270:	lbu	a5,3(a0) # 2003 <BRISC_LOCAL_MEM_LENGTH+0x3>
   10274:	lbu	a4,2(a0)
 *
 * @param tensor_shape: Tensor shape to validate
 * @return true if tensor shape is valid, false otherwise
 **/
__attribute__((noinline)) bool validate_tensor_shape_tile_dependent_ops_(const TensorShape &tensor_shape)
{
   10278:	mv	a3,a0
        return num_faces_r_dim * num_faces_c_dim;
   1027c:	mul	a4,a4,a5
   10280:	zext.b	a4,a4
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t face_c_dim = tensor_shape.face_c_dim;
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
   10284:	addi	a5,a4,-1
   10288:	addi	a4,a4,-4
   1028c:	sltiu	a5,a5,2
   10290:	seqz	a4,a4
   10294:	or	a0,a5,a4
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
   10298:	beqz	a0,102cc <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
   1029c:	lbu	a4,0(a3)
   102a0:	li	a5,16
   102a4:	li	a0,0
   102a8:	bltu	a5,a4,102cc <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
   102ac:	lui	a5,0x10
   102b0:	addi	a5,a5,278 # 10116 <main+0x56>
   102b4:	srl	a5,a5,a4
   102b8:	andi	a0,a5,1
   102bc:	beqz	a0,102cc <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
   102c0:	lbu	a0,1(a3)
   102c4:	addi	a0,a0,-16
   102c8:	seqz	a0,a0
}
   102cc:	ret

000102d0 <ckernel::packer::is_packer_fp32_late_column_output(DataFormat)>:
/**
 * \brief Returns true if `out_l1` is a valid output of the FP32 late-conversion column.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_fp32_late_column_output(const DataFormat out_l1)
{
    switch (out_l1)
   102d0:	li	a5,15
   102d4:	bltu	a5,a0,102ec <ckernel::packer::is_packer_fp32_late_column_output(DataFormat)+0x1c>
   102d8:	lui	a5,0x9
   102dc:	addi	a5,a5,-785 # 8cef <BRISC_LOCAL_MEM_LENGTH+0x6cef>
   102e0:	srl	a0,a5,a0
   102e4:	andi	a0,a0,1
   102e8:	ret
   102ec:	li	a0,0
        case DataFormat::Bfp2_b:
            return true;
        default:
            return false;
    }
}
   102f0:	ret

000102f4 <ckernel::packer::is_packer_combined_late_column_output(DataFormat)>:
 * Combined column: "From TF32 or BF16 or E8M6 or FP16 or E5M7 or E5M6 or FP8".
 * This is exactly the FP32 column plus Tf32.
 */
__attribute__((noinline)) bool is_packer_combined_late_column_output(const DataFormat out_l1)
{
    return out_l1 == DataFormat::Tf32 || is_packer_fp32_late_column_output(out_l1);
   102f4:	li	a5,4
   102f8:	beq	a0,a5,10300 <ckernel::packer::is_packer_combined_late_column_output(DataFormat)+0xc>
   102fc:	j	102d0 <ckernel::packer::is_packer_fp32_late_column_output(DataFormat)>
}
   10300:	li	a0,1
   10304:	ret

00010308 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)>:
 * Validates supported dst-register to intermediate-format pairs for Blackhole's early conversion
 * stage. For this API, `out_l1` is interpreted as the requested intermediate format code.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_to_L1_early_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
    switch (in_reg)
   10308:	li	a5,5
   1030c:	beq	a0,a5,103a8 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0xa0>
   10310:	bgeu	a5,a0,10334 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x2c>
   10314:	li	a5,8
   10318:	beq	a0,a5,10368 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x60>
   1031c:	addi	a0,a0,-9
                   out_l1 == DataFormat::Int32 ||     // INT32 (identity)
                   out_l1 == DataFormat::Int8 ||      // INT8
                   out_l1 == DataFormat::UInt8;       // UINT8

        case DataFormat::UInt16: // INT16 identity path
            return out_l1 == DataFormat::UInt16;
   10320:	addi	a1,a1,-9 # 1ff7 <TRISC_LOCAL_MEM_LENGTH+0xff7>
   10324:	seqz	a1,a1
    switch (in_reg)
   10328:	seqz	a0,a0
   1032c:	and	a0,a0,a1
   10330:	ret
   10334:	beqz	a0,1038c <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x84>
   10338:	li	a5,1
   1033c:	bne	a0,a5,10360 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x58>
            return out_l1 == DataFormat::Float16 || // FP16
   10340:	li	a5,14
                   out_l1 == DataFormat::Int8 ||      // INT8
   10344:	li	a0,0
   10348:	bltu	a5,a1,10388 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x80>
   1034c:	lui	a0,0x4
   10350:	addi	a0,a0,1030 # 4406 <BRISC_LOCAL_MEM_LENGTH+0x2406>
   10354:	srl	a0,a0,a1
   10358:	andi	a0,a0,1
   1035c:	ret
   10360:	li	a0,0
   10364:	ret
            return out_l1 == DataFormat::Float32 ||   // FP32 (bitcast)
   10368:	bltu	a0,a1,1037c <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x74>
   1036c:	li	a0,305
   10370:	srl	a0,a0,a1
   10374:	andi	a0,a0,1
   10378:	bnez	a0,10388 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x80>
                   out_l1 == DataFormat::Int8 ||      // INT8
   1037c:	andi	a1,a1,239
   10380:	addi	a1,a1,-14
   10384:	seqz	a0,a1

        default:
            return false;
    }
}
   10388:	ret
            return out_l1 == DataFormat::Float32 ||   // FP32 (identity)
   1038c:	li	a5,30
   10390:	bltu	a5,a1,10388 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x80>
   10394:	lui	a0,0x40004
   10398:	addi	a0,a0,369 # 40004171 <__runtime_args_end+0x3ffe3d71>
   1039c:	srl	a0,a0,a1
   103a0:	andi	a0,a0,1
   103a4:	ret
                   out_l1 == DataFormat::Float16_b || // BF16
   103a8:	addi	a0,a1,-4
                   out_l1 == DataFormat::Int8;        // INT8
   103ac:	addi	a1,a1,-14
   103b0:	seqz	a1,a1
                   out_l1 == DataFormat::Float16_b || // BF16
   103b4:	sltiu	a0,a0,3
                   out_l1 == DataFormat::Int8;        // INT8
   103b8:	or	a0,a0,a1
   103bc:	ret

000103c0 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)>:
 * Validates supported intermediate (LateFromFormat) to L1 pairs for Blackhole's late conversion
 * stage.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_to_L1_late_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
    switch (in_reg)
   103c0:	li	a5,9
   103c4:	beq	a0,a5,10444 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x84>
   103c8:	bgeu	a5,a0,103fc <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x3c>
   103cc:	li	a5,24
   103d0:	beq	a0,a5,10450 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x90>
   103d4:	bltu	a5,a0,10410 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x50>
   103d8:	li	a5,14
   103dc:	beq	a0,a5,10418 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x58>
   103e0:	bltu	a5,a0,10428 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x68>
   103e4:	addi	a5,a0,-10
   103e8:	zext.b	a5,a5
   103ec:	li	a4,1
   103f0:	bltu	a4,a5,10430 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x70>
        case DataFormat::Float16:
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2:
        case DataFormat::Lf8:
            return is_packer_combined_late_column_output(out_l1);
   103f4:	mv	a0,a1
   103f8:	j	102f4 <ckernel::packer::is_packer_combined_late_column_output(DataFormat)>
    switch (in_reg)
   103fc:	li	a5,8
   10400:	beq	a0,a5,10438 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x78>
   10404:	bnez	a0,103f4 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x34>
            return is_packer_fp32_late_column_output(out_l1);
   10408:	mv	a0,a1
   1040c:	j	102d0 <ckernel::packer::is_packer_fp32_late_column_output(DataFormat)>
    switch (in_reg)
   10410:	li	a5,30
   10414:	bne	a0,a5,10430 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x70>
        case DataFormat::UInt16: // From INT16 column
            return out_l1 == DataFormat::UInt16;

        case DataFormat::Int8: // From INT8/UINT8 column
        case DataFormat::UInt8:
            return out_l1 == DataFormat::Int8 || out_l1 == DataFormat::UInt8;
   10418:	andi	a1,a1,239
   1041c:	addi	a1,a1,-14
   10420:	seqz	a0,a1
   10424:	ret
    switch (in_reg)
   10428:	li	a5,15
   1042c:	beq	a0,a5,103f4 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x34>
   10430:	li	a0,0
            return out_l1 == DataFormat::UInt32;

        default:
            return false;
    }
}
   10434:	ret
            return out_l1 == DataFormat::Int32;
   10438:	addi	a1,a1,-8
   1043c:	seqz	a0,a1
   10440:	ret
            return out_l1 == DataFormat::UInt16;
   10444:	addi	a1,a1,-9
   10448:	seqz	a0,a1
   1044c:	ret
            return out_l1 == DataFormat::UInt32;
   10450:	addi	a1,a1,-24
   10454:	seqz	a0,a1
   10458:	ret

0001045c <ckernel::packer::is_packer_to_L1_conversion_supported(DataFormat, DataFormat)>:

/**
 * \brief Returns true if either EARLY or LATE packer conversion stage supports the conversion.
 */
__attribute__((noinline)) bool is_packer_to_L1_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
   1045c:	addi	sp,sp,-16
   10460:	sw	s0,8(sp)
   10464:	sw	s1,4(sp)
   10468:	sw	ra,12(sp)
   1046c:	mv	s1,a0
   10470:	mv	s0,a1
    return is_packer_to_L1_early_conversion_supported(in_reg, out_l1) || is_packer_to_L1_late_conversion_supported(in_reg, out_l1);
   10474:	jal	10308 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)>
   10478:	bnez	a0,10498 <ckernel::packer::is_packer_to_L1_conversion_supported(DataFormat, DataFormat)+0x3c>
   1047c:	mv	a1,s0
}
   10480:	lw	s0,8(sp)
   10484:	lw	ra,12(sp)
    return is_packer_to_L1_early_conversion_supported(in_reg, out_l1) || is_packer_to_L1_late_conversion_supported(in_reg, out_l1);
   10488:	mv	a0,s1
}
   1048c:	lw	s1,4(sp)
   10490:	addi	sp,sp,16
    return is_packer_to_L1_early_conversion_supported(in_reg, out_l1) || is_packer_to_L1_late_conversion_supported(in_reg, out_l1);
   10494:	j	103c0 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)>
}
   10498:	lw	ra,12(sp)
   1049c:	lw	s0,8(sp)
   104a0:	lw	s1,4(sp)
   104a4:	addi	sp,sp,16
   104a8:	ret

000104ac <ckernel::packer::read_pack_config()>:
    if (cfg_state_id == 0)
   104ac:	lw	a4,-2000(gp) # ffb00030 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
   104b0:	lui	a5,0xffef0
    if (cfg_state_id == 0)
   104b4:	beqz	a4,104bc <ckernel::packer::read_pack_config()+0x10>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
   104b8:	addi	a5,a5,896 # ffef0380 <__instrn_buffer+0xb0380>

inline pack_config_t read_pack_config_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr* cfg)
{
    pack_config_u config = {.val = 0};

    config.val[0] = cfg[reg_addr];
   104bc:	lw	a3,272(a5)
    config.val[1] = cfg[reg_addr + 1];
   104c0:	lw	a4,276(a5)
    config.val[2] = cfg[reg_addr + 2];
   104c4:	lw	a5,280(a5)
    std::array<pack_config_t, NUM_PACKERS> config_vec;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    config_vec[0] = read_pack_config_helper(THCON_SEC0_REG1_Row_start_section_size_ADDR32, cfg);
   104c8:	sw	a3,0(a0)
   104cc:	sw	a4,4(a0)
   104d0:	sw	a5,8(a0)

    return config_vec;
}
   104d4:	ret

000104d8 <ckernel::packer::read_pack_counters()>:
    if (cfg_state_id == 0)
   104d8:	lw	a4,-2000(gp) # ffb00030 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
   104dc:	lui	a5,0xffef0
    if (cfg_state_id == 0)
   104e0:	beqz	a4,104e8 <ckernel::packer::read_pack_counters()+0x10>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
   104e4:	addi	a5,a5,896 # ffef0380 <__instrn_buffer+0xb0380>
}

inline pack_counters_t read_pack_counters_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr* cfg)
{
    pack_counters_u counters = {.val = 0};
    counters.val             = cfg[reg_addr];
   104e8:	lw	a0,112(a5)
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    config_vec[0] = read_pack_counters_helper(PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32, cfg);

    return config_vec;
}
   104ec:	ret

000104f0 <run_kernel(RuntimeParams const&)>:

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
   104f0:	addi	sp,sp,-48
   104f4:	sw	s1,36(sp)
#endif
} // namespace detail

__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    std::uint32_t n = detail::next_zone_id;
   104f8:	addi	s1,gp,-1984 # ffb00040 <llk_perf::detail::next_zone_id>
   104fc:	lw	a5,0(s1) # ffb00000 <llk_profiler::buffer>
   10500:	sw	ra,44(sp)
   10504:	sw	s0,40(sp)
   10508:	sw	s2,32(sp)
   1050c:	sw	s3,28(sp)
    for (std::uint32_t i = 0; i < n; ++i)
   10510:	beqz	a5,10ac8 <run_kernel(RuntimeParams const&)+0x5d8>
    {
        if (detail::zone_hashes[i] == hash_val)
   10514:	lui	a4,0x7c867
   10518:	lw	a3,4(s1)
   1051c:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
   10520:	beq	a3,a4,105a4 <run_kernel(RuntimeParams const&)+0xb4>
    for (std::uint32_t i = 0; i < n; ++i)
   10524:	li	a3,1
   10528:	beq	a5,a3,10ac8 <run_kernel(RuntimeParams const&)+0x5d8>
        if (detail::zone_hashes[i] == hash_val)
   1052c:	lw	a3,8(s1)
   10530:	beq	a3,a4,105a4 <run_kernel(RuntimeParams const&)+0xb4>
    for (std::uint32_t i = 0; i < n; ++i)
   10534:	li	a3,2
   10538:	beq	a5,a3,10ac8 <run_kernel(RuntimeParams const&)+0x5d8>
        if (detail::zone_hashes[i] == hash_val)
   1053c:	lw	a3,12(s1)
   10540:	beq	a3,a4,105a4 <run_kernel(RuntimeParams const&)+0xb4>
    for (std::uint32_t i = 0; i < n; ++i)
   10544:	li	a3,3
   10548:	beq	a5,a3,10ac8 <run_kernel(RuntimeParams const&)+0x5d8>
        if (detail::zone_hashes[i] == hash_val)
   1054c:	lw	a3,16(s1)
   10550:	beq	a3,a4,105a4 <run_kernel(RuntimeParams const&)+0xb4>
    for (std::uint32_t i = 0; i < n; ++i)
   10554:	li	a3,4
   10558:	beq	a5,a3,10ac8 <run_kernel(RuntimeParams const&)+0x5d8>
        if (detail::zone_hashes[i] == hash_val)
   1055c:	lw	a3,20(s1)
   10560:	beq	a3,a4,105a4 <run_kernel(RuntimeParams const&)+0xb4>
    for (std::uint32_t i = 0; i < n; ++i)
   10564:	li	a4,5
   10568:	beq	a5,a4,10ac8 <run_kernel(RuntimeParams const&)+0x5d8>
        if (detail::zone_hashes[i] == hash_val)
   1056c:	lui	a4,0x7c867
   10570:	lw	a3,24(s1)
   10574:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
   10578:	beq	a3,a4,105a4 <run_kernel(RuntimeParams const&)+0xb4>
    for (std::uint32_t i = 0; i < n; ++i)
   1057c:	li	a3,6
   10580:	beq	a5,a3,10ac8 <run_kernel(RuntimeParams const&)+0x5d8>
        if (detail::zone_hashes[i] == hash_val)
   10584:	lw	a3,28(s1)
   10588:	beq	a3,a4,105a4 <run_kernel(RuntimeParams const&)+0xb4>
    for (std::uint32_t i = 0; i < n; ++i)
   1058c:	li	a3,7
   10590:	beq	a5,a3,10ac8 <run_kernel(RuntimeParams const&)+0x5d8>
        if (detail::zone_hashes[i] == hash_val)
   10594:	lw	a3,32(s1)
   10598:	beq	a3,a4,105a4 <run_kernel(RuntimeParams const&)+0xb4>
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
   1059c:	li	a4,8
   105a0:	bne	a5,a4,10ac8 <run_kernel(RuntimeParams const&)+0x5d8>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   105a4:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   105a8:	lw	a5,32(a4)
                ckernel::semaphore_post(PERF_ENTRY_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
   105ac:	zext.b	a5,a5
   105b0:	bnez	a5,105c8 <run_kernel(RuntimeParams const&)+0xd8>
            {
                asm volatile("nop");
   105b4:	nop
   105b8:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   105bc:	lw	a5,32(a4)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
   105c0:	zext.b	a5,a5
   105c4:	beqz	a5,105b4 <run_kernel(RuntimeParams const&)+0xc4>
   105c8:	lw	a5,32(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
   105cc:	zext.b	a5,a5
   105d0:	beqz	a5,10b0c <run_kernel(RuntimeParams const&)+0x61c>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
   105d4:	li	a2,1
   105d8:	sw	a2,32(a4)
    {
   105dc:	sb	zero,4(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   105e0:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
   105e4:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
   105e8:	li	a1,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   105ec:	add	a5,a4,a3
   105f0:	addi	a5,a5,-1021
        if (!is_buffer_full())
   105f4:	bgeu	a1,a5,10644 <run_kernel(RuntimeParams const&)+0x154>
    return p_reg[0];
   105f8:	lui	a5,0xffb12
   105fc:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10600:	lw	a5,504(a5)
            is_opened = true;
   10604:	sb	a2,4(sp)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10608:	lui	a2,0xffb00
   1060c:	lw	a2,0(a2) # ffb00000 <llk_profiler::buffer>
   10610:	lui	a6,0xa91eb
   10614:	lui	a0,0x2
            ++open_zone_cnt;
   10618:	addi	a3,a3,1
   1061c:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
   10620:	sh2add	a2,a4,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10624:	add	a2,a2,a0
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10628:	addi	a4,a4,2
   1062c:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10630:	slli	a5,a5,0x14
   10634:	srli	a5,a5,0x14
   10638:	or	a5,a5,a6
   1063c:	sw	a5,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10640:	sw	a1,4(a2)
    LLK_ASSERT(
   10644:	li	a1,6
   10648:	mv	a0,a1
   1064c:	jal	1045c <ckernel::packer::is_packer_to_L1_conversion_supported(DataFormat, DataFormat)>
   10650:	beqz	a0,10b18 <run_kernel(RuntimeParams const&)+0x628>
    if (cfg_state_id == 0)
   10654:	lw	a5,-2000(gp) # ffb00030 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
   10658:	lui	a4,0xffef0
    if (cfg_state_id == 0)
   1065c:	beqz	a5,10664 <run_kernel(RuntimeParams const&)+0x174>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
   10660:	addi	a4,a4,896 # ffef0380 <__instrn_buffer+0xb0380>
    TT_SETDMAREG(0, LOWER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0)); // x-stride not used!
   10664:	lui	a5,0xffe40
   10668:	lui	a3,0x45000
   1066c:	mv	a5,a5
   10670:	addi	a3,a3,56 # 45000038 <__runtime_args_end+0x44fdfc38>
   10674:	sw	a3,0(a5) # ffe40000 <__instrn_buffer>
    TT_SETDMAREG(0, UPPER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
   10678:	lui	a3,0x45001
   1067c:	addi	a3,a3,57 # 45001039 <__runtime_args_end+0x44fe0c39>
   10680:	sw	a3,0(a5)
    TT_SETDMAREG(0, LOWER_HALFWORD((z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP1));
   10684:	lui	a3,0x45010
   10688:	addi	a3,a3,58 # 4501003a <__runtime_args_end+0x44fefc3a>
   1068c:	sw	a3,0(a5)
    TT_SETDMAREG(0, UPPER_HALFWORD((w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, HI_16(p_gpr_pack::TMP1));
   10690:	lui	a3,0x45040
   10694:	addi	a3,a3,59 # 4504003b <__runtime_args_end+0x4501fc3b>
   10698:	sw	a3,0(a5)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   1069c:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
   106a0:	ttwrcfg	28,0,12
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
   106a4:	ttwrcfg	29,0,13
    TTI_NOP;
   106a8:	ttnop
    TTI_NOP;
   106ac:	ttnop
    TTI_ATGETM(index);
   106b0:	ttatgetm	0
    std::uint8_t mask_b2 = (Mask >> 16) & 0xff;

    if (mask_b2 != 0)
    {
        std::uint8_t data_b2 = (wrdata) & 0xff;
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   106b4:	lui	a3,0xb5800
   106b8:	addi	a3,a3,71 # b5800047 <__runtime_args_end+0xb57dfc47>
   106bc:	sw	a3,0(a5)
    wrdata >>= 8;
    std::uint8_t mask_b3 = (Mask >> 24) & 0xff;
    if (mask_b3 != 0)
    {
        std::uint8_t data_b3 = (wrdata) & 0xff;
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
   106c0:	lui	a3,0xb61e1
   106c4:	addi	a3,a3,-1023 # b61e0c01 <__runtime_args_end+0xb61c0801>
   106c8:	sw	a3,0(a5)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
   106cc:	lui	a3,0xb3fc0
   106d0:	addi	a3,a3,2 # b3fc0002 <__runtime_args_end+0xb3f9fc02>
   106d4:	sw	a3,0(a5)
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
   106d8:	lui	a3,0xb4ff0
   106dc:	addi	a3,a3,2 # b4ff0002 <__runtime_args_end+0xb4fcfc02>
   106e0:	sw	a3,0(a5)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   106e4:	lui	a3,0xb53f0
   106e8:	addi	a3,a3,2 # b53f0002 <__runtime_args_end+0xb53cfc02>
   106ec:	sw	a3,0(a5)
    TTI_ATRELM(index);
   106f0:	ttatrelm	0
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   106f4:	lui	a3,0xb5100
   106f8:	addi	a3,a3,71 # b5100047 <__runtime_args_end+0xb50dfc47>
   106fc:	sw	a3,0(a5)
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
   10700:	lui	a3,0xb6ff0
   10704:	addi	a3,a3,71 # b6ff0047 <__runtime_args_end+0xb6fcfc47>
   10708:	sw	a3,0(a5)
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
   1070c:	lui	a2,0x40
   10710:	sw	a2,272(a4)
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
   10714:	li	a3,1633
   10718:	sw	a3,280(a4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
   1071c:	ttstallwait	128,8
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
   10720:	lui	a3,0xb3040
   10724:	addi	a3,a3,70 # b3040046 <__runtime_args_end+0xb301fc46>
   10728:	sw	a3,0(a5)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   1072c:	lui	a3,0xb5080
   10730:	addi	a3,a3,71 # b5080047 <__runtime_args_end+0xb505fc47>
   10734:	sw	a3,0(a5)
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
   10738:	lw	a3,-2040(gp) # ffb00008 <ckernel::regfile>
    cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] = dest_rd_ctrl.val;
   1073c:	li	a1,1
   10740:	sw	a1,72(a4)
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
   10744:	sw	a2,208(a3)
    volatile std::uint32_t foo     = 0x0;
   10748:	sw	zero,12(sp)
    *fooptr                        = regfile[index];
   1074c:	lw	a6,208(a3)
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
   10750:	lui	a0,0x1
    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
   10754:	lui	a2,0x10
   10758:	addi	a2,a2,-1 # ffff <BRISC_LOCAL_MEM_LENGTH+0xdfff>
    cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x0; // All packers use row set mapping 0, edge offset 0 mask
   1075c:	sw	zero,80(a4)
    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
   10760:	li	a1,1024
   10764:	sw	a6,12(sp)
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
   10768:	sw	a0,112(a4)
    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
   1076c:	sw	a2,96(a4)
    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
   10770:	sw	a1,64(a3)
   10774:	lw	a4,76(a3)
    regfile[p_gpr_pack::TILE_HEADER + 1] = 0;
   10778:	sw	zero,68(a3)
    regfile[p_gpr_pack::TILE_HEADER + 2] = 0;
   1077c:	sw	zero,72(a3)
    regfile[p_gpr_pack::TILE_HEADER + 3] = 0;
   10780:	sw	zero,76(a3)
    volatile std::uint32_t foo     = 0x0;
   10784:	sw	zero,8(sp)
    *fooptr                        = regfile[index];
   10788:	sw	a4,8(sp)
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
   1078c:	ttsetadcxx	4,15,0
        ADDR_MOD_PACK_SEC0_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC1_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC2_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC3_YsrcIncr_ADDR32};

    // Program source and dest registers
    __attribute__((always_inline)) inline void set(const std::uint8_t mod_index) const
    {
        TTI_SETC16(addr_mod_pack_reg_addr[mod_index], pack_val());
   10790:	ttsetc16	37,260
   10794:	ttsetc16	38,10272
   10798:	ttsetc16	39,4384
    store_blocking(&pc_buf_base[2], 0);
   1079c:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
   107a0:	li	a3,0
    store_blocking(&pc_buf_base[2], 0);
   107a4:	addi	a4,a4,8
    asm volatile(
   107a8:	mv	a2,a3
   107ac:	sw	a2,0(a4)
   107b0:	lw	a2,0(a4)
   107b4:	and	zero,zero,a2
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
   107b8:	lui	a4,0xffb80
   107bc:	li	a2,4
   107c0:	sw	a2,0(a4) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
   107c4:	sw	a2,4(a4)
    mop_cfg[2] = m_start_op0;
   107c8:	lui	a2,0x2000
   107cc:	sw	a2,8(a4)
    mop_cfg[3] = m_end_op0;
   107d0:	sw	a2,12(a4)
    mop_cfg[4] = m_end_op1;
   107d4:	sw	a2,16(a4)
    mop_cfg[5] = m_loop_op0;
   107d8:	lui	a1,0x41000
   107dc:	sw	a1,20(a4)
    mop_cfg[6] = m_loop_op1;
    mop_cfg[7] = m_loop0_last_instr;
   107e0:	lui	a1,0x41008
    mop_cfg[6] = m_loop_op1;
   107e4:	sw	a2,24(a4)
    mop_cfg[7] = m_loop0_last_instr;
   107e8:	addi	a2,a1,1 # 41008001 <__runtime_args_end+0x40fe7c01>
   107ec:	sw	a2,28(a4)
    mop_cfg[8] = m_loop1_last_instr;
   107f0:	lui	a2,0x41010
   107f4:	sw	a2,32(a4)
    store_blocking(&pc_buf_base[1], 0);
   107f8:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
   107fc:	mv	a2,a3
    store_blocking(&pc_buf_base[1], 0);
   10800:	addi	a4,a4,4
    asm volatile(
   10804:	sw	a2,0(a4)
   10808:	lw	a2,0(a4)
   1080c:	and	zero,zero,a2
    dest_offset_id = 0;
   10810:	sw	zero,-2004(gp) # ffb0002c <ckernel::dest_offset_id>
inline void _llk_init_packer_dest_offset_registers_(
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM, [[maybe_unused]] const bool narrow_tile = false)
{
    LLK_ASSERT(face_r_dim == FACE_R_DIM, "face_r_dim: this parameter is unused");
    LLK_ASSERT(!narrow_tile, "narrow_tile: this parameter is unused");
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK); // wait for pack to finish
   10814:	ttstallwait	33,8

    // RowMajor order
    TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
   10818:	ttsetdmareg	0,0,0,8
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
   1081c:	ttsetdmareg	0,512,0,16

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10820:	ttstallwait	128,1
        TT_WRCFG(get_packer_dest_offset_index(), p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
   10824:	lui	a4,0xb0048
   10828:	addi	a4,a4,180 # b00480b4 <__runtime_args_end+0xb0027cb4>
   1082c:	sw	a4,0(a5)
    TTI_DMANOP;
   10830:	ttdmanop
    TTI_DMANOP;
   10834:	ttdmanop
    TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
   10838:	ttsetadcxy	4,0,0,0,0,11
    TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
   1083c:	ttsetadczw	4,0,0,0,0,15
    store_blocking(&pc_buf_base[1], 0);
   10840:	lw	a5,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   10844:	addi	a5,a5,4
{
    tensix_sync();
    reset_dest_offset_id();
    _llk_init_packer_dest_offset_registers_<Dst>(face_r_dim, narrow_tile);
    packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
   10848:	sw	zero,-1992(gp) # ffb00038 <pack_sync_tile_dst_ptr>
    asm volatile(
   1084c:	sw	a3,0(a5)
   10850:	lw	a3,0(a5)
   10854:	and	zero,zero,a3
        if (is_opened)
   10858:	lbu	a5,4(sp)
   1085c:	beqz	a5,108b0 <run_kernel(RuntimeParams const&)+0x3c0>
    return p_reg[0];
   10860:	lui	a5,0xffb12
   10864:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10868:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   1086c:	lui	a5,0xffb00
   10870:	lw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
   10874:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10878:	addi	a0,a0,-1 # fff <__firmware_stack_size+0xdff>
   1087c:	lw	a5,0(a5) # ffb00000 <llk_profiler::buffer>
   10880:	lui	a6,0xb91eb
   10884:	and	a4,a4,a0
   10888:	lui	a0,0x2
   1088c:	or	a4,a4,a6
            --open_zone_cnt;
   10890:	addi	a2,a2,-1 # 4100ffff <__runtime_args_end+0x40fefbff>
   10894:	sh2add	a5,a3,a5
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10898:	add	a5,a5,a0
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   1089c:	addi	a3,a3,2
   108a0:	sw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   108a4:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   108a8:	sw	a1,4(a5)
            --open_zone_cnt;
   108ac:	sw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   108b0:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   108b4:	lw	a5,40(a4)
                ckernel::semaphore_post(PERF_EXIT_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
   108b8:	zext.b	a5,a5
   108bc:	bnez	a5,108d4 <run_kernel(RuntimeParams const&)+0x3e4>
            {
                asm volatile("nop");
   108c0:	nop
   108c4:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   108c8:	lw	a5,40(a4)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
   108cc:	zext.b	a5,a5
   108d0:	beqz	a5,108c0 <run_kernel(RuntimeParams const&)+0x3d0>
   108d4:	lw	a5,40(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
   108d8:	zext.b	a5,a5
   108dc:	beqz	a5,10b00 <run_kernel(RuntimeParams const&)+0x610>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
   108e0:	li	a3,1
   108e4:	sw	a3,40(a4)
    std::uint32_t n = detail::next_zone_id;
   108e8:	lw	a5,0(s1)
    for (std::uint32_t i = 0; i < n; ++i)
   108ec:	beqz	a5,10ae4 <run_kernel(RuntimeParams const&)+0x5f4>
        if (detail::zone_hashes[i] == hash_val)
   108f0:	lui	a4,0xbd77
   108f4:	lw	a2,4(s1)
   108f8:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
   108fc:	beq	a2,a4,1097c <run_kernel(RuntimeParams const&)+0x48c>
    for (std::uint32_t i = 0; i < n; ++i)
   10900:	beq	a5,a3,10ae4 <run_kernel(RuntimeParams const&)+0x5f4>
        if (detail::zone_hashes[i] == hash_val)
   10904:	lw	a3,8(s1)
   10908:	beq	a3,a4,1097c <run_kernel(RuntimeParams const&)+0x48c>
    for (std::uint32_t i = 0; i < n; ++i)
   1090c:	li	a3,2
   10910:	beq	a5,a3,10ae4 <run_kernel(RuntimeParams const&)+0x5f4>
        if (detail::zone_hashes[i] == hash_val)
   10914:	lw	a3,12(s1)
   10918:	beq	a3,a4,1097c <run_kernel(RuntimeParams const&)+0x48c>
    for (std::uint32_t i = 0; i < n; ++i)
   1091c:	li	a3,3
   10920:	beq	a5,a3,10ae4 <run_kernel(RuntimeParams const&)+0x5f4>
        if (detail::zone_hashes[i] == hash_val)
   10924:	lw	a3,16(s1)
   10928:	beq	a3,a4,1097c <run_kernel(RuntimeParams const&)+0x48c>
    for (std::uint32_t i = 0; i < n; ++i)
   1092c:	li	a3,4
   10930:	beq	a5,a3,10ae4 <run_kernel(RuntimeParams const&)+0x5f4>
        if (detail::zone_hashes[i] == hash_val)
   10934:	lw	a3,20(s1)
   10938:	beq	a3,a4,1097c <run_kernel(RuntimeParams const&)+0x48c>
    for (std::uint32_t i = 0; i < n; ++i)
   1093c:	li	a4,5
   10940:	beq	a5,a4,10ae4 <run_kernel(RuntimeParams const&)+0x5f4>
        if (detail::zone_hashes[i] == hash_val)
   10944:	lui	a4,0xbd77
   10948:	lw	a3,24(s1)
   1094c:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
   10950:	beq	a3,a4,1097c <run_kernel(RuntimeParams const&)+0x48c>
    for (std::uint32_t i = 0; i < n; ++i)
   10954:	li	a3,6
   10958:	beq	a5,a3,10ae4 <run_kernel(RuntimeParams const&)+0x5f4>
        if (detail::zone_hashes[i] == hash_val)
   1095c:	lw	a3,28(s1)
   10960:	beq	a3,a4,1097c <run_kernel(RuntimeParams const&)+0x48c>
    for (std::uint32_t i = 0; i < n; ++i)
   10964:	li	a3,7
   10968:	beq	a5,a3,10ae4 <run_kernel(RuntimeParams const&)+0x5f4>
        if (detail::zone_hashes[i] == hash_val)
   1096c:	lw	a3,32(s1)
   10970:	beq	a3,a4,1097c <run_kernel(RuntimeParams const&)+0x48c>
    if (n < PERF_COUNTERS_MAX_ZONES)
   10974:	li	a4,8
   10978:	bne	a5,a4,10ae4 <run_kernel(RuntimeParams const&)+0x5f4>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   1097c:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   10980:	lw	a5,32(a4)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
   10984:	zext.b	a5,a5
   10988:	bnez	a5,109a0 <run_kernel(RuntimeParams const&)+0x4b0>
                asm volatile("nop");
   1098c:	nop
   10990:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   10994:	lw	a5,32(a4)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
   10998:	zext.b	a5,a5
   1099c:	beqz	a5,1098c <run_kernel(RuntimeParams const&)+0x49c>
   109a0:	lw	a5,32(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
   109a4:	zext.b	a5,a5
   109a8:	beqz	a5,10b20 <run_kernel(RuntimeParams const&)+0x630>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
   109ac:	li	a2,1
   109b0:	sw	a2,32(a4)
    {
   109b4:	sb	zero,4(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   109b8:	lw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
   109bc:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
   109c0:	li	a1,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   109c4:	add	a5,a4,a3
   109c8:	addi	a5,a5,-1021
        if (!is_buffer_full())
   109cc:	bgeu	a1,a5,10a1c <run_kernel(RuntimeParams const&)+0x52c>
    return p_reg[0];
   109d0:	lui	a5,0xffb12
   109d4:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   109d8:	lw	a5,504(a5)
            is_opened = true;
   109dc:	sb	a2,4(sp)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   109e0:	lui	a2,0xffb00
   109e4:	lw	a2,0(a2) # ffb00000 <llk_profiler::buffer>
   109e8:	lui	a6,0xac6d0
   109ec:	lui	a0,0x2
            ++open_zone_cnt;
   109f0:	addi	a3,a3,1
   109f4:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
   109f8:	sh2add	a2,a4,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   109fc:	add	a2,a2,a0
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10a00:	addi	a4,a4,2
   10a04:	sw	a4,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10a08:	slli	a5,a5,0x14
   10a0c:	srli	a5,a5,0x14
   10a10:	or	a5,a5,a6
   10a14:	sw	a5,0(a2)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10a18:	sw	a1,4(a2)
        if (is_opened)
   10a1c:	lbu	a5,4(sp)
   10a20:	beqz	a5,10a74 <run_kernel(RuntimeParams const&)+0x584>
   10a24:	lui	a5,0xffb12
   10a28:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10a2c:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10a30:	lui	a4,0xffb00
   10a34:	lw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
   10a38:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
   10a3c:	lw	a4,0(a4) # ffb00000 <llk_profiler::buffer>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10a40:	lui	a6,0xbc6d0
   10a44:	lui	a0,0x2
   10a48:	sh2add	a4,a3,a4
   10a4c:	add	a4,a4,a0
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10a50:	addi	a3,a3,2
   10a54:	sw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10a58:	slli	a5,a5,0x14
   10a5c:	srli	a5,a5,0x14
   10a60:	or	a5,a5,a6
   10a64:	sw	a5,0(a4)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10a68:	sw	a1,4(a4)
            --open_zone_cnt;
   10a6c:	addi	a2,a2,-1
   10a70:	sw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10a74:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   10a78:	lw	a5,40(a4)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
   10a7c:	zext.b	a5,a5
   10a80:	bnez	a5,10a98 <run_kernel(RuntimeParams const&)+0x5a8>
                asm volatile("nop");
   10a84:	nop
   10a88:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   10a8c:	lw	a5,40(a4)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
   10a90:	zext.b	a5,a5
   10a94:	beqz	a5,10a84 <run_kernel(RuntimeParams const&)+0x594>
   10a98:	lw	a5,40(a4)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
   10a9c:	zext.b	a5,a5
   10aa0:	beqz	a5,10b2c <run_kernel(RuntimeParams const&)+0x63c>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
   10aa4:	li	a5,1
   10aa8:	sw	a5,40(a4)
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
   10aac:	lw	ra,44(sp)
   10ab0:	lw	s0,40(sp)
   10ab4:	lw	s1,36(sp)
   10ab8:	lw	s2,32(sp)
   10abc:	lw	s3,28(sp)
   10ac0:	addi	sp,sp,48
   10ac4:	ret
        detail::zone_hashes[n] = hash_val;
   10ac8:	lui	a4,0x7c867
   10acc:	sh2add	a3,a5,s1
   10ad0:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
        detail::next_zone_id   = n + 1;
   10ad4:	addi	a5,a5,1
        detail::zone_hashes[n] = hash_val;
   10ad8:	sw	a4,4(a3)
        detail::next_zone_id   = n + 1;
   10adc:	sw	a5,0(s1)
        return n;
   10ae0:	j	105a4 <run_kernel(RuntimeParams const&)+0xb4>
        detail::zone_hashes[n] = hash_val;
   10ae4:	lui	a4,0xbd77
   10ae8:	sh2add	a3,a5,s1
   10aec:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
        detail::next_zone_id   = n + 1;
   10af0:	addi	a5,a5,1
        detail::zone_hashes[n] = hash_val;
   10af4:	sw	a4,4(a3)
        detail::next_zone_id   = n + 1;
   10af8:	sw	a5,0(s1)
        return n;
   10afc:	j	1097c <run_kernel(RuntimeParams const&)+0x48c>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
   10b00:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
   10b04:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   10b08:	j	108e0 <run_kernel(RuntimeParams const&)+0x3f0>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
   10b0c:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
   10b10:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   10b14:	j	105d4 <run_kernel(RuntimeParams const&)+0xe4>
    LLK_ASSERT(
   10b18:	ebreak
   10b1c:	j	10654 <run_kernel(RuntimeParams const&)+0x164>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
   10b20:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
   10b24:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   10b28:	j	109ac <run_kernel(RuntimeParams const&)+0x4bc>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
   10b2c:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
   10b30:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
   10b34:	j	10aa4 <run_kernel(RuntimeParams const&)+0x5b4>

00010b38 <_init()>:
    }
}

void _init(void)
{
}
   10b38:	ret

00010b3c <_fini()>:

void _fini(void)
   10b3c:	ret

00010b40 <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
   10b40:	lui	a5,0x20
   10b44:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
   10b48:	sb	a5,0(a0) # 2000 <BRISC_LOCAL_MEM_LENGTH>
        (void)(dstc[i]);
   10b4c:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
   10b50:	fence
}
   10b54:	ret

00010b58 <memset>:
   10b58:	li	t1,15
   10b5c:	mv	a4,a0
   10b60:	bgeu	t1,a2,10b9c <memset+0x44>
   10b64:	andi	a5,a4,15
   10b68:	bnez	a5,10c08 <memset+0xb0>
   10b6c:	bnez	a1,10bf0 <memset+0x98>
   10b70:	andi	a3,a2,-16
   10b74:	andi	a2,a2,15
   10b78:	add	a3,a3,a4
   10b7c:	sw	a1,0(a4)
   10b80:	sw	a1,4(a4)
   10b84:	sw	a1,8(a4)
   10b88:	sw	a1,12(a4)
   10b8c:	addi	a4,a4,16
   10b90:	bltu	a4,a3,10b7c <memset+0x24>
   10b94:	bnez	a2,10b9c <memset+0x44>
   10b98:	ret
   10b9c:	sub	a3,t1,a2
   10ba0:	slli	a3,a3,0x2
   10ba4:	auipc	t0,0x0
   10ba8:	add	a3,a3,t0
   10bac:	jr	12(a3)
   10bb0:	sb	a1,14(a4)
   10bb4:	sb	a1,13(a4)
   10bb8:	sb	a1,12(a4)
   10bbc:	sb	a1,11(a4)
   10bc0:	sb	a1,10(a4)
   10bc4:	sb	a1,9(a4)
   10bc8:	sb	a1,8(a4)
   10bcc:	sb	a1,7(a4)
   10bd0:	sb	a1,6(a4)
   10bd4:	sb	a1,5(a4)
   10bd8:	sb	a1,4(a4)
   10bdc:	sb	a1,3(a4)
   10be0:	sb	a1,2(a4)
   10be4:	sb	a1,1(a4)
   10be8:	sb	a1,0(a4)
   10bec:	ret
   10bf0:	zext.b	a1,a1
   10bf4:	slli	a3,a1,0x8
   10bf8:	or	a1,a1,a3
   10bfc:	slli	a3,a1,0x10
   10c00:	or	a1,a1,a3
   10c04:	j	10b70 <memset+0x18>
   10c08:	slli	a3,a5,0x2
   10c0c:	auipc	t0,0x0
   10c10:	add	a3,a3,t0
   10c14:	mv	t0,ra
   10c18:	jalr	-96(a3)
   10c1c:	mv	ra,t0
   10c20:	addi	a5,a5,-16
   10c24:	sub	a4,a4,a5
   10c28:	add	a2,a2,a5
   10c2c:	bgeu	t1,a2,10b9c <memset+0x44>
   10c30:	j	10b6c <memset+0x14>
