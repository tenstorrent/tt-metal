
/tmp/perf_overhead_artifacts/wc/L1_TO_L1/pack.elf:     file format elf32-littleriscv


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
   10018:	addi	a5,a5,72 # ffb00048 <llk_profiler::open_zone_cnt>
   1001c:	addi	a4,a4,140 # ffb0008c <__gcov_info_end>
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
   10064:	addi	a3,a3,72 # ffb00048 <llk_profiler::open_zone_cnt>
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
   10088:	sw	a1,0(a4) # ffb00000 <llk_perf::freeze_and_read_all_counters(unsigned long)::banks>
   1008c:	addi	a4,a4,4
        while (dst < end)
   10090:	bne	a5,a3,10080 <_start+0x80>
        }
    }

    // Execute global constructors
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
   10094:	lui	s0,0xffb00
   10098:	lui	s1,0xffb00
   1009c:	addi	s0,s0,40 # ffb00028 <llk_profiler::buffer>
   100a0:	addi	s1,s1,40 # ffb00028 <llk_profiler::buffer>
   100a4:	bgeu	s0,s1,100b8 <_start+0xb8>
    {
        (*temp_constructor)();
   100a8:	lw	a5,0(s0)
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
   100ec:	lw	a5,-2000(gp) # ffb00030 <ckernel::regfile>
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
   10104:	addi	a4,a5,-12 # 16aff4 <__runtime_args_end+0x14abf4>
    buffer        = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
   10108:	sw	a5,-2008(gp) # ffb00028 <llk_profiler::buffer>
    TTI_NOP;
}

inline void reset_cfg_state_id()
{
    cfg_state_id = 0;
   1010c:	sw	zero,-1960(gp) # ffb00058 <ckernel::cfg_state_id>
    write_idx     = 0;
    open_zone_cnt = 0;

    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
   10110:	lui	a2,0x1
   10114:	li	a1,0
   10118:	lui	a0,0x16d
}

inline void reset_dest_offset_id()
{
    dest_offset_id = 0;
   1011c:	sw	zero,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
   10120:	sw	a4,-2004(gp) # ffb0002c <llk_profiler::barrier_ptr>
    write_idx     = 0;
   10124:	sw	zero,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    open_zone_cnt = 0;
   10128:	sw	zero,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
   1012c:	jal	10efc <memset>
    auto& barrier = *barrier_ptr;
   10130:	lw	a5,-2004(gp) # ffb0002c <llk_profiler::barrier_ptr>
    barrier[TRISC_ID] = 1;
   10134:	li	a3,1
   10138:	sw	a3,8(a5)
    asm volatile("fence" ::: "memory");
   1013c:	fence
        while (barrier[i] != 1)
   10140:	lw	a4,0(a5)
   10144:	bne	a4,a3,1013c <main+0x7c>
   10148:	lw	a4,4(a5)
   1014c:	li	a3,1
   10150:	beq	a4,a3,10160 <main+0xa0>
            asm volatile("fence" ::: "memory");
   10154:	fence
        while (barrier[i] != 1)
   10158:	lw	a4,4(a5)
   1015c:	bne	a4,a3,10154 <main+0x94>
    zone_scoped(zone_scoped&&)                 = delete;
    zone_scoped& operator=(const zone_scoped&) = delete;
    zone_scoped& operator=(zone_scoped&&)      = delete;

    inline __attribute__((always_inline)) zone_scoped()
    {
   10160:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   10164:	lw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
   10168:	lw	a3,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        asm volatile("" ::: "memory");
        if (!is_buffer_full())
   1016c:	li	a2,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   10170:	add	a5,a4,a3
   10174:	addi	a5,a5,-1021
        if (!is_buffer_full())
   10178:	bgeu	a2,a5,101c8 <main+0x108>
// now handled by the compiler)
// workaround is needed only for GS
inline std::uint32_t reg_read(std::uint32_t addr)
{
    volatile std::uint32_t tt_reg_ptr *p_reg = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(addr);
    return p_reg[0];
   1017c:	lui	a5,0xffb12
   10180:	lw	a0,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10184:	lw	a5,504(a5)
   10188:	lw	a2,-2008(gp) # ffb00028 <llk_profiler::buffer>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   1018c:	lui	a1,0x2
   10190:	lui	a6,0xa5104
        {
            is_opened = true;
            write_entry(EntryType::ZONE_START, id16);
            ++open_zone_cnt;
   10194:	addi	a3,a3,1
   10198:	sh2add	a2,a4,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   1019c:	add	a2,a2,a1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   101a0:	addi	a4,a4,2
            is_opened = true;
   101a4:	li	a1,1
            ++open_zone_cnt;
   101a8:	sw	a3,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   101ac:	sw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   101b0:	slli	a5,a5,0x14
   101b4:	srli	a5,a5,0x14
   101b8:	or	a5,a5,a6
   101bc:	sw	a5,0(a2) # 1000 <TRISC_LOCAL_MEM_LENGTH>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   101c0:	sw	a0,4(a2)
            is_opened = true;
   101c4:	sb	a1,12(sp)
        run_kernel(temp_args);
   101c8:	addi	a0,sp,8
   101cc:	jal	104e8 <run_kernel(RuntimeParams const&)>
    store_blocking(&pc_buf_base[1], 0);
   101d0:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
   101d4:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
   101d8:	addi	a4,a4,4
    asm volatile(
   101dc:	sw	a5,0(a4)
   101e0:	lw	a5,0(a4)
   101e4:	and	zero,zero,a5
    }

    ~zone_scoped()
    {
        asm volatile("" ::: "memory");
        if (is_opened)
   101e8:	lbu	a5,12(sp)
   101ec:	beqz	a5,1023c <main+0x17c>
    return p_reg[0];
   101f0:	lui	a5,0xffb12
   101f4:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   101f8:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   101fc:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
        {
            write_entry(EntryType::ZONE_END, id16);
            --open_zone_cnt;
   10200:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
   10204:	lw	a4,-2008(gp) # ffb00028 <llk_profiler::buffer>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10208:	lui	a6,0xb5104
   1020c:	lui	a0,0x2
   10210:	sh2add	a4,a3,a4
   10214:	add	a4,a4,a0
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10218:	addi	a3,a3,2
   1021c:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10220:	slli	a5,a5,0x14
   10224:	srli	a5,a5,0x14
   10228:	or	a5,a5,a6
   1022c:	sw	a5,0(a4)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10230:	sw	a1,4(a4)
            --open_zone_cnt;
   10234:	addi	a2,a2,-1
   10238:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    *mailbox = ckernel::KERNEL_COMPLETE;
   1023c:	lui	a5,0x20
}
   10240:	lw	ra,44(sp)
   10244:	lw	s0,40(sp)
    *mailbox = ckernel::KERNEL_COMPLETE;
   10248:	li	a4,255
   1024c:	sw	a4,-64(a5) # 1ffc0 <__loader_init_end+0xafc0>
}
   10250:	lw	s1,36(sp)
   10254:	lw	s2,32(sp)
   10258:	lw	s3,28(sp)
   1025c:	li	a0,0
   10260:	addi	sp,sp,48
   10264:	ret

00010268 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>:
    }

    /// @brief Get total number of faces
    constexpr std::uint8_t total_num_faces() const
    {
        return num_faces_r_dim * num_faces_c_dim;
   10268:	lbu	a5,3(a0) # 2003 <BRISC_LOCAL_MEM_LENGTH+0x3>
   1026c:	lbu	a4,2(a0)
 *
 * @param tensor_shape: Tensor shape to validate
 * @return true if tensor shape is valid, false otherwise
 **/
__attribute__((noinline)) bool validate_tensor_shape_tile_dependent_ops_(const TensorShape &tensor_shape)
{
   10270:	mv	a3,a0
        return num_faces_r_dim * num_faces_c_dim;
   10274:	mul	a4,a4,a5
   10278:	zext.b	a4,a4
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t face_c_dim = tensor_shape.face_c_dim;
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
   1027c:	addi	a5,a4,-1
   10280:	addi	a4,a4,-4
   10284:	sltiu	a5,a5,2
   10288:	seqz	a4,a4
   1028c:	or	a0,a5,a4
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
   10290:	beqz	a0,102c4 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
   10294:	lbu	a4,0(a3)
   10298:	li	a5,16
   1029c:	li	a0,0
   102a0:	bltu	a5,a4,102c4 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
   102a4:	lui	a5,0x10
   102a8:	addi	a5,a5,278 # 10116 <main+0x56>
   102ac:	srl	a5,a5,a4
   102b0:	andi	a0,a5,1
   102b4:	beqz	a0,102c4 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
   102b8:	lbu	a0,1(a3)
   102bc:	addi	a0,a0,-16
   102c0:	seqz	a0,a0
}
   102c4:	ret

000102c8 <ckernel::packer::is_packer_fp32_late_column_output(DataFormat)>:
/**
 * \brief Returns true if `out_l1` is a valid output of the FP32 late-conversion column.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_fp32_late_column_output(const DataFormat out_l1)
{
    switch (out_l1)
   102c8:	li	a5,15
   102cc:	bltu	a5,a0,102e4 <ckernel::packer::is_packer_fp32_late_column_output(DataFormat)+0x1c>
   102d0:	lui	a5,0x9
   102d4:	addi	a5,a5,-785 # 8cef <BRISC_LOCAL_MEM_LENGTH+0x6cef>
   102d8:	srl	a0,a5,a0
   102dc:	andi	a0,a0,1
   102e0:	ret
   102e4:	li	a0,0
        case DataFormat::Bfp2_b:
            return true;
        default:
            return false;
    }
}
   102e8:	ret

000102ec <ckernel::packer::is_packer_combined_late_column_output(DataFormat)>:
 * Combined column: "From TF32 or BF16 or E8M6 or FP16 or E5M7 or E5M6 or FP8".
 * This is exactly the FP32 column plus Tf32.
 */
__attribute__((noinline)) bool is_packer_combined_late_column_output(const DataFormat out_l1)
{
    return out_l1 == DataFormat::Tf32 || is_packer_fp32_late_column_output(out_l1);
   102ec:	li	a5,4
   102f0:	beq	a0,a5,102f8 <ckernel::packer::is_packer_combined_late_column_output(DataFormat)+0xc>
   102f4:	j	102c8 <ckernel::packer::is_packer_fp32_late_column_output(DataFormat)>
}
   102f8:	li	a0,1
   102fc:	ret

00010300 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)>:
 * Validates supported dst-register to intermediate-format pairs for Blackhole's early conversion
 * stage. For this API, `out_l1` is interpreted as the requested intermediate format code.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_to_L1_early_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
    switch (in_reg)
   10300:	li	a5,5
   10304:	beq	a0,a5,103a0 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0xa0>
   10308:	bgeu	a5,a0,1032c <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x2c>
   1030c:	li	a5,8
   10310:	beq	a0,a5,10360 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x60>
   10314:	addi	a0,a0,-9
                   out_l1 == DataFormat::Int32 ||     // INT32 (identity)
                   out_l1 == DataFormat::Int8 ||      // INT8
                   out_l1 == DataFormat::UInt8;       // UINT8

        case DataFormat::UInt16: // INT16 identity path
            return out_l1 == DataFormat::UInt16;
   10318:	addi	a1,a1,-9 # 1ff7 <TRISC_LOCAL_MEM_LENGTH+0xff7>
   1031c:	seqz	a1,a1
    switch (in_reg)
   10320:	seqz	a0,a0
   10324:	and	a0,a0,a1
   10328:	ret
   1032c:	beqz	a0,10384 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x84>
   10330:	li	a5,1
   10334:	bne	a0,a5,10358 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x58>
            return out_l1 == DataFormat::Float16 || // FP16
   10338:	li	a5,14
                   out_l1 == DataFormat::Int8 ||      // INT8
   1033c:	li	a0,0
   10340:	bltu	a5,a1,10380 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x80>
   10344:	lui	a0,0x4
   10348:	addi	a0,a0,1030 # 4406 <BRISC_LOCAL_MEM_LENGTH+0x2406>
   1034c:	srl	a0,a0,a1
   10350:	andi	a0,a0,1
   10354:	ret
   10358:	li	a0,0
   1035c:	ret
            return out_l1 == DataFormat::Float32 ||   // FP32 (bitcast)
   10360:	bltu	a0,a1,10374 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x74>
   10364:	li	a0,305
   10368:	srl	a0,a0,a1
   1036c:	andi	a0,a0,1
   10370:	bnez	a0,10380 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x80>
                   out_l1 == DataFormat::Int8 ||      // INT8
   10374:	andi	a1,a1,239
   10378:	addi	a1,a1,-14
   1037c:	seqz	a0,a1

        default:
            return false;
    }
}
   10380:	ret
            return out_l1 == DataFormat::Float32 ||   // FP32 (identity)
   10384:	li	a5,30
   10388:	bltu	a5,a1,10380 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)+0x80>
   1038c:	lui	a0,0x40004
   10390:	addi	a0,a0,369 # 40004171 <__runtime_args_end+0x3ffe3d71>
   10394:	srl	a0,a0,a1
   10398:	andi	a0,a0,1
   1039c:	ret
                   out_l1 == DataFormat::Float16_b || // BF16
   103a0:	addi	a0,a1,-4
                   out_l1 == DataFormat::Int8;        // INT8
   103a4:	addi	a1,a1,-14
   103a8:	seqz	a1,a1
                   out_l1 == DataFormat::Float16_b || // BF16
   103ac:	sltiu	a0,a0,3
                   out_l1 == DataFormat::Int8;        // INT8
   103b0:	or	a0,a0,a1
   103b4:	ret

000103b8 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)>:
 * Validates supported intermediate (LateFromFormat) to L1 pairs for Blackhole's late conversion
 * stage.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_to_L1_late_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
    switch (in_reg)
   103b8:	li	a5,9
   103bc:	beq	a0,a5,1043c <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x84>
   103c0:	bgeu	a5,a0,103f4 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x3c>
   103c4:	li	a5,24
   103c8:	beq	a0,a5,10448 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x90>
   103cc:	bltu	a5,a0,10408 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x50>
   103d0:	li	a5,14
   103d4:	beq	a0,a5,10410 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x58>
   103d8:	bltu	a5,a0,10420 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x68>
   103dc:	addi	a5,a0,-10
   103e0:	zext.b	a5,a5
   103e4:	li	a4,1
   103e8:	bltu	a4,a5,10428 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x70>
        case DataFormat::Float16:
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2:
        case DataFormat::Lf8:
            return is_packer_combined_late_column_output(out_l1);
   103ec:	mv	a0,a1
   103f0:	j	102ec <ckernel::packer::is_packer_combined_late_column_output(DataFormat)>
    switch (in_reg)
   103f4:	li	a5,8
   103f8:	beq	a0,a5,10430 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x78>
   103fc:	bnez	a0,103ec <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x34>
            return is_packer_fp32_late_column_output(out_l1);
   10400:	mv	a0,a1
   10404:	j	102c8 <ckernel::packer::is_packer_fp32_late_column_output(DataFormat)>
    switch (in_reg)
   10408:	li	a5,30
   1040c:	bne	a0,a5,10428 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x70>
        case DataFormat::UInt16: // From INT16 column
            return out_l1 == DataFormat::UInt16;

        case DataFormat::Int8: // From INT8/UINT8 column
        case DataFormat::UInt8:
            return out_l1 == DataFormat::Int8 || out_l1 == DataFormat::UInt8;
   10410:	andi	a1,a1,239
   10414:	addi	a1,a1,-14
   10418:	seqz	a0,a1
   1041c:	ret
    switch (in_reg)
   10420:	li	a5,15
   10424:	beq	a0,a5,103ec <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)+0x34>
   10428:	li	a0,0
            return out_l1 == DataFormat::UInt32;

        default:
            return false;
    }
}
   1042c:	ret
            return out_l1 == DataFormat::Int32;
   10430:	addi	a1,a1,-8
   10434:	seqz	a0,a1
   10438:	ret
            return out_l1 == DataFormat::UInt16;
   1043c:	addi	a1,a1,-9
   10440:	seqz	a0,a1
   10444:	ret
            return out_l1 == DataFormat::UInt32;
   10448:	addi	a1,a1,-24
   1044c:	seqz	a0,a1
   10450:	ret

00010454 <ckernel::packer::is_packer_to_L1_conversion_supported(DataFormat, DataFormat)>:

/**
 * \brief Returns true if either EARLY or LATE packer conversion stage supports the conversion.
 */
__attribute__((noinline)) bool is_packer_to_L1_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
   10454:	addi	sp,sp,-16
   10458:	sw	s0,8(sp)
   1045c:	sw	s1,4(sp)
   10460:	sw	ra,12(sp)
   10464:	mv	s1,a0
   10468:	mv	s0,a1
    return is_packer_to_L1_early_conversion_supported(in_reg, out_l1) || is_packer_to_L1_late_conversion_supported(in_reg, out_l1);
   1046c:	jal	10300 <ckernel::packer::is_packer_to_L1_early_conversion_supported(DataFormat, DataFormat)>
   10470:	bnez	a0,10490 <ckernel::packer::is_packer_to_L1_conversion_supported(DataFormat, DataFormat)+0x3c>
   10474:	mv	a1,s0
}
   10478:	lw	s0,8(sp)
   1047c:	lw	ra,12(sp)
    return is_packer_to_L1_early_conversion_supported(in_reg, out_l1) || is_packer_to_L1_late_conversion_supported(in_reg, out_l1);
   10480:	mv	a0,s1
}
   10484:	lw	s1,4(sp)
   10488:	addi	sp,sp,16
    return is_packer_to_L1_early_conversion_supported(in_reg, out_l1) || is_packer_to_L1_late_conversion_supported(in_reg, out_l1);
   1048c:	j	103b8 <ckernel::packer::is_packer_to_L1_late_conversion_supported(DataFormat, DataFormat)>
}
   10490:	lw	ra,12(sp)
   10494:	lw	s0,8(sp)
   10498:	lw	s1,4(sp)
   1049c:	addi	sp,sp,16
   104a0:	ret

000104a4 <ckernel::packer::read_pack_config()>:
    if (cfg_state_id == 0)
   104a4:	lw	a4,-1960(gp) # ffb00058 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
   104a8:	lui	a5,0xffef0
    if (cfg_state_id == 0)
   104ac:	beqz	a4,104b4 <ckernel::packer::read_pack_config()+0x10>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
   104b0:	addi	a5,a5,896 # ffef0380 <__instrn_buffer+0xb0380>

inline pack_config_t read_pack_config_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr* cfg)
{
    pack_config_u config = {.val = 0};

    config.val[0] = cfg[reg_addr];
   104b4:	lw	a3,272(a5)
    config.val[1] = cfg[reg_addr + 1];
   104b8:	lw	a4,276(a5)
    config.val[2] = cfg[reg_addr + 2];
   104bc:	lw	a5,280(a5)
    std::array<pack_config_t, NUM_PACKERS> config_vec;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    config_vec[0] = read_pack_config_helper(THCON_SEC0_REG1_Row_start_section_size_ADDR32, cfg);
   104c0:	sw	a3,0(a0)
   104c4:	sw	a4,4(a0)
   104c8:	sw	a5,8(a0)

    return config_vec;
}
   104cc:	ret

000104d0 <ckernel::packer::read_pack_counters()>:
    if (cfg_state_id == 0)
   104d0:	lw	a4,-1960(gp) # ffb00058 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
   104d4:	lui	a5,0xffef0
    if (cfg_state_id == 0)
   104d8:	beqz	a4,104e0 <ckernel::packer::read_pack_counters()+0x10>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
   104dc:	addi	a5,a5,896 # ffef0380 <__instrn_buffer+0xb0380>
}

inline pack_counters_t read_pack_counters_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr* cfg)
{
    pack_counters_u counters = {.val = 0};
    counters.val             = cfg[reg_addr];
   104e0:	lw	a0,112(a5)
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    config_vec[0] = read_pack_counters_helper(PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32, cfg);

    return config_vec;
}
   104e4:	ret

000104e8 <run_kernel(RuntimeParams const&)>:

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
   104e8:	addi	sp,sp,-64
   104ec:	sw	s1,52(sp)
#endif
} // namespace detail

__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    std::uint32_t n = detail::next_zone_id;
   104f0:	addi	s1,gp,-1944 # ffb00068 <llk_perf::detail::next_zone_id>
   104f4:	lw	a5,0(s1)
   104f8:	sw	ra,60(sp)
   104fc:	sw	s0,56(sp)
   10500:	sw	s2,48(sp)
   10504:	sw	s3,44(sp)
   10508:	sw	s4,40(sp)
   1050c:	sw	s5,36(sp)
   10510:	sw	s6,32(sp)
   10514:	sw	s7,28(sp)
   10518:	sw	s8,24(sp)
   1051c:	sw	s9,20(sp)
   10520:	sw	s10,16(sp)
    for (std::uint32_t i = 0; i < n; ++i)
   10524:	beqz	a5,10e0c <run_kernel(RuntimeParams const&)+0x924>
    {
        if (detail::zone_hashes[i] == hash_val)
   10528:	lui	a4,0x7c867
   1052c:	lw	a3,4(s1)
   10530:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
   10534:	beq	a3,a4,105b8 <run_kernel(RuntimeParams const&)+0xd0>
    for (std::uint32_t i = 0; i < n; ++i)
   10538:	li	a3,1
   1053c:	beq	a5,a3,10e0c <run_kernel(RuntimeParams const&)+0x924>
        if (detail::zone_hashes[i] == hash_val)
   10540:	lw	a2,8(s1)
   10544:	beq	a2,a4,10e4c <run_kernel(RuntimeParams const&)+0x964>
    for (std::uint32_t i = 0; i < n; ++i)
   10548:	li	a3,2
   1054c:	beq	a5,a3,10e0c <run_kernel(RuntimeParams const&)+0x924>
        if (detail::zone_hashes[i] == hash_val)
   10550:	lw	a2,12(s1)
   10554:	beq	a2,a4,10e4c <run_kernel(RuntimeParams const&)+0x964>
    for (std::uint32_t i = 0; i < n; ++i)
   10558:	li	a3,3
   1055c:	beq	a5,a3,10e0c <run_kernel(RuntimeParams const&)+0x924>
        if (detail::zone_hashes[i] == hash_val)
   10560:	lw	a2,16(s1)
   10564:	beq	a2,a4,10e4c <run_kernel(RuntimeParams const&)+0x964>
    for (std::uint32_t i = 0; i < n; ++i)
   10568:	li	a3,4
   1056c:	beq	a5,a3,10e0c <run_kernel(RuntimeParams const&)+0x924>
        if (detail::zone_hashes[i] == hash_val)
   10570:	lw	a3,20(s1)
   10574:	beq	a3,a4,10ecc <run_kernel(RuntimeParams const&)+0x9e4>
    for (std::uint32_t i = 0; i < n; ++i)
   10578:	li	a3,5
   1057c:	beq	a5,a3,10e0c <run_kernel(RuntimeParams const&)+0x924>
        if (detail::zone_hashes[i] == hash_val)
   10580:	lui	a4,0x7c867
   10584:	lw	a2,24(s1)
   10588:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
   1058c:	beq	a2,a4,10e4c <run_kernel(RuntimeParams const&)+0x964>
    for (std::uint32_t i = 0; i < n; ++i)
   10590:	li	a3,6
   10594:	beq	a5,a3,10e0c <run_kernel(RuntimeParams const&)+0x924>
        if (detail::zone_hashes[i] == hash_val)
   10598:	lw	a2,28(s1)
   1059c:	beq	a2,a4,10e4c <run_kernel(RuntimeParams const&)+0x964>
    for (std::uint32_t i = 0; i < n; ++i)
   105a0:	li	a3,7
   105a4:	beq	a5,a3,10e0c <run_kernel(RuntimeParams const&)+0x924>
        if (detail::zone_hashes[i] == hash_val)
   105a8:	lw	a2,32(s1)
   105ac:	beq	a2,a4,10e4c <run_kernel(RuntimeParams const&)+0x964>
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
   105b0:	li	a4,8
   105b4:	bne	a5,a4,10e0c <run_kernel(RuntimeParams const&)+0x924>
    {
        detail::zone_hashes[n] = hash_val;
        detail::next_zone_id   = n + 1;
        return n;
    }
    return 0;
   105b8:	li	a5,0
    perf_counter_scoped(const perf_counter_scoped&)            = delete;
    perf_counter_scoped(perf_counter_scoped&&)                 = delete;
    perf_counter_scoped& operator=(const perf_counter_scoped&) = delete;
    perf_counter_scoped& operator=(perf_counter_scoped&&)      = delete;

    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
   105bc:	sw	a5,4(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 1u; // PERF_CNT_ALL (INSTRN+FPU)
   105c0:	lui	a5,0xffb12
   105c4:	li	a4,1
   105c8:	sw	a4,60(a5) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 1u; // TDMA_UNPACK
   105cc:	sw	a4,20(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 1u; // L1
   105d0:	sw	a4,56(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 1u; // TDMA_PACK
   105d4:	sw	a4,248(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   105d8:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   105dc:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   105e0:	li	a3,14
   105e4:	zext.b	a4,a4
   105e8:	bltu	a3,a4,10e54 <run_kernel(RuntimeParams const&)+0x96c>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   105ec:	sw	zero,32(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   105f0:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   105f4:	li	a3,14
   105f8:	zext.b	a4,a4
   105fc:	bltu	a3,a4,10e70 <run_kernel(RuntimeParams const&)+0x988>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10600:	sw	zero,32(a5)
    {
   10604:	sb	zero,0(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   10608:	lw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
   1060c:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
   10610:	li	a3,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   10614:	add	a5,a4,a2
   10618:	addi	a5,a5,-1021
        if (!is_buffer_full())
   1061c:	bgeu	a3,a5,1066c <run_kernel(RuntimeParams const&)+0x184>
    return p_reg[0];
   10620:	lui	a5,0xffb12
   10624:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10628:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   1062c:	lw	a3,-2008(gp) # ffb00028 <llk_profiler::buffer>
            is_opened = true;
   10630:	li	a0,1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10634:	lui	a7,0xa91eb
   10638:	lui	a6,0x2
            is_opened = true;
   1063c:	sb	a0,0(sp)
            ++open_zone_cnt;
   10640:	add	a2,a2,a0
   10644:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
   10648:	sh2add	a3,a4,a3
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   1064c:	add	a3,a3,a6
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10650:	addi	a4,a4,2
   10654:	sw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10658:	slli	a5,a5,0x14
   1065c:	srli	a5,a5,0x14
   10660:	or	a5,a5,a7
   10664:	sw	a5,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10668:	sw	a1,4(a3)
    LLK_ASSERT(
   1066c:	li	a1,6
   10670:	mv	a0,a1
   10674:	jal	10454 <ckernel::packer::is_packer_to_L1_conversion_supported(DataFormat, DataFormat)>
   10678:	beqz	a0,10ec4 <run_kernel(RuntimeParams const&)+0x9dc>
    if (cfg_state_id == 0)
   1067c:	lw	a4,-1960(gp) # ffb00058 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
   10680:	lui	a5,0xffef0
    if (cfg_state_id == 0)
   10684:	beqz	a4,1068c <run_kernel(RuntimeParams const&)+0x1a4>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
   10688:	addi	a5,a5,896 # ffef0380 <__instrn_buffer+0xb0380>
    TT_SETDMAREG(0, LOWER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0)); // x-stride not used!
   1068c:	lui	t4,0xffe40
   10690:	lui	a4,0x45000
   10694:	mv	t4,t4
   10698:	addi	a4,a4,56 # 45000038 <__runtime_args_end+0x44fdfc38>
   1069c:	sw	a4,0(t4) # ffe40000 <__instrn_buffer>
    TT_SETDMAREG(0, UPPER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
   106a0:	lui	a4,0x45001
   106a4:	addi	a4,a4,57 # 45001039 <__runtime_args_end+0x44fe0c39>
   106a8:	sw	a4,0(t4)
    TT_SETDMAREG(0, LOWER_HALFWORD((z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP1));
   106ac:	lui	a4,0x45010
   106b0:	addi	a4,a4,58 # 4501003a <__runtime_args_end+0x44fefc3a>
   106b4:	sw	a4,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD((w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, HI_16(p_gpr_pack::TMP1));
   106b8:	lui	a4,0x45040
   106bc:	addi	a4,a4,59 # 4504003b <__runtime_args_end+0x4501fc3b>
   106c0:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   106c4:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
   106c8:	ttwrcfg	28,0,12
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
   106cc:	ttwrcfg	29,0,13
    TTI_NOP;
   106d0:	ttnop
    TTI_NOP;
   106d4:	ttnop
    TTI_ATGETM(index);
   106d8:	ttatgetm	0
    std::uint8_t mask_b2 = (Mask >> 16) & 0xff;

    if (mask_b2 != 0)
    {
        std::uint8_t data_b2 = (wrdata) & 0xff;
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   106dc:	lui	a4,0xb5800
   106e0:	addi	a4,a4,71 # b5800047 <__runtime_args_end+0xb57dfc47>
   106e4:	sw	a4,0(t4)
    wrdata >>= 8;
    std::uint8_t mask_b3 = (Mask >> 24) & 0xff;
    if (mask_b3 != 0)
    {
        std::uint8_t data_b3 = (wrdata) & 0xff;
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
   106e8:	lui	a4,0xb61e1
   106ec:	addi	a4,a4,-1023 # b61e0c01 <__runtime_args_end+0xb61c0801>
   106f0:	sw	a4,0(t4)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
   106f4:	lui	a4,0xb3fc0
   106f8:	addi	a4,a4,2 # b3fc0002 <__runtime_args_end+0xb3f9fc02>
   106fc:	sw	a4,0(t4)
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
   10700:	lui	a4,0xb4ff0
   10704:	addi	a4,a4,2 # b4ff0002 <__runtime_args_end+0xb4fcfc02>
   10708:	sw	a4,0(t4)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   1070c:	lui	a4,0xb53f0
   10710:	addi	a4,a4,2 # b53f0002 <__runtime_args_end+0xb53cfc02>
   10714:	sw	a4,0(t4)
    TTI_ATRELM(index);
   10718:	ttatrelm	0
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   1071c:	lui	a4,0xb5100
   10720:	addi	a4,a4,71 # b5100047 <__runtime_args_end+0xb50dfc47>
   10724:	sw	a4,0(t4)
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
   10728:	lui	a4,0xb6ff0
   1072c:	addi	a4,a4,71 # b6ff0047 <__runtime_args_end+0xb6fcfc47>
   10730:	sw	a4,0(t4)
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
   10734:	lui	a3,0x40
   10738:	sw	a3,272(a5)
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
   1073c:	li	a4,1633
   10740:	sw	a4,280(a5)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
   10744:	ttstallwait	128,8
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
   10748:	lui	a4,0xb3040
   1074c:	addi	a4,a4,70 # b3040046 <__runtime_args_end+0xb301fc46>
   10750:	sw	a4,0(t4)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   10754:	lui	a4,0xb5080
   10758:	addi	a4,a4,71 # b5080047 <__runtime_args_end+0xb505fc47>
   1075c:	sw	a4,0(t4)
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
   10760:	lw	a4,-2000(gp) # ffb00030 <ckernel::regfile>
    cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] = dest_rd_ctrl.val;
   10764:	li	a2,1
   10768:	sw	a2,72(a5)
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
   1076c:	sw	a3,208(a4)
    volatile std::uint32_t foo     = 0x0;
   10770:	sw	zero,12(sp)
    *fooptr                        = regfile[index];
   10774:	lw	a0,208(a4)
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
   10778:	lui	a2,0x1
    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
   1077c:	lui	a3,0x10
   10780:	addi	a3,a3,-1 # ffff <BRISC_LOCAL_MEM_LENGTH+0xdfff>
    cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x0; // All packers use row set mapping 0, edge offset 0 mask
   10784:	sw	zero,80(a5)
    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
   10788:	li	a1,1024
   1078c:	sw	a0,12(sp)
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
   10790:	sw	a2,112(a5)
    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
   10794:	sw	a3,96(a5)
    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
   10798:	sw	a1,64(a4)
   1079c:	lw	a5,76(a4)
    regfile[p_gpr_pack::TILE_HEADER + 1] = 0;
   107a0:	sw	zero,68(a4)
    regfile[p_gpr_pack::TILE_HEADER + 2] = 0;
   107a4:	sw	zero,72(a4)
    regfile[p_gpr_pack::TILE_HEADER + 3] = 0;
   107a8:	sw	zero,76(a4)
    volatile std::uint32_t foo     = 0x0;
   107ac:	sw	zero,8(sp)
    *fooptr                        = regfile[index];
   107b0:	sw	a5,8(sp)
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
   107b4:	ttsetadcxx	4,15,0
        ADDR_MOD_PACK_SEC0_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC1_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC2_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC3_YsrcIncr_ADDR32};

    // Program source and dest registers
    __attribute__((always_inline)) inline void set(const std::uint8_t mod_index) const
    {
        TTI_SETC16(addr_mod_pack_reg_addr[mod_index], pack_val());
   107b8:	ttsetc16	37,260
   107bc:	ttsetc16	38,10272
   107c0:	ttsetc16	39,4384
    store_blocking(&pc_buf_base[2], 0);
   107c4:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
   107c8:	li	a4,0
    store_blocking(&pc_buf_base[2], 0);
   107cc:	addi	a5,a5,8
    asm volatile(
   107d0:	mv	a3,a4
   107d4:	sw	a3,0(a5)
   107d8:	lw	a3,0(a5)
   107dc:	and	zero,zero,a3
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
   107e0:	lui	a5,0xffb80
   107e4:	li	a3,4
   107e8:	sw	a3,0(a5) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
   107ec:	sw	a3,4(a5)
    mop_cfg[2] = m_start_op0;
   107f0:	lui	a3,0x2000
   107f4:	sw	a3,8(a5)
    mop_cfg[3] = m_end_op0;
   107f8:	sw	a3,12(a5)
    mop_cfg[4] = m_end_op1;
   107fc:	sw	a3,16(a5)
    mop_cfg[5] = m_loop_op0;
   10800:	lui	a1,0x41000
   10804:	sw	a1,20(a5)
    mop_cfg[6] = m_loop_op1;
   10808:	sw	a3,24(a5)
    mop_cfg[7] = m_loop0_last_instr;
   1080c:	lui	a3,0x41008
   10810:	addi	a3,a3,1 # 41008001 <__runtime_args_end+0x40fe7c01>
   10814:	sw	a3,28(a5)
    mop_cfg[8] = m_loop1_last_instr;
   10818:	lui	a3,0x41010
   1081c:	sw	a3,32(a5)
    store_blocking(&pc_buf_base[1], 0);
   10820:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
   10824:	mv	a3,a4
    store_blocking(&pc_buf_base[1], 0);
   10828:	addi	a5,a5,4
    asm volatile(
   1082c:	sw	a3,0(a5)
   10830:	lw	a3,0(a5)
   10834:	and	zero,zero,a3
    dest_offset_id = 0;
   10838:	sw	zero,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
inline void _llk_init_packer_dest_offset_registers_(
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM, [[maybe_unused]] const bool narrow_tile = false)
{
    LLK_ASSERT(face_r_dim == FACE_R_DIM, "face_r_dim: this parameter is unused");
    LLK_ASSERT(!narrow_tile, "narrow_tile: this parameter is unused");
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK); // wait for pack to finish
   1083c:	ttstallwait	33,8

    // RowMajor order
    TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
   10840:	ttsetdmareg	0,0,0,8
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
   10844:	ttsetdmareg	0,512,0,16

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10848:	ttstallwait	128,1
        TT_WRCFG(get_packer_dest_offset_index(), p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
   1084c:	lui	a5,0xb0048
   10850:	addi	a5,a5,180 # b00480b4 <__runtime_args_end+0xb0027cb4>
   10854:	sw	a5,0(t4)
    TTI_DMANOP;
   10858:	ttdmanop
    TTI_DMANOP;
   1085c:	ttdmanop
    TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
   10860:	ttsetadcxy	4,0,0,0,0,11
    TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
   10864:	ttsetadczw	4,0,0,0,0,15
    store_blocking(&pc_buf_base[1], 0);
   10868:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   1086c:	addi	a5,a5,4
{
    tensix_sync();
    reset_dest_offset_id();
    _llk_init_packer_dest_offset_registers_<Dst>(face_r_dim, narrow_tile);
    packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
   10870:	sw	zero,-1952(gp) # ffb00060 <pack_sync_tile_dst_ptr>
    asm volatile(
   10874:	sw	a4,0(a5)
   10878:	lw	a4,0(a5)
   1087c:	and	zero,zero,a4
        if (is_opened)
   10880:	lbu	a5,0(sp)
   10884:	beqz	a5,108d4 <run_kernel(RuntimeParams const&)+0x3ec>
    return p_reg[0];
   10888:	lui	a5,0xffb12
   1088c:	lw	a0,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10890:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10894:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
   10898:	lw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   1089c:	addi	a2,a2,-1 # fff <__firmware_stack_size+0xdff>
   108a0:	lw	a5,-2008(gp) # ffb00028 <llk_profiler::buffer>
   108a4:	lui	a6,0xb91eb
   108a8:	and	a4,a4,a2
   108ac:	lui	a2,0x2
   108b0:	or	a4,a4,a6
   108b4:	sh2add	a5,a3,a5
   108b8:	add	a5,a5,a2
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   108bc:	addi	a3,a3,2 # 41010002 <__runtime_args_end+0x40fefc02>
            --open_zone_cnt;
   108c0:	addi	a2,a1,-1 # 40ffffff <__runtime_args_end+0x40fdfbff>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   108c4:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   108c8:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   108cc:	sw	a0,4(a5)
            --open_zone_cnt;
   108d0:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        asm volatile("" ::: "memory");
        if constexpr (is_active_perf_thread<run_type>())
        {
            freeze_and_read_all_counters(zone_id);
   108d4:	lw	t2,4(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u;
   108d8:	lui	t0,0xffb12
   108dc:	li	a5,2
   108e0:	sw	a5,60(t0) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u;
   108e4:	sw	a5,20(t0)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u;
   108e8:	sw	a5,56(t0)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u;
   108ec:	sw	a5,248(t0)
    std::uint32_t shared_cycles            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
   108f0:	lw	a5,256(t0)
    std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
   108f4:	li	a4,860
   108f8:	mul	t2,t2,a4
   108fc:	lui	a4,0x169
   10900:	addi	t3,a4,800 # 169320 <__runtime_args_end+0x148f20>
   10904:	add	a7,t2,t3
    bank_cycles[0]                         = shared_cycles;
   10908:	sw	a5,0(a7) # a91eb000 <__runtime_args_end+0xa91cac00>
    bank_cycles[1]                         = shared_cycles;
   1090c:	sw	a5,4(a7)
    bank_cycles[2]                         = shared_cycles;
   10910:	sw	a5,8(a7)
    bank_cycles[3]                         = shared_cycles;
   10914:	sw	a5,12(a7)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10918:	lui	t5,0xffb00
   1091c:	lui	t1,0x20
    bank_cycles[4]                         = shared_cycles;
   10920:	sw	a5,16(a7)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   10924:	mv	s5,a4
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10928:	mv	t5,t5
   1092c:	addi	t1,t1,-256 # 1ff00 <__loader_init_end+0xaf00>
    std::uint32_t out_idx             = 0;
   10930:	li	a2,0
        if (bank_id == 3u)
   10934:	li	t6,3
        std::uint32_t cw = cfg[i];
   10938:	lw	a5,0(a4)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   1093c:	sh2add	a3,a2,a7
   10940:	addi	a3,a3,20
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10944:	and	a6,a5,t1
        if (!(cw & 0x80000000u))
   10948:	bgez	a5,10988 <run_kernel(RuntimeParams const&)+0x4a0>
        std::uint32_t bank_id    = cw & 0xFFu;
   1094c:	zext.b	a0,a5
        ++out_idx;
   10950:	addi	a2,a2,1 # 2001 <BRISC_LOCAL_MEM_LENGTH+0x1>
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10954:	sh3add	a1,a0,t5
        if (bank_id == 3u)
   10958:	bne	a0,t6,10974 <run_kernel(RuntimeParams const&)+0x48c>
            *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
   1095c:	lw	a0,536(t0)
   10960:	srli	a5,a5,0xd
   10964:	andi	a5,a5,112
   10968:	andi	a0,a0,-113
   1096c:	or	a5,a5,a0
   10970:	sw	a5,536(t0)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10974:	lw	a0,0(a1)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10978:	lw	a5,4(a1)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   1097c:	sw	a6,0(a0)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10980:	lw	a5,4(a5)
   10984:	sw	a5,0(a3)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   10988:	addi	a4,a4,4
   1098c:	bne	a4,t3,10938 <run_kernel(RuntimeParams const&)+0x450>
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10990:	li	a5,255
    return perf_counters_zone_data_addr(zone) + PERF_COUNTERS_ZONE_DATA_BYTES;
   10994:	add	t2,t2,s5
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10998:	sw	a5,1620(t2)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   1099c:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   109a0:	li	a3,14
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   109a4:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   109a8:	zext.b	a4,a4
   109ac:	bltu	a3,a4,10eb8 <run_kernel(RuntimeParams const&)+0x9d0>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   109b0:	sw	zero,40(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   109b4:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   109b8:	li	a3,14
   109bc:	zext.b	a4,a4
   109c0:	bltu	a3,a4,10eac <run_kernel(RuntimeParams const&)+0x9c4>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   109c4:	sw	zero,40(a5)
    std::uint32_t n = detail::next_zone_id;
   109c8:	lw	a5,0(s1)
    for (std::uint32_t i = 0; i < n; ++i)
   109cc:	beqz	a5,10e28 <run_kernel(RuntimeParams const&)+0x940>
        if (detail::zone_hashes[i] == hash_val)
   109d0:	lui	a4,0xbd77
   109d4:	lw	a3,4(s1)
   109d8:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
   109dc:	beq	a3,a4,10a60 <run_kernel(RuntimeParams const&)+0x578>
    for (std::uint32_t i = 0; i < n; ++i)
   109e0:	li	a3,1
   109e4:	beq	a5,a3,10e28 <run_kernel(RuntimeParams const&)+0x940>
        if (detail::zone_hashes[i] == hash_val)
   109e8:	lw	a2,8(s1)
   109ec:	beq	a2,a4,10e44 <run_kernel(RuntimeParams const&)+0x95c>
    for (std::uint32_t i = 0; i < n; ++i)
   109f0:	li	a3,2
   109f4:	beq	a5,a3,10e28 <run_kernel(RuntimeParams const&)+0x940>
        if (detail::zone_hashes[i] == hash_val)
   109f8:	lw	a2,12(s1)
   109fc:	beq	a2,a4,10e44 <run_kernel(RuntimeParams const&)+0x95c>
    for (std::uint32_t i = 0; i < n; ++i)
   10a00:	li	a3,3
   10a04:	beq	a5,a3,10e28 <run_kernel(RuntimeParams const&)+0x940>
        if (detail::zone_hashes[i] == hash_val)
   10a08:	lw	a2,16(s1)
   10a0c:	beq	a2,a4,10e44 <run_kernel(RuntimeParams const&)+0x95c>
    for (std::uint32_t i = 0; i < n; ++i)
   10a10:	li	a3,4
   10a14:	beq	a5,a3,10e28 <run_kernel(RuntimeParams const&)+0x940>
        if (detail::zone_hashes[i] == hash_val)
   10a18:	lw	a3,20(s1)
   10a1c:	beq	a3,a4,10ed4 <run_kernel(RuntimeParams const&)+0x9ec>
    for (std::uint32_t i = 0; i < n; ++i)
   10a20:	li	a3,5
   10a24:	beq	a5,a3,10e28 <run_kernel(RuntimeParams const&)+0x940>
        if (detail::zone_hashes[i] == hash_val)
   10a28:	lui	a4,0xbd77
   10a2c:	lw	a2,24(s1)
   10a30:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
   10a34:	beq	a2,a4,10e44 <run_kernel(RuntimeParams const&)+0x95c>
    for (std::uint32_t i = 0; i < n; ++i)
   10a38:	li	a3,6
   10a3c:	beq	a5,a3,10e28 <run_kernel(RuntimeParams const&)+0x940>
        if (detail::zone_hashes[i] == hash_val)
   10a40:	lw	a2,28(s1)
   10a44:	beq	a2,a4,10e44 <run_kernel(RuntimeParams const&)+0x95c>
    for (std::uint32_t i = 0; i < n; ++i)
   10a48:	li	a3,7
   10a4c:	beq	a5,a3,10e28 <run_kernel(RuntimeParams const&)+0x940>
        if (detail::zone_hashes[i] == hash_val)
   10a50:	lw	a2,32(s1)
   10a54:	beq	a2,a4,10e44 <run_kernel(RuntimeParams const&)+0x95c>
    if (n < PERF_COUNTERS_MAX_ZONES)
   10a58:	li	a4,8
   10a5c:	bne	a5,a4,10e28 <run_kernel(RuntimeParams const&)+0x940>
    return 0;
   10a60:	li	a5,0
    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
   10a64:	sw	a5,4(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 1u; // PERF_CNT_ALL (INSTRN+FPU)
   10a68:	lui	a5,0xffb12
   10a6c:	li	a4,1
   10a70:	sw	a4,60(a5) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 1u; // TDMA_UNPACK
   10a74:	sw	a4,20(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 1u; // L1
   10a78:	sw	a4,56(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 1u; // TDMA_PACK
   10a7c:	sw	a4,248(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10a80:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10a84:	li	a3,14
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10a88:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10a8c:	zext.b	a4,a4
   10a90:	bltu	a3,a4,10ea0 <run_kernel(RuntimeParams const&)+0x9b8>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10a94:	sw	zero,32(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10a98:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10a9c:	li	a3,14
   10aa0:	zext.b	a4,a4
   10aa4:	bltu	a3,a4,10e94 <run_kernel(RuntimeParams const&)+0x9ac>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10aa8:	sw	zero,32(a5)
    {
   10aac:	sb	zero,0(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   10ab0:	lw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
   10ab4:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
   10ab8:	li	a3,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   10abc:	add	a5,a2,a4
   10ac0:	addi	a5,a5,-1021
        if (!is_buffer_full())
   10ac4:	bgeu	a3,a5,10b14 <run_kernel(RuntimeParams const&)+0x62c>
    return p_reg[0];
   10ac8:	lui	a5,0xffb12
   10acc:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10ad0:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10ad4:	lw	a3,-2008(gp) # ffb00028 <llk_profiler::buffer>
            is_opened = true;
   10ad8:	li	a0,1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10adc:	lui	a7,0xac6d0
   10ae0:	lui	a6,0x2
            is_opened = true;
   10ae4:	sb	a0,0(sp)
            ++open_zone_cnt;
   10ae8:	add	a2,a2,a0
   10aec:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
   10af0:	sh2add	a3,a4,a3
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10af4:	add	a3,a3,a6
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10af8:	addi	a4,a4,2
   10afc:	sw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10b00:	slli	a5,a5,0x14
   10b04:	srli	a5,a5,0x14
   10b08:	or	a5,a5,a7
   10b0c:	sw	a5,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10b10:	sw	a1,4(a3)
constexpr std::uint32_t PERF_OUTPUT  = PERF_INPUT_C + 16 * 4096;

constexpr std::uint32_t PERF_ADDRESS(std::uint32_t buffer, std::uint32_t tile)
{
    std::uint32_t address = buffer + (tile % 16) * 4096; // Loop every 16 tiles, to prevent escaping memory
    return address / 16 - 1;                             // Correct the L1 Address for Tensix
   10b14:	lui	a6,0x5
    }
}

inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);
   10b18:	lui	t1,0x508c0
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10b1c:	lui	a4,0x45000
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b20:	lui	a3,0x45800
   10b24:	lw	a2,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
   10b28:	addi	s5,a6,255 # 50ff <BRISC_LOCAL_MEM_LENGTH+0x30ff>
   10b2c:	addi	s1,a6,511
   10b30:	addi	t2,a6,767
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10b34:	addi	a0,a4,24 # 45000018 <__runtime_args_end+0x44fdfc18>
   10b38:	addi	s8,t1,1 # 508c0001 <__runtime_args_end+0x5089fc01>
   10b3c:	addi	s7,t1,2
   10b40:	addi	s6,t1,3
   10b44:	addi	a6,a6,1023
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b48:	addi	a4,a4,25
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b4c:	addi	a3,a3,25 # 45800019 <__runtime_args_end+0x457dfc19>
        asm volatile("" ::: "memory");
   10b50:	li	a1,0
        TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2);
   10b54:	lui	t0,0x10144
    dest_offset_id = 1 - dest_offset_id;
   10b58:	li	a7,1
   10b5c:	lui	t6,0xb0048
    return (dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO);
   10b60:	lui	s9,0xb0088
                }
            }
        }
        else
        {
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
   10b64:	lui	t5,0x4
   10b68:	lui	t3,0x10
    TTI_SEMWAIT(p_stall::STALL_TDMA, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
   10b6c:	ttsemwait	1,2,1
   10b70:	srli	a5,a1,0x4
   10b74:	add	s10,a5,s5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10b78:	slli	s10,s10,0x8
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);
   10b7c:	sw	t1,0(t4)
   10b80:	add	s10,s10,a0
   10b84:	sw	s10,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b88:	sw	a3,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10b8c:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10b90:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b94:	sw	a4,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10b98:	ttdmanop
    TTI_MOP(1, 0, 0); // run the double-loop template
   10b9c:	ttmop	1,0,0

    program_packer_destination(address);

    ckernel::ckernel_template::run();

    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101); // reset z counters
   10ba0:	ttsetadczw	4,0,0,0,0,5
   10ba4:	add	s10,a5,s1
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10ba8:	slli	s10,s10,0x8
   10bac:	sw	s8,0(t4)
   10bb0:	add	s10,s10,a0
   10bb4:	sw	s10,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10bb8:	sw	a3,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10bbc:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10bc0:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10bc4:	sw	a4,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10bc8:	ttdmanop
   10bcc:	ttmop	1,0,0
   10bd0:	ttsetadczw	4,0,0,0,0,5
   10bd4:	add	s10,a5,t2
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10bd8:	slli	s10,s10,0x8
   10bdc:	sw	s7,0(t4)
   10be0:	add	s10,s10,a0
   10be4:	sw	s10,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10be8:	sw	a3,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10bec:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10bf0:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10bf4:	sw	a4,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10bf8:	ttdmanop
   10bfc:	ttmop	1,0,0
   10c00:	ttsetadczw	4,0,0,0,0,5
   10c04:	add	a5,a5,a6
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10c08:	slli	a5,a5,0x8
   10c0c:	sw	s6,0(t4)
   10c10:	add	a5,a5,a0
   10c14:	sw	a5,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10c18:	sw	a3,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10c1c:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10c20:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10c24:	sw	a4,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10c28:	ttdmanop
   10c2c:	ttmop	1,0,0
   10c30:	ttsetadczw	4,0,0,0,0,5
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK); // wait for pack to finish
   10c34:	ttstallwait	64,8
        TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2);
   10c38:	andi	a5,a2,1
   10c3c:	add	a5,a5,t0
   10c40:	sw	a5,0(t4)
    TTI_SEMGET(semaphore::t6_sem(index));
   10c44:	ttsemget	2
    dest_offset_id = 1 - dest_offset_id;
   10c48:	mv	s10,a2
   10c4c:	addi	a5,t6,180 # b00480b4 <__runtime_args_end+0xb0027cb4>
   10c50:	sub	a2,a7,a2
    return (dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO);
   10c54:	beq	s10,a7,10c5c <run_kernel(RuntimeParams const&)+0x774>
   10c58:	addi	a5,s9,180 # b00880b4 <__runtime_args_end+0xb0067cb4>
        TT_WRCFG(get_packer_dest_offset_index(), p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
   10c5c:	sw	a5,0(t4)
    TTI_DMANOP;
   10c60:	ttdmanop
    TTI_DMANOP;
   10c64:	ttdmanop
   10c68:	add	a1,a1,t5
   10c6c:	bne	a1,t3,10b6c <run_kernel(RuntimeParams const&)+0x684>
    store_blocking(&pc_buf_base[1], 0);
   10c70:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
   10c74:	li	a5,0
   10c78:	sw	a2,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
    store_blocking(&pc_buf_base[1], 0);
   10c7c:	addi	a4,a4,4
    asm volatile(
   10c80:	sw	a5,0(a4)
   10c84:	lw	a5,0(a4)
   10c88:	and	zero,zero,a5
        if (is_opened)
   10c8c:	lbu	a5,0(sp)
   10c90:	beqz	a5,10ce0 <run_kernel(RuntimeParams const&)+0x7f8>
    return p_reg[0];
   10c94:	lui	a5,0xffb12
   10c98:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10c9c:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10ca0:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
   10ca4:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
   10ca8:	lw	a4,-2008(gp) # ffb00028 <llk_profiler::buffer>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10cac:	lui	a6,0xbc6d0
   10cb0:	lui	a0,0x2
   10cb4:	sh2add	a4,a3,a4
   10cb8:	add	a4,a4,a0
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10cbc:	addi	a3,a3,2
   10cc0:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10cc4:	slli	a5,a5,0x14
   10cc8:	srli	a5,a5,0x14
   10ccc:	or	a5,a5,a6
   10cd0:	sw	a5,0(a4)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10cd4:	sw	a1,4(a4)
            --open_zone_cnt;
   10cd8:	addi	a2,a2,-1
   10cdc:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
            freeze_and_read_all_counters(zone_id);
   10ce0:	lw	t0,4(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u;
   10ce4:	lui	t6,0xffb12
   10ce8:	li	a5,2
   10cec:	sw	a5,60(t6) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u;
   10cf0:	sw	a5,20(t6)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u;
   10cf4:	sw	a5,56(t6)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u;
   10cf8:	sw	a5,248(t6)
    std::uint32_t shared_cycles            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
   10cfc:	lw	a5,256(t6)
    std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
   10d00:	li	a4,860
   10d04:	mul	t0,t0,a4
   10d08:	lui	a4,0x169
   10d0c:	addi	t3,a4,800 # 169320 <__runtime_args_end+0x148f20>
   10d10:	add	a7,t0,t3
    bank_cycles[0]                         = shared_cycles;
   10d14:	sw	a5,0(a7) # ac6d0000 <__runtime_args_end+0xac6afc00>
    bank_cycles[1]                         = shared_cycles;
   10d18:	sw	a5,4(a7)
    bank_cycles[2]                         = shared_cycles;
   10d1c:	sw	a5,8(a7)
    bank_cycles[3]                         = shared_cycles;
   10d20:	sw	a5,12(a7)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10d24:	lui	t4,0xffb00
   10d28:	lui	t1,0x20
    bank_cycles[4]                         = shared_cycles;
   10d2c:	sw	a5,16(a7)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   10d30:	mv	t2,a4
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10d34:	mv	t4,t4
   10d38:	addi	t1,t1,-256 # 1ff00 <__loader_init_end+0xaf00>
    std::uint32_t out_idx             = 0;
   10d3c:	li	a2,0
        if (bank_id == 3u)
   10d40:	li	t5,3
        std::uint32_t cw = cfg[i];
   10d44:	lw	a5,0(a4)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10d48:	sh2add	a3,a2,a7
   10d4c:	addi	a3,a3,20
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10d50:	and	a6,a5,t1
        if (!(cw & 0x80000000u))
   10d54:	bgez	a5,10d94 <run_kernel(RuntimeParams const&)+0x8ac>
        std::uint32_t bank_id    = cw & 0xFFu;
   10d58:	zext.b	a0,a5
        ++out_idx;
   10d5c:	addi	a2,a2,1
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10d60:	sh3add	a1,a0,t4
        if (bank_id == 3u)
   10d64:	bne	a0,t5,10d80 <run_kernel(RuntimeParams const&)+0x898>
            *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
   10d68:	lw	a0,536(t6)
   10d6c:	srli	a5,a5,0xd
   10d70:	andi	a5,a5,112
   10d74:	andi	a0,a0,-113
   10d78:	or	a5,a5,a0
   10d7c:	sw	a5,536(t6)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10d80:	lw	a0,0(a1)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10d84:	lw	a5,4(a1)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10d88:	sw	a6,0(a0) # 2000 <BRISC_LOCAL_MEM_LENGTH>
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10d8c:	lw	a5,4(a5)
   10d90:	sw	a5,0(a3)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   10d94:	addi	a4,a4,4
   10d98:	bne	a4,t3,10d44 <run_kernel(RuntimeParams const&)+0x85c>
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10d9c:	li	a5,255
    return perf_counters_zone_data_addr(zone) + PERF_COUNTERS_ZONE_DATA_BYTES;
   10da0:	add	t0,t0,t2
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10da4:	sw	a5,1620(t0) # 10144654 <__runtime_args_end+0x10124254>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10da8:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10dac:	li	a3,14
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10db0:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10db4:	zext.b	a4,a4
   10db8:	bltu	a3,a4,10e88 <run_kernel(RuntimeParams const&)+0x9a0>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10dbc:	sw	zero,40(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10dc0:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10dc4:	li	a3,14
   10dc8:	zext.b	a4,a4
   10dcc:	bltu	a3,a4,10e7c <run_kernel(RuntimeParams const&)+0x994>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10dd0:	sw	zero,40(a5)
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
   10dd4:	lw	ra,60(sp)
   10dd8:	lw	s0,56(sp)
   10ddc:	lw	s1,52(sp)
   10de0:	lw	s2,48(sp)
   10de4:	lw	s3,44(sp)
   10de8:	lw	s4,40(sp)
   10dec:	lw	s5,36(sp)
   10df0:	lw	s6,32(sp)
   10df4:	lw	s7,28(sp)
   10df8:	lw	s8,24(sp)
   10dfc:	lw	s9,20(sp)
   10e00:	lw	s10,16(sp)
   10e04:	addi	sp,sp,64
   10e08:	ret
        detail::zone_hashes[n] = hash_val;
   10e0c:	lui	a3,0x7c867
   10e10:	sh2add	a4,a5,s1
   10e14:	addi	a3,a3,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
        detail::next_zone_id   = n + 1;
   10e18:	addi	a2,a5,1
        detail::zone_hashes[n] = hash_val;
   10e1c:	sw	a3,4(a4)
        detail::next_zone_id   = n + 1;
   10e20:	sw	a2,0(s1)
        return n;
   10e24:	j	105bc <run_kernel(RuntimeParams const&)+0xd4>
        detail::zone_hashes[n] = hash_val;
   10e28:	lui	a4,0xbd77
   10e2c:	sh2add	a2,a5,s1
   10e30:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
        detail::next_zone_id   = n + 1;
   10e34:	addi	a3,a5,1
        detail::zone_hashes[n] = hash_val;
   10e38:	sw	a4,4(a2)
        detail::next_zone_id   = n + 1;
   10e3c:	sw	a3,0(s1)
        return n;
   10e40:	j	10a64 <run_kernel(RuntimeParams const&)+0x57c>
            return i;
   10e44:	mv	a5,a3
   10e48:	j	10a64 <run_kernel(RuntimeParams const&)+0x57c>
   10e4c:	mv	a5,a3
   10e50:	j	105bc <run_kernel(RuntimeParams const&)+0xd4>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10e54:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10e58:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10e5c:	li	a3,14
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10e60:	sw	zero,32(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10e64:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10e68:	zext.b	a4,a4
   10e6c:	bgeu	a3,a4,10600 <run_kernel(RuntimeParams const&)+0x118>
   10e70:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10e74:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10e78:	j	10600 <run_kernel(RuntimeParams const&)+0x118>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10e7c:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10e80:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10e84:	j	10dd0 <run_kernel(RuntimeParams const&)+0x8e8>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10e88:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10e8c:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10e90:	j	10dbc <run_kernel(RuntimeParams const&)+0x8d4>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10e94:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10e98:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10e9c:	j	10aa8 <run_kernel(RuntimeParams const&)+0x5c0>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10ea0:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10ea4:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10ea8:	j	10a94 <run_kernel(RuntimeParams const&)+0x5ac>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10eac:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10eb0:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10eb4:	j	109c4 <run_kernel(RuntimeParams const&)+0x4dc>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10eb8:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10ebc:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10ec0:	j	109b0 <run_kernel(RuntimeParams const&)+0x4c8>
    LLK_ASSERT(
   10ec4:	ebreak
   10ec8:	j	1067c <run_kernel(RuntimeParams const&)+0x194>
   10ecc:	li	a5,4
   10ed0:	j	105bc <run_kernel(RuntimeParams const&)+0xd4>
   10ed4:	li	a5,4
   10ed8:	j	10a64 <run_kernel(RuntimeParams const&)+0x57c>

00010edc <_init()>:
    }
}

void _init(void)
{
}
   10edc:	ret

00010ee0 <_fini()>:

void _fini(void)
   10ee0:	ret

00010ee4 <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
   10ee4:	lui	a5,0x20
   10ee8:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
   10eec:	sb	a5,0(a0)
        (void)(dstc[i]);
   10ef0:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
   10ef4:	fence
}
   10ef8:	ret

00010efc <memset>:
   10efc:	li	t1,15
   10f00:	mv	a4,a0
   10f04:	bgeu	t1,a2,10f40 <memset+0x44>
   10f08:	andi	a5,a4,15
   10f0c:	bnez	a5,10fac <memset+0xb0>
   10f10:	bnez	a1,10f94 <memset+0x98>
   10f14:	andi	a3,a2,-16
   10f18:	andi	a2,a2,15
   10f1c:	add	a3,a3,a4
   10f20:	sw	a1,0(a4)
   10f24:	sw	a1,4(a4)
   10f28:	sw	a1,8(a4)
   10f2c:	sw	a1,12(a4)
   10f30:	addi	a4,a4,16
   10f34:	bltu	a4,a3,10f20 <memset+0x24>
   10f38:	bnez	a2,10f40 <memset+0x44>
   10f3c:	ret
   10f40:	sub	a3,t1,a2
   10f44:	slli	a3,a3,0x2
   10f48:	auipc	t0,0x0
   10f4c:	add	a3,a3,t0
   10f50:	jr	12(a3)
   10f54:	sb	a1,14(a4)
   10f58:	sb	a1,13(a4)
   10f5c:	sb	a1,12(a4)
   10f60:	sb	a1,11(a4)
   10f64:	sb	a1,10(a4)
   10f68:	sb	a1,9(a4)
   10f6c:	sb	a1,8(a4)
   10f70:	sb	a1,7(a4)
   10f74:	sb	a1,6(a4)
   10f78:	sb	a1,5(a4)
   10f7c:	sb	a1,4(a4)
   10f80:	sb	a1,3(a4)
   10f84:	sb	a1,2(a4)
   10f88:	sb	a1,1(a4)
   10f8c:	sb	a1,0(a4)
   10f90:	ret
   10f94:	zext.b	a1,a1
   10f98:	slli	a3,a1,0x8
   10f9c:	or	a1,a1,a3
   10fa0:	slli	a3,a1,0x10
   10fa4:	or	a1,a1,a3
   10fa8:	j	10f14 <memset+0x18>
   10fac:	slli	a3,a5,0x2
   10fb0:	auipc	t0,0x0
   10fb4:	add	a3,a3,t0
   10fb8:	mv	t0,ra
   10fbc:	jalr	-96(a3)
   10fc0:	mv	ra,t0
   10fc4:	addi	a5,a5,-16
   10fc8:	sub	a4,a4,a5
   10fcc:	add	a2,a2,a5
   10fd0:	bgeu	t1,a2,10f40 <memset+0x44>
   10fd4:	j	10f10 <memset+0x14>
