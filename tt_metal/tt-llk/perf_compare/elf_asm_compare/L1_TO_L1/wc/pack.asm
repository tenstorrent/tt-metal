
/tmp/elf_compare/wc/L1_TO_L1/pack.elf:     file format elf32-littleriscv


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
   10088:	sw	a1,0(a4) # ffb00000 <llk_perf::perf_counter_scoped<(PerfRunType)0>::~perf_counter_scoped()::banks>
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
   1012c:	jal	10de4 <memset>
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
   104ec:	sw	s0,56(sp)
#endif
} // namespace detail

__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    std::uint32_t n = detail::next_zone_id;
   104f0:	addi	s0,gp,-1944 # ffb00068 <llk_perf::detail::next_zone_id>
   104f4:	lw	a5,0(s0)
   104f8:	sw	ra,60(sp)
   104fc:	sw	s1,52(sp)
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
   10524:	beqz	a5,10d64 <run_kernel(RuntimeParams const&)+0x87c>
    {
        if (detail::zone_hashes[i] == hash_val)
   10528:	lui	a4,0x7c867
   1052c:	lw	a3,4(s0)
   10530:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
   10534:	beq	a3,a4,105b8 <run_kernel(RuntimeParams const&)+0xd0>
    for (std::uint32_t i = 0; i < n; ++i)
   10538:	li	a3,1
   1053c:	beq	a5,a3,10d64 <run_kernel(RuntimeParams const&)+0x87c>
        if (detail::zone_hashes[i] == hash_val)
   10540:	lw	a2,8(s0)
   10544:	beq	a2,a4,10da4 <run_kernel(RuntimeParams const&)+0x8bc>
    for (std::uint32_t i = 0; i < n; ++i)
   10548:	li	a3,2
   1054c:	beq	a5,a3,10d64 <run_kernel(RuntimeParams const&)+0x87c>
        if (detail::zone_hashes[i] == hash_val)
   10550:	lw	a2,12(s0)
   10554:	beq	a2,a4,10da4 <run_kernel(RuntimeParams const&)+0x8bc>
    for (std::uint32_t i = 0; i < n; ++i)
   10558:	li	a3,3
   1055c:	beq	a5,a3,10d64 <run_kernel(RuntimeParams const&)+0x87c>
        if (detail::zone_hashes[i] == hash_val)
   10560:	lw	a2,16(s0)
   10564:	beq	a2,a4,10da4 <run_kernel(RuntimeParams const&)+0x8bc>
    for (std::uint32_t i = 0; i < n; ++i)
   10568:	li	a3,4
   1056c:	beq	a5,a3,10d64 <run_kernel(RuntimeParams const&)+0x87c>
        if (detail::zone_hashes[i] == hash_val)
   10570:	lw	a3,20(s0)
   10574:	beq	a3,a4,10db4 <run_kernel(RuntimeParams const&)+0x8cc>
    for (std::uint32_t i = 0; i < n; ++i)
   10578:	li	a3,5
   1057c:	beq	a5,a3,10d64 <run_kernel(RuntimeParams const&)+0x87c>
        if (detail::zone_hashes[i] == hash_val)
   10580:	lui	a4,0x7c867
   10584:	lw	a2,24(s0)
   10588:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
   1058c:	beq	a2,a4,10da4 <run_kernel(RuntimeParams const&)+0x8bc>
    for (std::uint32_t i = 0; i < n; ++i)
   10590:	li	a3,6
   10594:	beq	a5,a3,10d64 <run_kernel(RuntimeParams const&)+0x87c>
        if (detail::zone_hashes[i] == hash_val)
   10598:	lw	a2,28(s0)
   1059c:	beq	a2,a4,10da4 <run_kernel(RuntimeParams const&)+0x8bc>
    for (std::uint32_t i = 0; i < n; ++i)
   105a0:	li	a3,7
   105a4:	beq	a5,a3,10d64 <run_kernel(RuntimeParams const&)+0x87c>
        if (detail::zone_hashes[i] == hash_val)
   105a8:	lw	a2,32(s0)
   105ac:	beq	a2,a4,10da4 <run_kernel(RuntimeParams const&)+0x8bc>
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
   105b0:	li	a4,8
   105b4:	bne	a5,a4,10d64 <run_kernel(RuntimeParams const&)+0x87c>
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
    {
        if constexpr (perf_counter_thread_active<run_type>())
        {
            asm volatile("" ::: "memory");
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
    {
   105d8:	sb	zero,0(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   105dc:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
   105e0:	lw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
   105e4:	li	a0,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   105e8:	add	a2,a3,a1
   105ec:	addi	a2,a2,-1021
        if (!is_buffer_full())
   105f0:	bgeu	a0,a2,10638 <run_kernel(RuntimeParams const&)+0x150>
    return p_reg[0];
   105f4:	lw	a0,496(a5)
   105f8:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   105fc:	lw	a2,-2008(gp) # ffb00028 <llk_profiler::buffer>
   10600:	lui	a7,0xaa8fb
   10604:	lui	a6,0x2
            is_opened = true;
   10608:	sb	a4,0(sp)
            ++open_zone_cnt;
   1060c:	add	a1,a1,a4
   10610:	sw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
   10614:	sh2add	a2,a3,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10618:	add	a4,a2,a6
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   1061c:	addi	a3,a3,2
   10620:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10624:	slli	a5,a5,0x14
   10628:	srli	a5,a5,0x14
   1062c:	or	a5,a5,a7
   10630:	sw	a5,0(a4)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10634:	sw	a0,4(a4)
    LLK_ASSERT(
   10638:	li	a1,6
   1063c:	mv	a0,a1
   10640:	jal	10454 <ckernel::packer::is_packer_to_L1_conversion_supported(DataFormat, DataFormat)>
   10644:	beqz	a0,10dac <run_kernel(RuntimeParams const&)+0x8c4>
    if (cfg_state_id == 0)
   10648:	lw	a4,-1960(gp) # ffb00058 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
   1064c:	lui	a5,0xffef0
    if (cfg_state_id == 0)
   10650:	beqz	a4,10658 <run_kernel(RuntimeParams const&)+0x170>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
   10654:	addi	a5,a5,896 # ffef0380 <__instrn_buffer+0xb0380>
    TT_SETDMAREG(0, LOWER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0)); // x-stride not used!
   10658:	lui	t4,0xffe40
   1065c:	lui	a4,0x45000
   10660:	mv	t4,t4
   10664:	addi	a4,a4,56 # 45000038 <__runtime_args_end+0x44fdfc38>
   10668:	sw	a4,0(t4) # ffe40000 <__instrn_buffer>
    TT_SETDMAREG(0, UPPER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
   1066c:	lui	a4,0x45001
   10670:	addi	a4,a4,57 # 45001039 <__runtime_args_end+0x44fe0c39>
   10674:	sw	a4,0(t4)
    TT_SETDMAREG(0, LOWER_HALFWORD((z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP1));
   10678:	lui	a4,0x45010
   1067c:	addi	a4,a4,58 # 4501003a <__runtime_args_end+0x44fefc3a>
   10680:	sw	a4,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD((w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, HI_16(p_gpr_pack::TMP1));
   10684:	lui	a4,0x45040
   10688:	addi	a4,a4,59 # 4504003b <__runtime_args_end+0x4501fc3b>
   1068c:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10690:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
   10694:	ttwrcfg	28,0,12
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
   10698:	ttwrcfg	29,0,13
    TTI_NOP;
   1069c:	ttnop
    TTI_NOP;
   106a0:	ttnop
    TTI_ATGETM(index);
   106a4:	ttatgetm	0
    std::uint8_t mask_b2 = (Mask >> 16) & 0xff;

    if (mask_b2 != 0)
    {
        std::uint8_t data_b2 = (wrdata) & 0xff;
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   106a8:	lui	a4,0xb5800
   106ac:	addi	a4,a4,71 # b5800047 <__runtime_args_end+0xb57dfc47>
   106b0:	sw	a4,0(t4)
    wrdata >>= 8;
    std::uint8_t mask_b3 = (Mask >> 24) & 0xff;
    if (mask_b3 != 0)
    {
        std::uint8_t data_b3 = (wrdata) & 0xff;
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
   106b4:	lui	a4,0xb61e1
   106b8:	addi	a4,a4,-1023 # b61e0c01 <__runtime_args_end+0xb61c0801>
   106bc:	sw	a4,0(t4)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
   106c0:	lui	a4,0xb3fc0
   106c4:	addi	a4,a4,2 # b3fc0002 <__runtime_args_end+0xb3f9fc02>
   106c8:	sw	a4,0(t4)
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
   106cc:	lui	a4,0xb4ff0
   106d0:	addi	a4,a4,2 # b4ff0002 <__runtime_args_end+0xb4fcfc02>
   106d4:	sw	a4,0(t4)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   106d8:	lui	a4,0xb53f0
   106dc:	addi	a4,a4,2 # b53f0002 <__runtime_args_end+0xb53cfc02>
   106e0:	sw	a4,0(t4)
    TTI_ATRELM(index);
   106e4:	ttatrelm	0
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   106e8:	lui	a4,0xb5100
   106ec:	addi	a4,a4,71 # b5100047 <__runtime_args_end+0xb50dfc47>
   106f0:	sw	a4,0(t4)
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
   106f4:	lui	a4,0xb6ff0
   106f8:	addi	a4,a4,71 # b6ff0047 <__runtime_args_end+0xb6fcfc47>
   106fc:	sw	a4,0(t4)
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
   10700:	lui	a3,0x40
   10704:	sw	a3,272(a5)
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
   10708:	li	a4,1633
   1070c:	sw	a4,280(a5)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
   10710:	ttstallwait	128,8
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
   10714:	lui	a4,0xb3040
   10718:	addi	a4,a4,70 # b3040046 <__runtime_args_end+0xb301fc46>
   1071c:	sw	a4,0(t4)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   10720:	lui	a4,0xb5080
   10724:	addi	a4,a4,71 # b5080047 <__runtime_args_end+0xb505fc47>
   10728:	sw	a4,0(t4)
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
   1072c:	lw	a4,-2000(gp) # ffb00030 <ckernel::regfile>
    cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] = dest_rd_ctrl.val;
   10730:	li	a2,1
   10734:	sw	a2,72(a5)
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
   10738:	sw	a3,208(a4)
    volatile std::uint32_t foo     = 0x0;
   1073c:	sw	zero,12(sp)
    *fooptr                        = regfile[index];
   10740:	lw	a0,208(a4)
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
   10744:	lui	a2,0x1
    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
   10748:	lui	a3,0x10
   1074c:	addi	a3,a3,-1 # ffff <BRISC_LOCAL_MEM_LENGTH+0xdfff>
    cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x0; // All packers use row set mapping 0, edge offset 0 mask
   10750:	sw	zero,80(a5)
    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
   10754:	li	a1,1024
   10758:	sw	a0,12(sp)
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
   1075c:	sw	a2,112(a5)
    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
   10760:	sw	a3,96(a5)
    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
   10764:	sw	a1,64(a4)
   10768:	lw	a5,76(a4)
    regfile[p_gpr_pack::TILE_HEADER + 1] = 0;
   1076c:	sw	zero,68(a4)
    regfile[p_gpr_pack::TILE_HEADER + 2] = 0;
   10770:	sw	zero,72(a4)
    regfile[p_gpr_pack::TILE_HEADER + 3] = 0;
   10774:	sw	zero,76(a4)
    volatile std::uint32_t foo     = 0x0;
   10778:	sw	zero,8(sp)
    *fooptr                        = regfile[index];
   1077c:	sw	a5,8(sp)
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
   10780:	ttsetadcxx	4,15,0
        ADDR_MOD_PACK_SEC0_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC1_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC2_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC3_YsrcIncr_ADDR32};

    // Program source and dest registers
    __attribute__((always_inline)) inline void set(const std::uint8_t mod_index) const
    {
        TTI_SETC16(addr_mod_pack_reg_addr[mod_index], pack_val());
   10784:	ttsetc16	37,260
   10788:	ttsetc16	38,10272
   1078c:	ttsetc16	39,4384
    store_blocking(&pc_buf_base[2], 0);
   10790:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
   10794:	li	a4,0
    store_blocking(&pc_buf_base[2], 0);
   10798:	addi	a5,a5,8
    asm volatile(
   1079c:	mv	a3,a4
   107a0:	sw	a3,0(a5)
   107a4:	lw	a3,0(a5)
   107a8:	and	zero,zero,a3
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
   107ac:	lui	a5,0xffb80
   107b0:	li	a3,4
   107b4:	sw	a3,0(a5) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
   107b8:	sw	a3,4(a5)
    mop_cfg[2] = m_start_op0;
   107bc:	lui	a3,0x2000
   107c0:	sw	a3,8(a5)
    mop_cfg[3] = m_end_op0;
   107c4:	sw	a3,12(a5)
    mop_cfg[4] = m_end_op1;
   107c8:	sw	a3,16(a5)
    mop_cfg[5] = m_loop_op0;
   107cc:	lui	a1,0x41000
   107d0:	sw	a1,20(a5)
    mop_cfg[6] = m_loop_op1;
   107d4:	sw	a3,24(a5)
    mop_cfg[7] = m_loop0_last_instr;
   107d8:	lui	a3,0x41008
   107dc:	addi	a3,a3,1 # 41008001 <__runtime_args_end+0x40fe7c01>
   107e0:	sw	a3,28(a5)
    mop_cfg[8] = m_loop1_last_instr;
   107e4:	lui	a3,0x41010
   107e8:	sw	a3,32(a5)
    store_blocking(&pc_buf_base[1], 0);
   107ec:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
   107f0:	mv	a3,a4
    store_blocking(&pc_buf_base[1], 0);
   107f4:	addi	a5,a5,4
    asm volatile(
   107f8:	sw	a3,0(a5)
   107fc:	lw	a3,0(a5)
   10800:	and	zero,zero,a3
    dest_offset_id = 0;
   10804:	sw	zero,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
inline void _llk_init_packer_dest_offset_registers_(
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM, [[maybe_unused]] const bool narrow_tile = false)
{
    LLK_ASSERT(face_r_dim == FACE_R_DIM, "face_r_dim: this parameter is unused");
    LLK_ASSERT(!narrow_tile, "narrow_tile: this parameter is unused");
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK); // wait for pack to finish
   10808:	ttstallwait	33,8

    // RowMajor order
    TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
   1080c:	ttsetdmareg	0,0,0,8
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
   10810:	ttsetdmareg	0,512,0,16

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10814:	ttstallwait	128,1
        TT_WRCFG(get_packer_dest_offset_index(), p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
   10818:	lui	a5,0xb0048
   1081c:	addi	a5,a5,180 # b00480b4 <__runtime_args_end+0xb0027cb4>
   10820:	sw	a5,0(t4)
    TTI_DMANOP;
   10824:	ttdmanop
    TTI_DMANOP;
   10828:	ttdmanop
    TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
   1082c:	ttsetadcxy	4,0,0,0,0,11
    TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
   10830:	ttsetadczw	4,0,0,0,0,15
    store_blocking(&pc_buf_base[1], 0);
   10834:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10838:	addi	a5,a5,4
{
    tensix_sync();
    reset_dest_offset_id();
    _llk_init_packer_dest_offset_registers_<Dst>(face_r_dim, narrow_tile);
    packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
   1083c:	sw	zero,-1952(gp) # ffb00060 <pack_sync_tile_dst_ptr>
    asm volatile(
   10840:	sw	a4,0(a5)
   10844:	lw	a4,0(a5)
   10848:	and	zero,zero,a4
        if (is_opened)
   1084c:	lbu	a5,0(sp)
   10850:	beqz	a5,108a0 <run_kernel(RuntimeParams const&)+0x3b8>
    return p_reg[0];
   10854:	lui	a5,0xffb12
   10858:	lw	a0,496(a5) # ffb121f0 <__stack_top+0x111f0>
   1085c:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10860:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
   10864:	lw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10868:	addi	a2,a2,-1 # fff <__firmware_stack_size+0xdff>
   1086c:	lw	a5,-2008(gp) # ffb00028 <llk_profiler::buffer>
   10870:	lui	a6,0xba8fb
   10874:	and	a4,a4,a2
   10878:	lui	a2,0x2
   1087c:	or	a4,a4,a6
   10880:	sh2add	a5,a3,a5
   10884:	add	a5,a5,a2
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10888:	addi	a3,a3,2 # 41010002 <__runtime_args_end+0x40fefc02>
            --open_zone_cnt;
   1088c:	addi	a2,a1,-1 # 40ffffff <__runtime_args_end+0x40fdfbff>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10890:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10894:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10898:	sw	a0,4(a5)
            --open_zone_cnt;
   1089c:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        if constexpr (perf_counter_thread_active<run_type>())
        {
            asm volatile("" ::: "memory");
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u; // PERF_CNT_ALL
   108a0:	lui	t0,0xffb12
   108a4:	li	a5,2
   108a8:	sw	a5,60(t0) # ffb1203c <__stack_top+0x1103c>
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u; // TDMA_UNPACK
   108ac:	sw	a5,20(t0)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u; // L1
   108b0:	sw	a5,56(t0)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u; // TDMA_PACK
   108b4:	sw	a5,248(t0)
                {0xFFB12010u, 0xFFB12108u}, // 2 TDMA_UNPACK
                {0xFFB12034u, 0xFFB12118u}, // 3 L1
                {0xFFB120F4u, 0xFFB12110u}, // 4 TDMA_PACK
            };

            std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
   108b8:	lw	a7,4(sp)
            volatile std::uint32_t* bank_cycles    = reinterpret_cast<volatile std::uint32_t*>(cycles_base);
            volatile std::uint32_t* counter_counts = bank_cycles + PERF_COUNTERS_BANK_CYCLES_WORDS;

            // INSTRN OUT_L replicated to all banks: FPU/L1 OUT_L return 0 on 2nd+ zone
            // when counter_sel is high (HW quirk); INSTRN cycles agree within ±30 cyc.
            std::uint32_t shared_cycles = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
   108bc:	lw	a5,256(t0)
            std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
   108c0:	li	a4,860
   108c4:	mul	a7,a7,a4
   108c8:	lui	a4,0x169
   108cc:	addi	t3,a4,800 # 169320 <__runtime_args_end+0x148f20>
   108d0:	add	a7,a7,t3
            bank_cycles[0]              = shared_cycles;
   108d4:	sw	a5,0(a7) # aa8fb000 <__runtime_args_end+0xaa8dac00>
            bank_cycles[1]              = shared_cycles;
   108d8:	sw	a5,4(a7)
            bank_cycles[2]              = shared_cycles;
   108dc:	sw	a5,8(a7)
            bank_cycles[3]              = shared_cycles;
   108e0:	sw	a5,12(a7)
                if (bank_id == 3u)
                {
                    volatile std::uint32_t tt_reg_ptr* mux = reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12218u);
                    *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
                }
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   108e4:	lui	t5,0xffb00
   108e8:	lui	t1,0x20
            bank_cycles[4]              = shared_cycles;
   108ec:	sw	a5,16(a7)
            for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   108f0:	mv	s4,a4
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   108f4:	mv	t5,t5
   108f8:	addi	t1,t1,-256 # 1ff00 <__loader_init_end+0xaf00>
            std::uint32_t out_idx             = 0;
   108fc:	li	a2,0
                if (bank_id == 3u)
   10900:	li	t6,3
                std::uint32_t cw = cfg[i];
   10904:	lw	a5,0(a4)
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10908:	sh2add	a3,a2,a7
   1090c:	addi	a3,a3,20
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10910:	and	a6,a5,t1
                if (!(cw & 0x80000000u))
   10914:	bgez	a5,10954 <run_kernel(RuntimeParams const&)+0x46c>
                std::uint32_t bank_id    = cw & 0xFFu;
   10918:	zext.b	a0,a5
                ++out_idx;
   1091c:	addi	a2,a2,1 # 2001 <BRISC_LOCAL_MEM_LENGTH+0x1>
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10920:	sh3add	a1,a0,t5
                if (bank_id == 3u)
   10924:	bne	a0,t6,10940 <run_kernel(RuntimeParams const&)+0x458>
                    *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
   10928:	lw	a0,536(t0)
   1092c:	srli	a5,a5,0xd
   10930:	andi	a5,a5,112
   10934:	andi	a0,a0,-113
   10938:	or	a5,a5,a0
   1093c:	sw	a5,536(t0)
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10940:	lw	a0,0(a1)
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10944:	lw	a5,4(a1)
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10948:	sw	a6,0(a0)
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   1094c:	lw	a5,4(a5)
   10950:	sw	a5,0(a3)
            for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   10954:	addi	a4,a4,4
   10958:	bne	a4,t3,10904 <run_kernel(RuntimeParams const&)+0x41c>
            }

            std::uint32_t sync_addr                               = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE + PERF_COUNTERS_ZONE_DATA_BYTES;
   1095c:	lw	a5,4(sp)
   10960:	li	a4,860
   10964:	mul	a5,a5,a4
            *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10968:	li	a4,255
            std::uint32_t sync_addr                               = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE + PERF_COUNTERS_ZONE_DATA_BYTES;
   1096c:	add	a5,a5,s4
            *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10970:	sw	a4,1620(a5)
    std::uint32_t n = detail::next_zone_id;
   10974:	lw	a5,0(s0)
    for (std::uint32_t i = 0; i < n; ++i)
   10978:	beqz	a5,10d80 <run_kernel(RuntimeParams const&)+0x898>
        if (detail::zone_hashes[i] == hash_val)
   1097c:	lui	a4,0xbd77
   10980:	lw	a3,4(s0)
   10984:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
   10988:	beq	a3,a4,10a0c <run_kernel(RuntimeParams const&)+0x524>
    for (std::uint32_t i = 0; i < n; ++i)
   1098c:	li	a3,1
   10990:	beq	a5,a3,10d80 <run_kernel(RuntimeParams const&)+0x898>
        if (detail::zone_hashes[i] == hash_val)
   10994:	lw	a2,8(s0)
   10998:	beq	a2,a4,10d9c <run_kernel(RuntimeParams const&)+0x8b4>
    for (std::uint32_t i = 0; i < n; ++i)
   1099c:	li	a3,2
   109a0:	beq	a5,a3,10d80 <run_kernel(RuntimeParams const&)+0x898>
        if (detail::zone_hashes[i] == hash_val)
   109a4:	lw	a2,12(s0)
   109a8:	beq	a2,a4,10d9c <run_kernel(RuntimeParams const&)+0x8b4>
    for (std::uint32_t i = 0; i < n; ++i)
   109ac:	li	a3,3
   109b0:	beq	a5,a3,10d80 <run_kernel(RuntimeParams const&)+0x898>
        if (detail::zone_hashes[i] == hash_val)
   109b4:	lw	a2,16(s0)
   109b8:	beq	a2,a4,10d9c <run_kernel(RuntimeParams const&)+0x8b4>
    for (std::uint32_t i = 0; i < n; ++i)
   109bc:	li	a4,4
   109c0:	beq	a5,a4,10d80 <run_kernel(RuntimeParams const&)+0x898>
        if (detail::zone_hashes[i] == hash_val)
   109c4:	lui	a4,0xbd77
   109c8:	lw	a3,20(s0)
   109cc:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
   109d0:	beq	a3,a4,10dbc <run_kernel(RuntimeParams const&)+0x8d4>
    for (std::uint32_t i = 0; i < n; ++i)
   109d4:	li	a3,5
   109d8:	beq	a5,a3,10d80 <run_kernel(RuntimeParams const&)+0x898>
        if (detail::zone_hashes[i] == hash_val)
   109dc:	lw	a2,24(s0)
   109e0:	beq	a2,a4,10d9c <run_kernel(RuntimeParams const&)+0x8b4>
    for (std::uint32_t i = 0; i < n; ++i)
   109e4:	li	a3,6
   109e8:	beq	a5,a3,10d80 <run_kernel(RuntimeParams const&)+0x898>
        if (detail::zone_hashes[i] == hash_val)
   109ec:	lw	a2,28(s0)
   109f0:	beq	a2,a4,10d9c <run_kernel(RuntimeParams const&)+0x8b4>
    for (std::uint32_t i = 0; i < n; ++i)
   109f4:	li	a3,7
   109f8:	beq	a5,a3,10d80 <run_kernel(RuntimeParams const&)+0x898>
        if (detail::zone_hashes[i] == hash_val)
   109fc:	lw	a2,32(s0)
   10a00:	beq	a2,a4,10d9c <run_kernel(RuntimeParams const&)+0x8b4>
    if (n < PERF_COUNTERS_MAX_ZONES)
   10a04:	li	a4,8
   10a08:	bne	a5,a4,10d80 <run_kernel(RuntimeParams const&)+0x898>
    return 0;
   10a0c:	li	a5,0
    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
   10a10:	sw	a5,4(sp)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 1u; // PERF_CNT_ALL (INSTRN+FPU)
   10a14:	lui	a5,0xffb12
   10a18:	li	a4,1
   10a1c:	sw	a4,60(a5) # ffb1203c <__stack_top+0x1103c>
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 1u; // TDMA_UNPACK
   10a20:	sw	a4,20(a5)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 1u; // L1
   10a24:	sw	a4,56(a5)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 1u; // TDMA_PACK
   10a28:	sw	a4,248(a5)
    {
   10a2c:	sb	zero,0(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   10a30:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
   10a34:	lw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
   10a38:	li	a0,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   10a3c:	add	a2,a1,a3
   10a40:	addi	a2,a2,-1021
        if (!is_buffer_full())
   10a44:	bgeu	a0,a2,10a8c <run_kernel(RuntimeParams const&)+0x5a4>
   10a48:	lw	a0,496(a5)
   10a4c:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10a50:	lw	a2,-2008(gp) # ffb00028 <llk_profiler::buffer>
   10a54:	lui	a7,0xa21b2
   10a58:	lui	a6,0x2
            is_opened = true;
   10a5c:	sb	a4,0(sp)
            ++open_zone_cnt;
   10a60:	add	a1,a1,a4
   10a64:	sw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
   10a68:	sh2add	a2,a3,a2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10a6c:	add	a4,a2,a6
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10a70:	addi	a3,a3,2
   10a74:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10a78:	slli	a5,a5,0x14
   10a7c:	srli	a5,a5,0x14
   10a80:	or	a5,a5,a7
   10a84:	sw	a5,0(a4)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10a88:	sw	a0,4(a4)
constexpr std::uint32_t PERF_OUTPUT  = PERF_INPUT_C + 16 * 4096;

constexpr std::uint32_t PERF_ADDRESS(std::uint32_t buffer, std::uint32_t tile)
{
    std::uint32_t address = buffer + (tile % 16) * 4096; // Loop every 16 tiles, to prevent escaping memory
    return address / 16 - 1;                             // Correct the L1 Address for Tensix
   10a8c:	lui	a6,0x5
    }
}

inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);
   10a90:	lui	t1,0x508c0
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10a94:	lui	a4,0x45000
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10a98:	lui	a3,0x45800
   10a9c:	lw	a2,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
   10aa0:	addi	s5,a6,255 # 50ff <BRISC_LOCAL_MEM_LENGTH+0x30ff>
   10aa4:	addi	s4,a6,511
   10aa8:	addi	s0,a6,767
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10aac:	addi	a0,a4,24 # 45000018 <__runtime_args_end+0x44fdfc18>
   10ab0:	addi	s8,t1,1 # 508c0001 <__runtime_args_end+0x5089fc01>
   10ab4:	addi	s7,t1,2
   10ab8:	addi	s6,t1,3
   10abc:	addi	a6,a6,1023
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10ac0:	addi	a4,a4,25
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10ac4:	addi	a3,a3,25 # 45800019 <__runtime_args_end+0x457dfc19>
        asm volatile("" ::: "memory");
   10ac8:	li	a1,0
        TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2);
   10acc:	lui	t0,0x10144
    dest_offset_id = 1 - dest_offset_id;
   10ad0:	li	a7,1
   10ad4:	lui	t6,0xb0048
    return (dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO);
   10ad8:	lui	s9,0xb0088
                }
            }
        }
        else
        {
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
   10adc:	lui	t5,0x4
   10ae0:	lui	t3,0x10
    TTI_SEMWAIT(p_stall::STALL_TDMA, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
   10ae4:	ttsemwait	1,2,1
   10ae8:	srli	a5,a1,0x4
   10aec:	add	s10,a5,s5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10af0:	slli	s10,s10,0x8
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);
   10af4:	sw	t1,0(t4)
   10af8:	add	s10,s10,a0
   10afc:	sw	s10,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b00:	sw	a3,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10b04:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10b08:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b0c:	sw	a4,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10b10:	ttdmanop
    TTI_MOP(1, 0, 0); // run the double-loop template
   10b14:	ttmop	1,0,0

    program_packer_destination(address);

    ckernel::ckernel_template::run();

    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101); // reset z counters
   10b18:	ttsetadczw	4,0,0,0,0,5
   10b1c:	add	s10,a5,s4
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10b20:	slli	s10,s10,0x8
   10b24:	sw	s8,0(t4)
   10b28:	add	s10,s10,a0
   10b2c:	sw	s10,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b30:	sw	a3,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10b34:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10b38:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b3c:	sw	a4,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10b40:	ttdmanop
   10b44:	ttmop	1,0,0
   10b48:	ttsetadczw	4,0,0,0,0,5
   10b4c:	add	s10,a5,s0
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10b50:	slli	s10,s10,0x8
   10b54:	sw	s7,0(t4)
   10b58:	add	s10,s10,a0
   10b5c:	sw	s10,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b60:	sw	a3,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10b64:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10b68:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b6c:	sw	a4,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10b70:	ttdmanop
   10b74:	ttmop	1,0,0
   10b78:	ttsetadczw	4,0,0,0,0,5
   10b7c:	add	a5,a5,a6
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10b80:	slli	a5,a5,0x8
   10b84:	sw	s6,0(t4)
   10b88:	add	a5,a5,a0
   10b8c:	sw	a5,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b90:	sw	a3,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10b94:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10b98:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b9c:	sw	a4,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10ba0:	ttdmanop
   10ba4:	ttmop	1,0,0
   10ba8:	ttsetadczw	4,0,0,0,0,5
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK); // wait for pack to finish
   10bac:	ttstallwait	64,8
        TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2);
   10bb0:	andi	a5,a2,1
   10bb4:	add	a5,a5,t0
   10bb8:	sw	a5,0(t4)
    TTI_SEMGET(semaphore::t6_sem(index));
   10bbc:	ttsemget	2
    dest_offset_id = 1 - dest_offset_id;
   10bc0:	mv	s10,a2
   10bc4:	addi	a5,t6,180 # b00480b4 <__runtime_args_end+0xb0027cb4>
   10bc8:	sub	a2,a7,a2
    return (dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO);
   10bcc:	beq	s10,a7,10bd4 <run_kernel(RuntimeParams const&)+0x6ec>
   10bd0:	addi	a5,s9,180 # b00880b4 <__runtime_args_end+0xb0067cb4>
        TT_WRCFG(get_packer_dest_offset_index(), p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
   10bd4:	sw	a5,0(t4)
    TTI_DMANOP;
   10bd8:	ttdmanop
    TTI_DMANOP;
   10bdc:	ttdmanop
   10be0:	add	a1,a1,t5
   10be4:	bne	a1,t3,10ae4 <run_kernel(RuntimeParams const&)+0x5fc>
    store_blocking(&pc_buf_base[1], 0);
   10be8:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
   10bec:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
   10bf0:	addi	a4,a4,4
   10bf4:	sw	a2,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
    asm volatile(
   10bf8:	sw	a5,0(a4)
   10bfc:	lw	a5,0(a4)
   10c00:	and	zero,zero,a5
        if (is_opened)
   10c04:	lbu	a5,0(sp)
   10c08:	beqz	a5,10c58 <run_kernel(RuntimeParams const&)+0x770>
    return p_reg[0];
   10c0c:	lui	a5,0xffb12
   10c10:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10c14:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10c18:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
   10c1c:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
   10c20:	lw	a4,-2008(gp) # ffb00028 <llk_profiler::buffer>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10c24:	lui	a6,0xb21b2
   10c28:	lui	a0,0x2
   10c2c:	sh2add	a4,a3,a4
   10c30:	add	a4,a4,a0
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10c34:	addi	a3,a3,2
   10c38:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10c3c:	slli	a5,a5,0x14
   10c40:	srli	a5,a5,0x14
   10c44:	or	a5,a5,a6
   10c48:	sw	a5,0(a4)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10c4c:	sw	a1,4(a4)
            --open_zone_cnt;
   10c50:	addi	a2,a2,-1
   10c54:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u; // PERF_CNT_ALL
   10c58:	lui	t6,0xffb12
   10c5c:	li	a5,2
   10c60:	sw	a5,60(t6) # ffb1203c <__stack_top+0x1103c>
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u; // TDMA_UNPACK
   10c64:	sw	a5,20(t6)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u; // L1
   10c68:	sw	a5,56(t6)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u; // TDMA_PACK
   10c6c:	sw	a5,248(t6)
            std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
   10c70:	lw	a7,4(sp)
            std::uint32_t shared_cycles = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
   10c74:	lw	a5,256(t6)
            std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
   10c78:	li	a4,860
   10c7c:	mul	a7,a7,a4
   10c80:	lui	a4,0x169
   10c84:	addi	t3,a4,800 # 169320 <__runtime_args_end+0x148f20>
   10c88:	add	a7,a7,t3
            bank_cycles[0]              = shared_cycles;
   10c8c:	sw	a5,0(a7) # a21b2000 <__runtime_args_end+0xa2191c00>
            bank_cycles[1]              = shared_cycles;
   10c90:	sw	a5,4(a7)
            bank_cycles[2]              = shared_cycles;
   10c94:	sw	a5,8(a7)
            bank_cycles[3]              = shared_cycles;
   10c98:	sw	a5,12(a7)
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10c9c:	lui	t4,0xffb00
   10ca0:	lui	t1,0x20
            bank_cycles[4]              = shared_cycles;
   10ca4:	sw	a5,16(a7)
            for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   10ca8:	mv	t0,a4
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10cac:	mv	t4,t4
   10cb0:	addi	t1,t1,-256 # 1ff00 <__loader_init_end+0xaf00>
            std::uint32_t out_idx             = 0;
   10cb4:	li	a2,0
                if (bank_id == 3u)
   10cb8:	li	t5,3
                std::uint32_t cw = cfg[i];
   10cbc:	lw	a5,0(a4)
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10cc0:	sh2add	a3,a2,a7
   10cc4:	addi	a3,a3,20
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10cc8:	and	a6,a5,t1
                if (!(cw & 0x80000000u))
   10ccc:	bgez	a5,10d0c <run_kernel(RuntimeParams const&)+0x824>
                std::uint32_t bank_id    = cw & 0xFFu;
   10cd0:	zext.b	a0,a5
                ++out_idx;
   10cd4:	addi	a2,a2,1
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10cd8:	sh3add	a1,a0,t4
                if (bank_id == 3u)
   10cdc:	bne	a0,t5,10cf8 <run_kernel(RuntimeParams const&)+0x810>
                    *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
   10ce0:	lw	a0,536(t6)
   10ce4:	srli	a5,a5,0xd
   10ce8:	andi	a5,a5,112
   10cec:	andi	a0,a0,-113
   10cf0:	or	a5,a5,a0
   10cf4:	sw	a5,536(t6)
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10cf8:	lw	a0,0(a1)
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10cfc:	lw	a5,4(a1)
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10d00:	sw	a6,0(a0) # 2000 <BRISC_LOCAL_MEM_LENGTH>
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10d04:	lw	a5,4(a5)
   10d08:	sw	a5,0(a3)
            for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   10d0c:	addi	a4,a4,4
   10d10:	bne	a4,t3,10cbc <run_kernel(RuntimeParams const&)+0x7d4>
            std::uint32_t sync_addr                               = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE + PERF_COUNTERS_ZONE_DATA_BYTES;
   10d14:	lw	a5,4(sp)
   10d18:	li	a3,860
   10d1c:	mul	a5,a5,a3
            *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10d20:	li	a4,255
            std::uint32_t sync_addr                               = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE + PERF_COUNTERS_ZONE_DATA_BYTES;
   10d24:	add	a5,a5,t0
            *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10d28:	sw	a4,1620(a5)
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
   10d2c:	lw	ra,60(sp)
   10d30:	lw	s0,56(sp)
   10d34:	lw	s1,52(sp)
   10d38:	lw	s2,48(sp)
   10d3c:	lw	s3,44(sp)
   10d40:	lw	s4,40(sp)
   10d44:	lw	s5,36(sp)
   10d48:	lw	s6,32(sp)
   10d4c:	lw	s7,28(sp)
   10d50:	lw	s8,24(sp)
   10d54:	lw	s9,20(sp)
   10d58:	lw	s10,16(sp)
   10d5c:	addi	sp,sp,64
   10d60:	ret
        detail::zone_hashes[n] = hash_val;
   10d64:	lui	a3,0x7c867
   10d68:	sh2add	a4,a5,s0
   10d6c:	addi	a3,a3,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
        detail::next_zone_id   = n + 1;
   10d70:	addi	a2,a5,1
        detail::zone_hashes[n] = hash_val;
   10d74:	sw	a3,4(a4)
        detail::next_zone_id   = n + 1;
   10d78:	sw	a2,0(s0)
        return n;
   10d7c:	j	105bc <run_kernel(RuntimeParams const&)+0xd4>
        detail::zone_hashes[n] = hash_val;
   10d80:	lui	a4,0xbd77
   10d84:	sh2add	a2,a5,s0
   10d88:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
        detail::next_zone_id   = n + 1;
   10d8c:	addi	a3,a5,1
        detail::zone_hashes[n] = hash_val;
   10d90:	sw	a4,4(a2)
        detail::next_zone_id   = n + 1;
   10d94:	sw	a3,0(s0)
        return n;
   10d98:	j	10a10 <run_kernel(RuntimeParams const&)+0x528>
            return i;
   10d9c:	mv	a5,a3
   10da0:	j	10a10 <run_kernel(RuntimeParams const&)+0x528>
   10da4:	mv	a5,a3
   10da8:	j	105bc <run_kernel(RuntimeParams const&)+0xd4>
    LLK_ASSERT(
   10dac:	ebreak
   10db0:	j	10648 <run_kernel(RuntimeParams const&)+0x160>
   10db4:	li	a5,4
   10db8:	j	105bc <run_kernel(RuntimeParams const&)+0xd4>
   10dbc:	li	a5,4
   10dc0:	j	10a10 <run_kernel(RuntimeParams const&)+0x528>

00010dc4 <_init()>:
    }
}

void _init(void)
{
}
   10dc4:	ret

00010dc8 <_fini()>:

void _fini(void)
   10dc8:	ret

00010dcc <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
   10dcc:	lui	a5,0x20
   10dd0:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
   10dd4:	sb	a5,0(a0)
        (void)(dstc[i]);
   10dd8:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
   10ddc:	fence
}
   10de0:	ret

00010de4 <memset>:
   10de4:	li	t1,15
   10de8:	mv	a4,a0
   10dec:	bgeu	t1,a2,10e28 <memset+0x44>
   10df0:	andi	a5,a4,15
   10df4:	bnez	a5,10e94 <memset+0xb0>
   10df8:	bnez	a1,10e7c <memset+0x98>
   10dfc:	andi	a3,a2,-16
   10e00:	andi	a2,a2,15
   10e04:	add	a3,a3,a4
   10e08:	sw	a1,0(a4)
   10e0c:	sw	a1,4(a4)
   10e10:	sw	a1,8(a4)
   10e14:	sw	a1,12(a4)
   10e18:	addi	a4,a4,16
   10e1c:	bltu	a4,a3,10e08 <memset+0x24>
   10e20:	bnez	a2,10e28 <memset+0x44>
   10e24:	ret
   10e28:	sub	a3,t1,a2
   10e2c:	slli	a3,a3,0x2
   10e30:	auipc	t0,0x0
   10e34:	add	a3,a3,t0
   10e38:	jr	12(a3)
   10e3c:	sb	a1,14(a4)
   10e40:	sb	a1,13(a4)
   10e44:	sb	a1,12(a4)
   10e48:	sb	a1,11(a4)
   10e4c:	sb	a1,10(a4)
   10e50:	sb	a1,9(a4)
   10e54:	sb	a1,8(a4)
   10e58:	sb	a1,7(a4)
   10e5c:	sb	a1,6(a4)
   10e60:	sb	a1,5(a4)
   10e64:	sb	a1,4(a4)
   10e68:	sb	a1,3(a4)
   10e6c:	sb	a1,2(a4)
   10e70:	sb	a1,1(a4)
   10e74:	sb	a1,0(a4)
   10e78:	ret
   10e7c:	zext.b	a1,a1
   10e80:	slli	a3,a1,0x8
   10e84:	or	a1,a1,a3
   10e88:	slli	a3,a1,0x10
   10e8c:	or	a1,a1,a3
   10e90:	j	10dfc <memset+0x18>
   10e94:	slli	a3,a5,0x2
   10e98:	auipc	t0,0x0
   10e9c:	add	a3,a3,t0
   10ea0:	mv	t0,ra
   10ea4:	jalr	-96(a3)
   10ea8:	mv	ra,t0
   10eac:	addi	a5,a5,-16
   10eb0:	sub	a4,a4,a5
   10eb4:	add	a2,a2,a5
   10eb8:	bgeu	t1,a2,10e28 <memset+0x44>
   10ebc:	j	10df8 <memset+0x14>
