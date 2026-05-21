
/tmp/perf_overhead_artifacts/wc/PACK_ISOLATE/pack.elf:     file format elf32-littleriscv


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
   1012c:	jal	1104c <memset>
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
   104e8:	addi	sp,sp,-48
   104ec:	sw	s1,36(sp)
#endif
} // namespace detail

__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    std::uint32_t n = detail::next_zone_id;
   104f0:	addi	s1,gp,-1944 # ffb00068 <llk_perf::detail::next_zone_id>
   104f4:	lw	a5,0(s1)
   104f8:	sw	ra,44(sp)
   104fc:	sw	s0,40(sp)
   10500:	sw	s2,32(sp)
   10504:	sw	s3,28(sp)
   10508:	sw	s4,24(sp)
    for (std::uint32_t i = 0; i < n; ++i)
   1050c:	beqz	a5,10f5c <run_kernel(RuntimeParams const&)+0xa74>
    {
        if (detail::zone_hashes[i] == hash_val)
   10510:	lui	a4,0x7c867
   10514:	lw	a3,4(s1)
   10518:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
   1051c:	beq	a3,a4,105a0 <run_kernel(RuntimeParams const&)+0xb8>
    for (std::uint32_t i = 0; i < n; ++i)
   10520:	li	a3,1
   10524:	beq	a5,a3,10f5c <run_kernel(RuntimeParams const&)+0xa74>
        if (detail::zone_hashes[i] == hash_val)
   10528:	lw	a2,8(s1)
   1052c:	beq	a2,a4,10f9c <run_kernel(RuntimeParams const&)+0xab4>
    for (std::uint32_t i = 0; i < n; ++i)
   10530:	li	a3,2
   10534:	beq	a5,a3,10f5c <run_kernel(RuntimeParams const&)+0xa74>
        if (detail::zone_hashes[i] == hash_val)
   10538:	lw	a2,12(s1)
   1053c:	beq	a2,a4,10f9c <run_kernel(RuntimeParams const&)+0xab4>
    for (std::uint32_t i = 0; i < n; ++i)
   10540:	li	a3,3
   10544:	beq	a5,a3,10f5c <run_kernel(RuntimeParams const&)+0xa74>
        if (detail::zone_hashes[i] == hash_val)
   10548:	lw	a2,16(s1)
   1054c:	beq	a2,a4,10f9c <run_kernel(RuntimeParams const&)+0xab4>
    for (std::uint32_t i = 0; i < n; ++i)
   10550:	li	a3,4
   10554:	beq	a5,a3,10f5c <run_kernel(RuntimeParams const&)+0xa74>
        if (detail::zone_hashes[i] == hash_val)
   10558:	lw	a3,20(s1)
   1055c:	beq	a3,a4,1101c <run_kernel(RuntimeParams const&)+0xb34>
    for (std::uint32_t i = 0; i < n; ++i)
   10560:	li	a3,5
   10564:	beq	a5,a3,10f5c <run_kernel(RuntimeParams const&)+0xa74>
        if (detail::zone_hashes[i] == hash_val)
   10568:	lui	a4,0x7c867
   1056c:	lw	a2,24(s1)
   10570:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
   10574:	beq	a2,a4,10f9c <run_kernel(RuntimeParams const&)+0xab4>
    for (std::uint32_t i = 0; i < n; ++i)
   10578:	li	a3,6
   1057c:	beq	a5,a3,10f5c <run_kernel(RuntimeParams const&)+0xa74>
        if (detail::zone_hashes[i] == hash_val)
   10580:	lw	a2,28(s1)
   10584:	beq	a2,a4,10f9c <run_kernel(RuntimeParams const&)+0xab4>
    for (std::uint32_t i = 0; i < n; ++i)
   10588:	li	a3,7
   1058c:	beq	a5,a3,10f5c <run_kernel(RuntimeParams const&)+0xa74>
        if (detail::zone_hashes[i] == hash_val)
   10590:	lw	a2,32(s1)
   10594:	beq	a2,a4,10f9c <run_kernel(RuntimeParams const&)+0xab4>
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
   10598:	li	a4,8
   1059c:	bne	a5,a4,10f5c <run_kernel(RuntimeParams const&)+0xa74>
    {
        detail::zone_hashes[n] = hash_val;
        detail::next_zone_id   = n + 1;
        return n;
    }
    return 0;
   105a0:	li	a5,0
    perf_counter_scoped(const perf_counter_scoped&)            = delete;
    perf_counter_scoped(perf_counter_scoped&&)                 = delete;
    perf_counter_scoped& operator=(const perf_counter_scoped&) = delete;
    perf_counter_scoped& operator=(perf_counter_scoped&&)      = delete;

    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
   105a4:	sw	a5,4(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 1u; // PERF_CNT_ALL (INSTRN+FPU)
   105a8:	lui	a5,0xffb12
   105ac:	li	a4,1
   105b0:	sw	a4,60(a5) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 1u; // TDMA_UNPACK
   105b4:	sw	a4,20(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 1u; // L1
   105b8:	sw	a4,56(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 1u; // TDMA_PACK
   105bc:	sw	a4,248(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   105c0:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   105c4:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   105c8:	li	a3,14
   105cc:	zext.b	a4,a4
   105d0:	bltu	a3,a4,10fa4 <run_kernel(RuntimeParams const&)+0xabc>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   105d4:	sw	zero,32(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   105d8:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   105dc:	li	a3,14
   105e0:	zext.b	a4,a4
   105e4:	bltu	a3,a4,10fc0 <run_kernel(RuntimeParams const&)+0xad8>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   105e8:	sw	zero,32(a5)
    {
   105ec:	sb	zero,0(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   105f0:	lw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
   105f4:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
   105f8:	li	a3,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   105fc:	add	a5,a4,a2
   10600:	addi	a5,a5,-1021
        if (!is_buffer_full())
   10604:	bgeu	a3,a5,10654 <run_kernel(RuntimeParams const&)+0x16c>
    return p_reg[0];
   10608:	lui	a5,0xffb12
   1060c:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10610:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10614:	lw	a3,-2008(gp) # ffb00028 <llk_profiler::buffer>
            is_opened = true;
   10618:	li	a0,1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   1061c:	lui	a7,0xa91eb
   10620:	lui	a6,0x2
            is_opened = true;
   10624:	sb	a0,0(sp)
            ++open_zone_cnt;
   10628:	add	a2,a2,a0
   1062c:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
   10630:	sh2add	a3,a4,a3
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10634:	add	a3,a3,a6
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10638:	addi	a4,a4,2
   1063c:	sw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10640:	slli	a5,a5,0x14
   10644:	srli	a5,a5,0x14
   10648:	or	a5,a5,a7
   1064c:	sw	a5,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10650:	sw	a1,4(a3)
    LLK_ASSERT(
   10654:	li	a1,6
   10658:	mv	a0,a1
   1065c:	jal	10454 <ckernel::packer::is_packer_to_L1_conversion_supported(DataFormat, DataFormat)>
   10660:	beqz	a0,11014 <run_kernel(RuntimeParams const&)+0xb2c>
    if (cfg_state_id == 0)
   10664:	lw	a4,-1960(gp) # ffb00058 <ckernel::cfg_state_id>
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
   10668:	lui	a5,0xffef0
    if (cfg_state_id == 0)
   1066c:	beqz	a4,10674 <run_kernel(RuntimeParams const&)+0x18c>
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
   10670:	addi	a5,a5,896 # ffef0380 <__instrn_buffer+0xb0380>
    TT_SETDMAREG(0, LOWER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0)); // x-stride not used!
   10674:	lui	t4,0xffe40
   10678:	lui	a4,0x45000
   1067c:	mv	t4,t4
   10680:	addi	a4,a4,56 # 45000038 <__runtime_args_end+0x44fdfc38>
   10684:	sw	a4,0(t4) # ffe40000 <__instrn_buffer>
    TT_SETDMAREG(0, UPPER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
   10688:	lui	a4,0x45001
   1068c:	addi	a4,a4,57 # 45001039 <__runtime_args_end+0x44fe0c39>
   10690:	sw	a4,0(t4)
    TT_SETDMAREG(0, LOWER_HALFWORD((z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP1));
   10694:	lui	a4,0x45010
   10698:	addi	a4,a4,58 # 4501003a <__runtime_args_end+0x44fefc3a>
   1069c:	sw	a4,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD((w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, HI_16(p_gpr_pack::TMP1));
   106a0:	lui	a4,0x45040
   106a4:	addi	a4,a4,59 # 4504003b <__runtime_args_end+0x4501fc3b>
   106a8:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   106ac:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
   106b0:	ttwrcfg	28,0,12
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
   106b4:	ttwrcfg	29,0,13
    TTI_NOP;
   106b8:	ttnop
    TTI_NOP;
   106bc:	ttnop
    TTI_ATGETM(index);
   106c0:	ttatgetm	0
    std::uint8_t mask_b2 = (Mask >> 16) & 0xff;

    if (mask_b2 != 0)
    {
        std::uint8_t data_b2 = (wrdata) & 0xff;
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   106c4:	lui	a4,0xb5800
   106c8:	addi	a4,a4,71 # b5800047 <__runtime_args_end+0xb57dfc47>
   106cc:	sw	a4,0(t4)
    wrdata >>= 8;
    std::uint8_t mask_b3 = (Mask >> 24) & 0xff;
    if (mask_b3 != 0)
    {
        std::uint8_t data_b3 = (wrdata) & 0xff;
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
   106d0:	lui	a4,0xb61e1
   106d4:	addi	a4,a4,-1023 # b61e0c01 <__runtime_args_end+0xb61c0801>
   106d8:	sw	a4,0(t4)
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
   106dc:	lui	a4,0xb3fc0
   106e0:	addi	a4,a4,2 # b3fc0002 <__runtime_args_end+0xb3f9fc02>
   106e4:	sw	a4,0(t4)
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
   106e8:	lui	a4,0xb4ff0
   106ec:	addi	a4,a4,2 # b4ff0002 <__runtime_args_end+0xb4fcfc02>
   106f0:	sw	a4,0(t4)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   106f4:	lui	a4,0xb53f0
   106f8:	addi	a4,a4,2 # b53f0002 <__runtime_args_end+0xb53cfc02>
   106fc:	sw	a4,0(t4)
    TTI_ATRELM(index);
   10700:	ttatrelm	0
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   10704:	lui	a4,0xb5100
   10708:	addi	a4,a4,71 # b5100047 <__runtime_args_end+0xb50dfc47>
   1070c:	sw	a4,0(t4)
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
   10710:	lui	a4,0xb6ff0
   10714:	addi	a4,a4,71 # b6ff0047 <__runtime_args_end+0xb6fcfc47>
   10718:	sw	a4,0(t4)
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
   1071c:	lui	a3,0x40
   10720:	sw	a3,272(a5)
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
   10724:	li	a4,1633
   10728:	sw	a4,280(a5)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
   1072c:	ttstallwait	128,8
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
   10730:	lui	a4,0xb3040
   10734:	addi	a4,a4,70 # b3040046 <__runtime_args_end+0xb301fc46>
   10738:	sw	a4,0(t4)
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
   1073c:	lui	a4,0xb5080
   10740:	addi	a4,a4,71 # b5080047 <__runtime_args_end+0xb505fc47>
   10744:	sw	a4,0(t4)
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
   10748:	lw	a4,-2000(gp) # ffb00030 <ckernel::regfile>
    cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] = dest_rd_ctrl.val;
   1074c:	li	a2,1
   10750:	sw	a2,72(a5)
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
   10754:	sw	a3,208(a4)
    volatile std::uint32_t foo     = 0x0;
   10758:	sw	zero,12(sp)
    *fooptr                        = regfile[index];
   1075c:	lw	a0,208(a4)
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
   10760:	lui	a1,0x1
    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
   10764:	lui	a3,0x10
   10768:	addi	a3,a3,-1 # ffff <BRISC_LOCAL_MEM_LENGTH+0xdfff>
    cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x0; // All packers use row set mapping 0, edge offset 0 mask
   1076c:	sw	zero,80(a5)
    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
   10770:	li	a2,1024
   10774:	sw	a0,12(sp)
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
   10778:	sw	a1,112(a5)
    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
   1077c:	sw	a3,96(a5)
    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
   10780:	sw	a2,64(a4)
   10784:	lw	a5,76(a4)
    regfile[p_gpr_pack::TILE_HEADER + 1] = 0;
   10788:	sw	zero,68(a4)
    regfile[p_gpr_pack::TILE_HEADER + 2] = 0;
   1078c:	sw	zero,72(a4)
    regfile[p_gpr_pack::TILE_HEADER + 3] = 0;
   10790:	sw	zero,76(a4)
    volatile std::uint32_t foo     = 0x0;
   10794:	sw	zero,8(sp)
    *fooptr                        = regfile[index];
   10798:	sw	a5,8(sp)
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
   1079c:	ttsetadcxx	4,15,0
        ADDR_MOD_PACK_SEC0_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC1_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC2_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC3_YsrcIncr_ADDR32};

    // Program source and dest registers
    __attribute__((always_inline)) inline void set(const std::uint8_t mod_index) const
    {
        TTI_SETC16(addr_mod_pack_reg_addr[mod_index], pack_val());
   107a0:	ttsetc16	37,260
   107a4:	ttsetc16	38,10272
   107a8:	ttsetc16	39,4384
    store_blocking(&pc_buf_base[2], 0);
   107ac:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
   107b0:	li	a4,0
    store_blocking(&pc_buf_base[2], 0);
   107b4:	addi	a5,a5,8
    asm volatile(
   107b8:	mv	a3,a4
   107bc:	sw	a3,0(a5)
   107c0:	lw	a3,0(a5)
   107c4:	and	zero,zero,a3
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
   107c8:	lui	a5,0xffb80
   107cc:	li	a3,4
   107d0:	sw	a3,0(a5) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
   107d4:	sw	a3,4(a5)
    mop_cfg[2] = m_start_op0;
   107d8:	lui	a3,0x2000
   107dc:	sw	a3,8(a5)
    mop_cfg[3] = m_end_op0;
   107e0:	sw	a3,12(a5)
    mop_cfg[4] = m_end_op1;
   107e4:	sw	a3,16(a5)
    mop_cfg[5] = m_loop_op0;
   107e8:	lui	a2,0x41000
   107ec:	sw	a2,20(a5)
    mop_cfg[6] = m_loop_op1;
    mop_cfg[7] = m_loop0_last_instr;
   107f0:	lui	a2,0x41008
    mop_cfg[6] = m_loop_op1;
   107f4:	sw	a3,24(a5)
    mop_cfg[7] = m_loop0_last_instr;
   107f8:	addi	a3,a2,1 # 41008001 <__runtime_args_end+0x40fe7c01>
   107fc:	sw	a3,28(a5)
    mop_cfg[8] = m_loop1_last_instr;
   10800:	lui	a3,0x41010
   10804:	sw	a3,32(a5)
    store_blocking(&pc_buf_base[1], 0);
   10808:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
   1080c:	mv	a3,a4
    store_blocking(&pc_buf_base[1], 0);
   10810:	addi	a5,a5,4
    asm volatile(
   10814:	sw	a3,0(a5)
   10818:	lw	a3,0(a5)
   1081c:	and	zero,zero,a3
    dest_offset_id = 0;
   10820:	sw	zero,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
inline void _llk_init_packer_dest_offset_registers_(
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM, [[maybe_unused]] const bool narrow_tile = false)
{
    LLK_ASSERT(face_r_dim == FACE_R_DIM, "face_r_dim: this parameter is unused");
    LLK_ASSERT(!narrow_tile, "narrow_tile: this parameter is unused");
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK); // wait for pack to finish
   10824:	ttstallwait	33,8

    // RowMajor order
    TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
   10828:	ttsetdmareg	0,0,0,8
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
   1082c:	ttsetdmareg	0,512,0,16

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10830:	ttstallwait	128,1
        TT_WRCFG(get_packer_dest_offset_index(), p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
   10834:	lui	a5,0xb0048
   10838:	addi	a5,a5,180 # b00480b4 <__runtime_args_end+0xb0027cb4>
   1083c:	sw	a5,0(t4)
    TTI_DMANOP;
   10840:	ttdmanop
    TTI_DMANOP;
   10844:	ttdmanop
    TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
   10848:	ttsetadcxy	4,0,0,0,0,11
    TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
   1084c:	ttsetadczw	4,0,0,0,0,15
    store_blocking(&pc_buf_base[1], 0);
   10850:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10854:	addi	a5,a5,4
{
    tensix_sync();
    reset_dest_offset_id();
    _llk_init_packer_dest_offset_registers_<Dst>(face_r_dim, narrow_tile);
    packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
   10858:	sw	zero,-1952(gp) # ffb00060 <pack_sync_tile_dst_ptr>
    asm volatile(
   1085c:	sw	a4,0(a5)
   10860:	lw	a4,0(a5)
   10864:	and	zero,zero,a4
        if (is_opened)
   10868:	lbu	a5,0(sp)
   1086c:	beqz	a5,108bc <run_kernel(RuntimeParams const&)+0x3d4>
    return p_reg[0];
   10870:	lui	a5,0xffb12
   10874:	lw	a0,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10878:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   1087c:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
   10880:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10884:	addi	a1,a1,-1 # fff <__firmware_stack_size+0xdff>
   10888:	lw	a5,-2008(gp) # ffb00028 <llk_profiler::buffer>
   1088c:	lui	a6,0xb91eb
   10890:	and	a4,a4,a1
   10894:	lui	a1,0x2
   10898:	or	a4,a4,a6
            --open_zone_cnt;
   1089c:	addi	a2,a2,-1
   108a0:	sh2add	a5,a3,a5
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   108a4:	add	a5,a5,a1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   108a8:	addi	a3,a3,2 # 41010002 <__runtime_args_end+0x40fefc02>
   108ac:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   108b0:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   108b4:	sw	a0,4(a5)
            --open_zone_cnt;
   108b8:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        asm volatile("" ::: "memory");
        if constexpr (is_active_perf_thread<run_type>())
        {
            freeze_and_read_all_counters(zone_id);
   108bc:	lw	t2,4(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u;
   108c0:	lui	t0,0xffb12
   108c4:	li	a5,2
   108c8:	sw	a5,60(t0) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u;
   108cc:	sw	a5,20(t0)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u;
   108d0:	sw	a5,56(t0)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u;
   108d4:	sw	a5,248(t0)
    std::uint32_t shared_cycles            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
   108d8:	lw	a5,256(t0)
    std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
   108dc:	li	a4,860
   108e0:	mul	t2,t2,a4
   108e4:	lui	a4,0x169
   108e8:	addi	t3,a4,800 # 169320 <__runtime_args_end+0x148f20>
   108ec:	add	a7,t2,t3
    bank_cycles[0]                         = shared_cycles;
   108f0:	sw	a5,0(a7) # a91eb000 <__runtime_args_end+0xa91cac00>
    bank_cycles[1]                         = shared_cycles;
   108f4:	sw	a5,4(a7)
    bank_cycles[2]                         = shared_cycles;
   108f8:	sw	a5,8(a7)
    bank_cycles[3]                         = shared_cycles;
   108fc:	sw	a5,12(a7)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10900:	lui	t5,0xffb00
   10904:	lui	t1,0x20
    bank_cycles[4]                         = shared_cycles;
   10908:	sw	a5,16(a7)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   1090c:	mv	s4,a4
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10910:	mv	t5,t5
   10914:	addi	t1,t1,-256 # 1ff00 <__loader_init_end+0xaf00>
    std::uint32_t out_idx             = 0;
   10918:	li	a2,0
        if (bank_id == 3u)
   1091c:	li	t6,3
        std::uint32_t cw = cfg[i];
   10920:	lw	a5,0(a4)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10924:	sh2add	a3,a2,a7
   10928:	addi	a3,a3,20
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   1092c:	and	a6,a5,t1
        if (!(cw & 0x80000000u))
   10930:	bgez	a5,10970 <run_kernel(RuntimeParams const&)+0x488>
        std::uint32_t bank_id    = cw & 0xFFu;
   10934:	zext.b	a0,a5
        ++out_idx;
   10938:	addi	a2,a2,1
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   1093c:	sh3add	a1,a0,t5
        if (bank_id == 3u)
   10940:	bne	a0,t6,1095c <run_kernel(RuntimeParams const&)+0x474>
            *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
   10944:	lw	a0,536(t0)
   10948:	srli	a5,a5,0xd
   1094c:	andi	a5,a5,112
   10950:	andi	a0,a0,-113
   10954:	or	a5,a5,a0
   10958:	sw	a5,536(t0)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   1095c:	lw	a0,0(a1) # 2000 <BRISC_LOCAL_MEM_LENGTH>
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10960:	lw	a5,4(a1)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10964:	sw	a6,0(a0)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10968:	lw	a5,4(a5)
   1096c:	sw	a5,0(a3)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   10970:	addi	a4,a4,4
   10974:	bne	a4,t3,10920 <run_kernel(RuntimeParams const&)+0x438>
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10978:	li	a5,255
    return perf_counters_zone_data_addr(zone) + PERF_COUNTERS_ZONE_DATA_BYTES;
   1097c:	add	t2,t2,s4
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10980:	sw	a5,1620(t2)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10984:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10988:	li	a3,14
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   1098c:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10990:	zext.b	a4,a4
   10994:	bltu	a3,a4,11008 <run_kernel(RuntimeParams const&)+0xb20>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10998:	sw	zero,40(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   1099c:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   109a0:	li	a3,14
   109a4:	zext.b	a4,a4
   109a8:	bltu	a3,a4,10ffc <run_kernel(RuntimeParams const&)+0xb14>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   109ac:	sw	zero,40(a5)
    std::uint32_t n = detail::next_zone_id;
   109b0:	lw	a5,0(s1)
    for (std::uint32_t i = 0; i < n; ++i)
   109b4:	beqz	a5,10f78 <run_kernel(RuntimeParams const&)+0xa90>
        if (detail::zone_hashes[i] == hash_val)
   109b8:	lui	a4,0xbd77
   109bc:	lw	a3,4(s1)
   109c0:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
   109c4:	beq	a3,a4,10a48 <run_kernel(RuntimeParams const&)+0x560>
    for (std::uint32_t i = 0; i < n; ++i)
   109c8:	li	a3,1
   109cc:	beq	a5,a3,10f78 <run_kernel(RuntimeParams const&)+0xa90>
        if (detail::zone_hashes[i] == hash_val)
   109d0:	lw	a2,8(s1)
   109d4:	beq	a2,a4,10f94 <run_kernel(RuntimeParams const&)+0xaac>
    for (std::uint32_t i = 0; i < n; ++i)
   109d8:	li	a3,2
   109dc:	beq	a5,a3,10f78 <run_kernel(RuntimeParams const&)+0xa90>
        if (detail::zone_hashes[i] == hash_val)
   109e0:	lw	a2,12(s1)
   109e4:	beq	a2,a4,10f94 <run_kernel(RuntimeParams const&)+0xaac>
    for (std::uint32_t i = 0; i < n; ++i)
   109e8:	li	a3,3
   109ec:	beq	a5,a3,10f78 <run_kernel(RuntimeParams const&)+0xa90>
        if (detail::zone_hashes[i] == hash_val)
   109f0:	lw	a2,16(s1)
   109f4:	beq	a2,a4,10f94 <run_kernel(RuntimeParams const&)+0xaac>
    for (std::uint32_t i = 0; i < n; ++i)
   109f8:	li	a3,4
   109fc:	beq	a5,a3,10f78 <run_kernel(RuntimeParams const&)+0xa90>
        if (detail::zone_hashes[i] == hash_val)
   10a00:	lw	a3,20(s1)
   10a04:	beq	a3,a4,11024 <run_kernel(RuntimeParams const&)+0xb3c>
    for (std::uint32_t i = 0; i < n; ++i)
   10a08:	li	a3,5
   10a0c:	beq	a5,a3,10f78 <run_kernel(RuntimeParams const&)+0xa90>
        if (detail::zone_hashes[i] == hash_val)
   10a10:	lui	a4,0xbd77
   10a14:	lw	a2,24(s1)
   10a18:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
   10a1c:	beq	a2,a4,10f94 <run_kernel(RuntimeParams const&)+0xaac>
    for (std::uint32_t i = 0; i < n; ++i)
   10a20:	li	a3,6
   10a24:	beq	a5,a3,10f78 <run_kernel(RuntimeParams const&)+0xa90>
        if (detail::zone_hashes[i] == hash_val)
   10a28:	lw	a2,28(s1)
   10a2c:	beq	a2,a4,10f94 <run_kernel(RuntimeParams const&)+0xaac>
    for (std::uint32_t i = 0; i < n; ++i)
   10a30:	li	a3,7
   10a34:	beq	a5,a3,10f78 <run_kernel(RuntimeParams const&)+0xa90>
        if (detail::zone_hashes[i] == hash_val)
   10a38:	lw	a2,32(s1)
   10a3c:	beq	a2,a4,10f94 <run_kernel(RuntimeParams const&)+0xaac>
    if (n < PERF_COUNTERS_MAX_ZONES)
   10a40:	li	a4,8
   10a44:	bne	a5,a4,10f78 <run_kernel(RuntimeParams const&)+0xa90>
    return 0;
   10a48:	li	a5,0
    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
   10a4c:	sw	a5,4(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 1u; // PERF_CNT_ALL (INSTRN+FPU)
   10a50:	lui	a5,0xffb12
   10a54:	li	a4,1
   10a58:	sw	a4,60(a5) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 1u; // TDMA_UNPACK
   10a5c:	sw	a4,20(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 1u; // L1
   10a60:	sw	a4,56(a5)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 1u; // TDMA_PACK
   10a64:	sw	a4,248(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10a68:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10a6c:	li	a3,14
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10a70:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10a74:	zext.b	a4,a4
   10a78:	bltu	a3,a4,10ff0 <run_kernel(RuntimeParams const&)+0xb08>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10a7c:	sw	zero,32(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10a80:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10a84:	li	a3,14
   10a88:	zext.b	a4,a4
   10a8c:	bltu	a3,a4,10fe4 <run_kernel(RuntimeParams const&)+0xafc>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10a90:	sw	zero,32(a5)
    {
   10a94:	sb	zero,0(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   10a98:	lw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
   10a9c:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
   10aa0:	li	a3,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
   10aa4:	add	a5,a2,a4
   10aa8:	addi	a5,a5,-1021
        if (!is_buffer_full())
   10aac:	bgeu	a3,a5,10afc <run_kernel(RuntimeParams const&)+0x614>
    return p_reg[0];
   10ab0:	lui	a5,0xffb12
   10ab4:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10ab8:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10abc:	lw	a3,-2008(gp) # ffb00028 <llk_profiler::buffer>
            is_opened = true;
   10ac0:	li	a0,1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10ac4:	lui	a7,0xac6d0
   10ac8:	lui	a6,0x2
            is_opened = true;
   10acc:	sb	a0,0(sp)
            ++open_zone_cnt;
   10ad0:	add	a2,a2,a0
   10ad4:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
   10ad8:	sh2add	a3,a4,a3
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10adc:	add	a3,a3,a6
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10ae0:	addi	a4,a4,2
   10ae4:	sw	a4,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10ae8:	slli	a5,a5,0x14
   10aec:	srli	a5,a5,0x14
   10af0:	or	a5,a5,a7
   10af4:	sw	a5,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10af8:	sw	a1,4(a3)
    }
}

inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);
   10afc:	lui	a3,0x508c0
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10b00:	lui	a5,0x45510
   10b04:	sw	a3,0(t4)
   10b08:	addi	a5,a5,-232 # 4550ff18 <__runtime_args_end+0x454efb18>
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b0c:	lui	a4,0x45800
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10b10:	sw	a5,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b14:	addi	a4,a4,25 # 45800019 <__runtime_args_end+0x457dfc19>
   10b18:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10b1c:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10b20:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b24:	lui	a5,0x45000
   10b28:	addi	a5,a5,25 # 45000019 <__runtime_args_end+0x44fdfc19>
   10b2c:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10b30:	ttdmanop
    TTI_MOP(1, 0, 0); // run the double-loop template
   10b34:	ttmop	1,0,0

    program_packer_destination(address);

    ckernel::ckernel_template::run();

    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101); // reset z counters
   10b38:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10b3c:	lui	a2,0x45520
   10b40:	addi	a0,a3,1 # 508c0001 <__runtime_args_end+0x5089fc01>
   10b44:	sw	a0,0(t4)
   10b48:	addi	a2,a2,-232 # 4551ff18 <__runtime_args_end+0x454ffb18>
   10b4c:	sw	a2,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b50:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10b54:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10b58:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b5c:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10b60:	ttdmanop
   10b64:	ttmop	1,0,0
   10b68:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10b6c:	lui	a2,0x45530
   10b70:	addi	a1,a3,2
   10b74:	sw	a1,0(t4)
   10b78:	addi	a2,a2,-232 # 4552ff18 <__runtime_args_end+0x4550fb18>
   10b7c:	sw	a2,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b80:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10b84:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10b88:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10b8c:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10b90:	ttdmanop
   10b94:	ttmop	1,0,0
   10b98:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10b9c:	lui	a6,0x45540
   10ba0:	addi	a2,a3,3
   10ba4:	sw	a2,0(t4)
   10ba8:	addi	a6,a6,-232 # 4553ff18 <__runtime_args_end+0x4551fb18>
   10bac:	sw	a6,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10bb0:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10bb4:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10bb8:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10bbc:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10bc0:	ttdmanop
   10bc4:	ttmop	1,0,0
   10bc8:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10bcc:	lui	a6,0x45550
   10bd0:	sw	a3,0(t4)
   10bd4:	addi	a6,a6,-232 # 4554ff18 <__runtime_args_end+0x4552fb18>
   10bd8:	sw	a6,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10bdc:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10be0:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10be4:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10be8:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10bec:	ttdmanop
   10bf0:	ttmop	1,0,0
   10bf4:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10bf8:	lui	a6,0x45560
   10bfc:	sw	a0,0(t4)
   10c00:	addi	a6,a6,-232 # 4555ff18 <__runtime_args_end+0x4553fb18>
   10c04:	sw	a6,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10c08:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10c0c:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10c10:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10c14:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10c18:	ttdmanop
   10c1c:	ttmop	1,0,0
   10c20:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10c24:	lui	a6,0x45570
   10c28:	sw	a1,0(t4)
   10c2c:	addi	a6,a6,-232 # 4556ff18 <__runtime_args_end+0x4554fb18>
   10c30:	sw	a6,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10c34:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10c38:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10c3c:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10c40:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10c44:	ttdmanop
   10c48:	ttmop	1,0,0
   10c4c:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10c50:	lui	a6,0x45580
   10c54:	sw	a2,0(t4)
   10c58:	addi	a6,a6,-232 # 4557ff18 <__runtime_args_end+0x4555fb18>
   10c5c:	sw	a6,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10c60:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10c64:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10c68:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10c6c:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10c70:	ttdmanop
   10c74:	ttmop	1,0,0
   10c78:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10c7c:	lui	a6,0x45590
   10c80:	sw	a3,0(t4)
   10c84:	addi	a6,a6,-232 # 4558ff18 <__runtime_args_end+0x4556fb18>
   10c88:	sw	a6,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10c8c:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10c90:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10c94:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10c98:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10c9c:	ttdmanop
   10ca0:	ttmop	1,0,0
   10ca4:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10ca8:	lui	a6,0x455a0
   10cac:	sw	a0,0(t4)
   10cb0:	addi	a6,a6,-232 # 4559ff18 <__runtime_args_end+0x4557fb18>
   10cb4:	sw	a6,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10cb8:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10cbc:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10cc0:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10cc4:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10cc8:	ttdmanop
   10ccc:	ttmop	1,0,0
   10cd0:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10cd4:	lui	a6,0x455b0
   10cd8:	sw	a1,0(t4)
   10cdc:	addi	a6,a6,-232 # 455aff18 <__runtime_args_end+0x4558fb18>
   10ce0:	sw	a6,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10ce4:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10ce8:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10cec:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10cf0:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10cf4:	ttdmanop
   10cf8:	ttmop	1,0,0
   10cfc:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10d00:	lui	a6,0x455c0
   10d04:	sw	a2,0(t4)
   10d08:	addi	a6,a6,-232 # 455bff18 <__runtime_args_end+0x4559fb18>
   10d0c:	sw	a6,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10d10:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10d14:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10d18:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10d1c:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10d20:	ttdmanop
   10d24:	ttmop	1,0,0
   10d28:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10d2c:	lui	a6,0x455d0
   10d30:	sw	a3,0(t4)
   10d34:	addi	a3,a6,-232 # 455cff18 <__runtime_args_end+0x455afb18>
   10d38:	sw	a3,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10d3c:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10d40:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10d44:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10d48:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10d4c:	ttdmanop
   10d50:	ttmop	1,0,0
   10d54:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10d58:	lui	a3,0x455e0
   10d5c:	sw	a0,0(t4)
   10d60:	addi	a3,a3,-232 # 455dff18 <__runtime_args_end+0x455bfb18>
   10d64:	sw	a3,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10d68:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10d6c:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10d70:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10d74:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10d78:	ttdmanop
   10d7c:	ttmop	1,0,0
   10d80:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10d84:	lui	a3,0x455f0
   10d88:	sw	a1,0(t4)
   10d8c:	addi	a3,a3,-232 # 455eff18 <__runtime_args_end+0x455cfb18>
   10d90:	sw	a3,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10d94:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10d98:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10d9c:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10da0:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10da4:	ttdmanop
   10da8:	ttmop	1,0,0
   10dac:	ttsetadczw	4,0,0,0,0,5
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
   10db0:	lui	a3,0x45600
   10db4:	sw	a2,0(t4)
   10db8:	addi	a3,a3,-232 # 455fff18 <__runtime_args_end+0x455dfb18>
   10dbc:	sw	a3,0(t4)
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10dc0:	sw	a4,0(t4)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
   10dc4:	ttstallwait	128,1
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   10dc8:	ttwrcfg	12,0,69
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   10dcc:	sw	a5,0(t4)
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
   10dd0:	ttdmanop
   10dd4:	ttmop	1,0,0
   10dd8:	ttsetadczw	4,0,0,0,0,5
    store_blocking(&pc_buf_base[1], 0);
   10ddc:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
   10de0:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
   10de4:	addi	a4,a4,4
    asm volatile(
   10de8:	sw	a5,0(a4)
   10dec:	lw	a5,0(a4)
   10df0:	and	zero,zero,a5
        if (is_opened)
   10df4:	lbu	a5,0(sp)
   10df8:	beqz	a5,10e48 <run_kernel(RuntimeParams const&)+0x960>
    return p_reg[0];
   10dfc:	lui	a5,0xffb12
   10e00:	lw	a1,496(a5) # ffb121f0 <__stack_top+0x111f0>
   10e04:	lw	a5,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10e08:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
   10e0c:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
   10e10:	lw	a4,-2008(gp) # ffb00028 <llk_profiler::buffer>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10e14:	lui	a6,0xbc6d0
   10e18:	lui	a0,0x2
   10e1c:	sh2add	a4,a3,a4
   10e20:	add	a4,a4,a0
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10e24:	addi	a3,a3,2
   10e28:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
   10e2c:	slli	a5,a5,0x14
   10e30:	srli	a5,a5,0x14
   10e34:	or	a5,a5,a6
   10e38:	sw	a5,0(a4)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
   10e3c:	sw	a1,4(a4)
            --open_zone_cnt;
   10e40:	addi	a2,a2,-1
   10e44:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
            freeze_and_read_all_counters(zone_id);
   10e48:	lw	t0,4(sp)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u;
   10e4c:	lui	t6,0xffb12
   10e50:	li	a5,2
   10e54:	sw	a5,60(t6) # ffb1203c <__stack_top+0x1103c>
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u;
   10e58:	sw	a5,20(t6)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u;
   10e5c:	sw	a5,56(t6)
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u;
   10e60:	sw	a5,248(t6)
    std::uint32_t shared_cycles            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
   10e64:	lw	a5,256(t6)
    std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
   10e68:	li	a4,860
   10e6c:	mul	t0,t0,a4
   10e70:	lui	a4,0x169
   10e74:	addi	t3,a4,800 # 169320 <__runtime_args_end+0x148f20>
   10e78:	add	a7,t0,t3
    bank_cycles[0]                         = shared_cycles;
   10e7c:	sw	a5,0(a7) # ac6d0000 <__runtime_args_end+0xac6afc00>
    bank_cycles[1]                         = shared_cycles;
   10e80:	sw	a5,4(a7)
    bank_cycles[2]                         = shared_cycles;
   10e84:	sw	a5,8(a7)
    bank_cycles[3]                         = shared_cycles;
   10e88:	sw	a5,12(a7)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10e8c:	lui	t4,0xffb00
   10e90:	lui	t1,0x20
    bank_cycles[4]                         = shared_cycles;
   10e94:	sw	a5,16(a7)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   10e98:	mv	t2,a4
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10e9c:	mv	t4,t4
   10ea0:	addi	t1,t1,-256 # 1ff00 <__loader_init_end+0xaf00>
    std::uint32_t out_idx             = 0;
   10ea4:	li	a2,0
        if (bank_id == 3u)
   10ea8:	li	t5,3
        std::uint32_t cw = cfg[i];
   10eac:	lw	a5,0(a4)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10eb0:	sh2add	a3,a2,a7
   10eb4:	addi	a3,a3,20
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10eb8:	and	a6,a5,t1
        if (!(cw & 0x80000000u))
   10ebc:	bgez	a5,10efc <run_kernel(RuntimeParams const&)+0xa14>
        std::uint32_t bank_id    = cw & 0xFFu;
   10ec0:	zext.b	a0,a5
        ++out_idx;
   10ec4:	addi	a2,a2,1
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10ec8:	sh3add	a1,a0,t4
        if (bank_id == 3u)
   10ecc:	bne	a0,t5,10ee8 <run_kernel(RuntimeParams const&)+0xa00>
            *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
   10ed0:	lw	a0,536(t6)
   10ed4:	srli	a5,a5,0xd
   10ed8:	andi	a5,a5,112
   10edc:	andi	a0,a0,-113
   10ee0:	or	a5,a5,a0
   10ee4:	sw	a5,536(t6)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10ee8:	lw	a0,0(a1)
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10eec:	lw	a5,4(a1)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
   10ef0:	sw	a6,0(a0) # 2000 <BRISC_LOCAL_MEM_LENGTH>
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
   10ef4:	lw	a5,4(a5)
   10ef8:	sw	a5,0(a3)
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
   10efc:	addi	a4,a4,4
   10f00:	bne	a4,t3,10eac <run_kernel(RuntimeParams const&)+0x9c4>
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10f04:	li	a5,255
    return perf_counters_zone_data_addr(zone) + PERF_COUNTERS_ZONE_DATA_BYTES;
   10f08:	add	t0,t0,t2
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
   10f0c:	sw	a5,1620(t0)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10f10:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10f14:	li	a3,14
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10f18:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10f1c:	zext.b	a4,a4
   10f20:	bltu	a3,a4,10fd8 <run_kernel(RuntimeParams const&)+0xaf0>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10f24:	sw	zero,40(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10f28:	lw	a4,40(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10f2c:	li	a3,14
   10f30:	zext.b	a4,a4
   10f34:	bltu	a3,a4,10fcc <run_kernel(RuntimeParams const&)+0xae4>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10f38:	sw	zero,40(a5)
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
   10f3c:	lw	ra,44(sp)
   10f40:	lw	s0,40(sp)
   10f44:	lw	s1,36(sp)
   10f48:	lw	s2,32(sp)
   10f4c:	lw	s3,28(sp)
   10f50:	lw	s4,24(sp)
   10f54:	addi	sp,sp,48
   10f58:	ret
        detail::zone_hashes[n] = hash_val;
   10f5c:	lui	a3,0x7c867
   10f60:	sh2add	a4,a5,s1
   10f64:	addi	a3,a3,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
        detail::next_zone_id   = n + 1;
   10f68:	addi	a2,a5,1
        detail::zone_hashes[n] = hash_val;
   10f6c:	sw	a3,4(a4)
        detail::next_zone_id   = n + 1;
   10f70:	sw	a2,0(s1)
        return n;
   10f74:	j	105a4 <run_kernel(RuntimeParams const&)+0xbc>
        detail::zone_hashes[n] = hash_val;
   10f78:	lui	a4,0xbd77
   10f7c:	sh2add	a2,a5,s1
   10f80:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
        detail::next_zone_id   = n + 1;
   10f84:	addi	a3,a5,1
        detail::zone_hashes[n] = hash_val;
   10f88:	sw	a4,4(a2)
        detail::next_zone_id   = n + 1;
   10f8c:	sw	a3,0(s1)
        return n;
   10f90:	j	10a4c <run_kernel(RuntimeParams const&)+0x564>
    for (std::uint32_t i = 0; i < n; ++i)
   10f94:	mv	a5,a3
   10f98:	j	10a4c <run_kernel(RuntimeParams const&)+0x564>
   10f9c:	mv	a5,a3
   10fa0:	j	105a4 <run_kernel(RuntimeParams const&)+0xbc>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10fa4:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10fa8:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10fac:	li	a3,14
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10fb0:	sw	zero,32(a5)
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
   10fb4:	lw	a4,32(a5)
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10fb8:	zext.b	a4,a4
   10fbc:	bgeu	a3,a4,105e8 <run_kernel(RuntimeParams const&)+0x100>
   10fc0:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10fc4:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10fc8:	j	105e8 <run_kernel(RuntimeParams const&)+0x100>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10fcc:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10fd0:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10fd4:	j	10f38 <run_kernel(RuntimeParams const&)+0xa50>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10fd8:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10fdc:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10fe0:	j	10f24 <run_kernel(RuntimeParams const&)+0xa3c>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10fe4:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10fe8:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10fec:	j	10a90 <run_kernel(RuntimeParams const&)+0x5a8>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10ff0:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   10ff4:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   10ff8:	j	10a7c <run_kernel(RuntimeParams const&)+0x594>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   10ffc:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   11000:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   11004:	j	109ac <run_kernel(RuntimeParams const&)+0x4c4>
    LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
   11008:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear → SEMPOST: increment (cap at 15)
   1100c:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
   11010:	j	10998 <run_kernel(RuntimeParams const&)+0x4b0>
    LLK_ASSERT(
   11014:	ebreak
   11018:	j	10664 <run_kernel(RuntimeParams const&)+0x17c>
   1101c:	li	a5,4
   11020:	j	105a4 <run_kernel(RuntimeParams const&)+0xbc>
   11024:	li	a5,4
   11028:	j	10a4c <run_kernel(RuntimeParams const&)+0x564>

0001102c <_init()>:
    }
}

void _init(void)
{
}
   1102c:	ret

00011030 <_fini()>:

void _fini(void)
   11030:	ret

00011034 <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
   11034:	lui	a5,0x20
   11038:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
   1103c:	sb	a5,0(a0)
        (void)(dstc[i]);
   11040:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
   11044:	fence
}
   11048:	ret

0001104c <memset>:
   1104c:	li	t1,15
   11050:	mv	a4,a0
   11054:	bgeu	t1,a2,11090 <memset+0x44>
   11058:	andi	a5,a4,15
   1105c:	bnez	a5,110fc <memset+0xb0>
   11060:	bnez	a1,110e4 <memset+0x98>
   11064:	andi	a3,a2,-16
   11068:	andi	a2,a2,15
   1106c:	add	a3,a3,a4
   11070:	sw	a1,0(a4)
   11074:	sw	a1,4(a4)
   11078:	sw	a1,8(a4)
   1107c:	sw	a1,12(a4)
   11080:	addi	a4,a4,16
   11084:	bltu	a4,a3,11070 <memset+0x24>
   11088:	bnez	a2,11090 <memset+0x44>
   1108c:	ret
   11090:	sub	a3,t1,a2
   11094:	slli	a3,a3,0x2
   11098:	auipc	t0,0x0
   1109c:	add	a3,a3,t0
   110a0:	jr	12(a3)
   110a4:	sb	a1,14(a4)
   110a8:	sb	a1,13(a4)
   110ac:	sb	a1,12(a4)
   110b0:	sb	a1,11(a4)
   110b4:	sb	a1,10(a4)
   110b8:	sb	a1,9(a4)
   110bc:	sb	a1,8(a4)
   110c0:	sb	a1,7(a4)
   110c4:	sb	a1,6(a4)
   110c8:	sb	a1,5(a4)
   110cc:	sb	a1,4(a4)
   110d0:	sb	a1,3(a4)
   110d4:	sb	a1,2(a4)
   110d8:	sb	a1,1(a4)
   110dc:	sb	a1,0(a4)
   110e0:	ret
   110e4:	zext.b	a1,a1
   110e8:	slli	a3,a1,0x8
   110ec:	or	a1,a1,a3
   110f0:	slli	a3,a1,0x10
   110f4:	or	a1,a1,a3
   110f8:	j	11064 <memset+0x18>
   110fc:	slli	a3,a5,0x2
   11100:	auipc	t0,0x0
   11104:	add	a3,a3,t0
   11108:	mv	t0,ra
   1110c:	jalr	-96(a3)
   11110:	mv	ra,t0
   11114:	addi	a5,a5,-16
   11118:	sub	a4,a4,a5
   1111c:	add	a2,a2,a5
   11120:	bgeu	t1,a2,11090 <memset+0x44>
   11124:	j	11060 <memset+0x14>
