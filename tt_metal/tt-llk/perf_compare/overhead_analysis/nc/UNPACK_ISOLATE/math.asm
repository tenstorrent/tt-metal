
/tmp/perf_overhead_artifacts/nc/UNPACK_ISOLATE/math.elf:     file format elf32-littleriscv


Disassembly of section .init:

0000b000 <_start>:
// even though -fno-asynchronous-unwind-tables -fno-exceptions flags are set
void* __gxx_personality_v0;

__attribute__((no_profile_instrument_function)) TT_ALWAYS_INLINE void do_crt0()
{
    asm volatile(
    b000:	auipc	gp,0xffaf6
    b004:	addi	gp,gp,-2048 # ffb00800 <__global_pointer$>
        "la gp, __global_pointer$\n"
        ".option pop" ::
            : "memory");

    // Set stack pointer
    asm volatile("la sp, %0" : : "i"(__stack_top) : "memory");
    b008:	auipc	sp,0xffaf6
    b00c:	addi	sp,sp,-8 # ffb01000 <__stack_top>

    // Initialize .bss
    for (volatile std::uint32_t* p = (volatile std::uint32_t*)__ldm_bss_start; p < (volatile std::uint32_t*)__ldm_bss_end; p++)
    b010:	lui	a5,0xffb00
    b014:	lui	a4,0xffb00
    b018:	addi	a5,a5,32 # ffb00020 <llk_profiler::open_zone_cnt>
    b01c:	addi	a4,a4,64 # ffb00040 <__fw_export_ldm_end>
    b020:	bgeu	a5,a4,b044 <_start+0x44>
    b024:	addi	a4,a4,-1
    b028:	sub	a4,a4,a5
    b02c:	andi	a4,a4,-4
    b030:	addi	a4,a4,4
    b034:	add	a4,a4,a5
    {
        *p = 0;
    b038:	sw	zero,0(a5)
    for (volatile std::uint32_t* p = (volatile std::uint32_t*)__ldm_bss_start; p < (volatile std::uint32_t*)__ldm_bss_end; p++)
    b03c:	addi	a5,a5,4
    b040:	bne	a5,a4,b038 <_start+0x38>
    }

    // Copy .loader_init to .ldm_data
    if ((std::uint32_t)__loader_init_start != (std::uint32_t)__loader_init_end)
    b044:	lui	a5,0xf
    b048:	lui	a4,0x10
    b04c:	mv	a5,a5
    b050:	mv	a4,a4
    b054:	beq	a5,a4,b094 <_start+0x94>
    {
        volatile std::uint32_t* src = (volatile std::uint32_t*)__loader_init_start;
        volatile std::uint32_t* dst = (volatile std::uint32_t*)__ldm_data_start;
        volatile std::uint32_t* end = (volatile std::uint32_t*)__ldm_data_end;
        while (dst < end)
    b058:	lui	a4,0xffb00
    b05c:	lui	a3,0xffb00
    b060:	mv	a4,a4
    b064:	addi	a3,a3,32 # ffb00020 <llk_profiler::open_zone_cnt>
    b068:	bgeu	a4,a3,b094 <_start+0x94>
    b06c:	addi	a3,a3,-1
    b070:	sub	a3,a3,a4
    b074:	andi	a3,a3,-4
    b078:	addi	a3,a3,4
    b07c:	add	a3,a3,a5
        {
            *dst++ = *src++;
    b080:	lw	a1,0(a5) # f000 <__loader_init_start>
    b084:	addi	a5,a5,4
    b088:	sw	a1,0(a4) # ffb00000 <llk_profiler::buffer>
    b08c:	addi	a4,a4,4
        while (dst < end)
    b090:	bne	a5,a3,b080 <_start+0x80>
        }
    }

    // Execute global constructors
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
    b094:	lui	s0,0xffb00
    b098:	lui	s1,0xffb00
    b09c:	mv	s0,s0
    b0a0:	mv	s1,s1
    b0a4:	bgeu	s0,s1,b0b8 <_start+0xb8>
    {
        (*temp_constructor)();
    b0a8:	lw	a5,0(s0) # ffb00000 <llk_profiler::buffer>
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
    b0ac:	addi	s0,s0,4
        (*temp_constructor)();
    b0b0:	jalr	a5
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
    b0b4:	bltu	s0,s1,b0a8 <_start+0xa8>

extern "C" __attribute__((section(".init"), naked, noreturn, no_profile_instrument_function)) std::uint32_t _start()
{
    do_crt0();

    main();
    b0b8:	jal	b0c0 <main>

#ifdef COVERAGE
    gcov_dump();
#endif

    for (;;)
    b0bc:	j	b0bc <_start+0xbc>

Disassembly of section .text:

0000b0c0 <main>:
    volatile char *dstc       = reinterpret_cast<volatile char *>(dst);
    const volatile char *srcc = reinterpret_cast<const volatile char *>(src);

    for (std::size_t i = 0; i < len; i++)
    {
        dstc[i] = srcc[i];
    b0c0:	lui	a5,0x20
    b0c4:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
{
    b0c8:	addi	sp,sp,-48
    b0cc:	sb	a5,8(sp)
    b0d0:	sw	ra,44(sp)
    b0d4:	sw	s0,40(sp)
    b0d8:	sw	s1,36(sp)
    b0dc:	sw	s2,32(sp)
    b0e0:	sw	s3,28(sp)
    }

    for (std::size_t i = 0; i < len; i++)
    {
        (void)(dstc[i]);
    b0e4:	lbu	a5,8(sp)
    }

    asm volatile("fence" ::: "memory");
    b0e8:	fence
    std::fill(ckernel::regfile, ckernel::regfile + 64, 0);
    b0ec:	lw	a5,-2040(gp) # ffb00008 <ckernel::regfile>
      // otherwise we just use another reference.
      typedef typename __gnu_cxx::__conditional_type<__load_outside_loop,
						     const _Tp,
						     const _Tp&>::__type _Up;
      _Up __val(__value);
      for (; __first != __last; ++__first)
    b0f0:	addi	a4,a5,256
	*__first = __val;
    b0f4:	sw	zero,0(a5)
      for (; __first != __last; ++__first)
    b0f8:	addi	a5,a5,4
    b0fc:	bne	a4,a5,b0f4 <main+0x34>
    }
}

__attribute__((always_inline)) inline void reset()
{
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
    b100:	lui	a5,0x16b
    buffer        = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
    b104:	lui	s2,0xffb00
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
    b108:	addi	a4,a5,-12 # 16aff4 <__runtime_args_end+0x14abf4>
    buffer        = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
    b10c:	sw	a5,0(s2) # ffb00000 <llk_profiler::buffer>
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
    b110:	lui	s3,0xffb00
    TTI_NOP;
}

inline void reset_cfg_state_id()
{
    cfg_state_id = 0;
    b114:	sw	zero,-2000(gp) # ffb00030 <ckernel::cfg_state_id>
    write_idx     = 0;
    open_zone_cnt = 0;

    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
    b118:	lui	a2,0x1
    b11c:	li	a1,0
    b120:	lui	a0,0x16c
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
    b124:	sw	a4,4(s3) # ffb00004 <llk_profiler::barrier_ptr>
}

inline void reset_dest_offset_id()
{
    dest_offset_id = 0;
    b128:	sw	zero,-2004(gp) # ffb0002c <ckernel::dest_offset_id>
    write_idx     = 0;
    b12c:	sw	zero,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    open_zone_cnt = 0;
    b130:	sw	zero,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
    b134:	jal	b5b4 <memset>
    auto& barrier = *barrier_ptr;
    b138:	lw	a2,4(s3)
    barrier[TRISC_ID] = 1;
    b13c:	li	a4,1
    b140:	sw	a4,4(a2) # 1004 <TRISC_LOCAL_MEM_LENGTH+0x4>
    asm volatile("fence" ::: "memory");
    b144:	fence
    for (std::uint32_t i = 0; i < NUM_CORES; ++i)
    b148:	li	a5,0
    b14c:	li	a6,2
        if (i == TRISC_ID)
    b150:	sh3add	a3,a5,a2
        while (barrier[i] != 1)
    b154:	lw	a0,0(a3)
        if (i == TRISC_ID)
    b158:	slli	a1,a5,0x1
        while (barrier[i] != 1)
    b15c:	beq	a0,a4,b16c <main+0xac>
            asm volatile("fence" ::: "memory");
    b160:	fence
        while (barrier[i] != 1)
    b164:	lw	a5,0(a3)
    b168:	bne	a5,a4,b160 <main+0xa0>
    for (std::uint32_t i = 0; i < NUM_CORES; ++i)
    b16c:	li	a5,1
    b170:	bne	a1,a6,b150 <main+0x90>
    zone_scoped(zone_scoped&&)                 = delete;
    zone_scoped& operator=(const zone_scoped&) = delete;
    zone_scoped& operator=(zone_scoped&&)      = delete;

    inline __attribute__((always_inline)) zone_scoped()
    {
    b174:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b178:	lw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    b17c:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        asm volatile("" ::: "memory");
        if (!is_buffer_full())
    b180:	li	a1,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b184:	add	a5,a3,a2
    b188:	addi	a5,a5,-1021
        if (!is_buffer_full())
    b18c:	bgeu	a1,a5,b1d8 <main+0x118>
// now handled by the compiler)
// workaround is needed only for GS
inline std::uint32_t reg_read(std::uint32_t addr)
{
    volatile std::uint32_t tt_reg_ptr *p_reg = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(addr);
    return p_reg[0];
    b190:	lui	a5,0xffb12
    b194:	lw	a0,496(a5) # ffb121f0 <__stack_top+0x111f0>
        {
            is_opened = true;
    b198:	sb	a4,12(sp)
    b19c:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b1a0:	lui	a1,0x1
    b1a4:	addi	a6,a1,-1 # fff <__firmware_stack_size+0xdff>
    b1a8:	lw	a5,0(s2)
            write_entry(EntryType::ZONE_START, id16);
            ++open_zone_cnt;
    b1ac:	addi	a2,a2,1
    b1b0:	sw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b1b4:	and	a4,a4,a6
    b1b8:	lui	a6,0xa5104
    b1bc:	or	a4,a4,a6
    b1c0:	sh2add	a5,a3,a5
    b1c4:	add	a5,a5,a1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b1c8:	addi	a3,a3,2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b1cc:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b1d0:	sw	a0,4(a5)
    b1d4:	sw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
        run_kernel(temp_args);
    b1d8:	addi	a0,sp,8
    b1dc:	jal	b2e0 <run_kernel(RuntimeParams const&)>
    store_blocking(&pc_buf_base[1], 0);
    b1e0:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
    b1e4:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    b1e8:	addi	a4,a4,4
    asm volatile(
    b1ec:	sw	a5,0(a4)
    b1f0:	lw	a5,0(a4)
    b1f4:	and	zero,zero,a5
    }

    ~zone_scoped()
    {
        asm volatile("" ::: "memory");
        if (is_opened)
    b1f8:	lbu	a5,12(sp)
    b1fc:	beqz	a5,b24c <main+0x18c>
    return p_reg[0];
    b200:	lui	a5,0xffb12
    b204:	lw	a0,496(a5) # ffb121f0 <__stack_top+0x111f0>
    b208:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b20c:	lui	a1,0x1
    b210:	lw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
        {
            write_entry(EntryType::ZONE_END, id16);
            --open_zone_cnt;
    b214:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b218:	addi	a6,a1,-1 # fff <__firmware_stack_size+0xdff>
    b21c:	lw	a5,0(s2)
    b220:	and	a4,a4,a6
    b224:	lui	a6,0xb5104
    b228:	or	a4,a4,a6
    b22c:	sh2add	a5,a3,a5
    b230:	add	a5,a5,a1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b234:	addi	a3,a3,2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b238:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b23c:	sw	a0,4(a5)
    b240:	sw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b244:	addi	a2,a2,-1
    b248:	sw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    *mailbox = ckernel::KERNEL_COMPLETE;
    b24c:	lui	a5,0x20
}
    b250:	lw	ra,44(sp)
    b254:	lw	s0,40(sp)
    *mailbox = ckernel::KERNEL_COMPLETE;
    b258:	li	a4,255
    b25c:	sw	a4,-68(a5) # 1ffbc <__loader_init_end+0xffbc>
}
    b260:	lw	s1,36(sp)
    b264:	lw	s2,32(sp)
    b268:	lw	s3,28(sp)
    b26c:	li	a0,0
    b270:	addi	sp,sp,48
    b274:	ret

0000b278 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&) [clone .constprop.0]>:
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t face_c_dim = tensor_shape.face_c_dim;
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
}
    b278:	li	a0,1
    b27c:	ret

0000b280 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>:
        return num_faces_r_dim * num_faces_c_dim;
    b280:	lbu	a5,3(a0) # 16c003 <__runtime_args_end+0x14bc03>
    b284:	lbu	a4,2(a0)
{
    b288:	mv	a3,a0
        return num_faces_r_dim * num_faces_c_dim;
    b28c:	mul	a4,a4,a5
    b290:	zext.b	a4,a4
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
    b294:	addi	a5,a4,-1
    b298:	addi	a4,a4,-4
    b29c:	sltiu	a5,a5,2
    b2a0:	seqz	a4,a4
    b2a4:	or	a0,a5,a4
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
    b2a8:	beqz	a0,b2dc <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    b2ac:	lbu	a4,0(a3)
    b2b0:	li	a5,16
    b2b4:	li	a0,0
    b2b8:	bltu	a5,a4,b2dc <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
    b2bc:	lui	a5,0x10
    b2c0:	addi	a5,a5,278 # 10116 <__loader_init_end+0x116>
    b2c4:	srl	a5,a5,a4
    b2c8:	andi	a0,a5,1
    b2cc:	beqz	a0,b2dc <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
    b2d0:	lbu	a0,1(a3)
    b2d4:	addi	a0,a0,-16
    b2d8:	seqz	a0,a0
}
    b2dc:	ret

0000b2e0 <run_kernel(RuntimeParams const&)>:

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    b2e0:	addi	sp,sp,-16
    {
    b2e4:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b2e8:	lw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    b2ec:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    b2f0:	li	a0,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b2f4:	add	a4,a5,a3
    b2f8:	addi	a4,a4,-1021
        if (!is_buffer_full())
    b2fc:	bgeu	a0,a4,b350 <run_kernel(RuntimeParams const&)+0x70>
    b300:	lui	a4,0xffb12
    b304:	lw	a7,496(a4) # ffb121f0 <__stack_top+0x111f0>
    b308:	lw	a0,504(a4)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b30c:	lui	a6,0x1
    b310:	lui	a4,0xffb00
    b314:	addi	t1,a6,-1 # fff <__firmware_stack_size+0xdff>
    b318:	lw	a4,0(a4) # ffb00000 <llk_profiler::buffer>
    b31c:	lui	t3,0xaf7a9
    b320:	sh2add	a4,a5,a4
    b324:	add	a4,a4,a6
    b328:	and	a0,a0,t1
            is_opened = true;
    b32c:	li	t1,1
    b330:	sb	t1,12(sp)
            ++open_zone_cnt;
    b334:	add	a3,a3,t1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b338:	or	a0,a0,t3
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b33c:	addi	a5,a5,2
            ++open_zone_cnt;
    b340:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b344:	sw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b348:	sw	a0,0(a4)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b34c:	sw	a7,4(a4)
    store_blocking(&pc_buf_base[1], 0);
    b350:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
    b354:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    b358:	addi	a4,a4,4
    asm volatile(
    b35c:	sw	a5,0(a4)
    b360:	lw	a5,0(a4)
    b364:	and	zero,zero,a5
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    b368:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b36c:	lw	a5,36(a4)

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0)
    b370:	zext.b	a5,a5
    b374:	bnez	a5,b36c <run_kernel(RuntimeParams const&)+0x8c>
        set_dest_section_base<StartZero>();
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
    b378:	ttseminit	2,0,2
    dest_offset_id = 0;
    b37c:	sw	zero,-2004(gp) # ffb0002c <ckernel::dest_offset_id>
template <DstStart Dst>
inline void set_dest_section_base()
{
    if constexpr (Dst == DstStart::StartZero)
    {
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
    b380:	ttsetc16	1,0
    std::uint8_t mask_b0 = Mask & 0xff;

    if (mask_b0 != 0)
    {
        std::uint8_t data_b0 = wrdata & 0xff;
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    b384:	lui	a3,0xffe40
    b388:	lui	a6,0xb3080
    b38c:	mv	a3,a3
    b390:	addi	a6,a6,220 # b30800dc <__runtime_args_end+0xb305fcdc>
    b394:	sw	a6,0(a3) # ffe40000 <__instrn_buffer>
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    b398:	ttstallwait	128,16
    wrdata >>= 8;
    std::uint8_t mask_b3 = (Mask >> 24) & 0xff;
    if (mask_b3 != 0)
    {
        std::uint8_t data_b3 = (wrdata) & 0xff;
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
    b39c:	lui	a6,0xb6800
    b3a0:	addi	a6,a6,1 # b6800001 <__runtime_args_end+0xb67dfc01>
    b3a4:	lui	a7,0xb6202
    b3a8:	sw	a6,0(a3)
    b3ac:	addi	a7,a7,1 # b6202001 <__runtime_args_end+0xb61e1c01>
    b3b0:	lui	a6,0xb6404
    b3b4:	sw	a7,0(a3)
    b3b8:	addi	a6,a6,1 # b6404001 <__runtime_args_end+0xb63e3c01>
    b3bc:	sw	a6,0(a3)
    // Program source and dest registers
    __attribute__((always_inline)) inline void set(const std::uint8_t mod_index) const
    {
        // KCM - This gets around issue: error: impossible constraint in 'asm'
        // TTI_SETC16(addr_mod_src_reg_addr[mod_index], src_val());
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b3c0:	ttsetc16	12,2056
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b3c4:	ttsetc16	28,8
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b3c8:	ttsetc16	47,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b3cc:	ttsetc16	13,0
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b3d0:	ttsetc16	29,0
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b3d4:	ttsetc16	48,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b3d8:	ttsetc16	14,32896
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b3dc:	ttsetc16	30,9216
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b3e0:	ttsetc16	49,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b3e4:	ttsetc16	15,32896
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b3e8:	ttsetc16	31,36872
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b3ec:	ttsetc16	50,0
    store_blocking(&pc_buf_base[2], 0);
    b3f0:	addi	a4,a4,8
    asm volatile(
    b3f4:	mv	a3,a5
    b3f8:	sw	a3,0(a4)
    b3fc:	lw	a3,0(a4)
    b400:	and	zero,zero,a3
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
    b404:	lui	a4,0xffb80
    b408:	li	a3,2
    b40c:	sw	a3,0(a4) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
    b410:	sw	a3,4(a4)
    mop_cfg[2] = m_start_op0;
    b414:	lui	a3,0x2000
    b418:	sw	a3,8(a4)
    mop_cfg[3] = m_end_op0;
    b41c:	sw	a3,12(a4)
    mop_cfg[4] = m_end_op1;
    b420:	sw	a3,16(a4)
    mop_cfg[5] = m_loop_op0;
    b424:	lui	a6,0x27000
    b428:	sw	a6,20(a4)
    mop_cfg[6] = m_loop_op1;
    b42c:	sw	a3,24(a4)
    mop_cfg[7] = m_loop0_last_instr;
    b430:	lui	a3,0x27c0c
    b434:	sw	a3,28(a4)
    mop_cfg[8] = m_loop1_last_instr;
    b438:	lui	a3,0x27008
    b43c:	sw	a3,32(a4)
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    b440:	ttsetc16	7,0
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, setrwc);
    b444:	ttsetrwc	0,0,0,0,0,15
    store_blocking(&pc_buf_base[1], 0);
    b448:	lw	a4,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b44c:	addi	a4,a4,4
    asm volatile(
    b450:	sw	a5,0(a4)
    b454:	lw	a5,0(a4)
    b458:	and	zero,zero,a5
        if (is_opened)
    b45c:	lbu	a5,12(sp)
    b460:	beqz	a5,b4b4 <run_kernel(RuntimeParams const&)+0x1d4>
    return p_reg[0];
    b464:	lui	a5,0xffb12
    b468:	lw	a7,496(a5) # ffb121f0 <__stack_top+0x111f0>
    b46c:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b470:	lui	a6,0x1
    b474:	lui	a5,0xffb00
    b478:	lw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b47c:	lw	a0,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b480:	addi	t1,a6,-1 # fff <__firmware_stack_size+0xdff>
    b484:	lw	a5,0(a5) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    b488:	addi	a0,a0,-1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b48c:	and	a4,a4,t1
    b490:	lui	t1,0xbf7a9
    b494:	or	a4,a4,t1
    b498:	sh2add	a5,a3,a5
    b49c:	add	a5,a5,a6
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b4a0:	addi	a3,a3,2 # 27008002 <__runtime_args_end+0x26fe7c02>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b4a4:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b4a8:	sw	a7,4(a5)
    b4ac:	sw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b4b0:	sw	a0,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    {
    b4b4:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b4b8:	lw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    b4bc:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    b4c0:	li	a0,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b4c4:	add	a4,a5,a3
    b4c8:	addi	a4,a4,-1021
        if (!is_buffer_full())
    b4cc:	bgeu	a0,a4,b520 <run_kernel(RuntimeParams const&)+0x240>
    b4d0:	lui	a4,0xffb12
    b4d4:	lw	a7,496(a4) # ffb121f0 <__stack_top+0x111f0>
    b4d8:	lw	a0,504(a4)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b4dc:	lui	a6,0x1
    b4e0:	lui	a4,0xffb00
    b4e4:	addi	t1,a6,-1 # fff <__firmware_stack_size+0xdff>
    b4e8:	lw	a4,0(a4) # ffb00000 <llk_profiler::buffer>
    b4ec:	lui	t3,0xaa945
    b4f0:	sh2add	a4,a5,a4
    b4f4:	add	a4,a4,a6
    b4f8:	and	a0,a0,t1
            is_opened = true;
    b4fc:	li	t1,1
    b500:	sb	t1,12(sp)
            ++open_zone_cnt;
    b504:	add	a3,a3,t1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b508:	or	a0,a0,t3
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b50c:	addi	a5,a5,2
            ++open_zone_cnt;
    b510:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b514:	sw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b518:	sw	a0,0(a4)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b51c:	sw	a7,4(a4)
        asm volatile("" ::: "memory");
    b520:	li	a5,64
        constexpr std::uint32_t cond_valid_b = clear_b ? ckernel::p_stall::SRCB_VLD : 0;
#ifdef ARCH_QUASAR
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a, cond_valid_b, 0);
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0, 0, 0, 0, 0);
#else
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a | cond_valid_b);
    b524:	ttstallwait	64,384
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0);
    b528:	ttcleardvalid	3,0
    while (iterations-- > 0)
    b52c:	addi	a5,a5,-1
    b530:	bnez	a5,b524 <run_kernel(RuntimeParams const&)+0x244>
        if (is_opened)
    b534:	lbu	a5,12(sp)
    b538:	beqz	a5,b58c <run_kernel(RuntimeParams const&)+0x2ac>
    b53c:	lui	a5,0xffb12
    b540:	lw	a7,496(a5) # ffb121f0 <__stack_top+0x111f0>
    b544:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b548:	lui	a6,0x1
    b54c:	lui	a5,0xffb00
    b550:	lw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b554:	lw	a0,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b558:	addi	t1,a6,-1 # fff <__firmware_stack_size+0xdff>
    b55c:	lw	a5,0(a5) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    b560:	addi	a0,a0,-1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b564:	and	a4,a4,t1
    b568:	lui	t1,0xba945
    b56c:	or	a4,a4,t1
    b570:	sh2add	a5,a3,a5
    b574:	add	a5,a5,a6
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b578:	addi	a3,a3,2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b57c:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b580:	sw	a7,4(a5)
    b584:	sw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b588:	sw	a0,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
    b58c:	addi	sp,sp,16
    b590:	ret

0000b594 <_init()>:
    }
}

void _init(void)
{
}
    b594:	ret

0000b598 <_fini()>:

void _fini(void)
    b598:	ret

0000b59c <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
    b59c:	lui	a5,0x20
    b5a0:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
    b5a4:	sb	a5,0(a0)
        (void)(dstc[i]);
    b5a8:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
    b5ac:	fence
}
    b5b0:	ret

0000b5b4 <memset>:
    b5b4:	li	t1,15
    b5b8:	mv	a4,a0
    b5bc:	bgeu	t1,a2,b5f8 <memset+0x44>
    b5c0:	andi	a5,a4,15
    b5c4:	bnez	a5,b664 <memset+0xb0>
    b5c8:	bnez	a1,b64c <memset+0x98>
    b5cc:	andi	a3,a2,-16
    b5d0:	andi	a2,a2,15
    b5d4:	add	a3,a3,a4
    b5d8:	sw	a1,0(a4)
    b5dc:	sw	a1,4(a4)
    b5e0:	sw	a1,8(a4)
    b5e4:	sw	a1,12(a4)
    b5e8:	addi	a4,a4,16
    b5ec:	bltu	a4,a3,b5d8 <memset+0x24>
    b5f0:	bnez	a2,b5f8 <memset+0x44>
    b5f4:	ret
    b5f8:	sub	a3,t1,a2
    b5fc:	slli	a3,a3,0x2
    b600:	auipc	t0,0x0
    b604:	add	a3,a3,t0
    b608:	jr	12(a3)
    b60c:	sb	a1,14(a4)
    b610:	sb	a1,13(a4)
    b614:	sb	a1,12(a4)
    b618:	sb	a1,11(a4)
    b61c:	sb	a1,10(a4)
    b620:	sb	a1,9(a4)
    b624:	sb	a1,8(a4)
    b628:	sb	a1,7(a4)
    b62c:	sb	a1,6(a4)
    b630:	sb	a1,5(a4)
    b634:	sb	a1,4(a4)
    b638:	sb	a1,3(a4)
    b63c:	sb	a1,2(a4)
    b640:	sb	a1,1(a4)
    b644:	sb	a1,0(a4)
    b648:	ret
    b64c:	zext.b	a1,a1
    b650:	slli	a3,a1,0x8
    b654:	or	a1,a1,a3
    b658:	slli	a3,a1,0x10
    b65c:	or	a1,a1,a3
    b660:	j	b5cc <memset+0x18>
    b664:	slli	a3,a5,0x2
    b668:	auipc	t0,0x0
    b66c:	add	a3,a3,t0
    b670:	mv	t0,ra
    b674:	jalr	-96(a3)
    b678:	mv	ra,t0
    b67c:	addi	a5,a5,-16
    b680:	sub	a4,a4,a5
    b684:	add	a2,a2,a5
    b688:	bgeu	t1,a2,b5f8 <memset+0x44>
    b68c:	j	b5c8 <memset+0x14>
