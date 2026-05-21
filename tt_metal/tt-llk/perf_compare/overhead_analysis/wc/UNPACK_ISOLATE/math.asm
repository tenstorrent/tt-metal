
/tmp/perf_overhead_artifacts/wc/UNPACK_ISOLATE/math.elf:     file format elf32-littleriscv


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
    b01c:	addi	a4,a4,100 # ffb00064 <__gcov_info_end>
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
    b134:	jal	b830 <memset>
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
#endif
} // namespace detail

__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    std::uint32_t n = detail::next_zone_id;
    b2e0:	addi	a2,gp,-1984 # ffb00040 <llk_perf::detail::next_zone_id>
    b2e4:	lw	a5,0(a2)

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    b2e8:	addi	sp,sp,-16
    for (std::uint32_t i = 0; i < n; ++i)
    b2ec:	beqz	a5,b7b8 <run_kernel(RuntimeParams const&)+0x4d8>
    {
        if (detail::zone_hashes[i] == hash_val)
    b2f0:	lui	a4,0x7c867
    b2f4:	lw	a3,4(a2)
    b2f8:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    b2fc:	beq	a3,a4,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b300:	li	a3,1
    b304:	beq	a5,a3,b7b8 <run_kernel(RuntimeParams const&)+0x4d8>
        if (detail::zone_hashes[i] == hash_val)
    b308:	lw	a3,8(a2)
    b30c:	beq	a3,a4,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b310:	li	a3,2
    b314:	beq	a5,a3,b7b8 <run_kernel(RuntimeParams const&)+0x4d8>
        if (detail::zone_hashes[i] == hash_val)
    b318:	lw	a3,12(a2)
    b31c:	beq	a3,a4,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b320:	li	a3,3
    b324:	beq	a5,a3,b7b8 <run_kernel(RuntimeParams const&)+0x4d8>
        if (detail::zone_hashes[i] == hash_val)
    b328:	lw	a3,16(a2)
    b32c:	beq	a3,a4,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b330:	li	a3,4
    b334:	beq	a5,a3,b7b8 <run_kernel(RuntimeParams const&)+0x4d8>
        if (detail::zone_hashes[i] == hash_val)
    b338:	lw	a3,20(a2)
    b33c:	beq	a3,a4,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b340:	li	a4,5
    b344:	beq	a5,a4,b7b8 <run_kernel(RuntimeParams const&)+0x4d8>
        if (detail::zone_hashes[i] == hash_val)
    b348:	lui	a4,0x7c867
    b34c:	lw	a3,24(a2)
    b350:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    b354:	beq	a3,a4,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b358:	li	a3,6
    b35c:	beq	a5,a3,b7b8 <run_kernel(RuntimeParams const&)+0x4d8>
        if (detail::zone_hashes[i] == hash_val)
    b360:	lw	a3,28(a2)
    b364:	beq	a3,a4,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b368:	li	a3,7
    b36c:	beq	a5,a3,b7b8 <run_kernel(RuntimeParams const&)+0x4d8>
        if (detail::zone_hashes[i] == hash_val)
    b370:	lw	a3,32(a2)
    b374:	beq	a3,a4,b380 <run_kernel(RuntimeParams const&)+0xa0>
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
    b378:	li	a4,8
    b37c:	bne	a5,a4,b7b8 <run_kernel(RuntimeParams const&)+0x4d8>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    b380:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b384:	lw	a5,32(a3)
                ckernel::semaphore_post(PERF_ENTRY_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    b388:	zext.b	a5,a5
    b38c:	bnez	a5,b3a4 <run_kernel(RuntimeParams const&)+0xc4>
            {
                asm volatile("nop");
    b390:	nop
    b394:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b398:	lw	a5,32(a3)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    b39c:	zext.b	a5,a5
    b3a0:	beqz	a5,b390 <run_kernel(RuntimeParams const&)+0xb0>
    b3a4:	lw	a5,32(a3)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    b3a8:	zext.b	a5,a5
    b3ac:	beqz	a5,b804 <run_kernel(RuntimeParams const&)+0x524>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    b3b0:	li	a7,1
    b3b4:	sw	a7,32(a3)
    {
    b3b8:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b3bc:	lw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    b3c0:	lw	a1,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    b3c4:	li	t1,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b3c8:	add	a5,a3,a1
    b3cc:	addi	a5,a5,-1021
        if (!is_buffer_full())
    b3d0:	bgeu	t1,a5,b420 <run_kernel(RuntimeParams const&)+0x140>
    return p_reg[0];
    b3d4:	lui	a5,0xffb12
    b3d8:	lw	t3,496(a5) # ffb121f0 <__stack_top+0x111f0>
            is_opened = true;
    b3dc:	sb	a7,12(sp)
    b3e0:	lw	a7,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b3e4:	lui	t1,0x1
    b3e8:	lui	a5,0xffb00
    b3ec:	addi	t4,t1,-1 # fff <__firmware_stack_size+0xdff>
    b3f0:	lw	a5,0(a5) # ffb00000 <llk_profiler::buffer>
    b3f4:	sh2add	a5,a3,a5
            ++open_zone_cnt;
    b3f8:	addi	a1,a1,1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b3fc:	add	a5,a5,t1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b400:	addi	a3,a3,2
            ++open_zone_cnt;
    b404:	sw	a1,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b408:	and	a7,a7,t4
    b40c:	lui	t4,0xaf7a9
    b410:	or	a7,a7,t4
    b414:	sw	a7,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b418:	sw	t3,4(a5)
    b41c:	sw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    store_blocking(&pc_buf_base[1], 0);
    b420:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
    b424:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    b428:	addi	a3,a3,4
    asm volatile(
    b42c:	sw	a5,0(a3)
    b430:	lw	a5,0(a3)
    b434:	and	zero,zero,a5
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    b438:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b43c:	lw	a5,36(a3)

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0)
    b440:	zext.b	a5,a5
    b444:	bnez	a5,b43c <run_kernel(RuntimeParams const&)+0x15c>
        set_dest_section_base<StartZero>();
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
    b448:	ttseminit	2,0,2
    dest_offset_id = 0;
    b44c:	sw	zero,-2004(gp) # ffb0002c <ckernel::dest_offset_id>
template <DstStart Dst>
inline void set_dest_section_base()
{
    if constexpr (Dst == DstStart::StartZero)
    {
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
    b450:	ttsetc16	1,0
    std::uint8_t mask_b0 = Mask & 0xff;

    if (mask_b0 != 0)
    {
        std::uint8_t data_b0 = wrdata & 0xff;
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    b454:	lui	a1,0xffe40
    b458:	lui	a7,0xb3080
    b45c:	mv	a1,a1
    b460:	addi	a7,a7,220 # b30800dc <__runtime_args_end+0xb305fcdc>
    b464:	sw	a7,0(a1) # ffe40000 <__instrn_buffer>
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    b468:	ttstallwait	128,16
    wrdata >>= 8;
    std::uint8_t mask_b3 = (Mask >> 24) & 0xff;
    if (mask_b3 != 0)
    {
        std::uint8_t data_b3 = (wrdata) & 0xff;
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
    b46c:	lui	a7,0xb6800
    b470:	addi	a7,a7,1 # b6800001 <__runtime_args_end+0xb67dfc01>
    b474:	lui	t1,0xb6202
    b478:	sw	a7,0(a1)
    b47c:	addi	t1,t1,1 # b6202001 <__runtime_args_end+0xb61e1c01>
    b480:	lui	a7,0xb6404
    b484:	sw	t1,0(a1)
    b488:	addi	a7,a7,1 # b6404001 <__runtime_args_end+0xb63e3c01>
    b48c:	sw	a7,0(a1)
    // Program source and dest registers
    __attribute__((always_inline)) inline void set(const std::uint8_t mod_index) const
    {
        // KCM - This gets around issue: error: impossible constraint in 'asm'
        // TTI_SETC16(addr_mod_src_reg_addr[mod_index], src_val());
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b490:	ttsetc16	12,2056
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b494:	ttsetc16	28,8
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b498:	ttsetc16	47,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b49c:	ttsetc16	13,0
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b4a0:	ttsetc16	29,0
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b4a4:	ttsetc16	48,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b4a8:	ttsetc16	14,32896
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b4ac:	ttsetc16	30,9216
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b4b0:	ttsetc16	49,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b4b4:	ttsetc16	15,32896
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b4b8:	ttsetc16	31,36872
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b4bc:	ttsetc16	50,0
    store_blocking(&pc_buf_base[2], 0);
    b4c0:	addi	a3,a3,8
    asm volatile(
    b4c4:	mv	a1,a5
    b4c8:	sw	a1,0(a3)
    b4cc:	lw	a1,0(a3)
    b4d0:	and	zero,zero,a1
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
    b4d4:	lui	a3,0xffb80
    b4d8:	li	a1,2
    b4dc:	sw	a1,0(a3) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
    b4e0:	sw	a1,4(a3)
    mop_cfg[2] = m_start_op0;
    b4e4:	lui	a1,0x2000
    b4e8:	sw	a1,8(a3)
    mop_cfg[3] = m_end_op0;
    b4ec:	sw	a1,12(a3)
    mop_cfg[4] = m_end_op1;
    b4f0:	sw	a1,16(a3)
    mop_cfg[5] = m_loop_op0;
    b4f4:	lui	a7,0x27000
    b4f8:	sw	a7,20(a3)
    mop_cfg[6] = m_loop_op1;
    b4fc:	sw	a1,24(a3)
    mop_cfg[7] = m_loop0_last_instr;
    b500:	lui	a1,0x27c0c
    b504:	sw	a1,28(a3)
    mop_cfg[8] = m_loop1_last_instr;
    b508:	lui	a1,0x27008
    b50c:	sw	a1,32(a3)
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    b510:	ttsetc16	7,0
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, setrwc);
    b514:	ttsetrwc	0,0,0,0,0,15
    store_blocking(&pc_buf_base[1], 0);
    b518:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b51c:	addi	a3,a3,4
    asm volatile(
    b520:	sw	a5,0(a3)
    b524:	lw	a5,0(a3)
    b528:	and	zero,zero,a5
        if (is_opened)
    b52c:	lbu	a5,12(sp)
    b530:	beqz	a5,b584 <run_kernel(RuntimeParams const&)+0x2a4>
    return p_reg[0];
    b534:	lui	a5,0xffb12
    b538:	lw	t3,496(a5) # ffb121f0 <__stack_top+0x111f0>
    b53c:	lw	a3,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b540:	lui	t1,0x1
    b544:	lui	a5,0xffb00
    b548:	lw	a1,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b54c:	lw	a7,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b550:	addi	t4,t1,-1 # fff <__firmware_stack_size+0xdff>
    b554:	lw	a5,0(a5) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    b558:	addi	a7,a7,-1 # 26ffffff <__runtime_args_end+0x26fdfbff>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b55c:	and	a3,a3,t4
    b560:	lui	t4,0xbf7a9
    b564:	or	a3,a3,t4
    b568:	sh2add	a5,a1,a5
    b56c:	add	a5,a5,t1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b570:	addi	a1,a1,2 # 27008002 <__runtime_args_end+0x26fe7c02>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b574:	sw	a3,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b578:	sw	t3,4(a5)
    b57c:	sw	a1,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b580:	sw	a7,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    b584:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b588:	lw	a5,40(a3)
                ckernel::semaphore_post(PERF_EXIT_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    b58c:	zext.b	a5,a5
    b590:	bnez	a5,b5a8 <run_kernel(RuntimeParams const&)+0x2c8>
            {
                asm volatile("nop");
    b594:	nop
    b598:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b59c:	lw	a5,40(a3)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    b5a0:	zext.b	a5,a5
    b5a4:	beqz	a5,b594 <run_kernel(RuntimeParams const&)+0x2b4>
    b5a8:	lw	a5,40(a3)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    b5ac:	zext.b	a5,a5
    b5b0:	beqz	a5,b7d4 <run_kernel(RuntimeParams const&)+0x4f4>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    b5b4:	li	a1,1
    b5b8:	sw	a1,40(a3)
    std::uint32_t n = detail::next_zone_id;
    b5bc:	lw	a5,0(a2)
    for (std::uint32_t i = 0; i < n; ++i)
    b5c0:	beqz	a5,b79c <run_kernel(RuntimeParams const&)+0x4bc>
        if (detail::zone_hashes[i] == hash_val)
    b5c4:	lui	a3,0xbd77
    b5c8:	lw	a7,4(a2)
    b5cc:	addi	a3,a3,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    b5d0:	beq	a7,a3,b650 <run_kernel(RuntimeParams const&)+0x370>
    for (std::uint32_t i = 0; i < n; ++i)
    b5d4:	beq	a5,a1,b79c <run_kernel(RuntimeParams const&)+0x4bc>
        if (detail::zone_hashes[i] == hash_val)
    b5d8:	lw	a1,8(a2)
    b5dc:	beq	a1,a3,b650 <run_kernel(RuntimeParams const&)+0x370>
    for (std::uint32_t i = 0; i < n; ++i)
    b5e0:	li	a1,2
    b5e4:	beq	a5,a1,b79c <run_kernel(RuntimeParams const&)+0x4bc>
        if (detail::zone_hashes[i] == hash_val)
    b5e8:	lw	a1,12(a2)
    b5ec:	beq	a1,a3,b650 <run_kernel(RuntimeParams const&)+0x370>
    for (std::uint32_t i = 0; i < n; ++i)
    b5f0:	li	a1,3
    b5f4:	beq	a5,a1,b79c <run_kernel(RuntimeParams const&)+0x4bc>
        if (detail::zone_hashes[i] == hash_val)
    b5f8:	lw	a1,16(a2)
    b5fc:	beq	a1,a3,b650 <run_kernel(RuntimeParams const&)+0x370>
    for (std::uint32_t i = 0; i < n; ++i)
    b600:	li	a1,4
    b604:	beq	a5,a1,b79c <run_kernel(RuntimeParams const&)+0x4bc>
        if (detail::zone_hashes[i] == hash_val)
    b608:	lw	a1,20(a2)
    b60c:	beq	a1,a3,b650 <run_kernel(RuntimeParams const&)+0x370>
    for (std::uint32_t i = 0; i < n; ++i)
    b610:	li	a3,5
    b614:	beq	a5,a3,b79c <run_kernel(RuntimeParams const&)+0x4bc>
        if (detail::zone_hashes[i] == hash_val)
    b618:	lui	a3,0xbd77
    b61c:	lw	a1,24(a2)
    b620:	addi	a3,a3,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    b624:	beq	a1,a3,b650 <run_kernel(RuntimeParams const&)+0x370>
    for (std::uint32_t i = 0; i < n; ++i)
    b628:	li	a1,6
    b62c:	beq	a5,a1,b79c <run_kernel(RuntimeParams const&)+0x4bc>
        if (detail::zone_hashes[i] == hash_val)
    b630:	lw	a1,28(a2)
    b634:	beq	a1,a3,b650 <run_kernel(RuntimeParams const&)+0x370>
    for (std::uint32_t i = 0; i < n; ++i)
    b638:	li	a1,7
    b63c:	beq	a5,a1,b79c <run_kernel(RuntimeParams const&)+0x4bc>
        if (detail::zone_hashes[i] == hash_val)
    b640:	lw	a1,32(a2)
    b644:	beq	a1,a3,b650 <run_kernel(RuntimeParams const&)+0x370>
    if (n < PERF_COUNTERS_MAX_ZONES)
    b648:	li	a3,8
    b64c:	bne	a5,a3,b79c <run_kernel(RuntimeParams const&)+0x4bc>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    b650:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b654:	lw	a5,32(a3)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    b658:	zext.b	a5,a5
    b65c:	bnez	a5,b674 <run_kernel(RuntimeParams const&)+0x394>
                asm volatile("nop");
    b660:	nop
    b664:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b668:	lw	a5,32(a3)
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
    b66c:	zext.b	a5,a5
    b670:	beqz	a5,b660 <run_kernel(RuntimeParams const&)+0x380>
    b674:	lw	a5,32(a3)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    b678:	zext.b	a5,a5
    b67c:	beqz	a5,b7e0 <run_kernel(RuntimeParams const&)+0x500>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    b680:	li	a1,1
    b684:	sw	a1,32(a3)
    {
    b688:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b68c:	lw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    b690:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    b694:	li	a7,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b698:	add	a5,a3,a2
    b69c:	addi	a5,a5,-1021
        if (!is_buffer_full())
    b6a0:	bgeu	a7,a5,b6f0 <run_kernel(RuntimeParams const&)+0x410>
    return p_reg[0];
    b6a4:	lui	a5,0xffb12
    b6a8:	lw	t1,496(a5) # ffb121f0 <__stack_top+0x111f0>
            is_opened = true;
    b6ac:	sb	a1,12(sp)
    b6b0:	lw	a1,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b6b4:	lui	a7,0x1
    b6b8:	lui	a5,0xffb00
    b6bc:	addi	t3,a7,-1 # fff <__firmware_stack_size+0xdff>
    b6c0:	lw	a5,0(a5) # ffb00000 <llk_profiler::buffer>
    b6c4:	sh2add	a5,a3,a5
            ++open_zone_cnt;
    b6c8:	addi	a2,a2,1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b6cc:	add	a5,a5,a7
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b6d0:	addi	a3,a3,2
            ++open_zone_cnt;
    b6d4:	sw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b6d8:	and	a1,a1,t3
    b6dc:	lui	t3,0xaa945
    b6e0:	or	a1,a1,t3
    b6e4:	sw	a1,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b6e8:	sw	t1,4(a5)
    b6ec:	sw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
        asm volatile("" ::: "memory");
    b6f0:	li	a5,64
        constexpr std::uint32_t cond_valid_b = clear_b ? ckernel::p_stall::SRCB_VLD : 0;
#ifdef ARCH_QUASAR
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a, cond_valid_b, 0);
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0, 0, 0, 0, 0);
#else
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a | cond_valid_b);
    b6f4:	ttstallwait	64,384
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0);
    b6f8:	ttcleardvalid	3,0
    while (iterations-- > 0)
    b6fc:	addi	a5,a5,-1
    b700:	bnez	a5,b6f4 <run_kernel(RuntimeParams const&)+0x414>
        if (is_opened)
    b704:	lbu	a5,12(sp)
    b708:	beqz	a5,b75c <run_kernel(RuntimeParams const&)+0x47c>
    b70c:	lui	a5,0xffb12
    b710:	lw	t1,496(a5) # ffb121f0 <__stack_top+0x111f0>
    b714:	lw	a3,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b718:	lui	a7,0x1
    b71c:	lui	a5,0xffb00
    b720:	lw	a2,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b724:	lw	a1,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b728:	addi	t3,a7,-1 # fff <__firmware_stack_size+0xdff>
    b72c:	lw	a5,0(a5) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    b730:	addi	a1,a1,-1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b734:	and	a3,a3,t3
    b738:	lui	t3,0xba945
    b73c:	or	a3,a3,t3
    b740:	sh2add	a5,a2,a5
    b744:	add	a5,a5,a7
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b748:	addi	a2,a2,2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b74c:	sw	a3,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b750:	sw	t1,4(a5)
    b754:	sw	a2,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b758:	sw	a1,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    b75c:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b760:	lw	a5,40(a3)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    b764:	zext.b	a5,a5
    b768:	bnez	a5,b780 <run_kernel(RuntimeParams const&)+0x4a0>
                asm volatile("nop");
    b76c:	nop
    b770:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b774:	lw	a5,40(a3)
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
    b778:	zext.b	a5,a5
    b77c:	beqz	a5,b76c <run_kernel(RuntimeParams const&)+0x48c>
    b780:	lw	a5,40(a3)
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    b784:	zext.b	a5,a5
    b788:	beqz	a5,b7ec <run_kernel(RuntimeParams const&)+0x50c>
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    b78c:	li	a5,1
    b790:	sw	a5,40(a3)
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
    b794:	addi	sp,sp,16
    b798:	ret
        detail::zone_hashes[n] = hash_val;
    b79c:	lui	a3,0xbd77
    b7a0:	sh2add	a1,a5,a2
    b7a4:	addi	a3,a3,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
        detail::next_zone_id   = n + 1;
    b7a8:	addi	a5,a5,1
        detail::zone_hashes[n] = hash_val;
    b7ac:	sw	a3,4(a1)
        detail::next_zone_id   = n + 1;
    b7b0:	sw	a5,0(a2)
        return n;
    b7b4:	j	b650 <run_kernel(RuntimeParams const&)+0x370>
        detail::zone_hashes[n] = hash_val;
    b7b8:	lui	a4,0x7c867
    b7bc:	sh2add	a3,a5,a2
    b7c0:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
        detail::next_zone_id   = n + 1;
    b7c4:	addi	a5,a5,1
        detail::zone_hashes[n] = hash_val;
    b7c8:	sw	a4,4(a3)
        detail::next_zone_id   = n + 1;
    b7cc:	sw	a5,0(a2)
        return n;
    b7d0:	j	b380 <run_kernel(RuntimeParams const&)+0xa0>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    b7d4:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    b7d8:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b7dc:	j	b5b4 <run_kernel(RuntimeParams const&)+0x2d4>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    b7e0:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    b7e4:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b7e8:	j	b680 <run_kernel(RuntimeParams const&)+0x3a0>
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    b7ec:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    b7f0:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b7f4:	li	a5,1
    b7f8:	sw	a5,40(a3)
    b7fc:	addi	sp,sp,16
    b800:	ret
    LLK_ASSERT(semaphore_read(index) > 0, "Semaphore must not be already at 0.");
    b804:	ebreak
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set → SEMGET: decrement (only if > 0)
    b808:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b80c:	j	b3b0 <run_kernel(RuntimeParams const&)+0xd0>

0000b810 <_init()>:
    }
}

void _init(void)
{
}
    b810:	ret

0000b814 <_fini()>:

void _fini(void)
    b814:	ret

0000b818 <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
    b818:	lui	a5,0x20
    b81c:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
    b820:	sb	a5,0(a0)
        (void)(dstc[i]);
    b824:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
    b828:	fence
}
    b82c:	ret

0000b830 <memset>:
    b830:	li	t1,15
    b834:	mv	a4,a0
    b838:	bgeu	t1,a2,b874 <memset+0x44>
    b83c:	andi	a5,a4,15
    b840:	bnez	a5,b8e0 <memset+0xb0>
    b844:	bnez	a1,b8c8 <memset+0x98>
    b848:	andi	a3,a2,-16
    b84c:	andi	a2,a2,15
    b850:	add	a3,a3,a4
    b854:	sw	a1,0(a4)
    b858:	sw	a1,4(a4)
    b85c:	sw	a1,8(a4)
    b860:	sw	a1,12(a4)
    b864:	addi	a4,a4,16
    b868:	bltu	a4,a3,b854 <memset+0x24>
    b86c:	bnez	a2,b874 <memset+0x44>
    b870:	ret
    b874:	sub	a3,t1,a2
    b878:	slli	a3,a3,0x2
    b87c:	auipc	t0,0x0
    b880:	add	a3,a3,t0
    b884:	jr	12(a3)
    b888:	sb	a1,14(a4)
    b88c:	sb	a1,13(a4)
    b890:	sb	a1,12(a4)
    b894:	sb	a1,11(a4)
    b898:	sb	a1,10(a4)
    b89c:	sb	a1,9(a4)
    b8a0:	sb	a1,8(a4)
    b8a4:	sb	a1,7(a4)
    b8a8:	sb	a1,6(a4)
    b8ac:	sb	a1,5(a4)
    b8b0:	sb	a1,4(a4)
    b8b4:	sb	a1,3(a4)
    b8b8:	sb	a1,2(a4)
    b8bc:	sb	a1,1(a4)
    b8c0:	sb	a1,0(a4)
    b8c4:	ret
    b8c8:	zext.b	a1,a1
    b8cc:	slli	a3,a1,0x8
    b8d0:	or	a1,a1,a3
    b8d4:	slli	a3,a1,0x10
    b8d8:	or	a1,a1,a3
    b8dc:	j	b848 <memset+0x18>
    b8e0:	slli	a3,a5,0x2
    b8e4:	auipc	t0,0x0
    b8e8:	add	a3,a3,t0
    b8ec:	mv	t0,ra
    b8f0:	jalr	-96(a3)
    b8f4:	mv	ra,t0
    b8f8:	addi	a5,a5,-16
    b8fc:	sub	a4,a4,a5
    b900:	add	a2,a2,a5
    b904:	bgeu	t1,a2,b874 <memset+0x44>
    b908:	j	b844 <memset+0x14>
