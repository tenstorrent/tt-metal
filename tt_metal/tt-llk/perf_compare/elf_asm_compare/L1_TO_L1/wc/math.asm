
/tmp/elf_compare/wc/L1_TO_L1/math.elf:     file format elf32-littleriscv


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
    b134:	jal	b7a8 <memset>
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
    b2e0:	addi	a4,gp,-1984 # ffb00040 <llk_perf::detail::next_zone_id>
    b2e4:	lw	a5,0(a4)

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    b2e8:	addi	sp,sp,-16
    for (std::uint32_t i = 0; i < n; ++i)
    b2ec:	beqz	a5,b76c <run_kernel(RuntimeParams const&)+0x48c>
    {
        if (detail::zone_hashes[i] == hash_val)
    b2f0:	lui	a3,0x7c867
    b2f4:	lw	a2,4(a4)
    b2f8:	addi	a3,a3,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    b2fc:	beq	a2,a3,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b300:	li	a2,1
    b304:	beq	a5,a2,b76c <run_kernel(RuntimeParams const&)+0x48c>
        if (detail::zone_hashes[i] == hash_val)
    b308:	lw	a2,8(a4)
    b30c:	beq	a2,a3,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b310:	li	a2,2
    b314:	beq	a5,a2,b76c <run_kernel(RuntimeParams const&)+0x48c>
        if (detail::zone_hashes[i] == hash_val)
    b318:	lw	a2,12(a4)
    b31c:	beq	a2,a3,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b320:	li	a2,3
    b324:	beq	a5,a2,b76c <run_kernel(RuntimeParams const&)+0x48c>
        if (detail::zone_hashes[i] == hash_val)
    b328:	lw	a2,16(a4)
    b32c:	beq	a2,a3,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b330:	li	a2,4
    b334:	beq	a5,a2,b76c <run_kernel(RuntimeParams const&)+0x48c>
        if (detail::zone_hashes[i] == hash_val)
    b338:	lw	a2,20(a4)
    b33c:	beq	a2,a3,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b340:	li	a3,5
    b344:	beq	a5,a3,b76c <run_kernel(RuntimeParams const&)+0x48c>
        if (detail::zone_hashes[i] == hash_val)
    b348:	lui	a3,0x7c867
    b34c:	lw	a2,24(a4)
    b350:	addi	a3,a3,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    b354:	beq	a2,a3,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b358:	li	a2,6
    b35c:	beq	a5,a2,b76c <run_kernel(RuntimeParams const&)+0x48c>
        if (detail::zone_hashes[i] == hash_val)
    b360:	lw	a2,28(a4)
    b364:	beq	a2,a3,b380 <run_kernel(RuntimeParams const&)+0xa0>
    for (std::uint32_t i = 0; i < n; ++i)
    b368:	li	a2,7
    b36c:	beq	a5,a2,b76c <run_kernel(RuntimeParams const&)+0x48c>
        if (detail::zone_hashes[i] == hash_val)
    b370:	lw	a2,32(a4)
    b374:	beq	a2,a3,b380 <run_kernel(RuntimeParams const&)+0xa0>
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
    b378:	li	a3,8
    b37c:	bne	a5,a3,b76c <run_kernel(RuntimeParams const&)+0x48c>
    {
    b380:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b384:	lw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    b388:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    b38c:	li	a1,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b390:	add	a3,a5,a2
    b394:	addi	a3,a3,-1021
        if (!is_buffer_full())
    b398:	bgeu	a1,a3,b3ec <run_kernel(RuntimeParams const&)+0x10c>
    b39c:	lui	a3,0xffb12
    b3a0:	lw	a6,496(a3) # ffb121f0 <__stack_top+0x111f0>
    b3a4:	lw	a1,504(a3)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b3a8:	lui	a0,0x1
    b3ac:	lui	a3,0xffb00
    b3b0:	addi	t3,a0,-1 # fff <__firmware_stack_size+0xdff>
    b3b4:	lw	a3,0(a3) # ffb00000 <llk_profiler::buffer>
    b3b8:	lui	t4,0xa00db
    b3bc:	sh2add	a3,a5,a3
    b3c0:	add	a3,a3,a0
    b3c4:	and	a1,a1,t3
            is_opened = true;
    b3c8:	li	t3,1
    b3cc:	sb	t3,12(sp)
            ++open_zone_cnt;
    b3d0:	add	a2,a2,t3
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b3d4:	or	a1,a1,t4
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b3d8:	addi	a5,a5,2
            ++open_zone_cnt;
    b3dc:	sw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b3e0:	sw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b3e4:	sw	a1,0(a3)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b3e8:	sw	a6,4(a3)
    store_blocking(&pc_buf_base[1], 0);
    b3ec:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    asm volatile(
    b3f0:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    b3f4:	addi	a3,a3,4
    asm volatile(
    b3f8:	sw	a5,0(a3)
    b3fc:	lw	a5,0(a3)
    b400:	and	zero,zero,a5
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    b404:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b408:	lw	a5,36(a3)

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0)
    b40c:	zext.b	a5,a5
    b410:	bnez	a5,b408 <run_kernel(RuntimeParams const&)+0x128>
        set_dest_section_base<StartZero>();
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
    b414:	ttseminit	2,0,2
    dest_offset_id = 0;
    b418:	sw	zero,-2004(gp) # ffb0002c <ckernel::dest_offset_id>
template <DstStart Dst>
inline void set_dest_section_base()
{
    if constexpr (Dst == DstStart::StartZero)
    {
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
    b41c:	ttsetc16	1,0
    std::uint8_t mask_b0 = Mask & 0xff;

    if (mask_b0 != 0)
    {
        std::uint8_t data_b0 = wrdata & 0xff;
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    b420:	lui	a2,0xffe40
    b424:	lui	a1,0xb3080
    b428:	mv	a2,a2
    b42c:	addi	a1,a1,220 # b30800dc <__runtime_args_end+0xb305fcdc>
    b430:	sw	a1,0(a2) # ffe40000 <__instrn_buffer>
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    b434:	ttstallwait	128,16
    wrdata >>= 8;
    std::uint8_t mask_b3 = (Mask >> 24) & 0xff;
    if (mask_b3 != 0)
    {
        std::uint8_t data_b3 = (wrdata) & 0xff;
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
    b438:	lui	a1,0xb6800
    b43c:	addi	a1,a1,1 # b6800001 <__runtime_args_end+0xb67dfc01>
    b440:	sw	a1,0(a2)
    b444:	lui	a1,0xb6202
    b448:	addi	a1,a1,1 # b6202001 <__runtime_args_end+0xb61e1c01>
    b44c:	sw	a1,0(a2)
    b450:	lui	a1,0xb6404
    b454:	addi	a1,a1,1 # b6404001 <__runtime_args_end+0xb63e3c01>
    b458:	sw	a1,0(a2)
    // Program source and dest registers
    __attribute__((always_inline)) inline void set(const std::uint8_t mod_index) const
    {
        // KCM - This gets around issue: error: impossible constraint in 'asm'
        // TTI_SETC16(addr_mod_src_reg_addr[mod_index], src_val());
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b45c:	ttsetc16	12,2056
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b460:	ttsetc16	28,8
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b464:	ttsetc16	47,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b468:	ttsetc16	13,0
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b46c:	ttsetc16	29,0
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b470:	ttsetc16	48,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b474:	ttsetc16	14,32896
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b478:	ttsetc16	30,9216
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b47c:	ttsetc16	49,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b480:	ttsetc16	15,32896
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b484:	ttsetc16	31,36872
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b488:	ttsetc16	50,0
    store_blocking(&pc_buf_base[2], 0);
    b48c:	addi	a3,a3,8
    asm volatile(
    b490:	mv	a1,a5
    b494:	sw	a1,0(a3)
    b498:	lw	a1,0(a3)
    b49c:	and	zero,zero,a1
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
    b4a0:	lui	a3,0xffb80
    b4a4:	li	a1,2
    b4a8:	sw	a1,0(a3) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
    b4ac:	sw	a1,4(a3)
    mop_cfg[2] = m_start_op0;
    b4b0:	lui	a1,0x2000
    b4b4:	sw	a1,8(a3)
    mop_cfg[3] = m_end_op0;
    b4b8:	sw	a1,12(a3)
    mop_cfg[4] = m_end_op1;
    b4bc:	sw	a1,16(a3)
    mop_cfg[5] = m_loop_op0;
    b4c0:	lui	a0,0x27000
    b4c4:	sw	a0,20(a3)
    mop_cfg[6] = m_loop_op1;
    b4c8:	sw	a1,24(a3)
    mop_cfg[7] = m_loop0_last_instr;
    b4cc:	lui	a1,0x27c0c
    b4d0:	sw	a1,28(a3)
    mop_cfg[8] = m_loop1_last_instr;
    b4d4:	lui	a1,0x27008
    b4d8:	sw	a1,32(a3)
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    b4dc:	ttsetc16	7,0
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, setrwc);
    b4e0:	ttsetrwc	0,0,0,0,0,15
    store_blocking(&pc_buf_base[1], 0);
    b4e4:	lw	a3,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b4e8:	addi	a3,a3,4
    asm volatile(
    b4ec:	sw	a5,0(a3)
    b4f0:	lw	a5,0(a3)
    b4f4:	and	zero,zero,a5
        if (is_opened)
    b4f8:	lbu	a5,12(sp)
    b4fc:	beqz	a5,b550 <run_kernel(RuntimeParams const&)+0x270>
    return p_reg[0];
    b500:	lui	a5,0xffb12
    b504:	lw	t5,496(a5) # ffb121f0 <__stack_top+0x111f0>
    b508:	lw	a3,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b50c:	lui	a6,0x1
    b510:	lui	a5,0xffb00
    b514:	lw	a1,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b518:	lw	a0,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b51c:	addi	t6,a6,-1 # fff <__firmware_stack_size+0xdff>
    b520:	lw	a5,0(a5) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    b524:	addi	a0,a0,-1 # 26ffffff <__runtime_args_end+0x26fdfbff>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b528:	and	a3,a3,t6
    b52c:	lui	t6,0xb00db
    b530:	or	a3,a3,t6
    b534:	sh2add	a5,a1,a5
    b538:	add	a5,a5,a6
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b53c:	addi	a1,a1,2 # 27008002 <__runtime_args_end+0x26fe7c02>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b540:	sw	a3,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b544:	sw	t5,4(a5)
    b548:	sw	a1,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b54c:	sw	a0,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    std::uint32_t n = detail::next_zone_id;
    b550:	lw	a5,0(a4)
    for (std::uint32_t i = 0; i < n; ++i)
    b554:	beqz	a5,b750 <run_kernel(RuntimeParams const&)+0x470>
        if (detail::zone_hashes[i] == hash_val)
    b558:	lui	a3,0xbd77
    b55c:	lw	a1,4(a4)
    b560:	addi	a3,a3,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    b564:	beq	a1,a3,b5e8 <run_kernel(RuntimeParams const&)+0x308>
    for (std::uint32_t i = 0; i < n; ++i)
    b568:	li	a1,1
    b56c:	beq	a5,a1,b750 <run_kernel(RuntimeParams const&)+0x470>
        if (detail::zone_hashes[i] == hash_val)
    b570:	lw	a1,8(a4)
    b574:	beq	a1,a3,b5e8 <run_kernel(RuntimeParams const&)+0x308>
    for (std::uint32_t i = 0; i < n; ++i)
    b578:	li	a1,2
    b57c:	beq	a5,a1,b750 <run_kernel(RuntimeParams const&)+0x470>
        if (detail::zone_hashes[i] == hash_val)
    b580:	lw	a1,12(a4)
    b584:	beq	a1,a3,b5e8 <run_kernel(RuntimeParams const&)+0x308>
    for (std::uint32_t i = 0; i < n; ++i)
    b588:	li	a1,3
    b58c:	beq	a5,a1,b750 <run_kernel(RuntimeParams const&)+0x470>
        if (detail::zone_hashes[i] == hash_val)
    b590:	lw	a1,16(a4)
    b594:	beq	a1,a3,b5e8 <run_kernel(RuntimeParams const&)+0x308>
    for (std::uint32_t i = 0; i < n; ++i)
    b598:	li	a1,4
    b59c:	beq	a5,a1,b750 <run_kernel(RuntimeParams const&)+0x470>
        if (detail::zone_hashes[i] == hash_val)
    b5a0:	lw	a1,20(a4)
    b5a4:	beq	a1,a3,b5e8 <run_kernel(RuntimeParams const&)+0x308>
    for (std::uint32_t i = 0; i < n; ++i)
    b5a8:	li	a3,5
    b5ac:	beq	a5,a3,b750 <run_kernel(RuntimeParams const&)+0x470>
        if (detail::zone_hashes[i] == hash_val)
    b5b0:	lui	a3,0xbd77
    b5b4:	lw	a1,24(a4)
    b5b8:	addi	a3,a3,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    b5bc:	beq	a1,a3,b5e8 <run_kernel(RuntimeParams const&)+0x308>
    for (std::uint32_t i = 0; i < n; ++i)
    b5c0:	li	a1,6
    b5c4:	beq	a5,a1,b750 <run_kernel(RuntimeParams const&)+0x470>
        if (detail::zone_hashes[i] == hash_val)
    b5c8:	lw	a1,28(a4)
    b5cc:	beq	a1,a3,b5e8 <run_kernel(RuntimeParams const&)+0x308>
    for (std::uint32_t i = 0; i < n; ++i)
    b5d0:	li	a1,7
    b5d4:	beq	a5,a1,b750 <run_kernel(RuntimeParams const&)+0x470>
        if (detail::zone_hashes[i] == hash_val)
    b5d8:	lw	a1,32(a4)
    b5dc:	beq	a1,a3,b5e8 <run_kernel(RuntimeParams const&)+0x308>
    if (n < PERF_COUNTERS_MAX_ZONES)
    b5e0:	li	a3,8
    b5e4:	bne	a5,a3,b750 <run_kernel(RuntimeParams const&)+0x470>
    {
    b5e8:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b5ec:	lw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    b5f0:	lw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    b5f4:	li	a1,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b5f8:	add	a4,a5,a3
    b5fc:	addi	a4,a4,-1021
        if (!is_buffer_full())
    b600:	bgeu	a1,a4,b654 <run_kernel(RuntimeParams const&)+0x374>
    b604:	lui	a4,0xffb12
    b608:	lw	a6,496(a4) # ffb121f0 <__stack_top+0x111f0>
    b60c:	lw	a1,504(a4)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b610:	lui	a0,0x1
    b614:	lui	a4,0xffb00
    b618:	addi	t5,a0,-1 # fff <__firmware_stack_size+0xdff>
    b61c:	lw	a4,0(a4) # ffb00000 <llk_profiler::buffer>
    b620:	lui	t6,0xa1448
    b624:	sh2add	a4,a5,a4
    b628:	add	a4,a4,a0
    b62c:	and	a1,a1,t5
            is_opened = true;
    b630:	li	t5,1
    b634:	sb	t5,12(sp)
            ++open_zone_cnt;
    b638:	add	a3,a3,t5
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b63c:	or	a1,a1,t6
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b640:	addi	a5,a5,2
            ++open_zone_cnt;
    b644:	sw	a3,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b648:	sw	a5,-2012(gp) # ffb00024 <llk_profiler::write_idx>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b64c:	sw	a1,0(a4)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b650:	sw	a6,4(a4)
                }
            }
        }
        else
        {
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
    b654:	lw	a3,-2004(gp) # ffb0002c <ckernel::dest_offset_id>
        asm volatile("" ::: "memory");
    b658:	li	a0,4
    b65c:	lui	a6,0xb2010
    dest_offset_id = 1 - dest_offset_id;
    b660:	li	t5,1
    TTI_SEMWAIT(p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_MAX);
    b664:	ttsemwait	322,2,2
    return (0 != dest_offset_id) ? DEST_REGISTER_HALF_SIZE : 0x0;
    b668:	snez	a4,a3
    b66c:	slli	a4,a4,0x9
    b670:	add	a4,a4,a6
    b674:	addi	a1,a6,768 # b2010300 <__runtime_args_end+0xb1feff00>
    b678:	bnez	a3,b680 <run_kernel(RuntimeParams const&)+0x3a0>
    b67c:	addi	a1,a6,256
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
    b680:	sw	a4,0(a2)
    b684:	li	a5,4
    TTI_MOP(1, 0, 0); // run the double-loop template
    b688:	ttmop	1,0,0
        {
            // NONE/ROW/SCALAR: MOP handles all faces, fidelity requires multiple runs
            const std::uint32_t num_faces     = tensor_shape.total_num_faces();
            const std::uint32_t fidelity_loop = high_fidelity ? num_faces : 1;
#pragma GCC unroll 0
            for (std::uint32_t i = 0; i < fidelity_loop; i++)
    b68c:	addi	a5,a5,-1
    b690:	bnez	a5,b688 <run_kernel(RuntimeParams const&)+0x3a8>
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    b694:	ttsetrwc	0,0,0,0,0,4
            {
                std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
    b698:	addi	a4,a4,64
    b69c:	bne	a4,a1,b680 <run_kernel(RuntimeParams const&)+0x3a0>
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes);
    b6a0:	ttstallwait	2,2064
    TTI_SEMPOST(semaphore::t6_sem(index));
    b6a4:	ttsempost	2
    dest_offset_id = 1 - dest_offset_id;
    b6a8:	sub	a4,t5,a3
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
    b6ac:	ttstallwait	128,2064
    return (0 != dest_offset_id) ? DEST_REGISTER_HALF_SIZE : 0x0;
    b6b0:	addi	a5,a3,-1
    b6b4:	snez	a5,a5
    b6b8:	slli	a5,a5,0x9
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, base_addr);
    b6bc:	add	a5,a5,a6
    b6c0:	sw	a5,0(a2)
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
    b6c4:	addi	a0,a0,-1
    b6c8:	beqz	a0,b6d4 <run_kernel(RuntimeParams const&)+0x3f4>
    dest_offset_id = 1 - dest_offset_id;
    b6cc:	mv	a3,a4
    b6d0:	j	b664 <run_kernel(RuntimeParams const&)+0x384>
    store_blocking(&pc_buf_base[1], 0);
    b6d4:	lw	a5,-2036(gp) # ffb0000c <ckernel::pc_buf_base>
    b6d8:	addi	a5,a5,4
    b6dc:	sw	a4,-2004(gp) # ffb0002c <ckernel::dest_offset_id>
inline void store_blocking(volatile T *ptr, U &&val)
    b6e0:	sw	zero,-1996(gp) # ffb00034 <math_sync_tile_dst_index>
    asm volatile(
    b6e4:	sw	a0,0(a5)
    b6e8:	lw	a0,0(a5)
    b6ec:	and	zero,zero,a0
        if (is_opened)
    b6f0:	lbu	a5,12(sp)
    b6f4:	beqz	a5,b748 <run_kernel(RuntimeParams const&)+0x468>
    return p_reg[0];
    b6f8:	lui	a5,0xffb12
    b6fc:	lw	a0,496(a5) # ffb121f0 <__stack_top+0x111f0>
    b700:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b704:	lui	a1,0x1
    b708:	lui	a5,0xffb00
    b70c:	lw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b710:	lw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b714:	addi	a6,a1,-1 # fff <__firmware_stack_size+0xdff>
    b718:	lw	a5,0(a5) # ffb00000 <llk_profiler::buffer>
            --open_zone_cnt;
    b71c:	addi	a2,a2,-1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b720:	and	a4,a4,a6
    b724:	lui	a6,0xb1448
    b728:	or	a4,a4,a6
    b72c:	sh2add	a5,a3,a5
    b730:	add	a5,a5,a1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b734:	addi	a3,a3,2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b738:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b73c:	sw	a0,4(a5)
    b740:	sw	a3,-2012(gp) # ffb00024 <llk_profiler::write_idx>
            --open_zone_cnt;
    b744:	sw	a2,-2016(gp) # ffb00020 <llk_profiler::open_zone_cnt>
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
    b748:	addi	sp,sp,16
    b74c:	ret
    {
        detail::zone_hashes[n] = hash_val;
    b750:	lui	a3,0xbd77
    b754:	sh2add	a1,a5,a4
    b758:	addi	a3,a3,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
        detail::next_zone_id   = n + 1;
    b75c:	addi	a5,a5,1
        detail::zone_hashes[n] = hash_val;
    b760:	sw	a3,4(a1)
        detail::next_zone_id   = n + 1;
    b764:	sw	a5,0(a4)
        return n;
    b768:	j	b5e8 <run_kernel(RuntimeParams const&)+0x308>
        detail::zone_hashes[n] = hash_val;
    b76c:	lui	a3,0x7c867
    b770:	sh2add	a2,a5,a4
    b774:	addi	a3,a3,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
        detail::next_zone_id   = n + 1;
    b778:	addi	a5,a5,1
        detail::zone_hashes[n] = hash_val;
    b77c:	sw	a3,4(a2)
        detail::next_zone_id   = n + 1;
    b780:	sw	a5,0(a4)
        return n;
    b784:	j	b380 <run_kernel(RuntimeParams const&)+0xa0>

0000b788 <_init()>:
    }
}

void _init(void)
{
}
    b788:	ret

0000b78c <_fini()>:

void _fini(void)
    b78c:	ret

0000b790 <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
    b790:	lui	a5,0x20
    b794:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
    b798:	sb	a5,0(a0)
        (void)(dstc[i]);
    b79c:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
    b7a0:	fence
}
    b7a4:	ret

0000b7a8 <memset>:
    b7a8:	li	t1,15
    b7ac:	mv	a4,a0
    b7b0:	bgeu	t1,a2,b7ec <memset+0x44>
    b7b4:	andi	a5,a4,15
    b7b8:	bnez	a5,b858 <memset+0xb0>
    b7bc:	bnez	a1,b840 <memset+0x98>
    b7c0:	andi	a3,a2,-16
    b7c4:	andi	a2,a2,15
    b7c8:	add	a3,a3,a4
    b7cc:	sw	a1,0(a4)
    b7d0:	sw	a1,4(a4)
    b7d4:	sw	a1,8(a4)
    b7d8:	sw	a1,12(a4)
    b7dc:	addi	a4,a4,16
    b7e0:	bltu	a4,a3,b7cc <memset+0x24>
    b7e4:	bnez	a2,b7ec <memset+0x44>
    b7e8:	ret
    b7ec:	sub	a3,t1,a2
    b7f0:	slli	a3,a3,0x2
    b7f4:	auipc	t0,0x0
    b7f8:	add	a3,a3,t0
    b7fc:	jr	12(a3)
    b800:	sb	a1,14(a4)
    b804:	sb	a1,13(a4)
    b808:	sb	a1,12(a4)
    b80c:	sb	a1,11(a4)
    b810:	sb	a1,10(a4)
    b814:	sb	a1,9(a4)
    b818:	sb	a1,8(a4)
    b81c:	sb	a1,7(a4)
    b820:	sb	a1,6(a4)
    b824:	sb	a1,5(a4)
    b828:	sb	a1,4(a4)
    b82c:	sb	a1,3(a4)
    b830:	sb	a1,2(a4)
    b834:	sb	a1,1(a4)
    b838:	sb	a1,0(a4)
    b83c:	ret
    b840:	zext.b	a1,a1
    b844:	slli	a3,a1,0x8
    b848:	or	a1,a1,a3
    b84c:	slli	a3,a1,0x10
    b850:	or	a1,a1,a3
    b854:	j	b7c0 <memset+0x18>
    b858:	slli	a3,a5,0x2
    b85c:	auipc	t0,0x0
    b860:	add	a3,a3,t0
    b864:	mv	t0,ra
    b868:	jalr	-96(a3)
    b86c:	mv	ra,t0
    b870:	addi	a5,a5,-16
    b874:	sub	a4,a4,a5
    b878:	add	a2,a2,a5
    b87c:	bgeu	t1,a2,b7ec <memset+0x44>
    b880:	j	b7bc <memset+0x14>
