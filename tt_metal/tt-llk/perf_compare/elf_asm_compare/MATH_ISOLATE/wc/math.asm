
/tmp/elf_compare/wc/MATH_ISOLATE/math.elf:     file format elf32-littleriscv


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
    b018:	addi	a5,a5,72 # ffb00048 <llk_profiler::open_zone_cnt>
    b01c:	addi	a4,a4,140 # ffb0008c <__gcov_info_end>
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
    b064:	addi	a3,a3,72 # ffb00048 <llk_profiler::open_zone_cnt>
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
    b088:	sw	a1,0(a4) # ffb00000 <llk_perf::perf_counter_scoped<(PerfRunType)2>::~perf_counter_scoped()::banks>
    b08c:	addi	a4,a4,4
        while (dst < end)
    b090:	bne	a5,a3,b080 <_start+0x80>
        }
    }

    // Execute global constructors
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
    b094:	lui	s0,0xffb00
    b098:	lui	s1,0xffb00
    b09c:	addi	s0,s0,40 # ffb00028 <llk_profiler::buffer>
    b0a0:	addi	s1,s1,40 # ffb00028 <llk_profiler::buffer>
    b0a4:	bgeu	s0,s1,b0b8 <_start+0xb8>
    {
        (*temp_constructor)();
    b0a8:	lw	a5,0(s0)
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
    b0ec:	lw	a5,-2000(gp) # ffb00030 <ckernel::regfile>
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
    b104:	addi	a4,a5,-12 # 16aff4 <__runtime_args_end+0x14abf4>
    buffer        = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
    b108:	sw	a5,-2008(gp) # ffb00028 <llk_profiler::buffer>
    TTI_NOP;
}

inline void reset_cfg_state_id()
{
    cfg_state_id = 0;
    b10c:	sw	zero,-1960(gp) # ffb00058 <ckernel::cfg_state_id>
    write_idx     = 0;
    open_zone_cnt = 0;

    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
    b110:	lui	a2,0x1
    b114:	li	a1,0
    b118:	lui	a0,0x16c
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
    b11c:	sw	a4,-2004(gp) # ffb0002c <llk_profiler::barrier_ptr>
}

inline void reset_dest_offset_id()
{
    dest_offset_id = 0;
    b120:	sw	zero,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
    write_idx     = 0;
    b124:	sw	zero,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    open_zone_cnt = 0;
    b128:	sw	zero,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
    b12c:	jal	b978 <memset>
    auto& barrier = *barrier_ptr;
    b130:	lw	a2,-2004(gp) # ffb0002c <llk_profiler::barrier_ptr>
    barrier[TRISC_ID] = 1;
    b134:	li	a4,1
    b138:	sw	a4,4(a2) # 1004 <TRISC_LOCAL_MEM_LENGTH+0x4>
    asm volatile("fence" ::: "memory");
    b13c:	fence
    for (std::uint32_t i = 0; i < NUM_CORES; ++i)
    b140:	li	a5,0
    b144:	li	a6,2
        if (i == TRISC_ID)
    b148:	sh3add	a3,a5,a2
        while (barrier[i] != 1)
    b14c:	lw	a0,0(a3)
        if (i == TRISC_ID)
    b150:	slli	a1,a5,0x1
        while (barrier[i] != 1)
    b154:	beq	a0,a4,b164 <main+0xa4>
            asm volatile("fence" ::: "memory");
    b158:	fence
        while (barrier[i] != 1)
    b15c:	lw	a5,0(a3)
    b160:	bne	a5,a4,b158 <main+0x98>
    for (std::uint32_t i = 0; i < NUM_CORES; ++i)
    b164:	li	a5,1
    b168:	bne	a1,a6,b148 <main+0x88>
    zone_scoped(zone_scoped&&)                 = delete;
    zone_scoped& operator=(const zone_scoped&) = delete;
    zone_scoped& operator=(zone_scoped&&)      = delete;

    inline __attribute__((always_inline)) zone_scoped()
    {
    b16c:	sb	zero,12(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b170:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    b174:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        asm volatile("" ::: "memory");
        if (!is_buffer_full())
    b178:	li	a1,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b17c:	add	a5,a3,a2
    b180:	addi	a5,a5,-1021
        if (!is_buffer_full())
    b184:	bgeu	a1,a5,b1d0 <main+0x110>
// now handled by the compiler)
// workaround is needed only for GS
inline std::uint32_t reg_read(std::uint32_t addr)
{
    volatile std::uint32_t tt_reg_ptr *p_reg = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(addr);
    return p_reg[0];
    b188:	lui	a5,0xffb12
    b18c:	lw	a0,496(a5) # ffb121f0 <__stack_top+0x111f0>
        {
            is_opened = true;
    b190:	sb	a4,12(sp)
    b194:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b198:	lui	a1,0x1
    b19c:	addi	a6,a1,-1 # fff <__firmware_stack_size+0xdff>
    b1a0:	lw	a5,-2008(gp) # ffb00028 <llk_profiler::buffer>
            write_entry(EntryType::ZONE_START, id16);
            ++open_zone_cnt;
    b1a4:	addi	a2,a2,1
    b1a8:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b1ac:	and	a4,a4,a6
    b1b0:	lui	a6,0xa5104
    b1b4:	or	a4,a4,a6
    b1b8:	sh2add	a5,a3,a5
    b1bc:	add	a5,a5,a1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b1c0:	addi	a3,a3,2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b1c4:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b1c8:	sw	a0,4(a5)
    b1cc:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
        run_kernel(temp_args);
    b1d0:	addi	a0,sp,8
    b1d4:	jal	b2d8 <run_kernel(RuntimeParams const&)>
    store_blocking(&pc_buf_base[1], 0);
    b1d8:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
    b1dc:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    b1e0:	addi	a4,a4,4
    asm volatile(
    b1e4:	sw	a5,0(a4)
    b1e8:	lw	a5,0(a4)
    b1ec:	and	zero,zero,a5
    }

    ~zone_scoped()
    {
        asm volatile("" ::: "memory");
        if (is_opened)
    b1f0:	lbu	a5,12(sp)
    b1f4:	beqz	a5,b244 <main+0x184>
    return p_reg[0];
    b1f8:	lui	a5,0xffb12
    b1fc:	lw	a0,496(a5) # ffb121f0 <__stack_top+0x111f0>
    b200:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b204:	lui	a1,0x1
    b208:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
        {
            write_entry(EntryType::ZONE_END, id16);
            --open_zone_cnt;
    b20c:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b210:	addi	a6,a1,-1 # fff <__firmware_stack_size+0xdff>
    b214:	lw	a5,-2008(gp) # ffb00028 <llk_profiler::buffer>
    b218:	and	a4,a4,a6
    b21c:	lui	a6,0xb5104
    b220:	or	a4,a4,a6
    b224:	sh2add	a5,a3,a5
    b228:	add	a5,a5,a1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b22c:	addi	a3,a3,2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b230:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b234:	sw	a0,4(a5)
    b238:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
    b23c:	addi	a2,a2,-1
    b240:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    *mailbox = ckernel::KERNEL_COMPLETE;
    b244:	lui	a5,0x20
}
    b248:	lw	ra,44(sp)
    b24c:	lw	s0,40(sp)
    *mailbox = ckernel::KERNEL_COMPLETE;
    b250:	li	a4,255
    b254:	sw	a4,-68(a5) # 1ffbc <__loader_init_end+0xffbc>
}
    b258:	lw	s1,36(sp)
    b25c:	lw	s2,32(sp)
    b260:	lw	s3,28(sp)
    b264:	li	a0,0
    b268:	addi	sp,sp,48
    b26c:	ret

0000b270 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&) [clone .constprop.0]>:
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t face_c_dim = tensor_shape.face_c_dim;
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
}
    b270:	li	a0,1
    b274:	ret

0000b278 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)>:
        return num_faces_r_dim * num_faces_c_dim;
    b278:	lbu	a5,3(a0) # 16c003 <__runtime_args_end+0x14bc03>
    b27c:	lbu	a4,2(a0)
{
    b280:	mv	a3,a0
        return num_faces_r_dim * num_faces_c_dim;
    b284:	mul	a4,a4,a5
    b288:	zext.b	a4,a4
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
    b28c:	addi	a5,a4,-1
    b290:	addi	a4,a4,-4
    b294:	sltiu	a5,a5,2
    b298:	seqz	a4,a4
    b29c:	or	a0,a5,a4
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
    b2a0:	beqz	a0,b2d4 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    b2a4:	lbu	a4,0(a3)
    b2a8:	li	a5,16
    b2ac:	li	a0,0
    b2b0:	bltu	a5,a4,b2d4 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
    b2b4:	lui	a5,0x10
    b2b8:	addi	a5,a5,278 # 10116 <__loader_init_end+0x116>
    b2bc:	srl	a5,a5,a4
    b2c0:	andi	a0,a5,1
    b2c4:	beqz	a0,b2d4 <ckernel::validate_tensor_shape_tile_dependent_ops_(ckernel::TensorShape const&)+0x5c>
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
    b2c8:	lbu	a0,1(a3)
    b2cc:	addi	a0,a0,-16
    b2d0:	seqz	a0,a0
}
    b2d4:	ret

0000b2d8 <run_kernel(RuntimeParams const&)>:
#endif
} // namespace detail

__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    std::uint32_t n = detail::next_zone_id;
    b2d8:	addi	t2,gp,-1944 # ffb00068 <llk_perf::detail::next_zone_id>

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    b2dc:	addi	sp,sp,-48
    b2e0:	lw	a5,0(t2)
    b2e4:	sw	s0,44(sp)
    b2e8:	sw	s1,40(sp)
    b2ec:	sw	s2,36(sp)
    b2f0:	sw	s3,32(sp)
    b2f4:	sw	s4,28(sp)
    for (std::uint32_t i = 0; i < n; ++i)
    b2f8:	beqz	a5,b900 <run_kernel(RuntimeParams const&)+0x628>
    {
        if (detail::zone_hashes[i] == hash_val)
    b2fc:	lui	a4,0x7c867
    b300:	lw	a3,4(t2)
    b304:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    b308:	beq	a3,a4,b38c <run_kernel(RuntimeParams const&)+0xb4>
    for (std::uint32_t i = 0; i < n; ++i)
    b30c:	li	a3,1
    b310:	beq	a5,a3,b900 <run_kernel(RuntimeParams const&)+0x628>
        if (detail::zone_hashes[i] == hash_val)
    b314:	lw	a2,8(t2)
    b318:	beq	a2,a4,b940 <run_kernel(RuntimeParams const&)+0x668>
    for (std::uint32_t i = 0; i < n; ++i)
    b31c:	li	a3,2
    b320:	beq	a5,a3,b900 <run_kernel(RuntimeParams const&)+0x628>
        if (detail::zone_hashes[i] == hash_val)
    b324:	lw	a2,12(t2)
    b328:	beq	a2,a4,b940 <run_kernel(RuntimeParams const&)+0x668>
    for (std::uint32_t i = 0; i < n; ++i)
    b32c:	li	a3,3
    b330:	beq	a5,a3,b900 <run_kernel(RuntimeParams const&)+0x628>
        if (detail::zone_hashes[i] == hash_val)
    b334:	lw	a2,16(t2)
    b338:	beq	a2,a4,b940 <run_kernel(RuntimeParams const&)+0x668>
    for (std::uint32_t i = 0; i < n; ++i)
    b33c:	li	a3,4
    b340:	beq	a5,a3,b900 <run_kernel(RuntimeParams const&)+0x628>
        if (detail::zone_hashes[i] == hash_val)
    b344:	lw	a3,20(t2)
    b348:	beq	a3,a4,b950 <run_kernel(RuntimeParams const&)+0x678>
    for (std::uint32_t i = 0; i < n; ++i)
    b34c:	li	a3,5
    b350:	beq	a5,a3,b900 <run_kernel(RuntimeParams const&)+0x628>
        if (detail::zone_hashes[i] == hash_val)
    b354:	lui	a4,0x7c867
    b358:	lw	a2,24(t2)
    b35c:	addi	a4,a4,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
    b360:	beq	a2,a4,b940 <run_kernel(RuntimeParams const&)+0x668>
    for (std::uint32_t i = 0; i < n; ++i)
    b364:	li	a3,6
    b368:	beq	a5,a3,b900 <run_kernel(RuntimeParams const&)+0x628>
        if (detail::zone_hashes[i] == hash_val)
    b36c:	lw	a2,28(t2)
    b370:	beq	a2,a4,b940 <run_kernel(RuntimeParams const&)+0x668>
    for (std::uint32_t i = 0; i < n; ++i)
    b374:	li	a3,7
    b378:	beq	a5,a3,b900 <run_kernel(RuntimeParams const&)+0x628>
        if (detail::zone_hashes[i] == hash_val)
    b37c:	lw	a2,32(t2)
    b380:	beq	a2,a4,b940 <run_kernel(RuntimeParams const&)+0x668>
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
    b384:	li	a4,8
    b388:	bne	a5,a4,b900 <run_kernel(RuntimeParams const&)+0x628>
    {
        detail::zone_hashes[n] = hash_val;
        detail::next_zone_id   = n + 1;
        return n;
    }
    return 0;
    b38c:	li	a5,0
    perf_counter_scoped(const perf_counter_scoped&)            = delete;
    perf_counter_scoped(perf_counter_scoped&&)                 = delete;
    perf_counter_scoped& operator=(const perf_counter_scoped&) = delete;
    perf_counter_scoped& operator=(perf_counter_scoped&&)      = delete;

    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
    b390:	sw	a5,12(sp)
    {
        if constexpr (perf_counter_thread_active<run_type>())
        {
            asm volatile("" ::: "memory");
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 1u; // PERF_CNT_ALL (INSTRN+FPU)
    b394:	lui	a5,0xffb12
    b398:	li	a4,1
    b39c:	sw	a4,60(a5) # ffb1203c <__stack_top+0x1103c>
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 1u; // TDMA_UNPACK
    b3a0:	sw	a4,20(a5)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 1u; // L1
    b3a4:	sw	a4,56(a5)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 1u; // TDMA_PACK
    b3a8:	sw	a4,248(a5)
    {
    b3ac:	sb	zero,8(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b3b0:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    b3b4:	lw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    b3b8:	li	a0,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b3bc:	add	a2,a3,a1
    b3c0:	addi	a2,a2,-1021
        if (!is_buffer_full())
    b3c4:	bgeu	a0,a2,b40c <run_kernel(RuntimeParams const&)+0x134>
    b3c8:	lw	a0,496(a5)
            is_opened = true;
    b3cc:	sb	a4,8(sp)
    b3d0:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b3d4:	lui	a2,0x1
    b3d8:	addi	a6,a2,-1 # fff <__firmware_stack_size+0xdff>
    b3dc:	lw	a5,-2008(gp) # ffb00028 <llk_profiler::buffer>
    b3e0:	sh2add	a5,a3,a5
            ++open_zone_cnt;
    b3e4:	addi	a1,a1,1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b3e8:	add	a5,a5,a2
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b3ec:	addi	a3,a3,2
            ++open_zone_cnt;
    b3f0:	sw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b3f4:	and	a4,a4,a6
    b3f8:	lui	a6,0xa00db
    b3fc:	or	a4,a4,a6
    b400:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b404:	sw	a0,4(a5)
    b408:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    store_blocking(&pc_buf_base[1], 0);
    b40c:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
    b410:	li	a5,0
    store_blocking(&pc_buf_base[1], 0);
    b414:	addi	a4,a4,4
    asm volatile(
    b418:	sw	a5,0(a4)
    b41c:	lw	a5,0(a4)
    b420:	and	zero,zero,a5
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
    b424:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    b428:	lw	a5,36(a4)

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0)
    b42c:	zext.b	a5,a5
    b430:	bnez	a5,b428 <run_kernel(RuntimeParams const&)+0x150>
        set_dest_section_base<StartZero>();
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
    b434:	ttseminit	2,0,2
    dest_offset_id = 0;
    b438:	sw	zero,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
template <DstStart Dst>
inline void set_dest_section_base()
{
    if constexpr (Dst == DstStart::StartZero)
    {
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
    b43c:	ttsetc16	1,0
    std::uint8_t mask_b0 = Mask & 0xff;

    if (mask_b0 != 0)
    {
        std::uint8_t data_b0 = wrdata & 0xff;
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    b440:	lui	t0,0xffe40
    b444:	lui	a3,0xb3080
    b448:	mv	t0,t0
    b44c:	addi	a3,a3,220 # b30800dc <__runtime_args_end+0xb305fcdc>
    b450:	sw	a3,0(t0) # ffe40000 <__instrn_buffer>
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    b454:	ttstallwait	128,16
    wrdata >>= 8;
    std::uint8_t mask_b3 = (Mask >> 24) & 0xff;
    if (mask_b3 != 0)
    {
        std::uint8_t data_b3 = (wrdata) & 0xff;
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
    b458:	lui	a3,0xb6800
    b45c:	addi	a3,a3,1 # b6800001 <__runtime_args_end+0xb67dfc01>
    b460:	sw	a3,0(t0)
    b464:	lui	a3,0xb6202
    b468:	addi	a3,a3,1 # b6202001 <__runtime_args_end+0xb61e1c01>
    b46c:	sw	a3,0(t0)
    b470:	lui	a3,0xb6404
    b474:	addi	a3,a3,1 # b6404001 <__runtime_args_end+0xb63e3c01>
    b478:	sw	a3,0(t0)
    // Program source and dest registers
    __attribute__((always_inline)) inline void set(const std::uint8_t mod_index) const
    {
        // KCM - This gets around issue: error: impossible constraint in 'asm'
        // TTI_SETC16(addr_mod_src_reg_addr[mod_index], src_val());
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b47c:	ttsetc16	12,2056
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b480:	ttsetc16	28,8
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b484:	ttsetc16	47,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b488:	ttsetc16	13,0
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b48c:	ttsetc16	29,0
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b490:	ttsetc16	48,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b494:	ttsetc16	14,32896
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b498:	ttsetc16	30,9216
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b49c:	ttsetc16	49,0
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
    b4a0:	ttsetc16	15,32896
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
    b4a4:	ttsetc16	31,36872
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    b4a8:	ttsetc16	50,0
    store_blocking(&pc_buf_base[2], 0);
    b4ac:	addi	a4,a4,8
    asm volatile(
    b4b0:	mv	a3,a5
    b4b4:	sw	a3,0(a4)
    b4b8:	lw	a3,0(a4)
    b4bc:	and	zero,zero,a3
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
    b4c0:	lui	a4,0xffb80
    b4c4:	li	a3,2
    b4c8:	sw	a3,0(a4) # ffb80000 <__stack_top+0x7f000>
    mop_cfg[1] = m_inner_loop_len;
    b4cc:	sw	a3,4(a4)
    mop_cfg[2] = m_start_op0;
    b4d0:	lui	a3,0x2000
    b4d4:	sw	a3,8(a4)
    mop_cfg[3] = m_end_op0;
    b4d8:	sw	a3,12(a4)
    mop_cfg[4] = m_end_op1;
    b4dc:	sw	a3,16(a4)
    mop_cfg[5] = m_loop_op0;
    b4e0:	lui	a2,0x27000
    b4e4:	sw	a2,20(a4)
    mop_cfg[6] = m_loop_op1;
    b4e8:	sw	a3,24(a4)
    mop_cfg[7] = m_loop0_last_instr;
    b4ec:	lui	a3,0x27c0c
    b4f0:	sw	a3,28(a4)
    mop_cfg[8] = m_loop1_last_instr;
    b4f4:	lui	a3,0x27008
    b4f8:	sw	a3,32(a4)
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    b4fc:	ttsetc16	7,0
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, setrwc);
    b500:	ttsetrwc	0,0,0,0,0,15
    store_blocking(&pc_buf_base[1], 0);
    b504:	lw	a4,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    b508:	addi	a4,a4,4
    asm volatile(
    b50c:	sw	a5,0(a4)
    b510:	lw	a5,0(a4)
    b514:	and	zero,zero,a5
        if (is_opened)
    b518:	lbu	a5,8(sp)
    b51c:	beqz	a5,b56c <run_kernel(RuntimeParams const&)+0x294>
    return p_reg[0];
    b520:	lui	a5,0xffb12
    b524:	lw	a0,496(a5) # ffb121f0 <__stack_top+0x111f0>
    b528:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b52c:	lui	a1,0x1
    b530:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
    b534:	lw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b538:	addi	a6,a1,-1 # fff <__firmware_stack_size+0xdff>
    b53c:	lw	a5,-2008(gp) # ffb00028 <llk_profiler::buffer>
            --open_zone_cnt;
    b540:	addi	a2,a2,-1 # 26ffffff <__runtime_args_end+0x26fdfbff>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b544:	and	a4,a4,a6
    b548:	lui	a6,0xb00db
    b54c:	or	a4,a4,a6
    b550:	sh2add	a5,a3,a5
    b554:	add	a5,a5,a1
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b558:	addi	a3,a3,2 # 27008002 <__runtime_args_end+0x26fe7c02>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b55c:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b560:	sw	a0,4(a5)
    b564:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
    b568:	sw	a2,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        if constexpr (perf_counter_thread_active<run_type>())
        {
            asm volatile("" ::: "memory");
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u; // PERF_CNT_ALL
    b56c:	lui	t6,0xffb12
    b570:	li	a5,2
    b574:	sw	a5,60(t6) # ffb1203c <__stack_top+0x1103c>
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u; // TDMA_UNPACK
    b578:	sw	a5,20(t6)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u; // L1
    b57c:	sw	a5,56(t6)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u; // TDMA_PACK
    b580:	sw	a5,248(t6)
                {0xFFB12010u, 0xFFB12108u}, // 2 TDMA_UNPACK
                {0xFFB12034u, 0xFFB12118u}, // 3 L1
                {0xFFB120F4u, 0xFFB12110u}, // 4 TDMA_PACK
            };

            std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
    b584:	lw	a7,12(sp)
            volatile std::uint32_t* bank_cycles    = reinterpret_cast<volatile std::uint32_t*>(cycles_base);
            volatile std::uint32_t* counter_counts = bank_cycles + PERF_COUNTERS_BANK_CYCLES_WORDS;

            // INSTRN OUT_L replicated to all banks: FPU/L1 OUT_L return 0 on 2nd+ zone
            // when counter_sel is high (HW quirk); INSTRN cycles agree within ±30 cyc.
            std::uint32_t shared_cycles = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
    b588:	lw	a5,256(t6)
            std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
    b58c:	li	a4,860
    b590:	mul	a7,a7,a4
    b594:	lui	a4,0x169
    b598:	addi	t3,a4,800 # 169320 <__runtime_args_end+0x148f20>
    b59c:	add	a7,a7,t3
            bank_cycles[0]              = shared_cycles;
    b5a0:	sw	a5,0(a7)
            bank_cycles[1]              = shared_cycles;
    b5a4:	sw	a5,4(a7)
            bank_cycles[2]              = shared_cycles;
    b5a8:	sw	a5,8(a7)
            bank_cycles[3]              = shared_cycles;
    b5ac:	sw	a5,12(a7)
                if (bank_id == 3u)
                {
                    volatile std::uint32_t tt_reg_ptr* mux = reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12218u);
                    *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
                }
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b5b0:	lui	t4,0xffb00
    b5b4:	lui	t1,0x20
            bank_cycles[4]              = shared_cycles;
    b5b8:	sw	a5,16(a7)
            for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
    b5bc:	mv	s3,a4
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b5c0:	mv	t4,t4
    b5c4:	addi	t1,t1,-256 # 1ff00 <__loader_init_end+0xff00>
            std::uint32_t out_idx             = 0;
    b5c8:	li	a2,0
                if (bank_id == 3u)
    b5cc:	li	t5,3
                std::uint32_t cw = cfg[i];
    b5d0:	lw	a5,0(a4)
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    b5d4:	sh2add	a3,a2,a7
    b5d8:	addi	a3,a3,20
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b5dc:	and	a6,a5,t1
                if (!(cw & 0x80000000u))
    b5e0:	bgez	a5,b620 <run_kernel(RuntimeParams const&)+0x348>
                std::uint32_t bank_id    = cw & 0xFFu;
    b5e4:	zext.b	a0,a5
                ++out_idx;
    b5e8:	addi	a2,a2,1
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b5ec:	sh3add	a1,a0,t4
                if (bank_id == 3u)
    b5f0:	bne	a0,t5,b60c <run_kernel(RuntimeParams const&)+0x334>
                    *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
    b5f4:	lw	a0,536(t6)
    b5f8:	srli	a5,a5,0xd
    b5fc:	andi	a5,a5,112
    b600:	andi	a0,a0,-113
    b604:	or	a5,a5,a0
    b608:	sw	a5,536(t6)
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b60c:	lw	a0,0(a1)
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    b610:	lw	a5,4(a1)
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b614:	sw	a6,0(a0)
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    b618:	lw	a5,4(a5)
    b61c:	sw	a5,0(a3)
            for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
    b620:	addi	a4,a4,4
    b624:	bne	a4,t3,b5d0 <run_kernel(RuntimeParams const&)+0x2f8>
            }

            std::uint32_t sync_addr                               = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE + PERF_COUNTERS_ZONE_DATA_BYTES;
    b628:	lw	a5,12(sp)
    b62c:	li	a4,860
    b630:	mul	a5,a5,a4
            *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
    b634:	li	a4,255
            std::uint32_t sync_addr                               = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE + PERF_COUNTERS_ZONE_DATA_BYTES;
    b638:	add	a5,a5,s3
            *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
    b63c:	sw	a4,1620(a5)
    std::uint32_t n = detail::next_zone_id;
    b640:	lw	a5,0(t2)
    for (std::uint32_t i = 0; i < n; ++i)
    b644:	beqz	a5,b91c <run_kernel(RuntimeParams const&)+0x644>
        if (detail::zone_hashes[i] == hash_val)
    b648:	lui	a4,0xbd77
    b64c:	lw	a3,4(t2)
    b650:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    b654:	beq	a3,a4,b6d8 <run_kernel(RuntimeParams const&)+0x400>
    for (std::uint32_t i = 0; i < n; ++i)
    b658:	li	a3,1
    b65c:	beq	a5,a3,b91c <run_kernel(RuntimeParams const&)+0x644>
        if (detail::zone_hashes[i] == hash_val)
    b660:	lw	a2,8(t2)
    b664:	beq	a2,a4,b938 <run_kernel(RuntimeParams const&)+0x660>
    for (std::uint32_t i = 0; i < n; ++i)
    b668:	li	a3,2
    b66c:	beq	a5,a3,b91c <run_kernel(RuntimeParams const&)+0x644>
        if (detail::zone_hashes[i] == hash_val)
    b670:	lw	a2,12(t2)
    b674:	beq	a2,a4,b938 <run_kernel(RuntimeParams const&)+0x660>
    for (std::uint32_t i = 0; i < n; ++i)
    b678:	li	a3,3
    b67c:	beq	a5,a3,b91c <run_kernel(RuntimeParams const&)+0x644>
        if (detail::zone_hashes[i] == hash_val)
    b680:	lw	a2,16(t2)
    b684:	beq	a2,a4,b938 <run_kernel(RuntimeParams const&)+0x660>
    for (std::uint32_t i = 0; i < n; ++i)
    b688:	li	a4,4
    b68c:	beq	a5,a4,b91c <run_kernel(RuntimeParams const&)+0x644>
        if (detail::zone_hashes[i] == hash_val)
    b690:	lui	a4,0xbd77
    b694:	lw	a3,20(t2)
    b698:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
    b69c:	beq	a3,a4,b948 <run_kernel(RuntimeParams const&)+0x670>
    for (std::uint32_t i = 0; i < n; ++i)
    b6a0:	li	a3,5
    b6a4:	beq	a5,a3,b91c <run_kernel(RuntimeParams const&)+0x644>
        if (detail::zone_hashes[i] == hash_val)
    b6a8:	lw	a2,24(t2)
    b6ac:	beq	a2,a4,b938 <run_kernel(RuntimeParams const&)+0x660>
    for (std::uint32_t i = 0; i < n; ++i)
    b6b0:	li	a3,6
    b6b4:	beq	a5,a3,b91c <run_kernel(RuntimeParams const&)+0x644>
        if (detail::zone_hashes[i] == hash_val)
    b6b8:	lw	a2,28(t2)
    b6bc:	beq	a2,a4,b938 <run_kernel(RuntimeParams const&)+0x660>
    for (std::uint32_t i = 0; i < n; ++i)
    b6c0:	li	a3,7
    b6c4:	beq	a5,a3,b91c <run_kernel(RuntimeParams const&)+0x644>
        if (detail::zone_hashes[i] == hash_val)
    b6c8:	lw	a2,32(t2)
    b6cc:	beq	a2,a4,b938 <run_kernel(RuntimeParams const&)+0x660>
    if (n < PERF_COUNTERS_MAX_ZONES)
    b6d0:	li	a4,8
    b6d4:	bne	a5,a4,b91c <run_kernel(RuntimeParams const&)+0x644>
    return 0;
    b6d8:	li	a5,0
    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
    b6dc:	sw	a5,12(sp)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 1u; // PERF_CNT_ALL (INSTRN+FPU)
    b6e0:	lui	a5,0xffb12
    b6e4:	li	a4,1
    b6e8:	sw	a4,60(a5) # ffb1203c <__stack_top+0x1103c>
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 1u; // TDMA_UNPACK
    b6ec:	sw	a4,20(a5)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 1u; // L1
    b6f0:	sw	a4,56(a5)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 1u; // TDMA_PACK
    b6f4:	sw	a4,248(a5)
    {
    b6f8:	sb	zero,8(sp)
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b6fc:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    b700:	lw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
        if (!is_buffer_full())
    b704:	li	a0,3
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
    b708:	add	a2,a3,a1
    b70c:	addi	a2,a2,-1021
        if (!is_buffer_full())
    b710:	bgeu	a0,a2,b758 <run_kernel(RuntimeParams const&)+0x480>
    b714:	lw	a0,496(a5)
            is_opened = true;
    b718:	sb	a4,8(sp)
    b71c:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b720:	lui	a2,0x1
    b724:	addi	a6,a2,-1 # fff <__firmware_stack_size+0xdff>
    b728:	lw	a5,-2008(gp) # ffb00028 <llk_profiler::buffer>
    b72c:	sh2add	a5,a3,a5
            ++open_zone_cnt;
    b730:	addi	a1,a1,1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b734:	add	a5,a5,a2
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b738:	addi	a3,a3,2
            ++open_zone_cnt;
    b73c:	sw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b740:	and	a4,a4,a6
    b744:	lui	a6,0xa1448
    b748:	or	a4,a4,a6
    b74c:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b750:	sw	a0,4(a5)
    b754:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
    return (0 != dest_offset_id) ? DEST_REGISTER_HALF_SIZE : 0x0;
    b758:	lw	a6,-1964(gp) # ffb00054 <ckernel::dest_offset_id>
    b75c:	lui	a0,0xb2010
    b760:	snez	a1,a6
    b764:	slli	a1,a1,0x9
    b768:	add	a1,a1,a0
    b76c:	li	a2,4
        {
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
    b770:	addi	a3,a0,768 # b2010300 <__runtime_args_end+0xb1feff00>
    b774:	bnez	a6,b77c <run_kernel(RuntimeParams const&)+0x4a4>
    b778:	addi	a3,a0,256
    return 0;
    b77c:	mv	a4,a1
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
    b780:	sw	a4,0(t0)
    b784:	li	a5,4
    TTI_MOP(1, 0, 0); // run the double-loop template
    b788:	ttmop	1,0,0
        {
            // NONE/ROW/SCALAR: MOP handles all faces, fidelity requires multiple runs
            const std::uint32_t num_faces     = tensor_shape.total_num_faces();
            const std::uint32_t fidelity_loop = high_fidelity ? num_faces : 1;
#pragma GCC unroll 0
            for (std::uint32_t i = 0; i < fidelity_loop; i++)
    b78c:	addi	a5,a5,-1
    b790:	bnez	a5,b788 <run_kernel(RuntimeParams const&)+0x4b0>
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    b794:	ttsetrwc	0,0,0,0,0,4
    b798:	addi	a4,a4,64
    b79c:	bne	a3,a4,b780 <run_kernel(RuntimeParams const&)+0x4a8>
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
    b7a0:	addi	a2,a2,-1
    b7a4:	bnez	a2,b770 <run_kernel(RuntimeParams const&)+0x498>
    store_blocking(&pc_buf_base[1], 0);
    b7a8:	lw	a5,-1996(gp) # ffb00034 <ckernel::pc_buf_base>
    asm volatile(
    b7ac:	mv	a4,a2
    store_blocking(&pc_buf_base[1], 0);
    b7b0:	addi	a5,a5,4
    asm volatile(
    b7b4:	sw	a4,0(a5)
    b7b8:	lw	a4,0(a5)
    b7bc:	and	zero,zero,a4
        if (is_opened)
    b7c0:	lbu	a5,8(sp)
    b7c4:	beqz	a5,b814 <run_kernel(RuntimeParams const&)+0x53c>
    return p_reg[0];
    b7c8:	lui	a5,0xffb12
    b7cc:	lw	a6,496(a5) # ffb121f0 <__stack_top+0x111f0>
    b7d0:	lw	a4,504(a5)
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b7d4:	lui	a0,0x1
    b7d8:	lw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
    b7dc:	lw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b7e0:	addi	a7,a0,-1 # fff <__firmware_stack_size+0xdff>
    b7e4:	lw	a5,-2008(gp) # ffb00028 <llk_profiler::buffer>
            --open_zone_cnt;
    b7e8:	addi	a1,a1,-1
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b7ec:	and	a4,a4,a7
    b7f0:	lui	a7,0xb1448
    b7f4:	or	a4,a4,a7
    b7f8:	sh2add	a5,a3,a5
    b7fc:	add	a5,a5,a0
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b800:	addi	a3,a3,2
    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    b804:	sw	a4,0(a5)
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
    b808:	sw	a6,4(a5)
    b80c:	sw	a3,-1972(gp) # ffb0004c <llk_profiler::write_idx>
            --open_zone_cnt;
    b810:	sw	a1,-1976(gp) # ffb00048 <llk_profiler::open_zone_cnt>
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u; // PERF_CNT_ALL
    b814:	lui	t6,0xffb12
    b818:	li	a5,2
    b81c:	sw	a5,60(t6) # ffb1203c <__stack_top+0x1103c>
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u; // TDMA_UNPACK
    b820:	sw	a5,20(t6)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u; // L1
    b824:	sw	a5,56(t6)
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u; // TDMA_PACK
    b828:	sw	a5,248(t6)
            std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
    b82c:	lw	a7,12(sp)
            std::uint32_t shared_cycles = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
    b830:	lw	a5,256(t6)
            std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
    b834:	li	a4,860
    b838:	mul	a7,a7,a4
    b83c:	lui	a4,0x169
    b840:	addi	t3,a4,800 # 169320 <__runtime_args_end+0x148f20>
    b844:	add	a7,a7,t3
            bank_cycles[0]              = shared_cycles;
    b848:	sw	a5,0(a7) # b1448000 <__runtime_args_end+0xb1427c00>
            bank_cycles[1]              = shared_cycles;
    b84c:	sw	a5,4(a7)
            bank_cycles[2]              = shared_cycles;
    b850:	sw	a5,8(a7)
            bank_cycles[3]              = shared_cycles;
    b854:	sw	a5,12(a7)
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b858:	lui	t4,0xffb00
    b85c:	lui	t1,0x20
            bank_cycles[4]              = shared_cycles;
    b860:	sw	a5,16(a7)
            for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
    b864:	mv	t0,a4
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b868:	mv	t4,t4
    b86c:	addi	t1,t1,-256 # 1ff00 <__loader_init_end+0xff00>
                if (bank_id == 3u)
    b870:	li	t5,3
                std::uint32_t cw = cfg[i];
    b874:	lw	a5,0(a4)
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    b878:	sh2add	a3,a2,a7
    b87c:	addi	a3,a3,20
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b880:	and	a6,a5,t1
                if (!(cw & 0x80000000u))
    b884:	bgez	a5,b8c4 <run_kernel(RuntimeParams const&)+0x5ec>
                std::uint32_t bank_id    = cw & 0xFFu;
    b888:	zext.b	a0,a5
                ++out_idx;
    b88c:	addi	a2,a2,1
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b890:	sh3add	a1,a0,t4
                if (bank_id == 3u)
    b894:	bne	a0,t5,b8b0 <run_kernel(RuntimeParams const&)+0x5d8>
                    *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
    b898:	lw	a0,536(t6)
    b89c:	srli	a5,a5,0xd
    b8a0:	andi	a5,a5,112
    b8a4:	andi	a0,a0,-113
    b8a8:	or	a5,a5,a0
    b8ac:	sw	a5,536(t6)
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b8b0:	lw	a0,0(a1)
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    b8b4:	lw	a5,4(a1)
                *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
    b8b8:	sw	a6,0(a0)
                counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
    b8bc:	lw	a5,4(a5)
    b8c0:	sw	a5,0(a3)
            for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
    b8c4:	addi	a4,a4,4
    b8c8:	bne	a4,t3,b874 <run_kernel(RuntimeParams const&)+0x59c>
            std::uint32_t sync_addr                               = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE + PERF_COUNTERS_ZONE_DATA_BYTES;
    b8cc:	lw	a5,12(sp)
    b8d0:	li	a3,860
    b8d4:	mul	a5,a5,a3
            *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
    b8d8:	li	a4,255
            std::uint32_t sync_addr                               = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE + PERF_COUNTERS_ZONE_DATA_BYTES;
    b8dc:	add	a5,a5,t0
            *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
    b8e0:	sw	a4,1620(a5)
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
    b8e4:	lw	s0,44(sp)
    b8e8:	lw	s1,40(sp)
    b8ec:	lw	s2,36(sp)
    b8f0:	lw	s3,32(sp)
    b8f4:	lw	s4,28(sp)
    b8f8:	addi	sp,sp,48
    b8fc:	ret
        detail::zone_hashes[n] = hash_val;
    b900:	lui	a3,0x7c867
    b904:	sh2add	a4,a5,t2
    b908:	addi	a3,a3,-839 # 7c866cb9 <__runtime_args_end+0x7c8468b9>
        detail::next_zone_id   = n + 1;
    b90c:	addi	a2,a5,1
        detail::zone_hashes[n] = hash_val;
    b910:	sw	a3,4(a4)
        detail::next_zone_id   = n + 1;
    b914:	sw	a2,0(t2)
        return n;
    b918:	j	b390 <run_kernel(RuntimeParams const&)+0xb8>
        detail::zone_hashes[n] = hash_val;
    b91c:	lui	a4,0xbd77
    b920:	sh2add	a2,a5,t2
    b924:	addi	a4,a4,-244 # bd76f0c <__runtime_args_end+0xbd56b0c>
        detail::next_zone_id   = n + 1;
    b928:	addi	a3,a5,1
        detail::zone_hashes[n] = hash_val;
    b92c:	sw	a4,4(a2)
        detail::next_zone_id   = n + 1;
    b930:	sw	a3,0(t2)
        return n;
    b934:	j	b6dc <run_kernel(RuntimeParams const&)+0x404>
    for (std::uint32_t i = 0; i < n; ++i)
    b938:	mv	a5,a3
    b93c:	j	b6dc <run_kernel(RuntimeParams const&)+0x404>
    b940:	mv	a5,a3
    b944:	j	b390 <run_kernel(RuntimeParams const&)+0xb8>
    b948:	li	a5,4
    b94c:	j	b6dc <run_kernel(RuntimeParams const&)+0x404>
    b950:	li	a5,4
    b954:	j	b390 <run_kernel(RuntimeParams const&)+0xb8>

0000b958 <_init()>:
    }
}

void _init(void)
{
}
    b958:	ret

0000b95c <_fini()>:

void _fini(void)
    b95c:	ret

0000b960 <copy_runtimes_from_L1(RuntimeParams*)>:
        dstc[i] = srcc[i];
    b960:	lui	a5,0x20
    b964:	lbu	a5,0(a5) # 20000 <RUNTIME_ARGS_START>
    b968:	sb	a5,0(a0)
        (void)(dstc[i]);
    b96c:	lbu	a5,0(a0)
    asm volatile("fence" ::: "memory");
    b970:	fence
}
    b974:	ret

0000b978 <memset>:
    b978:	li	t1,15
    b97c:	mv	a4,a0
    b980:	bgeu	t1,a2,b9bc <memset+0x44>
    b984:	andi	a5,a4,15
    b988:	bnez	a5,ba28 <memset+0xb0>
    b98c:	bnez	a1,ba10 <memset+0x98>
    b990:	andi	a3,a2,-16
    b994:	andi	a2,a2,15
    b998:	add	a3,a3,a4
    b99c:	sw	a1,0(a4)
    b9a0:	sw	a1,4(a4)
    b9a4:	sw	a1,8(a4)
    b9a8:	sw	a1,12(a4)
    b9ac:	addi	a4,a4,16
    b9b0:	bltu	a4,a3,b99c <memset+0x24>
    b9b4:	bnez	a2,b9bc <memset+0x44>
    b9b8:	ret
    b9bc:	sub	a3,t1,a2
    b9c0:	slli	a3,a3,0x2
    b9c4:	auipc	t0,0x0
    b9c8:	add	a3,a3,t0
    b9cc:	jr	12(a3)
    b9d0:	sb	a1,14(a4)
    b9d4:	sb	a1,13(a4)
    b9d8:	sb	a1,12(a4)
    b9dc:	sb	a1,11(a4)
    b9e0:	sb	a1,10(a4)
    b9e4:	sb	a1,9(a4)
    b9e8:	sb	a1,8(a4)
    b9ec:	sb	a1,7(a4)
    b9f0:	sb	a1,6(a4)
    b9f4:	sb	a1,5(a4)
    b9f8:	sb	a1,4(a4)
    b9fc:	sb	a1,3(a4)
    ba00:	sb	a1,2(a4)
    ba04:	sb	a1,1(a4)
    ba08:	sb	a1,0(a4)
    ba0c:	ret
    ba10:	zext.b	a1,a1
    ba14:	slli	a3,a1,0x8
    ba18:	or	a1,a1,a3
    ba1c:	slli	a3,a1,0x10
    ba20:	or	a1,a1,a3
    ba24:	j	b990 <memset+0x18>
    ba28:	slli	a3,a5,0x2
    ba2c:	auipc	t0,0x0
    ba30:	add	a3,a3,t0
    ba34:	mv	t0,ra
    ba38:	jalr	-96(a3)
    ba3c:	mv	ra,t0
    ba40:	addi	a5,a5,-16
    ba44:	sub	a4,a4,a5
    ba48:	add	a2,a2,a5
    ba4c:	bgeu	t1,a2,b9bc <memset+0x44>
    ba50:	j	b98c <memset+0x14>
