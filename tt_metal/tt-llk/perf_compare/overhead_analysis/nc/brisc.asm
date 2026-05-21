
/tmp/perf_overhead_artifacts/nc/brisc.elf:     file format elf32-littleriscv


Disassembly of section .init:

00000000 <_start>:
// even though -fno-asynchronous-unwind-tables -fno-exceptions flags are set
void* __gxx_personality_v0;

__attribute__((no_profile_instrument_function)) TT_ALWAYS_INLINE void do_crt0()
{
    asm volatile(
   0:	auipc	gp,0xffb01
   4:	addi	gp,gp,-2048 # ffb00800 <__global_pointer$>
        "la gp, __global_pointer$\n"
        ".option pop" ::
            : "memory");

    // Set stack pointer
    asm volatile("la sp, %0" : : "i"(__stack_top) : "memory");
   8:	auipc	sp,0xffb02
   c:	addi	sp,sp,-8 # ffb02000 <__stack_top>

    // Initialize .bss
    for (volatile std::uint32_t* p = (volatile std::uint32_t*)__ldm_bss_start; p < (volatile std::uint32_t*)__ldm_bss_end; p++)
  10:	lui	a5,0xffb00
  14:	lui	a4,0xffb00
  18:	addi	a5,a5,36 # ffb00024 <brisc_bread1>
  1c:	addi	a4,a4,76 # ffb0004c <__gcov_info_end>
  20:	bgeu	a5,a4,44 <_start+0x44>
  24:	addi	a4,a4,-1
  28:	sub	a4,a4,a5
  2c:	andi	a4,a4,-4
  30:	addi	a4,a4,4
  34:	add	a4,a4,a5
    {
        *p = 0;
  38:	sw	zero,0(a5)
    for (volatile std::uint32_t* p = (volatile std::uint32_t*)__ldm_bss_start; p < (volatile std::uint32_t*)__ldm_bss_end; p++)
  3c:	addi	a5,a5,4
  40:	bne	a5,a4,38 <_start+0x38>
    }

    // Copy .loader_init to .ldm_data
    if ((std::uint32_t)__loader_init_start != (std::uint32_t)__loader_init_end)
  44:	lui	a5,0x4
  48:	lui	a4,0x6
  4c:	mv	a5,a5
  50:	mv	a4,a4
  54:	beq	a5,a4,94 <_start+0x94>
    {
        volatile std::uint32_t* src = (volatile std::uint32_t*)__loader_init_start;
        volatile std::uint32_t* dst = (volatile std::uint32_t*)__ldm_data_start;
        volatile std::uint32_t* end = (volatile std::uint32_t*)__ldm_data_end;
        while (dst < end)
  58:	lui	a4,0xffb00
  5c:	lui	a3,0xffb00
  60:	mv	a4,a4
  64:	addi	a3,a3,36 # ffb00024 <brisc_bread1>
  68:	bgeu	a4,a3,94 <_start+0x94>
  6c:	addi	a3,a3,-1
  70:	sub	a3,a3,a4
  74:	andi	a3,a3,-4
  78:	addi	a3,a3,4
  7c:	add	a3,a3,a5
        {
            *dst++ = *src++;
  80:	lw	a1,0(a5) # 4000 <__loader_init_start>
  84:	addi	a5,a5,4
  88:	sw	a1,0(a4) # ffb00000 <LOCAL_MEM_BASE>
  8c:	addi	a4,a4,4
        while (dst < end)
  90:	bne	a5,a3,80 <_start+0x80>
        }
    }

    // Execute global constructors
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
  94:	lui	s0,0xffb00
  98:	lui	s1,0xffb00
  9c:	mv	s0,s0
  a0:	addi	s1,s1,4 # ffb00004 <profiler_barrier>
  a4:	bgeu	s0,s1,b8 <_start+0xb8>
    {
        (*temp_constructor)();
  a8:	lw	a5,0(s0) # ffb00000 <LOCAL_MEM_BASE>
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
  ac:	addi	s0,s0,4
        (*temp_constructor)();
  b0:	jalr	a5
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
  b4:	bltu	s0,s1,a8 <_start+0xa8>

extern "C" __attribute__((section(".init"), naked, noreturn)) std::uint32_t _start()
{
    do_crt0();

    main();
  b8:	jal	bc <main>

Disassembly of section .text:

000000bc <main>:
{
  bc:	addi	sp,sp,-16
}

// Return pointer to CFG with the right base address for the current state
inline volatile std::uint32_t *tt_reg_ptr get_cfg_pointer()
{
    if (cfg_state_id == 0)
  c0:	lw	a5,-1976(gp) # ffb00048 <ckernel::cfg_state_id>
  c4:	sw	s0,12(sp)
  c8:	sw	s1,8(sp)
  cc:	sw	s2,4(sp)
  d0:	sw	s3,0(sp)
    {
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
  d4:	lui	a4,0xffef0
    if (cfg_state_id == 0)
  d8:	beqz	a5,e0 <main+0x24>
    }

    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
  dc:	addi	a4,a4,896 # ffef0380 <__stack_top+0x3ee380>
}

TT_ALWAYS_INLINE void disable_branch_prediction()
{
    volatile std::uint32_t* tt_reg_ptr cfg_ptr = ckernel::get_cfg_pointer();
    cfg_ptr[DISABLE_RISC_BP_Disable_main_ADDR32] |= DISABLE_RISC_BP_Disable_main_MASK;
  e0:	lw	a3,8(a4)
  e4:	lui	a2,0x400
    asm volatile(
  e8:	li	a5,0
  ec:	or	a3,a3,a2
  f0:	sw	a3,8(a4)
  f4:	lw	a3,-2000(gp) # ffb00030 <brisc_command_buffer>
  f8:	mv	a4,a5
  fc:	sw	a4,0(a3)
 100:	lw	a4,0(a3)
 104:	and	zero,zero,a4
    ckernel::store_blocking(brisc_command_buffer + 1, 0);
 108:	lw	a4,-2000(gp) # ffb00030 <brisc_command_buffer>
 10c:	mv	a3,a5
 110:	addi	a4,a4,4
 114:	sw	a3,0(a4)
 118:	lw	a3,0(a4)
 11c:	and	zero,zero,a3
 120:	mv	a4,a5
 124:	lw	a3,-2008(gp) # ffb00028 <brisc_bread0>
 128:	sw	a4,0(a3)
 12c:	lw	a4,0(a3)
 130:	and	zero,zero,a4
 134:	lw	a4,-2012(gp) # ffb00024 <brisc_bread1>
 138:	sw	a5,0(a4)
 13c:	lw	a5,0(a4)
 140:	and	zero,zero,a5
 144:	lui	a4,0xb001d
 148:	addi	a4,a4,-1282 # b001cafe <__runtime_args_end+0xafffc6fe>
 14c:	mv	a5,a4
    commit_store(brisc_counter, BRISC_BOOT_READY_SENTINEL);
 150:	lw	a3,-2004(gp) # ffb0002c <brisc_counter>
 154:	sw	a5,0(a3)
 158:	lw	a5,0(a3)
 15c:	and	zero,zero,a5
{
    ckernel::store_blocking(ptr, val);

    do
    {
        asm volatile("nop");
 160:	nop
    asm volatile(
 164:	lw	a5,0(a3)
 168:	and	zero,zero,a5
    } while (ckernel::load_blocking(ptr) != val);
 16c:	bne	a5,a4,160 <main+0xa4>
#endif

TT_ALWAYS_INLINE void clear_trisc_soft_reset()
{
    std::uint32_t soft_reset = ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0);
    soft_reset &= ~TRISC_SOFT_RESET_MASK;
 170:	lui	t4,0xffff9
 174:	addi	t4,t4,-1 # ffff8fff <__stack_top+0x4f6fff>
    std::uint32_t counter = 0;
 178:	li	a2,0
 17c:	li	s2,0
 180:	lui	a7,0xffb00
        switch (static_cast<BriscCommandState>(ckernel::load_blocking(brisc_command_buffer + (counter & 1))))
 184:	li	a6,2
// now handled by the compiler)
// workaround is needed only for GS
inline std::uint32_t reg_read(std::uint32_t addr)
{
    volatile std::uint32_t tt_reg_ptr *p_reg = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(addr);
    return p_reg[0];
 188:	lui	a5,0xffb12
}

TT_ALWAYS_INLINE void set_triscs_soft_reset()
{
    std::uint32_t soft_reset = ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0);
    soft_reset |= TRISC_SOFT_RESET_MASK;
 18c:	lui	t0,0x7
 190:	li	t1,1
 * On Blackhole this happens as a side effect of the FENCE instruction.
 */
inline void invalidate_data_cache()
{
    // clobber memory to prevent code reordering by the compiler.
    asm volatile("fence" ::: "memory");
 194:	fence
 198:	lw	a4,-2000(gp) # ffb00030 <brisc_command_buffer>
 19c:	sh2add	a4,s2,a4
    asm volatile(
 1a0:	lw	a4,0(a4)
 1a4:	and	zero,zero,a4
 1a8:	beq	a4,a6,1f4 <main+0x138>
 1ac:	andi	a4,a4,-3
 1b0:	beq	a4,t1,278 <main+0x1bc>
    std::uint64_t wall_clock_timestamp          = clock_lo[0] | (static_cast<std::uint64_t>(clock_hi[0]) << 32);
 1b4:	lw	a3,496(a5) # ffb121f0 <__stack_top+0x101f0>
 1b8:	lw	a4,504(a5)
    } while (wall_clock < (wall_clock_timestamp + cycles));
 1bc:	addi	s3,a3,1350
 1c0:	sltu	a3,s3,a3
 1c4:	add	a3,a3,a4
        wall_clock = clock_lo[0] | (static_cast<std::uint64_t>(clock_hi[0]) << 32);
 1c8:	lw	a1,496(a5)
 1cc:	lw	a4,504(a5)
    } while (wall_clock < (wall_clock_timestamp + cycles));
 1d0:	bltu	a4,a3,1c8 <main+0x10c>
 1d4:	bne	a3,a4,194 <main+0xd8>
 1d8:	bltu	a1,s3,1c8 <main+0x10c>
    asm volatile("fence" ::: "memory");
 1dc:	fence
 1e0:	lw	a4,-2000(gp) # ffb00030 <brisc_command_buffer>
 1e4:	sh2add	a4,s2,a4
    asm volatile(
 1e8:	lw	a4,0(a4)
 1ec:	and	zero,zero,a4
 1f0:	bne	a4,a6,1ac <main+0xf0>
    return p_reg[0];
 1f4:	lw	a3,432(a5)
 1f8:	or	a3,a3,t0
    p_reg[0]                                 = data;
 1fc:	sw	a3,432(a5)
    ckernel::reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset);
    do
    {
        asm volatile("nop");
 200:	nop
    return p_reg[0];
 204:	lw	a4,432(a5)
    } while (ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0) != soft_reset);
 208:	bne	a3,a4,200 <main+0x144>
    ckernel::store_blocking(brisc_command_buffer + (counter & 1), static_cast<std::uint32_t>(BriscCommandState::IDLE_STATE));
 20c:	lw	a3,-2000(gp) # ffb00030 <brisc_command_buffer>
    counter++;
 210:	addi	a2,a2,1 # 400001 <__runtime_args_end+0x3dfc01>
    ckernel::store_blocking(brisc_command_buffer + (counter & 1), static_cast<std::uint32_t>(BriscCommandState::IDLE_STATE));
 214:	andi	s2,a2,1
    asm volatile(
 218:	li	a4,0
 21c:	sh2add	a3,s2,a3
 220:	sw	a4,0(a3)
 224:	lw	a4,0(a3)
 228:	and	zero,zero,a4
    commit_store(brisc_counter, counter);
 22c:	lw	a3,-2004(gp) # ffb0002c <brisc_counter>
 230:	mv	a4,a2
 234:	sw	a4,0(a3)
 238:	lw	a4,0(a3)
 23c:	and	zero,zero,a4
        asm volatile("nop");
 240:	nop
    asm volatile(
 244:	lw	a4,0(a3)
 248:	and	zero,zero,a4
    } while (ckernel::load_blocking(ptr) != val);
 24c:	bne	a2,a4,240 <main+0x184>
                commit_store(brisc_bread1, counter);
 250:	lw	a3,-2012(gp) # ffb00024 <brisc_bread1>
    asm volatile(
 254:	mv	a4,a2
 258:	sw	a4,0(a3)
 25c:	lw	a4,0(a3)
 260:	and	zero,zero,a4
        asm volatile("nop");
 264:	nop
    asm volatile(
 268:	lw	a4,0(a3)
 26c:	and	zero,zero,a4
    } while (ckernel::load_blocking(ptr) != val);
 270:	bne	a4,a2,264 <main+0x1a8>
 274:	j	1b4 <main+0xf8>
    asm volatile(
 278:	li	a4,0
                commit_store(mailbox_math, ckernel::RESET_VAL);
 27c:	lw	a3,-1992(gp) # ffb00038 <mailbox_math>
 280:	sw	a4,0(a3)
 284:	lw	a4,0(a3)
 288:	and	zero,zero,a4
        asm volatile("nop");
 28c:	nop
    asm volatile(
 290:	lw	a4,0(a3)
 294:	and	zero,zero,a4
    } while (ckernel::load_blocking(ptr) != val);
 298:	bnez	a4,28c <main+0x1d0>
                commit_store(mailbox_unpack, ckernel::RESET_VAL);
 29c:	lw	a3,-1988(gp) # ffb0003c <mailbox_unpack>
    asm volatile(
 2a0:	sw	a4,0(a3)
 2a4:	lw	a4,0(a3)
 2a8:	and	zero,zero,a4
        asm volatile("nop");
 2ac:	nop
    asm volatile(
 2b0:	lw	a4,0(a3)
 2b4:	and	zero,zero,a4
    } while (ckernel::load_blocking(ptr) != val);
 2b8:	bnez	a4,2ac <main+0x1f0>
                commit_store(mailbox_pack, ckernel::RESET_VAL);
 2bc:	lw	a3,-1996(gp) # ffb00034 <mailbox_pack>
    asm volatile(
 2c0:	sw	a4,0(a3)
 2c4:	lw	a4,0(a3)
 2c8:	and	zero,zero,a4
        asm volatile("nop");
 2cc:	nop
    asm volatile(
 2d0:	lw	a4,0(a3)
 2d4:	and	zero,zero,a4
    } while (ckernel::load_blocking(ptr) != val);
 2d8:	bnez	a4,2cc <main+0x210>
                commit_store(profiler_barrier, 0U);
 2dc:	lw	a3,4(a7) # ffb00004 <profiler_barrier>
    asm volatile(
 2e0:	sw	a4,0(a3)
 2e4:	lw	a4,0(a3)
 2e8:	and	zero,zero,a4
        asm volatile("nop");
 2ec:	nop
    asm volatile(
 2f0:	lw	a4,0(a3)
 2f4:	and	zero,zero,a4
    } while (ckernel::load_blocking(ptr) != val);
 2f8:	bnez	a4,2ec <main+0x230>
                commit_store(profiler_barrier + 1, 0U);
 2fc:	lw	a3,4(a7)
 300:	addi	a3,a3,4
    asm volatile(
 304:	sw	a4,0(a3)
 308:	lw	a4,0(a3)
 30c:	and	zero,zero,a4
        asm volatile("nop");
 310:	nop
    asm volatile(
 314:	lw	a4,0(a3)
 318:	and	zero,zero,a4
    } while (ckernel::load_blocking(ptr) != val);
 31c:	bnez	a4,310 <main+0x254>
                commit_store(profiler_barrier + 2, 0U);
 320:	lw	a3,4(a7)
 324:	addi	a3,a3,8
    asm volatile(
 328:	sw	a4,0(a3)
 32c:	lw	a4,0(a3)
 330:	and	zero,zero,a4
        asm volatile("nop");
 334:	nop
    asm volatile(
 338:	lw	a4,0(a3)
 33c:	and	zero,zero,a4
    } while (ckernel::load_blocking(ptr) != val);
 340:	bnez	a4,334 <main+0x278>
    p_reg[0]                                 = data;
 344:	sw	zero,576(a5)
    TTI_ZEROACC(ckernel::p_zeroacc::CLR_ALL, 0, 0, 1, 0);
 348:	.insn	2, 0x0000
 34a:	.insn	2, 0x4061
    TTI_SFPENCC(3, 0, 0, 10);
 34c:	.insn	2, 0xc02a
 34e:	.insn	2, 0x2800
    TTI_NOP;
 350:	.insn	2, 0x0000
 352:	.insn	2, 0x0800
    TTI_SFPCONFIG(0, 11, 1); // loading -1 to LREG11 where sfpi expects it
 354:	.insn	2, 0x02c6
 356:	.insn	2, 0x4400
    TTI_SEMINIT(max_value, min_value, semaphore::t6_sem(index));
 358:	.insn	2, 0x0042
 35a:	.insn	2, 0x8c40
 35c:	.insn	2, 0x0802
 35e:	.insn	2, 0x8c40
 360:	.insn	2, 0x0102
 362:	.insn	2, 0x8c40
    return p_reg[0];
 364:	lw	a3,432(a5)
    soft_reset &= ~TRISC_SOFT_RESET_MASK;
 368:	and	a3,a3,t4
    p_reg[0]                                 = data;
 36c:	sw	a3,432(a5)
        asm volatile("nop");
 370:	nop
    return p_reg[0];
 374:	lw	a4,432(a5)
    } while (ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0) != soft_reset);
 378:	bne	a3,a4,370 <main+0x2b4>
    ckernel::store_blocking(brisc_command_buffer + (counter & 1), static_cast<std::uint32_t>(BriscCommandState::IDLE_STATE));
 37c:	lw	a3,-2000(gp) # ffb00030 <brisc_command_buffer>
    counter++;
 380:	addi	a2,a2,1
    ckernel::store_blocking(brisc_command_buffer + (counter & 1), static_cast<std::uint32_t>(BriscCommandState::IDLE_STATE));
 384:	andi	s2,a2,1
    asm volatile(
 388:	li	a4,0
 38c:	sh2add	a3,s2,a3
 390:	sw	a4,0(a3)
 394:	lw	a4,0(a3)
 398:	and	zero,zero,a4
    commit_store(brisc_counter, counter);
 39c:	lw	a3,-2004(gp) # ffb0002c <brisc_counter>
 3a0:	mv	a4,a2
 3a4:	sw	a4,0(a3)
 3a8:	lw	a4,0(a3)
 3ac:	and	zero,zero,a4
        asm volatile("nop");
 3b0:	nop
    asm volatile(
 3b4:	lw	a4,0(a3)
 3b8:	and	zero,zero,a4
    } while (ckernel::load_blocking(ptr) != val);
 3bc:	bne	a2,a4,3b0 <main+0x2f4>
                commit_store(brisc_bread0, counter);
 3c0:	lw	a3,-2008(gp) # ffb00028 <brisc_bread0>
    asm volatile(
 3c4:	mv	a4,a2
 3c8:	sw	a4,0(a3)
 3cc:	lw	a4,0(a3)
 3d0:	and	zero,zero,a4
        asm volatile("nop");
 3d4:	nop
    asm volatile(
 3d8:	lw	a4,0(a3)
 3dc:	and	zero,zero,a4
    } while (ckernel::load_blocking(ptr) != val);
 3e0:	bne	a4,a2,3d4 <main+0x318>
 3e4:	j	1b4 <main+0xf8>

000003e8 <_GLOBAL__sub_I__ZN7ckernel11pc_buf_baseE>:
mailbox_t mailbox_unpack = mailboxes_arr;
 3e8:	lw	a5,-2040(gp) # ffb00008 <mailboxes_arr>
 3ec:	sw	a5,-1988(gp) # ffb0003c <mailbox_unpack>
mailbox_t mailbox_math   = mailboxes_arr + 1;
 3f0:	addi	t5,a5,4
mailbox_t mailbox_pack   = mailboxes_arr + 2;
 3f4:	addi	t3,a5,8
mailbox_t brisc_command_buffer = mailboxes_arr + 3; // 2 entries
 3f8:	addi	a7,a5,12
mailbox_t brisc_counter        = mailboxes_arr + 5;
 3fc:	addi	a0,a5,20
mailbox_t brisc_bread0 = mailboxes_arr + 6;
 400:	addi	a2,a5,24
mailbox_t brisc_bread1 = mailboxes_arr + 7;
 404:	addi	a5,a5,28
mailbox_t mailbox_math   = mailboxes_arr + 1;
 408:	sw	t5,-1992(gp) # ffb00038 <mailbox_math>
mailbox_t mailbox_pack   = mailboxes_arr + 2;
 40c:	sw	t3,-1996(gp) # ffb00034 <mailbox_pack>
mailbox_t brisc_command_buffer = mailboxes_arr + 3; // 2 entries
 410:	sw	a7,-2000(gp) # ffb00030 <brisc_command_buffer>
mailbox_t brisc_counter        = mailboxes_arr + 5;
 414:	sw	a0,-2004(gp) # ffb0002c <brisc_counter>
mailbox_t brisc_bread0 = mailboxes_arr + 6;
 418:	sw	a2,-2008(gp) # ffb00028 <brisc_bread0>
mailbox_t brisc_bread1 = mailboxes_arr + 7;
 41c:	sw	a5,-2012(gp) # ffb00024 <brisc_bread1>

    for (;;)
    {
    } // Loop forever
}
 420:	ret

00000424 <_init()>:
}
 424:	ret

00000428 <_fini()>:
void _fini(void)
 428:	ret

0000042c <reset_state(unsigned long&)>:
    counter++;
 42c:	lw	a4,0(a0)
    ckernel::store_blocking(brisc_command_buffer + (counter & 1), static_cast<std::uint32_t>(BriscCommandState::IDLE_STATE));
 430:	lw	a2,-2000(gp) # ffb00030 <brisc_command_buffer>
    counter++;
 434:	addi	a4,a4,1
    ckernel::store_blocking(brisc_command_buffer + (counter & 1), static_cast<std::uint32_t>(BriscCommandState::IDLE_STATE));
 438:	andi	a3,a4,1
    asm volatile(
 43c:	li	a5,0
 440:	sh2add	a3,a3,a2
    counter++;
 444:	sw	a4,0(a0)
 448:	sw	a5,0(a3)
 44c:	lw	a5,0(a3)
 450:	and	zero,zero,a5
 454:	lw	a5,0(a0)
    commit_store(brisc_counter, counter);
 458:	lw	a3,-2004(gp) # ffb0002c <brisc_counter>
 45c:	sw	a5,0(a3)
 460:	lw	a5,0(a3)
 464:	and	zero,zero,a5
        asm volatile("nop");
 468:	nop
    asm volatile(
 46c:	lw	a4,0(a3)
 470:	and	zero,zero,a4
    } while (ckernel::load_blocking(ptr) != val);
 474:	lw	a5,0(a0)
 478:	bne	a4,a5,468 <reset_state(unsigned long&)+0x3c>
}
 47c:	ret
