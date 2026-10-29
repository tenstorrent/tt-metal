// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Data-movement kernel of test_noc_event_mesh_fill (tmp, issues/metal-profiler-noc-spill-reader): a stand-in for the
// profiler-relevant behaviour of a persistent blaze kernel. ITERS iterations of NUM_OPS "micro-ops", each a
// DeviceZoneScopedN with its own name holding NEST sub-zones; every sub-zone reads READS DRAM pages, then the op
// writes WRITES pages to its right neighbour, bumps that neighbour's semaphore and waits (bounded) on its own, and
// column-0 cores multicast a page to the rest of their row every MCAST_EVERY ops as a chain of MCAST_LINKED linked
// writes plus one unlinked. BRISC and NCRISC run the same source on their own NoC.
//
// Compile-time defines: NUM_OPS (<= 128), NEST, READS, WRITES, SEMS, MCAST_EVERY, MCAST_LINKED, DELAY_NOPS, ITERS,
// WAIT_SPINS. Compile args: TensorAccessorArgs of the DRAM page buffer at 0. Runtime args: see the host.
#include <cstdint>

struct Ctx {
    uint32_t page_size;
    uint32_t num_pages;
    uint32_t core_index;
    uint32_t has_right, right_x, right_y;
    uint32_t has_left;
    uint32_t sem_addr;
    uint32_t is_sender, mcast_num_dests;
    uint64_t mcast_noc_addr;
    uint32_t l1_src, l1_read_dst, l1_write_dst;
    uint32_t op_counter;
    uint32_t sem_expected;
};

// One shared, out-of-line body: inlined into 128 zone blocks it made the kernel 434 KB (job 162391), far over the
// kernel config buffer; the per-op code is now only the zone enter/exit and a call.
template <typename Accessor>
__attribute__((noinline)) void micro_op(Ctx& c, const Accessor& dram) {
    for (uint32_t k = 0; k < NEST; k++) {
        DeviceZoneScopedN("SUBOP");
        for (uint32_t r = 0; r < READS; r++) {
            const uint32_t page = (c.core_index * 131u + c.op_counter * 7u + k * 3u + r) % c.num_pages;
            noc_async_read_page(page, dram, c.l1_read_dst);
        }
        if constexpr (READS > 0) {
            noc_async_read_barrier();
        }
    }
    if constexpr (WRITES > 0) {
        if (c.has_right) {
            const uint64_t dst = get_noc_addr(c.right_x, c.right_y, c.l1_write_dst);
            for (uint32_t w = 0; w < WRITES; w++) {
                noc_async_write(c.l1_src, dst, c.page_size);
            }
            noc_async_write_barrier();
        }
    }
    if constexpr (SEMS > 0) {
        if (c.has_right) {
            noc_semaphore_inc(get_noc_addr(c.right_x, c.right_y, c.sem_addr), 1);
        }
        if (c.has_left) {
            c.sem_expected++;
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.sem_addr);
            // Bounded: a slow neighbour must never hang the galaxy (a hang poisons the chips for later phases).
            for (uint32_t spin = 0; spin < WAIT_SPINS && *sem < c.sem_expected; spin++) {
                asm volatile("nop");
            }
        }
    }
    if constexpr (MCAST_EVERY > 0) {
        if (c.is_sender && (c.op_counter % (MCAST_EVERY > 0 ? MCAST_EVERY : 1)) == 0) {  // no div-by-zero when compiled out
            for (uint32_t m = 0; m < MCAST_LINKED; m++) {
                noc_async_write_multicast(c.l1_src, c.mcast_noc_addr, c.page_size, c.mcast_num_dests, /*linked=*/true);
            }
            noc_async_write_multicast(c.l1_src, c.mcast_noc_addr, c.page_size, c.mcast_num_dests, /*linked=*/false);
            noc_async_write_barrier();
        }
    }
    for (uint32_t d = 0; d < DELAY_NOPS; d++) {
        asm volatile("nop");
    }
    c.op_counter++;
}

// One zone name per micro-op, like the generated blaze kernel (125 distinct names).
#define OP(n)                                      \
    if constexpr ((n) < NUM_OPS) {                 \
        DeviceZoneScopedN("MICROOP_" #n);          \
        micro_op(c, dram);                         \
    }

// Kernel-size knobs: the blaze raw dumps (job 162153) show the first bad quick push issued at ring index 0, i.e. with
// the RISC's profiler globals (firmware .bss) zeroed at a program boundary, on the cores running the largest kernels.
// BLOAT_BSS_WORDS grows this kernel's .bss (touched so it is kept), BLOAT_CODE adds unrolled text.
#if BLOAT_BSS_WORDS > 0
static uint32_t bloat_bss[BLOAT_BSS_WORDS];
#endif
#if BLOAT_DATA_WORDS > 0
static uint32_t bloat_data[BLOAT_DATA_WORDS] = {1};
#endif

void kernel_main() {
    Ctx c{};
#if BLOAT_BSS_WORDS > 0
    bloat_bss[get_arg_val<uint32_t>(3) % BLOAT_BSS_WORDS] = get_arg_val<uint32_t>(0);
    asm volatile("" ::: "memory");
#endif
#if BLOAT_DATA_WORDS > 0
    bloat_data[get_arg_val<uint32_t>(3) % BLOAT_DATA_WORDS] += get_arg_val<uint32_t>(1);
    asm volatile("" ::: "memory");
#endif
#if BLOAT_CODE > 0
#pragma GCC unroll 65534
    for (uint32_t b = 0; b < BLOAT_CODE; b++) {
        asm volatile("nop");
    }
#endif
    c.page_size = get_arg_val<uint32_t>(0);
    c.num_pages = get_arg_val<uint32_t>(1);
    const uint32_t dram_base = get_arg_val<uint32_t>(2);
    c.core_index = get_arg_val<uint32_t>(3);
    c.has_right = get_arg_val<uint32_t>(4);
    c.right_x = get_arg_val<uint32_t>(5);
    c.right_y = get_arg_val<uint32_t>(6);
    c.has_left = get_arg_val<uint32_t>(7);
    c.sem_addr = get_semaphore(get_arg_val<uint32_t>(8));
    c.is_sender = get_arg_val<uint32_t>(9);
    const uint32_t mcast_x_a = get_arg_val<uint32_t>(10);
    const uint32_t mcast_x_b = get_arg_val<uint32_t>(11);
    const uint32_t my_noc_y = get_arg_val<uint32_t>(12);
    c.mcast_num_dests = get_arg_val<uint32_t>(13);

    constexpr auto accessor_args = TensorAccessorArgs<0>();
    const auto dram = TensorAccessor(accessor_args, dram_base, c.page_size);

    // CB 0 holds 4 pages: 0 = source, 1 = DRAM read destination, 2 = neighbour write destination, 3 = multicast dst.
    const uint32_t l1_base = get_write_ptr(0);
    c.l1_src = l1_base;
    c.l1_read_dst = l1_base + c.page_size;
    c.l1_write_dst = l1_base + 2 * c.page_size;
    const uint32_t l1_mcast_dst = l1_base + 3 * c.page_size;
    // A multicast rectangle is given in the issuing NoC's traversal order: NOC0 start->end, NOC1 end->start.
    const uint32_t x_lo = mcast_x_a < mcast_x_b ? mcast_x_a : mcast_x_b;
    const uint32_t x_hi = mcast_x_a < mcast_x_b ? mcast_x_b : mcast_x_a;
    c.mcast_noc_addr = (noc_index == 0) ? get_noc_multicast_addr(x_lo, my_noc_y, x_hi, my_noc_y, l1_mcast_dst)
                                        : get_noc_multicast_addr(x_hi, my_noc_y, x_lo, my_noc_y, l1_mcast_dst);

    for (uint32_t it = 0; it < ITERS; it++) {
        DeviceZoneScopedN("ITERATION");
        OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7) OP(8) OP(9) OP(10) OP(11) OP(12) OP(13) OP(14) OP(15)
        OP(16) OP(17) OP(18) OP(19) OP(20) OP(21) OP(22) OP(23) OP(24) OP(25) OP(26) OP(27) OP(28) OP(29) OP(30) OP(31)
        OP(32) OP(33) OP(34) OP(35) OP(36) OP(37) OP(38) OP(39) OP(40) OP(41) OP(42) OP(43) OP(44) OP(45) OP(46) OP(47)
        OP(48) OP(49) OP(50) OP(51) OP(52) OP(53) OP(54) OP(55) OP(56) OP(57) OP(58) OP(59) OP(60) OP(61) OP(62) OP(63)
        OP(64) OP(65) OP(66) OP(67) OP(68) OP(69) OP(70) OP(71) OP(72) OP(73) OP(74) OP(75) OP(76) OP(77) OP(78) OP(79)
        OP(80) OP(81) OP(82) OP(83) OP(84) OP(85) OP(86) OP(87) OP(88) OP(89) OP(90) OP(91) OP(92) OP(93) OP(94) OP(95)
        OP(96) OP(97) OP(98) OP(99) OP(100) OP(101) OP(102) OP(103) OP(104) OP(105) OP(106) OP(107) OP(108) OP(109) OP(110) OP(111)
        OP(112) OP(113) OP(114) OP(115) OP(116) OP(117) OP(118) OP(119) OP(120) OP(121) OP(122) OP(123) OP(124) OP(125) OP(126) OP(127)
    }
}
