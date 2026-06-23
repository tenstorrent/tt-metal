/*
 * noc.h - X280 NOC 2 MiB TLB-window programming + read helpers.
 *
 * Trimmed/vendored from tenstorrent/tt-llm-engine x280/include/noc.h. The X280
 * reaches another tile's address space (e.g. a Tensix core's L1) by programming
 * a 2 MiB TLB window (config regs on the Peripheral Port) then doing uncached
 * loads through the System Port data window. Each load stalls the in-order core
 * until the NOC read response returns — that latency is what the poller measures.
 */
#ifndef X280_NOC_H
#define X280_NOC_H

#include <stdint.h>

/* Peripheral Port: TLB config registers (224 entries x 16 bytes). */
#define NOC_TLB_2M_CONFIG_BASE 0x2FF00000UL
/* System Port: 2 MiB window data base (uncached, no atomics per PMA). */
#define NOC_2M_WINDOW_BASE 0x430000000ULL
#define NOC_2M_WINDOW_STRIDE 0x200000ULL /* 2 MiB per window */

/* 128-bit TLB descriptor (written as 4 x u32). For unicast,
 * x_start=x_end=noc_x and y_start=y_end=noc_y; addr holds target>>21 and the
 * low 21 bits of the address become the window offset. */
typedef union {
    struct {
        uint64_t addr : 43; /* target NOC address >> 21 */
        uint64_t reserved0 : 21;
        uint64_t x_end : 6;
        uint64_t y_end : 6;
        uint64_t x_start : 6;
        uint64_t y_start : 6;
        uint64_t multicast_en : 1;
        uint64_t strict_order : 1;
        uint64_t posted : 1;
        uint64_t linked : 1;
        uint64_t static_en : 1;
        uint64_t stream_header : 1;
        uint64_t reserved1 : 1;
        uint64_t noc_selector : 1; /* 0=NOC0, 1=NOC1 */
        uint64_t static_vc : 3;
        uint64_t strided : 8;
        uint64_t exclude_coord_x : 5;
        uint64_t exclude_coord_y : 4;
        uint64_t exclude_dir_x : 1;
        uint64_t exclude_dir_y : 1;
        uint64_t exclude_enable : 1;
        uint64_t exclude_routing_option : 1;
        uint64_t num_destinations : 8;
    };
    uint32_t data[4];
} noc_tlb_2m_t;

/* Configure 2 MiB TLB window `index` to (noc_x, noc_y, addr); return the
 * uncached System-Port pointer to use for loads (offset = addr & 0x1FFFFF). */
static inline volatile void* noc_configure_tlb_2m(
    uint32_t index, uint32_t noc_x, uint32_t noc_y, uint64_t addr, int posted, int strict_order) {
    volatile uint32_t* cfg = (volatile uint32_t*)(NOC_TLB_2M_CONFIG_BASE + (uint64_t)index * 0x10UL);

    /* Zero the 4 descriptor words explicitly (an aggregate {0} init would
     * emit a memset call, which we don't have under -nostdlib). */
    noc_tlb_2m_t tlb;
    tlb.data[0] = 0;
    tlb.data[1] = 0;
    tlb.data[2] = 0;
    tlb.data[3] = 0;
    tlb.addr = addr >> 21;
    tlb.x_end = noc_x;
    tlb.y_end = noc_y;
    tlb.x_start = noc_x;
    tlb.y_start = noc_y;
    tlb.posted = posted ? 1u : 0u;
    tlb.strict_order = strict_order ? 1u : 0u;

    __asm__ volatile("fence iorw, iorw");
    cfg[0] = tlb.data[0];
    cfg[1] = tlb.data[1];
    cfg[2] = tlb.data[2];
    cfg[3] = tlb.data[3];
    __asm__ volatile("fence iorw, iorw");

    uint64_t offset = addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    return (volatile void*)(NOC_2M_WINDOW_BASE + (uint64_t)index * NOC_2M_WINDOW_STRIDE + offset);
}

/* Uncached reads through a window pointer. Each load stalls until the NOC
 * response arrives (the load itself is the synchronization). */
static inline uint32_t noc_read_u32(volatile void* window_ptr) { return *(volatile uint32_t*)window_ptr; }

static inline uint64_t noc_read_u64(volatile void* window_ptr) {
    volatile uint32_t* lo = (volatile uint32_t*)window_ptr;
    return ((uint64_t)lo[1] << 32) | lo[0];
}

#endif /* X280_NOC_H */
