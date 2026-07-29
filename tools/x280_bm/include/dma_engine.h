/*
 * dma_engine.h - X280 DMA channel-0 helpers for the migration worker.
 *
 * Programming model: Synopsys DesignWare DMAC (the same controller
 * used by the L2CPU).
 *
 * MASTER MODEL
 *
 *   The DMAC's CTL register selects a SOURCE master (sms) and a
 *   DESTINATION master (dms). On Blackhole's L2CPU the two valid
 *   masters are:
 *
 *     EXTERN (= 0)  external bus port; reaches the NOC and the X280's
 *                   outbound DMA TLB. Used for OFF-tile traffic.
 *     L2     (= 1)  L2 cache port; reaches the X280's local memory
 *                   subsystem (LIM/L3 SRAM and X280 DDR/PMEM).
 *                   Used for ON-tile traffic.
 *
 *   The address that goes into SAR / DAR is interpreted by the chosen
 *   master:
 *
 *     EXTERN: SAR / DAR is a 32-bit DMA address. The upper 4 bits
 *             select an entry in the per-channel outbound DMA TLB
 *             table (DMA_TLB[0..15]); the TLB entry is a 20-bit value
 *             that contributes the upper 20 bits of a 48-bit X280
 *             physical address ( = {16'b0, dma_tlb[idx], sar[27:0]} ).
 *             For DDR-mapped traffic we use DMA_TLB[0] = 0x00000;
 *             for NOC-routed traffic we use DMA_TLB[8] = 0x00200
 *             (lands at X280 system port, where the configured CPU
 *              NOC TLB slot then routes the access off-tile).
 *
 *     L2:     SAR / DAR is the natural X280 physical address; the
 *             L2 cache port resolves it directly. So for LIM staging
 *             at e.g. 0x08194000, the DMA helper passes that address
 *             unchanged with master = L2. No DMA_TLB involved.
 *
 *   Migration worker traffic patterns and master selection:
 *     NOC -> LIM staging   (READ  payload  fetch)  EXTERN -> L2
 *     LIM staging -> NOC   (WRITE payload  push )  L2 -> EXTERN
 *     PCIe -> LIM staging  (h2d-data DEVICE_PULL)  EXTERN -> L2
 *     LIM staging -> PCIe  (d2h-data write)        L2 -> EXTERN
 *
 * NOC TLB CONTRACT
 *
 *   Every NOC-routed transfer needs the X280 hart to first program a
 *   2 MiB CPU NOC TLB so the chosen DMA-TLB[8] window has a target.
 *   Reference uses tlb_index 0 (config_noc_tlb_2M_sysport(0, ...))
 *   and reprograms it per call. We use X280_DMA_NOC_TLB_INDEX (= 0)
 *   for the same reason: on Blackhole the DMA NIU master only honors
 *   the single window the CPU programs immediately before the kick;
 *   higher TLB indices have shown empirical flakiness from this
 *   path. Fences after the configuration writes drain the peripheral
 *   port write pipeline so the DMA channel observes the new TLB
 *   state when its kick fires.
 *
 *   posted = 1, strict_order = 0 are the bench-validated flags. The
 *   posted bit lets writes complete asynchronously over the NOC
 *   (the DMA channel polls STATUSINT for done; ordering vs the
 *   post-DMA notify is enforced by a fence ow,ow at the call site).
 *
 * KICK SEQUENCE (software handshake mode)
 *
 *     1. Disable + re-enable the DMAC, clear all interrupt status.
 *     2. Initialize CTL    (int_en = 1, done bit pre-armed).
 *     3. Initialize CFG    (ch_prior = 1, hs_sel_src = hs_sel_dst = 1
 *                          for software handshaking, polarity = active
 *                          high, fcmode = 0, fifo_mode = 0, protctl = 0).
 *     4. Set transfer type = PERIPH -> PERIPH (forces software
 *        handshaking flow-control).
 *     5. Set address-change (INCR/INCR), burst length, word size,
 *        block_ts (== bytes / word_size).
 *     6. Disable LLP, scatter/gather, reload-addr.
 *     7. Set linked-list head address (set even though LLP is off;
 *        reference does this).
 *     8. Set sms = src master, dms = dst master, SAR, DAR.
 *     9. Unmask channel-0 interrupts, clear-interrupts.
 *    10. Enable channel: write CHENREG with ch_en_we = 1, ch_en = 1.
 *    11. Trigger software handshake: assert LSTSRC/LSTDST/REQSRC/
 *        REQDST/SGLRQSRC/SGLRQDST in the right order (burst-mode:
 *        REQ before SGLRQ).
 *    12. Poll STATUSINT until DONE (bit 0) or ERR (bit 4).
 *
 * BLOCK / WORD SIZE
 *
 *   Each "transfer" is `1 << src_tr_width` bytes wide; block_ts is
 *   12 bits so the maximum bytes per single block transfer is
 *   max_word_size * (4096 - 1) = 32 * 4095 = 131040 B. For arbitrary
 *   byte counts we descend through (32, 16, 8, 4, 2, 1)-byte word
 *   sizes, so the tail of a non-multiple-of-32 transfer still rides
 *   one program.
 *
 * THIS HEADER IS INLINE-ONLY. All helpers are static inline; there
 * is no companion .c file. Consumers include this from any firmware
 * source that needs the DMA channel (C or C++).
 */
#ifndef DMA_ENGINE_H
#define DMA_ENGINE_H

#include <stdint.h>

#include "noc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------
 * Address constants.
 *
 * All DMAC registers are reached through the L2CPU peripheral port
 * mapping at X280_DMA_PB (= 0x20000000) + the per-block offset from
 * the reference dma.hpp (DMAC_REG_MAP_BASE_ADDR = 0xFFFFF7FEFFF80000;
 * we drop the upper bits and OR with the peripheral port base).
 *
 * The local DMA TLB lives at peripheral-port offset 0x0FF10200
 * (mirrors L2CPU_OUTBOUND_DMA_TLB_0_REG_OFFSET = 0x200 from the
 * L2CPU register block). 16 32-bit entries; entry `i` is reachable
 * at X280_DMA_TLB + i * 4.
 * ------------------------------------------------------------------ */

#define X280_DMA_PB 0x20000000ULL
#define X280_DMA_CH0 (X280_DMA_PB + 0x0FF80000ULL)
#define X280_DMA_INT (X280_DMA_CH0 + 0x2C0ULL)
#define X280_DMA_SWHS (X280_DMA_CH0 + 0x368ULL)
#define X280_DMA_MISC (X280_DMA_CH0 + 0x398ULL)
#define X280_DMA_TLB (X280_DMA_PB + 0x0FF10200ULL)

/* CPU NOC TLB index every DMA-routed NOC access reprograms before
 * kicking the channel. See "NOC TLB CONTRACT" in the file header. */
#define X280_DMA_NOC_TLB_INDEX 0u

/* DMA-side base of the NOC system-port window (DMA address [0, 256 MiB)
 * mapped through DMA_TLB[8] = 0x200). For NOC traffic the SAR / DAR
 * is X280_DMA_NOC_BASE + idx * NOC_2M_WINDOW_STRIDE + window_off;
 * idx = X280_DMA_NOC_TLB_INDEX. */
#define X280_DMA_NOC_BASE 0x80000000ULL

/* Outbound DMA TLB slot we point at the NOC window. Together with
 * X280_DMA_NOC_BASE this gives DMA addresses in [0x80000000,
 * 0x90000000) -> X280 system port (256 MiB region). The slot index
 * is fixed by DMA_TLB layout: slot N covers DMA addresses
 * [N * 256 MiB, (N+1) * 256 MiB). */
#define X280_DMA_TLB_NOC_SLOT 8u

/* ------------------------------------------------------------------
 * NOC->NOC DMA TLB selector
 *
 * dma_engine_noc_to_noc has two implementations -- a 2 MiB CPU-NOC-TLB
 * path and a 128 GiB CPU-NOC-TLB path -- selected at firmware build
 * time by X280_NOC_TO_NOC_TLB. Both paths cover the migration-worker
 * use case correctly; the trade-off is per-kick byte ceiling and
 * outbound-DMA-TLB ownership:
 *
 *   X280_NOC_TO_NOC_TLB_2M (default)
 *     - 2 MiB CPU NOC TLB at indices SRC = 0, DST = 1.
 *     - Reuses the with-staging path's outbound DMA TLB slot 8
 *       (= 0x200, programmed once at init).
 *     - Per-kick max = 2 MiB. No per-call outbound-slot rewriting.
 *     - TLB.address (43 bits) carries the high bits of the NOC
 *       address; the low 21 bits arrive via the DMA SAR/DAR window
 *       offset. Full 64-bit NOC addressability with no straddle
 *       handling.
 *
 *   X280_NOC_TO_NOC_TLB_128G
 *     - 128 GiB CPU NOC TLB at indices SRC = 0, DST = 1.
 *     - Owns outbound DMA TLB slots 0 and 1 (rewritten per call).
 *     - Per-kick max = 256 MiB. The helper splits transfers that
 *       straddle a 256 MiB phys[27:0] boundary into multiple kicks.
 *     - TLB.address (27 bits) carries noc_addr >> 37; phys[36:28]
 *       is baked into the outbound DMA TLB slot value per call;
 *       phys[27:0] arrives via the DMA SAR/DAR.
 * ------------------------------------------------------------------ */
#define X280_NOC_TO_NOC_TLB_2M 0
#define X280_NOC_TO_NOC_TLB_128G 1

#ifndef X280_NOC_TO_NOC_TLB
#define X280_NOC_TO_NOC_TLB X280_NOC_TO_NOC_TLB_2M
#endif

#if X280_NOC_TO_NOC_TLB != X280_NOC_TO_NOC_TLB_2M && X280_NOC_TO_NOC_TLB != X280_NOC_TO_NOC_TLB_128G
#error "X280_NOC_TO_NOC_TLB must be X280_NOC_TO_NOC_TLB_2M or X280_NOC_TO_NOC_TLB_128G"
#endif

/* NOC->NOC DMA: per-call CPU NOC TLB indices (shared across both
 * 2 MiB and 128 GiB arms; the index space for 2 MiB and 128 GiB
 * blocks is independent per the noc.h cfg-base layout). */
#define X280_DMA_NOC_SRC_TLB_INDEX X280_DMA_NOC_TLB_INDEX /* CPU NOC TLB 0 */
#define X280_DMA_NOC_DST_TLB_INDEX 1u                     /* CPU NOC TLB 1 */

/* Outbound DMA TLB slots used by the 128 GiB arm. Rewritten per
 * call to bake phys[36:28] of each NOC offset into the slot value;
 * see dma_engine_noc_to_noc body. */
#if X280_NOC_TO_NOC_TLB == X280_NOC_TO_NOC_TLB_128G
#define X280_DMA_TLB_NOC_SRC_SLOT 0u
#define X280_DMA_TLB_NOC_DST_SLOT 1u
#endif

/* DMAC channel-0 timeout in cycles (matches the bench's 5e8). */
#define X280_DMA_TIMEOUT_CYCLES 500000000ULL

/* ------------------------------------------------------------------
 * Enums
 * ------------------------------------------------------------------ */

typedef enum { X280_DMA_DEV_MEM = 0, X280_DMA_DEV_PERIPH = 1 } x280_dma_device_t;

typedef enum { X280_DMA_INCR = 0, X280_DMA_DECR = 1, X280_DMA_CONST = 2 } x280_dma_addr_change_t;

typedef enum {
    X280_DMA_WORD_1 = 0,
    X280_DMA_WORD_2 = 1,
    X280_DMA_WORD_4 = 2,
    X280_DMA_WORD_8 = 3,
    X280_DMA_WORD_16 = 4,
    X280_DMA_WORD_32 = 5
} x280_dma_word_t;

typedef enum {
    X280_DMA_BURST_1 = 0,
    X280_DMA_BURST_4 = 1,
    X280_DMA_BURST_8 = 2,
    X280_DMA_BURST_16 = 3,
    X280_DMA_BURST_32 = 4,
    X280_DMA_BURST_64 = 5,
    X280_DMA_BURST_128 = 6,
    X280_DMA_BURST_256 = 7
} x280_dma_burst_t;

typedef enum {
    X280_DMA_MASTER_EXTERN = 0, /* off-tile; goes through DMA_TLB */
    X280_DMA_MASTER_L2 = 1      /* on-tile; L2 cache port (LIM/DDR) */
} x280_dma_master_t;

typedef enum { X280_DMA_DISABLE = 0, X280_DMA_ENABLE = 1 } x280_dma_enable_t;

/* ------------------------------------------------------------------
 * Bitfield definitions (lifted verbatim from dma.hpp).
 *
 * Used to compose CTL / CFG / LLP register values without manual
 * shift-and-OR (which is how earlier versions of this header silently
 * dropped master/transfer-type bits and routed traffic to DDR by
 * default). The unions allow a `.f.<field> = ...` write followed by
 * a single 64-bit MMIO store.
 * ------------------------------------------------------------------ */

typedef struct {
    uint64_t int_en : 1;       /* bit  0    */
    uint64_t dst_tr_width : 3; /* bits 1-3  */
    uint64_t src_tr_width : 3; /* bits 4-6  */
    uint64_t dinc : 2;         /* bits 7-8  */
    uint64_t sinc : 2;         /* bits 9-10 */
    uint64_t dest_msize : 3;   /* bits 11-13*/
    uint64_t src_msize : 3;    /* bits 14-16*/
    uint64_t src_gather_en : 1;
    uint64_t dst_scatter_en : 1;
    uint64_t rsvd_ctl0 : 1;
    uint64_t tt_fc : 3; /* bits 20-22*/
    uint64_t dms : 2;   /* bits 23-24*/
    uint64_t sms : 2;   /* bits 25-26*/
    uint64_t llp_dst_en : 1;
    uint64_t llp_src_en : 1;
    uint64_t rsvd_ctl1 : 3;
    uint64_t block_ts : 12; /* bits 32-43*/
    uint64_t done : 1;      /* bit 44    */
    uint64_t rsvd_ctl2 : 19;
} x280_dma_ctl0_bits_t;

typedef union {
    uint64_t val;
    x280_dma_ctl0_bits_t f;
} x280_dma_ctl0_t;

typedef struct {
    uint64_t rsvd_cfg0 : 5;
    uint64_t ch_prior : 3;
    uint64_t ch_susp : 1;
    uint64_t fifo_empty : 1;
    uint64_t hs_sel_dst : 1;
    uint64_t hs_sel_src : 1;
    uint64_t rsvd_lock_l : 4;
    uint64_t rsvd_lock : 2;
    uint64_t dst_hs_pol : 1;
    uint64_t src_hs_pol : 1;
    uint64_t rsvd_max_abrst : 10;
    uint64_t reload_src : 1;
    uint64_t reload_dst : 1;
    uint64_t fcmode : 1;
    uint64_t fifo_mode : 1;
    uint64_t protctl : 3;
    uint64_t rsvd_ds_upd_en : 1;
    uint64_t rsvd_ss_upd_en : 1;
    uint64_t src_per : 1;
    uint64_t rsvd_cfg1 : 3;
    uint64_t dest_per : 1;
    uint64_t rsvd_cfg2 : 3;
    uint64_t rsvd_cfg3 : 17;
} x280_dma_cfg0_bits_t;

typedef union {
    uint64_t val;
    x280_dma_cfg0_bits_t f;
} x280_dma_cfg0_t;

typedef struct {
    uint64_t lms : 2;
    uint64_t loc : 30;
    uint64_t rsvd_llp : 32;
} x280_dma_llp0_bits_t;

typedef union {
    uint64_t val;
    x280_dma_llp0_bits_t f;
} x280_dma_llp0_t;

/* ------------------------------------------------------------------
 * Low-level register helpers.
 * ------------------------------------------------------------------ */

static inline void x280_dma_w64_(uint64_t addr, uint64_t value) { *(volatile uint64_t*)(uintptr_t)addr = value; }

static inline uint64_t x280_dma_r64_(uint64_t addr) { return *(volatile uint64_t*)(uintptr_t)addr; }

static inline void x280_dma_w32_(uint64_t addr, uint32_t value) { *(volatile uint32_t*)(uintptr_t)addr = value; }

static inline void x280_dma_fence_(void) { __asm__ volatile("fence iorw, iorw"); }

static inline uint64_t x280_dma_rdcycle_(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

/* Channel-0 register offsets (relative to X280_DMA_CH0). */
#define X280_DMA_CH0_SAR0 0x00ULL
#define X280_DMA_CH0_DAR0 0x08ULL
#define X280_DMA_CH0_LLP0 0x10ULL
#define X280_DMA_CH0_CTL0 0x18ULL
#define X280_DMA_CH0_CFG0 0x40ULL

/* Misc register offsets (relative to X280_DMA_MISC). */
#define X280_DMA_MISC_DMACFG 0x00ULL
#define X280_DMA_MISC_CHEN 0x08ULL

/* Software-handshake register offsets (relative to X280_DMA_SWHS). */
#define X280_DMA_SWHS_REQSRC 0x00ULL
#define X280_DMA_SWHS_REQDST 0x08ULL
#define X280_DMA_SWHS_SGLRQSRC 0x10ULL
#define X280_DMA_SWHS_SGLRQDST 0x18ULL
#define X280_DMA_SWHS_LSTSRC 0x20ULL
#define X280_DMA_SWHS_LSTDST 0x28ULL

/* Interrupt-block register offsets (relative to X280_DMA_INT, which
 * itself starts at the RAW_TFR register). The status / clear /
 * STATUSINT regs follow the DesignWare layout from dma.hpp. */
#define X280_DMA_INT_RAWTFR 0x00ULL
#define X280_DMA_INT_RAWBLK 0x08ULL
#define X280_DMA_INT_RAWSRCT 0x10ULL
#define X280_DMA_INT_RAWDSTT 0x18ULL
#define X280_DMA_INT_RAWERR 0x20ULL
#define X280_DMA_INT_STATUSTFR 0x28ULL
#define X280_DMA_INT_STATUSBLK 0x30ULL
#define X280_DMA_INT_STATUSSRCT 0x38ULL
#define X280_DMA_INT_STATUSDSTT 0x40ULL
#define X280_DMA_INT_STATUSERR 0x48ULL
#define X280_DMA_INT_MASKTFR 0x50ULL
#define X280_DMA_INT_MASKBLK 0x58ULL
#define X280_DMA_INT_MASKSRCT 0x60ULL
#define X280_DMA_INT_MASKDSTT 0x68ULL
#define X280_DMA_INT_MASKERR 0x70ULL
#define X280_DMA_INT_CLEARTFR 0x78ULL
#define X280_DMA_INT_CLEARBLK 0x80ULL
#define X280_DMA_INT_CLEARSRCT 0x88ULL
#define X280_DMA_INT_CLEARDSTT 0x90ULL
#define X280_DMA_INT_CLEARERR 0x98ULL
#define X280_DMA_INT_STATUSINT 0xA0ULL

/* ------------------------------------------------------------------
 * Channel-0 read/modify/write helpers for CTL / CFG / LLP.
 * ------------------------------------------------------------------ */

static inline x280_dma_ctl0_t x280_dma_read_ctl0(void) {
    x280_dma_ctl0_t c;
    c.val = x280_dma_r64_(X280_DMA_CH0 + X280_DMA_CH0_CTL0);
    return c;
}

static inline void x280_dma_write_ctl0(x280_dma_ctl0_t c) { x280_dma_w64_(X280_DMA_CH0 + X280_DMA_CH0_CTL0, c.val); }

static inline x280_dma_cfg0_t x280_dma_read_cfg0(void) {
    x280_dma_cfg0_t c;
    c.val = x280_dma_r64_(X280_DMA_CH0 + X280_DMA_CH0_CFG0);
    return c;
}

static inline void x280_dma_write_cfg0(x280_dma_cfg0_t c) { x280_dma_w64_(X280_DMA_CH0 + X280_DMA_CH0_CFG0, c.val); }

/* ------------------------------------------------------------------
 * Per-field setters
 * ------------------------------------------------------------------ */

static inline void x280_dma_set_transfer_type(x280_dma_device_t src, x280_dma_device_t dst) {
    x280_dma_ctl0_t c = x280_dma_read_ctl0();
    if (src == X280_DMA_DEV_MEM && dst == X280_DMA_DEV_MEM) {
        c.f.tt_fc = 0x0;
    } else if (src == X280_DMA_DEV_MEM && dst == X280_DMA_DEV_PERIPH) {
        c.f.tt_fc = 0x1;
    } else if (src == X280_DMA_DEV_PERIPH && dst == X280_DMA_DEV_MEM) {
        c.f.tt_fc = 0x2;
    } else {
        c.f.tt_fc = 0x3;
    }
    x280_dma_write_ctl0(c);
}

static inline void x280_dma_set_addr_change(x280_dma_addr_change_t src, x280_dma_addr_change_t dst) {
    x280_dma_ctl0_t c = x280_dma_read_ctl0();
    c.f.sinc = (uint64_t)src;
    c.f.dinc = (uint64_t)dst;
    x280_dma_write_ctl0(c);
}

static inline void x280_dma_set_word_sizes(x280_dma_word_t src, x280_dma_word_t dst) {
    x280_dma_ctl0_t c = x280_dma_read_ctl0();
    c.f.src_tr_width = (uint64_t)src;
    c.f.dst_tr_width = (uint64_t)dst;
    x280_dma_write_ctl0(c);
}

static inline void x280_dma_set_burst_len(x280_dma_burst_t src, x280_dma_burst_t dst) {
    x280_dma_ctl0_t c = x280_dma_read_ctl0();
    c.f.src_msize = (uint64_t)src;
    c.f.dest_msize = (uint64_t)dst;
    x280_dma_write_ctl0(c);
}

static inline void x280_dma_set_block_ts(uint32_t block_ts) {
    x280_dma_ctl0_t c = x280_dma_read_ctl0();
    c.f.block_ts = (uint64_t)block_ts;
    x280_dma_write_ctl0(c);
}

static inline void x280_dma_set_masters(x280_dma_master_t src, x280_dma_master_t dst) {
    x280_dma_ctl0_t c = x280_dma_read_ctl0();
    c.f.sms = (uint64_t)src;
    c.f.dms = (uint64_t)dst;
    x280_dma_write_ctl0(c);
}

static inline void x280_dma_set_llp_en(x280_dma_enable_t src, x280_dma_enable_t dst) {
    x280_dma_ctl0_t c = x280_dma_read_ctl0();
    c.f.llp_src_en = (uint64_t)src;
    c.f.llp_dst_en = (uint64_t)dst;
    x280_dma_write_ctl0(c);
}

static inline void x280_dma_set_scatter_gather(x280_dma_enable_t scatter, x280_dma_enable_t gather) {
    x280_dma_ctl0_t c = x280_dma_read_ctl0();
    c.f.dst_scatter_en = (uint64_t)scatter;
    c.f.src_gather_en = (uint64_t)gather;
    x280_dma_write_ctl0(c);
}

static inline void x280_dma_set_reload_en(x280_dma_enable_t src, x280_dma_enable_t dst) {
    x280_dma_cfg0_t c = x280_dma_read_cfg0();
    c.f.reload_src = (uint64_t)src;
    c.f.reload_dst = (uint64_t)dst;
    x280_dma_write_cfg0(c);
}

static inline void x280_dma_set_linked_list_head(uint32_t head_addr, x280_dma_master_t master) {
    x280_dma_llp0_t l;
    l.val = 0;
    l.f.lms = (uint64_t)master;
    l.f.loc = (uint64_t)head_addr & 0x3FFFFFFFULL;
    x280_dma_w64_(X280_DMA_CH0 + X280_DMA_CH0_LLP0, l.val);
}

static inline void x280_dma_set_sar(uint64_t addr) { x280_dma_w64_(X280_DMA_CH0 + X280_DMA_CH0_SAR0, addr); }

static inline void x280_dma_set_dar(uint64_t addr) { x280_dma_w64_(X280_DMA_CH0 + X280_DMA_CH0_DAR0, addr); }

static inline void x280_dma_transfer_x_to_y(
    x280_dma_master_t src, uint64_t src_addr, x280_dma_master_t dst, uint64_t dst_addr) {
    x280_dma_set_masters(src, dst);
    x280_dma_set_sar(src_addr);
    x280_dma_set_dar(dst_addr);
}

/* ------------------------------------------------------------------
 * Interrupt + DMAC enable helpers.
 * ------------------------------------------------------------------ */

static inline void x280_dma_clear_interrupts(void) {
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARTFR, 1ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARBLK, 1ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARSRCT, 1ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARDSTT, 1ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARERR, 1ULL);
}

static inline void x280_dma_unmask_channel0(void) {
    /* mask reg layout: bit 0 = mask, bit 8 = write-enable. Setting both
     * = 0x0101 unmasks the interrupt source (= writes 1, with WE = 1). */
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_MASKTFR, 0x0101ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_MASKBLK, 0x0101ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_MASKSRCT, 0x0101ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_MASKDSTT, 0x0101ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_MASKERR, 0x0101ULL);
}

static inline void x280_dma_enable(void) { x280_dma_w64_(X280_DMA_MISC + X280_DMA_MISC_DMACFG, 1ULL); }

static inline void x280_dma_disable(void) { x280_dma_w64_(X280_DMA_MISC + X280_DMA_MISC_DMACFG, 0ULL); }

/* Initialize CTL0 with the bench's defaults: int_en = 1, done = 1
 * (the rest are filled by individual setters before the kick). */
static inline void x280_dma_init_ctl(void) {
    x280_dma_ctl0_t c;
    c.val = 0;
    c.f.int_en = 1;
    c.f.done = 1;
    x280_dma_write_ctl0(c);
}

/* Initialize CFG0 for software-handshake mode (the only mode we use):
 * channel-priority = 1, hs_sel_src/dst = 1 (= software handshake),
 * polarity = active-high, fcmode = 0, fifo_mode = 0, protctl = 0. */
static inline void x280_dma_init_cfg(void) {
    x280_dma_cfg0_t c;
    c.val = 0;
    c.f.ch_prior = 1;
    c.f.ch_susp = 0;
    c.f.hs_sel_src = 1;
    c.f.hs_sel_dst = 1;
    c.f.src_hs_pol = 0;
    c.f.dst_hs_pol = 0;
    c.f.fcmode = 0;
    c.f.fifo_mode = 0;
    c.f.protctl = 0;
    x280_dma_write_cfg0(c);
}

/* Reset DMA channel-0 between transfers (toggle DMAC_EN and clear all
 * raw + status interrupt registers). Mirrors test_noc_bw_matrix.c
 * dma_reset() exactly so a future trace cross-check still matches the
 * bench. */
static inline void x280_dma_reset_channel(void) {
    x280_dma_disable();
    x280_dma_fence_();
    for (volatile int d = 0; d < 100; d++) { /* small delay */
    }
    x280_dma_enable();

    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARTFR, 1ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARBLK, 1ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARSRCT, 1ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARDSTT, 1ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARERR, 1ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_MASKTFR, 0x0101ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_MASKBLK, 0x0101ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_MASKSRCT, 0x0101ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_MASKDSTT, 0x0101ULL);
    x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_MASKERR, 0x0101ULL);
    x280_dma_fence_();
}

/* ------------------------------------------------------------------
 * Kick the DMA channel via the software-handshake registers.
 *
 * Burst mode: write LSTSRC/LSTDST = 1 (with WE), then REQSRC/REQDST,
 * then SGLRQSRC/SGLRQDST. Order is REQ-before-SGL for burst transfers.
 * The DMAC then advances autonomously through block_ts transfers and
 * asserts STATUSINT bit 0 (DONE) on completion.
 *
 * Each handshake register has a bit-0 value and a bit-8 write-enable;
 * 0x0101 sets both = 1.
 * ------------------------------------------------------------------ */

static inline void x280_dma_start_burst(void) {
    x280_dma_w64_(X280_DMA_SWHS + X280_DMA_SWHS_LSTSRC, 0x0101ULL);
    x280_dma_w64_(X280_DMA_SWHS + X280_DMA_SWHS_LSTDST, 0x0101ULL);
    x280_dma_w64_(X280_DMA_SWHS + X280_DMA_SWHS_REQSRC, 0x0101ULL);
    x280_dma_w64_(X280_DMA_SWHS + X280_DMA_SWHS_REQDST, 0x0101ULL);
    x280_dma_w64_(X280_DMA_SWHS + X280_DMA_SWHS_SGLRQSRC, 0x0101ULL);
    x280_dma_w64_(X280_DMA_SWHS + X280_DMA_SWHS_SGLRQDST, 0x0101ULL);
}

/* Enable channel 0 in CHENREG (write-enable + ch_en bit). After this
 * write the channel is armed and the next handshake assert kicks the
 * transfer. */
static inline void x280_dma_channel_enable(void) {
    /* CHENREG layout: bit 0 = ch_en[0], bit 8 = ch_en_we[0]. */
    x280_dma_w64_(X280_DMA_MISC + X280_DMA_MISC_CHEN, 0x0101ULL);
}

/* Poll the DMAC interrupt block for the per-burst / per-block status
 * bits, re-kicking the software handshake whenever a burst (rather
 * than the whole block) finishes.
 *
 * In software-handshake mode (hs_sel_src = hs_sel_dst = 1, tt_fc =
 * PERIPH-PERIPH) the DMAC executes ONE burst per handshake assertion.
 * After each burst it raises IntDstTran (and IntSrcTran), parks the
 * channel, and waits for the firmware to assert the handshake again.
 * Only after the last burst of the block does it raise IntTfr (the
 * "block is fully done" event) and de-assert ch_en.
 *
 * STATUSINT is the OR of all five {Tfr, Block, SrcTran, DstTran, Err}
 * status registers, so a non-zero STATUSINT does NOT mean the block is
 * complete -- it can mean a single burst finished. We disambiguate by
 * reading the per-event raw status registers (RAWTFR / RAWERR / RAW
 * DstTran). If only DstTran fired, we clear it and re-issue the
 * handshake to release the next burst. If Tfr fires we return 0
 * (block done); if Err fires we return 1; on cycle-deadline timeout
 * we return 2.
 */
static inline int x280_dma_wait_done(uint64_t timeout_cycles) {
    uint64_t t0 = x280_dma_rdcycle_();
    for (;;) {
        uint64_t st = x280_dma_r64_(X280_DMA_INT + X280_DMA_INT_STATUSINT);
        if (st != 0ULL) {
            uint64_t err = x280_dma_r64_(X280_DMA_INT + X280_DMA_INT_RAWERR);
            if (err & 0x1ULL) {
                return 1;
            }
            uint64_t tfr = x280_dma_r64_(X280_DMA_INT + X280_DMA_INT_RAWTFR);
            if (tfr & 0x1ULL) {
                return 0;
            }
            uint64_t dsttran = x280_dma_r64_(X280_DMA_INT + X280_DMA_INT_RAWDSTT);
            uint64_t srctran = x280_dma_r64_(X280_DMA_INT + X280_DMA_INT_RAWSRCT);
            uint64_t blk = x280_dma_r64_(X280_DMA_INT + X280_DMA_INT_RAWBLK);
            if ((dsttran | srctran | blk) & 0x1ULL) {
                /* One burst done but the block isn't; clear the raw
                 * status bits and re-assert the handshake to release
                 * the next burst. The block_ts field decrements
                 * internally, so eventually IntTfr fires. */
                x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARDSTT, 1ULL);
                x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARSRCT, 1ULL);
                x280_dma_w64_(X280_DMA_INT + X280_DMA_INT_CLEARBLK, 1ULL);
                x280_dma_fence_();
                x280_dma_start_burst();
                x280_dma_fence_();
                continue;
            }
            /* STATUSINT was non-zero but no event we recognize fired:
             * fall through to the timeout check so we don't spin
             * forever on a phantom status bit. */
        }
        if (x280_dma_rdcycle_() - t0 > timeout_cycles) {
            return 2;
        }
    }
}

/* ------------------------------------------------------------------
 * High-level: one-time DMA channel + outbound TLB setup.
 *
 * Programs the outbound DMA TLB so that EXTERN-master accesses in
 * [2 GiB, 2 GiB + 256 MiB) land at the X280 system port for NOC
 * routing (TLB[8] = 0x00200). Enables the DMAC.
 *
 * Idempotent. Cheap (~3 register writes); the migration worker calls
 * it unconditionally even in SCALAR mode. */
static inline void dma_engine_init(void) {
    x280_dma_disable();
    x280_dma_fence_();
    x280_dma_w32_(X280_DMA_TLB + X280_DMA_TLB_NOC_SLOT * 4u, 0x00200u);
    x280_dma_fence_();
    x280_dma_enable();
    x280_dma_fence_();
}

/* Program X280 NOC TLB X280_DMA_NOC_TLB_INDEX for a DMA-routed access
 * to (noc_x, noc_y, addr). posted = 1, strict_order = 0 (the bench-
 * validated combination). Reprogrammed per call; callers must not
 * race the TLB across harts. The trailing config-readback drains the
 * peripheral-port write pipeline so the DMA NIU sees the new
 * programming before the next channel kick. */
static inline void dma_engine_program_noc_tlb(uint32_t noc_x, uint32_t noc_y, uint64_t addr) {
    (void)noc_configure_tlb_2m(
        X280_DMA_NOC_TLB_INDEX,
        noc_x,
        noc_y,
        addr,
        /*posted=*/1,
        /*strict_order=*/0);
    /* Read-back fence: drains the peripheral-port write pipeline
     * before we program SAR/DAR + kick the DMA channel. */
    volatile uint32_t* cfg =
        (volatile uint32_t*)(uintptr_t)(NOC_TLB_2M_CONFIG_BASE + (uint64_t)X280_DMA_NOC_TLB_INDEX * 0x10UL);
    (void)cfg[0];
    x280_dma_fence_();
}

/* Compute the DMA-side SAR/DAR for a NOC-routed access whose CPU NOC
 * TLB slot is X280_DMA_NOC_TLB_INDEX. Valid only after a matching
 * dma_engine_program_noc_tlb. */
static inline uint64_t dma_engine_noc_sar_dar(uint64_t noc_addr) {
    uint64_t window_off = noc_addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    return X280_DMA_NOC_BASE + (uint64_t)X280_DMA_NOC_TLB_INDEX * NOC_2M_WINDOW_STRIDE + window_off;
}

/* ------------------------------------------------------------------
 * do_dma -- low-level transfer driver.
 *
 * Programs CTL/CFG/SAR/DAR with the supplied masters and addresses,
 * descends through (32, 16, 8, 4, 2, 1)-byte word sizes to consume
 * `bytes` total, and kicks each block via the software handshake.
 *
 * The (src_master, dst_master) pair determines flow:
 *   EXTERN -> L2     NOC -> X280 LIM/DDR        (READ payload fetch)
 *   L2 -> EXTERN     X280 LIM/DDR -> NOC        (WRITE payload push)
 *   EXTERN -> EXTERN NOC -> NOC (fabric-style transfer)
 *   L2 -> L2         on-tile copy (LIM <-> DDR)
 *
 * `src_addr` / `dst_addr` are interpreted by their respective master
 * (see file header). Returns 0 on success, non-zero on DMA error or
 * timeout.
 * ------------------------------------------------------------------ */
static inline int x280_dma_do_dma(
    x280_dma_master_t src_master, uint64_t src_addr, x280_dma_master_t dst_master, uint64_t dst_addr, uint32_t bytes) {
    static const uint32_t MAX_BLOCK = 4095u;
    static const x280_dma_word_t word_size_enum[6] = {
        X280_DMA_WORD_32,
        X280_DMA_WORD_16,
        X280_DMA_WORD_8,
        X280_DMA_WORD_4,
        X280_DMA_WORD_2,
        X280_DMA_WORD_1,
    };
    static const uint32_t word_size_bytes[6] = {32, 16, 8, 4, 2, 1};

    x280_dma_reset_channel();

    /* CTL: int_en=1, done=1; remaining fields filled by setters. */
    x280_dma_init_ctl();
    x280_dma_set_transfer_type(X280_DMA_DEV_PERIPH, X280_DMA_DEV_PERIPH);
    x280_dma_set_addr_change(X280_DMA_INCR, X280_DMA_INCR);
    x280_dma_set_burst_len(X280_DMA_BURST_8, X280_DMA_BURST_8);
    x280_dma_set_llp_en(X280_DMA_DISABLE, X280_DMA_DISABLE);
    x280_dma_set_scatter_gather(X280_DMA_DISABLE, X280_DMA_DISABLE);

    /* CFG: ch_prior=1, hs_sel_*=1 (software handshaking). */
    x280_dma_init_cfg();
    x280_dma_set_reload_en(X280_DMA_DISABLE, X280_DMA_DISABLE);

    /* LLP head: bench writes a non-zero value even with LLP disabled.
     * We use the L2-master DDR base (0x00000000) so the linked-list
     * pointer is always parsable as a valid X280 phys address; the
     * DMAC won't follow it because llp_*_en = 0. */
    x280_dma_set_linked_list_head(0x00000000u, X280_DMA_MASTER_L2);

    x280_dma_unmask_channel0();
    x280_dma_clear_interrupts();

    /* Set masters + SAR + DAR ONCE for the whole multi-word descent.
     * The DMAC auto-increments SAR/DAR (sinc/dinc = INCR_ADDR) inside
     * each block AND across blocks, so the next set_block_ts kick
     * resumes from where the previous one left off. Manual SAR/DAR
     * bumps between iterations would double-count the offset and
     * scribble the destination at 2x stride. */
    x280_dma_transfer_x_to_y(src_master, src_addr, dst_master, dst_addr);
    x280_dma_fence_();

    uint32_t remaining = bytes;
    for (uint32_t w = 0; w < 6 && remaining > 0; w++) {
        uint32_t ws = word_size_bytes[w];
        while (remaining >= ws) {
            uint32_t blk = remaining / ws;
            if (blk > MAX_BLOCK) {
                blk = MAX_BLOCK;
            }
            remaining -= blk * ws;

            x280_dma_set_word_sizes(word_size_enum[w], word_size_enum[w]);
            x280_dma_set_block_ts(blk);
            x280_dma_clear_interrupts();
            x280_dma_fence_();

            x280_dma_channel_enable();
            x280_dma_fence_();
            x280_dma_start_burst();
            x280_dma_fence_();

            int rc = x280_dma_wait_done(X280_DMA_TIMEOUT_CYCLES);
            if (rc != 0) {
                return rc;
            }
            /* DMAC auto-incremented SAR/DAR; do NOT re-write them. */
        }
    }
    return 0;
}

/* ------------------------------------------------------------------
 * High-level NOC<->X280 transfer helpers.
 *
 * The `x280_master` argument lets callers steer the on-tile side at L2
 * (LIM/L3) or EXTERN (X280 DDR via outbound DMA TLB[0]):
 *
 *   - LIM staging buffer (typical migration worker target):  L2
 *   - X280 DDR-resident buffer (e.g. a debug scratchpad):    EXTERN
 *
 * Most migration worker call sites want L2 (the LIM staging buffer
 * sits at 0x08194000, which the L2 cache port handles directly).
 *
 * Returns 0 on success, non-zero on DMA error / timeout.
 * ------------------------------------------------------------------ */

static inline int dma_engine_noc_to_x280(
    uint32_t noc_x,
    uint32_t noc_y,
    uint64_t noc_addr,
    x280_dma_master_t x280_master,
    uint64_t x280_addr,
    uint32_t bytes) {
    dma_engine_program_noc_tlb(noc_x, noc_y, noc_addr);
    uint64_t sar = dma_engine_noc_sar_dar(noc_addr);
    uint64_t dar = x280_addr;
    return x280_dma_do_dma(X280_DMA_MASTER_EXTERN, sar, x280_master, dar, bytes);
}

static inline int dma_engine_x280_to_noc(
    x280_dma_master_t x280_master,
    uint64_t x280_addr,
    uint32_t noc_x,
    uint32_t noc_y,
    uint64_t noc_addr,
    uint32_t bytes) {
    dma_engine_program_noc_tlb(noc_x, noc_y, noc_addr);
    uint64_t sar = x280_addr;
    uint64_t dar = dma_engine_noc_sar_dar(noc_addr);
    return x280_dma_do_dma(x280_master, sar, X280_DMA_MASTER_EXTERN, dar, bytes);
}

/* ------------------------------------------------------------------
 * NOC->NOC DMA (no LIM staging).
 *
 * Two implementations selected at firmware build time by
 * X280_NOC_TO_NOC_TLB:
 *
 *   X280_NOC_TO_NOC_TLB_2M (default)
 *     Programs two 2 MiB CPU NOC TLBs (SRC at index 0, DST at index
 *     1). Both DMA addresses share the single outbound DMA TLB slot
 *     X280_DMA_TLB_NOC_SLOT (= 8, value 0x200) that
 *     dma_engine_init() programs once at boot, since SLOT * 256 MiB
 *     = 2 GiB and 2 MiB window indices 0 and 1 sit at offsets 0 and
 *     2 MiB inside that slot's 256 MiB DMA-address window. The
 *     2 MiB TLB.address (43 bits) carries the high bits of the NOC
 *     address; the low 21 bits arrive via the DMA SAR/DAR window
 *     offset. Per-call max transfer is the 2 MiB window stride;
 *     migration-worker data_page_size is well below this, so no
 *     chunking is needed.
 *
 *   X280_NOC_TO_NOC_TLB_128G
 *     Programs two 128 GiB CPU NOC TLBs (SRC at index 0, DST at
 *     index 1). The 128 GiB TLB.address field is 27 bits wide --
 *     it captures only `noc_addr >> 37`. The remaining 37 bits of
 *     the NOC offset must be split:
 *         phys[36:28] (9 bits)   -> baked into the outbound DMA
 *                                   TLB slot value (rewritten per
 *                                   call into slots SRC=0, DST=1).
 *         phys[27:0]  (28 bits)  -> the DMA SAR/DAR.
 *     This requires this helper to own outbound DMA TLB slots 0 and
 *     1 (dma_engine_init no longer touches slot 0). Per-call max
 *     transfer is the outbound DMA TLB slot width (256 MiB =
 *     1 << 28). A transfer that would straddle a 256 MiB phys[27:0]
 *     boundary on either endpoint is split into multiple kicks,
 *     each reprogramming the TLB and slot value before kicking the
 *     channel.
 *
 * Used by the migration worker's X280_XFER_MODE_DMA_NO_STAGING
 * path: READ moves DRAM -> pinned host D2H FIFO directly; WRITE
 * moves pinned host H2D-data FIFO -> DRAM directly. No intermediate
 * LIM hop.
 *
 * Returns 0 on success, non-zero on DMA error / timeout.
 * ------------------------------------------------------------------ */

#if X280_NOC_TO_NOC_TLB == X280_NOC_TO_NOC_TLB_2M

static inline int dma_engine_noc_to_noc(
    uint32_t src_noc_x,
    uint32_t src_noc_y,
    uint64_t src_noc_addr,
    uint32_t dst_noc_x,
    uint32_t dst_noc_y,
    uint64_t dst_noc_addr,
    uint32_t bytes) {
    (void)noc_configure_tlb_2m(
        X280_DMA_NOC_SRC_TLB_INDEX,
        src_noc_x,
        src_noc_y,
        src_noc_addr,
        /*posted=*/1,
        /*strict_order=*/0);
    (void)noc_configure_tlb_2m(
        X280_DMA_NOC_DST_TLB_INDEX,
        dst_noc_x,
        dst_noc_y,
        dst_noc_addr,
        /*posted=*/1,
        /*strict_order=*/0);

    /* Read-back fence on both TLB cfg blocks: drains the peripheral-
     * port write pipeline before the next channel kick. */
    volatile uint32_t* src_cfg =
        (volatile uint32_t*)(uintptr_t)(NOC_TLB_2M_CONFIG_BASE + (uint64_t)X280_DMA_NOC_SRC_TLB_INDEX * 0x10UL);
    volatile uint32_t* dst_cfg =
        (volatile uint32_t*)(uintptr_t)(NOC_TLB_2M_CONFIG_BASE + (uint64_t)X280_DMA_NOC_DST_TLB_INDEX * 0x10UL);
    (void)src_cfg[0];
    (void)dst_cfg[0];
    x280_dma_fence_();

    /* DMA-side SAR/DAR: same formula as dma_engine_noc_sar_dar, one
     * for each end. Both fall inside the single outbound DMA TLB
     * slot X280_DMA_TLB_NOC_SLOT (= 8, value 0x200, programmed by
     * dma_engine_init), since SLOT * 256 MiB = 2 GiB and 2 MiB
     * window indices 0 and 1 sit at offsets 0 and 2 MiB inside that
     * slot's 256 MiB DMA-address window. */
    uint64_t src_window_off = src_noc_addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t dst_window_off = dst_noc_addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t src_dma_addr =
        X280_DMA_NOC_BASE + (uint64_t)X280_DMA_NOC_SRC_TLB_INDEX * NOC_2M_WINDOW_STRIDE + src_window_off;
    uint64_t dst_dma_addr =
        X280_DMA_NOC_BASE + (uint64_t)X280_DMA_NOC_DST_TLB_INDEX * NOC_2M_WINDOW_STRIDE + dst_window_off;
    return x280_dma_do_dma(X280_DMA_MASTER_EXTERN, src_dma_addr, X280_DMA_MASTER_EXTERN, dst_dma_addr, bytes);
}

#else /* X280_NOC_TO_NOC_TLB == X280_NOC_TO_NOC_TLB_128G */

/* Internal helper: program one CPU NOC TLB + one outbound DMA TLB
 * slot for a single 128 GiB-arm sub-kick, and return the DMA
 * SAR/DAR to feed x280_dma_do_dma.
 *
 *   tlb_index    CPU NOC TLB index in the 128 GiB block
 *                (X280_DMA_NOC_SRC_TLB_INDEX or _DST_TLB_INDEX).
 *   dma_slot     Outbound DMA TLB slot to rewrite per call
 *                (X280_DMA_TLB_NOC_SRC_SLOT or _DST_SLOT).
 *
 * The DMA SAR/DAR returned is `(dma_slot << 28) | phys[27:0]`,
 * where phys[27:0] is the low 28 bits of `noc_addr & ((1<<37)-1)`.
 * phys[36:28] is OR'd into the outbound DMA TLB slot value so the
 * full 37-bit offset is reconstructed inside the X280's address
 * fabric. */
static inline uint64_t dma_engine_program_128g_endpoint_(
    uint32_t tlb_index, uint32_t dma_slot, uint32_t noc_x, uint32_t noc_y, uint64_t noc_addr) {
    uint64_t window_base = noc_configure_tlb_128g(
        tlb_index,
        noc_x,
        noc_y,
        noc_addr,
        /*posted=*/1,
        /*strict_order=*/0);

    /* Outbound DMA TLB slot value: low 12 bits of the 20-bit
     * field. Bits [11:9] capture peripheral selection
     * (0x200 = system port for slot 8, encoded here as
     * (window_base - SYSTEM_PORT) >> 28); bits [8:0] capture
     * phys[36:28] of the NOC offset. */
    const uint64_t k_system_port = 0x30000000ULL;
    uint64_t window_local = window_base - k_system_port;
    uint64_t local_off = noc_addr & ((1ULL << 37) - 1ULL);
    uint64_t local_off_hi9 = local_off & ~((1ULL << 28) - 1ULL);
    uint64_t slot_val = (window_local + local_off_hi9) >> 28;

    x280_dma_fence_();
    x280_dma_w32_(X280_DMA_TLB + dma_slot * 4u, (uint32_t)slot_val);
    x280_dma_fence_();

    /* DMA SAR/DAR: phys[27:0] of the offset, anchored at the
     * outbound DMA TLB slot's DMA-address base ((slot << 28)). */
    uint64_t local_off_lo28 = local_off & ((1ULL << 28) - 1ULL);
    return ((uint64_t)dma_slot << 28) | local_off_lo28;
}

static inline int dma_engine_noc_to_noc(
    uint32_t src_noc_x,
    uint32_t src_noc_y,
    uint64_t src_noc_addr,
    uint32_t dst_noc_x,
    uint32_t dst_noc_y,
    uint64_t dst_noc_addr,
    uint32_t bytes) {
    /* Per-kick max transfer is the outbound DMA TLB slot width
     * (256 MiB). Compute the largest contiguous chunk that fits in
     * the *current* phys[27:0] window for both endpoints; loop if
     * the transfer straddles a 256 MiB boundary on either side. */
    uint32_t remaining = bytes;
    while (remaining > 0u) {
        const uint64_t k_window = 1ULL << 28; /* 256 MiB */
        uint64_t src_off = src_noc_addr & (k_window - 1ULL);
        uint64_t dst_off = dst_noc_addr & (k_window - 1ULL);
        uint64_t src_room = k_window - src_off;
        uint64_t dst_room = k_window - dst_off;
        uint64_t room = src_room < dst_room ? src_room : dst_room;

        uint32_t chunk = remaining;
        if ((uint64_t)chunk > room) {
            chunk = (uint32_t)room;
        }

        uint64_t sar = dma_engine_program_128g_endpoint_(
            X280_DMA_NOC_SRC_TLB_INDEX, X280_DMA_TLB_NOC_SRC_SLOT, src_noc_x, src_noc_y, src_noc_addr);
        uint64_t dar = dma_engine_program_128g_endpoint_(
            X280_DMA_NOC_DST_TLB_INDEX, X280_DMA_TLB_NOC_DST_SLOT, dst_noc_x, dst_noc_y, dst_noc_addr);

        /* Read-back fence on both 128 GiB TLB cfg blocks: drains the
         * peripheral-port write pipeline before the next channel
         * kick. */
        volatile uint32_t* src_cfg =
            (volatile uint32_t*)(uintptr_t)(NOC_128G_CONFIG_BASE + (uint64_t)X280_DMA_NOC_SRC_TLB_INDEX * 0xCUL);
        volatile uint32_t* dst_cfg =
            (volatile uint32_t*)(uintptr_t)(NOC_128G_CONFIG_BASE + (uint64_t)X280_DMA_NOC_DST_TLB_INDEX * 0xCUL);
        (void)src_cfg[0];
        (void)dst_cfg[0];
        x280_dma_fence_();

        int rc = x280_dma_do_dma(X280_DMA_MASTER_EXTERN, sar, X280_DMA_MASTER_EXTERN, dar, chunk);
        if (rc != 0) {
            return rc;
        }

        src_noc_addr += chunk;
        dst_noc_addr += chunk;
        remaining -= chunk;
    }
    return 0;
}

#endif /* X280_NOC_TO_NOC_TLB */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DMA_ENGINE_H */
