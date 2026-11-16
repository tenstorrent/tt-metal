// Weight stream of one data-movement RISC. The core's plan units (processing order) are cut into chunks of CH tiles;
// both RISCs claim chunks from a shared counter (semaphore CTR_SEM, zero at launch), so the RISC whose NoC delivers
// faster takes more chunks and both finish together. The non-atomic claim can rarely hand one chunk to both RISCs;
// that only costs a duplicate read. A RISC's k-th claimed chunk lands in slots [k * CH, ...) of its CB, its chunk id
// is recorded in word RISC * TABLE_STRIDE + k of CB TABLE_CB (read by compute) before the reads go out, and chunks are pushed
// in claim order as they land (transaction id (k % 7) + 1, at most WIN chunks in flight: a shallow DRAM queue lets
// the activation reads through early).
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "wqb_plan.h"

namespace wqb {

constexpr uint32_t MAX_UNITS = 64;

template <uint32_t KH, uint32_t NT>
inline uint32_t plan_pages(const Plan& plan, uint32_t* pages) {
    uint32_t n = 0;
    for (uint32_t si = 0; si < plan.n; ++si) {
        const auto& sg = plan.s[si];
        for (uint32_t kk = sg.kb; kk < sg.ke; ++kk) {
            pages[n++] = (plan.half * KH + kk) * NT + sg.col;
        }
    }
    return n;
}

template <uint32_t CH, uint32_t WIN, uint32_t CB, uint32_t TABLE_CB, uint32_t CTR_SEM, uint32_t RISC, uint32_t KH,
          uint32_t NT, typename Acc>
struct WStream {
    static constexpr uint32_t W_TILE = 1088;
    static_assert(WIN >= 1 && WIN <= 7);
    const Acc& acc;
    uint32_t pages[MAX_UNITS];
    uint32_t n = 0, nch = 0, issued = 0, done = 0, base = 0;
    bool exhausted = false;
    volatile tt_l1_ptr uint32_t* ctr;
    volatile tt_l1_ptr uint32_t* table;

    WStream(const Acc& a, const Plan& plan) : acc(a) {
        n = plan_pages<KH, NT>(plan, pages);
        nch = (n + CH - 1) / CH;
        cb_reserve_back(CB, n);
        base = get_write_ptr(CB);
        ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(CTR_SEM));
        table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(TABLE_CB)) + RISC * TABLE_STRIDE;
    }

    uint32_t chunk_tiles(uint32_t c) const { return (c + 1) * CH <= n ? CH : n - c * CH; }

    void issue() {
        while (!exhausted && issued < done + WIN) {
            const uint32_t c = *ctr;
            if (c >= nch) {
                exhausted = true;
                break;
            }
            *ctr = c + 1;
            table[issued] = c;
            noc_async_read_set_trid(issued % 7 + 1);
            const uint32_t dst = base + issued * CH * W_TILE;
            for (uint32_t i = 0; i < chunk_tiles(c); ++i) {
                noc_async_read(acc.get_noc_addr(pages[c * CH + i]), dst + i * W_TILE, W_TILE);
            }
            ++issued;
        }
        noc_async_read_set_trid(0);
    }

    bool pending() const { return !exhausted || done < issued; }

    void step() {
        issue();
        if (done < issued && ncrisc_noc_read_with_transaction_id_flushed(noc_index, done % 7 + 1)) {
            cb_push_back(CB, chunk_tiles(table[done]));
            ++done;
        }
    }
};

}  // namespace wqb
