// sdpa2 floor measurement: pure K gather of the sparse_sdpa decode access pattern (no compute). One core = (row, split
// g); it gathers keys [g*K/G, (g+1)*K/G) of its index row (MODE 0) or sequential pages (MODE 1) into one L1 buffer,
// REPS times. Compiled for both the reader (NCRISC) and writer (BRISC) data movers: NOCS=2 splits the keys across both,
// SEL picks the half.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "../sparse_split_kernels/ssplit_gather.hpp"

void kernel_main() {
    constexpr uint32_t TOPK = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(2);
    constexpr uint32_t REPS = get_compile_time_arg_val(3);
    constexpr uint32_t MODE = get_compile_time_arg_val(4);
    constexpr uint32_t SEL = get_compile_time_arg_val(5);  // 0: lower half, 1: upper half, 2: all
    constexpr uint32_t CB_DST = get_compile_time_arg_val(6);
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(7);
    constexpr uint32_t ROWB = 512;
    constexpr auto kv_args = TensorAccessorArgs<8, 2>();
    constexpr auto idx_args =
        TensorAccessorArgs<kv_args.next_compile_time_args_offset(), kv_args.next_common_runtime_args_offset()>();
    const uint32_t kv_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t row = get_arg_val<uint32_t>(0);
    const uint32_t g = get_arg_val<uint32_t>(1);
    Noc noc;
    experimental::CB dst(CB_DST), idxcb(CB_IDX);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    idxcb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ip = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idxcb.get_write_ptr());
    noc.async_read(idx, idxcb, TOPK * 4, {.page_id = row}, {.offset_bytes = 0});
    noc.async_read_barrier();
    constexpr uint32_t per = TOPK / G;  // keys of this core
    const uint32_t base = g * per;
    constexpr uint32_t lo = SEL == 1 ? per / 2 : 0;
    constexpr uint32_t hi = SEL == 0 ? per / 2 : per;
    dst.reserve_back(1);
    const uint32_t dst_l1 = dst.get_write_ptr();
    DeviceZoneScopedN("GB");
    for (uint32_t r = 0; r < REPS; ++r) {
        if constexpr (MODE == 0) {
            ssplit::trid_ring_gather<false, 0, 0, 0, 0, DEPTH>(noc, kv, dst_l1, ip, base, lo, hi, ROWB, 0);
        } else {
            const UnicastEndpoint local_l1;
            for (uint32_t i = lo; i < hi; ++i) {
                noc.async_read(kv, local_l1, ROWB, {.page_id = row * 640 + base + i}, {.addr = dst_l1 + i * ROWB});
            }
            noc.async_read_barrier();
        }
    }
}
