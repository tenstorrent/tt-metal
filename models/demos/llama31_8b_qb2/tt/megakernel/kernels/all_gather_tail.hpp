// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Inside the native writer, only after the final layer's second reduction.
if (get_arg_val<uint32_t>(TAIL_RT_OFFSET + 20) && (invocation & 1u))
{
    DeviceZoneScopedN("DECODER-FINAL-GATHER");
    constexpr auto ag_args = TensorAccessorArgs<AG_CT_OFFSET>();
    const auto gathered = TensorAccessor(ag_args, get_arg_val<uint32_t>(TAIL_RT_OFFSET), 2048);
    const uint32_t ready = get_arg_val<uint32_t>(TAIL_RT_OFFSET + 1);
    const uint32_t coordinator_x = get_arg_val<uint32_t>(TAIL_RT_OFFSET + 2);
    const uint32_t coordinator_y = get_arg_val<uint32_t>(TAIL_RT_OFFSET + 3);
    const uint64_t ready_remote = safe_get_noc_addr(coordinator_x, coordinator_y, ready, 0);
    const uint32_t first = device_idx * 32 + tile_start;
    // The two output chunks fill and wrap this sixteen-tile CB exactly once.
    // The producer has finished; borrow its storage after the native pops.
    noc.async_write_barrier();
    const uint32_t residual = get_read_ptr(16);
    for (uint32_t dst = 0; dst < 3; ++dst) {
        const uint8_t hops = static_cast<uint8_t>(get_arg_val<uint32_t>(dest_hops + dst));
        auto* sender = &fabric_connection.get(get_arg_val<uint32_t>(dest_conn + dst)).sender;
        fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(data_pkt, hops);
        fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
            UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush>(
            fused_pkt, hops, tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0u, 0u, 1u}, 0);
        set_route_2d(data_pkt, dst);
        set_route_2d(fused_pkt, dst);
        for (uint32_t t = 0; t < 16; t += 4) {
            const uint64_t dest = tt::tt_fabric::addrgen_detail::get_noc_address(gathered, first + t, 0);
            if (t == 12) {
                fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                    UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
                    UnicastFusedAtomicIncUpdateMask::PayloadSize>(
                    sender, fused_pkt, residual + t * 2048,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dest, ready_remote, 1u}, 8192);
            } else {
                fabric_api::fabric_unicast_noc_unicast_write_with_state<
                    UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                    sender, data_pkt, residual + t * 2048, tt::tt_fabric::NocUnicastCommandHeader{dest}, 8192);
            }
            noc.async_writes_flushed();
        }
    }
    for (uint32_t t = 0; t < 16; ++t) {
        noc_async_write_page(first + t, gathered, residual + t * 2048);
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(coordinator_x, coordinator_y, ready), 1);
    noc_async_atomic_barrier();
    if (tile_start == 0) {
        const uint32_t layers_per_token = get_arg_val<uint32_t>(TAIL_RT_OFFSET + 21);
        const uint32_t token_generation = (invocation + 1) / (2 * layers_per_token);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), token_generation * 8);
        for (uint32_t worker = 0; worker < 8; ++worker) {
            const uint32_t x = get_arg_val<uint32_t>(TAIL_RT_OFFSET + 4 + 2 * worker);
            const uint32_t y = get_arg_val<uint32_t>(TAIL_RT_OFFSET + 5 + 2 * worker);
#if SINGLE_LAYER_BARRIER
            noc_semaphore_inc(get_noc_addr(x, y, qb2_program_semaphore(12)), 1);
#else
            noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(12)), 1);
#endif
        }
        noc_async_atomic_barrier();
    }
}
