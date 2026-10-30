// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

// Stateful set_state/with_state coverage: read a DRAM region into L1 in chunks
// through ONE read state (one-packet flavor: LEN latched at set_state), then
// write it back to a second DRAM region through ONE write state (any-len
// flavor: LEN written per issue). The states are set with a zero local address
// - the state base carries only the target identity - and every chunk
// re-supplies its full local address, per the stateful contract.
void kernel_main() {
    const uint32_t dram_in_addr = get_arg(args::dram_in_addr);
    const uint32_t dram_out_addr = get_arg(args::dram_out_addr);
    const uint32_t l1_addr = get_arg(args::l1_addr);
    const uint32_t dram_bank_id = get_arg(args::dram_bank_id);

    constexpr uint32_t chunk_bytes = 64;
    constexpr uint32_t num_chunks = 4;

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram;

    // One-packet stateful reads: LEN and target identity latched once.
    noc.set_async_read_state<NocOptions::DEFAULT, chunk_bytes>(dram, chunk_bytes, {.bank_id = dram_bank_id, .addr = 0});
    for (uint32_t i = 0; i < num_chunks; i++) {
        CoreLocalMem<std::uint32_t> l1_chunk(l1_addr + i * chunk_bytes);
        noc.async_read_with_state<NocOptions::DEFAULT, chunk_bytes>(
            dram, l1_chunk, chunk_bytes, {.bank_id = dram_bank_id, .addr = dram_in_addr + i * chunk_bytes}, {});
    }
    noc.async_read_barrier();

    // Any-len stateful writes: target identity latched once, LEN per issue.
    noc.set_async_write_state(dram, chunk_bytes, {.bank_id = dram_bank_id, .addr = 0});
    for (uint32_t i = 0; i < num_chunks; i++) {
        CoreLocalMem<std::uint32_t> l1_chunk(l1_addr + i * chunk_bytes);
        noc.async_write_with_state(
            l1_chunk, dram, chunk_bytes, {}, {.bank_id = dram_bank_id, .addr = dram_out_addr + i * chunk_bytes});
    }
    noc.async_write_barrier();
}
