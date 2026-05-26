// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Worker-side kernel for the H2DStreamService worker-sync handshake test.
//
// Each invocation runs a single transfer: wait for the service kernel's
// multicast atomic-inc on `data_ready_sem_addr` to reach `target_data_ready`,
// copy the worker's assigned page range from the service's backing tensor
// (input) into a separately-allocated output tensor (same spec), then atomic-
// inc the service-core L1 counter at `consumed_counter_addr` to ack the
// iteration. The host re-enqueues this workload once per service iteration
// with an updated `target_data_ready` runtime arg.
//
// CT layout (must stay in sync with the test's worker-program build in
// tests/ttnn/unit_tests/gtests/tensor/test_h2d_stream_service.cpp):
//   [0] data_ready_sem_addr  (uint32, local worker-core L1)
//   [1] input_tensor_addr    (uint32, backing tensor base)
//   [2] output_tensor_addr   (uint32, output tensor base — same spec as input)
//   [3] page_size            (uint32, bytes per tensor page)
//   [4] scratch_cb_index     (uint32, single-slot scratch CB)
//   [5..] TensorAccessorArgs (single set; input and output share the same spec
//                             and reuse this args object with different bases)
//
// RT layout (per-worker; service-core fields are uniform across workers within
// a device but may differ across devices, so kept as RT):
//   [0] start_page                — first tensor page this worker handles (inclusive)
//   [1] end_page                  — last tensor page (exclusive)
//   [2] consumed_counter_addr     — L1 address of the counter on the service core
//   [3] service_noc_x             — physical NoC x of the service core
//   [4] service_noc_y             — physical NoC y of the service core
//
// data_ready_sem protocol: the host zero-initialises the GlobalSemaphore at
// service construction. Each service iteration the receiver kernel multicasts
// an inc-by-1 to every worker's local copy. Each worker spins until its local
// copy goes non-zero, resets it to 0, and proceeds. No iteration counter is
// needed on either side because every iteration is structurally identical.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t input_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t output_tensor_addr = get_compile_time_arg_val(2);
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(4);
// Metadata copy block (indices 5..7). When metadata_enabled is 0, the worker
// skips the L1 copy and metadata_{input,output}_addr / metadata_size_bytes
// are ignored. Indices stay reserved so the host build always emits 8 CT args
// before TensorAccessorArgs — the kernel constexpr resolves to 0 when the host
// passes 0, which `if constexpr` then drops at compile time.
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(5);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(6);
constexpr uint32_t metadata_input_addr = get_compile_time_arg_val(7);
constexpr uint32_t metadata_output_addr = get_compile_time_arg_val(8);
// Input and output share the same spec, so a single TensorAccessorArgs set is
// reused below with the two distinct base addresses.
constexpr auto acc_args = TensorAccessorArgs<9>();

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);

    auto input = TensorAccessor(acc_args, input_tensor_addr);
    auto output = TensorAccessor(acc_args, output_tensor_addr);
    const uint32_t cb_l1 = get_write_ptr(scratch_cb_index);

    // 1. Wait for the service kernel's multicast atomic-inc to land. The host
    //    zero-inits the GlobalSemaphore at service construction; each iteration
    //    the service kernel incs every worker's local copy by exactly 1. We
    //    spin until the local copy goes non-zero, then reset to 0 so the next
    //    iteration starts from a fresh state — no iteration counter / target
    //    value needs to flow between host and worker.
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_sem_addr);
    while (true) {
        invalidate_l1_cache();
        if (*sem > 0) {
            *sem = 0;
            break;
        }
    }

    // 2. Copy this worker's assigned page range input -> output, page by page,
    //    through the single-slot scratch CB.
    for (uint32_t p = start_page; p < end_page; ++p) {
        noc_async_read(input.get_noc_addr(p), cb_l1, page_size);
        noc_async_read_barrier();
        noc_async_write<page_size>(cb_l1, output.get_noc_addr(p), page_size);
    }
    noc_async_write_barrier();

    // 2b. Snapshot the multicast metadata from the service-owned L1 input region
    //     into a worker-owned L1 output region BEFORE atomic-incing the consumed
    //     counter. The service kernel can (and will, under high writer pressure)
    //     start the next iter's metadata multicast as soon as it sees the ack,
    //     which would overwrite the input region while the host is still trying
    //     to verify it. Copying into a separate output region — never written
    //     by the service kernel — gives the host a stable per-iter snapshot.
    //     volatile byte loops avoid the compiler reordering the copy past the
    //     atomic-inc below.
    if constexpr (metadata_enabled) {
        auto* src = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(metadata_input_addr);
        auto* dst = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(metadata_output_addr);
        for (uint32_t i = 0; i < metadata_size_bytes; ++i) {
            dst[i] = src[i];
        }
    }

    // 3. Atomic-inc the consumed counter on the service core. The service kernel
    //    polls for `num_workers` more incs since its last iteration and proceeds
    //    to the next transfer.
    const uint64_t consumed_noc = get_noc_addr(service_noc_x, service_noc_y, consumed_counter_addr);
    noc_semaphore_inc(consumed_noc, 1);
    noc_async_atomic_barrier();
}
