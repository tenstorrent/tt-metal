// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn_test_fixtures.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/fusion/device/fusion_dispatch_op_helpers.hpp"

namespace ttnn::operations::experimental::fusion::test {

using ::tt::tt_metal::CBDescriptor;
using ::tt::tt_metal::ProgramDescriptor;

namespace {

Tensor make_device_tensor(tt::tt_metal::distributed::MeshDevice* device) {
    ttnn::Shape shape{1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    return ttnn::random::random(shape, DataType::BFLOAT16).to_device(device);
}

const CBSlot* find_cb_slot(const AddressSlots& slots, std::uint32_t cb_idx) {
    for (const auto& s : slots.cb_slots) {
        if (s.cb_idx == cb_idx) {
            return &s;
        }
    }
    return nullptr;
}

}  // namespace

// The fusion op caches a ProgramDescriptor and refreshes IO-tensor-derived CB backings in place
// on every dispatch.  A CB may be backed by either a raw ``Buffer*`` (.buffer) or a ``MeshTensor*``
// (.tensor).  This verifies that compute_address_slots discovers BOTH kinds and that
// patch_stale_descriptor refreshes each to the live tensor while preserving the backing kind and
// the buffer-xor-tensor invariant.
TEST_F(TTNNFixtureWithDevice, FusionHelpersRefreshBufferAndTensorBackedCBs) {
    Tensor a = make_device_tensor(device_);  // CB 0 will be .buffer-backed by this tensor
    Tensor b = make_device_tensor(device_);  // CB 1 will be .tensor-backed by this tensor

    ProgramDescriptor desc;
    {
        CBDescriptor cb_buffer;
        cb_buffer.buffer = a.buffer();
        desc.cbs.push_back(cb_buffer);

        CBDescriptor cb_tensor;
        cb_tensor.tensor = &b.mesh_tensor();
        desc.cbs.push_back(cb_tensor);
    }

    const std::vector<Tensor> io_tensors = {a, b};
    const AddressSlots slots = compute_address_slots(desc, io_tensors);

    // Both CBs are discovered and matched to the right IO tensor, with the backing kind recorded.
    ASSERT_EQ(slots.cb_slots.size(), 2u);

    const CBSlot* buffer_slot = find_cb_slot(slots, /*cb_idx=*/0);
    const CBSlot* tensor_slot = find_cb_slot(slots, /*cb_idx=*/1);
    ASSERT_NE(buffer_slot, nullptr);
    ASSERT_NE(tensor_slot, nullptr);

    EXPECT_EQ(buffer_slot->io_tensor_index, 0u);
    EXPECT_FALSE(buffer_slot->tensor_backed);
    EXPECT_EQ(tensor_slot->io_tensor_index, 1u);
    EXPECT_TRUE(tensor_slot->tensor_backed);

    // Fresh tensors (distinct allocations, since a/b are still alive) stand in for a cache hit.
    Tensor a2 = make_device_tensor(device_);
    Tensor b2 = make_device_tensor(device_);
    ASSERT_NE(a2.buffer()->address(), a.buffer()->address());
    ASSERT_NE(b2.buffer()->address(), b.buffer()->address());

    const std::vector<Tensor> io_tensors_refreshed = {a2, b2};
    patch_stale_descriptor(desc, io_tensors_refreshed, slots);

    // Buffer-backed CB now points at the new tensor's buffer; tensor stays cleared.
    EXPECT_EQ(desc.cbs[0].buffer, a2.buffer());
    EXPECT_EQ(desc.cbs[0].tensor, nullptr);

    // Tensor-backed CB now points at the new tensor's MeshTensor; buffer stays cleared.
    EXPECT_EQ(desc.cbs[1].tensor, &b2.mesh_tensor());
    EXPECT_EQ(desc.cbs[1].buffer, nullptr);

    // The tensor-backed CB resolves to the new tensor's L1 address (the value the dynamic-CB
    // consumer reads via cb.tensor->mesh_buffer().get_reference_buffer()).
    EXPECT_EQ(desc.cbs[1].tensor->mesh_buffer().get_reference_buffer()->address(), b2.buffer()->address());
}

}  // namespace ttnn::operations::experimental::fusion::test
