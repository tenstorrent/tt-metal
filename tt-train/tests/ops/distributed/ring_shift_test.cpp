// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Test for ring_shift distributed operation.
 *
 * Ring shift moves tensor data between adjacent devices in a ring:
 * - Forward: device i sends to device (i+1) % ring_size
 * - Backward: device i sends to device (i-1+ring_size) % ring_size
 */

#include <gtest/gtest.h>

#include <array>
#include <core/xtensor_utils.hpp>
#include <optional>
#include <tt-metalium/distributed_context.hpp>
#include <tt-umd-workload/cluster.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/distributed/socket_manager.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "test_utils/mesh_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/create_socket.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

using ttml::ttnn_fixed::distributed::RingShiftDirection;

class GalaxyRingShiftTest : public ::testing::Test {
public:
    static inline const tt::tt_metal::distributed::MeshShape kMeshShape{8, 4};

    static void SetUpTestSuite() {
        if (ttml::test_utils::system_supports_mesh(kMeshShape)) {
            if (!ttml::autograd::ctx().is_distributed_context_initialized()) {
                ttml::autograd::ctx().initialize_distributed_context(0, nullptr);
            }
            ttml::ttnn_fixed::distributed::enable_fabric(static_cast<uint32_t>(kMeshShape.mesh_size()));
            ttml::autograd::ctx().open_device(kMeshShape);
            ttml::autograd::ctx().set_seed(42);
            ttml::autograd::ctx().initialize_socket_manager(ttnn::distributed::SocketType::FABRIC);
        }
    }
    static void TearDownTestSuite() {
        if (ttml::test_utils::system_supports_mesh(kMeshShape)) {
            ttml::autograd::ctx().close_device();
        }
    }

    void SetUp() override {
        SKIP_UNLESS_MESH_SUPPORTED(kMeshShape);
    }
};

static void TestRingShift(
    const size_t batch,
    const size_t seq,
    const size_t hidden,
    const uint32_t cluster_axis,
    const uint32_t shard_dim,
    const RingShiftDirection direction,
    const bool test_backward_grad = false,
    const float rtol = 1e-3F,
    const float atol = 1e-5F) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    const uint32_t num_devices = mesh_shape.mesh_size();
    const uint32_t mesh_rows = mesh_shape[0];
    const uint32_t mesh_cols = mesh_shape[1];

    // Create input tensor - full tensor that will be sharded
    auto& rng = autograd::ctx().get_generator();
    const auto seed = rng();
    const xt::xarray<float> xtensor = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1UL, seq, hidden}, 0.0F, 2.0F, seed);

    // Shard tensor across devices along the appropriate dimension
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, shard_dim, cluster_axis);
    const auto tt_tensor =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    // Inputs do not require grad by default; without this ring_shift records no backward node.
    auto tensor = autograd::create_tensor(tt_tensor, /*requires_grad=*/test_backward_grad);

    // Get original sharded tensors for comparison
    const auto original_xtensors = core::to_xtensor<float>(tensor->get_value(), core::IdentityComposer{});

    // Perform ring shift
    const auto shifted_tensor = ops::distributed::ring_shift(tensor, cluster_axis, direction);

    // Get output back to xtensor
    const auto shifted_xtensors = core::to_xtensor<float>(shifted_tensor->get_value(), core::IdentityComposer{});

    // Verify shape is preserved
    for (size_t i = 0; i < num_devices; ++i) {
        EXPECT_EQ(shifted_xtensors[i].shape(), original_xtensors[i].shape()) << "Shape mismatch on device " << i;
    }

    // Verify ring shift correctness:
    // After forward shift, device i should have data from device (i-1+ring_size) % ring_size
    // After backward shift, device i should have data from device (i+1) % ring_size
    const bool forward = (direction == ttnn_fixed::distributed::RingShiftDirection::Forward);
    for (uint32_t row = 0; row < mesh_rows; ++row) {
        for (uint32_t col = 0; col < mesh_cols; ++col) {
            const size_t device_idx = row * mesh_cols + col;

            uint32_t src_row = row;
            uint32_t src_col = col;
            if (cluster_axis == 0) {
                src_row = forward ? (row + mesh_rows - 1) % mesh_rows : (row + 1) % mesh_rows;
            } else {
                src_col = forward ? (col + mesh_cols - 1) % mesh_cols : (col + 1) % mesh_cols;
            }
            size_t src_device_idx = src_row * mesh_cols + src_col;

            EXPECT_TRUE(xt::allclose(original_xtensors[src_device_idx], shifted_xtensors[device_idx], rtol, atol))
                << "Value mismatch on device " << device_idx << " (expected data from device " << src_device_idx << ")";
        }
    }

    // Optionally test backward gradient flow
    if (test_backward_grad) {
        const auto grad_seed = rng();
        xt::xarray<float> grad_data = ttml::test_utils::make_uniform_xarray<float>(
            std::array<std::size_t, 4>{batch, 1UL, seq, hidden}, 0.0F, 1.0F, grad_seed);

        const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, shard_dim, cluster_axis);
        const auto tt_grad_tensor =
            core::from_xtensor<float, ttnn::DataType::BFLOAT16>(grad_data, device, ttnn::Layout::TILE, mapper.get());

        const auto original_grad_xtensors = core::to_xtensor<float>(tt_grad_tensor, core::IdentityComposer{});

        shifted_tensor->set_grad(tt_grad_tensor);
        shifted_tensor->backward();
        const auto grad_xtensors = core::to_xtensor<float>(tensor->get_grad(), core::IdentityComposer{});

        // Gradient flows in opposite direction
        for (uint32_t row = 0; row < mesh_rows; ++row) {
            for (uint32_t col = 0; col < mesh_cols; ++col) {
                const size_t device_idx = row * mesh_cols + col;

                uint32_t src_row = row;
                uint32_t src_col = col;
                if (cluster_axis == 0) {
                    src_row = forward ? (row + 1) % mesh_rows : (row + mesh_rows - 1) % mesh_rows;
                } else {
                    src_col = forward ? (col + 1) % mesh_cols : (col + mesh_cols - 1) % mesh_cols;
                }
                const size_t src_device_idx = src_row * mesh_cols + src_col;

                EXPECT_EQ(grad_xtensors[device_idx].shape(), original_grad_xtensors[device_idx].shape());
                EXPECT_TRUE(xt::allclose(original_grad_xtensors[src_device_idx], grad_xtensors[device_idx], rtol, atol))
                    << "Gradient mismatch on device " << device_idx;
            }
        }
    }
}

TEST_F(GalaxyRingShiftTest, ForwardAlongColumns) {
    // hidden=64 sharded across 8 cols -> 8 per device
    TestRingShift(1, 32, 64, /*cluster_axis=*/1, /*shard_dim=*/3, RingShiftDirection::Forward);
}

TEST_F(GalaxyRingShiftTest, ForwardAlongRows) {
    // batch=4 sharded across 4 rows -> 1 per device
    TestRingShift(8, 32, 64, /*cluster_axis=*/0, /*shard_dim=*/0, RingShiftDirection::Forward);
}

TEST_F(GalaxyRingShiftTest, ForwardBig) {
    TestRingShift(
        16,
        1024,
        8192,
        /*cluster_axis=*/1,
        /*shard_dim=*/3,
        RingShiftDirection::Backward,
        /*test_backward_grad=*/false);
}

TEST_F(GalaxyRingShiftTest, Llama8bSeqLen8192Tp8Cp8Bs16) {
    TestRingShift(
        16, 1024, 512, /*cluster_axis=*/1, /*shard_dim=*/0, RingShiftDirection::Backward, /*test_backward_grad=*/false);
}

TEST_F(GalaxyRingShiftTest, BackwardAlongColumns) {
    TestRingShift(1, 32, 64, /*cluster_axis=*/1, /*shard_dim=*/3, RingShiftDirection::Backward);
}

TEST_F(GalaxyRingShiftTest, ForwardWithGradient) {
    TestRingShift(
        1, 32, 64, /*cluster_axis=*/1, /*shard_dim=*/3, RingShiftDirection::Forward, /*test_backward_grad=*/true);
}

TEST_F(GalaxyRingShiftTest, BackwardWithGradient) {
    TestRingShift(
        1, 32, 64, /*cluster_axis=*/1, /*shard_dim=*/3, RingShiftDirection::Backward, /*test_backward_grad=*/true);
}

TEST_F(GalaxyRingShiftTest, BackwardBig) {
    TestRingShift(
        16, 1024, 8192, /*cluster_axis=*/1, /*shard_dim=*/3, RingShiftDirection::Forward, /*test_backward_grad=*/true);
}

// Back-to-back shifts with no host synchronization in between: the socket state has to carry over from one shift to
// the next on device, and each shift reads the tensor the previous one has just received. With both_directions a
// second tensor is shifted the other way, alternating with the first, so both directions (separate sockets) are in
// flight at once. Only the final values are read back.
static void TestChainedRingShifts(
    const size_t batch,
    const size_t seq,
    const size_t hidden,
    const uint32_t cluster_axis,
    const uint32_t shard_dim,
    const uint32_t num_shifts,
    const bool both_directions = true,
    const ttnn::DataType dtype = ttnn::DataType::BFLOAT16) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const auto mesh_shape = device->shape();
    const uint32_t mesh_rows = mesh_shape[0];
    const uint32_t mesh_cols = mesh_shape[1];
    const uint32_t ring_size = mesh_shape[cluster_axis];

    auto& rng = autograd::ctx().get_generator();
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, shard_dim, cluster_axis);
    const auto make_input = [&]() {
        const xt::xarray<float> data = ttml::test_utils::make_uniform_xarray<float>(
            std::array<std::size_t, 4>{batch, 1UL, seq, hidden}, -1.0F, 1.0F, rng());
        if (dtype == ttnn::DataType::FLOAT32) {
            return core::from_xtensor<float, ttnn::DataType::FLOAT32>(data, device, ttnn::Layout::TILE, mapper.get());
        }
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, device, ttnn::Layout::TILE, mapper.get());
    };
    const ttnn::Tensor a_in = make_input();
    ASSERT_EQ(a_in.dtype(), dtype) << "TestChainedRingShifts supports BFLOAT16 and FLOAT32";
    const auto a_orig = core::to_xtensor<float>(a_in, core::IdentityComposer{});
    std::optional<ttnn::Tensor> b_in;
    std::vector<xt::xarray<float>> b_orig;
    if (both_directions) {
        b_in = make_input();
        b_orig = core::to_xtensor<float>(*b_in, core::IdentityComposer{});
    }

    ttnn::Tensor a = a_in;
    std::optional<ttnn::Tensor> b = b_in;
    for (uint32_t i = 0; i < num_shifts; ++i) {
        a = ttnn_fixed::distributed::ring_shift(a, cluster_axis, RingShiftDirection::Backward);
        if (b) {
            b = ttnn_fixed::distributed::ring_shift(*b, cluster_axis, RingShiftDirection::Forward);
        }
    }

    const auto a_out = core::to_xtensor<float>(a, core::IdentityComposer{});
    std::vector<xt::xarray<float>> b_out;
    if (b) {
        b_out = core::to_xtensor<float>(*b, core::IdentityComposer{});
    }
    const uint32_t k = num_shifts % ring_size;
    for (uint32_t row = 0; row < mesh_rows; ++row) {
        for (uint32_t col = 0; col < mesh_cols; ++col) {
            const size_t device_idx = row * mesh_cols + col;
            // Backward: device i ends with the data of device (i + k); Forward: of device (i - k).
            const auto source = [&](const bool backward) {
                const uint32_t pos = cluster_axis == 0 ? row : col;
                const uint32_t src = backward ? (pos + k) % ring_size : (pos + ring_size - k) % ring_size;
                return cluster_axis == 0 ? src * mesh_cols + col : row * mesh_cols + src;
            };
            // A shift only moves data, so the values must match exactly.
            EXPECT_TRUE(xt::allclose(a_orig[source(true)], a_out[device_idx], 0.0F, 0.0F))
                << "Backward chain mismatch on device " << device_idx;
            if (both_directions) {
                EXPECT_TRUE(xt::allclose(b_orig[source(false)], b_out[device_idx], 0.0F, 0.0F))
                    << "Forward chain mismatch on device " << device_idx;
            }
        }
    }
}

TEST_F(GalaxyRingShiftTest, ChainedShiftsAlongRowsNoHostSync) {
    // Ring of 8 on axis 0; 2 * 8 + 3 shifts so every device's data goes round the ring twice and ends 3 hops away.
    // Sequence sharded 8 ways: 2048 x 1024 bf16 = 4 MiB per device per tensor.
    TestChainedRingShifts(1, 16384, 1024, /*cluster_axis=*/0, /*shard_dim=*/2, /*num_shifts=*/19);
}

// Ring attention's backward shifts its float32 gradient accumulators. A float32 tile is one fabric packet, where two
// bf16 tiles share one, so the sender takes a different path. 2048 x 1024 float32 = 8 MiB per device per tensor.
TEST_F(GalaxyRingShiftTest, ChainedShiftsFloat32AlongRowsNoHostSync) {
    TestChainedRingShifts(
        1,
        16384,
        1024,
        /*cluster_axis=*/0,
        /*shard_dim=*/2,
        /*num_shifts=*/19,
        /*both_directions=*/true,
        ttnn::DataType::FLOAT32);
}

TEST_F(GalaxyRingShiftTest, ChainedShiftsAlongColumnsNoHostSync) {
    // Ring of 4 on axis 1: 2048 x 1024 bf16 = 4 MiB per device per tensor.
    TestChainedRingShifts(1, 8192, 1024, /*cluster_axis=*/1, /*shard_dim=*/2, /*num_shifts=*/11);
}

// Volume test: 45056 x 1024 bf16 is 88 MiB per device (45056 tiles), which one direct send writes into the
// receiver's tensor two tiles per fabric packet. 3 shifts back to back with no host sync in between, so the data
// ends 3 hops away.
TEST_F(GalaxyRingShiftTest, ChainedShiftsOfLargeShardsAlongRows) {
    TestChainedRingShifts(
        1, 8 * 45056, 1024, /*cluster_axis=*/0, /*shard_dim=*/2, /*num_shifts=*/3, /*both_directions=*/false);
}

TEST_F(GalaxyRingShiftTest, ChainedShiftsOfLargeShardsAlongColumns) {
    TestChainedRingShifts(
        1, 4 * 45056, 1024, /*cluster_axis=*/1, /*shard_dim=*/2, /*num_shifts=*/3, /*both_directions=*/false);
}

// The socket manager outlives the device, and its intra-mesh sockets hold buffers of the mesh they were created on.
// close_device drops them, so the first shift on the reopened mesh creates new ones. The reopened mesh has the same
// shape and fabric config, so tests that run after this one use it as the fixture's device.
TEST_F(GalaxyRingShiftTest, ShiftAfterDeviceReopen) {
    using namespace ttml;
    TestChainedRingShifts(1, 2048, 64, /*cluster_axis=*/0, /*shard_dim=*/2, /*num_shifts=*/1);

    const auto mesh_shape = autograd::ctx().get_device().shape();
    autograd::ctx().close_device();
    ttnn_fixed::distributed::enable_fabric(static_cast<uint32_t>(mesh_shape.mesh_size()));
    autograd::ctx().open_device(mesh_shape);

    TestChainedRingShifts(1, 2048, 64, /*cluster_axis=*/0, /*shard_dim=*/2, /*num_shifts=*/3);
}
