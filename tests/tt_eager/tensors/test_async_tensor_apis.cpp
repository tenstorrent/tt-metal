// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/types.hpp"
#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_impl.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "common/bfloat16.hpp"
#include "common/constants.hpp"

#include "tt_numpy/functions.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;

TEST_F(CommonFixture, TestTensorOwnershipSanity) {
    // Sanity test tensor read, write and update paths with synchronous
    // Ensure that tensor data is copied and owned as expected
    Device* device = this->devices_[0];
    Tensor host_tensor = tt::numpy::arange<float>(0, 32 * 32 * 4, 1);
    Tensor readback_tensor;

    auto func = [device, host_tensor, readback_tensor]() mutable {
        // Ensure that both the lambda and global scope have ownership to this tensor
        EXPECT_EQ(host_tensor.tensor_attributes.use_count(), 2);
        std::visit([](auto&& storage) {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                std::visit(
                    [](auto&& buf) {
                        using buf_type = std::decay_t<decltype(buf)>;
                        if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                            EXPECT_EQ(buf.use_count(), 1);
                        }
                    },
                storage.buffer);
            }
        }, host_tensor.get_storage());
        // Send tensor to device, read it back and copy it to empty tensor initialized by main thread
        Tensor reshaped_tensor = host_tensor.reshape(1, 1, 32, 128);
        auto device_tensor = reshaped_tensor.to(Layout::TILE).to(device);
        auto thread_local_tensor = device_tensor.cpu().to(Layout::ROW_MAJOR);
        readback_tensor.set_storage(thread_local_tensor.get_storage());
        readback_tensor.set_shape(thread_local_tensor.get_shape());
        readback_tensor.set_dtype(thread_local_tensor.get_dtype());
        readback_tensor.set_layout(thread_local_tensor.get_layout());
        // Ensure that the readback buffer is owned inside and outside the lambda
        std::visit([](auto&& storage) {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                std::visit(
                    [](auto&& buf) {
                        using buf_type = std::decay_t<decltype(buf)>;
                        if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                            EXPECT_EQ(buf.use_count(), 2);
                        }
                    },
                storage.buffer);
            }
        }, readback_tensor.get_storage());
    };

    func();
     std::visit([](auto&& storage) {
        using T = std::decay_t<decltype(storage)>;
        if constexpr (std::is_same_v<T, OwnedStorage>) {
            std::visit(
                [](auto&& buf) {
                    using buf_type = std::decay_t<decltype(buf)>;
                    if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                        EXPECT_EQ(buf.use_count(), 1);
                        for (int i = 0; i < 128 * 32; i++) {
                            EXPECT_EQ(buf[i], i);
                        }
                    }
                },
            storage.buffer);
        }
    },
    readback_tensor.get_storage());
    EXPECT_EQ(readback_tensor.get_dtype(), DataType::FLOAT32);
    EXPECT_EQ(readback_tensor.get_layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(readback_tensor.get_shape(), ttnn::Shape(Shape({1, 1, 32, 128})));
}

TEST_F(CommonFixture, TestTensorAsyncDataMovement) {
    // Test 2 data paths here (resembles async mode):
    // 1. Main -> Worker: Create a tensor in the main thread. Ensure that it is accessible in the worker thread even after its destroyed
    //                    by the main thread. This resembles host -> device data movement
    // 2. Worker -> Main: Create an empty tensor in the mainb thread. Populate it in the worker thread. Ensure that the tensor is correctly
    //                    populated in the main thread once the worker is done.
    Device* device = this->devices_[0];
    uint32_t tensor_start = 0;
    uint32_t num_tiles = 128;
    uint32_t tensor_stop = TILE_HEIGHT * TILE_WIDTH * num_tiles;
    Tensor readback_tensor;
    std::thread worker;

    {
        // host_tensor only lives in this scope
        Tensor host_tensor = tt::numpy::arange<float>(tensor_start, tensor_stop, 1);
        log_info(LogTest, "Spawning worker thread");
        worker = std::thread([tensor_stop, host_tensor, readback_tensor, device] () mutable {
            // Sleep for 3 seconds to ensure that main thread deallocates host_tensor
            std::this_thread::sleep_for(std::chrono::milliseconds(3000));
            log_info(LogTest, "Worker started");
            // Main thread should have deallocated host_tensor by this point
            EXPECT_EQ(host_tensor.tensor_attributes.use_count(), 1);
            // Ensure that the buffer inside host_buffer is owned by a single tensor_attr object
            // This buffer will not go out of scope until the last object owning it is destroyed (i.e. until the thread is done)
            std::visit([](auto&& storage) {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, OwnedStorage>) {
                    std::visit(
                        [](auto&& buf) {
                            using buf_type = std::decay_t<decltype(buf)>;
                            if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                                EXPECT_EQ(buf.use_count(), 1);
                            }
                        },
                    storage.buffer);
                }
            }, host_tensor.get_storage());

            Tensor reshaped_tensor = host_tensor.reshape(1, 1, 32, tensor_stop / 32);
            auto device_tensor = reshaped_tensor.to(Layout::TILE).to(device);
            auto thread_local_tensor = device_tensor.cpu().to(Layout::ROW_MAJOR);
            log_info(LogTest, "Worker populating empty host readback_tensor");
            readback_tensor.set_storage(thread_local_tensor.get_storage());
            readback_tensor.set_shape(thread_local_tensor.get_shape());
            readback_tensor.set_dtype(thread_local_tensor.get_dtype());
            readback_tensor.set_layout(thread_local_tensor.get_layout());
            // Ensure that this buffer is currently owned by both the thread_local and read_back tensors
            // This is because we explictly pass in the buffer to a new tensor_attr object
            std::visit([](auto&& storage) {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, OwnedStorage>) {
                    std::visit(
                        [](auto&& buf) {
                            using buf_type = std::decay_t<decltype(buf)>;
                            if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                                EXPECT_EQ(buf.use_count(), 2);
                            }
                        },
                    storage.buffer);
                }
            }, readback_tensor.get_storage());
            log_info(LogTest, "Worker Done");
        });
        // Call deallocate on the tensor in the main thread to ensure that this call is safe
        // i.e.: the tensor should not be deallocated until the thread is done with it
        log_info(LogTest, "Main thread calling deallocate on tensor passed to worker");
        host_tensor.deallocate();
    }
    worker.join();
    log_info(LogTest, "Verifying populated tensor in main thread");
    std::visit(
    [tensor_start, tensor_stop](auto&& storage) {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, OwnedStorage>) {
                    std::visit(
                        [tensor_start, tensor_stop](auto&& buf) {
                            using buf_type = std::decay_t<decltype(buf)>;
                            if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                                EXPECT_EQ(buf.use_count(), 1);
                                for (int i = tensor_start; i < tensor_stop; i++) {
                                    EXPECT_EQ(buf[i], i);
                                }
                            }
                        },
                    storage.buffer);
                }
            },
        readback_tensor.get_storage());
    EXPECT_EQ(readback_tensor.get_dtype(), DataType::FLOAT32);
    EXPECT_EQ(readback_tensor.get_layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(readback_tensor.get_shape(), ttnn::Shape(Shape({1, 1, 32, tensor_stop / 32})));
}
