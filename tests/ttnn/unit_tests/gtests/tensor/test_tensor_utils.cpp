// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <optional>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::distributed::test {
namespace {

using namespace tt::tt_metal;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TensorSpec make_test_tensor_spec() {
    return TensorSpec(ttnn::Shape{1, 1, 32, 32}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
}

using AsOptionalMeshTensorTest = GenericMeshDeviceFixture;

TEST_F(AsOptionalMeshTensorTest, EmptyOptionalReturnsEmptyReference) {
    std::optional<Tensor> opt;
    auto ref = as_optional_mesh_tensor(opt);
    EXPECT_FALSE(ref.has_value());
}

TEST_F(AsOptionalMeshTensorTest, DeviceTensorReturnsReferenceToMeshTensor) {
    std::optional<Tensor> opt = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());

    auto ref = as_optional_mesh_tensor(opt);

    ASSERT_TRUE(ref.has_value());
    EXPECT_EQ(&ref.value(), &opt->mesh_tensor());
}

TEST_F(AsOptionalMeshTensorTest, HostTensorFatals) {
    std::optional<Tensor> opt = allocate_tensor_on_host(make_test_tensor_spec(), mesh_device_.get());

    EXPECT_THAT(
        ([&]() { (void)as_optional_mesh_tensor(opt); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("as_optional_mesh_tensor: expected device tensor")));
}

using GetCbAddressTest = GenericMeshDeviceFixture;

TEST(GetCbAddressHostTest, EmptyDescriptorReturnsAddressOffset) {
    CBDescriptor desc{};
    desc.address_offset = 1024;
    EXPECT_EQ(get_cb_address(desc), 1024u);
}

TEST(GetCbAddressHostTest, EmptyDescriptorZeroOffsetReturnsZero) {
    CBDescriptor desc{};
    EXPECT_EQ(get_cb_address(desc), 0u);
}

TEST_F(GetCbAddressTest, BufferOnlyReturnsBufferAddressPlusOffset) {
    Tensor input_tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    Buffer* buf = input_tensor.buffer();
    ASSERT_NE(buf, nullptr);

    CBDescriptor desc{};
    desc.buffer = buf;
    desc.address_offset = 64;

    EXPECT_EQ(get_cb_address(desc), buf->address() + 64u);
}

TEST_F(GetCbAddressTest, TensorOnlyReturnsTensorAddressPlusOffset) {
    Tensor input_tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    const MeshTensor* mt = &input_tensor.mesh_tensor();

    CBDescriptor desc{};
    desc.tensor = mt;
    desc.address_offset = 128;

    EXPECT_EQ(get_cb_address(desc), mt->address() + 128u);
}

TEST_F(GetCbAddressTest, BufferTakesPrecedenceOverTensor) {
    Tensor input_tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    Buffer* buf = input_tensor.buffer();
    const MeshTensor* mt = &input_tensor.mesh_tensor();
    ASSERT_NE(buf, nullptr);

    CBDescriptor desc{};
    desc.buffer = buf;
    desc.tensor = mt;
    desc.address_offset = 32;

    EXPECT_EQ(get_cb_address(desc), buf->address() + 32u);
}

}  // namespace
}  // namespace ttnn::distributed::test
