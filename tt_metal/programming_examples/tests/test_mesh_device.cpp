// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/system_mesh.hpp>

using namespace tt::tt_metal;

int main() {
    // Make sure creating a mesh device with the full system mesh shape works. This is specifically meant to catch
    // regressions such as https://github.com/tenstorrent/tt-metal/issues/30899. Ideally it should be ran against
    // a single host context (requires building `--without-distributed`) and the MPI one.
    auto mesh_shape = distributed::SystemMesh::instance().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create(mesh_device_config);

    auto context = distributed::multihost::DistributedContext::get_current_world();

    // Make sure distributed context collective operations don't throw.
    context->broadcast(tt::stl::Span<std::byte>({}), distributed::multihost::Rank{0});
    context->gather(tt::stl::Span<std::byte>({}), tt::stl::Span<std::byte>({}), distributed::multihost::Rank{0});
    context->scatter(tt::stl::Span<std::byte>({}), tt::stl::Span<std::byte>({}), distributed::multihost::Rank{0});
    context->all_gather(tt::stl::Span<std::byte>({}), tt::stl::Span<std::byte>({}));
    context->all_to_all(tt::stl::Span<std::byte>({}), tt::stl::Span<std::byte>({}));
    context->reduce(
        tt::stl::Span<int>({}),
        tt::stl::Span<int>({}),
        distributed::multihost::ReduceOp::SUM,
        distributed::multihost::Rank{0});

    return 0;
}
