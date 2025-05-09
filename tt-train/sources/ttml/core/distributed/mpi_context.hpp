// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <span>

namespace ttml::core::distributed {

class MPIContext {
private:
    class Pimpl;
    std::unique_ptr<Pimpl> m_pimpl;

public:
    MPIContext(int argc, char** argv);
    ~MPIContext();
    void finalize();
    int get_rank() const;
    int get_size() const;
    void barrier() const;
    void send(std::span<std::byte> data, int dest, int tag = -1) const;
    void recv(std::span<std::byte> data, int source, int tag = -1) const;
    void broadcast(std::span<std::byte> data, int root) const;
    // TODO: implement split
    // MPIContext split(int color, int key) const;
};

}  // namespace ttml::core::distributed
