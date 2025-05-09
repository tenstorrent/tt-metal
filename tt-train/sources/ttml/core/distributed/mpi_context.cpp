// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mpi_context.hpp"

#include <mpi.h>

#include <span>
#include <stdexcept>
namespace ttml::core::distributed {

class MPIContext::Pimpl {
public:
    MPI_Comm m_comm{};
    int m_rank = 0;
    int m_size = 0;

    Pimpl() : m_comm(MPI_COMM_WORLD), m_rank(0), m_size(0) {
    }

    void init(int argc, char** argv) {
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init_thread failed");
        }
        update_rank_size();
    }

    void finalize() {
        MPI_Finalize();
    }

    int get_rank() const {
        return m_rank;
    }
    int get_size() const {
        return m_size;
    }

    void barrier() const {
        MPI_Barrier(m_comm);
    }

    void send(std::span<std::byte> data, int dest, int tag) const {
        int t = (tag < 0 ? 0 : tag);
        MPI_Send(data.data(), static_cast<int>(data.size()), MPI_BYTE, dest, t, m_comm);
    }

    void recv(std::span<std::byte> data, int source, int tag) const {
        int t = (tag < 0 ? MPI_ANY_TAG : tag);
        int src = (source < 0 ? MPI_ANY_SOURCE : source);
        MPI_Status status;
        MPI_Recv(data.data(), static_cast<int>(data.size()), MPI_BYTE, src, t, m_comm, &status);
    }

    void broadcast(std::span<std::byte> data, int root) const {
        MPI_Bcast(data.data(), static_cast<int>(data.size()), MPI_BYTE, root, m_comm);
    }

private:
    void update_rank_size() {
        MPI_Comm_rank(m_comm, &m_rank);
        MPI_Comm_size(m_comm, &m_size);
    }
};

// ——— MPIContext public wrappers ———

MPIContext::MPIContext(int argc, char** argv) : m_pimpl(std::make_unique<Pimpl>()) {
    m_pimpl->init(argc, argv);
}

MPIContext::~MPIContext() = default;

void MPIContext::finalize() {
    m_pimpl->finalize();
}

int MPIContext::get_rank() const {
    return m_pimpl->get_rank();
}

int MPIContext::get_size() const {
    return m_pimpl->get_size();
}

void MPIContext::barrier() const {
    m_pimpl->barrier();
}

void MPIContext::send(std::span<std::byte> data, int dest, int tag) const {
    m_pimpl->send(data, dest, tag);
}

void MPIContext::recv(std::span<std::byte> data, int source, int tag) const {
    m_pimpl->recv(data, source, tag);
}

void MPIContext::broadcast(std::span<std::byte> data, int root) const {
    m_pimpl->broadcast(data, root);
}
}  // namespace ttml::core::distributed
