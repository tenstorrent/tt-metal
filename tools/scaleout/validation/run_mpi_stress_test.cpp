// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

void abort_if(bool condition, const std::string& message, int rank) {
    if (condition) {
        std::cerr << "Rank " << rank << ": " << message << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long iterations = 100000;
    int message_size = 1024 * 1024;

    if (argc > 1) {
        iterations = std::strtol(argv[1], nullptr, 10);
    }
    if (argc > 2) {
        message_size = std::atoi(argv[2]);
    }

    abort_if(size < 2, "At least two ranks are required", rank);
    abort_if(iterations <= 0, "Invalid iteration count", rank);
    abort_if(message_size <= 0, "Invalid message size", rank);

    std::vector<uint8_t> sendbuf(static_cast<std::size_t>(message_size));
    std::vector<uint8_t> recvbuf(static_cast<std::size_t>(message_size));

    std::memset(sendbuf.data(), rank & 0xff, sendbuf.size());

    int next = (rank + 1) % size;
    int previous = (rank - 1 + size) % size;

    std::string hostname(MPI_MAX_PROCESSOR_NAME, '\0');
    int hostname_length = 0;
    MPI_Get_processor_name(hostname.data(), &hostname_length);
    hostname.resize(static_cast<std::size_t>(hostname_length));

    std::cout << "Rank " << rank << "/" << size << " on " << hostname << ", pid " << getpid() << "\n" << std::flush;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (long iteration = 0; iteration < iterations; ++iteration) {
        MPI_Sendrecv(
            sendbuf.data(),
            message_size,
            MPI_BYTE,
            next,
            100,
            recvbuf.data(),
            message_size,
            MPI_BYTE,
            previous,
            100,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        uint8_t expected = static_cast<uint8_t>(previous & 0xff);

        for (std::size_t offset = 0; offset < recvbuf.size(); ++offset) {
            if (recvbuf[offset] != expected) {
                std::cerr << "Rank " << rank << ": corruption at iteration " << iteration << "; offset=" << offset
                          << " expected=" << static_cast<unsigned>(expected)
                          << " actual=" << static_cast<unsigned>(recvbuf[offset]) << "\n";
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }

        if ((iteration % 1000) == 0) {
            MPI_Allreduce(MPI_IN_PLACE, &expected, 1, MPI_UINT8_T, MPI_BXOR, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;

    double maximum_elapsed = 0.0;
    MPI_Reduce(&elapsed, &maximum_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "PASS: " << iterations << " iterations, " << message_size << " bytes/message, " << maximum_elapsed
                  << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
