// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace tt::tt_metal::distributed;

void interface_h2d_workload(
    std::size_t page_size, std::size_t data_size, const std::string& socket_id, uint32_t num_iterations) {
    auto connected_socket = H2DSocket::connect(socket_id);
    connected_socket->set_page_size(page_size);

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    uint32_t num_writes = data_size / page_size;

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));

    for (uint32_t i = 0; i < num_iterations; i++) {
        std::iota(src_vec.begin(), src_vec.end(), i);
        for (uint32_t j = 0; j < num_writes; j++) {
            connected_socket->write(src_vec.data() + (j * page_size_words), 1);
        }
        std::cout << "H2D Barrier on socket for iteration " << i << std::endl;
        connected_socket->barrier();
        std::cout << "H2D Barrier done for iteration " << i << std::endl;
    }
}

void interface_d2h_workload(std::size_t page_size, std::size_t data_size, const std::string& socket_id) {
    auto connected_socket = D2HSocket::connect(socket_id);
    connected_socket->set_page_size(page_size);

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    uint32_t num_reads = data_size / page_size;

    std::vector<uint32_t> dst_vec(data_size / sizeof(uint32_t));
    std::cout << "Validating: " << data_size << " bytes" << std::endl;
    for (uint32_t i = 0; i < num_reads; i++) {
        connected_socket->read(dst_vec.data() + (i * page_size_words), 1);
    }
    std::cout << "D2H Barrier on socket" << std::endl;
    connected_socket->barrier();
    std::cout << "D2H Barrier done" << std::endl;

    std::vector<uint32_t> expected(data_size / sizeof(uint32_t));
    std::iota(expected.begin(), expected.end(), 0);
    if (dst_vec == expected) {
        std::cout << "D2H data verification PASSED" << std::endl;
    } else {
        std::cerr << "D2H data verification FAILED" << std::endl;
        exit(1);
    }
}

void interface_loopback_workload(
    std::size_t page_size,
    std::size_t data_size,
    const std::string& h2d_socket_id,
    const std::string& d2h_socket_id,
    uint32_t num_iterations) {
    std::cout << "Creating H2D Socket" << std::endl;
    auto h2d_socket = H2DSocket::connect(h2d_socket_id);
    std::cout << "Creating D2H Socket" << std::endl;
    auto d2h_socket = D2HSocket::connect(d2h_socket_id);
    std::cout << "Setting Page Size" << std::endl;
    h2d_socket->set_page_size(page_size);
    d2h_socket->set_page_size(page_size);
    std::cout << "Setting Page Size Done" << std::endl;
    uint32_t page_size_words = page_size / sizeof(uint32_t);
    uint32_t data_size_words = data_size / sizeof(uint32_t);
    uint32_t num_txns = data_size / page_size;

    std::vector<uint32_t> src_vec(data_size_words * num_iterations);
    std::vector<uint32_t> dst_vec(data_size_words * num_iterations, 0);
    std::iota(src_vec.begin(), src_vec.end(), 0);

    std::cout << "Issue Transactions" << std::endl;
    std::thread write_thread([&]() {
        for (uint32_t i = 0; i < num_iterations; i++) {
            for (uint32_t j = 0; j < num_txns; j++) {
                h2d_socket->write(src_vec.data() + (i * data_size_words) + (j * page_size_words), 1);
            }
        }
    });

    std::thread read_thread([&]() {
        for (uint32_t i = 0; i < num_iterations; i++) {
            for (uint32_t j = 0; j < num_txns; j++) {
                d2h_socket->read(dst_vec.data() + (i * data_size_words) + (j * page_size_words), 1);
            }
        }
    });

    write_thread.join();
    read_thread.join();

    std::cout << "Loopback H2D Barrier" << std::endl;
    h2d_socket->barrier();
    std::cout << "Loopback D2H Barrier" << std::endl;
    d2h_socket->barrier();

    if (src_vec == dst_vec) {
        std::cout << "Loopback data verification PASSED" << std::endl;
    } else {
        std::cerr << "Loopback data verification FAILED" << std::endl;
        exit(1);
    }
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <h2d|d2h|loopback> <socket_id> [page_size] [data_size] [num_iterations]"
              << std::endl;
    std::cerr << "  For loopback mode, socket_id is the H2D socket id; D2H socket id is derived by replacing 'h2d' "
                 "with 'd2h'."
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    std::string socket_id = argv[2];
    std::size_t page_size = (argc > 3) ? std::stoull(argv[3]) : 64;
    std::size_t data_size = (argc > 4) ? std::stoull(argv[4]) : 1024;
    uint32_t num_iterations = (argc > 5) ? std::stoul(argv[5]) : 10;

    std::cout << "MultiProcSocket: mode=" << mode << " socket_id=" << socket_id << " page_size=" << page_size
              << " data_size=" << data_size << " num_iterations=" << num_iterations << std::endl;

    if (mode == "h2d") {
        interface_h2d_workload(page_size, data_size, socket_id, num_iterations);
    } else if (mode == "d2h") {
        interface_d2h_workload(page_size, data_size, socket_id);
    } else if (mode == "loopback") {
        std::string d2h_socket_id = (argc > 6) ? argv[6] : "test_d2h_xproc_0";
        interface_loopback_workload(page_size, data_size, socket_id, d2h_socket_id, num_iterations);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
