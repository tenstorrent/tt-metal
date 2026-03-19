#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>

namespace models::demos::deepseek_v3_b1::pipeline_manager {

class H2DWriterSocket {
public:
    H2DWriterSocket(const std::string& socket_id, uint32_t page_size_bytes, uint32_t connect_timeout_ms);

    void write_token(uint32_t token_id);
    void barrier();

private:
    std::unique_ptr<tt::tt_metal::distributed::H2DSocket> socket_;
    std::vector<uint32_t> page_buffer_;
};

class D2HReaderSocket {
public:
    D2HReaderSocket(const std::string& socket_id, uint32_t page_size_bytes, uint32_t connect_timeout_ms);

    uint32_t read_token();
    void barrier();

private:
    std::unique_ptr<tt::tt_metal::distributed::D2HSocket> socket_;
    std::vector<uint32_t> page_buffer_;
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
