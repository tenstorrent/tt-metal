#include "models/demos/deepseek_v3_b1/pipeline_manager/socket_connector.hpp"

#include <algorithm>

namespace models::demos::deepseek_v3_b1::pipeline_manager {

H2DWriterSocket::H2DWriterSocket(const std::string& socket_id, uint32_t page_size_bytes, uint32_t connect_timeout_ms) :
    socket_(tt::tt_metal::distributed::H2DSocket::connect(socket_id, connect_timeout_ms)),
    page_buffer_(page_size_bytes / sizeof(uint32_t), 0) {
    socket_->set_page_size(page_size_bytes);
}

void H2DWriterSocket::write_token(uint32_t token_id) {
    std::fill(page_buffer_.begin(), page_buffer_.end(), 0);
    page_buffer_.front() = token_id;
    socket_->write(page_buffer_.data(), 1);
}

void H2DWriterSocket::barrier() { socket_->barrier(); }

D2HReaderSocket::D2HReaderSocket(const std::string& socket_id, uint32_t page_size_bytes, uint32_t connect_timeout_ms) :
    socket_(tt::tt_metal::distributed::D2HSocket::connect(socket_id, connect_timeout_ms)),
    page_buffer_(page_size_bytes / sizeof(uint32_t), 0) {
    socket_->set_page_size(page_size_bytes);
}

uint32_t D2HReaderSocket::read_token() {
    socket_->read(page_buffer_.data(), 1);
    return page_buffer_.front();
}

void D2HReaderSocket::barrier() { socket_->barrier(); }

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
