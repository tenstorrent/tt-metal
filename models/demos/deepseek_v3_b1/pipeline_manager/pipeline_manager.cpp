#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include "models/demos/deepseek_v3_b1/pipeline_manager/socket_connector.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {
namespace {

struct RequestInput {
    std::string request_id;
    std::vector<uint32_t> prompt_token_ids;
    uint32_t max_new_tokens = 0;
    std::optional<uint32_t> eos_token_id;
};

RequestInput create_request_input(const PipelineManagerRequest& request) {
    return RequestInput{
        .request_id = request.request_id,
        .prompt_token_ids = request.prompt_token_ids,
        .max_new_tokens = request.max_new_tokens,
        .eos_token_id = request.eos_token_id,
    };
}

std::string join_generated_tokens(const std::vector<uint32_t>& generated_token_ids) {
    std::ostringstream output;
    for (size_t idx = 0; idx < generated_token_ids.size(); ++idx) {
        if (idx != 0) {
            output << ",";
        }
        output << generated_token_ids[idx];
    }
    return output.str();
}

class TokenStream {
public:
    explicit TokenStream(std::ostream& output_stream) : output_stream_(output_stream) {}

    void emit_token(const std::string& request_id, size_t token_index, uint32_t token_id) {
        output_stream_ << "TOKEN\t" << sanitize_field(request_id) << "\t" << token_index << "\t" << token_id
                       << std::endl;
    }

    void emit_complete(const std::string& request_id, const std::vector<uint32_t>& generated_token_ids) {
        output_stream_ << "COMPLETE\t" << sanitize_field(request_id) << "\t" << generated_token_ids.size() << "\t"
                       << join_generated_tokens(generated_token_ids) << std::endl;
    }

    void emit_error(const std::string& request_id, const std::string& message) {
        output_stream_ << "ERROR\t" << sanitize_field(request_id) << "\t" << sanitize_field(message) << std::endl;
    }

private:
    static std::string sanitize_field(const std::string& value) {
        std::string sanitized = value;
        std::replace(sanitized.begin(), sanitized.end(), '\t', ' ');
        std::replace(sanitized.begin(), sanitized.end(), '\n', ' ');
        std::replace(sanitized.begin(), sanitized.end(), '\r', ' ');
        return sanitized;
    }

    std::ostream& output_stream_;
};

}  // namespace

struct PipelineManager::Impl {
    Impl(std::string h2d_socket_id, std::string d2h_socket_id, uint32_t page_size_bytes, uint32_t connect_timeout_ms) :
        writer_socket_(std::move(h2d_socket_id), page_size_bytes, connect_timeout_ms),
        reader_socket_(std::move(d2h_socket_id), page_size_bytes, connect_timeout_ms) {}

    ~Impl() { stop_threads(); }

    void start() { start_threads(); }

    void stop() {
        stop_threads();
        writer_socket_.barrier();
        reader_socket_.barrier();
    }

    void write_token(uint32_t token_id) {
        ensure_started();
        enqueue_write(token_id);
    }

    uint32_t read_token() {
        ensure_started();
        enqueue_read();
        return await_read();
    }

    void run_one_shot(PipelineManagerRequest& request, std::ostream& output_stream) {
        if (busy_) {
            throw std::runtime_error("PipelineManager currently supports one in-flight request");
        }
        if (request.request_id.empty()) {
            throw std::runtime_error("request_id must not be empty");
        }

        RequestInput request_input = create_request_input(request);
        std::vector<uint32_t> generated_token_ids;
        TokenStream token_stream(output_stream);

        busy_ = true;
        start_threads();

        try {
            run_request(request_input, generated_token_ids, token_stream);
        } catch (const std::exception& error) {
            token_stream.emit_error(request_input.request_id, error.what());
            stop_threads();
            busy_ = false;
            throw;
        }

        stop_threads();
        busy_ = false;
    }

    void start_threads() {
        if (threads_started_) {
            return;
        }
        stop_requested_.store(false, std::memory_order_release);
        writer_thread_ = std::thread([this]() { writer_loop(); });
        reader_thread_ = std::thread([this]() { reader_loop(); });
        threads_started_ = true;
    }

    void stop_threads() {
        if (!threads_started_) {
            clear_queues();
            return;
        }
        stop_requested_.store(true, std::memory_order_release);
        write_cv_.notify_all();
        read_request_cv_.notify_all();

        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
        if (reader_thread_.joinable()) {
            reader_thread_.join();
        }

        clear_queues();
        threads_started_ = false;
    }

    void clear_queues() {
        {
            std::lock_guard<std::mutex> write_lock(write_mutex_);
            pending_writes_.clear();
        }
        {
            std::lock_guard<std::mutex> read_request_lock(read_request_mutex_);
            pending_read_count_ = 0;
        }
        {
            std::lock_guard<std::mutex> read_result_lock(read_result_mutex_);
            read_results_.clear();
        }
    }

    void ensure_started() const {
        if (!threads_started_) {
            throw std::runtime_error("PipelineManager threads have not been started");
        }
    }

    void writer_loop() {
        while (true) {
            uint32_t pending_write = 0;
            {
                std::unique_lock<std::mutex> lock(write_mutex_);
                write_cv_.wait(lock, [this]() {
                    return stop_requested_.load(std::memory_order_acquire) || !pending_writes_.empty();
                });
                if (stop_requested_.load(std::memory_order_acquire) && pending_writes_.empty()) {
                    break;
                }
                pending_write = pending_writes_.front();
                pending_writes_.pop_front();
            }
            writer_socket_.write_token(pending_write);
        }
        writer_socket_.barrier();
    }

    void reader_loop() {
        while (true) {
            bool should_read = false;
            {
                std::unique_lock<std::mutex> lock(read_request_mutex_);
                read_request_cv_.wait(lock, [this]() {
                    return stop_requested_.load(std::memory_order_acquire) || pending_read_count_ > 0;
                });
                if (stop_requested_.load(std::memory_order_acquire) && pending_read_count_ == 0) {
                    break;
                }
                --pending_read_count_;
                should_read = true;
            }

            if (should_read) {
                const uint32_t token_id = reader_socket_.read_token();
                {
                    std::lock_guard<std::mutex> lock(read_result_mutex_);
                    read_results_.push_back(token_id);
                }
                read_result_cv_.notify_one();
            }
        }
        reader_socket_.barrier();
    }

    void enqueue_write(uint32_t token_id) {
        {
            std::lock_guard<std::mutex> lock(write_mutex_);
            pending_writes_.push_back(token_id);
        }
        write_cv_.notify_one();
    }

    void enqueue_read() {
        {
            std::lock_guard<std::mutex> lock(read_request_mutex_);
            ++pending_read_count_;
        }
        read_request_cv_.notify_one();
    }

    uint32_t await_read() {
        std::unique_lock<std::mutex> lock(read_result_mutex_);
        read_result_cv_.wait(lock, [this]() { return !read_results_.empty(); });
        const uint32_t token_id = read_results_.front();
        read_results_.pop_front();
        return token_id;
    }

    uint32_t run_prefill(
        const RequestInput& request, std::vector<uint32_t>& generated_token_ids, TokenStream& token_stream) {
        if (request.prompt_token_ids.empty()) {
            throw std::runtime_error("Prompt token list must not be empty");
        }

        uint32_t last_output_token = 0;
        for (uint32_t prompt_token : request.prompt_token_ids) {
            enqueue_write(prompt_token);
            enqueue_read();
            last_output_token = await_read();
        }

        generated_token_ids.push_back(last_output_token);
        token_stream.emit_token(request.request_id, 0, last_output_token);
        return last_output_token;
    }

    void run_decode(
        const RequestInput& request,
        std::vector<uint32_t>& generated_token_ids,
        uint32_t first_generated_token,
        TokenStream& token_stream) {
        uint32_t input_token = first_generated_token;
        for (uint32_t step = 0; step + 1 < request.max_new_tokens; ++step) {
            if (request.eos_token_id.has_value() && input_token == request.eos_token_id.value()) {
                break;
            }

            enqueue_write(input_token);
            enqueue_read();
            const uint32_t output_token = await_read();
            generated_token_ids.push_back(output_token);
            token_stream.emit_token(request.request_id, generated_token_ids.size() - 1, output_token);
            input_token = output_token;
        }
    }

    void run_request(
        const RequestInput& request, std::vector<uint32_t>& generated_token_ids, TokenStream& token_stream) {
        if (request.max_new_tokens == 0) {
            throw std::runtime_error("max_new_tokens must be greater than zero");
        }

        const uint32_t first_generated_token = run_prefill(request, generated_token_ids, token_stream);
        run_decode(request, generated_token_ids, first_generated_token, token_stream);
        token_stream.emit_complete(request.request_id, generated_token_ids);
    }

    void write_over_socket(uint32_t token_id) { writer_socket_.write_token(token_id); }

    uint32_t read_over_socket() { return reader_socket_.read_token(); }

    H2DWriterSocket writer_socket_;
    D2HReaderSocket reader_socket_;
    std::thread writer_thread_;
    std::thread reader_thread_;
    std::mutex write_mutex_;
    std::condition_variable write_cv_;
    std::deque<uint32_t> pending_writes_;
    std::mutex read_request_mutex_;
    std::condition_variable read_request_cv_;
    size_t pending_read_count_ = 0;
    std::mutex read_result_mutex_;
    std::condition_variable read_result_cv_;
    std::deque<uint32_t> read_results_;
    bool busy_ = false;
    bool threads_started_ = false;
    std::atomic<bool> stop_requested_{false};
};

PipelineManager::PipelineManager(
    std::string h2d_socket_id, std::string d2h_socket_id, uint32_t page_size_bytes, uint32_t connect_timeout_ms) :
    impl_(std::make_unique<Impl>(
        std::move(h2d_socket_id), std::move(d2h_socket_id), page_size_bytes, connect_timeout_ms)) {}

PipelineManager::~PipelineManager() = default;

void PipelineManager::start() { impl_->start(); }

void PipelineManager::stop() { impl_->stop(); }

void PipelineManager::write_token(uint32_t token_id) { impl_->write_token(token_id); }

uint32_t PipelineManager::read_token() { return impl_->read_token(); }

void PipelineManager::write_over_socket(uint32_t token_id) { impl_->write_over_socket(token_id); }

uint32_t PipelineManager::read_over_socket() { return impl_->read_over_socket(); }

void PipelineManager::run_one_shot(PipelineManagerRequest& request, std::ostream& output_stream) {
    impl_->run_one_shot(request, output_stream);
}

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
