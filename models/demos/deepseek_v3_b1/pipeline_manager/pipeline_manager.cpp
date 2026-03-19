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
    struct PendingWrite {
        std::string request_id;
        uint32_t token_id = 0;
    };

    struct PendingRead {
        std::string request_id;
    };

    struct ReadResult {
        std::string request_id;
        uint32_t token_id = 0;
    };

    Impl(std::string h2d_socket_id, std::string d2h_socket_id, uint32_t page_size_bytes, uint32_t connect_timeout_ms) :
        writer_socket_(std::move(h2d_socket_id), page_size_bytes, connect_timeout_ms),
        reader_socket_(std::move(d2h_socket_id), page_size_bytes, connect_timeout_ms) {}

    ~Impl() { stop_threads(); }

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
        stop_requested_.store(false, std::memory_order_release);
        writer_thread_ = std::thread([this]() { writer_loop(); });
        reader_thread_ = std::thread([this]() { reader_loop(); });
    }

    void stop_threads() {
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
    }

    void clear_queues() {
        {
            std::lock_guard<std::mutex> write_lock(write_mutex_);
            pending_writes_.clear();
        }
        {
            std::lock_guard<std::mutex> read_request_lock(read_request_mutex_);
            pending_reads_.clear();
        }
        {
            std::lock_guard<std::mutex> read_result_lock(read_result_mutex_);
            read_results_.clear();
        }
    }

    void writer_loop() {
        while (true) {
            PendingWrite pending_write;
            {
                std::unique_lock<std::mutex> lock(write_mutex_);
                write_cv_.wait(lock, [this]() {
                    return stop_requested_.load(std::memory_order_acquire) || !pending_writes_.empty();
                });
                if (stop_requested_.load(std::memory_order_acquire) && pending_writes_.empty()) {
                    break;
                }
                pending_write = std::move(pending_writes_.front());
                pending_writes_.pop_front();
            }
            writer_socket_.write_token(pending_write.token_id);
        }
        writer_socket_.barrier();
    }

    void reader_loop() {
        while (true) {
            PendingRead pending_read;
            {
                std::unique_lock<std::mutex> lock(read_request_mutex_);
                read_request_cv_.wait(lock, [this]() {
                    return stop_requested_.load(std::memory_order_acquire) || !pending_reads_.empty();
                });
                if (stop_requested_.load(std::memory_order_acquire) && pending_reads_.empty()) {
                    break;
                }
                pending_read = std::move(pending_reads_.front());
                pending_reads_.pop_front();
            }

            ReadResult result{
                .request_id = pending_read.request_id,
                .token_id = reader_socket_.read_token(),
            };
            {
                std::lock_guard<std::mutex> lock(read_result_mutex_);
                read_results_.push_back(std::move(result));
            }
            read_result_cv_.notify_one();
        }
        reader_socket_.barrier();
    }

    void enqueue_write(const std::string& request_id, uint32_t token_id) {
        {
            std::lock_guard<std::mutex> lock(write_mutex_);
            pending_writes_.push_back(PendingWrite{.request_id = request_id, .token_id = token_id});
        }
        write_cv_.notify_one();
    }

    void enqueue_read(const std::string& request_id) {
        {
            std::lock_guard<std::mutex> lock(read_request_mutex_);
            pending_reads_.push_back(PendingRead{.request_id = request_id});
        }
        read_request_cv_.notify_one();
    }

    uint32_t await_read(const std::string& request_id) {
        std::unique_lock<std::mutex> lock(read_result_mutex_);
        read_result_cv_.wait(lock, [this, &request_id]() {
            return std::any_of(read_results_.begin(), read_results_.end(), [&request_id](const ReadResult& result) {
                return result.request_id == request_id;
            });
        });

        const auto result_it =
            std::find_if(read_results_.begin(), read_results_.end(), [&request_id](const ReadResult& result) {
                return result.request_id == request_id;
            });
        const uint32_t token_id = result_it->token_id;
        read_results_.erase(result_it);
        return token_id;
    }

    uint32_t run_prefill(
        const RequestInput& request, std::vector<uint32_t>& generated_token_ids, TokenStream& token_stream) {
        if (request.prompt_token_ids.empty()) {
            throw std::runtime_error("Prompt token list must not be empty");
        }

        uint32_t last_output_token = 0;
        for (uint32_t prompt_token : request.prompt_token_ids) {
            enqueue_write(request.request_id, prompt_token);
            enqueue_read(request.request_id);
            last_output_token = await_read(request.request_id);
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

            enqueue_write(request.request_id, input_token);
            enqueue_read(request.request_id);
            const uint32_t output_token = await_read(request.request_id);
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

    H2DWriterSocket writer_socket_;
    D2HReaderSocket reader_socket_;
    std::thread writer_thread_;
    std::thread reader_thread_;
    std::mutex write_mutex_;
    std::condition_variable write_cv_;
    std::deque<PendingWrite> pending_writes_;
    std::mutex read_request_mutex_;
    std::condition_variable read_request_cv_;
    std::deque<PendingRead> pending_reads_;
    std::mutex read_result_mutex_;
    std::condition_variable read_result_cv_;
    std::deque<ReadResult> read_results_;
    bool busy_ = false;
    std::atomic<bool> stop_requested_{false};
};

PipelineManager::PipelineManager(
    std::string h2d_socket_id, std::string d2h_socket_id, uint32_t page_size_bytes, uint32_t connect_timeout_ms) :
    impl_(std::make_unique<Impl>(
        std::move(h2d_socket_id), std::move(d2h_socket_id), page_size_bytes, connect_timeout_ms)) {}

PipelineManager::~PipelineManager() = default;

void PipelineManager::run_one_shot(PipelineManagerRequest& request, std::ostream& output_stream) {
    impl_->run_one_shot(request, output_stream);
}

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
