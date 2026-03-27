// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pipeline_manager/pipeline_manager.hpp"

#include <algorithm>
#include <atomic>
#include <thread>
#include <iostream>

#include "bounded_queue.hpp"
#include "decode_staging.hpp"
#include "free_id_pool.hpp"
#include "pipeline_manager/pipeline_interface.hpp"
#include "prefill_queue.hpp"
#include "user_table.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

struct PipelineManager::Impl {
    PipelineInterface& pipeline;
    int chunk_size;

    FreeIdPool free_ids;
    UserTable user_table;
    PromptTable prompt_table;
    CancelBitmap cancel_pending;
    DecodeStaging decode_staging;
    PrefillQueue prefill_queue;

    BoundedQueue<ISRequest, MAX_USERS * 2> request_queue;
    BoundedQueue<PMResponse, MAX_USERS> response_queue;
    BoundedQueue<OutputMessage, MAX_USERS * 256> output_queue;

    std::thread writer_thread;
    std::thread reader_thread;
    std::atomic<bool> running{false};

    Impl(PipelineInterface& p, int cs) : pipeline(p), chunk_size(cs) {}

    ~Impl() { stop(); }

    void start() {
        if (running.load(std::memory_order_acquire)) {
            return;
        }
        running.store(true, std::memory_order_release);
        writer_thread = std::thread([this] { writer_loop(); });
        reader_thread = std::thread([this] { reader_loop(); });
    }

    void stop() {
        if (!running.load(std::memory_order_acquire)) {
            return;
        }

        running.store(false, std::memory_order_release);
        pipeline.request_stop();

        std::cout << "Start joining writer thread" << std::endl;
        if (writer_thread.joinable()) {
            writer_thread.join();
        }
        if (reader_thread.joinable()) {
            reader_thread.join();
        }

        std::cout << "Start shutting down pipeline" << std::endl;
        pipeline.shutdown();
        std::cout << "Finished shutting down pipeline" << std::endl;
    }

    // ========================================================================
    //  Writer Thread — hot path
    //  Priority: 1) decode tokens  2) prefill tokens (chunked)
    // ========================================================================

    void writer_loop() {
        while (running.load(std::memory_order_acquire)) {
            int uid;
            int32_t tok;
            int32_t pos;
            if (decode_staging.try_pop(uid, tok, pos)) {
                if (cancel_pending.is_set(uid)) {
                    maybe_finalize_cleanup(uid);
                    continue;
                }

                user_table.in_flight_count[uid].fetch_add(1, std::memory_order_release);
                pipeline.inject(InjectDescriptor{
                    .user_id = static_cast<int32_t>(uid),
                    .token_id = tok,
                    .position = pos,
                    .mode = TokenMode::DECODE,
                    .spec_flag = false,
                    .temperature = user_table.temperature[uid],
                    .top_p = user_table.top_p[uid],
                    .top_k = user_table.top_k[uid],
                });
                continue;
            }

            int pfuid;
            if (prefill_queue.try_front(pfuid)) {
                if (cancel_pending.is_set(pfuid)) {
                    prefill_queue.pop_front();
                    maybe_finalize_cleanup(pfuid);
                    continue;
                }

                int prompt_len = prompt_table.get_length(pfuid);
                int cur_pos = user_table.prefill_pos[pfuid];
                int chunk_rem = user_table.prefill_chunk_remaining[pfuid];

                if (chunk_rem <= 0) {
                    user_table.prefill_chunk_remaining[pfuid] = chunk_size;
                    prefill_queue.rotate();
                    continue;
                }

                if (cur_pos >= prompt_len) {
                    prefill_queue.pop_front();
                    prompt_table.clear(pfuid);
                    continue;
                }

                bool is_last = (cur_pos == prompt_len - 1);

                user_table.prefill_pos[pfuid] = cur_pos + 1;
                user_table.current_position[pfuid].store(cur_pos + 1, std::memory_order_release);
                user_table.prefill_chunk_remaining[pfuid] = chunk_rem - 1;

                user_table.in_flight_count[pfuid].fetch_add(1, std::memory_order_release);
                pipeline.inject(InjectDescriptor{
                    .user_id = static_cast<int32_t>(pfuid),
                    .token_id = prompt_table.get_token(pfuid, cur_pos),
                    .position = cur_pos,
                    .mode = is_last ? TokenMode::DECODE : TokenMode::PREFILL,
                    .spec_flag = false,
                    .temperature = user_table.temperature[pfuid],
                    .top_p = user_table.top_p[pfuid],
                    .top_k = user_table.top_k[pfuid],
                });

                if (is_last) {
                    prefill_queue.pop_front();
                    user_table.state[pfuid].store(UserState::DECODE, std::memory_order_release);
                    prompt_table.clear(pfuid);
                }
                continue;
            }

            std::this_thread::yield();
        }
    }

    // ========================================================================
    //  Reader Thread — hot path
    //  Processes pipeline results: decode loopback, completion, cancellation.
    // ========================================================================

    void reader_loop() {
        while (running.load(std::memory_order_acquire)) {
            ResultDescriptor result = pipeline.read_result();

            int uid = result.user_id;
            if (uid < 0) {
                break;
            }

            user_table.in_flight_count[uid].fetch_sub(1, std::memory_order_relaxed);

            if (cancel_pending.is_set(uid)) {
                maybe_finalize_cleanup(uid);
                continue;
            }

            if (!result.sampled) {
                continue;
            }

            int32_t tok = result.actual_token;
            user_table.tokens_generated[uid]++;

            bool is_eos = (tok == EOS_TOKEN);
            bool is_max = (user_table.tokens_generated[uid] >= user_table.max_new_tokens[uid]);
            bool is_complete = is_eos || is_max;

            OutputMessage msg{
                .user_id = static_cast<int32_t>(uid),
                .token_id = tok,
                .is_eos = is_eos,
                .is_complete = is_complete,
                .tokens_generated = user_table.tokens_generated[uid],
                .generation = decode_staging.generation[uid].load(std::memory_order_relaxed),
            };
            while (!output_queue.try_push(msg)) {
                std::this_thread::yield();
            }

            if (is_complete) {
                user_table.state[uid].store(UserState::COMPLETE, std::memory_order_release);
            } else {
                int32_t next_pos = user_table.current_position[uid].load(std::memory_order_acquire);
                decode_staging.stage(uid, tok, next_pos);
                user_table.current_position[uid].store(next_pos + 1, std::memory_order_release);
            }
        }
    }

    // ========================================================================
    //  API Request Handler — called by main thread via tick()
    // ========================================================================

    void handle_api_requests() {
        ISRequest req;
        while (request_queue.try_pop(req)) {
            switch (req.type) {
                case RequestType::ALLOCATE: {
                    int uid = free_ids.allocate();
                    if (uid >= 0) {
                        user_table.reset(uid);
                        cancel_pending.clear(uid);
                    }
                    PMResponse resp{
                        .request_id = req.request_id,
                        .user_id = static_cast<int32_t>(uid),
                        .error_code = (uid < 0) ? 1 : 0,
                    };
                    while (!response_queue.try_push(resp)) {
                        std::this_thread::yield();
                    }
                    break;
                }

                case RequestType::SUBMIT: {
                    int uid = req.user_id;
                    prompt_table.store(uid, req.tokens.data(), req.token_count);
                    user_table.state[uid].store(UserState::PREFILL, std::memory_order_release);
                    user_table.current_position[uid].store(0, std::memory_order_relaxed);
                    user_table.prefill_pos[uid] = 0;
                    user_table.max_new_tokens[uid] = req.max_new_tokens;
                    user_table.tokens_generated[uid] = 0;
                    user_table.in_flight_count[uid].store(0, std::memory_order_relaxed);
                    user_table.prefill_chunk_remaining[uid] = chunk_size;
                    user_table.spec_decode_enabled[uid] = req.spec_decode;
                    user_table.temperature[uid] = req.temperature;
                    user_table.top_p[uid] = req.top_p;
                    user_table.top_k[uid] = req.top_k;
                    prefill_queue.push(uid);
                    break;
                }

                case RequestType::CONTINUE: {
                    int uid = req.user_id;
                    int start_pos = user_table.current_position[uid].load(std::memory_order_relaxed);
                    prompt_table.store(uid, req.tokens.data(), req.token_count, start_pos);

                    user_table.state[uid].store(UserState::PREFILL, std::memory_order_release);
                    user_table.prefill_pos[uid] = start_pos;
                    user_table.max_new_tokens[uid] = req.max_new_tokens;
                    user_table.tokens_generated[uid] = 0;
                    user_table.prefill_chunk_remaining[uid] = chunk_size;
                    user_table.temperature[uid] = req.temperature;
                    user_table.top_p[uid] = req.top_p;
                    user_table.top_k[uid] = req.top_k;
                    prefill_queue.push(uid);
                    break;
                }

                case RequestType::CANCEL: {
                    int uid = req.user_id;
                    cancel_pending.mark(uid);
                    decode_staging.advance_generation(uid);
                    prefill_queue.remove(uid);
                    maybe_finalize_cleanup(uid);
                    break;
                }
            }
        }
    }

    // Safe to call from any thread (writer, reader, or API handler).
    void maybe_finalize_cleanup(int uid) {
        if (cancel_pending.is_set(uid) && user_table.in_flight_count[uid].load(std::memory_order_acquire) == 0) {
            pipeline.reset_kv(uid);
            user_table.state[uid].store(UserState::INACTIVE, std::memory_order_release);
            free_ids.free(uid);
        }
    }
};

// ============================================================================
//  PipelineManager public API — forwards to Impl
// ============================================================================

PipelineManager::PipelineManager(PipelineInterface& pipeline, int chunk_size) :
    impl_(std::make_unique<Impl>(pipeline, chunk_size)) {}

PipelineManager::~PipelineManager() = default;

void PipelineManager::start() { impl_->start(); }

void PipelineManager::stop() { impl_->stop(); }

bool PipelineManager::push_request(const ISRequest& request) { return impl_->request_queue.try_push(request); }

bool PipelineManager::try_pop_response(PMResponse& response) { return impl_->response_queue.try_pop(response); }

bool PipelineManager::try_pop_output(OutputMessage& output) { return impl_->output_queue.try_pop(output); }

void PipelineManager::tick() { impl_->handle_api_requests(); }

UserState PipelineManager::get_user_state(int user_id) const {
    return impl_->user_table.state[user_id].load(std::memory_order_acquire);
}

int32_t PipelineManager::get_in_flight_count(int user_id) const {
    return impl_->user_table.in_flight_count[user_id].load(std::memory_order_acquire);
}

int32_t PipelineManager::get_tokens_generated(int user_id) const { return impl_->user_table.tokens_generated[user_id]; }

int32_t PipelineManager::get_max_new_tokens(int user_id) const { return impl_->user_table.max_new_tokens[user_id]; }

int32_t PipelineManager::get_current_position(int user_id) const {
    return impl_->user_table.current_position[user_id].load(std::memory_order_acquire);
}

bool PipelineManager::get_cancel_pending(int user_id) const { return impl_->cancel_pending.is_set(user_id); }

int PipelineManager::get_decode_staging_size() const { return impl_->decode_staging.fifo.size(); }

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
