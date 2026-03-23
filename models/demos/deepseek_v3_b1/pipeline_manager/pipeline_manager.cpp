// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager.hpp"

#include <algorithm>
#include <atomic>
#include <thread>

#include "models/demos/deepseek_v3_b1/pipeline_manager/bounded_queue.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/decode_staging.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/free_id_pool.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_interface.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/prefill_queue.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/user_table.hpp"

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
        pipeline.shutdown();
        if (writer_thread.joinable()) {
            writer_thread.join();
        }
        if (reader_thread.joinable()) {
            reader_thread.join();
        }
    }

    // ========================================================================
    //  Writer Thread — hot path
    //  Priority: 1) decode tokens  2) prefill tokens (chunked)
    // ========================================================================

    void writer_loop() {
        while (running.load(std::memory_order_acquire)) {
            // --- Priority 1: Decode tokens ---
            int uid;
            int32_t tok;
            int32_t pos;
            if (decode_staging.try_pop(uid, tok, pos)) {
                if (cancel_pending.is_set(uid)) {
                    maybe_finalize_cleanup(uid);
                    continue;
                }

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
                user_table.in_flight_count[uid].fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // --- Priority 2: Prefill tokens (chunked round-robin) ---
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
                    user_table.state[pfuid].store(UserState::DECODE, std::memory_order_release);
                    prompt_table.clear(pfuid);
                    continue;
                }

                bool is_last = (cur_pos == prompt_len - 1);

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
                user_table.in_flight_count[pfuid].fetch_add(1, std::memory_order_relaxed);
                user_table.prefill_pos[pfuid] = cur_pos + 1;
                user_table.current_position[pfuid] = cur_pos + 1;
                user_table.prefill_chunk_remaining[pfuid] = chunk_rem - 1;
                continue;
            }

            // No work
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

            output_queue.try_push(OutputMessage{
                .user_id = static_cast<int32_t>(uid),
                .token_id = tok,
                .is_eos = is_eos,
                .is_complete = is_complete,
                .tokens_generated = user_table.tokens_generated[uid],
            });

            if (is_complete) {
                user_table.state[uid].store(UserState::COMPLETE, std::memory_order_release);
            } else {
                // Decode loopback: stage token for re-injection by Writer.
                // current_position is the next free KV slot.
                int32_t next_pos = user_table.current_position[uid];
                decode_staging.stage(uid, tok, next_pos);
                user_table.current_position[uid] = next_pos + 1;
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
                    response_queue.try_push(PMResponse{
                        .request_id = req.request_id,
                        .user_id = static_cast<int32_t>(uid),
                        .error_code = (uid < 0) ? 1 : 0,
                    });
                    break;
                }

                case RequestType::SUBMIT: {
                    int uid = req.user_id;
                    prompt_table.store(uid, req.tokens.data(), req.token_count);
                    user_table.state[uid].store(UserState::PREFILL, std::memory_order_release);
                    user_table.current_position[uid] = 0;
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
                    int start_pos = user_table.current_position[uid];
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
                    prefill_queue.remove(uid);
                    maybe_finalize_cleanup(uid);
                    break;
                }
            }
        }
    }

    // Safe to call from any thread (writer, reader, or API handler).
    // cancel_pending stays set until the slot is re-allocated — this prevents
    // the writer from injecting stale decode staging entries after the uid
    // is freed. All operations here are idempotent, so concurrent calls from
    // multiple threads are harmless.
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

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
