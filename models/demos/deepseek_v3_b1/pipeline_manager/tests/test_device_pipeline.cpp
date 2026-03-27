// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Integration test: PipelineManager + SocketPipeline + on-device loopback kernel.
// Single-process: fixture creates device, sockets, launches kernel, then runs
// PipelineManager against the SocketPipeline connector.

#include <cstdlib>
#include <gtest/gtest.h>

#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/host_api.hpp>

#include "pipeline_manager/pipeline_manager.hpp"
#include "pipeline_manager/pipeline_manager_types.hpp"
#include "pipeline_manager/socket_pipeline.hpp"
#include "pipeline_manager/wire_format.hpp"

namespace pm = models::demos::deepseek_v3_b1::pipeline_manager;

using pm::ISRequest;
using pm::MAX_SEQ_LEN;
using pm::MAX_USERS;
using pm::OutputMessage;
using pm::PAGE_SIZE_BYTES;
using pm::PMResponse;
using pm::RequestType;

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace {

ISRequest make_allocate(int32_t request_id) {
    ISRequest req{};
    req.type = RequestType::ALLOCATE;
    req.request_id = request_id;
    return req;
}

ISRequest make_submit(int32_t request_id, int32_t user_id, const std::vector<int32_t>& prompt, int32_t max_new_tokens) {
    ISRequest req{};
    req.type = RequestType::SUBMIT;
    req.request_id = request_id;
    req.user_id = user_id;
    req.token_count = static_cast<int32_t>(prompt.size());
    for (size_t i = 0; i < prompt.size() && static_cast<int>(i) < MAX_SEQ_LEN; i++) {
        req.tokens[i] = prompt[i];
    }
    req.max_new_tokens = max_new_tokens;
    return req;
}

ISRequest make_continue(
    int32_t request_id, int32_t user_id, const std::vector<int32_t>& prompt, int32_t max_new_tokens) {
    ISRequest req{};
    req.type = RequestType::CONTINUE;
    req.request_id = request_id;
    req.user_id = user_id;
    req.token_count = static_cast<int32_t>(prompt.size());
    for (size_t i = 0; i < prompt.size() && static_cast<int>(i) < MAX_SEQ_LEN; i++) {
        req.tokens[i] = prompt[i];
    }
    req.max_new_tokens = max_new_tokens;
    return req;
}

ISRequest make_cancel(int32_t request_id, int32_t user_id) {
    ISRequest req{};
    req.type = RequestType::CANCEL;
    req.request_id = request_id;
    req.user_id = user_id;
    return req;
}

static constexpr int MAX_POLL_ITERATIONS = 100000000;

template <typename Pred>
bool poll_until(pm::PipelineManager& mgr, Pred pred, std::vector<OutputMessage>& outputs) {
    for (int i = 0; i < MAX_POLL_ITERATIONS; i++) {
        mgr.tick();
        OutputMessage out;
        while (mgr.try_pop_output(out)) {
            outputs.push_back(out);
        }
        if (pred()) {
            // Final drain: the reader does push(msg) then store(COMPLETE).
            // If the predicate saw COMPLETE, the push already happened, so
            // the message is in the queue — capture it now.
            while (mgr.try_pop_output(out)) {
                outputs.push_back(out);
            }
            return true;
        }
        std::this_thread::yield();
    }
    return false;
}

}  // namespace

class DevicePipelineFixture : public ::testing::Test {
protected:
    void SetUp() override {
        if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
            GTEST_SKIP() << "Mesh-device pipeline test requires fast dispatch (unset TT_METAL_SLOW_DISPATCH_MODE).";
        }
        mesh_device_ = MeshDevice::create(MeshDeviceConfig{MeshShape{1, 1}});
        ASSERT_NE(mesh_device_, nullptr);

        socket_core_ = MeshCoreCoord{MeshCoordinate(0, 0), CoreCoord(0, 0)};

        h2d_socket_ =
            std::make_unique<H2DSocket>(mesh_device_, socket_core_, BufferType::L1, FIFO_SIZE, H2DMode::HOST_PUSH);
        h2d_socket_->export_descriptor(H2D_SOCKET_ID);

        d2h_socket_ = std::make_unique<D2HSocket>(mesh_device_, socket_core_, FIFO_SIZE);
        d2h_socket_->export_descriptor(D2H_SOCKET_ID);

        auto program = CreateProgram();

        auto output_cb_index = tt::CBIndex::c_0;
        auto output_cb_config = CircularBufferConfig(PAGE_SIZE_BYTES, {{output_cb_index, tt::DataFormat::UInt32}})
                                    .set_page_size(output_cb_index, PAGE_SIZE_BYTES);
        CreateCircularBuffer(program, socket_core_.core_coord, output_cb_config);

        CreateKernel(
            program,
            "models/demos/deepseek_v3_b1/pipeline_manager/kernels/pipeline_loopback.cpp",
            socket_core_.core_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(h2d_socket_->get_config_buffer_address()),
                    static_cast<uint32_t>(d2h_socket_->get_config_buffer_address()),
                    PAGE_SIZE_BYTES,
                    output_cb_index,
                }});

        auto mesh_workload = MeshWorkload();
        mesh_workload.add_program(MeshCoordinateRange(socket_core_.device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);

        pipeline_ = std::make_unique<pm::SocketPipeline>(H2D_SOCKET_ID, D2H_SOCKET_ID, CONNECT_TIMEOUT_MS);
        mgr_ = std::make_unique<pm::PipelineManager>(*pipeline_, chunk_size_);
        mgr_->start();
    }

    void TearDown() override {
        if (mgr_) {
            mgr_->stop();
            mgr_.reset();
        }
        pipeline_.reset();

        if (mesh_device_) {
            Finish(mesh_device_->mesh_command_queue());
        }

        h2d_socket_.reset();
        d2h_socket_.reset();

        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    static constexpr uint32_t FIFO_SIZE = 1024;
    static constexpr uint32_t CONNECT_TIMEOUT_MS = 30000;
    static constexpr const char* H2D_SOCKET_ID = "pm_test_h2d";
    static constexpr const char* D2H_SOCKET_ID = "pm_test_d2h";

    int chunk_size_ = pm::DEFAULT_CHUNK_SIZE;

    std::shared_ptr<MeshDevice> mesh_device_;
    MeshCoreCoord socket_core_;
    std::unique_ptr<H2DSocket> h2d_socket_;
    std::unique_ptr<D2HSocket> d2h_socket_;
    std::unique_ptr<pm::SocketPipeline> pipeline_;
    std::unique_ptr<pm::PipelineManager> mgr_;
};

class DevicePipelineChunkedFixture : public DevicePipelineFixture {
protected:
    DevicePipelineChunkedFixture() { chunk_size_ = 32; }
};

// =============================================================================
//  Single user end-to-end through device: allocate, submit, complete.
//  Same logic as the MockPipeline test — verifies the full
//  host → H2D socket → device kernel → D2H socket → host round trip.
// =============================================================================
TEST_F(DevicePipelineFixture, SingleUserAllocateSubmitComplete) {
    ASSERT_TRUE(mgr_->push_request(make_allocate(1)));
    mgr_->tick();

    PMResponse resp{};
    ASSERT_TRUE(mgr_->try_pop_response(resp));
    EXPECT_EQ(resp.request_id, 1);
    EXPECT_EQ(resp.error_code, 0);
    EXPECT_GE(resp.user_id, 0);
    EXPECT_LT(resp.user_id, MAX_USERS);
    int32_t uid = resp.user_id;

    std::vector<int32_t> prompt = {100, 200, 300};
    int32_t max_new = 5;
    ASSERT_TRUE(mgr_->push_request(make_submit(2, uid, prompt, max_new)));
    mgr_->tick();

    std::vector<OutputMessage> outputs;
    bool done = poll_until(*mgr_, [&] { return !outputs.empty() && outputs.back().is_complete; }, outputs);

    ASSERT_TRUE(done) << "Timed out waiting for completion";
    EXPECT_EQ(static_cast<int>(outputs.size()), max_new);

    for (const auto& o : outputs) {
        EXPECT_EQ(o.user_id, uid);
    }

    for (int i = 0; i < static_cast<int>(outputs.size()); i++) {
        EXPECT_EQ(outputs[i].tokens_generated, i + 1);
    }

    for (int i = 0; i < static_cast<int>(outputs.size()) - 1; i++) {
        EXPECT_FALSE(outputs[i].is_complete);
    }
    EXPECT_TRUE(outputs.back().is_complete);

    EXPECT_EQ(outputs[0].token_id, 301);
    EXPECT_EQ(outputs[1].token_id, 302);
    EXPECT_EQ(outputs[2].token_id, 303);
    EXPECT_EQ(outputs[3].token_id, 304);
    EXPECT_EQ(outputs[4].token_id, 305);
}

// =============================================================================
//  Test 2: Allocation exhaustion — 64 succeed, 65th fails
// =============================================================================
TEST_F(DevicePipelineFixture, AllocationExhaustion) {
    std::set<int32_t> allocated_ids;

    for (int i = 0; i < MAX_USERS; i++) {
        ASSERT_TRUE(mgr_->push_request(make_allocate(i)));
    }
    mgr_->tick();

    for (int i = 0; i < MAX_USERS; i++) {
        PMResponse resp{};
        ASSERT_TRUE(mgr_->try_pop_response(resp));
        EXPECT_EQ(resp.error_code, 0) << "Allocation " << i << " failed";
        EXPECT_GE(resp.user_id, 0);
        EXPECT_LT(resp.user_id, MAX_USERS);
        allocated_ids.insert(resp.user_id);
    }

    EXPECT_EQ(static_cast<int>(allocated_ids.size()), MAX_USERS);

    ASSERT_TRUE(mgr_->push_request(make_allocate(MAX_USERS)));
    mgr_->tick();

    PMResponse fail_resp{};
    ASSERT_TRUE(mgr_->try_pop_response(fail_resp));
    EXPECT_EQ(fail_resp.user_id, -1);
    EXPECT_NE(fail_resp.error_code, 0);
}

// =============================================================================
//  Test 3: Cancel mid-decode frees user ID for re-allocation
// =============================================================================
TEST_F(DevicePipelineFixture, CancelMidDecodeFreesUserIdForReallocation) {
    ASSERT_TRUE(mgr_->push_request(make_allocate(1)));
    mgr_->tick();

    PMResponse resp{};
    ASSERT_TRUE(mgr_->try_pop_response(resp));
    ASSERT_EQ(resp.error_code, 0);
    int32_t uid = resp.user_id;

    std::vector<int32_t> prompt = {10, 20, 30};
    ASSERT_TRUE(mgr_->push_request(make_submit(2, uid, prompt, 1000000)));
    mgr_->tick();

    std::vector<OutputMessage> outputs;
    bool got_output = poll_until(*mgr_, [&] { return outputs.size() >= 3; }, outputs);
    ASSERT_TRUE(got_output) << "Timed out waiting for decode output tokens";
    EXPECT_EQ(mgr_->get_user_state(uid), pm::UserState::DECODE);

    size_t tokens_before_cancel = outputs.size();

    std::mt19937 rng(std::random_device{}());
    auto delay_us = std::uniform_int_distribution<int>(0, 2000)(rng);
    std::this_thread::sleep_for(std::chrono::microseconds(delay_us));

    ASSERT_TRUE(mgr_->push_request(make_cancel(3, uid)));
    mgr_->tick();

    bool cleaned_up = poll_until(*mgr_, [&] { return mgr_->get_user_state(uid) == pm::UserState::INACTIVE; }, outputs);
    ASSERT_TRUE(cleaned_up) << "Timed out waiting for cancel cleanup";

    size_t tokens_after_cancel = outputs.size() - tokens_before_cancel;
    std::cout << "[  INFO   ] Random delay before cancel: " << delay_us << " us" << std::endl;
    std::cout << "[  INFO   ] Tokens before cancel: " << tokens_before_cancel
              << ", drained after cancel: " << tokens_after_cancel << ", total: " << outputs.size() << std::endl;

    ASSERT_TRUE(mgr_->push_request(make_allocate(4)));
    mgr_->tick();

    PMResponse realloc_resp{};
    ASSERT_TRUE(mgr_->try_pop_response(realloc_resp));
    EXPECT_EQ(realloc_resp.error_code, 0);
    EXPECT_GE(realloc_resp.user_id, 0);
    EXPECT_LT(realloc_resp.user_id, MAX_USERS);
    int32_t uid2 = realloc_resp.user_id;

    ASSERT_TRUE(mgr_->push_request(make_cancel(5, uid2)));
    mgr_->tick();

    bool cleaned_up2 =
        poll_until(*mgr_, [&] { return mgr_->get_user_state(uid2) == pm::UserState::INACTIVE; }, outputs);
    ASSERT_TRUE(cleaned_up2) << "Timed out waiting for cancel-before-submit cleanup";

    ASSERT_TRUE(mgr_->push_request(make_allocate(6)));
    mgr_->tick();

    PMResponse realloc_resp2{};
    ASSERT_TRUE(mgr_->try_pop_response(realloc_resp2));
    EXPECT_EQ(realloc_resp2.error_code, 0);
    EXPECT_GE(realloc_resp2.user_id, 0);
    EXPECT_LT(realloc_resp2.user_id, MAX_USERS);
}

// =============================================================================
//  Test 4: Max users stress — randomized prompts and token counts, all complete.
//  Verifies deterministic token output through the device loopback kernel.
// =============================================================================
TEST_F(DevicePipelineFixture, MultipleUsersStressTest) {
    static constexpr int NUM_USERS = MAX_USERS;
    static constexpr int MIN_PROMPT_LEN = 10;
    static constexpr int MAX_PROMPT_LEN = 500;
    static constexpr int MIN_NEW_TOKENS = 10;
    static constexpr int MAX_NEW_TOKENS = 100;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> prompt_len_dist(MIN_PROMPT_LEN, MAX_PROMPT_LEN);
    std::uniform_int_distribution<int> token_count_dist(MIN_NEW_TOKENS, MAX_NEW_TOKENS);

    struct UserSpec {
        int32_t uid = -1;
        int32_t prompt_len = 0;
        int32_t max_new_tokens = 0;
    };

    int32_t req_id = 0;
    std::vector<UserSpec> users(NUM_USERS);

    for (int i = 0; i < NUM_USERS; i++) {
        ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
    }
    mgr_->tick();
    for (int i = 0; i < NUM_USERS; i++) {
        PMResponse resp{};
        ASSERT_TRUE(mgr_->try_pop_response(resp));
        ASSERT_EQ(resp.error_code, 0);
        users[i].uid = resp.user_id;
    }

    int total_expected = 0;
    for (int i = 0; i < NUM_USERS; i++) {
        users[i].prompt_len = prompt_len_dist(rng);
        users[i].max_new_tokens = token_count_dist(rng);
        total_expected += users[i].max_new_tokens;

        std::vector<int32_t> prompt(users[i].prompt_len);
        int32_t base = users[i].uid * 10000;
        std::iota(prompt.begin(), prompt.end(), base);

        ASSERT_TRUE(mgr_->push_request(make_submit(req_id++, users[i].uid, prompt, users[i].max_new_tokens)));
    }
    mgr_->tick();

    std::cout << "[  INFO   ] " << NUM_USERS << " users, total expected tokens: " << total_expected << std::endl;

    std::vector<OutputMessage> outputs;
    auto count_complete = [&]() {
        std::set<int32_t> done;
        for (auto& o : outputs) {
            if (o.is_complete) {
                done.insert(o.user_id);
            }
        }
        return static_cast<int>(done.size());
    };

    bool all_done = poll_until(*mgr_, [&] { return count_complete() == NUM_USERS; }, outputs);
    ASSERT_TRUE(all_done) << "Timed out: only " << count_complete() << "/" << NUM_USERS << " completed";

    std::map<int32_t, std::vector<OutputMessage>> per_user;
    for (auto& o : outputs) {
        per_user[o.user_id].push_back(o);
    }

    for (auto& u : users) {
        auto it = per_user.find(u.uid);
        ASSERT_NE(it, per_user.end()) << "No output for user " << u.uid;
        auto& user_outputs = it->second;

        EXPECT_EQ(static_cast<int32_t>(user_outputs.size()), u.max_new_tokens)
            << "User " << u.uid << " (prompt_len=" << u.prompt_len << ")"
            << " expected " << u.max_new_tokens << " tokens, got " << user_outputs.size();

        int32_t base = u.uid * 10000;
        for (int i = 0; i < static_cast<int>(user_outputs.size()); i++) {
            EXPECT_EQ(user_outputs[i].tokens_generated, i + 1);
            EXPECT_EQ(user_outputs[i].user_id, u.uid);
            EXPECT_EQ(user_outputs[i].token_id, base + u.prompt_len + i)
                << "User " << u.uid << " token " << i << " mismatch";
        }

        EXPECT_TRUE(user_outputs.back().is_complete);
        EXPECT_EQ(mgr_->get_user_state(u.uid), pm::UserState::COMPLETE);
    }

    std::cout << "[  INFO   ] All " << NUM_USERS << " users completed, " << outputs.size() << " total output tokens"
              << std::endl;
}

// =============================================================================
//  Test 5: Chunked prefill — 8 users with identical long prompts produce
//  interleaved output (no user completes entirely before another starts).
//  Uses DevicePipelineChunkedFixture (chunk_size=32) so the writer
//  round-robins prefill tokens across users.
// =============================================================================
TEST_F(DevicePipelineChunkedFixture, ChunkedPrefillInterleaved) {
    static constexpr int NUM_USERS = 8;
    static constexpr int PROMPT_LEN = 2048;
    static constexpr int MAX_NEW = 30;

    int32_t req_id = 0;
    std::vector<int32_t> uids(NUM_USERS);

    for (int i = 0; i < NUM_USERS; i++) {
        ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
    }
    mgr_->tick();
    for (int i = 0; i < NUM_USERS; i++) {
        PMResponse resp{};
        ASSERT_TRUE(mgr_->try_pop_response(resp));
        ASSERT_EQ(resp.error_code, 0);
        uids[i] = resp.user_id;
    }

    for (int i = 0; i < NUM_USERS; i++) {
        int32_t base = uids[i] * 10000;
        std::vector<int32_t> prompt(PROMPT_LEN);
        std::iota(prompt.begin(), prompt.end(), base);
        ASSERT_TRUE(mgr_->push_request(make_submit(req_id++, uids[i], prompt, MAX_NEW)));
    }
    mgr_->tick();

    std::vector<OutputMessage> outputs;
    auto count_complete = [&]() {
        std::set<int32_t> done;
        for (auto& o : outputs) {
            if (o.is_complete) {
                done.insert(o.user_id);
            }
        }
        return static_cast<int>(done.size());
    };

    bool all_done = poll_until(*mgr_, [&] { return count_complete() == NUM_USERS; }, outputs);
    ASSERT_TRUE(all_done) << "Timed out: only " << count_complete() << "/" << NUM_USERS << " completed";

    std::map<int32_t, int> first_idx, last_idx;
    std::map<int32_t, std::vector<OutputMessage>> per_user;
    for (int i = 0; i < static_cast<int>(outputs.size()); i++) {
        int32_t uid = outputs[i].user_id;
        per_user[uid].push_back(outputs[i]);
        if (first_idx.find(uid) == first_idx.end()) {
            first_idx[uid] = i;
        }
        last_idx[uid] = i;
    }

    int interleave_violations = 0;
    for (int i = 0; i < NUM_USERS; i++) {
        for (int j = i + 1; j < NUM_USERS; j++) {
            int32_t a = uids[i], b = uids[j];
            bool overlaps = (first_idx[a] < last_idx[b]) && (first_idx[b] < last_idx[a]);
            if (!overlaps) {
                interleave_violations++;
                std::cout << "[  WARN   ] No interleave between user " << a << " [" << first_idx[a] << ".."
                          << last_idx[a] << "] and user " << b << " [" << first_idx[b] << ".." << last_idx[b] << "]"
                          << std::endl;
            }
        }
    }
    EXPECT_EQ(interleave_violations, 0) << interleave_violations << " user pairs had non-overlapping output ranges";

    for (int i = 0; i < NUM_USERS; i++) {
        int32_t uid = uids[i];
        auto& user_outputs = per_user[uid];

        EXPECT_EQ(static_cast<int>(user_outputs.size()), MAX_NEW)
            << "User " << uid << " expected " << MAX_NEW << " tokens, got " << user_outputs.size();

        int32_t base = uid * 10000;
        for (int t = 0; t < static_cast<int>(user_outputs.size()); t++) {
            EXPECT_EQ(user_outputs[t].tokens_generated, t + 1);
            EXPECT_EQ(user_outputs[t].user_id, uid);
            EXPECT_EQ(user_outputs[t].token_id, base + PROMPT_LEN + t)
                << "User " << uid << " token " << t << " mismatch";
        }

        EXPECT_TRUE(user_outputs.back().is_complete);
        EXPECT_EQ(mgr_->get_user_state(uid), pm::UserState::COMPLETE);
    }
}

// =============================================================================
//  Test 6: Multi-turn continue — incremental ramp to 64 users, each doing
//  multiple CONTINUE turns with longer prompts and generation counts.
//
//  Users are allocated in batches of 8.  Each batch submits, all active users
//  run to completion, then every active user issues a CONTINUE.  Repeats for
//  TURNS_PER_USER turns total.  Token values are verified per-user per-turn
//  using the loopback kernel (token_id + 1).
// =============================================================================
TEST_F(DevicePipelineFixture, MultiTurnContinue) {
    static constexpr int RAMP_BATCH = 8;
    static constexpr int TURNS_PER_USER = 4;
    static constexpr int MIN_PROMPT = 50;
    static constexpr int MAX_PROMPT = 200;
    static constexpr int MIN_NEW = 20;
    static constexpr int MAX_NEW = 80;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> prompt_dist(MIN_PROMPT, MAX_PROMPT);
    std::uniform_int_distribution<int> token_dist(MIN_NEW, MAX_NEW);

    int32_t req_id = 0;

    struct TurnSpec {
        int32_t prompt_len;
        int32_t max_new;
        int32_t token_base;
    };

    struct UserInfo {
        int32_t uid = -1;
        int slot_idx = -1;
        std::vector<TurnSpec> turns;
    };

    std::vector<UserInfo> users;
    std::vector<OutputMessage> all_outputs;

    auto wait_all_complete = [&](const char* label) {
        bool ok = poll_until(
            *mgr_,
            [&] {
                for (auto& u : users) {
                    if (mgr_->get_user_state(u.uid) != pm::UserState::COMPLETE) {
                        return false;
                    }
                }
                return true;
            },
            all_outputs);
        ASSERT_TRUE(ok) << label << ": timed out waiting for all users to complete";
    };

    // Incremental ramp: add RAMP_BATCH users per iteration until 64
    for (int batch_start = 0; batch_start < MAX_USERS; batch_start += RAMP_BATCH) {
        int batch_end = std::min(batch_start + RAMP_BATCH, static_cast<int>(MAX_USERS));
        int batch_size = batch_end - batch_start;

        for (int i = 0; i < batch_size; i++) {
            ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
        }
        mgr_->tick();

        for (int i = 0; i < batch_size; i++) {
            PMResponse resp{};
            ASSERT_TRUE(mgr_->try_pop_response(resp));
            ASSERT_EQ(resp.error_code, 0);

            UserInfo u;
            u.uid = resp.user_id;
            u.slot_idx = batch_start + i;
            users.push_back(u);
        }

        // Turn 1 (SUBMIT) for the new batch
        for (int i = batch_start; i < batch_end; i++) {
            auto& u = users[i];
            int32_t plen = prompt_dist(rng);
            int32_t max_new = token_dist(rng);
            int32_t base = u.uid * 100000;
            u.turns.push_back({plen, max_new, base});

            std::vector<int32_t> prompt(plen);
            std::iota(prompt.begin(), prompt.end(), base);
            ASSERT_TRUE(mgr_->push_request(make_submit(req_id++, u.uid, prompt, max_new)));
        }
        mgr_->tick();

        // Wait for ALL currently active users (including earlier batches
        // that may still be running a CONTINUE) to complete this turn.
        wait_all_complete("Ramp SUBMIT");

        // Issue CONTINUE for ALL active users that still have turns left.
        // Earlier batches are further ahead; they continue in lockstep.
        int turn_for_earlier = static_cast<int>(users[0].turns.size());
        if (turn_for_earlier < TURNS_PER_USER) {
            for (auto& u : users) {
                int next_turn = static_cast<int>(u.turns.size());
                if (next_turn >= TURNS_PER_USER) {
                    continue;
                }

                int32_t plen = prompt_dist(rng);
                int32_t max_new = token_dist(rng);
                int32_t base = u.uid * 100000 + next_turn * 10000;
                u.turns.push_back({plen, max_new, base});

                std::vector<int32_t> prompt(plen);
                std::iota(prompt.begin(), prompt.end(), base);
                ASSERT_TRUE(mgr_->push_request(make_continue(req_id++, u.uid, prompt, max_new)));
            }
            mgr_->tick();
        }
    }

    // Remaining CONTINUE turns until every user has completed TURNS_PER_USER
    for (;;) {
        bool any_remaining = false;
        for (auto& u : users) {
            if (static_cast<int>(u.turns.size()) < TURNS_PER_USER) {
                any_remaining = true;
                break;
            }
        }
        if (!any_remaining) {
            break;
        }

        wait_all_complete("Continue round");

        for (auto& u : users) {
            int next_turn = static_cast<int>(u.turns.size());
            if (next_turn >= TURNS_PER_USER) {
                continue;
            }

            int32_t plen = prompt_dist(rng);
            int32_t max_new = token_dist(rng);
            int32_t base = u.uid * 100000 + next_turn * 10000;
            u.turns.push_back({plen, max_new, base});

            std::vector<int32_t> prompt(plen);
            std::iota(prompt.begin(), prompt.end(), base);
            ASSERT_TRUE(mgr_->push_request(make_continue(req_id++, u.uid, prompt, max_new)));
        }
        mgr_->tick();
    }

    // Final wait
    wait_all_complete("Final round");

    // ---- Verification ----
    std::map<int32_t, std::vector<OutputMessage>> per_user;
    for (auto& o : all_outputs) {
        per_user[o.user_id].push_back(o);
    }

    int total_tokens_verified = 0;
    for (auto& u : users) {
        auto it = per_user.find(u.uid);
        ASSERT_NE(it, per_user.end()) << "No output for uid " << u.uid;
        auto& stream = it->second;

        // Split stream into turns at is_complete boundaries
        std::vector<std::vector<OutputMessage>> turn_outputs;
        std::vector<OutputMessage> buf;
        for (auto& o : stream) {
            buf.push_back(o);
            if (o.is_complete) {
                turn_outputs.push_back(std::move(buf));
                buf.clear();
            }
        }

        ASSERT_EQ(static_cast<int>(turn_outputs.size()), TURNS_PER_USER)
            << "uid " << u.uid << ": expected " << TURNS_PER_USER << " completed turns, got " << turn_outputs.size();

        for (int t = 0; t < TURNS_PER_USER; t++) {
            auto& turn = turn_outputs[t];
            auto& spec = u.turns[t];

            EXPECT_EQ(static_cast<int32_t>(turn.size()), spec.max_new) << "uid " << u.uid << " turn " << t;

            for (int i = 0; i < static_cast<int>(turn.size()); i++) {
                EXPECT_EQ(turn[i].tokens_generated, i + 1) << "uid " << u.uid << " turn " << t << " token " << i;
                EXPECT_EQ(turn[i].token_id, spec.token_base + spec.prompt_len + i)
                    << "uid " << u.uid << " turn " << t << " token " << i;
            }

            EXPECT_TRUE(turn.back().is_complete) << "uid " << u.uid << " turn " << t << " last token not complete";
            total_tokens_verified += static_cast<int>(turn.size());
        }

        EXPECT_EQ(mgr_->get_user_state(u.uid), pm::UserState::COMPLETE);
    }

    std::cout << "[  INFO   ] " << MAX_USERS << " users x " << TURNS_PER_USER << " turns, " << total_tokens_verified
              << " tokens verified" << std::endl;
}

// =============================================================================
//  Test 7: EOS early termination.
//  Loopback kernel produces token_id + 1. EOS_TOKEN = 2.
//  If the last prefill token is 1, the first decode output is 2 = EOS.
//  Generation should stop after 1 output despite a large max_new_tokens.
// =============================================================================
TEST_F(DevicePipelineFixture, EosEarlyTermination) {
    int32_t req_id = 0;

    ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
    mgr_->tick();
    PMResponse resp{};
    ASSERT_TRUE(mgr_->try_pop_response(resp));
    ASSERT_EQ(resp.error_code, 0);
    int32_t uid = resp.user_id;

    std::vector<int32_t> prompt = {10, 20, 1};
    ASSERT_TRUE(mgr_->push_request(make_submit(req_id++, uid, prompt, 1000)));
    mgr_->tick();

    std::vector<OutputMessage> outputs;
    bool done = poll_until(*mgr_, [&] { return !outputs.empty() && outputs.back().is_complete; }, outputs);
    ASSERT_TRUE(done) << "Timed out waiting for EOS completion";

    ASSERT_EQ(static_cast<int>(outputs.size()), 1);
    EXPECT_EQ(outputs[0].token_id, pm::EOS_TOKEN);
    EXPECT_TRUE(outputs[0].is_eos);
    EXPECT_TRUE(outputs[0].is_complete);
    EXPECT_EQ(outputs[0].tokens_generated, 1);
    EXPECT_EQ(mgr_->get_user_state(uid), pm::UserState::COMPLETE);
}

// =============================================================================
//  Test 8: Cancel mid-prefill.
//  Submit a max-length prompt and cancel immediately — the writer is on
//  another thread and won't have finished injecting all tokens yet.
//  Verify cleanup completes and the slot is freed.
// =============================================================================
TEST_F(DevicePipelineFixture, CancelMidPrefill) {
    int32_t req_id = 0;

    ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
    mgr_->tick();
    PMResponse resp{};
    ASSERT_TRUE(mgr_->try_pop_response(resp));
    ASSERT_EQ(resp.error_code, 0);
    int32_t uid = resp.user_id;

    std::vector<int32_t> prompt(MAX_SEQ_LEN);
    std::iota(prompt.begin(), prompt.end(), 1000);
    ASSERT_TRUE(mgr_->push_request(make_submit(req_id++, uid, prompt, 100)));
    mgr_->tick();

    ASSERT_TRUE(mgr_->push_request(make_cancel(req_id++, uid)));
    mgr_->tick();

    std::vector<OutputMessage> outputs;
    bool cleaned_up = poll_until(*mgr_, [&] { return mgr_->get_user_state(uid) == pm::UserState::INACTIVE; }, outputs);
    ASSERT_TRUE(cleaned_up) << "Timed out waiting for mid-prefill cancel cleanup";

    int user_output_count = 0;
    for (auto& o : outputs) {
        if (o.user_id == uid) {
            user_output_count++;
        }
    }
    std::cout << "[  INFO   ] Output tokens before cancel took effect: " << user_output_count << std::endl;

    ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
    std::cout << "[  INFO   ] Allocated new user ID: " << req_id - 1 << std::endl;
    mgr_->tick();
    std::cout << "[  INFO   ] Ticked" << std::endl;
    PMResponse realloc_resp{};
    ASSERT_TRUE(mgr_->try_pop_response(realloc_resp));
    std::cout << "[  INFO   ] Popped response: " << realloc_resp.user_id << std::endl;
    EXPECT_EQ(realloc_resp.error_code, 0);
    EXPECT_GE(realloc_resp.user_id, 0);
    std::cout << "[  INFO   ] Re-allocated user ID: " << realloc_resp.user_id << std::endl;
}

// =============================================================================
//  Test 9: Lifecycle churn at full capacity.
//  All 64 slots active. Repeatedly: randomly cancel a user in decode,
//  re-allocate the slot, submit new work. Runs for a fixed number of churn
//  cycles. Verifies no corruption, no hangs, all slots eventually clean.
// =============================================================================
TEST_F(DevicePipelineFixture, LifecycleChurnFullCapacity) {
    static constexpr int CHURN_CYCLES = 1000;
    static constexpr int MIN_PROMPT = 100;
    static constexpr int MAX_PROMPT = 500;
    static constexpr int MIN_TOKENS = 50;
    static constexpr int MAX_TOKENS = 1000;

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> prompt_dist(MIN_PROMPT, MAX_PROMPT);
    std::uniform_int_distribution<int> token_dist(MIN_TOKENS, MAX_TOKENS);

    int32_t req_id = 0;

    struct SlotInfo {
        int32_t uid = -1;
        int32_t prompt_len = 0;
        int32_t max_new_tokens = 0;
        int generation = 0;
    };
    std::vector<SlotInfo> slots(MAX_USERS);

    for (int i = 0; i < MAX_USERS; i++) {
        ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
    }
    mgr_->tick();
    for (int i = 0; i < MAX_USERS; i++) {
        PMResponse resp{};
        ASSERT_TRUE(mgr_->try_pop_response(resp));
        ASSERT_EQ(resp.error_code, 0);
        slots[i].uid = resp.user_id;
    }

    auto submit_user = [&](int slot_idx) {
        auto& s = slots[slot_idx];
        s.prompt_len = prompt_dist(rng);
        s.max_new_tokens = token_dist(rng);
        s.generation++;

        std::vector<int32_t> prompt(s.prompt_len);
        int32_t base = s.uid * 10000 + s.generation * 1000;
        std::iota(prompt.begin(), prompt.end(), base);

        ASSERT_TRUE(mgr_->push_request(make_submit(req_id++, s.uid, prompt, s.max_new_tokens)));
    };

    for (int i = 0; i < MAX_USERS; i++) {
        submit_user(i);
    }
    mgr_->tick();

    int cancels_done = 0;
    int reallocs_done = 0;
    std::vector<OutputMessage> outputs;
    std::uniform_int_distribution<int> slot_dist(0, MAX_USERS - 1);
    std::uniform_int_distribution<int> delay_dist(0, 500);

    for (int cycle = 0; cycle < CHURN_CYCLES; cycle++) {
        int target = slot_dist(rng);
        auto& s = slots[target];

        std::this_thread::sleep_for(std::chrono::microseconds(delay_dist(rng)));

        mgr_->tick();
        OutputMessage out;
        while (mgr_->try_pop_output(out)) {
            outputs.push_back(out);
        }

        ASSERT_TRUE(mgr_->push_request(make_cancel(req_id++, s.uid)));
        mgr_->tick();
        cancels_done++;

        bool cleaned =
            poll_until(*mgr_, [&] { return mgr_->get_user_state(s.uid) == pm::UserState::INACTIVE; }, outputs);
        ASSERT_TRUE(cleaned) << "Cycle " << cycle << ": user " << s.uid << " stuck after cancel";

        ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
        mgr_->tick();
        PMResponse resp{};
        ASSERT_TRUE(mgr_->try_pop_response(resp));
        ASSERT_EQ(resp.error_code, 0) << "Cycle " << cycle << ": re-allocation failed";
        s.uid = resp.user_id;
        reallocs_done++;

        submit_user(target);
        mgr_->tick();
    }

    bool all_done = poll_until(
        *mgr_,
        [&] {
            for (auto& s : slots) {
                auto st = mgr_->get_user_state(s.uid);
                if (st != pm::UserState::COMPLETE && st != pm::UserState::INACTIVE) {
                    return false;
                }
            }
            return true;
        },
        outputs);
    if (!all_done) {
        for (auto& s : slots) {
            auto st = mgr_->get_user_state(s.uid);
            if (st != pm::UserState::COMPLETE && st != pm::UserState::INACTIVE) {
                std::cout << "[  STUCK  ] slot uid=" << s.uid << " state=" << static_cast<int>(st)
                          << " gen=" << s.generation << " prompt_len=" << s.prompt_len
                          << " max_new(test)=" << s.max_new_tokens << " max_new(pm)=" << mgr_->get_max_new_tokens(s.uid)
                          << " in_flight=" << mgr_->get_in_flight_count(s.uid)
                          << " tokens_gen=" << mgr_->get_tokens_generated(s.uid)
                          << " cur_pos=" << mgr_->get_current_position(s.uid)
                          << " cancel_pending=" << mgr_->get_cancel_pending(s.uid)
                          << " staging_size=" << mgr_->get_decode_staging_size() << std::endl;
            }
        }
    }
    ASSERT_TRUE(all_done) << "Timed out waiting for remaining users to complete";

    std::map<int32_t, std::vector<OutputMessage>> per_user;
    for (auto& o : outputs) {
        per_user[o.user_id].push_back(o);
    }

    int verified_users = 0;
    for (auto& s : slots) {
        if (mgr_->get_user_state(s.uid) != pm::UserState::COMPLETE) {
            continue;
        }
        auto it = per_user.find(s.uid);
        if (it == per_user.end()) {
            continue;
        }

        auto& raw = it->second;
        uint32_t latest_gen = 0;
        for (auto& o : raw) {
            latest_gen = std::max(latest_gen, o.generation);
        }
        std::vector<OutputMessage> user_outputs;
        for (auto& o : raw) {
            if (o.generation == latest_gen) {
                user_outputs.push_back(o);
            }
        }

        EXPECT_EQ(static_cast<int32_t>(user_outputs.size()), s.max_new_tokens)
            << "User " << s.uid << " gen " << s.generation;

        int32_t base = s.uid * 10000 + s.generation * 1000;
        for (int i = 0; i < static_cast<int>(user_outputs.size()); i++) {
            EXPECT_EQ(user_outputs[i].token_id, base + s.prompt_len + i)
                << "User " << s.uid << " gen " << s.generation << " token " << i;
            EXPECT_EQ(user_outputs[i].tokens_generated, i + 1);
        }
        if (!user_outputs.empty()) {
            EXPECT_TRUE(user_outputs.back().is_complete);
        }
        verified_users++;
    }

    std::cout << "[  INFO   ] Churn cycles: " << CHURN_CYCLES << ", cancels: " << cancels_done
              << ", reallocs: " << reallocs_done << ", verified users: " << verified_users
              << ", total outputs: " << outputs.size() << std::endl;
}

// =============================================================================
//  Test 10: Cancel-and-resubmit storm on a single slot.
//  Hammers one user ID with rapid cancel/resubmit cycles.
//  Verifies no stale state leaks between sessions on the same slot.
// =============================================================================
TEST_F(DevicePipelineFixture, CancelResubmitStormSingleSlot) {
    static constexpr int CYCLES = 200;

    std::mt19937 rng(99);
    std::uniform_int_distribution<int> prompt_dist(3, 50);
    std::uniform_int_distribution<int> token_dist(5, 50);
    std::uniform_int_distribution<int> delay_dist(0, 200);

    int32_t req_id = 0;

    ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
    mgr_->tick();
    PMResponse resp{};
    ASSERT_TRUE(mgr_->try_pop_response(resp));
    ASSERT_EQ(resp.error_code, 0);
    int32_t uid = resp.user_id;

    std::vector<OutputMessage> drain;
    auto flush_outputs = [&]() {
        mgr_->tick();
        OutputMessage out;
        while (mgr_->try_pop_output(out)) {
            drain.push_back(out);
        }
    };

    for (int cycle = 0; cycle < CYCLES; cycle++) {
        int32_t plen = prompt_dist(rng);
        int32_t max_new = token_dist(rng);
        int32_t base = (cycle + 1) * 10000;

        std::vector<int32_t> prompt(plen);
        std::iota(prompt.begin(), prompt.end(), base);

        ASSERT_TRUE(mgr_->push_request(make_submit(req_id++, uid, prompt, max_new)));
        mgr_->tick();

        std::this_thread::sleep_for(std::chrono::microseconds(delay_dist(rng)));
        flush_outputs();

        ASSERT_TRUE(mgr_->push_request(make_cancel(req_id++, uid)));
        mgr_->tick();

        bool cleaned = poll_until(*mgr_, [&] { return mgr_->get_user_state(uid) == pm::UserState::INACTIVE; }, drain);
        ASSERT_TRUE(cleaned) << "Cycle " << cycle << ": stuck after cancel";
    }

    // Re-allocate before the final submission
    ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
    mgr_->tick();
    PMResponse final_alloc{};
    ASSERT_TRUE(mgr_->try_pop_response(final_alloc));
    ASSERT_EQ(final_alloc.error_code, 0);
    uid = final_alloc.user_id;

    int32_t final_plen = prompt_dist(rng);
    int32_t final_max_new = token_dist(rng);
    int32_t final_base = (CYCLES + 1) * 10000;

    std::vector<int32_t> final_prompt(final_plen);
    std::iota(final_prompt.begin(), final_prompt.end(), final_base);

    ASSERT_TRUE(mgr_->push_request(make_submit(req_id++, uid, final_prompt, final_max_new)));
    mgr_->tick();

    std::vector<OutputMessage> outputs;
    bool done = poll_until(*mgr_, [&] { return !outputs.empty() && outputs.back().is_complete; }, outputs);
    ASSERT_TRUE(done) << "Final submission timed out";

    uint32_t latest_gen = 0;
    for (auto& o : outputs) {
        if (o.user_id == uid) {
            latest_gen = std::max(latest_gen, o.generation);
        }
    }
    std::vector<OutputMessage> final_outputs;
    for (auto& o : outputs) {
        if (o.user_id == uid && o.generation == latest_gen) {
            final_outputs.push_back(o);
        }
    }

    ASSERT_EQ(static_cast<int32_t>(final_outputs.size()), final_max_new)
        << "Final session: expected " << final_max_new << " tokens, got " << final_outputs.size();

    for (int i = 0; i < static_cast<int>(final_outputs.size()); i++) {
        EXPECT_EQ(final_outputs[i].token_id, final_base + final_plen + i) << "Final session token " << i << " mismatch";
        EXPECT_EQ(final_outputs[i].tokens_generated, i + 1);
        EXPECT_EQ(final_outputs[i].user_id, uid);
    }
    EXPECT_TRUE(final_outputs.back().is_complete);
    EXPECT_EQ(mgr_->get_user_state(uid), pm::UserState::COMPLETE);

    std::cout << "[  INFO   ] " << CYCLES << " cancel/resubmit cycles on uid " << uid
              << ", final session: " << final_max_new << " tokens verified" << std::endl;
}

// =============================================================================
//  Test 11: Concurrent cancel of all 64 users.
//  Submit work for all 64 users, let decode start, then cancel all 64 in
//  one tick. Verify all reach INACTIVE and can be re-allocated.
// =============================================================================
TEST_F(DevicePipelineFixture, ConcurrentCancelAll64) {
    int32_t req_id = 0;
    std::vector<int32_t> uids(MAX_USERS);

    for (int i = 0; i < MAX_USERS; i++) {
        ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
    }
    mgr_->tick();
    for (int i = 0; i < MAX_USERS; i++) {
        PMResponse resp{};
        ASSERT_TRUE(mgr_->try_pop_response(resp));
        ASSERT_EQ(resp.error_code, 0);
        uids[i] = resp.user_id;
    }

    for (int i = 0; i < MAX_USERS; i++) {
        std::vector<int32_t> prompt(20);
        std::iota(prompt.begin(), prompt.end(), static_cast<int32_t>(uids[i] * 1000));
        ASSERT_TRUE(mgr_->push_request(make_submit(req_id++, uids[i], prompt, 500)));
    }
    mgr_->tick();

    std::vector<OutputMessage> outputs;
    poll_until(
        *mgr_,
        [&] {
            int decoding = 0;
            for (auto uid : uids) {
                if (mgr_->get_user_state(uid) == pm::UserState::DECODE) {
                    decoding++;
                }
            }
            return decoding >= MAX_USERS / 2;
        },
        outputs);

    for (auto uid : uids) {
        ASSERT_TRUE(mgr_->push_request(make_cancel(req_id++, uid)));
    }
    mgr_->tick();

    bool all_inactive = poll_until(
        *mgr_,
        [&] {
            for (auto uid : uids) {
                if (mgr_->get_user_state(uid) != pm::UserState::INACTIVE) {
                    return false;
                }
            }
            return true;
        },
        outputs);
    ASSERT_TRUE(all_inactive) << "Timed out waiting for all 64 cancels to complete";

    for (int i = 0; i < MAX_USERS; i++) {
        ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
    }
    mgr_->tick();
    for (int i = 0; i < MAX_USERS; i++) {
        PMResponse resp{};
        ASSERT_TRUE(mgr_->try_pop_response(resp));
        EXPECT_EQ(resp.error_code, 0) << "Re-allocation " << i << " failed";
    }

    std::cout << "[  INFO   ] All 64 users cancelled and re-allocated, " << outputs.size() << " outputs drained"
              << std::endl;
}

// =============================================================================
//  Test 12: Single-token fast cycling.
//  prompt_len=1, max_new_tokens=1 — fastest possible lifecycle.
//  All 64 slots: allocate, submit, complete, cancel, repeat 100 times.
// =============================================================================
TEST_F(DevicePipelineFixture, SingleTokenFastCycling) {
    static constexpr int CYCLES = 100;
    int32_t req_id = 0;

    std::vector<int32_t> uids(MAX_USERS);

    for (int i = 0; i < MAX_USERS; i++) {
        ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
    }
    mgr_->tick();
    for (int i = 0; i < MAX_USERS; i++) {
        PMResponse resp{};
        ASSERT_TRUE(mgr_->try_pop_response(resp));
        ASSERT_EQ(resp.error_code, 0);
        uids[i] = resp.user_id;
    }

    int total_completions = 0;

    for (int cycle = 0; cycle < CYCLES; cycle++) {
        for (int i = 0; i < MAX_USERS; i++) {
            int32_t base = cycle * MAX_USERS * 100 + uids[i] * 100;
            std::vector<int32_t> prompt = {base};
            ASSERT_TRUE(mgr_->push_request(make_submit(req_id++, uids[i], prompt, 1)));
        }
        mgr_->tick();

        std::vector<OutputMessage> outputs;
        auto count_complete = [&]() {
            std::set<int32_t> done;
            for (auto& o : outputs) {
                if (o.is_complete) {
                    done.insert(o.user_id);
                }
            }
            return static_cast<int>(done.size());
        };

        bool all_done = poll_until(*mgr_, [&] { return count_complete() == MAX_USERS; }, outputs);
        ASSERT_TRUE(all_done) << "Cycle " << cycle << ": only " << count_complete() << "/" << MAX_USERS << " completed";

        total_completions += MAX_USERS;

        std::map<int32_t, int> per_user_count;
        for (auto& o : outputs) {
            per_user_count[o.user_id]++;
        }
        for (auto uid : uids) {
            EXPECT_GE(per_user_count[uid], 1) << "Cycle " << cycle << ": user " << uid << " missing output";
        }

        for (auto uid : uids) {
            ASSERT_TRUE(mgr_->push_request(make_cancel(req_id++, uid)));
        }
        mgr_->tick();

        std::vector<OutputMessage> cancel_drain;
        bool all_inactive = poll_until(
            *mgr_,
            [&] {
                for (auto uid : uids) {
                    if (mgr_->get_user_state(uid) != pm::UserState::INACTIVE) {
                        return false;
                    }
                }
                return true;
            },
            cancel_drain);
        ASSERT_TRUE(all_inactive) << "Cycle " << cycle << ": not all users reached INACTIVE";

        for (int i = 0; i < MAX_USERS; i++) {
            ASSERT_TRUE(mgr_->push_request(make_allocate(req_id++)));
        }
        mgr_->tick();
        for (int i = 0; i < MAX_USERS; i++) {
            PMResponse resp{};
            ASSERT_TRUE(mgr_->try_pop_response(resp));
            ASSERT_EQ(resp.error_code, 0) << "Cycle " << cycle << ": re-alloc " << i << " failed";
            uids[i] = resp.user_id;
        }
    }

    std::cout << "[  INFO   ] " << CYCLES << " cycles x " << MAX_USERS << " users = " << total_completions
              << " completions" << std::endl;
}

// =============================================================================
//  Test 13: Realistic multi-turn serving scenario.
//
//  Simulates a real serving workload:
//    Phase 1 — Single user arrives, submits, and does 3 turns (CONTINUE).
//    Phase 2 — More users arrive in batches of 8 until all 64 slots are full.
//    Phase 3 — All 64 users complete multi-turn conversations (3 turns each).
//    Phase 4 — 16 users leave (cancel); 16 new users arrive and do 3 turns.
//    Phase 5 — All remaining users finish; verify token correctness.
// =============================================================================
// TEST_F(DevicePipelineFixture, RealisticMultiTurnServing) {
//     static constexpr int TURNS_PER_SESSION = 3;
//     static constexpr int RAMP_BATCH = 8;
//     static constexpr int CHURN_REPLACEMENTS = 16;

//     std::mt19937 rng(42);
//     std::uniform_int_distribution<int> prompt_dist(5, 50);
//     std::uniform_int_distribution<int> token_dist(5, 20);

//     int32_t req_id = 0;
//     int next_session_id = 0;

//     struct TurnSpec {
//         int32_t prompt_len;
//         int32_t max_new;
//         int32_t token_base;
//     };

//     struct Session {
//         int32_t uid = -1;
//         int session_id = -1;
//         int current_turn = 0;
//         bool awaiting_completion = false;
//         std::vector<TurnSpec> turns;
//     };

//     std::map<int32_t, Session> active;
//     std::vector<OutputMessage> all_outputs;
//     int turns_completed_total = 0;

//     auto do_allocate = [&]() -> int32_t {
//         mgr_->push_request(make_allocate(req_id++));
//         mgr_->tick();
//         PMResponse resp{};
//         bool got = mgr_->try_pop_response(resp);
//         EXPECT_TRUE(got);
//         EXPECT_EQ(resp.error_code, 0);
//         return resp.user_id;
//     };

//     auto begin_session = [&](int32_t uid) {
//         Session s;
//         s.uid = uid;
//         s.session_id = next_session_id++;
//         s.current_turn = 0;
//         s.awaiting_completion = true;

//         int32_t plen = prompt_dist(rng);
//         int32_t max_new = token_dist(rng);
//         int32_t base = s.session_id * 100000;
//         s.turns.push_back({plen, max_new, base});

//         std::vector<int32_t> prompt(plen);
//         std::iota(prompt.begin(), prompt.end(), base);
//         mgr_->push_request(make_submit(req_id++, uid, prompt, max_new));

//         active[uid] = std::move(s);
//     };

//     auto continue_session = [&](int32_t uid) {
//         auto& s = active[uid];
//         s.current_turn++;
//         s.awaiting_completion = true;

//         int32_t plen = prompt_dist(rng);
//         int32_t max_new = token_dist(rng);
//         int32_t base = s.session_id * 100000 + s.current_turn * 10000;
//         s.turns.push_back({plen, max_new, base});

//         std::vector<int32_t> prompt(plen);
//         std::iota(prompt.begin(), prompt.end(), base);
//         mgr_->push_request(make_continue(req_id++, uid, prompt, max_new));
//     };

//     auto do_cancel_wait = [&](int32_t uid) {
//         mgr_->push_request(make_cancel(req_id++, uid));
//         mgr_->tick();
//         std::vector<OutputMessage> drain;
//         bool ok = poll_until(*mgr_, [&] { return mgr_->get_user_state(uid) == pm::UserState::INACTIVE; }, drain);
//         EXPECT_TRUE(ok) << "Cancel cleanup timed out for uid " << uid;
//         all_outputs.insert(all_outputs.end(), drain.begin(), drain.end());
//         active.erase(uid);
//     };

//     // Drive the simulation: tick, drain outputs, detect completed turns,
//     // issue CONTINUEs.  Returns when all active sessions have finished
//     // all their turns, or on timeout.
//     auto run_until_all_done = [&](const char* phase_name) {
//         static constexpr int MAX_ITERS = 100000000;
//         for (int iter = 0; iter < MAX_ITERS; iter++) {
//             mgr_->tick();
//             OutputMessage out;
//             while (mgr_->try_pop_output(out)) {
//                 all_outputs.push_back(out);
//             }

//             std::vector<int32_t> just_completed;
//             for (auto& [uid, s] : active) {
//                 if (s.awaiting_completion &&
//                     mgr_->get_user_state(uid) == pm::UserState::COMPLETE) {
//                     s.awaiting_completion = false;
//                     just_completed.push_back(uid);
//                     turns_completed_total++;
//                 }
//             }

//             for (auto uid : just_completed) {
//                 auto& s = active[uid];
//                 if (s.current_turn < TURNS_PER_SESSION - 1) {
//                     continue_session(uid);
//                 }
//             }

//             if (!just_completed.empty()) {
//                 mgr_->tick();
//             }

//             bool all_done = true;
//             for (auto& [uid, s] : active) {
//                 if (s.current_turn < TURNS_PER_SESSION - 1 || s.awaiting_completion) {
//                     all_done = false;
//                     break;
//                 }
//             }
//             if (all_done) return;

//             std::this_thread::yield();
//         }
//         FAIL() << phase_name << ": simulation timed out";
//     };

//     // ---- Phase 1: Single user, full multi-turn conversation ----
//     std::cout << "[  INFO   ] Phase 1: single user, " << TURNS_PER_SESSION << " turns" << std::endl;
//     {
//         int32_t uid = do_allocate();
//         begin_session(uid);
//         mgr_->tick();
//         run_until_all_done("Phase 1");
//         EXPECT_EQ(mgr_->get_user_state(uid), pm::UserState::COMPLETE);
//     }

//     // ---- Phase 2: Gradual ramp to 64 users ----
//     std::cout << "[  INFO   ] Phase 2: ramping to " << MAX_USERS << " users" << std::endl;
//     {
//         int existing = static_cast<int>(active.size());
//         while (existing < MAX_USERS) {
//             int batch = std::min(RAMP_BATCH, MAX_USERS - existing);
//             for (int i = 0; i < batch; i++) {
//                 int32_t uid = do_allocate();
//                 begin_session(uid);
//             }
//             existing += batch;
//             mgr_->tick();

//             // Drain between batches so the pipeline stays healthy
//             OutputMessage out;
//             while (mgr_->try_pop_output(out)) {
//                 all_outputs.push_back(out);
//             }
//         }
//         EXPECT_EQ(static_cast<int>(active.size()), MAX_USERS);
//     }

//     // ---- Phase 3: All users run multi-turn to completion ----
//     std::cout << "[  INFO   ] Phase 3: all " << MAX_USERS << " users, multi-turn" << std::endl;
//     run_until_all_done("Phase 3");

//     // ---- Phase 4: Churn — cancel 16, replace with 16 new users ----
//     std::cout << "[  INFO   ] Phase 4: churn " << CHURN_REPLACEMENTS << " users" << std::endl;
//     {
//         std::vector<int32_t> targets;
//         for (auto& [uid, s] : active) {
//             if (static_cast<int>(targets.size()) >= CHURN_REPLACEMENTS) break;
//             targets.push_back(uid);
//         }

//         for (auto uid : targets) {
//             do_cancel_wait(uid);
//             int32_t new_uid = do_allocate();
//             begin_session(new_uid);
//             mgr_->tick();
//         }
//     }

//     // ---- Phase 5: Let all remaining/replacement users finish ----
//     std::cout << "[  INFO   ] Phase 5: drain remaining users" << std::endl;
//     run_until_all_done("Phase 5");

//     // ---- Verification ----
//     // Bucket outputs by user_id, split each user's stream by is_complete into turns.
//     std::map<int32_t, std::vector<OutputMessage>> per_user;
//     for (auto& o : all_outputs) {
//         per_user[o.user_id].push_back(o);
//     }

//     int users_verified = 0;
//     for (auto& [uid, s] : active) {
//         auto it = per_user.find(uid);
//         if (it == per_user.end()) continue;

//         auto& stream = it->second;

//         // Split into turns at each is_complete boundary
//         std::vector<std::vector<OutputMessage>> turn_outputs;
//         std::vector<OutputMessage> current_turn_buf;
//         for (auto& o : stream) {
//             current_turn_buf.push_back(o);
//             if (o.is_complete) {
//                 turn_outputs.push_back(std::move(current_turn_buf));
//                 current_turn_buf.clear();
//             }
//         }

//         EXPECT_EQ(static_cast<int>(turn_outputs.size()), TURNS_PER_SESSION)
//             << "uid " << uid << " (session " << s.session_id << "): expected "
//             << TURNS_PER_SESSION << " completed turns, got " << turn_outputs.size();

//         for (int t = 0; t < std::min(static_cast<int>(turn_outputs.size()),
//                                       static_cast<int>(s.turns.size())); t++) {
//             auto& turn = turn_outputs[t];
//             auto& spec = s.turns[t];

//             EXPECT_EQ(static_cast<int32_t>(turn.size()), spec.max_new)
//                 << "uid " << uid << " turn " << t << ": expected "
//                 << spec.max_new << " tokens, got " << turn.size();

//             for (int i = 0; i < static_cast<int>(turn.size()); i++) {
//                 EXPECT_EQ(turn[i].tokens_generated, i + 1)
//                     << "uid " << uid << " turn " << t << " token " << i;
//                 EXPECT_EQ(turn[i].token_id, spec.token_base + spec.prompt_len + i)
//                     << "uid " << uid << " turn " << t << " token " << i << " value mismatch";
//             }

//             if (!turn.empty()) {
//                 EXPECT_TRUE(turn.back().is_complete);
//             }
//         }
//         users_verified++;
//     }

//     std::cout << "[  INFO   ] Sessions: " << next_session_id
//               << ", turns completed: " << turns_completed_total
//               << ", users verified: " << users_verified
//               << ", total outputs: " << all_outputs.size() << std::endl;
// }
