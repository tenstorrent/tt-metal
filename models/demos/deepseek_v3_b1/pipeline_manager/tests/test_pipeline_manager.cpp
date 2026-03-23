// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_interface.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager_types.hpp"

namespace pm = models::demos::deepseek_v3_b1::pipeline_manager;

using pm::ISRequest;
using pm::MAX_SEQ_LEN;
using pm::MAX_USERS;
using pm::OutputMessage;
using pm::PMResponse;
using pm::RequestType;

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

static constexpr int MAX_POLL_ITERATIONS = 10000000;

// Poll tick + output until predicate is satisfied or iteration limit hit.
template <typename Pred>
bool poll_until(pm::PipelineManager& mgr, Pred pred, std::vector<OutputMessage>& outputs) {
    for (int i = 0; i < MAX_POLL_ITERATIONS; i++) {
        mgr.tick();
        OutputMessage out;
        while (mgr.try_pop_output(out)) {
            outputs.push_back(out);
        }
        if (pred()) {
            return true;
        }
        std::this_thread::yield();
    }
    return false;
}

}  // namespace

// =============================================================================
//  Test 1: Single user — allocate, submit, run to completion
// =============================================================================
TEST(PipelineManagerTest, SingleUserAllocateSubmitComplete) {
    pm::MockPipeline mock;
    pm::PipelineManager mgr(mock);
    mgr.start();

    // Allocate
    ASSERT_TRUE(mgr.push_request(make_allocate(1)));
    mgr.tick();

    PMResponse resp{};
    ASSERT_TRUE(mgr.try_pop_response(resp));
    EXPECT_EQ(resp.request_id, 1);
    EXPECT_EQ(resp.error_code, 0);
    EXPECT_GE(resp.user_id, 0);
    EXPECT_LT(resp.user_id, MAX_USERS);
    int32_t uid = resp.user_id;

    // Submit: 3-token prompt, generate 5 tokens
    std::vector<int32_t> prompt = {100, 200, 300};
    int32_t max_new = 5;
    ASSERT_TRUE(mgr.push_request(make_submit(2, uid, prompt, max_new)));
    mgr.tick();

    // Poll for outputs
    std::vector<OutputMessage> outputs;
    bool done = poll_until(mgr, [&] { return !outputs.empty() && outputs.back().is_complete; }, outputs);

    ASSERT_TRUE(done) << "Timed out waiting for completion";
    EXPECT_EQ(static_cast<int>(outputs.size()), max_new);

    // Verify all outputs belong to our user
    for (const auto& o : outputs) {
        EXPECT_EQ(o.user_id, uid);
    }

    // Verify tokens_generated increments correctly
    for (int i = 0; i < static_cast<int>(outputs.size()); i++) {
        EXPECT_EQ(outputs[i].tokens_generated, i + 1);
    }

    // Only the last output should be marked complete
    for (int i = 0; i < static_cast<int>(outputs.size()) - 1; i++) {
        EXPECT_FALSE(outputs[i].is_complete);
    }
    EXPECT_TRUE(outputs.back().is_complete);

    // Verify deterministic token values.
    // MockPipeline: decode produces token_id + 1.
    // Last prefill token (300) is injected as DECODE → output = 301.
    // Then loopback: 301 → 302 → 303 → 304 → 305.
    EXPECT_EQ(outputs[0].token_id, 301);
    EXPECT_EQ(outputs[1].token_id, 302);
    EXPECT_EQ(outputs[2].token_id, 303);
    EXPECT_EQ(outputs[3].token_id, 304);
    EXPECT_EQ(outputs[4].token_id, 305);

    mgr.stop();
}

// =============================================================================
//  Test 2: Allocation exhaustion — 64 succeed, 65th fails
// =============================================================================
TEST(PipelineManagerTest, AllocationExhaustion) {
    pm::MockPipeline mock;
    pm::PipelineManager mgr(mock);
    mgr.start();

    std::set<int32_t> allocated_ids;

    // Allocate all 64 users
    for (int i = 0; i < MAX_USERS; i++) {
        ASSERT_TRUE(mgr.push_request(make_allocate(i)));
    }
    mgr.tick();

    for (int i = 0; i < MAX_USERS; i++) {
        PMResponse resp{};
        ASSERT_TRUE(mgr.try_pop_response(resp));
        EXPECT_EQ(resp.error_code, 0) << "Allocation " << i << " failed";
        EXPECT_GE(resp.user_id, 0);
        EXPECT_LT(resp.user_id, MAX_USERS);
        allocated_ids.insert(resp.user_id);
    }

    // All 64 IDs should be unique
    EXPECT_EQ(static_cast<int>(allocated_ids.size()), MAX_USERS);

    // 65th allocation should fail
    ASSERT_TRUE(mgr.push_request(make_allocate(MAX_USERS)));
    mgr.tick();

    PMResponse fail_resp{};
    ASSERT_TRUE(mgr.try_pop_response(fail_resp));
    EXPECT_EQ(fail_resp.user_id, -1);
    EXPECT_NE(fail_resp.error_code, 0);

    mgr.stop();
}

// =============================================================================
//  Test 3: Cancel mid-decode frees user ID for re-allocation
// =============================================================================
TEST(PipelineManagerTest, CancelMidDecodeFreesUserIdForReallocation) {
    pm::MockPipeline mock;
    pm::PipelineManager mgr(mock);
    mgr.start();

    // Allocate
    ASSERT_TRUE(mgr.push_request(make_allocate(1)));
    mgr.tick();

    PMResponse resp{};
    ASSERT_TRUE(mgr.try_pop_response(resp));
    ASSERT_EQ(resp.error_code, 0);
    int32_t uid = resp.user_id;

    // Submit with a large max_new_tokens so decode runs for a while
    std::vector<int32_t> prompt = {10, 20, 30};
    ASSERT_TRUE(mgr.push_request(make_submit(2, uid, prompt, 1000000)));
    mgr.tick();

    // Wait until decode is actively producing output tokens
    std::vector<OutputMessage> outputs;
    bool got_output = poll_until(mgr, [&] { return outputs.size() >= 3; }, outputs);
    ASSERT_TRUE(got_output) << "Timed out waiting for decode output tokens";
    EXPECT_EQ(mgr.get_user_state(uid), pm::UserState::DECODE);

    size_t tokens_before_cancel = outputs.size();

    // Random delay up to 2ms to vary the point at which cancel arrives
    std::mt19937 rng(std::random_device{}());
    auto delay_us = std::uniform_int_distribution<int>(0, 2000)(rng);
    std::this_thread::sleep_for(std::chrono::microseconds(delay_us));

    // Cancel now that decode is confirmed active
    ASSERT_TRUE(mgr.push_request(make_cancel(3, uid)));
    mgr.tick();

    // Wait until the user slot is fully cleaned up (INACTIVE)
    bool cleaned_up = poll_until(mgr, [&] { return mgr.get_user_state(uid) == pm::UserState::INACTIVE; }, outputs);
    ASSERT_TRUE(cleaned_up) << "Timed out waiting for cancel cleanup";

    size_t tokens_after_cancel = outputs.size() - tokens_before_cancel;
    std::cout << "[  INFO   ] Random delay before cancel: " << delay_us << " us" << std::endl;
    std::cout << "[  INFO   ] Tokens before cancel: " << tokens_before_cancel
              << ", drained after cancel: " << tokens_after_cancel << ", total: " << outputs.size() << std::endl;

    // Re-allocate — should succeed now that the user ID was freed
    ASSERT_TRUE(mgr.push_request(make_allocate(4)));
    mgr.tick();

    PMResponse realloc_resp{};
    ASSERT_TRUE(mgr.try_pop_response(realloc_resp));
    EXPECT_EQ(realloc_resp.error_code, 0);
    EXPECT_GE(realloc_resp.user_id, 0);
    EXPECT_LT(realloc_resp.user_id, MAX_USERS);
    int32_t uid2 = realloc_resp.user_id;

    // ---- Cancel before submit (allocated but never started) ----
    ASSERT_TRUE(mgr.push_request(make_cancel(5, uid2)));
    mgr.tick();

    bool cleaned_up2 = poll_until(mgr, [&] { return mgr.get_user_state(uid2) == pm::UserState::INACTIVE; }, outputs);
    ASSERT_TRUE(cleaned_up2) << "Timed out waiting for cancel-before-submit cleanup";

    // Re-allocate again — freed without ever having in-flight tokens
    ASSERT_TRUE(mgr.push_request(make_allocate(6)));
    mgr.tick();

    PMResponse realloc_resp2{};
    ASSERT_TRUE(mgr.try_pop_response(realloc_resp2));
    EXPECT_EQ(realloc_resp2.error_code, 0);
    EXPECT_GE(realloc_resp2.user_id, 0);
    EXPECT_LT(realloc_resp2.user_id, MAX_USERS);

    mgr.stop();
}

// =============================================================================
//  Test 4: Max users — randomized prompts and token counts, all complete
// =============================================================================
TEST(PipelineManagerTest, MultipleUsersStressTest) {
    pm::MockPipeline mock(50, 200);
    pm::PipelineManager mgr(mock);
    mgr.start();

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

    // Allocate all users
    for (int i = 0; i < NUM_USERS; i++) {
        ASSERT_TRUE(mgr.push_request(make_allocate(req_id++)));
    }
    mgr.tick();
    for (int i = 0; i < NUM_USERS; i++) {
        PMResponse resp{};
        ASSERT_TRUE(mgr.try_pop_response(resp));
        ASSERT_EQ(resp.error_code, 0);
        users[i].uid = resp.user_id;
    }

    // Submit each user with randomized prompt and token count.
    // Each user gets a unique token base (uid * 10000) so tokens don't collide.
    int total_expected = 0;
    for (int i = 0; i < NUM_USERS; i++) {
        users[i].prompt_len = prompt_len_dist(rng);
        users[i].max_new_tokens = token_count_dist(rng);
        total_expected += users[i].max_new_tokens;

        std::vector<int32_t> prompt(users[i].prompt_len);
        int32_t base = users[i].uid * 10000;
        std::iota(prompt.begin(), prompt.end(), base);

        ASSERT_TRUE(mgr.push_request(make_submit(req_id++, users[i].uid, prompt, users[i].max_new_tokens)));
    }
    mgr.tick();

    std::cout << "[  INFO   ] " << NUM_USERS << " users, total expected tokens: " << total_expected << std::endl;

    // Poll until all users complete
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

    bool all_done = poll_until(mgr, [&] { return count_complete() == NUM_USERS; }, outputs);
    ASSERT_TRUE(all_done) << "Timed out: only " << count_complete() << "/" << NUM_USERS << " completed";

    // Bucket outputs by user and verify
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

        // MockPipeline: decode produces token_id + 1.
        // Last prompt token = base + prompt_len - 1, so first output = base + prompt_len.
        int32_t base = u.uid * 10000;
        for (int i = 0; i < static_cast<int>(user_outputs.size()); i++) {
            EXPECT_EQ(user_outputs[i].tokens_generated, i + 1);
            EXPECT_EQ(user_outputs[i].user_id, u.uid);
            EXPECT_EQ(user_outputs[i].token_id, base + u.prompt_len + i)
                << "User " << u.uid << " token " << i << " mismatch";
        }

        EXPECT_TRUE(user_outputs.back().is_complete);
        EXPECT_EQ(mgr.get_user_state(u.uid), pm::UserState::COMPLETE);
    }

    std::cout << "[  INFO   ] All " << NUM_USERS << " users completed, " << outputs.size() << " total output tokens"
              << std::endl;

    mgr.stop();
}

// =============================================================================
//  Test 5: Chunked prefill — 8 users with identical long prompts produce
//  interleaved output (no user completes entirely before another starts).
//
//  Uses simulated pipeline latency so the reader is slow and the writer
//  fills idle ticks with other users' prefill tokens via round-robin chunking.
// =============================================================================
TEST(PipelineManagerTest, ChunkedPrefillInterleaved) {
    static constexpr int CHUNK_SIZE = 32;
    static constexpr int NUM_USERS = 8;
    static constexpr int PROMPT_LEN = 2048;
    static constexpr int MAX_NEW = 30;

    pm::MockPipeline mock(50, 200);
    pm::PipelineManager mgr(mock, CHUNK_SIZE);
    mgr.start();

    int32_t req_id = 0;
    std::vector<int32_t> uids(NUM_USERS);

    // Allocate all users
    for (int i = 0; i < NUM_USERS; i++) {
        ASSERT_TRUE(mgr.push_request(make_allocate(req_id++)));
    }
    mgr.tick();
    for (int i = 0; i < NUM_USERS; i++) {
        PMResponse resp{};
        ASSERT_TRUE(mgr.try_pop_response(resp));
        ASSERT_EQ(resp.error_code, 0);
        uids[i] = resp.user_id;
    }

    // Submit all with identical prompt length, unique token bases
    for (int i = 0; i < NUM_USERS; i++) {
        int32_t base = uids[i] * 10000;
        std::vector<int32_t> prompt(PROMPT_LEN);
        std::iota(prompt.begin(), prompt.end(), base);
        ASSERT_TRUE(mgr.push_request(make_submit(req_id++, uids[i], prompt, MAX_NEW)));
    }
    mgr.tick();

    // Poll until all complete
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

    bool all_done = poll_until(mgr, [&] { return count_complete() == NUM_USERS; }, outputs);
    ASSERT_TRUE(all_done) << "Timed out: only " << count_complete() << "/" << NUM_USERS << " completed";

    // Find first and last output index for each user
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

    // Every pair of users should have overlapping output ranges.
    // If user X completed entirely before user Y started, chunking isn't working.
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

    // Verify per-user correctness: token count, deterministic values, completion
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
        EXPECT_EQ(mgr.get_user_state(uid), pm::UserState::COMPLETE);
    }

    mgr.stop();
}

// =============================================================================
//  Test 6: Multi-turn continue — SUBMIT, complete, CONTINUE, complete.
//  Verify the second turn's tokens pick up where the first left off
//  (KV position continuity) and the user ID is reused without re-allocation.
//
//  MockPipeline: decode produces token_id + 1.
//
//  Turn 1: prompt = [100, 200, 300], max_new = 3
//    Prefill injections: pos 0, 1, 2 (last as DECODE mode)
//    Decode injections:  pos 3, 4 (loopback; pos 4 produces the 3rd output, no re-inject)
//    Total injections: 5, positions 0..4
//
//  Turn 2: continue tokens = [400, 500], max_new = 2
//    Prefill injections: pos 5, 6 (continues from current_position=5)
//    Decode injections:  pos 7 (loopback; pos 7 produces the 2nd output, no re-inject)
//    Total injections: 3, positions 5..7
//
//  Combined: 8 injections at positions 0,1,2,3,4,5,6,7
// =============================================================================
TEST(PipelineManagerTest, MultiTurnContinue) {
    pm::MockPipeline mock;
    pm::PipelineManager mgr(mock);
    mgr.start();

    int32_t req_id = 0;

    // Allocate
    ASSERT_TRUE(mgr.push_request(make_allocate(req_id++)));
    mgr.tick();
    PMResponse resp{};
    ASSERT_TRUE(mgr.try_pop_response(resp));
    ASSERT_EQ(resp.error_code, 0);
    int32_t uid = resp.user_id;

    // ---- Turn 1 ----
    std::vector<int32_t> prompt_1 = {100, 200, 300};
    int32_t max_new_1 = 3;
    ASSERT_TRUE(mgr.push_request(make_submit(req_id++, uid, prompt_1, max_new_1)));
    mgr.tick();

    std::vector<OutputMessage> outputs_1;
    bool done_1 = poll_until(mgr, [&] { return !outputs_1.empty() && outputs_1.back().is_complete; }, outputs_1);
    ASSERT_TRUE(done_1) << "Turn 1 timed out";

    ASSERT_EQ(static_cast<int>(outputs_1.size()), max_new_1);
    EXPECT_EQ(mgr.get_user_state(uid), pm::UserState::COMPLETE);

    // Last prefill token is 300 (position 2), injected as DECODE → output = 301.
    // Loopback: 301 → 302 → 303.
    EXPECT_EQ(outputs_1[0].token_id, 301);
    EXPECT_EQ(outputs_1[1].token_id, 302);
    EXPECT_EQ(outputs_1[2].token_id, 303);

    // ---- Turn 2: CONTINUE ----
    std::vector<int32_t> prompt_2 = {400, 500};
    int32_t max_new_2 = 2;
    ASSERT_TRUE(mgr.push_request(make_continue(req_id++, uid, prompt_2, max_new_2)));
    mgr.tick();

    std::vector<OutputMessage> outputs_2;
    bool done_2 = poll_until(mgr, [&] { return !outputs_2.empty() && outputs_2.back().is_complete; }, outputs_2);
    ASSERT_TRUE(done_2) << "Turn 2 timed out";

    ASSERT_EQ(static_cast<int>(outputs_2.size()), max_new_2);
    EXPECT_EQ(mgr.get_user_state(uid), pm::UserState::COMPLETE);

    // Turn 2 prefills tokens [400, 500] at positions 6, 7.
    // Last prefill token is 500 (position 7), injected as DECODE → output = 501.
    // Loopback: 501 → 502.
    EXPECT_EQ(outputs_2[0].token_id, 501);
    EXPECT_EQ(outputs_2[1].token_id, 502);

    // tokens_generated resets each turn
    EXPECT_EQ(outputs_2[0].tokens_generated, 1);
    EXPECT_EQ(outputs_2[1].tokens_generated, 2);

    // Verify position continuity from the inject log.
    // Filter injections for our user and check positions form a contiguous sequence.
    auto log = mock.get_inject_log();
    std::vector<int32_t> positions;
    for (auto& entry : log) {
        if (entry.user_id == uid) {
            positions.push_back(entry.position);
        }
    }

    // Turn 1: prefill 0,1,2 + decode loopback 3,4 = 5
    // Turn 2: prefill 5,6 + decode loopback 7 = 3
    // Full sequence: 0,1,2,3,4,5,6,7
    ASSERT_EQ(static_cast<int>(positions.size()), 8);
    for (int i = 0; i < static_cast<int>(positions.size()); i++) {
        EXPECT_EQ(positions[i], i) << "Position discontinuity at index " << i;
    }

    mgr.stop();
}

// =============================================================================
//  Test 7: EOS early termination.
//  MockPipeline produces token_id + 1. EOS_TOKEN = 2.
//  If the last prefill token is 1, the first decode output is 2 = EOS.
//  Generation should stop after 1 output despite a large max_new_tokens.
// =============================================================================
TEST(PipelineManagerTest, EosEarlyTermination) {
    pm::MockPipeline mock;
    pm::PipelineManager mgr(mock);
    mgr.start();

    int32_t req_id = 0;

    // Allocate
    ASSERT_TRUE(mgr.push_request(make_allocate(req_id++)));
    mgr.tick();
    PMResponse resp{};
    ASSERT_TRUE(mgr.try_pop_response(resp));
    ASSERT_EQ(resp.error_code, 0);
    int32_t uid = resp.user_id;

    // Prompt ends with token 1. Mock: 1 + 1 = 2 = EOS_TOKEN.
    // max_new_tokens is large — EOS should stop generation early.
    std::vector<int32_t> prompt = {10, 20, 1};
    ASSERT_TRUE(mgr.push_request(make_submit(req_id++, uid, prompt, 1000)));
    mgr.tick();

    std::vector<OutputMessage> outputs;
    bool done = poll_until(mgr, [&] { return !outputs.empty() && outputs.back().is_complete; }, outputs);
    ASSERT_TRUE(done) << "Timed out waiting for EOS completion";

    ASSERT_EQ(static_cast<int>(outputs.size()), 1);
    EXPECT_EQ(outputs[0].token_id, pm::EOS_TOKEN);
    EXPECT_TRUE(outputs[0].is_eos);
    EXPECT_TRUE(outputs[0].is_complete);
    EXPECT_EQ(outputs[0].tokens_generated, 1);
    EXPECT_EQ(mgr.get_user_state(uid), pm::UserState::COMPLETE);

    mgr.stop();
}

// =============================================================================
//  Test 8: Cancel mid-prefill.
//  Submit a max-length prompt and cancel immediately — the writer is on
//  another thread and won't have finished injecting all tokens yet.
//  Verify cleanup completes and the slot is freed.
//  Uses simulated reader latency so in-flight draining takes real time,
//  confirming the reader properly handles cancelled prefill results.
// =============================================================================
TEST(PipelineManagerTest, CancelMidPrefill) {
    pm::MockPipeline mock(10, 50);
    pm::PipelineManager mgr(mock);
    mgr.start();

    int32_t req_id = 0;

    // Allocate
    ASSERT_TRUE(mgr.push_request(make_allocate(req_id++)));
    mgr.tick();
    PMResponse resp{};
    ASSERT_TRUE(mgr.try_pop_response(resp));
    ASSERT_EQ(resp.error_code, 0);
    int32_t uid = resp.user_id;

    // Submit max-length prompt
    std::vector<int32_t> prompt(MAX_SEQ_LEN);
    std::iota(prompt.begin(), prompt.end(), 1000);
    ASSERT_TRUE(mgr.push_request(make_submit(req_id++, uid, prompt, 100)));
    mgr.tick();

    // Cancel immediately — writer is racing through prefill on its thread
    ASSERT_TRUE(mgr.push_request(make_cancel(req_id++, uid)));
    mgr.tick();

    // Wait for cleanup (reader drains in-flight tokens, then finalize)
    std::vector<OutputMessage> outputs;
    bool cleaned_up = poll_until(mgr, [&] { return mgr.get_user_state(uid) == pm::UserState::INACTIVE; }, outputs);
    ASSERT_TRUE(cleaned_up) << "Timed out waiting for mid-prefill cancel cleanup";

    // Count any output tokens that leaked through before cancel took effect.
    // Ideally zero (cancel arrived during prefill), but a few are acceptable
    // if the writer raced past prefill into decode before seeing cancel.
    int user_output_count = 0;
    for (auto& o : outputs) {
        if (o.user_id == uid) {
            user_output_count++;
        }
    }
    std::cout << "[  INFO   ] Output tokens before cancel took effect: " << user_output_count << std::endl;

    // Re-allocate to verify the slot was freed
    ASSERT_TRUE(mgr.push_request(make_allocate(req_id++)));
    mgr.tick();
    PMResponse realloc_resp{};
    ASSERT_TRUE(mgr.try_pop_response(realloc_resp));
    EXPECT_EQ(realloc_resp.error_code, 0);
    EXPECT_GE(realloc_resp.user_id, 0);

    mgr.stop();
}
