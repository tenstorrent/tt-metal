// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal reproducer for the intra-step CCL semaphore-pool-wrap hang.
//
// Background
// ----------
// The Galaxy Llama 8B TP=4 DDP=8 training run hangs non-deterministically on
// some runners. It was bisected to the rotating GlobalSemaphore pool in
// CCLResources (`kNumSemaphoresPairs`): every collective draws a semaphore set
// from a small ring buffer that wraps modulo the pool depth. With depth 2, once
// more than two collectives that reuse the reduce_scatter/all_gather semaphores
// are in flight, a new collective grabs a semaphore set whose previous collective
// hasn't fully quiesced -> the ring/linear reduce_scatter writer wedges. Bumping
// `kNumSemaphoresPairs` from 2 to 8 made the full 2048-step run pass.
//
// Why the earlier isolated stress test missed it
// ----------------------------------------------
// The bug is about collective *overlap depth*, not collective size. A test that
// fires one big collective and reads it back keeps <=1-2 in flight, so the
// depth-2 pool never wraps under contention. To reproduce you must issue many
// back-to-back collectives through the *shared* CclResources pool with no drain
// between them (matching RowParallelLinear's per-block all_reduce, ~2 per block),
// and only drain once per "step" (like the trainer reading the loss scalar), so
// the intra-step collectives overlap.
//
// What this test does
// -------------------
// Recreates that pattern with no model: a chain of `collectives_per_step`
// `all_reduce`s on the TP axis (each followed by a cheap rescale so values stay
// bounded and there is a little compute between collectives), repeated for
// `num_steps`, draining exactly once per step. On the full 8x4 Galaxy with the
// torus (ring_ring) MGD and `kNumSemaphoresPairs=2` this is expected to hang
// (the harness `timeout` turns it into a failure); with `kNumSemaphoresPairs=8`
// it should complete all steps.
//
// It is opt-in (needs a real 32-device Galaxy + fabric) and gated behind
// TT_TRAIN_CCL_STRESS so it never runs in the normal unit-test sweep. The fabric
// topology (ring vs line) is whatever TT_MESH_GRAPH_DESC_PATH points at, so the
// same binary can A/B ring_ring vs line_line.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace {

uint32_t get_env_u32(const char* name, uint32_t default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
}

class GalaxyCCLStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (std::getenv("TT_TRAIN_CCL_STRESS") == nullptr) {
            GTEST_SKIP() << "Set TT_TRAIN_CCL_STRESS=1 (on a 32-device Galaxy) to run the "
                            "CCL semaphore-pool stress reproducer.";
        }
        // Fabric config (FABRIC_2D vs FABRIC_2D_TORUS_*) is inferred from the MGD
        // pointed to by TT_MESH_GRAPH_DESC_PATH, so ring_ring vs line_line is
        // selected purely by that env var -- the same binary reproduces on both.
        ttml::ttnn_fixed::distributed::enable_fabric(32U);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(8, 4));
        m_opened = true;
    }

    void TearDown() override {
        if (m_opened) {
            ttml::autograd::ctx().close_device();
            m_opened = false;
        }
    }

private:
    bool m_opened = false;
};

}  // namespace

TEST_F(GalaxyCCLStressTest, ReduceScatterSemaphorePoolWrapHang) {
    // Defaults mirror the failing run: 32 blocks x ~2 row-parallel all_reduce =
    // ~64 collectives/step, 2048 steps. All overridable so the repro window can be
    // widened without recompiling.
    const uint32_t num_steps = get_env_u32("TT_TRAIN_CCL_STRESS_STEPS", 2048U);
    const uint32_t collectives_per_step = get_env_u32("TT_TRAIN_CCL_STRESS_COLLECTIVES", 64U);
    const uint32_t hidden = get_env_u32("TT_TRAIN_CCL_STRESS_HIDDEN", 2048U);
    const uint32_t seq = get_env_u32("TT_TRAIN_CCL_STRESS_SEQ", 32U);
    // TP axis on the 8x4 mesh is axis 1 (size 4); axis 0 (size 8) is DDP.
    const uint32_t tp_axis = get_env_u32("TT_TRAIN_CCL_STRESS_TP_AXIS", 1U);

    auto* device = &ttml::autograd::ctx().get_device();
    const auto& mesh_shape = device->shape();
    ASSERT_LT(tp_axis, mesh_shape.dims()) << "tp_axis out of range for mesh " << mesh_shape;
    const uint32_t tp_size = mesh_shape[tp_axis];
    ASSERT_GT(tp_size, 1U) << "TP axis must span >1 device for all_reduce to communicate";

    // Replicated [1, 1, seq, hidden] tensor. all_reduce along tp_axis sums across
    // the TP group; rescaling by 1/tp_size keeps the value ~fixed across the long
    // chain (numerically sane) while still doing a real collective + a little
    // compute between collectives (like the matmul between block all_reduces).
    std::vector<float> data(static_cast<size_t>(seq) * static_cast<size_t>(hidden), 1.0F);
    auto shape = ttnn::Shape({1U, 1U, seq, hidden});
    auto base = ttml::core::from_vector(data, shape, device);

    fmt::print(
        "[ccl-stress] mesh={} tp_axis={} tp_size={} steps={} collectives/step={} shape=[1,1,{},{}]\n",
        mesh_shape,
        tp_axis,
        tp_size,
        num_steps,
        collectives_per_step,
        seq,
        hidden);
    std::fflush(stdout);

    const float inv_tp = 1.0F / static_cast<float>(tp_size);
    for (uint32_t step = 0; step < num_steps; ++step) {
        auto acc = base;
        for (uint32_t c = 0; c < collectives_per_step; ++c) {
            acc = ttml::ttnn_fixed::distributed::all_reduce(acc, tp_axis);
            acc = ttnn::multiply(acc, inv_tp);
        }
        // Exactly one drain per step (mirrors the trainer reading the loss): bounds
        // host run-ahead to ~1 step but does NOT serialize the intra-step
        // collectives, so the overlap that trips the depth-2 pool is preserved. If
        // this readback (or an all_reduce above) hangs, the harness timeout turns
        // it into a failure -- that IS the reproduction.
        auto probe = ttml::core::to_xtensor<float>(acc, ttml::core::IdentityComposer{});
        volatile float sink = probe[0](0, 0, 0, 0);
        (void)sink;

        if (step % 10U == 0U) {
            fmt::print("[ccl-stress] step {} ok\n", step);
            std::fflush(stdout);
        }
    }

    fmt::print("[ccl-stress] completed all {} steps with no hang\n", num_steps);
    std::fflush(stdout);
    SUCCEED();
}
