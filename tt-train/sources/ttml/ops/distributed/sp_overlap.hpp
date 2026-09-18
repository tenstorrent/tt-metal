// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <deque>
#include <functional>
#include <vector>

namespace ttml::ops::distributed {

// Two-stream scheduling of the sequence-parallel linears' backward (ops/distributed/sp_linear_ops.hpp, when the
// backward runs SPLinearImpl::Composed -- the forward may run Fused, see set_sp_linear_backward_impl): the
// collective of a linear's backward runs on the second command queue and
// the CCL sub-device while a matmul that does not depend on it runs on the compute queue. Nothing is
// computed differently -- the same ttnn ops, only issued on another queue and in another order -- so the
// results are bit-identical to the un-overlapped Composed sequence on the same core grid.
//
// What overlaps what. In a linear's backward the weight gradient never depends on the collective, and the
// input gradient's collective is either after its matmul (column parallel: dgrad = reduce_scatter(grad @ W))
// or before it (row parallel: dgrad = all_gather(grad) @ W). So:
//   * every linear DEFERS its weight gradient instead of issuing it;
//   * every collective is issued on the CCL queue and immediately followed by ONE deferred weight gradient
//     on the compute queue -- the previous linear's -- so the collective runs behind a matmul of the same
//     magnitude (in the backward order of a Llama block: w2's all-gather behind the previous block's qkv
//     wgrad, gate_up's reduce-scatter behind w2's wgrad, out_proj's all-gather behind gate_up's wgrad, qkv's
//     reduce-scatter behind out_proj's wgrad);
//   * the outermost Tensor::backward() drains what is left before it returns (AutoContext backward-end hook).
//
// Ordering between the queues is by device events, both ways. Before a collective is issued, the CCL queue
// waits for an event recorded on the compute queue (a "compute drain": the collective's input is complete,
// and every buffer the host has freed so far is really free on this device -- the host allocator runs far
// ahead of the device and is not ordered across queues). Before compute consumes a collective's output, the
// compute queue waits for the event recorded after it. Every tensor a collective reads, writes or stages
// through is kept referenced until two later collectives have been waited for; freeing it earlier lets the
// host hand its address to a compute-queue op while a slower peer's copy of the collective may still be in
// flight (that shows as rare bf16-level drift, not as a crash).
enum class SPOverlapMode { Off, Backward };

// Process-wide switch. Backward needs the device opened with two command queues and the CCL sub-device
// enabled (AutoContext::enable_ccl_sub_device); it only affects SPLinearImpl::Composed.
void set_sp_overlap_mode(SPOverlapMode mode);
SPOverlapMode get_sp_overlap_mode();

class SPOverlap {
public:
    static SPOverlap& instance();

    // True while a grad function of a Composed SP linear should schedule across the two queues.
    [[nodiscard]] bool backward_active() const;

    struct Collective {
        ttnn::Tensor output;
        tt::tt_metal::distributed::MeshEvent done;  // recorded on the CCL queue right after the collective
        std::vector<ttnn::Tensor> buffers;          // everything the collective reads, writes or stages through
    };
    // Compute drain, then `collective` issued on the CCL queue, then the event. `reads` are the tensors the
    // collective consumes; buffers the callee allocates for it (a reduce-scatter's staging tensors) go into
    // the vector it is handed; the output is kept too.
    Collective issue(
        const std::function<ttnn::Tensor(std::vector<ttnn::Tensor>& keep)>& collective, std::vector<ttnn::Tensor> reads);
    // The compute queue waits for `collective`; its buffers are released two waited collectives later.
    void wait(Collective collective);

    // Work held back so a later collective has something to hide behind; drained one per collective and
    // completely at the end of the outermost backward.
    void defer(std::function<void()> work);
    void drain_one();
    void drain_all();

    [[nodiscard]] size_t num_deferred() const;
    [[nodiscard]] size_t num_retained() const;

private:
    SPOverlap();
    void compute_drain();
    void finish_backward();
    void teardown(bool device_closing);

    std::deque<std::function<void()>> m_deferred;
    std::deque<std::vector<ttnn::Tensor>> m_retire;  // buffers of waited collectives, oldest first
};

}  // namespace ttml::ops::distributed
