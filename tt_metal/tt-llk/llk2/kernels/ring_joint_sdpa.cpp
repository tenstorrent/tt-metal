// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn's ring-joint scaled-dot-product-attention compute kernel (kernels/compute/ring_joint_sdpa.cpp on its streaming
// path, sdpa_ring_v2) on LLK 2.0: the online softmax of sdpa.cpp, over keys that arrive one device's share at a time.
//
// The keys of a sequence are split over the ring's devices, a share each. Ring iteration r brings the share of device
// ring_id (the reader's order: this device's own first), and every Q chunk of this core meets that share's K chunks
// with the m, L and U it ended the iteration before with; the last iteration normalizes them into the output. Within an
// iteration it is sdpa.cpp's flow, unchanged: the same passes, groups and buffers, included from there. What a ring
// adds (sdpa.cpp's RingKRange, with LLK2_SDPA_RING):
//   - The K chunks of a share run up to the sequence's end (logical_nt): those starting past it are not sent, and
//     the last one is narrower when the share's padding or the sequence's end cuts it (as the reader cuts them).
//     When the share holds the sequence's last key tile and it is not whole, the writer's mask tile of it is added
//     onto that tile of S (k_partial_col; the 1.0 kernel's global_n partial tile, index 1 of the mask buffer).
//   - m, L and U go from one iteration to the next. With one Q chunk on the core they stay in the A/B buffers, which
//     the chunks of all iterations take in turn (phase); with more, the writer takes them into DRAM after a Q
//     chunk's last K chunk and brings them back before its first K chunk of the next iteration (staged: the
//     staging buffers cb_max_out / cb_sum_out / cb_out out, and cb_max_in / cb_sum_in / cb_prev_out in, a whole Q
//     chunk each). A restored Q chunk's first group computes its alpha right after its m, as there is no pass before.
//   - An iteration that a later one continues writes m of its last K chunk and keeps L and U as they are: only the
//     last iteration normalizes.
//
// Syncs with the reader and writer, as the 1.0 kernel keeps them:
//   cb_signal      a page per Q chunk with more than one on the core: the writer waits for it before it saves or
//                  writes the Q chunk's state (here pushed when the Q chunk starts: the writer's other waits order it)
//   Q              sent anew every iteration with more than one Q chunk on the core, once with one (popped then at
//                  the end of the last iteration)
//   K and V        after every iteration, the reader sends one K and V more than it used when the iteration used an
//                  even number of them (dummy_kv_chunks_for_phase_alignment): drained here
//   the ring id    advanced once per iteration, active or not (RingSDPAOpIndexer: the compute side waits on nothing)
//
// What the 1.0 kernel does on this path that this one does not: causal, balanced and zigzag, joint Q/K/V, chunked,
// a sliding window, sinks, KV-pad rotation, a logical length from a tensor, a dynamic local length, latent V, the K
// split and segmented accumulation, rotated or padded Q slots, grouped KV sources, a 32-bit Dest. The host (the
// LLK 2.0 branch of ring_joint_sdpa_program_factory.cpp, TT_RING_SDPA_LLK2) refuses them.

#define LLK2_SDPA_RING
#include <cstdint>

#include "sdpa.cpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_rank_mapping.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"

namespace
{

// The ring, as the host's named compile-time args.
constexpr std::uint32_t kv_local_padded_Nt = get_named_compile_time_arg_val("kv_local_padded_Nt"); // key tiles of a share
constexpr std::uint32_t rank_mapping       = get_named_compile_time_arg_val("rank_mapping");       // transport rank -> share
constexpr auto snake_orientation           = static_cast<ttnn::ccl::snake_ring::Orientation>(get_named_compile_time_arg_val("snake_orientation"));
constexpr std::uint32_t mesh_rows          = get_named_compile_time_arg_val("mesh_rows");
constexpr std::uint32_t mesh_cols          = get_named_compile_time_arg_val("mesh_cols");
// Run-time args: the 1.0 kernel's.
constexpr int q_start_arg          = 0; // this core's Q chunks: [q_start, q_end) of all heads' Q chunks
constexpr int q_end_arg            = 1;
constexpr int ring_size_arg        = 2;
constexpr int ring_index_arg       = 3; // this device's transport rank
constexpr int forward_writes_arg   = 4;
constexpr int backward_writes_arg  = 5;
constexpr int logical_nt_arg       = 6;  // the sequence's key tiles, without the padding
constexpr int active_ring_iter_arg = 11; // bit r: iteration r has keys this core's Q chunks meet

static_assert(k_chunks * Sk_chunk_t >= kv_local_padded_Nt, "a share's K chunks hold its key tiles");
static_assert(
    LLK2_CB_FORMAT("cb_max_in") == LLK2_CB_FORMAT("cb_max_A") && LLK2_CB_FORMAT("cb_max_out") == LLK2_CB_FORMAT("cb_max_A") &&
        LLK2_CB_FORMAT("cb_sum_in") == LLK2_CB_FORMAT("cb_sum_A") && LLK2_CB_FORMAT("cb_sum_out") == LLK2_CB_FORMAT("cb_sum_A") &&
        LLK2_CB_FORMAT("cb_prev_out") == LLK2_CB_FORMAT("cb_out_im_A") && LLK2_CB_FORMAT("cb_out") == LLK2_CB_FORMAT("cb_out_im_A"),
    "a statistic's staging buffers hold the tensor its A/B buffers hold");

constexpr const auto& signal_spec = words_spec<8, LLK2_CB_FORMAT("cb_signal")>; // one 16-byte page, its words unread

/// The K chunks of share ring_id that a Q chunk meets, in an iteration the flags describe.
inline RingKRange share_range(std::uint32_t ring_id, std::uint32_t logical_nt, std::uint32_t phase, bool restored, bool continues, bool staged)
{
    const std::int32_t within  = static_cast<std::int32_t>(logical_nt) - static_cast<std::int32_t>(ring_id * kv_local_padded_Nt);
    const std::uint32_t tiles  = within <= 0 ? 0 : (static_cast<std::uint32_t>(within) < kv_local_padded_Nt ? within : kv_local_padded_Nt);
    const std::uint32_t chunks = (tiles + Sk_chunk_t - 1) / Sk_chunk_t;
    const bool partial_key     = partial_key_cols != 0 && within > 0 && static_cast<std::uint32_t>(within) <= kv_local_padded_Nt;
    return RingKRange {chunks, chunks == 0 ? 0 : tiles - (chunks - 1) * Sk_chunk_t, phase, restored, continues, staged, partial_key};
}

/// alpha(g) = exp(scale (m_prev - m)), m(g) being at the front of row_maxes and m_prev at the front of the buffer g
/// reads it from (popped here): its exp on the pack thread.
inline __attribute__((always_inline)) void alpha_of(const Group& g, Carried& m, DataflowBuffer& row_maxes, DataflowBuffer& alphas)
{
    const auto m_prev = wait_front<carried_m_spec>(m.read_by(g));
    const auto diff   = sub(m_prev, peek_front<m_spec>(row_maxes), reserve_back<StatRing>());
    const auto a      = vec_exp<exp_on_pack>(diff);
    release(pack(a, reserve_back<alpha_spec>(alphas)));
    release(a);
    release(m_prev);
}

} // namespace

void kernel_main()
{
    DataflowBuffer q_in(get_named_compile_time_arg_val("cb_q_in"));
    DataflowBuffer k_in(get_named_compile_time_arg_val("cb_k_in"));
    DataflowBuffer v_in(get_named_compile_time_arg_val("cb_v_in"));
    DataflowBuffer unit_in(get_named_compile_time_arg_val("cb_identity_scale_in"));
    DataflowBuffer scores(get_named_compile_time_arg_val("cb_scores"));
    DataflowBuffer probabilities(get_named_compile_time_arg_val("cb_probabilities"));
    DataflowBuffer row_maxes(get_named_compile_time_arg_val("cb_row_max"));
    DataflowBuffer alphas(get_named_compile_time_arg_val("cb_alpha"));
    DataflowBuffer inverse(get_named_compile_time_arg_val("cb_inverse"));
    DataflowBuffer out(get_named_compile_time_arg_val("cb_out"));
    DataflowBuffer signal(get_named_compile_time_arg_val("cb_signal"));
    Carried m {
        DataflowBuffer(get_named_compile_time_arg_val("cb_max_A")),
        DataflowBuffer(get_named_compile_time_arg_val("cb_max_B")),
        DataflowBuffer(get_named_compile_time_arg_val("cb_max_in")),
        DataflowBuffer(get_named_compile_time_arg_val("cb_max_out"))};
    Carried l {
        DataflowBuffer(get_named_compile_time_arg_val("cb_sum_A")),
        DataflowBuffer(get_named_compile_time_arg_val("cb_sum_B")),
        DataflowBuffer(get_named_compile_time_arg_val("cb_sum_in")),
        DataflowBuffer(get_named_compile_time_arg_val("cb_sum_out"))};
    // U's staging: cb_out out (the writer saves it from there, and writes the output from there), cb_prev_out in.
    Carried u {
        DataflowBuffer(get_named_compile_time_arg_val("cb_out_im_A")),
        DataflowBuffer(get_named_compile_time_arg_val("cb_out_im_B")),
        DataflowBuffer(get_named_compile_time_arg_val("cb_prev_out")),
        DataflowBuffer(get_named_compile_time_arg_val("cb_out"))};
    buffer_if<DataflowBuffer, false> mask_in(get_named_compile_time_arg_val("cb_mask_in")); // no mask tensor in a ring
    buffer_if<DataflowBuffer, made_masks_used> made_masks_in(get_named_compile_time_arg_val("cb_mask_in"));
    buffer_if<DataflowBuffer, false> sink_terms(get_named_compile_time_arg_val("cb_sink_term"));

    const std::uint32_t q_start           = get_arg_val<std::uint32_t>(q_start_arg);
    const std::uint32_t q_chunks          = get_arg_val<std::uint32_t>(q_end_arg) - q_start;
    const std::uint32_t ring_size         = get_arg_val<std::uint32_t>(ring_size_arg);
    const std::uint32_t logical_nt        = get_arg_val<std::uint32_t>(logical_nt_arg);
    const std::uint32_t active_ring_iters = get_arg_val<std::uint32_t>(active_ring_iter_arg);
    RingSDPAOpIndexer ring_ids(
        ring_size, get_arg_val<std::uint32_t>(ring_index_arg), get_arg_val<std::uint32_t>(forward_writes_arg), get_arg_val<std::uint32_t>(backward_writes_arg));
    // Pushed once by the writer and never popped (as 1.0): the reduce's scaler, and the mask tiles it makes.
    const auto unit       = peek_front<unit_spec>(unit_in);
    const auto made_masks = peek_front<made_masks_spec>(made_masks_in);

    const bool staged   = q_chunks > 1; // the writer takes m, L and U through DRAM between iterations
    bool seen_active    = false;
    std::uint32_t phase = 0;
    for (std::uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter)
    {
        const std::uint32_t ring_id = ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<rank_mapping>(
            ring_ids.get_next_ring_id_and_sync(), mesh_rows, mesh_cols, snake_orientation);
        if (((active_ring_iters >> ring_iter) & 1u) == 0)
        {
            continue;
        }
        const bool restored    = seen_active;
        const bool continues   = !is_last_active_ring_iter(active_ring_iters, ring_iter);
        seen_active            = true;
        const RingKRange range = share_range(ring_id, logical_nt, staged ? 0 : phase, restored, continues, staged);
        LLK_ASSERT(range.end_chunk > 0, "an active iteration has keys");

        for (std::uint32_t q_chunk = 0; q_chunk < q_chunks; ++q_chunk)
        {
            if (staged)
            {
                release(reserve_back<signal_spec>(signal)); // the writer may save or write this Q chunk's state
            }

            // S and m of the first group, and its alpha when it goes on from an earlier iteration.
            {
                const Group first = Group::opening(range);
                AbsentBuffer no_group(0);
                probabilities_and_scores(first, first, absent_back<l_spec>(), made_masks, q_in, k_in, mask_in, no_group, no_group, no_group, scores);
                running_row_max(first, unit, scores, row_maxes, m);
                if (!first.first())
                {
                    alpha_of(first, m, row_maxes, alphas);
                }
            }

            for (Group cur = Group::opening(range); cur.exists(); cur = cur.next())
            {
                const Group next    = cur.next(); // its S, m and alpha are computed in this pass
                const bool has_next = next.exists();
                const bool carried  = !cur.first();       // cur has m, L and U from the K chunk (or iteration) before
                const bool kv_done  = next.opens_chunk(); // cur ends its K chunk, and K and V go

                if (kv_done)
                {
                    pop_front<kt_spec>(k_in); // every S of cur's K chunk is issued
                }

                // P(cur) and S(next), block by block.
                const auto l_slot = reserve_back<l_spec>(l.written_by(cur));
                probabilities_and_scores(cur, next, l_slot, made_masks, q_in, k_in, mask_in, scores, row_maxes, probabilities, scores);

                if (has_next)
                {
                    running_row_max(next, unit, scores, row_maxes, m);
                }

                // L(cur) = alpha L_prev + the row sums the P blocks left in l_slot.
                const auto alpha = peek_front<alpha_spec>(alphas, carried);
                if (carried)
                {
                    const auto l_prev = wait_front<l_spec>(l.read_by(cur));
                    const auto scaled = mul<times_column>(l_prev, alpha, reserve_back<StatRing>());
                    pack<added>(scaled, l_slot);
                    release(scaled);
                    release(l_prev);
                }
                release(l_slot);

                // U(cur) = P V + alpha U_prev: all but the last rows, alpha(next) beside the last ones, then those.
                const auto p      = wait_front<p_spec>(probabilities);
                const auto v      = peek_front<v_spec>(v_in);
                const auto u_prev = peek_front<u_spec>(u.read_by(cur), carried);
                const auto u_slot = reserve_back<u_spec>(u.written_by(cur));
                for (std::uint32_t r0 = 0; r0 < group_rows - u_last_rows; r0 += u_rows)
                {
                    numerator_rows<u_rows>(cur.width(), carried, r0, p, v, u_prev, alpha, u_slot);
                }
                if (has_next && !next.first())
                {
                    alpha_of(next, m, row_maxes, alphas);
                }
                numerator_rows<u_last_rows>(cur.width(), carried, group_rows - u_last_rows, p, v, u_prev, alpha, u_slot);
                release(p);
                release(u_slot);
                if (carried)
                {
                    pop_front<alpha_spec>(alphas);
                    pop_front<u_spec>(u.read_by(cur));
                }
                if (kv_done)
                {
                    pop_front<v_spec>(v_in);
                }

                if (cur.last() && !cur.continues)
                {
                    output_rows(unit, l.written_by(cur), u.written_by(cur), inverse, out, sink_terms);
                }
            }
            if (staged || !continues)
            {
                pop_front<q_spec>(q_in); // sent anew next iteration, or this was the last
            }
        }

        if (!staged)
        {
            phase = (phase + range.end_chunk) % 2; // the next iteration's first chunk reads what this one's last wrote
        }
        // The reader's K and V to keep the K and V buffers' phase: one more when this iteration used an even number.
        if (dummy_kv_chunks_for_phase_alignment<false>(q_chunks * range.end_chunk) != 0)
        {
            release(wait_front<kt_spec>(k_in));
            release(wait_front<v_spec>(v_in));
        }
    }
}
