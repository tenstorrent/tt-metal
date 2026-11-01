"""Iteration 27: per-round chain stream bounds. Per chain and round the stream carries only the chunks the round's
items need (forward / Q backward: up to the heaviest row's diagonal chunk; KV backward: from the lightest item's
K row on), instead of the whole sequence every round. Apply from the tt-metal root; the compute per row is unchanged
(bit-exact expected)."""
import re
import sys

M = "tt-train/sources/ttml/metal/ops"
FWR, FWW, FWC = (
    f"{M}/sdpa_fw/device/kernels/dataflow/sdpa_fw_reader_kernel.cpp",
    f"{M}/sdpa_fw/device/kernels/dataflow/sdpa_fw_writer_kernel.cpp",
    f"{M}/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp",
)
QR, QW, QC = (
    f"{M}/sdpa_bw/device/kernels/dataflow/sdpa_bw_q_reader_kernel.cpp",
    f"{M}/sdpa_bw/device/kernels/dataflow/sdpa_bw_q_writer_kernel.cpp",
    f"{M}/sdpa_bw/device/kernels/compute/sdpa_bw_q_compute_kernel.cpp",
)
KR, KW, KC = (
    f"{M}/sdpa_bw/device/kernels/dataflow/sdpa_bw_kv_reader_kernel.cpp",
    f"{M}/sdpa_bw/device/kernels/dataflow/sdpa_bw_kv_writer_kernel.cpp",
    f"{M}/sdpa_bw/device/kernels/compute/sdpa_bw_kv_compute_kernel.cpp",
)
FF = f"{M}/sdpa_fw/device/sdpa_fw_program_factory.cpp"
QF = f"{M}/sdpa_bw/device/sdpa_bw_q_program_factory.cpp"
KF = f"{M}/sdpa_bw/device/sdpa_bw_kv_program_factory.cpp"


def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        print(f"FAILED {path}: {old[:90]!r} found {n}, expected {count}")
        sys.exit(2)
    open(path, "w").write(s.replace(old, new))


# ============ forward reader
sub(
    FWR,
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rows_base = runtime_args_counter;
""",
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rows_base = runtime_args_counter;
    const uint32_t chunks_base = rows_base + num_rounds;  // per round: the chunks the chain streams (iteration 27)
""",
)
sub(
    FWR,
    """    const uint32_t total_chunks = num_rounds * num_chunks;
    uint32_t credited = 0;  // chunks credited to the previous core (non-injectors)
    for (uint32_t round = 0; round < num_rounds; ++round) {
""",
    """    uint32_t total_chunks = 0;
    for (uint32_t round = 0; round < num_rounds; ++round) {
        total_chunks += get_arg_val<uint32_t>(chunks_base + round);
    }
    uint32_t credited = 0;  // chunks credited to the previous core (non-injectors)
    uint32_t c = 0;         // chunk counter over the rounds (CB slot = c % kChainSlots)
    for (uint32_t round = 0; round < num_rounds; ++round) {
""",
)
sub(
    FWR,
    """        for (uint32_t k_chunk = 0; k_chunk < num_chunks; ++k_chunk) {
            const uint32_t c = round * num_chunks + k_chunk;
""",
    """        const uint32_t round_chunks = get_arg_val<uint32_t>(chunks_base + round);
        for (uint32_t k_chunk = 0; k_chunk < round_chunks; ++k_chunk, ++c) {
""",
)
# ============ forward writer
sub(
    FWW,
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rows_base = runtime_args_counter;
""",
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rows_base = runtime_args_counter;
    const uint32_t chunks_base = rows_base + num_rounds;  // per round: the chunks the chain streams (iteration 27)
""",
)
sub(
    FWW,
    """    constexpr uint32_t num_chunks = Ht / Sk_chunk_t;
    for (uint32_t round = 0; round < num_rounds; ++round) {
        for (uint32_t h = 0; h < num_chunks; ++h) {
            const uint32_t c = round * num_chunks + h;
""",
    """    uint32_t c = 0;  // chunk counter over the rounds (CB slot = c % kChainSlots)
    for (uint32_t round = 0; round < num_rounds; ++round) {
        const uint32_t round_chunks = get_arg_val<uint32_t>(chunks_base + round);
        for (uint32_t h = 0; h < round_chunks; ++h, ++c) {
""",
)
# ============ forward compute
sub(
    FWC,
    "FORCE_INLINE void process_single_row(uint32_t global_row_idx) {",
    "// `round_chunks`: chain mode streams that many K/V chunks this round (iteration 27); the ones past the row's\n"
    "// diagonal are dropped.\nFORCE_INLINE void process_single_row(uint32_t global_row_idx, uint32_t round_chunks = Ht / Sk_chunk_t) {",
)
sub(
    FWC,
    """    // Chain mode streams every chunk of the sequence through this core: drop the ones past the diagonal.
    for (uint32_t k_chunk = num_kv_chunks; k_chunk < Ht / Sk_chunk_t; ++k_chunk) {""",
    """    // Chain mode streams the round's chunks through this core: drop the ones past the diagonal.
    for (uint32_t k_chunk = num_kv_chunks; k_chunk < round_chunks; ++k_chunk) {""",
)
sub(
    FWC,
    """    // Chain mode: the same list padded with 0xFFFFFFFF rounds in which this core only consumes the K/V stream.
    for (uint32_t i = 0; i < start_idx; ++i) {
        const uint32_t row = get_arg_val<uint32_t>(1U + i);
#ifdef CHAIN
        if (row == 0xFFFFFFFFU) {
            for (uint32_t k_chunk = 0; k_chunk < Ht / Sk_chunk_t; ++k_chunk) {
                cb_wait_front(cb_key, Sk_chunk_t * qWt);
                cb_pop_front(cb_key, Sk_chunk_t * qWt);
                cb_wait_front(cb_value, Sk_chunk_t * vWt);
                cb_pop_front(cb_value, Sk_chunk_t * vWt);
            }
            continue;
        }
#endif
        process_single_row(row);
    }
""",
    """    // Chain mode: the same list padded with 0xFFFFFFFF rounds in which this core only consumes the K/V stream,
    // then per round the chunk count the chain streams (iteration 27).
    for (uint32_t i = 0; i < start_idx; ++i) {
        const uint32_t row = get_arg_val<uint32_t>(1U + i);
#ifdef CHAIN
        const uint32_t round_chunks = get_arg_val<uint32_t>(1U + start_idx + i);
        if (row == 0xFFFFFFFFU) {
            for (uint32_t k_chunk = 0; k_chunk < round_chunks; ++k_chunk) {
                cb_wait_front(cb_key, Sk_chunk_t * qWt);
                cb_pop_front(cb_key, Sk_chunk_t * qWt);
                cb_wait_front(cb_value, Sk_chunk_t * vWt);
                cb_pop_front(cb_value, Sk_chunk_t * vWt);
            }
            continue;
        }
        process_single_row(row, round_chunks);
#else
        process_single_row(row);
#endif
    }
""",
)
# ============ Q reader
sub(
    QR,
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rows_base = runtime_args_counter;
""",
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rows_base = runtime_args_counter;
    const uint32_t chunks_base = rows_base + num_rounds;  // per round: the chunks the chain streams (iteration 27)
""",
)
sub(
    QR,
    """    const uint32_t total_chunks = num_rounds * chunks_per_round;
    uint32_t credited = 0;  // chunks credited to the previous core (non-injectors)
    for (uint32_t round = 0; round < num_rounds; ++round) {
""",
    """    uint32_t total_chunks = 0;
    for (uint32_t round = 0; round < num_rounds; ++round) {
        total_chunks += get_arg_val<uint32_t>(chunks_base + round);
    }
    uint32_t credited = 0;  // chunks credited to the previous core (non-injectors)
    uint32_t c = 0;         // chunk counter over the rounds (CB slot = c % kChainSlots)
    for (uint32_t round = 0; round < num_rounds; ++round) {
""",
)
sub(
    QR,
    """        for (uint32_t h0 = 0; h0 < Ht; h0 += chain_chunk) {
            const uint32_t c = round * chunks_per_round + h0 / chain_chunk;
""",
    """        const uint32_t round_rows = get_arg_val<uint32_t>(chunks_base + round) * chain_chunk;
        for (uint32_t h0 = 0; h0 < round_rows; h0 += chain_chunk, ++c) {
""",
)
# ============ Q writer
sub(
    QW,
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rows_base = runtime_args_counter;
""",
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rows_base = runtime_args_counter;
    const uint32_t chunks_base = rows_base + num_rounds;  // per round: the chunks the chain streams (iteration 27)
""",
)
sub(
    QW,
    """    constexpr uint32_t chunks_per_round = Ht / chain_chunk;
    for (uint32_t round = 0; round < num_rounds; ++round) {
        for (uint32_t h = 0; h < chunks_per_round; ++h) {
            const uint32_t c = round * chunks_per_round + h;
""",
    """    uint32_t c = 0;  // chunk counter over the rounds (CB slot = c % kChainSlots)
    for (uint32_t round = 0; round < num_rounds; ++round) {
        const uint32_t round_chunks = get_arg_val<uint32_t>(chunks_base + round);
        for (uint32_t h = 0; h < round_chunks; ++h, ++c) {
""",
)
# ============ Q compute
sub(
    QC,
    "FORCE_INLINE void process_single_row(uint32_t global_row_idx) {",
    "// `round_chunks`: chain mode streams that many K/V chunks this round (iteration 27); the ones past the row's\n"
    "// diagonal are dropped.\nFORCE_INLINE void process_single_row(uint32_t global_row_idx, uint32_t round_chunks = Ht / kv_block) {",
)
sub(
    QC,
    """    // Chain mode streams every K/V chunk of the sequence through this core: drop the ones past the diagonal.
    for (uint32_t h = ((num_kv_tiles_to_process + kv_block - 1U) / kv_block) * kv_block; h < Ht; h += kv_block) {""",
    """    // Chain mode streams the round's K/V chunks through this core: drop the ones past the diagonal.
    for (uint32_t h = ((num_kv_tiles_to_process + kv_block - 1U) / kv_block) * kv_block; h < round_chunks * kv_block;
         h += kv_block) {""",
)
sub(
    QC,
    """    // Chain mode: the same list padded with 0xFFFFFFFF rounds in which this core only consumes the K/V stream.
    for (uint32_t i = 0; i < start_idx; ++i) {
        const uint32_t row = get_arg_val<uint32_t>(1U + i);
#ifdef CHAIN
        if (row == 0xFFFFFFFFU) {
            for (uint32_t h = 0; h < Ht; h += kv_block) {
                cb_wait_front(cb_key, kv_block * qk_tiles);
                cb_pop_front(cb_key, kv_block * qk_tiles);
                cb_wait_front(cb_value, kv_block * v_tiles);
                cb_pop_front(cb_value, kv_block * v_tiles);
            }
            continue;
        }
#endif
        process_single_row(row);
    }
""",
    """    // Chain mode: the same list padded with 0xFFFFFFFF rounds in which this core only consumes the K/V stream,
    // then per round the chunk count the chain streams (iteration 27).
    for (uint32_t i = 0; i < start_idx; ++i) {
        const uint32_t row = get_arg_val<uint32_t>(1U + i);
#ifdef CHAIN
        const uint32_t round_chunks = get_arg_val<uint32_t>(1U + start_idx + i);
        if (row == 0xFFFFFFFFU) {
            for (uint32_t h = 0; h < round_chunks * kv_block; h += kv_block) {
                cb_wait_front(cb_key, kv_block * qk_tiles);
                cb_pop_front(cb_key, kv_block * qk_tiles);
                cb_wait_front(cb_value, kv_block * v_tiles);
                cb_pop_front(cb_value, kv_block * v_tiles);
            }
            continue;
        }
        process_single_row(row, round_chunks);
#else
        process_single_row(row);
#endif
    }
""",
)
# ============ KV reader
sub(
    KR,
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t items_base = runtime_args_counter;
""",
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t items_base = runtime_args_counter;
    const uint32_t starts_base = items_base + num_rounds;  // per round: the first query row the chain streams (iteration 27)
""",
)
sub(
    KR,
    """    const uint32_t total_chunks = num_rounds * chunks_per_round;
    uint32_t credited = 0;  // chunks credited to the previous core (non-injectors)
""",
    """    uint32_t total_chunks = 0;
    for (uint32_t round = 0; round < num_rounds; ++round) {
        total_chunks += (Ht - get_arg_val<uint32_t>(starts_base + round)) / chain_chunk;
    }
    uint32_t credited = 0;  // chunks credited to the previous core (non-injectors)
    uint32_t c = 0;         // chunk counter over the rounds (CB slot = c % kChainSlots)
""",
)
sub(
    KR,
    """        for (uint32_t h0 = 0; h0 < Ht; h0 += chain_chunk) {
            const uint32_t c = round * chunks_per_round + h0 / chain_chunk;
""",
    """        const uint32_t round_start = get_arg_val<uint32_t>(starts_base + round);
        for (uint32_t h0 = round_start; h0 < Ht; h0 += chain_chunk, ++c) {
""",
)
# ============ KV writer
sub(
    KW,
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t items_base = runtime_args_counter;
""",
    """    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t items_base = runtime_args_counter;
    const uint32_t starts_base = items_base + num_rounds;  // per round: the first query row the chain streams (iteration 27)
""",
)
sub(
    KW,
    """    constexpr uint32_t chunks_per_round = Ht / chain_chunk;
    for (uint32_t round = 0; round < num_rounds; ++round) {
        for (uint32_t h = 0; h < chunks_per_round; ++h) {
            const uint32_t c = round * chunks_per_round + h;
""",
    """    uint32_t c = 0;  // chunk counter over the rounds (CB slot = c % kChainSlots)
    for (uint32_t round = 0; round < num_rounds; ++round) {
        const uint32_t round_start = get_arg_val<uint32_t>(starts_base + round);
        for (uint32_t h = round_start; h < Ht; h += chain_chunk, ++c) {
""",
)
# ============ KV compute
sub(
    KC,
    "FORCE_INLINE void process_row_heads(uint32_t global_row_idx, uint32_t num_heads) {",
    "// `stream_start`: chain mode streams the query rows from this one on this round (iteration 27); the ones above\n"
    "// the diagonal are dropped.\nFORCE_INLINE void process_row_heads(uint32_t global_row_idx, uint32_t num_heads, uint32_t stream_start = 0) {",
)
sub(
    KC,
    """    // Chain mode streams every query row of the head through this core: drop the ones above the diagonal.
    for (uint32_t h = 0; h < q_start_tile; ++h) {""",
    """    // Chain mode streams the round's query rows through this core: drop the ones above the diagonal.
    for (uint32_t h = stream_start; h < q_start_tile; ++h) {""",
)
sub(
    KC,
    """    for (uint32_t i = 0; i < start_idx; ++i) {
        const uint32_t item = get_arg_val<uint32_t>(1U + i);
#ifdef CHAIN
        if (item == 0xFFFFFFFFU) {  // a round without an item: consume the stream
            for (uint32_t h = 0; h < Ht; ++h) {
                cb_wait_front(cb_query, qk_tiles);
                cb_pop_front(cb_query, qk_tiles);
                cb_wait_front(cb_grad_output, v_tiles);
                cb_pop_front(cb_grad_output, v_tiles);
                cb_wait_front(cb_intermediates, num_of_interm_tiles);
                cb_pop_front(cb_intermediates, num_of_interm_tiles);
                cb_wait_front(cb_u_scalar_row, onetile);
                cb_pop_front(cb_u_scalar_row, onetile);
            }
            continue;
        }
#endif
        process_row_heads(item / items_per_row, heads_per_item);
    }
""",
    """    for (uint32_t i = 0; i < start_idx; ++i) {
        const uint32_t item = get_arg_val<uint32_t>(1U + i);
#ifdef CHAIN
        // Per round the first query row the chain streams (iteration 27).
        const uint32_t round_start = get_arg_val<uint32_t>(1U + start_idx + i);
        if (item == 0xFFFFFFFFU) {  // a round without an item: consume the stream
            for (uint32_t h = round_start; h < Ht; ++h) {
                cb_wait_front(cb_query, qk_tiles);
                cb_pop_front(cb_query, qk_tiles);
                cb_wait_front(cb_grad_output, v_tiles);
                cb_pop_front(cb_grad_output, v_tiles);
                cb_wait_front(cb_intermediates, num_of_interm_tiles);
                cb_pop_front(cb_intermediates, num_of_interm_tiles);
                cb_wait_front(cb_u_scalar_row, onetile);
                cb_pop_front(cb_u_scalar_row, onetile);
            }
            continue;
        }
        process_row_heads(item / items_per_row, heads_per_item, round_start);
#else
        process_row_heads(item / items_per_row, heads_per_item);
#endif
    }
""",
)
# ============ forward factory
sub(
    FF,
    """        uint32_t prev = 0;  // linear index of the core it receives from
        uint32_t next = 0;  // linear index of the core it forwards to
        std::vector<uint32_t> rows;
    };
    std::vector<ChainCore> chain_cores;
    uint32_t chain_rounds = 0;
    if (use_chain) {
        chain_cores.resize(num_cores);
""",
    """        uint32_t prev = 0;   // linear index of the core it receives from
        uint32_t next = 0;   // linear index of the core it forwards to
        uint32_t chain = 0;  // global chain index
        std::vector<uint32_t> rows;
        std::vector<uint32_t> stream_chunks;  // per round: the chunks the chain streams (iteration 27)
    };
    std::vector<ChainCore> chain_cores;
    uint32_t chain_rounds = 0;
    uint32_t num_chains = 0;
    if (use_chain) {
        chain_cores.resize(num_cores);
""",
)
sub(
    FF,
    """                auto& cc = chain_cores[first_core + c];
                cc.stream = stream;
                const uint32_t pos = c % kChainLength;
""",
    """                auto& cc = chain_cores[first_core + c];
                cc.stream = stream;
                cc.chain = num_chains + c / kChainLength;
                const uint32_t pos = c % kChainLength;
""",
)
sub(
    FF,
    """            first_core += stream_cores;
        }
        TT_FATAL(
            chain_rounds + 24U <= tt::tt_metal::max_runtime_args,
            "SDPA chain: {} rounds exceed the runtime-arg limit",
            chain_rounds);
        num_rows_per_core_group_1 = chain_rounds;
    }
""",
    """            first_core += stream_cores;
            num_chains += (stream_cores + kChainLength - 1U) / kChainLength;
        }
        // Per chain and round the stream stops at the diagonal chunk of the round's heaviest row (iteration 27):
        // every core of the chain consumes the same chunks, so the chain's rows decide.
        std::vector<std::vector<uint32_t>> chain_chunks(num_chains, std::vector<uint32_t>(chain_rounds, 0U));
        for (const auto& cc : chain_cores) {
            for (uint32_t r = 0; r < cc.rows.size(); ++r) {
                const uint32_t q = cc.rows[r] % St;
                chain_chunks[cc.chain][r] = std::max(chain_chunks[cc.chain][r], (q + 1U + Sk_chunk_t - 1U) / Sk_chunk_t);
            }
        }
        for (auto& cc : chain_cores) {
            cc.stream_chunks = chain_chunks[cc.chain];
        }
        TT_FATAL(
            2U * chain_rounds + 24U <= tt::tt_metal::max_runtime_args,
            "SDPA chain: {} rounds exceed the runtime-arg limit",
            chain_rounds);
        num_rows_per_core_group_1 = chain_rounds;
    }
""",
)
s = open(FF).read()
a = s.index("            const auto& cc = chain_cores[i];\n            std::vector<uint32_t> rounds(cc.rows);")
b = s.index("    } else if (use_work_list) {", a)
block = s[a:b]
nb = block.replace(
    "            std::vector<uint32_t> rounds(cc.rows);\n            rounds.resize(chain_rounds, kNoRow);\n",
    "            std::vector<uint32_t> rounds(cc.rows);\n            rounds.resize(chain_rounds, kNoRow);\n"
    "            // The rows (padded) then, per round, the chunks the chain streams.\n"
    "            const auto with_chunks = [&](std::vector<uint32_t> args) {\n"
    "                args.insert(args.end(), cc.stream_chunks.begin(), cc.stream_chunks.end());\n"
    "                return args;\n            };\n",
)
assert nb.count("work_list_runtime_args(") == 3, nb.count("work_list_runtime_args(")
nb = re.sub(
    r"work_list_runtime_args\(\s*(\{.*?\}),\s*rounds\)",
    lambda m: "with_chunks(work_list_runtime_args(" + m.group(1) + ", rounds))",
    nb,
    flags=re.S,
)
assert nb.count("with_chunks(work_list_runtime_args(") == 3
open(FF, "w").write(s[:a] + nb + s[b:])
# ============ Q and KV factories (shared chain_schedule template in each file)
for path in (QF, KF):
    sub(
        path,
        """    uint32_t prev = 0;
    uint32_t next = 0;
    std::vector<uint32_t> items;
};

struct ChainSchedule {
    std::vector<ChainCore> cores;
    uint32_t rounds = 0;
};
""",
        """    uint32_t prev = 0;
    uint32_t next = 0;
    uint32_t chain = 0;  // global chain index
    std::vector<uint32_t> items;
    std::vector<uint32_t> stream_bounds;  // per round: the chain's stream bound (iteration 27), see set_stream_bounds
};

struct ChainSchedule {
    std::vector<ChainCore> cores;
    uint32_t rounds = 0;
    uint32_t num_chains = 0;
};
""",
    )
    sub(
        path,
        """            auto& cc = sched.cores[first_core + c];
            cc.stream = stream;
""",
        """            auto& cc = sched.cores[first_core + c];
            cc.stream = stream;
            cc.chain = sched.num_chains + c / kChainLength;
""",
    )
    sub(
        path,
        """        first_core += stream_cores;
    }
    TT_FATAL(
        sched.rounds + 24U <= tt::tt_metal::max_runtime_args,
        "SDPA chain: {} rounds exceed the runtime-arg limit",
        sched.rounds);
""",
        """        first_core += stream_cores;
        sched.num_chains += (stream_cores + kChainLength - 1U) / kChainLength;
    }
    TT_FATAL(
        2U * sched.rounds + 24U <= tt::tt_metal::max_runtime_args,
        "SDPA chain: {} rounds exceed the runtime-arg limit",
        sched.rounds);
""",
    )
    # a helper after chain_schedule: per chain and round, combine the items' bounds
    sub(
        path,
        "// `items_of_stream(s)` -> (item ids, weights) of stream s.\ntemplate <typename ItemsFn>\n",
        """// Per chain and round the stream bound of the round's items (`bound_of_item`, combined with `combine`, `empty` for a
// round without items): every core of a chain consumes the same chunks, so the chain's items decide (iteration 27).
template <typename BoundFn, typename CombineFn>
void set_stream_bounds(ChainSchedule& sched, BoundFn&& bound_of_item, CombineFn&& combine, uint32_t empty) {
    std::vector<std::vector<uint32_t>> bounds(sched.num_chains, std::vector<uint32_t>(sched.rounds, empty));
    for (const auto& cc : sched.cores) {
        for (uint32_t r = 0; r < cc.items.size(); ++r) {
            bounds[cc.chain][r] = combine(bounds[cc.chain][r], bound_of_item(cc.items[r]));
        }
    }
    for (auto& cc : sched.cores) {
        cc.stream_bounds = bounds[cc.chain];
    }
}

// `items_of_stream(s)` -> (item ids, weights) of stream s.
template <typename ItemsFn>
""",
    )
    s = open(path).read()
    a = s.index("            const auto& cc = chain.cores[i];\n            std::vector<uint32_t> rounds(cc.items);")
    b = s.index("    } else if (use_work_list) {", a)
    block = s[a:b]
    nb = block.replace(
        "            std::vector<uint32_t> rounds(cc.items);\n            rounds.resize(chain.rounds, kNoRow);\n",
        "            std::vector<uint32_t> rounds(cc.items);\n            rounds.resize(chain.rounds, kNoRow);\n"
        "            // The items (padded) then, per round, the chain's stream bound.\n"
        "            const auto with_bounds = [&](std::vector<uint32_t> args) {\n"
        "                args.insert(args.end(), cc.stream_bounds.begin(), cc.stream_bounds.end());\n"
        "                return args;\n            };\n",
    )
    assert nb.count("work_list_runtime_args(") == 3, (path, nb.count("work_list_runtime_args("))
    nb = re.sub(
        r"work_list_runtime_args\(\s*(\{.*?\}),\s*rounds\)",
        lambda m: "with_bounds(work_list_runtime_args(" + m.group(1) + ", rounds))",
        nb,
        flags=re.S,
    )
    assert nb.count("with_bounds(work_list_runtime_args(") == 3
    open(path, "w").write(s[:a] + nb + s[b:])
sub(
    QF,
    """        chain = chain_schedule(num_cores, num_streams, items_of_stream);
        num_rows_per_core_group_1 = chain.rounds;
""",
    """        chain = chain_schedule(num_cores, num_streams, items_of_stream);
        // The stream stops at the diagonal chunk of the round's heaviest row.
        set_stream_bounds(
            chain,
            [&](uint32_t item) { return ((item % St) + 1U + chain_chunk - 1U) / chain_chunk; },
            [](uint32_t a, uint32_t b) { return std::max(a, b); },
            /* empty */ 0U);
        num_rows_per_core_group_1 = chain.rounds;
""",
)
sub(
    KF,
    """            return std::make_pair(items, weights);
        });
        num_rows_per_core_group_1 = chain.rounds;
""",
    """            return std::make_pair(items, weights);
        });
        // The stream starts at the chunk holding the round's lightest item's K row (its query rows from the
        // diagonal on); a round without items streams nothing (start = St).
        set_stream_bounds(
            chain,
            [&](uint32_t item) { return (((item / heads_per_group) % St) / chain_chunk) * chain_chunk; },
            [](uint32_t a, uint32_t b) { return std::min(a, b); },
            /* empty */ St);
        num_rows_per_core_group_1 = chain.rounds;
""",
)
# the per-round constants are no longer the loop bounds
sub(
    FWR,
    "    constexpr uint32_t num_chunks = Ht / Sk_chunk_t;\n",
    "    [[maybe_unused]] constexpr uint32_t num_chunks = Ht / Sk_chunk_t;\n",
)
sub(
    QR,
    "    constexpr uint32_t chunks_per_round = Ht / chain_chunk;\n",
    "    [[maybe_unused]] constexpr uint32_t chunks_per_round = Ht / chain_chunk;\n",
)
sub(
    KR,
    "    constexpr uint32_t chunks_per_round = Ht / chain_chunk;\n",
    "    [[maybe_unused]] constexpr uint32_t chunks_per_round = Ht / chain_chunk;\n",
)
print("iteration 27 patch applied")
