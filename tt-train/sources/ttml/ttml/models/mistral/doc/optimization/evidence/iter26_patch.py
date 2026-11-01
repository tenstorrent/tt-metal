"""Iteration 26: credit-based chain hops with CHAIN_SLOTS chunks in flight (SDPA forward, Q backward, KV backward)."""
import re
import sys

M = "tt-train/sources/ttml/metal/ops"
FW = f"{M}/sdpa_fw/device/kernels/dataflow/sdpa_fw"
BW = f"{M}/sdpa_bw/device/kernels/dataflow/sdpa_bw"
done = []


def sub(path, old, new, count=1, regex=False):
    s = open(path).read()
    if regex:
        s2, n = re.subn(old, new, s, flags=re.S)
    else:
        n = s.count(old)
        s2 = s.replace(old, new)
    if n != count:
        print(f"FAILED {path}: {old[:70]!r} found {n}, expected {count}")
        sys.exit(2)
    open(path, "w").write(s2)
    done.append((path, old[:40]))


def credit_block(cbs, total_expr):
    checks = " &&\n                       ".join(
        f"cb_pages_reservable_at_back({cb}, (credited - c + 1U) * {tiles})" for cb, tiles in cbs
    )
    return f"""                // Credits (iteration 26): the previous core's writer forwards chunk n once this core has credited
                // it. Chunk c is credited here (its slots are reserved) and the chunks up to c + kChainSlots - 1
                // whose slots are already free (the compute popped and the writer forwarded chunk n - kChainSlots)
                // are credited ahead, so the hop's ready round trip leaves the critical path.
                if (credited <= c) {{
                    sender_sem.up(noc, prev_noc_x, prev_noc_y, 1);
                    credited = c + 1U;
                }}
                while (credited < {total_expr} && credited < c + kChainSlots &&
                       (is_sink || credited < kChainSlots || forwarded_sem.value() + kChainSlots >= credited + 1U) &&
                       {checks}) {{
                    sender_sem.up(noc, prev_noc_x, prev_noc_y, 1);
                    ++credited;
                }}
                receiver_sem.wait_min(c + 1U);
"""


# ---- readers
for path, cbs, total in [
    (f"{FW}_reader_kernel.cpp", [("cb_key", "k_chunk_tiles"), ("cb_value", "v_chunk_tiles")], "total_chunks"),
    (f"{BW}_q_reader_kernel.cpp", [("cb_key", "k_chunk_tiles"), ("cb_value", "v_chunk_tiles")], "total_chunks"),
    (
        f"{BW}_kv_reader_kernel.cpp",
        [
            ("cb_query", "q_chunk_tiles"),
            ("cb_intermediates", "one_chunk_tiles"),
            ("cb_grad_output", "vo_chunk_tiles"),
            ("cb_u_scalar_row", "one_chunk_tiles"),
        ],
        "total_chunks",
    ),
]:
    sub(
        path,
        """            if (!is_sink && c >= 2U) {
                forwarded_sem.wait_min(c - 1U);""",
        """            if (!is_sink && c >= kChainSlots) {
                forwarded_sem.wait_min(c + 1U - kChainSlots);""",
    )
    sub(
        path,
        """                receiver_sem.set(INVALID);
                sender_sem.up(noc, prev_noc_x, prev_noc_y, 1);
                receiver_sem.wait(VALID);
""",
        credit_block(cbs, total),
    )
chunks = {
    f"{FW}_reader_kernel.cpp": "num_chunks",
    f"{BW}_q_reader_kernel.cpp": "chunks_per_round",
    f"{BW}_kv_reader_kernel.cpp": "chunks_per_round",
}
for path, per_round in chunks.items():
    # declare kChainSlots, credited and total_chunks right before the round loop
    sub(
        path,
        "    for (uint32_t round = 0; round < num_rounds; ++round) {\n",
        f"""    constexpr uint32_t kChainSlots = CHAIN_SLOTS;  // chunks resident per stream CB (slot = chunk % kChainSlots)
    const uint32_t total_chunks = num_rounds * {per_round};
    uint32_t credited = 0;  // chunks credited to the previous core (non-injectors)
    for (uint32_t round = 0; round < num_rounds; ++round) {{
""",
    )
# the fw reader's comment line about forwarded >= c - 1
# ---- writers
for path in [f"{FW}_writer_kernel.cpp", f"{BW}_q_writer_kernel.cpp", f"{BW}_kv_writer_kernel.cpp"]:
    sub(path, "    valid_sem.set(VALID);\n", "    constexpr uint32_t kChainSlots = CHAIN_SLOTS;\n")
    sub(
        path,
        "                sender_sem.wait(1);\n                sender_sem.set(0);\n",
        "                sender_sem.wait_min(c + 1U);  // the next core's credits (iteration 26)\n",
    )
    sub(
        path,
        "                valid_sem.relay_unicast(noc, receiver_sem, next_noc_x, next_noc_y);\n",
        "                receiver_sem.up(noc, next_noc_x, next_noc_y, 1);  // valid count\n",
    )
    sub(path, "    noc.async_write_barrier();\n", "    noc.async_write_barrier();\n    noc.async_atomic_barrier();\n")
    sub(path, "(c % 2U)", "(c % kChainSlots)", count=None if False else open(path).read().count("(c % 2U)"))
    sub(path, "    Semaphore valid_sem(", "    [[maybe_unused]] Semaphore valid_sem(")
    sub(
        path,
        "Two chunks resident per CB; the reader's pushes are whole chunks, so chunk c sits in slot c % 2.",
        "kChainSlots chunks resident per CB; the reader's pushes are whole chunks, so chunk c sits in slot c % kChainSlots.",
    )
sub(f"{BW}_kv_writer_kernel.cpp", "const uint32_t slot = c % 2U;", "const uint32_t slot = c % kChainSlots;")
# ---- factories
F = f"{M}/sdpa_fw/device/sdpa_fw_program_factory.cpp"
Q = f"{M}/sdpa_bw/device/sdpa_bw_q_program_factory.cpp"
KV = f"{M}/sdpa_bw/device/sdpa_bw_kv_program_factory.cpp"
for path in (F, Q, KV):
    sub(
        path,
        "constexpr uint32_t kChainLength = 14U;",
        "constexpr uint32_t kChainSlots = 3U;      // stream chunks resident per core (credits run kChainSlots - 1 ahead)\nconstexpr uint32_t kChainLength = 14U;",
    )
sub(
    F,
    '        defines[kChainDefKey] = "1";\n',
    '        defines[kChainDefKey] = "1";\n        defines["CHAIN_SLOTS"] = std::to_string(kChainSlots);\n',
)
sub(F, "2 * Sk_chunk_t * kWt);", "(use_chain ? kChainSlots : 2U) * Sk_chunk_t * kWt);")
sub(F, "2 * Sk_chunk_t * vWt);", "(use_chain ? kChainSlots : 2U) * Sk_chunk_t * vWt);")
for path in (Q, KV):
    sub(
        path,
        '        reader_defines[kChainDefKey] = "1";\n        writer_defines[kChainDefKey] = "1";\n',
        '        reader_defines[kChainDefKey] = "1";\n        writer_defines[kChainDefKey] = "1";\n'
        '        reader_defines["CHAIN_SLOTS"] = std::to_string(kChainSlots);\n        writer_defines["CHAIN_SLOTS"] = std::to_string(kChainSlots);\n',
    )
sub(
    Q,
    "const uint32_t kv_cb_rows = 2U * kv_block;",
    "const uint32_t kv_cb_rows = (use_chain ? kChainSlots : 2U) * kv_block;",
)
sub(
    KV,
    "const uint32_t stream_cb_rows = use_chain ? 2U * chain_chunk : 2U;",
    "const uint32_t stream_cb_rows = use_chain ? kChainSlots * chain_chunk : 2U;",
)
print("patched", len(done), "sites")
