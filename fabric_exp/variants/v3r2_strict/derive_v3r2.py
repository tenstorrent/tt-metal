#!/usr/bin/env python3
"""derive_v3r2.py JOINED_V3R OUT CHANGE[,CHANGE...]|all: apply v3r2's codegen-changing cleanups to a joined v3r copy.

v3r2 = split(derive_v3r2(join(v3r), all)). To build one change alone (to bisect a hardware difference):
    python3 fabric_exp/consolidation/v3r_renames.py join fabric_exp/variants/v3r /tmp/j
    python3 fabric_exp/variants/v3r2/derive_v3r2.py /tmp/j /tmp/k bittest
    python3 fabric_exp/consolidation/v3r_renames.py split /tmp/k /tmp/v3r_bittest
Changes: templates, os, heartbeat, param, bittest, init (see V3_REVIEW.md, "v3r2"). Not a router source file: the
offline build and swap.py ignore it.
"""
import os, re, shutil, sys

R = "fabric_erisc_router_v3.cpp"


def sub1(s, a, b, n=1):
    c = s.count(a)
    assert c == n, (c, n, a[:80])
    return s.replace(a, b)


def c_hint(s):
    return sub1(s, "            if (sparse_write_and_forward) [[unlikely]] {", "            if (sparse_write_and_forward) {")


def c_templates(s):
    n = s.count("template <bool INSTANTIATED_ON_USE = true>\n__attribute__((noinline))")
    assert n == 7, n
    return s.replace("template <bool INSTANTIATED_ON_USE = true>\n__attribute__((noinline))", "__attribute__((noinline))")


def c_os(s):
    n = s.count('__attribute__((optimize("Os"))) ')
    assert n == 7, n
    return s.replace('__attribute__((optimize("Os"))) ', "")


def c_heartbeat(s):
    s = sub1(
        s,
        """#if defined(ARCH_BLACKHOLE)
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x7CC70;
#else
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x1F80;
#endif
""",
        """#if defined(ARCH_BLACKHOLE)
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x7CC70;
#else
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x1F80;
#endif
FORCE_INLINE void update_fabric_heartbeat(uint16_t& counter) {
    if ((++counter & 0x3F) == 0) {
        *reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR) = 0xDCBA0000 | counter;
    }
}
""",
    )
    s = sub1(
        s,
        """    uint16_t fabric_heartbeat_counter = 0;
    volatile uint32_t* fabric_heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR);
""",
        "    uint16_t fabric_heartbeat_counter = 0;\n",
        3,
    )
    n = 0
    for ind in ("            ", "        "):
        a = f"""{ind}if ((++fabric_heartbeat_counter & 0x3F) == 0) {{
{ind}    *fabric_heartbeat_ptr = 0xDCBA0000 | fabric_heartbeat_counter;
{ind}}}
"""
        n += s.count(a)
        s = s.replace(a, f"{ind}update_fabric_heartbeat(fabric_heartbeat_counter);\n")
    assert n == 3
    return s


def c_param(s):
    s = sub1(
        s,
        """FORCE_INLINE void establish_edm_connection(
    EdmChannelWorkerIFs& local_sender_channel_worker_interface, [[maybe_unused]] uint32_t stream_id) {""",
        """FORCE_INLINE void establish_edm_connection(EdmChannelWorkerIFs& local_sender_channel_worker_interface) {""",
    )
    return sub1(
        s,
        """        establish_edm_connection(
            interface,
            local_sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_idx)]);""",
        "        establish_edm_connection(interface);",
    )


def c_bittest(s):
    s = sub1(
        s,
        "    if (static_cast<int32_t>(pre_txq_status << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)) < 0) {",
        "    if ((pre_txq_status >> ETH_TXQ_STATUS_CMD_ONGOING_BIT) & 1u) {",
    )
    return sub1(
        s,
        "    if (static_cast<int32_t>(txq_status << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)) < 0) {",
        "    if ((txq_status >> ETH_TXQ_STATUS_CMD_ONGOING_BIT) & 1u) {",
    )


HS_BEGIN = "    if constexpr (enable_ethernet_handshake) {\n"


def handshake_block(s, start):
    """The `if constexpr (enable_ethernet_handshake) { ... }` block starting at index `start` (to its closing brace)."""
    i = start + len(HS_BEGIN)
    depth = 1
    while depth:
        c = s[i]
        depth += c == "{"
        depth -= c == "}"
        i += 1
    assert s[i] == "\n"
    return s[start : i + 1]


def c_init(s):
    # Pointer variables themselves volatile (`volatile auto p = reinterpret_cast<volatile T*>(...)`): plain pointers.
    s = sub1(s, "    volatile auto termination_signal_ptr =\n", "    auto termination_signal_ptr =\n", 2)
    s = sub1(s, "    [[maybe_unused]] volatile auto edm_local_sync_ptr =\n", "    [[maybe_unused]] auto edm_local_sync_ptr =\n")
    s = sub1(s, "    volatile auto edm_local_sync_ptr = reinterpret_cast", "    auto edm_local_sync_ptr = reinterpret_cast")
    s = sub1(s, "    volatile auto edm_status_ptr = reinterpret_cast", "    auto edm_status_ptr = reinterpret_cast", 2)
    # One handshake helper for both init paths.
    starts = [m.start() for m in re.finditer(re.escape(HS_BEGIN), s)]
    assert len(starts) == 2, len(starts)
    blocks = [handshake_block(s, x) for x in starts]
    keep = blocks[1]  # kernel_main's (it carries main's comment on the self-notification)
    body = keep[len(HS_BEGIN) :]
    helper = (
        """// The Ethernet handshake with the link peer and, when the host coordinates start-up, the local handshake with this
// chip's other routers and the wait for READY_FOR_TRAFFIC. Both init paths run it between the same two ERISC syncs.
FORCE_INLINE void handshake_with_link_peer_and_local_routers(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    volatile tt_l1_ptr tt::tt_fabric::EDMStatus* edm_status_ptr,
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* edm_local_sync_ptr,
    const tt_l1_ptr tt::tt_fabric::routing_l1_info_t* routing_table_l1) {
"""
        + HS_BEGIN
        + body
        + "}\n\n"
    )
    call = (
        "    handshake_with_link_peer_and_local_routers(\n"
        "        termination_signal_ptr, edm_status_ptr, edm_local_sync_ptr, routing_table_l1);\n"
    )
    s = s.replace(blocks[1], call, 1)
    s = s.replace(blocks[0], call, 1)
    anchor = "#if defined(ARCH_BLACKHOLE)\nnamespace receiver_erisc {\n"
    s = sub1(s, anchor, helper + anchor)
    return s


CHANGES = {
    "templates": c_templates,
    "os": c_os,
    "heartbeat": c_heartbeat,
    "param": c_param,
    "bittest": c_bittest,
    "init": c_init,
}

if len(sys.argv) != 4:
    sys.exit(__doc__)
src, out, which = sys.argv[1], sys.argv[2], sys.argv[3]
names = list(CHANGES) if which == "all" else which.split(",")
if os.path.exists(out):
    shutil.rmtree(out)
shutil.copytree(src, out)
p = os.path.join(out, R)
s = open(p).read()
for n in names:
    s = CHANGES[n](s)
open(p, "w").write(s)
print("applied", names)
