"""api_patch.py VARIANT: the smallest additive NoC API for partially preset forwards, in VARIANT's pinned copies only
(never the live tree; flagged for upstream review in V4_NS_PRESETS.md):
  - noc_write_to_node_with_state (blackhole and wormhole noc_nonblocking_api.h, next to noc_wwrite_with_state): the
    stateful write with the destination node given as a coordinate, not extracted from a 64-bit address;
  - noc_fast_write_dw_inline_to_node_with_state (both noc_nonblocking_api.h, next to the stateful inline helpers) and
    its dataflow wrapper noc_inline_dw_write_to_node_with_state (dataflow_api.h): the stateful inline write that
    updates both the local address and the node."""
import sys
V = f"/home/aagarwal/tt-metal/fabric_exp/variants/{sys.argv[1]}/tt_metal/hw/inc"
FN = '''
// clang-format off
/**
 * Additive (fabric router study; for upstream review). Like noc_fast_write_dw_inline_with_state, for a destination
 * whose node and local address both change per write:
 * writes the local address (NOC_TARG_ADDR_LO) and the node (NOC_TARG_ADDR_COORDINATE, encoded as
 * noc_fast_write_dw_inline_set_state encodes it: (noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK), then sends.
 * The value, byte enables, control and NOC_TARG_ADDR_MID stay as noc_fast_write_dw_inline_set_state left them, so the
 * destination must be a register or a word-aligned address (the byte enables are not rewritten).
 */
// clang-format on
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    bool posted = false,
    bool update_counter = true,
    bool skip_cmdbuf_chk = false>  // the caller has already seen the command buffer ready
inline __attribute__((always_inline)) void noc_fast_write_dw_inline_to_node_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t node_coordinate, uint32_t addr_lo) {
    if constexpr (update_counter && noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
        }
    }
    if constexpr (!skip_cmdbuf_chk) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, addr_lo);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, node_coordinate);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (update_counter && noc_mode == DM_DEDICATED_NOC) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;
        }
    }
}
'''

DATA_FN = {
    "blackhole": '''
// clang-format off
/**
 * Additive (fabric router study; for upstream review). Like noc_write_with_state, for a destination given as a node
 * coordinate and a local address: the coordinate is written as given (as NOC_RET_ADDR_COORDINATE takes it,
 * (noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK), not extracted from a 64-bit address. On-chip addresses
 * only: NOC_RET_ADDR_MID stays as the set_state call left it.
 */
// clang-format on
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    uint32_t cmd_buf,
    enum CQNocFlags flags,
    enum CQNocSend send = CQ_NOC_SEND,
    enum CQNocWait wait = CQ_NOC_WAIT,
    bool update_counter = true,
    bool posted = false>
inline __attribute__((always_inline)) void noc_write_to_node_with_state(
    uint32_t noc, uint32_t src_addr, uint32_t node_coordinate, uint32_t dst_addr, uint32_t size = 0, uint32_t ndests = 1) {
    noc_common_write_with_state<noc_mode, cmd_buf, flags, wait, update_counter, posted>(noc, src_addr, size, ndests);
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dst_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, node_coordinate);
    }
    if constexpr (send) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    }
}
''',
    "wormhole": '''
// clang-format off
/**
 * Additive (fabric router study; for upstream review). Like noc_write_with_state, for a destination given as a node
 * coordinate and a local address: the coordinate is written as given (as NOC_RET_ADDR_COORDINATE takes it,
 * (noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK), not extracted from a 64-bit address.
 */
// clang-format on
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    uint32_t cmd_buf,
    enum CQNocFlags flags,
    enum CQNocSend send = CQ_NOC_SEND,
    enum CQNocWait wait = CQ_NOC_WAIT,
    bool update_counter = true,
    bool posted = false>
inline __attribute__((always_inline)) void noc_write_to_node_with_state(
    uint32_t noc, uint32_t src_addr, uint32_t node_coordinate, uint32_t dst_addr, uint32_t size = 0, uint32_t ndests = 1) {
    noc_wwrite_with_state<noc_mode, cmd_buf, flags, send, wait, update_counter, posted>(
        noc, src_addr, node_coordinate, dst_addr, size, ndests);
}
''',
}
# Blackhole: right after noc_wwrite_with_state; Wormhole: right after noc_write_with_state (its wrapper of it).
DATA_ANCHOR = {
    "blackhole": '''        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, dst_noc_addr);
    }
    if constexpr (send) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    }
}
''',
    "wormhole": '''    noc_wwrite_with_state<noc_mode, cmd_buf, flags, send, wait, update_counter, posted>(
        noc, src_addr, dst_noc_addr, dst_addr, size, ndests);
}
''',
}
for arch in ("blackhole", "wormhole"):
    p = f"{V}/internal/tt-1xx/{arch}/noc_nonblocking_api.h"
    s = open(p).read()
    anchor = "\n// clang-format off\n/**\n * The stateful NOC commands provide granular control over NOC register programming"
    assert s.count(anchor) == 1, (arch, "anchor")
    s = s.replace(anchor, FN + anchor)
    da = DATA_ANCHOR[arch]
    assert s.count(da) == 1, (arch, "data anchor")
    s = s.replace(da, da + DATA_FN[arch])
    open(p, "w").write(s)
p = f"{V}/api/dataflow/dataflow_api.h"
s = open(p).read()
anchor = """// clang-format off
/**
 * The Tensix core executing this function call initiates an atomic increment
 * (with 32-bit wrap) of a remote Tensix core L1 memory address."""
assert s.count(anchor) == 1, "dataflow anchor"
WRAP = '''// clang-format off
/**
 * Additive (fabric router study; for upstream review). Like noc_inline_dw_write_with_state, for a destination whose
 * node and local address both change per write: writes the local address and the node, then sends; the value, byte
 * enables and control stay as noc_inline_dw_write_set_state left them. Only for registers and word-aligned addresses
 * (the byte enables are not rewritten).
 */
// clang-format on
template <bool update_counter = true, bool posted = false, bool skip_cmdbuf_chk = false>
FORCE_INLINE void noc_inline_dw_write_to_node_with_state(
    uint32_t node_coordinate, uint32_t addr, uint8_t cmd_buf = write_at_cmd_buf, uint8_t noc = noc_index) {
#ifdef ARCH_BLACKHOLE
    // As noc_inline_dw_write_with_state: always update the counter on Blackhole (issue 28758).
    constexpr bool update_counter_in_callee = true;
#else
    constexpr bool update_counter_in_callee = update_counter;
#endif
    WAYPOINT("NWNW");
    noc_fast_write_dw_inline_to_node_with_state<noc_mode, posted, update_counter_in_callee, skip_cmdbuf_chk>(
        noc, cmd_buf, node_coordinate, addr);
    WAYPOINT("NWND");
}

'''
s = s.replace(anchor, WRAP + anchor)
open(p, "w").write(s)
print("api added to", sys.argv[1])
