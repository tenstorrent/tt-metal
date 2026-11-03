"""stage_x0_regroup.py: v4txo0's regroup after stage_x0_names.py: the receiver's file layout (header with the pass steps,
channels and VCs, state by placement class, sections), and one block, "Where the channel states live, and the loop",
holding the loop-local state containers, their starts and the loop (moves only)."""
import sys
P = "/home/aagarwal/tt-metal/fabric_exp/variants/v4txo0/tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp"
s = open(P).read()

def cut(block):
    global s
    assert s.count(block) == 1, block[:80]
    s = s.replace(block, "")
    return block

def put_before(anchor, text):
    global s
    assert s.count(anchor) == 1, anchor[:80]
    s = s.replace(anchor, text + anchor)

# 1. the header
h0 = s.index("// The sender program:")
h1 = s.index('#include "router_common.hpp"')
s = s[:h0] + """// The sender program: sends packets from this router's sender channels over the Ethernet link into the far router's
// receiver channel of the same VC, and returns each slot to its producer once the far router completes it.
//
// Each pass (pass, below):
//   reads     the free-slots registers of VC0's channels and VC2's worker channel, and the worker channels' connection
//             semaphores, as one read group (begin_read_group)
//   connect   each worker channel's connection, checked every pass (by rule); a change is handled out of line
//   send      VC0's worker channel: send if it holds a packet and the remote receiver channel has room; take completions
//             and credit the worker (out of line, once per completion batch)
//   forward   each VC's forwarding channels (VC0's; VC1's after a read group of its own), skipped with one test when
//             none holds a packet or has packets in flight; each one with work sends, takes completions and credits
//             its upstream router
//   VC2       its one channel, a worker channel, as VC0's (neighbour exchange; V4_VC2.md)
// The pass makes no call while it sends; the worker credit and the connection event are out of line (LEDGER E0-11).
// It takes no closures: a lambda capturing the loop's state can keep that state in memory (RESULTS §98), so the helpers
// recurse over their channel mask with their inputs as arguments.
//
// Channels and VCs. Each sender channel (SenderChannel<CH>) belongs to one VC. A VC's first channel takes a worker
// (VC0's channel 0, VC2's channel); the others are forwarding channels, written by the routers on this chip that face
// other directions. A VC's packets go into the far router's receiver channel of that VC (RemoteReceiverChannel<VC>).
//
// State, by placement class: the worker channels' progress and connection state, each VC's remote receiver progress
// and which forwarding channels have packets in flight, loop locals; each forwarding channel's progress and its
// producer's address, in memory between its visits; the worker interfaces, in memory (they escape to out-of-line
// calls); ring bounds, registers and counters, compile-time. The block before pass() states where each lives and why.
//
// Sections, in order: channels and their features; state; free-slots registers; sending; credits back to producers;
// visits; where the channel states live, and the loop.

""" + s[h1:]

# 2. section header for the channels
put_before("constexpr uint8_t SENDER_TXQ = sender_txq_id;", """// ---------------------------------------------------------------------------------------------------------------------
// Channels and their features
// ---------------------------------------------------------------------------------------------------------------------

""")
put_before("// The remote ring as this router fills it.\nstruct RemoteReceiverChannelProgress {", """// ---------------------------------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------------------------------

""")

# 3. the loop-local containers move into the last block
workers = cut("""// The worker channels: VC0's channel 0 and VC2's channel, each where present.
struct Workers {
    [[no_unique_address]] Present<has_worker, Worker, 0> vc0;
    [[no_unique_address]] Present<has_vc2_worker, Worker, 2> vc2;
};

""")
remotes = cut("""// The remote rings this router fills, one per VC it sends on: VC0's (its worker and forwarding channels send into it),
// VC1's, VC2's.
struct RemoteReceiverChannelProgresses {
    [[no_unique_address]] Present<sends_vc0, RemoteReceiverChannelProgress, 0> vc0;
    [[no_unique_address]] Present<sends_vc1, RemoteReceiverChannelProgress, 1> vc1;
    [[no_unique_address]] Present<sends_vc2, RemoteReceiverChannelProgress, 2> vc2;
};

""")
conns = cut("""// The worker channels' interfaces, the loop's own copies (connection events and the workers' credit addresses): passed
// to out-of-line calls, so in memory, apart from the state above. (Placement: see "State" above.)
template <typename Vc0Interface, typename Vc2Interface>
struct WorkerConnections {
    [[no_unique_address]] Vc0Interface vc0;
    [[no_unique_address]] Vc2Interface vc2;
};

""")
loop_hdr = cut("""// ---------------------------------------------------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------------------------------------------------

""")
block = """// ---------------------------------------------------------------------------------------------------------------------
// Where the channel states live, and the loop
// ---------------------------------------------------------------------------------------------------------------------

// Where the sender's state lives. The worker channels' progress and connection state (Workers) and each VC's remote
// receiver progress (RemoteReceiverChannelProgresses) are run()'s loop locals, kept in registers where they fit. Each
// forwarding channel's record stays in memory between its visits (ForwardingTable; keep_in_memory after each visit),
// so it does not take registers from the state every pass uses. The send path stays out of memory: its read groups
// would store and reload memory state around them (V4_STATE.md §11). The worker interfaces are the loop's own copies
// (WorkerConnections), in memory. A one-ERISC router keeps every state in its state record (router_single.hpp).
""" + "\n" + workers + remotes + conns.replace(
    """// to out-of-line calls, so in memory, apart from the state above. (Placement: see "State" above.)""",
    """// to out-of-line calls, so in memory.""")
put_before("// The producers' credit addresses of the forwarding channels, cached when each upstream router connected.\n",
           block)
open(P, "w").write(s)
print("regrouped", P)
