"""stage_a2.py: v4txa2 = v4txa1 + Part A2 of ROUTER_HELPERS_REVIEW.md (router_init.hpp; user-approved 2026-11-03, the
stream-init design agreed with the main session):
  - stream registers by owner: receiver::init_channel_streams (each receiver channel the router has: its packets-
    received register) and sender::init_channel_streams (each sender channel the router has, at its fabric position:
    its free-slots, first-level-ack and completion registers), both before the ERISCs' first sync; the context-switch
    register's start state is router_round.hpp's init_context_switch_state. init_link_streams / init_2d_streams go.
    A behavior change: the 2D group moves before the first sync, and registers no channel of this router owns are no
    longer written (a2/stream_init_scan.py: on 3,104 of 3,124 configs the set written is the same; on 20 bh_neighbor
    configs position 1's free-slots register, owned by no channel there and written by nothing else, is left alone);
  - one helper (zero_credit_counters) per L1 credit-counter block in init_txqs;
  - one owner each, in router_common.hpp, for the termination signal, the routing table and the telemetry block
    (termination_signal, l1_routing_table, telemetry_block); router_init.hpp, router_round.hpp use them;
  - TeardownTrids: checked against the receiver's tracker (the receiver tags with CompletionGroupTrids over the same
    IDs; the library tracker is used only to wait for all of them), and said so."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txa1", "v4txa2", "a2.patch")
