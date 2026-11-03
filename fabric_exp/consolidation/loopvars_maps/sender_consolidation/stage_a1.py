"""stage_a1.py: v4txa1 = v4txc4 + Part A1 of ROUTER_HELPERS_REVIEW.md (router_common.hpp; user-approved 2026-11-03):
  - the hand-listed per-channel host-arg arrays leave router_common.hpp: the ct args file provides each as one array
    (sender_free_slots_streams by fabric position, sender_connection_info_addrs, sender_connection_semaphore_addrs,
    and local_downstream_credit_streams by VC and downstream entry); SenderChannel<CH> reads its own free-slots register;
  - the VC-number / position-arithmetic downstream helpers (downstream_sender_free_slots_stream,
    local_downstream_credit_stream) become DownstreamTraits<VC>'s local_credit_stream / forwarding_position /
    remote_credit_stream;
  - has_vc2 next to has_vc1; the heartbeat's interval and signature named (HEARTBEAT_INTERVAL, HEARTBEAT_SIGNATURE);
  - the router-independent helpers (compile-time loops, mask helpers, Present / Each, placement primitives) move to a
    new router_utils.hpp, so router_common.hpp holds the router's facts.
Expected: byte-identical to v4txc4."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txc4", "v4txa1", "a1.patch")
