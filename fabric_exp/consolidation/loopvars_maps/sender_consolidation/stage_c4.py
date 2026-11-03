"""stage_c4.py: v4txc4 = v4txc3 + stage C4 of V4_SENDER_CONSOLIDATION.md: comments and names only.
  - router_sender.hpp's header: the per-pass list by features (worker channel, forwarding channels, where each VC's
    registers are read), not VC numbers; the channel kinds named;
  - fabric_erisc_router_v4.cpp's header: "a VC's worker channel" for "channel 0 of a VC";
  - router_round.hpp: end_round's parameter `send_attempted` (was the field's name, send_yielded) and its description
    above it (it sat above RoundCounters; `idle_interval` named SWITCH_INTERVAL);
  - router_common.hpp: the sender channels' connection blocks under their own section header (they sat under
    "Downstream routers");
  - SenderChannels' static-connection assert reads the host's trimmed array sender_ch_wait_static_connection;
  - comment lines this pass made longer than 120 columns rewrapped, and three older ones in router_sender.hpp.
Expected: byte-identical to v4txc3."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txc3", "v4txc4", "c4.patch")
