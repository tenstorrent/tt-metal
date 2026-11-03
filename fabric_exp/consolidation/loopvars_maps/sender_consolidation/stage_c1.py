"""stage_c1.py: v4txc1 = v4txc0 + stage C1 of V4_SENDER_CONSOLIDATION.md (§2.1, §2.2): every sender fact in its owner.
  - SenderChannel<CH> holds its bubble flow-control threshold (remote_free_slots_to_send_by goes) and holds_packet;
  - SenderVc<VC> finds its channels by each channel's VC and its worker by role: a private worker_positions (was
    worker_channel_positions), has_worker_channel, worker_channel; channels_in and the dead first_level_ack go;
  - SenderChannels (the user's name) holds the router-wide masks (serviced, worker_channels, forwarding_channels,
    read_at_pass_start, serviced_vcs), the VC-count, mux and static-connection asserts (the last was K:163);
  - SenderVcState's worker member is typed by role (SenderVc<VC>::worker_channel), not by first_sender_channel_by_vc;
  - SENDER_TXQ goes (sender_txq_id directly); the dead forwarding_index goes;
  - ct: first_sender_channel_by_vc / sender_channel_count_by_vc go; the is_vc_crossover_to_vc1 comment is back above
    its array. Expected: byte-identical to v4txc0 on every image."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txc0", "v4txc1", "c1.patch")
