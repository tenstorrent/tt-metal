"""stage_c3.py: v4txc3 = v4txc2 + stage C3 of V4_SENDER_CONSOLIDATION.md (§2.4, §2.5): the producers.
  - SenderChannelInterfaces: the library interfaces' storage, start() (K's init_worker_interfaces and
    start_worker_interfaces in one, called only where the ERISC serves sender channels: v4txo8w, user's answer 7),
    of<CH>() and read_counter_addr(ch) (was worker_read_counter_addr);
  - WorkerChannel<CH>: connection semaphore, read counter, credit buffer, connected(), check_connection, credit,
    preset_credit and the private connection_event (were check_connection<CH>, credit_worker<CH>, preset_worker_credit<CH>,
    connection_event<CH>; bodies verbatim); open_or_close_worker_connection (was worker_connection_event) beside it;
  - ForwardingChannel<CH>: waits_for_upstream_router, upstream_free_slots_addr(), credit (was credit_upstream);
  - SenderChannel<CH>::producer_noc for WORKER_CREDIT_NOC; start_forwarding_channels(states) for init_forwarding<MASK>;
  - wait_for_connect_request / wait_for_upstream_routers move from the kernel file into router_sender.hpp's init
    section (user's answer 6); the file's sections follow §2.4.
Expected: sender::run, single::sender_pass, receiver::run, single::receiver_pass identical per function (renamed calls
aside); init, sender-less images (no interface storage) and address constants change."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txc2", "v4txc3", "c3.patch")
