"""stage_a10.py: v4txa10 = v4txa9 + stage A10, the kernel file's cleanup (user go 2026-11-03; design agreed with the
main session):
  - fabric_erisc_router_v4.cpp includes the kernel API and the router headers only; each router header includes the
    library headers it uses; the file-scope `using namespace tt::tt_fabric;` is a using-directive inside namespace
    fabric_router (router_common.hpp), said why; `<limits>` and the 2D API include moved to their users; five library
    headers nothing used are dropped (packet_header_validate, tmp_utils, packet_recorder, channel_traits,
    flow_control_helpers; the router adapter is replaced by the adapter utils it needs);
  - the bring-up gate is router_common.hpp's "Supported configurations": the list as its comment, one static_assert
    per unsupported feature with its own message, udm_ / vc2_ / speedy_config_supported with it;
  - finish_phase<Phase>(): sync_erisc plus the phase's postcodes from one table of main's values in main's order
    (two marked as main's codes kept for tooling); *edm_status() = STARTED and the WAYPOINTs kept;
  - init_router() (router_init.hpp): the prologue, with the code-profiling clear; set_l1_data_cache<false>() kept
    (it turns the data cache off, not a no-op: the design's drop was not applied, reported);
  - the teardown master's second INITIALIZING kept, with a one-line why;
  - receiver::set_up_channels returns the downstream connections (none on a sender-only ERISC); the role loop is a
    small template (run_loop) so the receiver's calls are named only where they exist;
  - FABRIC_ERISC_ROUTER_V4 says what it selects in the shared ct-args header.
Expected: every loop function byte-identical to v4txa9; kernel_main and _start change."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txa9", "v4txa10", "a10.patch")
