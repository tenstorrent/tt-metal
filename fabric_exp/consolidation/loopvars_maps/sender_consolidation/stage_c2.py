"""stage_c2.py: v4txc2 = v4txc1b + stage C2 of V4_SENDER_CONSOLIDATION.md (§2.3): one command-buffer owner for both
ERISCs, CommandBuffers in router_common.hpp: the host's buffer of each kind of command, which kinds keep presets (one
condition, with the DPRINT/watcher guard now on the worker credit too: user's answer 4), the table of kinds per buffer
on this ERISC and one rule as static_asserts (conflicts stop the build: user's answer 2). It replaces
worker_credit_cmd_bufs_distinct, forward_cmd_bufs_consistent, local_write_cmd_buf_is_exclusive, command_buffer_presets,
local_write_state_preset, forwards_can_preset, worker_credit_preset, uses_vc0_forward_cmd_bufs and LOCAL_WRITE_CMD_BUF;
is_sender_role / is_receiver_role (router_init.hpp) and vc_forwards_to_routers (router_downstream.hpp) move to
router_common.hpp. Expected: byte-identical to v4txc1b on every coverage image (no DPRINT/watcher case)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txc1b", "v4txc2", "c2.patch")
