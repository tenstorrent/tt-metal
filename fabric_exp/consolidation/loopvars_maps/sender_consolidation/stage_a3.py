"""stage_a3.py: v4txa3 = v4txa2 + Part A3 of ROUTER_HELPERS_REVIEW.md (the kernel file; user-approved 2026-11-03):
  - role setup into the roles, so kernel_main is the phase skeleton: sender::reset_connections (connection semaphores and
    producer cursors) and sender::set_up_channels (slot buffers and interfaces); receiver::set_up_channels (downstream
    links and progress, slot buffers; was router_init.hpp's init_downstream_connections), receiver::open_connections
    (wait_for_downstream_muxes and the opens) and receiver::preset_forward_commands; router_receiver.hpp includes
    router_init.hpp for RuntimeArgs;
  - the file header without the bring-up paragraph: the supported configurations are a list at the bring-up gate;
  - the 100000-poll constant named (POLLS_PER_ROUTING_RUN);
  - init_telemetry (the FabricTelemetry block) renamed init_telemetry_block, apart from telemetry::init (the counts)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txa2", "v4txa3", "a3.patch")
