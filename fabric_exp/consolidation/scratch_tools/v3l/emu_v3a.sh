#!/bin/bash
# emu_v3a.sh ELF HEADER [emu args]: run emu.py on a v3a-style ELF (main loop inline in kernel_main)
E=$1; H=$2; shift 2
python3 /tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3l/emu.py $E --func kernel_main --header $H \
  --skip wait_for_other_local_erisc,notify_subordinate_routers,wait_for_static_connection_to_ready,set_eth_txq_seq_update_timeouts \
  --force 0x15b60=170 "$@"
