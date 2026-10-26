#!/bin/bash
# side.sh VARIANT A_DIS_PREFIX B_DIS_PREFIX SCEN: side-by-side walk listings (A | B)
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
V=$1; A=$2; B=$3; sc=$4; VB=${5:-$1}
cd $S/dis
f() { python3 ../rxscen.py $1 $2.dis $2.elf $sc -v | awk '{printf "%-5s %-7s %-22s %s\n",$1,$2,$3,$4}' | sed 's#fabric_erisc_router_v3_receiver_erisc.hpp#rx#;s#noc_nonblocking_api.h#nna#;s#fabric_edm_packet_transmission.hpp#tx#;s#fabric_erisc_router_v2.cpp#v2#;s#fabric_common.h#fc#;s#fabric_stream_regs.hpp#sr#;s#edm_fabric_flow_control_helpers.hpp#fch#;s#dataflow_api.h#dfa#;s#fabric_router_flow_control.hpp#rfc#' | cut -c1-58; }
paste -d'|' <(f $V $A) <(f $VB $B)
