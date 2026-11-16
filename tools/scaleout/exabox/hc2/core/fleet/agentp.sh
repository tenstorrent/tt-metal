#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
# fleet/agentp.sh: sourced by gate.sh (this host's own idle gate).
# agentp PID COMM: true only for tt-fabric-manager-agent, never the controller. Reads ${PROC:-/proc}.
agentp() {  # $1 pid, $2 comm: tt-fabric-manager-agent. /proc comm is cut to 15 characters ("tt-fabric-manag"),
  # which tt-fabric-manager-controller shares, so the cut name is confirmed on argv[0]
  [ "$2" = tt-fabric-manager-agent ] && return 0; [ "$2" = tt-fabric-manag ] || return 1
  local a; a=$(tr '\0' '\n' < "${PROC:-/proc}/$1/cmdline" 2>/dev/null | head -1); [ "${a##*/}" = tt-fabric-manager-agent ]
}
