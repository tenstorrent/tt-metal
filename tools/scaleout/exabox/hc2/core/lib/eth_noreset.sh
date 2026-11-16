#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# lib/eth_noreset.sh: the not-judged rewrite of an in-box 'ETH core ... not active again at mesh open' line when no start
# reset ran (dashboard rule 9). Sourced by lib/fleet.sh and lib/fault_capture.sh; defines one function.
fault_eth_noreset() {  # <stage> <fault_eth line> <why no start reset ran, empty when one ran> [<where>]: the line, judged or not
  local nm=$1 fe=$2 why=$3 at=${4:-at C2} hd
  if [ -z "$why" ] || [[ $fe != "ETH FAIL "*": ETH core "*" not active again at mesh open"* ]]; then echo "$nm $fe"; return 0; fi
  hd=${fe%%: ETH core *}
  echo "ETH INCOMPLETE ${hd#ETH FAIL }: $nm: start reset skipped; ETH not-active-at-open not judged: ${fe#"$hd": }; no glx_reset ran $at ($why)"
}
