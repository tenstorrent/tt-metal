#!/bin/bash
# Path resolution shared by every runner and campaign in this directory (PORTABLE_CONTRACT.md).
# Two layouts are supported, with no edit to any script:
#   1) reference workspace: the scripts sit in $WORK/handoff/revamp/data/bh_zones and DD is that directory;
#   2) tt-metal checkout: the scripts sit in $TTM/analysis/campaigns, TTM is the checkout found by walking
#      up for a directory that holds both ttnn and tt_metal, and DD is $PWD so that nothing is ever
#      written into the repository tree.
# The caller sets SD (the directory the sourcing script sits in) before sourcing this file; sourcing this
# file on its own also works, SD then defaults to where this file lives.
# Every value is an environment override: DD, SDPA_WORK, TTM, TTM_FRESH, PYENV, HANDOFF.

SD=${SD:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}

_ttm_root() {   # nearest directory at or above $1 that holds both ttnn and tt_metal, i.e. a tt-metal checkout
  local d=$1
  while [ -n "$d" ] && [ "$d" != "/" ]; do
    if [ -d "$d/ttnn" ] && [ -d "$d/tt_metal" ]; then echo "$d"; return 0; fi
    d=$(dirname "$d")
  done
}

_SELF_TTM=$(_ttm_root "$SD")
if [ -n "$_SELF_TTM" ]; then
  TTM=${TTM:-$_SELF_TTM}
  WORK=${SDPA_WORK:-$(dirname "$TTM")}
  DD=${DD:-$PWD}
  HANDOFF=${HANDOFF:-$DD}
else
  DD=${DD:-$SD}
  WORK=${SDPA_WORK:-$(cd "$SD/../../../.." && pwd)}
  TTM=${TTM:-$WORK/tt-metal}
  HANDOFF=${HANDOFF:-$(cd "$SD/../.." && pwd)}
fi
TTM_FRESH=${TTM_FRESH:-$WORK/tt-metal-fresh}
PYENV=${PYENV:-$TTM/python_env/bin/activate}
export DD SD WORK HANDOFF TTM TTM_FRESH PYENV
case "$DD" in                 # writing a campaign into a git checkout is almost never wanted
  "$TTM"/*) echo "note: DD=$DD is inside the checkout $TTM; export DD to a directory of your own" >&2 ;;
esac
