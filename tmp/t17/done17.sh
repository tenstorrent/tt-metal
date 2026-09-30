#!/bin/bash
# retry_when for t17: exit 0 once drive17 submitted the A/B and it is no longer running/queued on blx03.
r=$(timeout 50 ssh -o BatchMode=yes g14blx03 '[ -f ~/fasth3/drive17.log ] || { echo wait; exit; }
  tt-device-mcp status 1 2>&1 | sed -n "/^RUNNING/,/^RECENT/p" | grep -q "fasth3/t17 " && echo wait || echo done' 2>/dev/null)
[ "$r" = done ] && exit 0 || exit 1
