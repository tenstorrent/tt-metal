#!/bin/bash
# PRRTE plm_rsh_agent that injects `ulimit -n 65536;` before the remote
# command so spawned daemons / pytest workers don't inherit the system-wide
# 1024 fd limit (which is too low for tt-metal fabric/socket allocations).
#
# Wired in via `--prtemca plm_rsh_agent <this-script>` in the run_*_test.sh
# launchers. mpirun calls this as: <wrapper> [ssh-options] HOST <remote-cmd...>
# We identify HOST (first non-option arg, skipping flags that take a value)
# and prepend `ulimit -n 65536;` to the remote command.
all=("$@")
n=$#
host_idx=-1
i=0
while [ $i -lt $n ]; do
  a="${all[$i]}"
  case "$a" in
    -*)
      case "$a" in
        -l|-i|-p|-F|-o|-c|-e|-w|-S|-J|-W|-Q|-D|-R|-L|-B|-b|-m)
          i=$((i+2)); continue;;
      esac
      i=$((i+1));;
    *)
      host_idx=$i; break;;
  esac
done

if [ $host_idx -ge 0 ] && [ $host_idx -lt $n ]; then
  pre=("${all[@]:0:$host_idx}")
  host="${all[$host_idx]}"
  post=("${all[@]:$((host_idx+1))}")
  if [ ${#post[@]} -gt 0 ]; then
    remote="ulimit -n 65536; ${post[*]}"
    exec /usr/bin/ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=5 \
      "${pre[@]}" "$host" "$remote"
  fi
fi
exec /usr/bin/ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=5 "$@"
