#!/bin/bash
# glxq_node.sh STEP: one node's part of a multi-Galaxy job's preflight or cleanup (glxq.sbatch runs it on every node
# with srun, one task per node). Prints one line per finding; prints nothing when a check finds nothing.
#   busy     other users' processes that may hold the devices
#   tools    missing tools (mpirun-ulfm, tt-smi, the tree's test binary and python_env, the MPI NIC, /data writable)
#   reset    tt-smi -glx_reset, then the Blackhole chips tt-smi lists: "rc=<exit> chips=<n>"
#   eth      read_eth_speed.py on 32 chips into $J/results/eth_<host>.txt: "up=<channels UP and trained>"
#   kill     this user's test processes on the node (the job holds the nodes --exclusive)
#   capture LABEL   hang_capture.py (read-only) into $J/results/LABEL_hangcap_<host>
# Environment: J, TREE, STAGE, MPI_IF, TSMI.
set -uo pipefail
h=${GLXQ_HOST:-$(hostname -s)}  # GLXQ_HOST: glxq.sbatch stub mode only
case ${1:?STEP} in
    busy)
        ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn|mpirun|prted/' |
            grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-160 | head -3 ;;
    tools)
        command -v mpirun-ulfm > /dev/null || echo "$h: no mpirun-ulfm"
        command -v "${TSMI:-tt-smi}" > /dev/null || echo "$h: no tt-smi"
        [ -x "$TREE/build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric" ] || echo "$h: tree not visible"
        [ -x "$TREE/python_env/bin/python" ] || echo "$h: no python_env"
        ip -o link show "$MPI_IF" > /dev/null 2>&1 || echo "$h: no interface $MPI_IF"
        touch "$J/results/.w_$h" 2> /dev/null || echo "$h: $J not writable"
        [ -d "${HOME:-/nonexistent}" ] && [ -w "$HOME" ] || echo "$h: HOME $HOME missing or not writable (Open MPI keeps session files there)" ;;
    reset)
        timeout 900 "${TSMI:-tt-smi}" -glx_reset > "$J/results/reset_${h}_$(date -u +%H%M%S).log" 2>&1
        rc=$?
        echo "rc=$rc chips=$("${TSMI:-tt-smi}" -ls 2> /dev/null | grep -c Blackhole)" ;;
    eth)
        (cd "$TREE" && timeout 300 python_env/bin/python -I "${READ_ETH:-$STAGE/read_eth_speed.py}" 32 > "$J/results/eth_$h.txt" 2>&1)
        echo "up=$(grep -c 'port_status=1 train_status=2' "$J/results/eth_$h.txt")" ;;
    kill)
        [ -n "${GLXQ_HOST:-}" ] && { echo "$h: stub mode, kill skipped"; exit 0; }  # never on the submit host
        pkill -KILL -u "$USER" -f "glxq_rank.sh|test_tt_fabric|pytest|prted" 2> /dev/null
        sleep 2 ;;
    capture)
        [ -n "${GLXQ_HOST:-}" ] && { echo "$h: stub mode, capture skipped"; exit 0; }  # reads devices: never on the submit host
        (cd "$TREE" && timeout 600 python_env/bin/python "$STAGE/hang_capture.py" "$J/results/${2:?LABEL}_hangcap_$h" \
            --time-budget 540 --meta "label=$2" --meta "host=$h" > "$J/results/${2}_hangcap_$h.log" 2>&1)
        echo "capture rc=$?" ;;
    *) echo "unknown step $1"; exit 2 ;;
esac
