#!/usr/bin/env bash
#SBATCH --job-name=host-socket
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#
# HostMeshSocket test launcher. Runs strictly under SLURM: submit with sbatch, or
# run inside an existing salloc. Never reserve a machine by hand.
#
#   sbatch --nodes=2 run_host_socket_tests.sh [transport|smoke|hotswap|perf|latency|sweep|soak|all]
#   sbatch --nodes=1 run_host_socket_tests.sh smoke    # both ranks on one host
#   salloc -N2 -t 2:00:00 ./run_host_socket_tests.sh perf
#
# Two nodes is the real configuration. A single node still exercises the whole
# datapath (both ranks on that host, different chips, one NIC port each), which
# makes it the fast correctness loop.
#
# Environment knobs (all optional):
# Modes:
#   transport  host-only page streaming between the two ranks;
#              needs no Tenstorrent device
#   smoke      transport, then the four correctness tests
#   multirank  3 ranks, 2 endpoints: non-participants must not be waited on
#   hotswap    the same body and kernels over a D2D MeshSocket and over this one
#   perf       throughput at 14 KiB pages
#   latency    idle device-to-device round trip, then ack latency under load
#   sweep      page-size sweep at fixed volume, then core-count scaling, then
#              ring-depth scaling (locates the bandwidth-delay product)
#   soak       long-running verified soak (see HOST_SOCKET_SOAK_SECONDS)
#   all        transport, smoke, multirank, hotswap, perf, latency
#
# Environment knobs (all optional):
#   TT_METAL_HOME              repo root; also resolved from SLURM_SUBMIT_DIR
#   HOST_SOCKET_BIN            test binary, if not under $TT_METAL_HOME/build*
#   HOST_SOCKET_VISIBLE_DEVICES  candidate chips, tried in order until one opens.
#                              An entry is "N", or "A:B" to give each rank its own
#                              chip when both share a host. perf/sweep default to
#                              the four x8 chips (5,13,21,29); correctness modes
#                              try a wider list, since those four are contended.
#   HOST_SOCKET_TIMEOUT        per-run timeout, seconds (600; 120 for smoke). A
#                              soak must raise it above HOST_SOCKET_SOAK_SECONDS.
#   HOST_SOCKET_DEVICE_ID      index within the visible set (default 0)
#   HOST_SOCKET_NIC_IF         interface for MPI out-of-band; else docker/flannel
#                              are excluded
#   HOST_SOCKET_MGD            mesh graph descriptor (default: config/two_bh_single_chip_mgd.textproto)
#   HOST_SOCKET_SOAK_SECONDS   soak duration in seconds (soak mode defaults to 3900)
#   HOST_SOCKET_CSV            where throughput rows are appended
#   HOST_SOCKET_MIN_GBPS       fail the perf test below this (default: unset, so
#                              throughput is reported but not gated)

set -uo pipefail

MODE="${1:-smoke}"

# sbatch copies this script to its spool dir, so BASH_SOURCE is not in the repo.
# Walk up from wherever we can see instead of assuming a submit directory.
if [[ -z "${TT_METAL_HOME:-}" ]]; then
    for start in "${SLURM_SUBMIT_DIR:-}" "$(dirname "${BASH_SOURCE[0]}")"; do
        [[ -n "$start" && -d "$start" ]] || continue
        candidate="$(cd "$start" && pwd)"
        while [[ "$candidate" != "/" ]]; do
            if [[ -d "$candidate/tt_metal" && -d "$candidate/tests/tt_metal/multihost/host_socket" ]]; then
                TT_METAL_HOME="$candidate"
                break 2
            fi
            candidate="$(dirname "$candidate")"
        done
    done
fi
[[ -n "${TT_METAL_HOME:-}" ]] || { echo "error: set TT_METAL_HOME to the tt-metal repo root" >&2; exit 2; }
export TT_METAL_HOME
# tt-metal reads this, not TT_METAL_HOME, to resolve relative kernel paths.
: "${TT_METAL_RUNTIME_ROOT:=$TT_METAL_HOME}"
export TT_METAL_RUNTIME_ROOT
HERE="$TT_METAL_HOME/tests/tt_metal/multihost/host_socket"

BIN="${HOST_SOCKET_BIN:-}"
if [[ -z "$BIN" ]]; then
    for b in "$TT_METAL_HOME/build_Release/test/tt_metal/multi_host_socket_transport_tests" \
             "$TT_METAL_HOME/build/test/tt_metal/multi_host_socket_transport_tests"; do
        [[ -x "$b" ]] && { BIN="$b"; break; }
    done
fi
[[ -x "$BIN" ]] || {
    echo "error: test binary not found under $TT_METAL_HOME (build multi_host_socket_transport_tests," >&2
    echo "       or point HOST_SOCKET_BIN at it)" >&2
    exit 2
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "error: no SLURM allocation. Use 'sbatch $0 $MODE' or run inside salloc." >&2
    exit 2
fi

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#NODES[@]} >= 1 )) || { echo "error: empty allocation" >&2; exit 2; }

# Two nodes is the real configuration. One node still exercises the whole
# datapath (both ranks, different chips) and is far easier to schedule.
# Ranks beyond the two endpoints are non-participants: they open no device, so
# they are packed onto the first node. The socket still spans two hosts because
# the test picks rank 0 and rank size-1 as its endpoints, and those land on
# different nodes under the layouts below.
#
# A mode that wants a different rank count calls set_layout again; NRANKS and
# HOSTSPEC must stay in step, so nothing sets them by hand.
set_layout() {
    NRANKS="${1:-2}"
    (( NRANKS >= 2 )) || { echo "error: need at least 2 ranks" >&2; exit 2; }
    local extra=$(( NRANKS - 2 ))
    if (( ${#NODES[@]} >= 2 )); then
        N0="${NODES[0]}"; N1="${NODES[1]}"
        # rank 0 on N0 with any spare ranks beside it, the last rank on N1.
        HOSTSPEC="$N0:$(( 1 + extra )),$N1:1"
        LOOPBACK=0
    else
        N0="${NODES[0]}"; N1="$N0"
        HOSTSPEC="$N0:$NRANKS"
        LOOPBACK=1
    fi

    # Unbound: the relay is a polling thread and sharing one core costs
    # throughput. SLURM hands out one slot per node, so any layout denser than
    # one rank per node needs --oversubscribe or mpirun refuses to map.
    MAP_ARGS="--bind-to none"
    if (( NRANKS > ${#NODES[@]} )); then
        MAP_ARGS="$MAP_ARGS --oversubscribe"
    fi
}
set_layout "${HOST_SOCKET_RANKS:-2}"

MPIRUN=/opt/openmpi-v5.0.7-ulfm/bin/mpirun
[[ -x "$MPIRUN" ]] || MPIRUN=$(command -v mpirun) || { echo "error: no mpirun" >&2; exit 2; }

# MPI now carries the data path too. These hosts have docker, flannel and Calico
# interfaces, and OpenMPI will otherwise try to reach a peer on 172.17.0.1 and
# abort. Exclude them so it picks the 100 GbE port.
if [[ -n "${HOST_SOCKET_NIC_IF:-}" ]]; then
    export OMPI_MCA_btl_tcp_if_include="$HOST_SOCKET_NIC_IF"
    export OMPI_MCA_oob_tcp_if_include="$HOST_SOCKET_NIC_IF"
else
    EXCLUDE="${HOST_SOCKET_NIC_EXCLUDE:-lo,docker0,flannel.1,virbr0,cali+,br-+}"
    export OMPI_MCA_btl_tcp_if_exclude="$EXCLUDE"
    export OMPI_MCA_oob_tcp_if_exclude="$EXCLUDE"
fi

# Open only the chip under test: all 32 costs minutes per rank and contends on
# other tenants' CHIP_IN_USE_* locks. TT_VISIBLE_DEVICES names the PHYSICAL chip
# but tt-metal sees the index within the visible set, so pinning chip 5 makes it
# device 0.
#
# Candidates are tried in order: a chip whose lock another tenant holds makes
# device open *block* rather than fail, hence the timeout-and-retry in run_gtest.
# An entry is "N", or "A:B" to give each rank its own chip when they share a host.
# A lock wedged by an older UMD build can never be recovered, only avoided.
case "$MODE" in
    perf|sweep) DEFAULT_CHIPS_2N="5,13,21,29"; DEFAULT_CHIPS_LB="5:13,21:29" ;;
    *)          DEFAULT_CHIPS_2N="5,13,21,29,0,1,2,3,4,6,7,8,9,10"
                DEFAULT_CHIPS_LB="5:13,21:29,0:1,2:3,6:7,8:9,10:11,12:14,15:16,17:18" ;;
esac
if (( LOOPBACK )); then
    CHIP_CANDIDATES="${HOST_SOCKET_VISIBLE_DEVICES:-$DEFAULT_CHIPS_LB}"
else
    CHIP_CANDIDATES="${HOST_SOCKET_VISIBLE_DEVICES:-$DEFAULT_CHIPS_2N}"
fi
export TT_HOST_SOCKET_DEVICE_ID="${HOST_SOCKET_DEVICE_ID:-0}"
RUN_TIMEOUT="${HOST_SOCKET_TIMEOUT:-600}"
# Many chip pairs to get past contended devices, so keep each attempt short.
case "$MODE" in smoke) RUN_TIMEOUT="${HOST_SOCKET_TIMEOUT:-120}" ;; esac

# World size > 1 makes the control plane require a per-rank mesh binding. The
# descriptor wires nothing between the two meshes: this runs over the host network.
export TT_MESH_GRAPH_DESC_PATH="${HOST_SOCKET_MGD:-$HERE/config/two_bh_single_chip_mgd.textproto}"
[[ -f "$TT_MESH_GRAPH_DESC_PATH" ]] || { echo "error: mesh graph descriptor not found: $TT_MESH_GRAPH_DESC_PATH" >&2; exit 2; }
# Benchmark rows land here so a sweep is machine-readable.
export TT_HOST_SOCKET_CSV="${HOST_SOCKET_CSV:-$HERE/results/host_socket_${SLURM_JOB_ID:-local}.csv}"
mkdir -p "$(dirname "$TT_HOST_SOCKET_CSV")"
[[ -n "${HOST_SOCKET_MIN_GBPS:-}" ]] && export TT_HOST_SOCKET_MIN_GBPS="$HOST_SOCKET_MIN_GBPS"

# Relative kernel paths resolve against the runtime root, so launch from there.
cd "$TT_METAL_HOME" || exit 2

echo "== job ${SLURM_JOB_ID}: $N0 + $N1 (loopback=$LOOPBACK)"
echo "== mode=$MODE binary=$BIN device=$TT_HOST_SOCKET_DEVICE_ID chips=$CHIP_CANDIDATES"
echo "== mpirun=$MPIRUN"
date

run_gtest() {  # label, gtest_filter, then VAR=VAL overrides
    local label="$1" filter="$2"; shift 2
    local rc=0

    local chip
    for chip in ${CHIP_CANDIDATES//,/ }; do
        export TT_VISIBLE_DEVICES_PER_RANK="$chip"
        echo; echo "########## $label (chips $chip) ##########"
        # A stalled socket is a device-side spin with no timeout (a lost arrival signal
        # hangs the kernel, exactly as it would on a D2D socket), and mpirun does
        # not always die on SIGTERM when its children are wedged in the driver, so
        # follow up with a kill.
        local log="${TMPDIR:-/tmp}/host_socket_${SLURM_JOB_ID:-local}_$$.log"
        timeout --kill-after=30s "$RUN_TIMEOUT" \
        env "$@" "$MPIRUN" -n "$NRANKS" --host "$HOSTSPEC" \
            --allow-run-as-root --tag-output $MAP_ARGS \
            -x TT_METAL_HOME -x TT_METAL_RUNTIME_ROOT -x TT_MESH_GRAPH_DESC_PATH \
            -x TT_HOST_SOCKET_DEVICE_ID -x TT_VISIBLE_DEVICES_PER_RANK \
            -x TT_HOST_SOCKET_PAGE_SIZE -x TT_HOST_SOCKET_FIFO_PAGES \
            -x TT_HOST_SOCKET_NUM_CORES -x TT_HOST_SOCKET_BYTES \
            -x TT_HOST_SOCKET_ITERS -x TT_HOST_SOCKET_SOAK_SECONDS \
            -x TT_HOST_SOCKET_BATCH_PAGES -x TT_HOST_SOCKET_MIN_GBPS \
                -x TT_HOST_SOCKET_CSV \
            -x TT_HOST_SOCKET_CSV_LATENCY -x TT_HOST_SOCKET_LAT_ITERS -x TT_HOST_SOCKET_IDLE_RTT_US \
            -x OMPI_MCA_btl_tcp_if_include -x OMPI_MCA_oob_tcp_if_include \
            -x OMPI_MCA_btl_tcp_if_exclude -x OMPI_MCA_oob_tcp_if_exclude \
            "$HERE/rank_env.sh" "$BIN" --gtest_filter="$filter" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        if (( rc == 124 )); then
            echo "----- chip $chip: timed out after ${RUN_TIMEOUT}s (device lock held?); trying next -----"
            rm -f "$log"; continue
        fi
        # A board that will not launch firmware is an environment problem, not a
        # test result, and the next candidate is usually healthy. Retry it rather
        # than reporting a failure the socket had no part in.
        if grep -qE 'failed to initialize FW|waiting for physical cores to finish' "$log"; then
            echo "----- chip $chip: board did not initialize firmware; trying next -----"
            rm -f "$log"; continue
        fi
        rm -f "$log"
        echo "----- rc=$rc ($label, chip $chip) -----"
        date
        return $rc
    done

    echo "----- $label: every candidate chip ($CHIP_CANDIDATES) timed out -----"
    date
    return 124
}

rc=0
case "$MODE" in
    transport)
        # Host-only: streams pages between the two ranks and verifies every byte.
        # Opens no TT device, but still goes through the scheduler rather than
        # being run by hand.
        run_gtest "host transport" 'HostTransportTest.*' || rc=$?
        ;;
    smoke)
        run_gtest "host transport" 'HostTransportTest.*' || rc=$?
        run_gtest "correctness" 'HostSocketTest.*Correctness*' || rc=$?
        ;;
    latency)
        # Idle device-to-device round trip, then the streaming ack round trip.
        # Both are timed on a single clock, so no cross-host sync is needed; the
        # second run is given the first's RTT so it can also report the forward
        # path with the return transit removed.
        export TT_HOST_SOCKET_CSV_LATENCY="${HOST_SOCKET_CSV_LATENCY:-$HERE/results/latency_${SLURM_JOB_ID:-local}.csv}"
        mkdir -p "$(dirname "$TT_HOST_SOCKET_CSV_LATENCY")"
        run_gtest "round-trip latency" 'HostSocketLatencyTest.RoundTrip' || rc=$?
        idle_rtt=$(awk -F, '$1=="d2d_round_trip"{print $5}' "$TT_HOST_SOCKET_CSV_LATENCY" 2>/dev/null | tail -1)
        [[ -n "${idle_rtt:-}" ]] && echo "== idle round trip p50 = ${idle_rtt} us"
        run_gtest "streaming ack latency" 'HostSocketLatencyTest.StreamingAckLatency' \
            TT_HOST_SOCKET_IDLE_RTT_US="${idle_rtt:-0}" || rc=$?
        ;;
    multirank)
        # Three ranks, two endpoints. Exercises the path where a socket is built
        # inside a world larger than itself: the non-participant never calls the
        # socket's barriers, so anything collective over the full context hangs.
        set_layout 3
        export TT_MESH_GRAPH_DESC_PATH="$HERE/config/three_bh_single_chip_mgd.textproto"
        # Every rank opens a device, and ranks 0 and 1 share a node, so each entry
        # names a chip per rank: "rank0:rank1:rank2". Rank 2 is on the other node
        # and may reuse rank 0's chip id.
        CHIP_CANDIDATES="${HOST_SOCKET_VISIBLE_DEVICES:-5:13:5,21:29:21,0:1:0,2:3:2,6:7:6}"
        echo "== multirank: $NRANKS ranks over $HOSTSPEC"
        run_gtest "multi-rank world" 'HostSocketMultiRankTest.*' || rc=$?
        ;;
    hotswap)
        # Same body and same kernels over a D2D MeshSocket and over a
        # HostMeshSocket; only the socket type and SOCKET_MODE differ.
        run_gtest "hot swap d2d vs host" 'SocketHotSwapTest.*' || rc=$?
        ;;
    perf)
        run_gtest "throughput @14K" 'HostSocketTest.Throughput' || rc=$?
        ;;
    sweep)
        # Page-size sweep. Volume is held FIXED across page sizes (the largest
        # multiple of the page size under 1 GiB) so the points are comparable;
        # scaling volume with page size instead would make the small-page points
        # measure kernel-launch overhead rather than link bandwidth. Kept under
        # 2^31 because the kernels take data_size as a uint32 compile-time arg.
        sweep_volume=1073741824
        for ps in 2048 4096 8192 14336 32768 65536; do
            run_gtest "throughput @${ps}B" 'HostSocketTest.Throughput' \
                TT_HOST_SOCKET_PAGE_SIZE="$ps" \
                TT_HOST_SOCKET_ITERS=4 \
                TT_HOST_SOCKET_BYTES=$(( (sweep_volume / ps) * ps )) || rc=$?
        done
        # Then core-count scaling at the target page size.
        for nc in 1 2 4 8; do
            run_gtest "throughput ${nc} core(s) @14336B" 'HostSocketTest.Throughput' \
                TT_HOST_SOCKET_NUM_CORES="$nc" || rc=$?
        done
        # Then ring depth, which locates the bandwidth-delay product empirically:
        # throughput climbs while the ring is too shallow to cover the credit
        # round trip and plateaus once it is. Anything past the knee is buying
        # latency for no bandwidth.
        for fp in 2 4 8 16 32 64 128; do
            run_gtest "throughput ring ${fp} pages @14336B" 'HostSocketTest.Throughput' \
                TT_HOST_SOCKET_FIFO_PAGES="$fp" \
                TT_HOST_SOCKET_BYTES=$(( 14336 * 4096 )) || rc=$?
        done
        ;;
    soak)
        run_gtest "soak" 'HostSocketTest.Soak' \
            TT_HOST_SOCKET_SOAK_SECONDS="${HOST_SOCKET_SOAK_SECONDS:-3900}" || rc=$?
        ;;
    all)
        # The qualification sequence: host-only transport, then correctness, then
        # throughput. 'sweep' is deliberately separate; it is long.
        "$0" transport || rc=$?
        "$0" smoke || rc=$?
        "$0" multirank || rc=$?
        "$0" hotswap || rc=$?
        "$0" perf || rc=$?
        "$0" latency || rc=$?
        ;;
    *)
        echo "unknown mode: $MODE (want transport|smoke|multirank|hotswap|perf|latency|sweep|soak|all)" >&2
        exit 2
        ;;
esac

echo; echo "== done (rc=$rc)"; date
exit $rc
