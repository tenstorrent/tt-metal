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
#   transport  host-only RDMA loopback checks; needs no Tenstorrent device
#   smoke      transport, then the four correctness tests
#   hotswap    the same body and kernels over a D2D MeshSocket and over this one
#   perf       throughput at 14 KiB pages
#   latency    idle device-to-device round trip, then ack latency under load
#   sweep      page-size sweep at fixed volume, then core-count scaling, then
#              ring-depth scaling (locates the bandwidth-delay product)
#   soak       long-running verified soak (see HOST_SOCKET_SOAK_SECONDS)
#   all        transport, smoke, hotswap, perf
#
# Environment knobs (all optional):
#   TT_METAL_HOME              repo root; also resolved from SLURM_SUBMIT_DIR
#   HOST_SOCKET_BIN            test binary, if not under $TT_METAL_HOME/build*
#   HOST_SOCKET_VISIBLE_DEVICES  candidate chips, tried in order until one opens.
#                              An entry is "N", or "A:B" to give each rank its own
#                              chip when both share a host. Defaults depend on the
#                              mode: perf and sweep use only the four PCIe x8 chips
#                              (5, 13, 21, 29), since the other 28 are x1 and
#                              cannot reach target throughput; the correctness
#                              modes try a wider list, because they do not need
#                              bandwidth and the x8 chips are the contended ones.
#   HOST_SOCKET_TIMEOUT        per-run timeout in seconds (default 600, 120 for
#                              smoke). Device open normally takes ~10 s, so a small
#                              value makes the occupied-chip retry quick; a soak
#                              must raise it above HOST_SOCKET_SOAK_SECONDS.
#   HOST_SOCKET_DEVICE_ID      index within the visible set (default 0)
#   HOST_SOCKET_NICS           per-rank RDMA device names, "a:b" (default: both
#                              ranks on rocep201s0f0 in the single-node mode,
#                              auto-selected otherwise)
#   HOST_SOCKET_NIC_IF         host interface MPI should use for out-of-band;
#                              otherwise the docker/flannel interfaces are excluded
#   HOST_SOCKET_MGD            mesh graph descriptor (default: config/two_bh_single_chip_mgd.textproto)
#   HOST_SOCKET_SOAK_SECONDS   soak duration in seconds (soak mode defaults to 3900)
#   HOST_SOCKET_CSV            where throughput rows are appended
#   HOST_SOCKET_MIN_GBPS       fail the perf test below this (default: unset, so
#                              throughput is reported but not gated)

set -uo pipefail

MODE="${1:-smoke}"

# sbatch copies the batch script into its spool directory, so BASH_SOURCE does not
# point into the repo. Resolve the repo from TT_METAL_HOME or the submit directory,
# and only fall back to the script's own location for a direct (salloc) run.
if [[ -z "${TT_METAL_HOME:-}" ]]; then
    for candidate in "${SLURM_SUBMIT_DIR:-}/../../../.." "$(dirname "${BASH_SOURCE[0]}")/../../../.."; do
        [[ -n "$candidate" ]] || continue
        if [[ -d "$candidate/tt_metal" && -d "$candidate/tests" ]]; then
            TT_METAL_HOME="$(cd "$candidate" && pwd)"
            break
        fi
    done
fi
[[ -n "${TT_METAL_HOME:-}" ]] || { echo "error: set TT_METAL_HOME to the tt-metal repo root" >&2; exit 2; }
export TT_METAL_HOME
# What tt-metal actually reads to locate its runtime and resolve relative kernel
# paths; TT_METAL_HOME alone is not enough.
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

# Two nodes is the real configuration: one Galaxy per host, RDMA between them.
# With a single node both ranks run there on different chips, with rank 0 on one
# NIC port and rank 1 on the other -- same-port self-connect would need the NIC
# to hairpin, using the two functions does not. That mode exercises the whole
# datapath and is far easier to schedule, so it is the fast correctness loop.
if (( ${#NODES[@]} >= 2 )); then
    N0="${NODES[0]}"; N1="${NODES[1]}"
    HOSTSPEC="$N0:1,$N1:1"
    LOOPBACK=0
else
    N0="${NODES[0]}"; N1="$N0"
    HOSTSPEC="$N0:2"
    LOOPBACK=1
fi

MPIRUN=/opt/openmpi-v5.0.7-ulfm/bin/mpirun
[[ -x "$MPIRUN" ]] || MPIRUN=$(command -v mpirun) || { echo "error: no mpirun" >&2; exit 2; }

# MPI carries only the out-of-band control path here; the socket's data path goes
# straight to the RDMA NIC through verbs. These hosts have a docker bridge and a
# flannel overlay alongside the cluster NIC, and OpenMPI will otherwise try to
# reach a peer on 172.17.0.1 and abort. Name an interface to pin it, else exclude
# the virtual ones, which is portable across differing NIC names.
if [[ -n "${HOST_SOCKET_NIC_IF:-}" ]]; then
    export OMPI_MCA_btl_tcp_if_include="$HOST_SOCKET_NIC_IF"
    export OMPI_MCA_oob_tcp_if_include="$HOST_SOCKET_NIC_IF"
else
    EXCLUDE="${HOST_SOCKET_NIC_EXCLUDE:-lo,docker0,flannel.1,virbr0}"
    export OMPI_MCA_btl_tcp_if_exclude="$EXCLUDE"
    export OMPI_MCA_oob_tcp_if_exclude="$EXCLUDE"
fi

# Open only the chip under test: constructing all 32 chips costs minutes of
# start-up per rank and contends on other tenants' CHIP_IN_USE_* device locks.
#
# On these Blackhole Galaxy hosts only 4 of the 32 chips have a PCIe x8 link (one
# per tray, ASIC location 6); the NIC<->BH p2p sweep measured chips 5, 13, 21 and
# 29 at ~11.3 GiB/s and every other chip at ~3 GiB/s. So TT_VISIBLE_DEVICES names
# the PHYSICAL chip, while the id tt-metal sees is the index within that visible
# set -- pinning one chip makes it device 0, not 5.
# Candidate chips, tried in order. These nodes are shared and another tenant's
# long-running process holds a chip's CHIP_IN_USE_<n>_PCIe lock for its lifetime,
# which makes device open block rather than fail -- hence the timeout-and-retry in
# run_gtest below.
# Candidate chip assignments, tried in order. An entry may be "N" (both ranks on
# chip N, for the two-node case) or "A:B" (rank 0 on A, rank 1 on B, needed when
# both ranks share a host).
# Throughput needs an x8 chip. Correctness does not, and the x8 chips are the ones
# every other tenant wants -- and a chip whose CHIP_IN_USE_<n>_PCIe lock was left
# wedged by an older UMD build can never be recovered, only avoided. So the
# correctness modes get a wide candidate list and perf keeps to the x8 four.
case "$MODE" in
    perf|sweep) DEFAULT_CHIPS_2N="5,13,21,29"; DEFAULT_CHIPS_LB="5:13,21:29" ;;
    *)          DEFAULT_CHIPS_2N="5,13,21,29,0,1,2,3,4,6,7,8,9,10"
                DEFAULT_CHIPS_LB="5:13,21:29,0:1,2:3,6:7,8:9,10:11,12:14,15:16,17:18" ;;
esac
if (( LOOPBACK )); then
    CHIP_CANDIDATES="${HOST_SOCKET_VISIBLE_DEVICES:-$DEFAULT_CHIPS_LB}"
    # Both ranks on the same port: only f0 carries an IP on these hosts, so f1 has no
    # RoCEv2 GID to select. Two RC queue pairs on one port loop back inside the HCA,
    # so this needs no switch hairpin.
    export TT_HOST_SOCKET_RDMA_DEV_PER_RANK="${HOST_SOCKET_NICS:-rocep201s0f0:rocep201s0f0}"
else
    CHIP_CANDIDATES="${HOST_SOCKET_VISIBLE_DEVICES:-$DEFAULT_CHIPS_2N}"
    [[ -n "${HOST_SOCKET_NICS:-}" ]] && export TT_HOST_SOCKET_RDMA_DEV_PER_RANK="$HOST_SOCKET_NICS"
fi
: "${TT_HOST_SOCKET_RDMA_DEV_PER_RANK:=}"
export TT_HOST_SOCKET_DEVICE_ID="${HOST_SOCKET_DEVICE_ID:-0}"
RUN_TIMEOUT="${HOST_SOCKET_TIMEOUT:-600}"
# Correctness sweeps many chip pairs to get past contended devices, so keep each
# attempt short; a working run finishes these tests in well under a minute.
case "$MODE" in smoke) RUN_TIMEOUT="${HOST_SOCKET_TIMEOUT:-120}" ;; esac

# Leave the ranks unbound: the relay is a polling thread and pinning it to one
# core alongside the test thread costs throughput. In loopback both ranks share a
# node, which usually has only one SLURM slot, so let mpirun oversubscribe it.
MAP_ARGS="--bind-to none"
(( LOOPBACK )) && MAP_ARGS="$MAP_ARGS --oversubscribe"

# With a world size above 1 the control plane requires each rank's mesh binding,
# so declare two independent single-chip meshes (one per rank). The socket runs
# over RDMA, not ethernet, so the descriptor deliberately wires nothing between
# them. Set per rank below via mpirun's MPMD segments.
export TT_MESH_GRAPH_DESC_PATH="${HOST_SOCKET_MGD:-$HERE/config/two_bh_single_chip_mgd.textproto}"
[[ -f "$TT_MESH_GRAPH_DESC_PATH" ]] || { echo "error: mesh graph descriptor not found: $TT_MESH_GRAPH_DESC_PATH" >&2; exit 2; }
# Benchmark rows land here so a sweep is machine-readable.
export TT_HOST_SOCKET_CSV="${HOST_SOCKET_CSV:-$HERE/results/host_socket_${SLURM_JOB_ID:-local}.csv}"
mkdir -p "$(dirname "$TT_HOST_SOCKET_CSV")"
[[ -n "${HOST_SOCKET_RDMA_DEV:-}" ]] && export TT_HOST_SOCKET_RDMA_DEV="$HOST_SOCKET_RDMA_DEV"
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
        # A stalled socket is a device-side spin with no timeout (a lost doorbell
        # hangs the kernel, exactly as it would on a D2D socket), and mpirun does
        # not always die on SIGTERM when its children are wedged in the driver, so
        # follow up with a kill.
        timeout --kill-after=30s "$RUN_TIMEOUT" \
        env "$@" "$MPIRUN" -n 2 --host "$HOSTSPEC" \
            --allow-run-as-root --tag-output $MAP_ARGS \
            -x TT_METAL_HOME -x TT_METAL_RUNTIME_ROOT -x TT_MESH_GRAPH_DESC_PATH \
            -x TT_HOST_SOCKET_DEVICE_ID -x TT_VISIBLE_DEVICES_PER_RANK \
            -x TT_HOST_SOCKET_RDMA_DEV_PER_RANK \
            -x TT_HOST_SOCKET_PAGE_SIZE -x TT_HOST_SOCKET_FIFO_PAGES \
            -x TT_HOST_SOCKET_NUM_CORES -x TT_HOST_SOCKET_BYTES \
            -x TT_HOST_SOCKET_ITERS -x TT_HOST_SOCKET_SOAK_SECONDS \
            -x TT_HOST_SOCKET_BATCH_PAGES -x TT_HOST_SOCKET_MIN_GBPS \
                -x TT_HOST_SOCKET_RDMA_DEV -x TT_HOST_SOCKET_GID_INDEX -x TT_HOST_SOCKET_CSV \
            -x TT_HOST_SOCKET_CSV_LATENCY -x TT_HOST_SOCKET_LAT_ITERS -x TT_HOST_SOCKET_IDLE_RTT_US \
            -x OMPI_MCA_btl_tcp_if_include -x OMPI_MCA_oob_tcp_if_include \
            -x OMPI_MCA_btl_tcp_if_exclude -x OMPI_MCA_oob_tcp_if_exclude \
            "$HERE/rank_env.sh" "$BIN" --gtest_filter="$filter"
        rc=$?
        if (( rc == 124 )); then
            echo "----- chip $chip: timed out after ${RUN_TIMEOUT}s (device lock held?); trying next -----"
            continue
        fi
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
        # Host-only: cross-connects two queue pairs on one NIC port, no TT device.
        # Still launched through the scheduler rather than run by hand.
        run_gtest "rdma transport" 'HostTransportTest.*' || rc=$?
        ;;
    smoke)
        run_gtest "rdma transport" 'HostTransportTest.*' || rc=$?
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
        "$0" hotswap || rc=$?
        "$0" perf || rc=$?
        "$0" latency || rc=$?
        ;;
    *)
        echo "unknown mode: $MODE (want transport|smoke|hotswap|perf|latency|sweep|soak|all)" >&2
        exit 2
        ;;
esac

echo; echo "== done (rc=$rc)"; date
exit $rc
