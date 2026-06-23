#!/usr/bin/env bash
# Single-host isolation + pinpoint for the PSD-discovery device bring-up hang.
#
# Background: in a multi-host run, one host (e.g. bh-glx-120-c04u14 / rank 6) hangs in UMD
# Cluster::start_device ("Starting devices in cluster", cluster.cpp) BEFORE PSD discovery, so
# every other rank blocks at the discovery entry barrier and the job is killed. This script
# reproduces that bring-up on a SINGLE host and names the exact chip that hangs.
#
# Usage (run ON the suspect host, with env.sh sourced so tt-run / tt-smi / TT_METAL_HOME are set):
#     source /data/rsong/tt-blaze-5/env.sh
#     bash tools/scaleout/isolate_host_device_init.sh [tcp_interface]
#
# Requires the build with the [PSD-DEBUG] per-chip start_device logging (UMD cluster.cpp).

set +e
HOST=$(hostname)
IFACE="${1:-ens5f0np0}"
TS=$(date +%H%M%S)
SMI_LOG="/tmp/isolate_${HOST}_ttsmi_${TS}.log"
INIT_LOG="/tmp/isolate_${HOST}_devinit_${TS}.log"
MGD="tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"

echo "================ single-host isolation on ${HOST} (iface=${IFACE}) ================"

echo
echo "----- [1] tt-smi health snapshot (per-chip: temp / AICLK / ARC / link / fw) -----"
echo "    A chip whose ARC is wedged will show up here as unreadable / error, or tt-smi itself will stall."
timeout 120 tt-smi -ls 2>&1 | tail -40   # non-interactive board list (no TUI)
timeout 120 tt-smi -s --snapshot_no_tty -f "${SMI_LOG}" >/dev/null 2>&1
echo "    (snapshot -> ${SMI_LOG}; rc=$? -- non-zero/timeout here itself implicates a wedged chip)"

echo
echo "----- [2] single-host device bring-up (reproduces Cluster::start_device) -----"
echo "    Watch the [PSD-DEBUG] 'bringing up chip N' lines: the LAST one with no matching"
echo "    'chip N brought up OK' is the chip whose start_device() hung."
echo "    (map logical chip N -> PCIe id via the 'Opening local chip ids/PCIe ids' line.)"
: "${TT_METAL_HOME:?source env.sh first}"
cd "${TT_METAL_HOME}" || exit 2

TT_METAL_LOGGER_TYPES=UMD,Fabric TT_METAL_LOGGER_LEVEL=debug TT_METAL_SLOW_DISPATCH_MODE=1 \
    timeout 240 tt-run \
      --tcp-interface "${IFACE}" \
      --hosts "${HOST}" \
      --force-rediscovery \
      --mesh-graph-descriptor "${MGD}" \
      true 2>&1 | tee "${INIT_LOG}" \
    | grep -aE "PSD-DEBUG|Starting devices in cluster|brought up|Opening local chip|AICLK|[Ee]rror|Completed topology|Cluster constructor"
rc=${PIPESTATUS[0]}

echo
echo "================ RESULT ================"
if [ "${rc}" = "124" ]; then
    echo "TIMED OUT (240s) -> device bring-up hung on ${HOST}. Stuck chip = last '[PSD-DEBUG] bringing up chip N' below:"
    grep -aE "\[PSD-DEBUG\] start_device: (bringing up|.* brought up OK)" "${INIT_LOG}" | tail -4
else
    echo "completed rc=${rc} (no hang on ${HOST} this run)."
fi
echo "Full device-init log: ${INIT_LOG}"
echo "tt-smi snapshot:      ${SMI_LOG}"
