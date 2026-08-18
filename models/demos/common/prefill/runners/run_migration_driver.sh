#!/usr/bin/env bash
# Launcher for the KV-migration driver — one driver process per pipeline host.
#
# Sibling of run_pipeline_prefill.sh, and for the same reason: launch concerns (host list, MPI transport,
# env forwarding) live in shell, not in the Python they launch. The driver module itself never spawns
# anything; it only splits by rank once MPI has placed it.
#
# WHY ONE PROCESS PER HOST: both of the driver's read-backs — the source golden PCC and the destination
# --verify-migration — go over read_dram_umd, which reaches only the chips in the machine it runs on. One
# process on an N-host pipeline runner therefore verifies 1/N of the layers and silently skips the rest.
# With a process per host, each rank reads its own galaxy and the per-rank verdicts are allgathered, so the
# union covers the whole model. Rank 0 additionally does the whole run: H2D feed, ack drain, migrate, both
# sidecar files.
#
# Usage:
#   run_migration_driver.sh <producer_manifest.yaml> [host_list] [tcp_iface] [extra driver args...]
#
#   <producer_manifest.yaml>  path (relative to TT_METAL_HOME or absolute) to the producer manifest.
#   [host_list]               mpirun --host value, in RANK ORDER. Rank 0 comes first and MUST be this host:
#                             it alone attaches the H2D service and the /mig_ep*_ queues, which exist only
#                             where the runner's rank 0 runs. Use the SAME list you passed to
#                             run_pipeline_prefill.sh. Omit it (or pass a single host) to run one process,
#                             covering this host's layers only — correct for a 1-galaxy runner.
#   [tcp_iface]               NIC for MPI TCP. Default: ens5f0np0. MUST match what the runner was launched
#                             with (run_pipeline_prefill.sh's 3rd argument): these hosts are multi-homed
#                             and docker0 carries the SAME address on every one, so an unpinned OpenMPI
#                             advertises on one NIC and connects on another and MPI_Init never completes —
#                             every rank logs "applied manifest" and then goes silent.
#   [extra driver args...]    Anything else is passed through to the module verbatim, on every rank, e.g.
#                             --verify-migration both --verify-migration-layers 0,30,60
#
# Examples:
#   # 2 galaxies, full coverage
#   ./run_migration_driver.sh \
#     models/demos/common/prefill/runners/producer_manifests/producer_manifest_ds_2galaxy_loopback.yaml \
#     bh-glx-110-c04u02:1,bh-glx-110-c04u08:1
#
#   # single galaxy — no MPI at all
#   ./run_migration_driver.sh models/demos/common/prefill/runners/producer_manifests/<manifest>.yaml
set -euo pipefail

MANIFEST="${1:?usage: run_migration_driver.sh <producer_manifest.yaml> [host_list] [tcp_iface] [args...]}"
shift
HOST_LIST="${1:-}"
[ $# -gt 0 ] && shift
TCP_IFACE="${1:-ens5f0np0}"
[ $# -gt 0 ] && shift

# TT_METAL_HOME = the tt-metal tree this script lives in
# (models/demos/common/prefill/runners -> 5 levels up). Matches run_pipeline_prefill.sh.
TT_METAL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export TT_METAL_HOME PYTHONPATH="${PYTHONPATH:-$TT_METAL_HOME}"
[ -z "${VIRTUAL_ENV:-}" ] && [ -f "$TT_METAL_HOME/python_env/bin/activate" ] && source "$TT_METAL_HOME/python_env/bin/activate"
cd "$TT_METAL_HOME"

MODULE="models.demos.common.prefill.runners.migration_driver"
NUM_HOSTS=$(printf '%s' "$HOST_LIST" | tr ',' '\n' | grep -c . || true)

# One host (or none): no MPI. The driver runs standalone, rank 0 of 1, and verifies this host's layers.
if [ "$NUM_HOSTS" -le 1 ]; then
  echo "[run_migration_driver] single process (host_list='${HOST_LIST}'); verifying THIS host's layers only."
  exec python3 -m "$MODULE" --manifest "$MANIFEST" "$@"
fi

# mpirun-ulfm is what tt-run prefers; fall back to plain mpirun.
LAUNCHER="$(command -v mpirun-ulfm || command -v mpirun || true)"
[ -n "$LAUNCHER" ] || { echo "[run_migration_driver] no mpirun-ulfm or mpirun on PATH" >&2; exit 1; }

# ttrun only auto-propagates TT_*/ARCH_*/... , so PATH/PYTHONPATH must be named or a peer rank resolves a
# bare python3 with no ttnn. Every exported PREFILL_*/MIGRATION_* goes too: the ranks coordinate over MPI
# collectives and must agree on the config, so a knob that reached only rank 0 would desynchronize them.
FWD=(-x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x TT_METAL_HOME)
[ -n "${TT_METAL_CACHE:-}" ] && FWD+=(-x TT_METAL_CACHE)
[ -n "${VIRTUAL_ENV:-}" ] && FWD+=(-x VIRTUAL_ENV)
[ -n "${LOGURU_LEVEL:-}" ] && FWD+=(-x LOGURU_LEVEL)
while IFS= read -r var; do
  FWD+=(-x "$var")
done < <(compgen -e | grep -E '^(PREFILL_|MIGRATION_)' | sort)

echo "[run_migration_driver] $NUM_HOSTS host(s): $HOST_LIST (rank 0 = ${HOST_LIST%%,*}, must be $(hostname))"
echo "[run_migration_driver] tcp interface: $TCP_IFACE — must match the runner's"

# --mca btl self,tcp --mca btl_tcp_if_include <iface>: the same transport arguments ttrun gives the runner
# (see default_multihost_mpi_args in ttnn/ttnn/distributed/ttrun.py). Both jobs span the same hosts, so
# they have to agree on the network.
exec "$LAUNCHER" \
  --host "$HOST_LIST" \
  --map-by slot \
  --bind-to none \
  --tag-output \
  --allow-run-as-root \
  --mca btl self,tcp \
  --mca btl_tcp_if_include "$TCP_IFACE" \
  "${FWD[@]}" \
  python3 -m "$MODULE" --manifest "$MANIFEST" "$@"
