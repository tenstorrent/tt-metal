#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:?usage: run_migration_driver.sh <producer_manifest.yaml> [host_list] [tcp_iface] [args...]}"
shift
HOST_LIST="${1:-}"
[ $# -gt 0 ] && shift
TCP_IFACE="${1:-ens5f0np0}"
[ $# -gt 0 ] && shift

TT_METAL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export TT_METAL_HOME PYTHONPATH="${PYTHONPATH:-$TT_METAL_HOME}"
[ -z "${VIRTUAL_ENV:-}" ] && [ -f "$TT_METAL_HOME/python_env/bin/activate" ] && source "$TT_METAL_HOME/python_env/bin/activate"
cd "$TT_METAL_HOME"

MODULE="models.demos.common.prefill.runners.migration_driver"
NUM_HOSTS=$(printf '%s' "$HOST_LIST" | tr ',' '\n' | grep -c . || true)

if [ "$NUM_HOSTS" -le 1 ]; then
  echo "[run_migration_driver] single process (host_list='${HOST_LIST}'); verifying THIS host's layers only."
  exec python3 -m "$MODULE" --manifest "$MANIFEST" "$@"
fi

LAUNCHER="$(command -v mpirun-ulfm || command -v mpirun || true)"
[ -n "$LAUNCHER" ] || { echo "[run_migration_driver] no mpirun-ulfm or mpirun on PATH" >&2; exit 1; }

FWD=(-x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x TT_METAL_HOME)
[ -n "${TT_METAL_CACHE:-}" ] && FWD+=(-x TT_METAL_CACHE)
[ -n "${VIRTUAL_ENV:-}" ] && FWD+=(-x VIRTUAL_ENV)
[ -n "${LOGURU_LEVEL:-}" ] && FWD+=(-x LOGURU_LEVEL)
while IFS= read -r var; do
  FWD+=(-x "$var")
done < <(compgen -e | grep -E '^(PREFILL_|MIGRATION_)' | sort)

echo "[run_migration_driver] $NUM_HOSTS host(s): $HOST_LIST (rank 0 = ${HOST_LIST%%,*}, must be $(hostname))"
echo "[run_migration_driver] tcp interface: $TCP_IFACE — must match the runner's"

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
