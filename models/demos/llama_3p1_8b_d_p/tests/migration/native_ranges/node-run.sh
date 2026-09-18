#!/usr/bin/env bash
set -euo pipefail
packet=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
environment_script=$(python3 -I -S -B -c 'import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]; d=json.loads(p.read_bytes()); f=pathlib.Path(d["environment_script"]); pins=d.get("pins",d.get("source_pins",{})); assert hashlib.sha256(f.read_bytes()).hexdigest()==pins[str(f)]; print(f)' "$1" "$2")
source "$environment_script"
export PREFILL_FABRIC_MODE=1d_ring PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TASK8_CPU_THREADS=1 GOMAXPROCS=1
export TT_MESH_GRAPH_DESC_PATH="$PREFILL_REPO/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
unset TT_METAL_SLOW_DISPATCH_MODE PYTEST_ADDOPTS
while IFS= read -r key; do
 case "$key" in TT_METAL_WATCHER*|TT_METAL_DEVICE_PROFILER*|TT_METAL_PROFILER*|TT_METAL_DPRINT*|TRACY_*) unset "$key";; esac
done < <(compgen -v)
cd "$PREFILL_REPO"
python3 -I -S -B "$packet/node-preflight.py" "$1" "$2" "$3"
exec "$PREFILL_PYTHON" -B "$packet/supervise_owner.py" --plan "$1" --plan-sha256 "$2" --role "$3"
