#!/usr/bin/env bash
set -euo pipefail
range_plan=$1
range_hash=$2
range_role=$3
launch_packet=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
owner_packet=$launch_packet
environment_script=$(python3 -I -S -B -c 'import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]; d=json.loads(p.read_bytes()); f=pathlib.Path(d["environment_script"]); assert hashlib.sha256(f.read_bytes()).hexdigest()==d["pins"][str(f)]; print(f)' "$range_plan" "$range_hash")
source "$environment_script"
export PREFILL_FABRIC_MODE=1d_ring
unset TT_METAL_SLOW_DISPATCH_MODE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TASK8_CPU_THREADS=1
export PYTHONUNBUFFERED=1 GOMAXPROCS=1
export TT_MESH_GRAPH_DESC_PATH="$PREFILL_REPO/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
while IFS= read -r key; do
    case "$key" in TT_METAL_WATCHER*|TT_METAL_DEVICE_PROFILER*|TT_METAL_PROFILER*|TT_METAL_DPRINT*|TRACY_*) unset "$key";; esac
done < <(compgen -v)
cd "$PREFILL_REPO"
python3 -I -S -B "$launch_packet/node-preflight.py" "$range_plan" "$range_hash" "$range_role"
exec "$PREFILL_PYTHON" -B "$owner_packet/single_cpu_exec.py" "$PREFILL_PYTHON" -B "$owner_packet/supervise_owner.py" --plan "$range_plan" --plan-sha256 "$range_hash" --role "$range_role"
