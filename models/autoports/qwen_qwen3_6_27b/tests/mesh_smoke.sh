#!/usr/bin/env bash
# 1x4 mesh open/close + fabric ring + global semaphore smoke.
#
# Settles first. A board reset returns before the ethernet cores have finished
# re-training, and opening the mesh too early produces
# "Timed out while waiting for active ethernet core N-M to become active again",
# which then reads as a hardware fault rather than as impatience. Default settle
# is 60 s on top of whatever the caller already slept; override with $1.
#
# Exit code is meaningful: 0 only if the mesh opened, a global semaphore was
# created, and the mesh closed cleanly. An earlier version ended in `| head -12`
# and then read $?, which is the exit status of head -- always 0 -- so it
# reported success on a wedged mesh.
SETTLE=${1:-60}
cd ~/tt-metal || exit 2
source python_env/bin/activate
echo "--- settling ${SETTLE}s after reset before opening the mesh"
sleep "$SETTLE"
OUT=$(mktemp)
echo "--- mesh open/close smoke (1x4, fabric ring + global semaphore)"
timeout 300 python - > "$OUT" 2>&1 <<'PY'
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
m = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
print("MESH_OPEN_OK", m.shape)
try:
    s = ttnn.create_global_semaphore(
        m, ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]), 0
    )
    print("SEM_OK")
except Exception as e:
    print("SEM_ERR", type(e).__name__, str(e)[:120])
ttnn.close_mesh_device(m)
print("MESH_CLOSE_OK")
PY
PY_RC=$?
grep -E "MESH_|SEM_|Timed out|TT_THROW|Traceback|RuntimeError" "$OUT" | head -12

RC=0
[ $PY_RC -ne 0 ] && { echo "  python exited $PY_RC"; RC=1; }
for marker in MESH_OPEN_OK SEM_OK MESH_CLOSE_OK; do
  grep -q "$marker" "$OUT" || { echo "  MISSING $marker"; RC=1; }
done
grep -qE "Timed out while waiting for active ethernet core" "$OUT" \
  && { echo "  WEDGED: ethernet core timeout -- reset and settle longer"; RC=1; }
rm -f "$OUT"
# Let the fabric quiesce after the smoke closes before a real workload opens it.
sleep 20
if [ $RC -eq 0 ]; then echo "SMOKE_OK"; else echo "SMOKE_FAILED"; fi
exit $RC
