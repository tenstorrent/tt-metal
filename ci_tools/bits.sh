#!/usr/bin/env bash
# Recheck bit dumps on CI (05 13:05 rule: the head merged with current main, sfpi 7.86.0): the exhaustive MAX ROW dumps and
# the ops' outputs on both main and the merge, compared by hash across the two jobs; the exact-fidelity and block-call
# exhaustive checks on the merge (their kernels use the PR's API). usage: bits.sh <main|merge>
set +e +o pipefail
D=/work/ci_tools; O=/tmp/bits; mkdir -p $O; MODE=${1:-merge}
export HWLOCK_HELD=1 PYTHONPATH=/work:$D:${PYTHONPATH:-}
run() { local n=$1; shift; timeout -k 30 1200 python3 "$@" > $O/$n.log 2>&1; echo "== $n rc=$? $(grep -c '^CASE' $O/$n.log) CASE lines $(grep -c ' FAIL' $O/$n.log) FAIL"; grep -aE "Traceback|TT_THROW|TT_FATAL" $O/$n.log | head -3 | cut -c1-300; }
run maxrow $D/exh_maxrow.py $O/maxrow.pt
run denorm $D/maxrow_denorm.py $O/denorm.pt
run bitid $D/rc_bitid.py $O/bitid.pt
run scalar $D/rc_scalar.py $O/scalar.pt
if [ "$MODE" = merge ]; then
  run pow2 $D/exh_pow2.py $O/pow2.txt; echo "-- pow2"; cat $O/pow2.txt
  run block $D/exh_block.py $O/block.txt; echo "-- block"; cat $O/block.txt
fi
python3 - <<'PY'
import hashlib, torch
for n in ("maxrow", "denorm", "bitid", "scalar"):
    try:
        d = torch.load(f"/tmp/bits/{n}.pt", weights_only=True)
    except Exception as e:
        print("HASH", n, "missing", str(e)[:200]); continue
    def walk(prefix, v):
        if isinstance(v, dict):
            for k in sorted(v, key=str): walk(f"{prefix}/{k}", v[k])
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v): walk(f"{prefix}/{i}", x)
        elif torch.is_tensor(v):
            t = v.detach().contiguous().reshape(-1)
            raw = t.numpy().tobytes() if t.dtype in (torch.bool, torch.int64, torch.int32, torch.float32) else t.view(torch.int16 if t.element_size() == 2 else torch.uint8).numpy().tobytes()
            print("HASH", prefix, tuple(v.shape), str(v.dtype), hashlib.sha256(raw).hexdigest()[:16])
        else:
            print("HASH", prefix, repr(v)[:80])
    walk(n, d)
PY
echo "== bits done"
