#!/usr/bin/env bash
# ttnn bits and op-level A/B on one CI Blackhole runner (knowledge/process/17_CI_AB.md). The job's tree is the PR head;
# before each run a variant's device files (ci_ab/var/<v>) are copied into /work and into the installed runtime root,
# with a fresh JIT cache per run. ci_ab/job.sh sets BITS_VARIANTS, bits() (one variant's dumps into $1), COMPARE
# pairs, PROF_PAIRS ("A B"), PROF_CASES and PROF_SCRIPT. Op level: every case in its own tracy process, ABBA twice.
# Kernel ELF hashes per variant show the swap reaches the compiler; results go to the log as compact lines.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; OUT=/tmp/sfo; rm -rf $OUT; mkdir -p $OUT/log $OUT/prof; cd $R
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== runner $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format=%h) runtime root $RT"
for v in $(ls $S/var); do echo "== var $v: $(cd $S/var/$v && find . -type f | wc -l) files, $(cd $S/var/$v && find . -type f | while read f; do [ -f "$RT/$f" ] && echo x; done | wc -l) also in the runtime root"; done
CUR=""; NC=0
use() { NC=$((NC+1)); (cd $S/var/$1 && find . -type f | while read f; do cp "$f" "$R/$f"; [ -f "$RT/$f" ] && cp "$f" "$RT/$f"; done); export TT_METAL_CACHE=$OUT/cache/$1_$NC; mkdir -p $TT_METAL_CACHE; CUR=$1; }
elfs() { python3 - "$TT_METAL_CACHE" <<'PYE'
import hashlib, os, struct, sys
def img(fn):
    b = open(fn, "rb").read(); h = hashlib.sha256(); shoff, = struct.unpack_from("<I", b, 0x20); se, sn = struct.unpack_from("<HH", b, 0x2E)
    for i in range(sn):
        _, typ, flags, addr, off, size = struct.unpack_from("<IIIIII", b, shoff + i * se)
        if flags & 2 and typ != 8 and size: h.update(struct.pack("<II", addr, size)); h.update(b[off:off + size])
    return h.hexdigest()[:16]
s = sorted({img(os.path.join(d, f)) for d, _, fs in os.walk(sys.argv[1]) for f in fs if f.startswith("trisc") and f.endswith(".elf")})
print(hashlib.sha256(" ".join(s).encode()).hexdigest()[:16], len(s))
PYE
}
source $S/job.sh
for v in $BITS_VARIANTS; do
  use ${v%2}; mkdir -p $OUT/bits_$v
  bits $OUT/bits_$v > $OUT/log/bits_$v.log 2>&1
  echo "== bits $v rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' $OUT/log/bits_$v.log | tail -1) files $(ls $OUT/bits_$v | wc -l) elfset $(elfs) $(date -u +%T)"
  grep -aE "^(FAILED|ERROR)" $OUT/log/bits_$v.log | head -5
done
for p in "${COMPARE[@]}"; do set -- $p; echo "== compare $1 against $2"; ${COMPARE_CMD:-python3 $S/tests/compare_r2_dumps.py} $OUT/bits_$1 $OUT/bits_$2 | grep -v ": identical;" | tail -40; done
k=0
for pr in "${PROF_PAIRS[@]}"; do
  set -- $pr; A=$1; B=$2
  for v in $A $B $B $A $A $B $B $A; do
    k=$((k+1)); use $v
    for c in $PROF_CASES; do
      O=$OUT/prof/${A}_${B}__${v}__r${k}__$c; mkdir -p $O
      TT_METAL_PROFILER_DIR=$O/pdir R3_NMEAS=6 timeout -s INT -k 60 900 python3 -m tracy -r -p --no-web-server -o $O $S/tools/$PROF_SCRIPT $c > $O.log 2>&1
      python3 - "$O" "$A" "$B" "$v" "$k" "$c" <<'PYEOF'
import csv, glob, statistics, sys
o, a, b, v, k, c = sys.argv[1:7]
fs = glob.glob(o + "/reports/*/ops_perf_results_*.csv")
if not fs:
    print("SFOP", a, b, k, v, c, "NOREPORT"); sys.exit(0)
rows = list(csv.DictReader(open(fs[0])))
idx = max([i for i, r in enumerate(rows) if r.get("OP TYPE") == "signpost"] or [-1])
meas = [r for r in rows[idx + 1:] if r.get("OP TYPE") != "signpost" and r.get("DEVICE KERNEL DURATION [ns]")]
ops = {}
for r in meas:
    ops.setdefault(r["OP CODE"], []).append(float(r["DEVICE KERNEL DURATION [ns]"]))
for op, xs in ops.items():
    print("SFOP", a, b, k, v, c, op.replace(" ", "_"), len(xs), statistics.median(xs), " ".join("%.0f" % x for x in xs))
X = ["DEVICE KERNEL DURATION PER CORE MIN [ns]", "DEVICE KERNEL DURATION PER CORE MAX [ns]", "DEVICE KERNEL DURATION PER CORE AVG [ns]",
     "DEVICE KERNEL FIRST TO LAST START [ns]", "DEVICE BRISC KERNEL DURATION [ns]", "DEVICE NCRISC KERNEL DURATION [ns]",
     "DEVICE TRISC0 KERNEL DURATION [ns]", "DEVICE TRISC1 KERNEL DURATION [ns]", "DEVICE TRISC2 KERNEL DURATION [ns]",
     "DEVICE COMPUTE CB WAIT FRONT [ns]", "DEVICE COMPUTE CB RESERVE BACK [ns]", "CORE COUNT"]
for op in ops:
    rs = [r for r in meas if r["OP CODE"] == op]
    out = []
    for col in X:
        xs = [float(r[col]) for r in rs if r.get(col) not in (None, "", "-")]
        out.append("%.0f" % statistics.median(xs) if xs else "na")
    print("SFOX", a, b, k, v, c, op.replace(" ", "_"), " ".join(out))
PYEOF
    done
    echo "   prof $A/$B round $k $v elfset $(elfs)"
  done
done
echo "== done $(date -u +%T)"
exit 0
