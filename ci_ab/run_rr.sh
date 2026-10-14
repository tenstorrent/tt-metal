#!/usr/bin/env bash
# reshuffle_rows op level on one CI Blackhole runner (knowledge/process/17_CI_AB.md): the job's tree is the PR head
# merged with main; before each run a variant's device files are copied into /work and the installed runtime root,
# each run with its own fresh JIT cache. Variants: P (head), L (replay per row), H3/H4 (replay from 3 or 4 live rows,
# the direct form below), P_aa (A/A copy of P), Pi/Li (P and L with the timing-instrumented embedding kernels).
# Each run is one tracy process over every case; ABBA twice per pair. Kernel ELF hashes per run prove the swap.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; OUT=/tmp/sfo; rm -rf $OUT; mkdir -p $OUT/prof; cd $R
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== runner $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format=%h) runtime root $RT"
cpv() { (cd $1 && find . -type f | while read f; do cp "$f" "$R/$f"; [ -f "$RT/$f" ] && cp "$f" "$RT/$f"; done); }
for d in $S/var/* $S/inst $S/plain; do echo "== $(basename $d): $(cd $d && find . -type f | wc -l) files, $(cd $d && find . -type f | while read f; do [ -f "$RT/$f" ] && echo x; done | wc -l) in the runtime root"; done
NC=0
use() { NC=$((NC+1)); local v=$1 b=${1%_aa}; b=${b%i}; cpv $S/var/$b; if [[ $v == *i ]]; then cpv $S/inst; else cpv $S/plain; fi
  export TT_METAL_CACHE=$OUT/cache/${v}_$NC; mkdir -p $TT_METAL_CACHE; }
elfs() { python3 - "$TT_METAL_CACHE" <<'PYE'
import hashlib, os, struct, sys
def img(fn):
    b = open(fn, "rb").read(); h = hashlib.sha256(); shoff, = struct.unpack_from("<I", b, 0x20); se, sn = struct.unpack_from("<HH", b, 0x2E)
    for i in range(sn):
        _, typ, flags, addr, off, size = struct.unpack_from("<IIIIII", b, shoff + i * se)
        if flags & 2 and typ != 8 and size: h.update(struct.pack("<II", addr, size)); h.update(b[off:off + size])
    return h.hexdigest()[:16]
s = sorted({img(os.path.join(d, f)) for d, _, fs in os.walk(sys.argv[1]) for f in fs if f.startswith("trisc1") and f.endswith(".elf")})
print(hashlib.sha256(" ".join(s).encode()).hexdigest()[:16], len(s))
PYE
}
CASES="emb_2x64x160x96 emb_2x1024x4096x3200 emb_1x2048x2048x32000 emb_8x128x768x30522 k1_8x128x768x30522 k2_8x128x768x30522 k3_8x128x768x30522 k4_8x128x768x30522 k8_8x128x768x30522"
run() { # variant round cases [extra tracy args]
  local v=$1 k=$2 cs=$3; shift 3; use $v
  local O=$OUT/prof/r${k}_$v; mkdir -p $O
  TT_METAL_PROFILER_DIR=$O/pdir R3_NMEAS=${NMEAS:-6} timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server "$@" -o $O $S/tools/prof_rr2.py $cs > $O.log 2>&1
  echo "   run $k $v rc=$? $(grep -c 'CASE .* ok' $O.log) ok, elfset $(elfs) $(date -u +%T)"; grep -a "FAILED\|Error\|error:" $O.log | head -3
  python3 $S/tools/parse_rr.py $O "r$k $v"
}
for v in P H3 H4; do use $v; mkdir -p $OUT/bits_$v
  python3 $S/tools/exh_rr.py $OUT/bits_$v/all.json > $OUT/bits_$v.log 2>&1
  echo "== bits $v rc=$? $(tail -1 $OUT/bits_$v.log) elfset $(elfs) $(date -u +%T)"; grep -a "Error\|error:" $OUT/bits_$v.log | head -3
done
for v in H3 H4; do echo "== compare P against $v"; python3 $S/tools/cmp_json.py $OUT/bits_P $OUT/bits_$v; done
k=0
for pr in "P L" "P H3" "P H4" "P P_aa"; do
  set -- $pr
  for v in $1 $2 $2 $1 $1 $2 $2 $1; do k=$((k+1)); run $v $k "$CASES"; done
done
IC="k1_8x128x768x30522 k4_8x128x768x30522 emb_8x128x768x30522 emb_2x64x160x96"
for v in Pi Li Li Pi; do k=$((k+1)); run $v $k "$IC"; done
for v in P L H3; do k=$((k+1)); NMEAS=2 run $v $k "$IC" --profiler-capture-perf-counters instrn; done
echo "== done $(date -u +%T)"
exit 0
