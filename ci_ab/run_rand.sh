#!/usr/bin/env bash
# rand_tile users at op level on one CI Blackhole runner (knowledge/process/17_CI_AB.md): the job's tree is the rand PR
# head merged with main; before each run a variant's device files are copied into /work and the installed runtime root,
# each run with its own fresh JIT cache. Variants: P (head, four calls per tile), O (one call per tile), Z (probe: no
# SFPU work, the op's data movement alone), P_aa (A/A copy of P). ABBA twice per pair; kernel ELF hashes per run.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; OUT=/tmp/sfo; rm -rf $OUT; mkdir -p $OUT/prof; cd $R
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== runner $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format=%h) runtime root $RT"
cpv() { (cd $1 && find . -type f | while read f; do cp "$f" "$R/$f"; [ -f "$RT/$f" ] && cp "$f" "$RT/$f"; done); }
for d in $S/var/*; do echo "== $(basename $d): $(cd $d && find . -type f | wc -l) files, $(cd $d && find . -type f | while read f; do [ -f "$RT/$f" ] && echo x; done | wc -l) in the runtime root"; done
NC=0
use() { NC=$((NC+1)); cpv $S/var/${1%_aa}; export TT_METAL_CACHE=$OUT/cache/${1}_$NC; mkdir -p $TT_METAL_CACHE; }
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
CASES="rand_bf16_4096x2048 rand_fp32_4096x2048 rand_bf16_1024x1024 rand_bf16_32x32 rand_bf16_8192x8192 uniform_fp32_4096x2048 uniform_bf16_4096x2048 bernoulli_fp32_4096x2048 sampling_32x256"
for v in P O; do use $v; R_BITS=$OUT/bits_$v.json timeout 900 python3 $S/prof_rand.py $CASES > $OUT/bits_$v.log 2>&1; echo "== bits $v rc=$? $(tail -1 $OUT/bits_$v.log) elfset $(elfs)"; done
python3 -c "import json; a=json.load(open('$OUT/bits_P.json')); b=json.load(open('$OUT/bits_O.json')); print('== bits P vs O:', {k: a[k]==b.get(k) for k in a})"
run() { local v=$1 k=$2; use $v; local O=$OUT/prof/r${k}_$v; mkdir -p $O
  TT_METAL_PROFILER_DIR=$O/pdir R3_NMEAS=6 timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $O $S/prof_rand.py $CASES > $O.log 2>&1
  echo "   run $k $v rc=$? $(grep -c 'CASE .* ok' $O.log) ok, elfset $(elfs) $(date -u +%T)"; grep -a "FAILED" $O.log | head -3
  python3 $S/parse_ops.py $O "r$k $v"; }
k=0
for pr in "P O" "P Z" "P P_aa"; do set -- $pr; for v in $1 $2 $2 $1 $1 $2 $2 $1; do k=$((k+1)); run $v $k; done; done
echo "== done $(date -u +%T)"
exit 0
