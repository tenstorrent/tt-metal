#!/usr/bin/env bash
# ttnn bits and op-level A/B on one CI Blackhole runner (knowledge/process/17_CI_AB.md). The job's tree is the stack top
# (variant base); before each run a variant's device files (ci_ab/variants/<v>) are copied into /work and into the
# installed runtime root, with its own fresh JIT cache. Bits: bits_oc_int.py (int add, sub, rsub) and exh_dump.py
# (add, subtract, multiply, rsub, xlogy, nextafter) per variant (main, base, int, bin). Op level: every case in its own
# tracy process, variants alternated ABBA twice per lever. Kernel ELF hashes per variant show the swap reaches the
# compiler. Results printed as one base64 tarball.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; OUT=$R/generated/sbt; rm -rf $OUT; mkdir -p $OUT/prof $OUT/log; cd $R
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci
echo "== runner $(hostname) $(date -u +%FT%TZ)"
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== runtime root $RT; $(cd $S/variants/base && find . -type f | while read f; do [ -f "$RT/$f" ] && echo x; done | wc -l) of $(wc -l < $S/paths.txt) paths also there"
CUR=""
use() { [[ $CUR == $1 ]] && return; (cd $S/variants/$1 && find . -type f | while read f; do cp "$f" "$R/$f"; [ -f "$RT/$f" ] && cp "$f" "$RT/$f"; done); export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE; CUR=$1; }
for v in main base int bin; do
  use $v
  mkdir -p $OUT/bits_$v
  R3_BITS_OUT=$OUT/bits_$v timeout -s INT -k 60 1200 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread $S/tools/bits_oc_int.py -k "ib_add or ib_sub or ib_rsub" > $OUT/log/bits_$v.log 2>&1
  echo "== bits $v rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' $OUT/log/bits_$v.log | tail -1) $(ls $OUT/bits_$v | wc -l) files"
  timeout -s INT -k 60 1800 python3 $S/tools/exh_dump.py $OUT/exh_$v.json add,subtract,multiply,rsub,xlogy,nextafter > $OUT/log/exh_$v.log 2>&1
  echo "== exh $v rc=$? $(tail -1 $OUT/log/exh_$v.log)"
  python3 - $OUT/bits_$v $v <<'PYB'
import glob, hashlib, os, sys
import numpy as np
d, v = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(d + "/*.npz")):
    z = np.load(f); h = hashlib.sha256()
    for k in sorted(z.files):
        h.update(k.encode()); h.update(np.ascontiguousarray(z[k]).tobytes())
    print("SBTB", v, os.path.basename(f)[:-4], h.hexdigest()[:20], sum(z[k].size for k in z.files))
PYB
  python3 -c "import json,sys; d=json.load(open('$OUT/exh_$v.json')); [print('SBTE', '$v', k, str(x).replace(' ','_')) for k, x in sorted(d.items()) if k not in ('other',)]" 2>/dev/null
done
PI="add_i32_hs add_i32_dram_1024x1024 sub_i32_hs rsub_i32_hs add_i32_penalties_32x128256"
PB="add_f32_hs add_f32_dram_1024x1024 sub_f32_hs rsub_f32_hs xlogy_bf16_hs xlogy_bf16_dram_1024x1024 mul_bf16_hs"
k=0
for lev in int bin; do
  C=$PI; [[ $lev == bin ]] && C=$PB
  for v in base $lev $lev base base $lev $lev base; do
    k=$((k+1)); use $v
    for c in $C; do
      O=$OUT/prof/${lev}__${v}__r${k}__$c; mkdir -p $O
      TT_METAL_PROFILER_DIR=$O/pdir R3_NMEAS=6 timeout -s INT -k 60 600 python3 -m tracy -r -p --no-web-server -o $O $S/tools/prof_bin.py $c > $O.log 2>&1
      python3 - "$O" "$lev" "$v" "$k" "$c" <<'PYEOF'
import csv, glob, statistics, sys
o, lev, v, k, c = sys.argv[1:6]
fs = glob.glob(o + "/reports/*/ops_perf_results_*.csv")
if not fs:
    print("SBT", lev, k, v, c, "NOREPORT"); sys.exit(0)
rows = [r for r in csv.DictReader(open(fs[0])) if r.get("OP TYPE") != "signpost" and r.get("DEVICE KERNEL DURATION [ns]")]
op = rows[-1]["OP CODE"]; xs = [float(r["DEVICE KERNEL DURATION [ns]"]) for r in rows if r["OP CODE"] == op][-6:]
print("SBT", lev, k, v, c, op.replace(" ", "_"), len(xs), statistics.median(xs) if xs else "nan", " ".join("%.0f" % x for x in xs))
PYEOF
    done
  done
done
for v in main base int bin; do
  find $OUT/cache_$v -name '*.elf' 2>/dev/null | while read e; do echo "SBTK $v ${e#$OUT/cache_$v/} $(sha256sum < $e | cut -c1-16)"; done | sort > $OUT/elf_$v.txt
  echo "== $v kernels $(wc -l < $OUT/elf_$v.txt)"
done
for v in main int bin; do echo "SBTKD $v $(join -j1 <(awk '{print $3, $4}' $OUT/elf_base.txt | sort) <(awk '{print $3, $4}' $OUT/elf_$v.txt | sort) | awk '$2!=$3' | wc -l) kernels differ from base of $(wc -l < $OUT/elf_base.txt)"; join -j1 <(awk '{print $3, $4}' $OUT/elf_base.txt | sort) <(awk '{print $3, $4}' $OUT/elf_$v.txt | sort) | awk '$2!=$3{print "SBTKN", v, $1}' v=$v | sed 's|/[0-9a-f]\{20,\}/| |' | sort -u | head -40; done
use base
exit 0
