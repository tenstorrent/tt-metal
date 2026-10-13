#!/usr/bin/env bash
# Scratch op-level A/B of the welford table-form read-ahead on the #60071 head merged with main: H (head), G (the lever), H2 (A/A).
set +e +o pipefail
cd /work
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
export HWLOCK_HELD=ci R3_NMEAS=6
CASES="ditpre_bf16_512x768 ditpre_bf16_512x1280 ditpre_bf16_4096x768 ditpre_bf16_4096x1280 ditpre_fp32_512x768 ditpre_fp32_4096x1280 lnpre_bf16_32x1024 lnpre_bf16_256x2048 lnpre_bf16_1024x4096 lnpre_fp32_32x1024 lnpre_fp32_256x2048 lnpre_fp32_1024x4096 lnpre_res_bf16_32x1024 lnpre_res_bf16_256x2048 lnpre_res_bf16_1024x4096 lnpre_res_fp32_32x1024 lnpre_res_fp32_256x2048 lnpre_res_fp32_1024x4096 lnpre_res_bf16nf_32x1024 lnpre_res_bf16nf_256x2048 lnpre_res_bf16nf_1024x4096"
FILES=$(cd ci_ab/v/H && find . -type f | sed 's#^\./##')
RR=$(python3 - <<'PY'
import glob, os, site
k = "tt_metal/tt-llk/common/ckernel_sfpu_welfords_common.h"
for sp in site.getsitepackages():
    for p in glob.glob(os.path.join(sp, "**", k), recursive=True):
        print(p[: -len(k)])
PY
)
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s'); runtime roots: ${RR:-none}"
for f in $FILES; do for d in $RR; do [[ -f $d$f ]] && echo "runtime root has $f"; done; done
mkdir -p /tmp/base && for f in $FILES; do mkdir -p /tmp/base/$(dirname $f); git show HEAD:$f > /tmp/base/$f; done
apply() { # apply <variant>: base files, then the variant's own, into /work and every runtime root that has the file
  local f src d
  for f in $FILES; do
    src=/tmp/base/$f; [[ -f ci_ab/v/$1/$f ]] && src=ci_ab/v/$1/$f
    cp $src /work/$f
    for d in $RR; do [[ -f $d$f ]] && cp $src $d$f; done
  done
  echo "apply $1: $(for f in $FILES; do md5sum < /work/$f | cut -c1-6; done | tr '\n' ' ')"
}
rm -rf /tmp/ttc_* /work/generated/ol; mkdir -p /work/generated/ol
n=0
for order in "H G H2" "H2 G H" "G H H2" "H2 H G"; do
  for v in $order; do
    n=$((n+1)); vv=${v%2}; apply $vv
    out=/work/generated/ol/${v}_$n; t0=$(date +%s)
    TT_METAL_CACHE=/tmp/ttc_$v timeout 1500 python3 -m tracy -r -p --no-web-server -o $out ci_ab/prof_stateful.py $CASES > $out.log 2>&1
    echo "== run $n $v rc=$? $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $out.log) ok, $(grep -c 'CASE .* FAILED' $out.log) failed"
    grep 'CASE .* FAILED' $out.log | head -3
    python3 - "$out" "$v" "$n" <<'PY'
import csv, glob, sys
out, v, n = sys.argv[1:4]
fs = glob.glob(out + "/reports/*/ops_perf_results_*.csv")
if not fs:
    print(f"OPL {v} {n} NO_REPORT"); sys.exit()
case = None
for r in csv.DictReader(open(fs[0])):
    if r.get("OP TYPE") == "signpost":
        case = r["OP CODE"]; continue
    d = r.get("DEVICE KERNEL DURATION [ns]")
    if case and d:
        print(f"OPL {v} {n} {case} {r['OP CODE']} {d}")
PY
  done
  if [[ $n -eq 4 ]]; then for v in H G H2; do echo "ELF $v $(find /tmp/ttc_$v -path '*layernorm_pre_allgather_welford*' -name '*.elf' | xargs -r md5sum | awk '{print $1}' | sort | md5sum | cut -c1-12) $(find /tmp/ttc_$v -path '*layernorm_pre_allgather_welford*' -name '*.elf' | wc -l)"; done; fi
done
apply H
