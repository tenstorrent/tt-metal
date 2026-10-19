#!/usr/bin/env bash
# Scratch A/B, not for merge. Device files ci_ab/v/<M|P|S>/ (files_dev.txt) into /work and the package, own JIT cache per arm.
# M main; P the PR head b0c77f9a2fb (EB_NOSCALAR); N the branch with the scalar block off (EB_NOSCALAR); S the branch; S2 A/A.
# Timing M S P S2 | S2 P S M under the device profiler (sum_op19.py), N once; bits per arm; ELF hashes of every kernel per arm.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-4} AB_NODES_FILE=/work/ci_ab/nodes.txt
S=/work/ci_ab; OUT=/work/generated/sc; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
export PYTHONPATH=$S:${PYTHONPATH:-}
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK
while read -r rel; do PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null); done < $S/files_dev.txt
echo "== package copies: $(for r in "${!PK[@]}"; do echo ${PK[$r]} | wc -w; done | awk '{s+=$1} END {print s}') for $(wc -l < $S/files_dev.txt) files"
use() {  # use <arm>
  local d; case $1 in M) d=M ;; P) d=P ;; *) d=S ;; esac
  while read -r rel; do cp "$S/v/$d/$rel" "/work/$rel"; for p in ${PK[$rel]}; do cp "$S/v/$d/$rel" "$p"; done; done < $S/files_dev.txt
  if [[ $1 == P || $1 == N ]]; then export EB_NOSCALAR=1; else unset EB_NOSCALAR; fi
  export TT_METAL_CACHE=$OUT/cache_$1 EB_ARM=$1 EB_BITS_OUT=$OUT/bits.txt; mkdir -p $TT_METAL_CACHE
}
r=0
for a in M S P S2 N S2 P S M; do
  r=$((r+1)); use $a
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 2400 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  echo "== run $r $a rc=$? $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  grep -a -E '^(FAILED|ERROR)' $o.log | sort -u | head -6 | cut -c1-250
  python3 $S/sum_op19.py $o $a $r
done
for a in M P N S S2; do ELFKEY_DROP=NOTHING python3 $S/elfhash.py $a $OUT/cache_$a > $OUT/elf_$a.txt 2>&1; tail -1 $OUT/elf_$a.txt; done
python3 - <<'PY'
import collections
def load(a):
    d = collections.defaultdict(dict)
    for l in open(f"/work/generated/sc/elf_{a}.txt"):
        p = l.split()
        if p and p[0] == "ELF":
            d[p[2]][p[3]] = " ".join(p[4:])
    return d
A = {a: load(a) for a in ("M", "P", "N", "S", "S2")}
for x, y in (("N", "P"), ("S", "N"), ("S", "S2"), ("S", "M")):
    tp = td = 0
    for name in sorted(set(A[x]) | set(A[y])):
        ks = set(A[x].get(name, {})) & set(A[y].get(name, {}))
        diff = [k for k in ks if A[x][name][k] != A[y][name][k]]
        tp += len(ks); td += len(diff)
        if diff:
            print(f"ELFCMP {x} {y} {name} paired {len(ks)} differ {len(diff)}")
    print(f"ELFSUM {x} {y} paired {tp} differ {td}")
PY
sort -u $OUT/bits.txt | sed 's/^/BITS /'
use M
echo "== done"
