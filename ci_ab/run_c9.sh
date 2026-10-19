#!/usr/bin/env bash
# Scratch, not for merge. Step 4 callers A/B. Device files ci_ab/v/<M|P|N|S>/ (files_dev.txt) into /work and the package,
# own JIT cache per arm. M main 129d0c50379, P the PR head 148493818c7, N the branch with its kernel opt-ins off, S the
# branch, S2 A/A of S. Timing M S P S2 N | N S2 P S M under the device profiler (sum_op19.py) on $C9_NODES; output sha1 per
# arm; ELF hashes of every kernel per arm (N against P must match, S against N differs only in the opted kernels); then the
# callers' ttnn tests ($C9_FUNC, pytest args) on S, and the failures again on P.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-4} AB_NODES_FILE=/work/ci_ab/${C9_NODES:-nodes.txt}
S=/work/ci_ab; OUT=/work/generated/c9; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
export PYTHONPATH=$S:${PYTHONPATH:-}
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(git log -1 --format='%h %s' | cut -c1-80) nodes $AB_NODES_FILE ($(wc -l < $AB_NODES_FILE))"
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK
while read -r rel; do PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null); done < $S/files_dev.txt
echo "== package copies: $(for r in "${!PK[@]}"; do echo ${PK[$r]} | wc -w; done | awk '{s+=$1} END {print s}') for $(wc -l < $S/files_dev.txt) files"
use() { local d=${1%2}
  while read -r rel; do
    if [[ -f $S/v/$d/$rel ]]; then cp "$S/v/$d/$rel" "/work/$rel"; for p in ${PK[$rel]}; do cp "$S/v/$d/$rel" "$p"; done
    else rm -f "/work/$rel"; for p in ${PK[$rel]}; do rm -f "$p"; done; fi
  done < $S/files_dev.txt
  echo "== use $1: $(while read -r rel; do md5sum < /work/$rel 2>/dev/null | cut -c1-6; done < $S/files_dev.txt | md5sum | cut -c1-12)"
  export TT_METAL_CACHE=$OUT/cache_$1 EB_ARM=$1 EB_BITS_OUT=$OUT/bits.txt; mkdir -p $TT_METAL_CACHE; }
r=0; T0=$(date +%s)
for a in ${C9_ARMS-M S P S2 N N S2 P S M}; do
  r=$((r+1)); use $a; o=$OUT/${a}_r$r; mkdir -p $o
  timeout -s INT -k 60 ${C9_RUN_TIMEOUT:-1500} python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  echo "== run $r $a rc=$? $(( $(date +%s) - T0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  grep -a -E '^(FAILED|ERROR)' $o.log | sort -u | head -6 | cut -c1-250
  python3 $S/sum_op19.py $o $a $r
done
ARMS=$(echo ${C9_ARMS-M S P S2 N N S2 P S M} | tr ' ' '\n' | sort -u | tr '\n' ' ')
for a in $ARMS; do ELFKEY_DROP=NOTHING python3 $S/elfhash.py $a $OUT/cache_$a > $OUT/elf_$a.txt 2>&1; tail -1 $OUT/elf_$a.txt; done
cat $OUT/elf_*.txt | grep '^ELF ' > $OUT/elf_all.txt
C9_ELF_PAIRS=${C9_ELF_PAIRS-N:P S:N S:S2 S:M} python3 - <<'PY'
import collections, os
def load(a):
    d = collections.defaultdict(dict)
    for l in open(f"/work/generated/c9/elf_{a}.txt"):
        p = l.split()
        if p and p[0] == "ELF":
            d[p[2]][p[3]] = " ".join(p[4:])
    return d
pairs = [tuple(x.split(":")) for x in os.environ["C9_ELF_PAIRS"].split()]
A = {a: load(a) for a in {a for pr in pairs for a in pr}}
for x, y in pairs:
    tp = td = 0
    for name in sorted(set(A[x]) | set(A[y])):
        ks = set(A[x].get(name, {})) & set(A[y].get(name, {}))
        diff = [k for k in ks if A[x][name][k] != A[y][name][k]]
        tp += len(ks); td += len(diff)
        if diff and (x, y) != ("S", "M"):
            print(f"ELFCMP {x} {y} {name} paired {len(ks)} differ {len(diff)}")
    print(f"ELFSUM {x} {y} paired {tp} differ {td}")
PY
sort -u $OUT/bits.txt | sed 's/^/BITS /'
if [[ -n ${C9_FUNC:-} ]]; then
  mapfile -t FA < $S/$C9_FUNC
  use S; unset TT_METAL_DEVICE_PROFILER
  timeout -s INT -k 60 ${C9_FUNC_TIMEOUT:-1800} python3 -m pytest -q -rfE -p no:cacheprovider -o timeout_method=thread "${FA[@]}" > $OUT/func_S.log 2>&1
  echo "== func S rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' $OUT/func_S.log | tail -1 | cut -c1-200)"
  grep -a -E '^(FAILED|ERROR) ' $OUT/func_S.log | cut -d' ' -f2 | sort -u > $OUT/func_fail.txt
  head -20 $OUT/func_fail.txt | sed 's/^/FUNCFAIL S /'
  if [[ -s $OUT/func_fail.txt ]]; then
    mapfile -t FF < <(head -40 $OUT/func_fail.txt)
    use P
    timeout -s INT -k 60 900 python3 -m pytest -q -rfE -p no:cacheprovider -o timeout_method=thread "${FF[@]}" > $OUT/func_P.log 2>&1
    echo "== func P (S failures) rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' $OUT/func_P.log | tail -1 | cut -c1-200)"
    grep -a -E '^(FAILED|ERROR) ' $OUT/func_P.log | cut -d' ' -f2 | sort -u | head -20 | sed 's/^/FUNCFAIL P /'
  fi
fi
use M
echo "== done $(( $(date +%s) - T0 )) s"
