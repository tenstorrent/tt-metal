#!/usr/bin/env bash
# Scratch, not for merge. Arms: M main's device files (now with ttnn/cpp/ttnn/kernel_lib), W the branch, X the branch without
# the PER_TILE_HANDOFF_GAINS define (EB_NODEFINE). One pytest pass per arm over test_bng_w.py's gate tests and test_kernels.py,
# failures printed with their error lines; then every compute kernel's ELF code hash per arm (key without the define line),
# compared W against X and W against M.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work; S=/work/ci_ab; OUT=/work/generated/elf; rm -rf $OUT; mkdir -p $OUT
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK
while read -r rel; do PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null); done < $S/files_dev.txt
echo "== package copies: $(for r in "${!PK[@]}"; do echo ${PK[$r]} | wc -w; done | awk '{s+=$1} END {print s}') for $(wc -l < $S/files_dev.txt) files"
for a in M W X; do
  d=$([[ $a == M ]] && echo M || echo B)
  while read -r rel; do cp "$S/v/$d/$rel" "/work/$rel"; for p in ${PK[$rel]}; do cp "$S/v/$d/$rel" "$p"; done; done < $S/files_dev.txt
  if [[ $a == X ]]; then export EB_NODEFINE=1; else unset EB_NODEFINE; fi
  export TT_METAL_CACHE=$OUT/cache_$a; mkdir -p $TT_METAL_CACHE
  timeout -s INT -k 60 2400 python3 -m pytest -p no:cacheprovider -o timeout_method=thread -o addopts= -q -rfE \
    "ci_ab/test_bng_w.py::test_gate" ci_ab/test_kernels.py > $OUT/log_$a.txt 2>&1
  echo "== arm $a rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/log_$a.txt | tail -1)"
  grep -a -E "^(FAILED|ERROR) " $OUT/log_$a.txt | cut -c1-200 | head -12
  grep -a -E "error:|Error|TT_FATAL|TT_THROW" $OUT/log_$a.txt | sort | uniq -c | sort -rn | head -12 | cut -c1-400
  ELFKEY_DROP=PER_TILE_HANDOFF_GAINS python3 $S/elfhash.py $a $TT_METAL_CACHE > $OUT/elf_$a.txt 2>&1; tail -1 $OUT/elf_$a.txt
done
python3 - <<'PY'
import collections
def load(a):
    d = collections.defaultdict(dict)
    for l in open(f"/work/generated/elf/elf_{a}.txt"):
        p = l.split()
        if p and p[0] == "ELF":
            d[p[2]][p[3]] = " ".join(p[4:])
    return d
M, W, X = load("M"), load("W"), load("X")
for x, y, nx, ny in ((W, X, "W", "X"), (W, M, "W", "M")):
    for name in sorted(set(x) | set(y)):
        ks = set(x.get(name, {})) & set(y.get(name, {}))
        diff = [k for k in ks if x[name][k] != y[name][k]]
        print(f"ELFCMP {nx} {ny} {name} paired {len(ks)} differ {len(diff)} only_{nx} {len(set(x.get(name, {})) - ks)} only_{ny} {len(set(y.get(name, {})) - ks)}")
        for k in diff[:2]:
            a, b = dict(t.split("=") for t in x[name][k].split()), dict(t.split("=") for t in y[name][k].split())
            print(f"ELFDIFF {nx} {ny} {name} {k} " + " ".join(p for p in a if a[p] != b.get(p)))
PY
echo "== done"
