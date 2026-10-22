#!/usr/bin/env bash
# Scratch, not for merge. #59135 candidate 7ff78ce4209 (early poll in transpose_tile) at the ttnn level, one job:
# 1 functional ttnn transpose/permute/sort tests on the candidate, failures rerun on the PR head file;
# 2 op level (tracy) ABBA of arms H (PR head transpose.h) and C (candidate) with A/A copies H2 C2, fresh caches;
# 3 bits: output sha1 per arm and shape (BITS lines), ELF guard on the transpose kernels.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-45} AB_NODES_FILE=/work/ci_ab/nodes_tb90b.txt
S=/work/ci_ab; OUT=/work/generated/tb90b; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
export PYTHONPATH=$S:${PYTHONPATH:-}
echo "== $(git log -1 --format='%h %s' | cut -c1-80)"
HDR=tt_metal/hw/inc/api/compute/transpose.h
TARGETS="/work/$HDR"
for d in $(python3 -c "import ttnn,os;print(os.path.dirname(ttnn.__file__))" 2>/dev/null) /work/runtime /work/ttnn; do
  [[ -f $d/$HDR ]] && TARGETS="$TARGETS $d/$HDR"
  f=$(find $d -maxdepth 6 -path '*api/compute/transpose.h' 2>/dev/null | head -1); [[ -n $f ]] && TARGETS="$TARGETS $f"
done
TARGETS=$(printf '%s\n' $TARGETS | sort -u); echo "== header targets: $TARGETS"
setarm() { for t in $TARGETS; do cp $S/var/$1/transpose.h $t; done; echo "== arm $1 $(md5sum /work/$HDR | cut -c1-8)"; }
setarm C
for a in H C C2 H2 H2 C2 C H; do
  k=$((k+1)); case $a in H|H2) v=H;; *) v=C;; esac
  setarm $v
  o=$OUT/r${k}_$a; mkdir -p $o; export TT_METAL_CACHE=$OUT/cache_${k}_$a EB_ARM=$a EB_BITS_OUT=$OUT/bits.txt
  timeout -s INT -k 60 2400 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  echo "== run $k $a rc=$?: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  python3 $S/sum_prof.py $o $a $k | sed 's/^/OPDATA /'
  python3 $S/elfhash.py $a $TT_METAL_CACHE transpose_wh transpose_xw | grep -a '^ELF ' | grep -av 'reader\|writer' | sed 's/^/ELFSUM '$k' /' | tee -a $OUT/elf_all.txt
done
sort -u $OUT/bits.txt | sed 's/^/BITS /'
python3 - <<'PY'
import collections
d=collections.defaultdict(set)
for l in open("/work/generated/tb90b/elf_all.txt"):
    p=l.split()
    # ELFSUM k ELF arm name key procs...
    k,arm,name,key=p[1],p[3],p[4],p[5]
    d[(arm,name)].add(" ".join(sorted(p[6:])))
names=sorted(set(n for _,n in d))
ok=True
for n in names:
    H=d[("H",n)]|d[("H2",n)]; C=d[("C",n)]|d[("C2",n)]
    same_aa=(d[("H",n)]==d[("H2",n)]) and (d[("C",n)]==d[("C2",n)])
    diff=H!=C
    print("ELFGUARD",n,"H==H2/C==C2" if same_aa else "AA-MISMATCH","H!=C" if diff else "H==C(VOID)")
    ok&=same_aa and diff
print("ELFGUARD", "ok" if ok and names else "VOID")
PY
echo "== done"
