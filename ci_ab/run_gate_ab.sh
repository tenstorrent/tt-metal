#!/usr/bin/env bash
# Scratch A/B job (split plan N2, the prefill gate), not for merge. The gate and hash gate at the model configurations
# under the device profiler with main's kernels and main's program (M: main's three kernel files, CI_AB_MAIN_FACTORY=1)
# against the branch (N) and an A/A copy of N (N2), each arm its own fresh JIT cache, its files swapped into the
# checkout and the installed package, M N N2 N2 N M in two blocks. Then the bits of the 20 gate_bitid cases per arm and
# the mock ELF builds (Wormhole N150, Blackhole p100a) of M and N. ELF guard: the gate kernels of M and N must differ
# on Blackhole, N and N2 must match.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1
S=/work/ci_ab; OUT=/work/generated/ab; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
while read -r rel; do echo "== $rel -> $(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/files.txt
use() {  # use <arm> <files> [cache]: the files into the checkout and every copy under the package root, the arm's cache
  while read -r rel; do
    cp "$S/v/$2/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/v/$2/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "RAB FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/${3:-cache_$1}; mkdir -p $TT_METAL_CACHE
  if [[ $2 == M ]]; then export CI_AB_MAIN_FACTORY=1; else unset CI_AB_MAIN_FACTORY; fi
}
OBJ=/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy; [[ -x $OBJ ]] || OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
elfsum() {  # .text md5 over every gate kernel ELF of a cache
  for e in $(ls $1/*/kernels/{moe_grouped_topk,writer_moe_grouped_topk,moe_hash_gate}/*/*/*.elf 2>/dev/null | sort); do
    $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin; done | md5sum | cut -c1-12; }
declare -A ESUM
r=0; fail=0
for blk in 1 2; do for a in M N N2 N2 N M; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/gate_ops.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c ' ok$' $o.log) ok, $(grep -c FAILED $o.log) failed"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'FAILED|Error' $o.log | cut -c1-300; }
  python3 $S/sum_op.py $o $a $r
  if [[ $r -le 3 ]]; then ESUM[$a]=$(elfsum $TT_METAL_CACHE); echo "RAB ELF $a ${ESUM[$a]} $(ls -d $TT_METAL_CACHE/*/kernels/*/* 2>/dev/null | wc -l) builds"; fi
done; done
g1=$([[ ${ESUM[M]} != "${ESUM[N]}" ]] && echo ok || echo VOID); g2=$([[ ${ESUM[N]} == "${ESUM[N2]}" ]] && echo ok || echo DIFF)
echo "RAB GUARD M_vs_N=$g1 N_vs_N2=$g2"; [[ $g1 == ok && $g2 == ok ]] || fail=1
for a in M N; do  # bits: the 20 gate_bitid cases, each arm on its own timing cache
  use $a $a; o=$OUT/bits_$a; timeout -s INT -k 60 900 python3 $S/gate_bitid.py $o > $o.log 2>&1
  echo "== bits $a rc=$? $(wc -l < $o/sha256.txt 2>/dev/null) outputs"
  sed "s/^/BITS $a /" $o/sha256.txt 2>/dev/null
done
for arch in wh bh; do for a in M N; do  # mock ELF builds, fresh caches
  desc=$([[ $arch == wh ]] && echo /work/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml || echo $S/p100a_cluster.yaml)
  [[ -f $desc ]] || { echo "== no mock descriptor $desc"; fail=1; continue; }
  use $a $a mock_${arch}_$a
  GATE_MOCK=1 TT_METAL_MOCK_CLUSTER_DESC_PATH=$desc timeout -s INT -k 60 1200 python3 $S/gate_ops.py > $OUT/mock_${arch}_$a.log 2>&1
  echo "== mock $arch $a rc=$? $(grep -c ' ok$' $OUT/mock_${arch}_$a.log) ok, $(grep -c FAILED $OUT/mock_${arch}_$a.log) failed"
  grep -a -m3 FAILED $OUT/mock_${arch}_$a.log | cut -c1-300
  python3 $S/elfhash.py ${arch}_$a $TT_METAL_CACHE moe_grouped_topk moe_hash_gate
done; done
use M M cache_M
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
