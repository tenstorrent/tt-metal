# Scratch, not for merge. Shared setup of the untilize A/B jobs (sourced by run_perf.sh and run_bits.sh).
# Arms (ci_ab/v/<arm>/, files in ci_ab/files.txt, copied into the checkout and the installed package before every run, fresh JIT cache per arm):
#   A main (merge-base 129d0c50379)   B PR head fd36d27b42b   C the fixed commit   C2 A/A copy of C
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-8} AB_DIR=/work/ci_ab
S=/work/ci_ab
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP $(hostname)"
OBJ=/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy; [[ -x $OBJ ]] || OBJ=$(which riscv-tt-elf-objcopy 2>/dev/null)
[[ -x $OBJ ]] || OBJ=$(find /work/runtime /opt -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK PD
while read -r rel; do
  PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  [[ -z ${PK[$rel]} ]] && { s=$(dirname "$rel" | awk -F/ '{n=NF; print $(n-2)"/"$(n-1)"/"$n}'); PD[$rel]=$(find "$SP" -type d -path "*/$s" 2>/dev/null); }
  echo "RAB FILE $rel package copies: $(echo ${PK[$rel]} | wc -w) dirs: $(echo ${PD[$rel]:-} | wc -w)"
done < $S/files.txt
use() {  # use <cache tag> <files dir>
  while read -r rel; do
    [[ -f "$S/v/$2/$rel" ]] || continue
    cp "$S/v/$2/$rel" "/work/$rel"
    for p in ${PK[$rel]}; do cp "$S/v/$2/$rel" "$p"; done
    for d in ${PD[$rel]:-}; do cp "$S/v/$2/$rel" "$d/"; done
  done < $S/files.txt
  echo "RAB FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() {  # .text hash of every trisc ELF of a cache, order independent
  for e in $(ls $1/*/kernels/*/*/trisc*/*.elf 2>/dev/null); do
    $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12
  done | sort | md5sum | cut -c1-12
}
