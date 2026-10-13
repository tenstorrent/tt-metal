#!/usr/bin/env bash
# #60138 sanity follow-up: the ring MLA accuracy test with the head's MIN_BLOCKED_PACK_TILES (2) against main's (4), alternated
# on one QB2 runner, each side with its own fresh kernel cache; the job fails if the two sides build the same SDPA ELFs.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); cd $R
F=ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
V=$R/generated/pcc; mkdir -p $V
cp $R/$F $V/head.hpp
sed 's/constexpr uint32_t MIN_BLOCKED_PACK_TILES = 2;/constexpr uint32_t MIN_BLOCKED_PACK_TILES = 4;/' $V/head.hpp > $V/main4.hpp
if cmp -s $V/head.hpp $V/main4.hpp; then echo "== VOID: variant files identical"; exit 1; fi
md5sum $V/head.hpp $V/main4.hpp
OBJ=$(find /opt /usr /root /home "$R" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); echo "== objcopy $OBJ"
place() { cp "$1" "$R/$F"; for p in $(find "$SP" -path "*/$F" -type f 2>/dev/null); do cp "$1" "$p"; echo "   placed in $p"; done; }
elfsum() { for e in $(ls $1/*/kernels/*sdpa*/*/trisc[012]/trisc[012].elf 2>/dev/null | sort); do $OBJ -O binary -j .text "$e" /tmp/elf.bin 2>/dev/null && md5sum < /tmp/elf.bin; done | md5sum | cut -c1-12; }
NODE="tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_balanced_rotated_q_pairs_accuracy_and_cache_reuse"
i=0
for v in head main4 main4 head; do
  i=$((i+1)); place $V/$v.hpp
  export TT_METAL_CACHE=$HOME/pcc_cache_$v
  if [[ " ${FRESHED:-} " != *" $v "* ]]; then rm -rf $TT_METAL_CACHE; FRESHED="${FRESHED:-} $v"; fi
  mkdir -p $TT_METAL_CACHE
  echo "== run $i $v $(date -u +%T)"
  timeout -s INT -k 60 1500 python -m pytest -p no:cacheprovider -q -rfE "$NODE" > $V/log_${i}_$v.txt 2>&1
  echo "== run $i $v rc=$? $(grep -E 'passed|failed' $V/log_${i}_$v.txt | tail -1)"
  grep -E "^(FAILED|ERROR) |PCC: " $V/log_${i}_$v.txt | sed -E 's/.*(test_ring_joint_balanced[^ ]*).*/\1/' | cut -c1-200 | head -40
done
place $V/head.hpp
H=$(elfsum $HOME/pcc_cache_head); M=$(elfsum $HOME/pcc_cache_main4); echo "== sdpa compute ELFs: head $H main4 $M"
[[ -n $H && $H != "$M" ]] || { echo "== VOID: the two sides built the same SDPA ELFs"; exit 1; }
