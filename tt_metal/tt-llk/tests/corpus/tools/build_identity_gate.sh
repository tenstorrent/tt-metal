#!/usr/bin/env bash
# laneMK — compile each op's sem+hand certified kernel (pin-59) and run the object-identity
# gate: extract math.elf .text sha256 per leg, assert sem != hand (cross-binary), record.
# Emits a manifest TSV (op, sem_node, hand_node, sem_text, hand_text, status). Parameterized;
# no hard-coded personal paths beyond the required --farm/--out args.
set -uo pipefail
FARM="${FARM:?set FARM=<tests/python_tests>}"
OUT="${OUT:?set OUT=<evidence dir>}"
MANIFEST="${MANIFEST:?set MANIFEST=<op<TAB>sem_node<TAB>hand_node per line>}"
OBJCOPY="${OBJCOPY:?set OBJCOPY=<riscv-tt-elf-objcopy>}"
VENV="${VENV:?set VENV=<python>}"
LLK_HOME_="${LLK_HOME:?set LLK_HOME}"
mkdir -p "$OUT"
GATE="$OUT/IDENTITY-GATE.tsv"
echo -e "op\tstatus\tsem_text_sha256\thand_text_sha256\tsem_node\thand_node" > "$GATE"
text_of(){ "$OBJCOPY" -O binary --only-section=.text "$1" /dev/stdout 2>/dev/null | sha256sum | cut -d' ' -f1; }
while IFS=$'\t' read -r op sem hand; do
  [ -n "$op" ] || continue
  rt="/tmp/lanemk-idg-$op"; rm -rf "$rt"; mkdir -p "$rt"
  ( cd "$FARM" && CHIP_ARCH=blackhole SHORT_ARCH=bh LLK_HOME="$LLK_HOME_" RUNNER_TEMP="$rt" PYTHONUNBUFFERED=1 \
      timeout 300 "$VENV" -m pytest -o addopts= -q --compile-producer "$sem" "$hand" >"$rt/compile.log" 2>&1 )
  crc=$?
  mapfile -t elfs < <(find "$rt" -name math.elf 2>/dev/null)
  if [ "$crc" -ne 0 ] || [ "${#elfs[@]}" -ne 2 ]; then
    echo -e "$op\tCOMPILE-FAIL(rc=$crc,elfs=${#elfs[@]})\t-\t-\t$sem\t$hand" >> "$GATE"
    rm -rf "$rt/tt-llk-build"; continue
  fi
  t1=$(text_of "${elfs[0]}"); t2=$(text_of "${elfs[1]}")
  if [ -z "$t1" ] || [ -z "$t2" ]; then
    echo -e "$op\tTEXT-EMPTY\t$t1\t$t2\t$sem\t$hand" >> "$GATE"
  elif [ "$t1" == "$t2" ]; then
    echo -e "$op\tIDENTITY-FAIL(sem==hand)\t$t1\t$t2\t$sem\t$hand" >> "$GATE"
  else
    echo -e "$op\tOK(sem!=hand)\t$t1\t$t2\t$sem\t$hand" >> "$GATE"
  fi
  rm -rf "$rt/tt-llk-build"
done < "$MANIFEST"
echo "=== IDENTITY GATE SUMMARY ==="
awk -F'\t' 'NR>1{c[$2 ~ /^OK/ ? "OK" : "REFUSE"]++} END{print "OK="c["OK"]+0" REFUSE="c["REFUSE"]+0}' "$GATE"
column -t -s$'\t' "$GATE"
