#!/usr/bin/env bash
# laneMK — compile each op's sem+hand certified kernel (pin-59) and run the object-identity
# gate: extract math.elf .text sha256 per leg, assert sem != hand (cross-binary), record.
# Emits a manifest TSV (op, sem_node, hand_node, sem_text, hand_text, status). Parameterized;
# no hard-coded personal paths beyond the required --farm/--out args.
set -uo pipefail
FARM="${FARM:?set FARM=<tests/python_tests>}"
OUT="${OUT:?set OUT=<evidence dir>}"
MANIFEST="${MANIFEST:-}"
OBJCOPY="${OBJCOPY:?set OBJCOPY=<riscv-tt-elf-objcopy>}"
VENV="${VENV:?set VENV=<python>}"
LLK_HOME_="${LLK_HOME:?set LLK_HOME}"
mkdir -p "$OUT"
text_of(){ "$OBJCOPY" -O binary --only-section=.text "$1" /dev/stdout 2>/dev/null | sha256sum | cut -d' ' -f1; }

# Tri-arm producer. Compile each arm alone so A/B (same semantic node, different
# compiler profiles) cannot be mislabeled by a compile-together invocation.
# Merge the three keyed variants into one staged tt-llk-build and emit the map
# consumed by galaxy_shard.sh/streamers.
if [ -n "${TRI_PROFILES:-}" ]; then
  [ -f "$TRI_PROFILES" ] || { echo "FATAL: no TRI_PROFILES=$TRI_PROFILES" >&2; exit 2; }
  STAGE_BUILD="${STAGE_BUILD:-$OUT/tt-llk-build}"
  if [ -d "$STAGE_BUILD" ] && [ -n "$(find "$STAGE_BUILD" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "FATAL: STAGE_BUILD must be empty: $STAGE_BUILD" >&2
    exit 2
  fi
  mkdir -p "$STAGE_BUILD"
  TRI_MAP="${TRI_IDMAP_OUT:-$OUT/TRI-IDENTITY-MAP.tsv}"
  : > "$TRI_MAP"
  while IFS= read -r profile_row; do
    field(){ printf '%s\n' "$profile_row" | awk -F'\t' -v n="$1" '{print $n}'; }
    op=$(field 1); a_node=$(field 5); a_flags=$(field 6)
    b_node=$(field 7); b_flags=$(field 8); c_node=$(field 9); c_flags=$(field 10)
    c_flags=${c_flags%$'\r'}
    [ "$op" != op ] || continue
    [ -n "$op" ] || continue
    [ "$(printf '%s\n' "$profile_row" | awk -F'\t' '{print NF}')" -eq 10 ] || {
      echo "FATAL: malformed tri profile row for '$op'" >&2; exit 2; }
    [ "$b_flags" = "$c_flags" ] || { echo "FATAL: $op B/C flags differ" >&2; exit 2; }
    [ "$a_node" = "$b_node" ] || { echo "FATAL: $op A/B nodes differ" >&2; exit 2; }
    variants=(); hashes=()
    for arm in a b c; do
      case "$arm" in
        a) node=$a_node; flags=$a_flags ;;
        b) node=$b_node; flags=$b_flags ;;
        c) node=$c_node; flags=$c_flags ;;
      esac
      rt=$(mktemp -d "${TMPDIR:-/tmp}/sfpu-tri-${op}-${arm}.XXXXXX") || exit 2
      ( cd "$FARM" && CHIP_ARCH=blackhole SHORT_ARCH=bh LLK_HOME="$LLK_HOME_" \
          RUNNER_TEMP="$rt" PYTHONUNBUFFERED=1 TT_LLK_EXTRA_COMPILER_OPTIONS="$flags" \
          timeout 300 "$VENV" -m pytest -o addopts= -q --compile-producer "$node" \
          >"$rt/compile.log" 2>&1 )
      crc=$?
      mapfile -t elfs < <(find "$rt/tt-llk-build/sources" -name math.elf 2>/dev/null | LC_ALL=C sort)
      if [ "$crc" -ne 0 ] || [ "${#elfs[@]}" -ne 1 ]; then
        cp "$rt/compile.log" "$OUT/${op}-${arm}-compile.log"
        rm -rf -- "$rt"
        echo "FATAL: $op arm=$arm compile rc=$crc elfs=${#elfs[@]}" >&2
        exit 1
      fi
      elf=${elfs[0]}
      variant=$(basename "$(dirname "$(dirname "$elf")")")
      hash=$(text_of "$elf")
      [ -n "$variant" ] && [ -n "$hash" ] || { rm -rf -- "$rt"; exit 1; }
      variants+=("$variant"); hashes+=("$hash")
      cp -a "$rt/tt-llk-build/." "$STAGE_BUILD/"
      rm -rf -- "$rt"
    done
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$op" \
      "${variants[0]}" "${hashes[0]}" "${variants[1]}" "${hashes[1]}" \
      "${variants[2]}" "${hashes[2]}" >> "$TRI_MAP"
  done < "$TRI_PROFILES"
  echo "tri identity map -> $TRI_MAP"
  echo "staged build -> $STAGE_BUILD"
  exit 0
fi

[ -n "$MANIFEST" ] || { echo "FATAL: set MANIFEST or TRI_PROFILES" >&2; exit 2; }
GATE="$OUT/IDENTITY-GATE.tsv"
echo -e "op\tstatus\tsem_text_sha256\thand_text_sha256\tsem_node\thand_node" > "$GATE"
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
