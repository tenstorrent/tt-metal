#!/usr/bin/env bash
# ttnop test suite. Two layers:
#   (1) functional: build a freestanding RV32 program, instrument it many ways, run
#       original vs instrumented under qemu-riscv32, assert identical output.
#   (2) structural: if real linked kernels are available, instrument them and check
#       the ELF stays valid, only the detoured words change, and --verify passes.
# Gated tests skip (not fail) when a tool is missing.
set -u
cd "$(dirname "$0")"
ROOT=..
TT=$ROOT/ttnop
GCC=riscv64-elf-gcc
QEMU=qemu-riscv32-static
WORK=work
rm -rf "$WORK"; mkdir -p "$WORK"

pass=0; fail=0; skip=0
ok()   { echo "  ok   - $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL - $1"; fail=$((fail+1)); }
skip_test(){ echo "  skip - $1"; skip=$((skip+1)); }
have() { command -v "$1" >/dev/null 2>&1; }

# Echo "LO HI" for the cave range from ttnop's "cave 0xLO..0xHI" report line.
cave_range() { sed -nE 's/.*cave (0x[0-9a-f]+)\.\.(0x[0-9a-f]+).*/\1 \2/p' | tail -1; }

[ -x "$TT" ] || { echo "build ttnop first (make)"; exit 2; }

# ---------------------------------------------------------------- functional
if have "$GCC" && have "$QEMU"; then
  BASE=$WORK/kern
  if $GCC -march=rv32im -mabi=ilp32 -O2 -nostdlib -nostartfiles -ffreestanding -static \
       -Wl,-T,fixtures/tensix_like.ld fixtures/kern.c -o "$BASE" 2>"$WORK/cc.log"; then
    EXP=$($QEMU "$BASE" 2>/dev/null)
    if [ -n "$EXP" ]; then ok "fixture builds and runs (baseline output: $EXP)"
    else bad "fixture produced no output"; EXP=__none__; fi

    # Equivalence under a spread of instrumentation strategies.
    configs=(
      "0x10000=1"
      "--every store=4"
      "--every branch=9"
      "--every call=5"
      "--every jal=3"
      "--every load=2 --every op=2"
      "--every system=4 --every fence=4"
      "--every all=1"
    )
    for cfg in "${configs[@]}"; do
      if $TT "$BASE" -o "$WORK/inst" $cfg --verify >"$WORK/t.log" 2>&1; then
        got=$($QEMU "$WORK/inst" 2>/dev/null)
        [ "$got" = "$EXP" ] && ok "equiv: $cfg" || bad "equiv: $cfg (got '$got' want '$EXP')"
      else
        bad "instrument failed: $cfg ($(tail -1 "$WORK/t.log"))"
      fi
    done

    # The injected delay must actually execute: the cave's address range must appear
    # in qemu's lazily-translated instruction trace.
    read CLO CHI < <($TT "$BASE" -o "$WORK/inst" --every store=3 2>&1 | cave_range)
    $QEMU -d in_asm -D "$WORK/tr.log" "$WORK/inst" >/dev/null 2>&1
    seen=$(grep -oE '0x[0-9a-f]{8}' "$WORK/tr.log" | sort -u | \
           awk -v lo="$CLO" -v hi="$CHI" '{a=strtonum($1); if(a>=strtonum(lo)&&a<strtonum(hi))c++} END{print c+0}')
    [ "${seen:-0}" -gt 0 ] && ok "cave executes under qemu ($seen cave words in trace)" \
                           || bad "cave never executed (0 cave words in trace)"

    # More NOPs => strictly larger cave (the delay knob scales).
    read _ H1 < <($TT "$BASE" -o "$WORK/a" 0x10010=10  2>&1 | cave_range)
    read _ H2 < <($TT "$BASE" -o "$WORK/b" 0x10010=100 2>&1 | cave_range)
    [ "$((H2))" -gt "$((H1))" ] && ok "cave grows with nop count" || bad "cave size did not scale with nops"
  else
    bad "fixture failed to build: $(tail -1 "$WORK/cc.log")"
  fi
else
  skip_test "functional tests need $GCC and $QEMU"
fi

# ---------------------------------------------------------------- structural
KDIR=$ROOT/../elfsight/elfs/metal_elfs_bulk
shopt -s nullglob
kernels=("$KDIR"/*trisc0*.elf "$KDIR"/*brisc*.elf)
if [ ${#kernels[@]} -gt 0 ]; then
  for K in "${kernels[@]}"; do
    [ -f "$K" ] || continue
    name=$(basename "$K"); O=$WORK/$name
    if ! $TT "$K" -o "$O" --every store=2 --verify >"$WORK/k.log" 2>&1; then
      bad "real kernel instrument/verify: $name ($(tail -1 "$WORK/k.log"))"; continue
    fi
    readelf -h "$O" >/dev/null 2>&1 && readelf -l "$O" >/dev/null 2>&1 \
      && ok "real kernel stays valid ELF: $name" || bad "real kernel output invalid: $name"

    # .text grows by the cave; its original prefix changes only at detoured words.
    riscv64-elf-objcopy -O binary --only-section=.text "$K" "$WORK/a.bin" 2>/dev/null
    riscv64-elf-objcopy -O binary --only-section=.text "$O" "$WORK/b.bin" 2>/dev/null
    asz=$(stat -c%s "$WORK/a.bin"); bsz=$(stat -c%s "$WORK/b.bin")
    if [ "$bsz" -gt "$asz" ]; then
      ndiff=$(cmp -l <(head -c "$asz" "$WORK/b.bin") "$WORK/a.bin" 2>/dev/null | \
              awk '{print int(($1-1)/4)}' | sort -u | wc -l)
      ndet=$(sed -nE 's/.* ([0-9]+) detour.*/\1/p' "$WORK/k.log" | tail -1)
      [ "${ndiff:-0}" = "${ndet:-x}" ] \
        && ok ".text grew, only $ndiff detoured word(s) changed: $name" \
        || bad ".text prefix changed in $ndiff words but $ndet detours: $name"
    else
      bad ".text did not grow: $name"
    fi
  done
else
  skip_test "structural tests need real kernels under $KDIR"
fi

echo
echo "ttnop tests: $pass passed, $fail failed, $skip skipped"
[ "$fail" -eq 0 ]
