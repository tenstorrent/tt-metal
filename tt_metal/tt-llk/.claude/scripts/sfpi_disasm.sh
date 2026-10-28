#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Build the Blackhole kernels with two sfpi toolchains and diff the disassembly,
# to see whether an sfpi bump changes the generated code.
#
# Both trees are the SAME commit. Only tt_metal/sfpi-version differs, so an ELF
# that differs is the toolchain's doing and nothing else. It compares
# disassembly, never bytes: ELF hashes embed the build path.
#
# The trees are `git clone -s` of a repo you already have, so they cost almost no
# disk and never write to it. Put --work somewhere local: a home directory over
# quota is what this script was written around.
#
# Usage:
#   sfpi_disasm.sh --repo <path> --work <dir> --python <venv python>
#                  --commit <sha> --other-sfpi <sha>
#                  [--test perf_math_matmul.py] [--select <-k expr>] [--jobs N]
#                  [--collect-only]

set -euo pipefail

REPO="" WORK="" PY="" COMMIT="" OTHER="" TEST="perf_math_matmul.py"
SELECT="" JOBS=4 COLLECT=no

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO=$2; shift 2 ;;
        --work) WORK=$2; shift 2 ;;
        --python) PY=$2; shift 2 ;;
        --commit) COMMIT=$2; shift 2 ;;
        --other-sfpi) OTHER=$2; shift 2 ;;
        --test) TEST=$2; shift 2 ;;
        --select) SELECT=$2; shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        --collect-only) COLLECT=yes; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -n "$REPO$WORK$PY$COMMIT" ] || { echo "usage: see the header" >&2; exit 2; }

say() { echo "[$(date -u +%H:%M:%S)] $*"; }
N() { nice -n 19 "$@"; }

mkdir -p "$WORK"

# --- the trees ---------------------------------------------------------------
tree() {
    local dir=$1 sfpi_from=$2
    if [ ! -d "$dir/.git" ]; then
        say "cloning $dir"
        N git clone -q -s --no-checkout "$REPO" "$dir"
    fi
    N git -C "$dir" checkout -q -f "$COMMIT"
    # Replace only the toolchain pin. The LLK sources stay byte-identical.
    [ -n "$sfpi_from" ] && N git -C "$REPO" show "${sfpi_from}:tt_metal/sfpi-version" > "$dir/tt_metal/sfpi-version"
    say "  $(basename "$dir"): $(grep -m1 '^sfpi_version=' "$dir/tt_metal/sfpi-version")"
}

# --- the toolchain -----------------------------------------------------------
# The download and hash check that tests/setup_testing_env.sh does, without the
# pre-commit hooks it also installs.
sfpi() {
    local tests=$1
    ( cd "$tests"
      eval "$(./sfpi-info.sh SHELL txz)"
      if [ -f sfpi/sfpi.version ] && [ "$sfpi_version" = "$(cat sfpi/sfpi.version)" ]; then
          say "  sfpi $sfpi_version already here"; return 0
      fi
      say "  fetching sfpi $sfpi_version"
      local tmp; tmp=$(mktemp -d -p "$(dirname "$tests")")
      curl -sSfL --retry 3 -o "$tmp/$sfpi_filename" "$sfpi_url/$sfpi_filename"
      local got; got=$("${sfpi_hashtype}sum" -b "$tmp/$sfpi_filename" | cut -d' ' -f1)
      if [ "$got" != "$sfpi_hash" ]; then
          echo "[ERROR] $sfpi_filename hash $got != $sfpi_hash" >&2; rm -rf "$tmp"; return 1
      fi
      say "  hash verified"
      rm -rf sfpi; tar -xJf "$tmp/$sfpi_filename" -C .; rm -rf "$tmp"
      echo "$sfpi_version" > sfpi/sfpi.version
    )
}

# --- compile -----------------------------------------------------------------
# --compile-producer only builds. It never opens a device.
build() {
    local tests=$1 tag=$2
    say "compiling $tag (jobs $JOBS) ..."
    ( cd "$tests"
      set +e
      CHIP_ARCH=blackhole TT_LLK_DISABLE_ASSERTS=1 \
        N "$PY" -m pytest --compile-producer -n "$JOBS" -q \
          ${SELECT:+-k "$SELECT"} "python_tests/$TEST" > "$WORK/build-$tag.log" 2>&1
      say "  pytest exit $?"
      set -e
    )
    say "  math ELFs: $(find "$tests" -name 'trisc1.elf' 2>/dev/null | wc -l)"
}

# --- disassemble -------------------------------------------------------------
dump() {
    local tests=$1 tag=$2
    local objdump; objdump=$(find "$tests/sfpi" -name '*objdump' -type f | head -1)
    local out="$WORK/dis-$tag"; rm -rf "$out"; mkdir -p "$out"
    local n=0
    while IFS= read -r elf; do
        local variant; variant=$(basename "$(dirname "$(dirname "$elf")")")
        # Drop addresses and the header: only the instruction stream is compared.
        "$objdump" -d --no-show-raw-insn "$elf" \
          | sed -e '1,/Disassembly of section/d' -e 's/^[[:space:]]*[0-9a-f]*:[[:space:]]*//' \
          > "$out/$variant.asm"
        n=$((n+1))
    done < <(find "$tests" -name 'trisc1.elf' | sort)
    say "  dumped $n ELFs for $tag (objdump: $objdump)"
}

main() {
    tree "$WORK/t_a" ""        # the commit's own sfpi
    tree "$WORK/t_b" "$OTHER"  # the same commit, other sfpi

    if [ "$COLLECT" = yes ]; then
        say "collecting variant ids ..."
        ( cd "$WORK/t_a/tt_metal/tt-llk/tests"
          CHIP_ARCH=blackhole N "$PY" -m pytest --collect-only -q \
            ${SELECT:+-k "$SELECT"} "python_tests/$TEST" > "$WORK/ids.txt" 2>&1 || true )
        tail -3 "$WORK/ids.txt"; head -5 "$WORK/ids.txt"
        return 0
    fi

    sfpi "$WORK/t_a/tt_metal/tt-llk/tests"
    sfpi "$WORK/t_b/tt_metal/tt-llk/tests"

    build "$WORK/t_a/tt_metal/tt-llk/tests" a
    build "$WORK/t_b/tt_metal/tt-llk/tests" b
    dump  "$WORK/t_a/tt_metal/tt-llk/tests" a
    dump  "$WORK/t_b/tt_metal/tt-llk/tests" b

    say "comparing ..."
    : > "$WORK/same.txt"; : > "$WORK/differ.txt"
    for f in "$WORK/dis-a"/*.asm; do
        [ -e "$f" ] || break
        local base other; base=$(basename "$f"); other="$WORK/dis-b/$base"
        if [ ! -f "$other" ]; then echo "$base  (only in A)" >> "$WORK/differ.txt"
        elif cmp -s "$f" "$other"; then echo "$base" >> "$WORK/same.txt"
        else echo "$base  ($(diff "$f" "$other" | grep -c '^[<>]') changed lines)" >> "$WORK/differ.txt"
        fi
    done

    echo
    echo "================ RESULT ================"
    echo "identical : $(wc -l < "$WORK/same.txt")"
    echo "different : $(wc -l < "$WORK/differ.txt")"
    echo
    sed -n '1,40p' "$WORK/differ.txt"
    echo
    echo "lists: $WORK/same.txt  $WORK/differ.txt ; dumps: $WORK/dis-a $WORK/dis-b"
}

main "$@"
