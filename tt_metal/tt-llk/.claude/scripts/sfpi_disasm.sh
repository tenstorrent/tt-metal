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
# Each tree is a depth-1 fetch of one commit over HTTPS, so it costs a checkout
# and no history. Put --work somewhere local: a home directory over quota is what
# this script was written around.
#
# The sfpi binaries need glibc 2.34 or later (Ubuntu 22.04 and later). On an
# older host, run this inside a container. The script checks and says so.
#
# Usage:
#   sfpi_disasm.sh --work <dir> --python <venv python>
#                  --commit <sha> --other-sfpi <sha>
#                  [--repo-url <url>] [--test perf_math_matmul.py]
#                  [--select <-k expr>] [--jobs N] [--collect-only]

set -euo pipefail

REPO="https://github.com/tenstorrent/tt-metal"
WORK="" PY="" COMMIT="" OTHER="" TEST="perf_math_matmul.py"
SELECT="" JOBS=4 COLLECT=no

while [ $# -gt 0 ]; do
    case "$1" in
        --repo-url) REPO=$2; shift 2 ;;
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

[ -n "$WORK$PY$COMMIT" ] || { echo "usage: see the header" >&2; exit 2; }

say() { echo "[$(date -u +%H:%M:%S)] $*"; }
N() { nice -n 19 "$@"; }

# The toolchain is built against glibc 2.34. An older host links it and fails
# per kernel, deep in a pytest error, so say it once and up front.
glibc_check() {
    local have; have=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')
    [ -n "$have" ] || return 0
    if [ "$(printf '%s\n2.34\n' "$have" | sort -V | head -1)" != "2.34" ]; then
        echo "[ERROR] glibc $have is too old for sfpi, which needs 2.34." >&2
        echo "        Run this inside a container with Ubuntu 22.04 or later." >&2
        exit 1
    fi
}

mkdir -p "$WORK"

# --- the trees ---------------------------------------------------------------
tree() {
    local dir=$1 sfpi_from=$2
    if [ ! -e "$dir/.git" ]; then
        say "fetching $(basename "$dir") at ${COMMIT:0:11}"
        mkdir -p "$dir"
        ( cd "$dir"; git init -q .; git remote add origin "$REPO" )
    fi
    N git -C "$dir" fetch -q --depth=1 origin "$COMMIT"
    N git -C "$dir" checkout -q -f FETCH_HEAD
    # Replace only the toolchain pin. The LLK sources stay byte-identical, so a
    # difference in the generated code can only come from the compiler.
    if [ -n "$sfpi_from" ]; then
        N git -C "$dir" fetch -q --depth=1 origin "$sfpi_from"
        N git -C "$dir" show "FETCH_HEAD:tt_metal/sfpi-version" > "$dir/tt_metal/sfpi-version"
    fi
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
# The kernels land in $RUNNER_TEMP/tt-llk-build, not in the tree, so each tag
# gets its own root. Without that the two toolchains overwrite each other.
art() { echo "$WORK/art-$1/tt-llk-build"; }

build() {
    local tests=$1 tag=$2
    say "compiling $tag (jobs $JOBS) ..."
    rm -rf "$WORK/art-$tag"; mkdir -p "$WORK/art-$tag"
    ( cd "$tests"
      set +e
      CHIP_ARCH=blackhole TT_LLK_DISABLE_ASSERTS=1 RUNNER_TEMP="$WORK/art-$tag" \
        N "$PY" -m pytest --compile-producer -n "$JOBS" -q \
          ${SELECT:+-k "$SELECT"} "python_tests/$TEST" > "$WORK/build-$tag.log" 2>&1
      say "  pytest exit $?"
      set -e
    )
    say "  ELFs: $(find "$(art "$tag")" -name '*.elf' 2>/dev/null | wc -l)"
}

# --- disassemble -------------------------------------------------------------
# Every ELF is dumped, not only the math one. A kernel the toolchain leaves
# alone is as much of an answer as one it changes, and the build names each
# variant by a hash of its source and flags, so the two trees agree on names.
dump() {
    local tests=$1 tag=$2
    local objdump; objdump=$(find "$tests/sfpi" -name '*objdump' -type f | head -1)
    local out="$WORK/dis-$tag"; rm -rf "$out"; mkdir -p "$out"
    local n=0
    while IFS= read -r elf; do
        # Name each dump by its path under the build root, so the two trees
        # produce the same names and a missing one is visible.
        local variant; variant=${elf#"$(art "$tag")"/}; variant=${variant//\//_}
        # Drop addresses and the header: only the instruction stream is compared.
        "$objdump" -d --no-show-raw-insn "$elf" \
          | sed -e '1,/Disassembly of section/d' -e 's/^[[:space:]]*[0-9a-f]*:[[:space:]]*//' \
          > "$out/$variant.asm"
        n=$((n+1))
    done < <(find "$(art "$tag")" -name '*.elf' | sort)
    say "  dumped $n ELFs for $tag (objdump: $objdump)"
}

main() {
    glibc_check
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
