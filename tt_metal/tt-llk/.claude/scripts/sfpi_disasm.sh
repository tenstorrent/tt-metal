#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Build the Blackhole math-matmul kernels with two sfpi toolchains and diff the
# disassembly, to see whether an sfpi bump changes the generated code.
#
# It builds the same commit's kernels twice: once with the sfpi the tree pins and
# once with the sfpi named by --other. Only the toolchain differs, so any ELF that
# differs is the toolchain's doing and nothing else.
#
# Usage: sfpi_disasm.sh --repo <path> --work <dir> [--test <file>] [--select <-k expr>]
#                       [--other <sfpi commit>] [--jobs N]

set -euo pipefail

REPO=""
WORK=""
TEST="perf_math_matmul.py"
SELECT=""
OTHER=""
JOBS="$(nproc)"

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO=$2; shift 2 ;;
        --work) WORK=$2; shift 2 ;;
        --test) TEST=$2; shift 2 ;;
        --select) SELECT=$2; shift 2 ;;
        --other) OTHER=$2; shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -n "$REPO" ] && [ -n "$WORK" ] && [ -n "$OTHER" ] || {
    echo "usage: $0 --repo <path> --work <dir> --other <sfpi commit>" >&2
    exit 2
}

mkdir -p "$WORK"
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- the two trees -----------------------------------------------------------
# Both are the SAME commit. The second one has only its sfpi-version replaced, so
# the toolchain is the only difference between the two builds.
prepare_tree() {
    local dir=$1 commit=$2 sfpi_from=$3
    if [ ! -d "$dir" ]; then
        say "worktree $dir at ${commit:0:11}"
        git -C "$REPO" worktree add --detach --force "$dir" "$commit" >/dev/null
    fi
    if [ -n "$sfpi_from" ]; then
        say "  using the sfpi-version from ${sfpi_from:0:11}"
        git -C "$REPO" show "${sfpi_from}:tt_metal/sfpi-version" > "$dir/tt_metal/sfpi-version"
    fi
    grep -m1 "^sfpi_version=" "$dir/tt_metal/sfpi-version"
}

# --- the toolchain -----------------------------------------------------------
# Same download and hash check that tests/setup_testing_env.sh does, without the
# pre-commit hooks it also installs.
install_sfpi() {
    local tests=$1
    ( cd "$tests"
      if [ -f sfpi/sfpi.version ]; then
          eval "$(./sfpi-info.sh SHELL txz)"
          [ "$sfpi_version" = "$(cat sfpi/sfpi.version)" ] && { say "  sfpi $sfpi_version already here"; return 0; }
      fi
      eval "$(./sfpi-info.sh SHELL txz)"
      say "  fetching sfpi $sfpi_version ($sfpi_filename)"
      local tmp; tmp=$(mktemp -d); trap 'rm -rf "$tmp"' RETURN
      curl -sSfL --retry 3 -o "$tmp/$sfpi_filename" "$sfpi_url/$sfpi_filename"
      local got; got=$("${sfpi_hashtype}sum" -b "$tmp/$sfpi_filename" | cut -d' ' -f1)
      [ "$got" = "$sfpi_hash" ] || { echo "[ERROR] $sfpi_filename hash mismatch: $got != $sfpi_hash" >&2; return 1; }
      say "  hash verified"
      rm -rf sfpi
      tar -xJf "$tmp/$sfpi_filename" -C .
      echo "$sfpi_version" > sfpi/sfpi.version
    )
}

# --- compile -----------------------------------------------------------------
# --compile-producer only builds; it never touches a board.
build() {
    local tests=$1 tag=$2
    say "compiling $tag ..."
    ( cd "$tests"
      set +e
      CHIP_ARCH=blackhole TT_LLK_DISABLE_ASSERTS=1 \
        python3 -m pytest --compile-producer -n "$JOBS" -q \
          ${SELECT:+-k "$SELECT"} "python_tests/$TEST" \
          > "$WORK/build-$tag.log" 2>&1
      echo "  pytest exit $?" >&2
      set -e
    )
    say "  ELFs: $(find "$tests" -name 'trisc1.elf' 2>/dev/null | wc -l)"
}

# --- disassemble and compare -------------------------------------------------
# ELF hashes embed the build path, so compare disassembly, never the bytes.
dump_all() {
    local tests=$1 tag=$2 objdump=$3
    local out="$WORK/dis-$tag"; rm -rf "$out"; mkdir -p "$out"
    local n=0
    while IFS= read -r elf; do
        # Name each dump after the variant directory the ELF sits in.
        local variant; variant=$(basename "$(dirname "$(dirname "$elf")")")
        "$objdump" -d --no-show-raw-insn "$elf" \
          | sed -e '1,/Disassembly of section/d' -e 's/^[[:space:]]*[0-9a-f]\+://' \
          > "$out/$variant.asm"
        n=$((n+1))
    done < <(find "$tests" -name 'trisc1.elf' | sort)
    say "  dumped $n math ELFs for $tag"
}

main() {
    local A=ff89f4eaec4629ce8ad12329c23b0b46c278de2a   # main, sfpi 7.91.0
    local B=2ef9d776d9ebaba50135646388169373d9afa79a   # PR #62951, sfpi 7.92.0

    git -C "$REPO" fetch -q origin "$A" "$B" 2>/dev/null || git -C "$REPO" fetch -q origin || true

    prepare_tree "$WORK/t791" "$A" ""
    prepare_tree "$WORK/t792" "$A" "$B"

    for t in t791 t792; do
        install_sfpi "$WORK/$t/tt_metal/tt-llk/tests"
    done

    for t in t791 t792; do
        build "$WORK/$t/tt_metal/tt-llk/tests" "$t"
    done

    local od791 od792
    od791=$(find "$WORK/t791/tt_metal/tt-llk/tests/sfpi" -name '*objdump' | head -1)
    od792=$(find "$WORK/t792/tt_metal/tt-llk/tests/sfpi" -name '*objdump' | head -1)
    say "objdump 7.91: $od791"
    say "objdump 7.92: $od792"

    dump_all "$WORK/t791/tt_metal/tt-llk/tests" t791 "$od791"
    dump_all "$WORK/t792/tt_metal/tt-llk/tests" t792 "$od792"

    say "comparing ..."
    : > "$WORK/same.txt"; : > "$WORK/differ.txt"
    for f in "$WORK/dis-t791"/*.asm; do
        local base; base=$(basename "$f")
        local other="$WORK/dis-t792/$base"
        [ -f "$other" ] || { echo "$base  (only in 7.91)" >> "$WORK/differ.txt"; continue; }
        if cmp -s "$f" "$other"; then
            echo "$base" >> "$WORK/same.txt"
        else
            echo "$base  (+$(diff "$f" "$other" | grep -c '^>') -$(diff "$f" "$other" | grep -c '^<'))" >> "$WORK/differ.txt"
        fi
    done

    echo
    echo "================ RESULT ================"
    echo "identical: $(wc -l < "$WORK/same.txt")"
    echo "different: $(wc -l < "$WORK/differ.txt")"
    echo
    echo "--- variants whose Blackhole math code changed ---"
    head -40 "$WORK/differ.txt"
    echo
    echo "(full lists in $WORK/same.txt and $WORK/differ.txt; dumps in $WORK/dis-t791 and $WORK/dis-t792)"
}

main "$@"
