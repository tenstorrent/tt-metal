#!/usr/bin/env bash
# Disassemble the newest generated TRISC object/ELF for the adhoc embedded run.
#
# Typical use:
#   export TT_METAL_FORCE_JIT_COMPILE=1
#   export TT_METAL_BACKEND_DUMP_RUN_CMD=1
#   export TT_METAL_CACHE=/tmp/tt-metal-cache-inspect
#   run_csv.sh <coeff.csv> --activation <act> --precision bf16 --tiles 256 --runs 1
#   tools/disassemble_adhoc.sh --cache /tmp/tt-metal-cache-inspect --out /tmp/adhoc.dis
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
BUILD_DIR="${TT_METAL_RUNTIME_ROOT:-$REPO_ROOT}/build_Release"
OBJDUMP="$BUILD_DIR/libexec/tt-metalium/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
CACHE="${TT_METAL_CACHE:-}"
OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache) CACHE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--cache <TT_METAL_CACHE>] [--out <path>]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -x "$OBJDUMP" ]]; then
    echo "ERROR: TT objdump not found or not executable: $OBJDUMP" >&2
    exit 1
fi

SEARCH_ROOTS=()
if [[ -n "$CACHE" && -d "$CACHE" ]]; then
    SEARCH_ROOTS+=("$CACHE")
fi
SEARCH_ROOTS+=("$BUILD_DIR")

# The embedded activation math body is the adhoc TRISC1 ELF. Prefer it over
# arbitrary newest ELF files: XIP dumps, pack/unpack kernels, or unrelated
# cached kernels can otherwise win the timestamp race and produce misleading
# disassembly.
artifact="$(
    find "${SEARCH_ROOTS[@]}" -path '*/kernels/adhoc/*/trisc1/trisc1.elf' -printf '%T@ %p\n' 2>/dev/null \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
)"

if [[ -z "$artifact" ]]; then
    artifact="$(
        find "${SEARCH_ROOTS[@]}" \( -name 'trisc*.elf' -o -name 'trisc*.o' -o -name '*.elf' \) -printf '%T@ %p\n' 2>/dev/null \
            | sort -n \
            | tail -1 \
            | cut -d' ' -f2-
    )"
fi

if [[ -z "$artifact" || ! -f "$artifact" ]]; then
    echo "ERROR: no TRISC ELF/object found under: ${SEARCH_ROOTS[*]}" >&2
    exit 1
fi

echo "Disassembling: $artifact" >&2
if [[ -n "$OUT" ]]; then
    mkdir -p "$(dirname "$OUT")"
    "$OBJDUMP" -d -S "$artifact" > "$OUT"
    echo "$OUT"
else
    "$OBJDUMP" -d -S "$artifact"
fi
