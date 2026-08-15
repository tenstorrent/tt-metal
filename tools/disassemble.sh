#!/usr/bin/env bash

# Disassemble Tensix firmware ELFs with the matching SFPI objdump.
#
# The script deliberately accepts an explicit --objdump path so archived
# compiler/toolchain provenance can be reproduced.  Its automatic lookup is a
# convenience for normal TT-Metal worktrees, not a substitute for pinning the
# tool in evidence manifests.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/disassemble.sh [options] ELF_OR_BUILD_DIR [...]

Disassemble one or more Tensix ELFs. A directory is accepted when it contains
exactly one math.elf; otherwise pass the desired ELF explicitly.

Options:
  --objdump PATH       Use this riscv-tt-elf-objdump binary.
  --symbol NAME        Show demangled function(s) whose name starts with NAME.
  --match REGEX        Show only matching lines, with optional context.
  --sfpu-only          Match SFPU/Tensix issue instructions.
  --context N          Context lines for --match/--sfpu-only (default: 2).
  --source             Intermix source with disassembly (objdump -S).
  --section NAME       Restrict disassembly to one section.
  --output PATH        Write combined output to PATH instead of stdout.
  --no-header          Omit ELF/tool/SHA256 provenance headers.
  -h, --help           Show this help.

Tool lookup order:
  1. --objdump PATH
  2. SFPI_OBJDUMP
  3. riscv-tt-elf-objdump on PATH
  4. SFPI_HOME and the current TT-Metal worktree's tt-llk test toolchain

Examples:
  tools/disassemble.sh build/.../elf/math.elf
  tools/disassemble.sh --symbol run_kernel build/.../elf/math.elf
  tools/disassemble.sh --sfpu-only --context 6 build/.../elf/math.elf
  tools/disassemble.sh --match 'SFPMAD|TTREPLAY' a.elf b.elf
  SFPI_OBJDUMP=/opt/sfpi/compiler/bin/riscv-tt-elf-objdump \
    tools/disassemble.sh --output math.objdump math.elf
EOF
}

die() {
    printf 'disassemble.sh: %s\n' "$*" >&2
    exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)

objdump_path=""
symbol=""
match_regex=""
context=2
with_source=0
section=""
output=""
with_header=1
declare -a inputs=()

while (($#)); do
    case "$1" in
        --objdump)
            (($# >= 2)) || die '--objdump requires a path'
            objdump_path=$2
            shift 2
            ;;
        --symbol)
            (($# >= 2)) || die '--symbol requires a name'
            symbol=$2
            shift 2
            ;;
        --match)
            (($# >= 2)) || die '--match requires a regular expression'
            match_regex=$2
            shift 2
            ;;
        --sfpu-only)
            [[ -z "$match_regex" ]] || die '--sfpu-only and --match are mutually exclusive'
            match_regex='sfp[a-z0-9_]*|tt(replay|mop|incrwc|setrwc|setc16|stallwait|sem[a-z0-9_]*)'
            shift
            ;;
        --context)
            (($# >= 2)) || die '--context requires a non-negative integer'
            [[ $2 =~ ^[0-9]+$ ]] || die '--context requires a non-negative integer'
            context=$2
            shift 2
            ;;
        --source)
            with_source=1
            shift
            ;;
        --section)
            (($# >= 2)) || die '--section requires a name'
            section=$2
            shift 2
            ;;
        --output)
            (($# >= 2)) || die '--output requires a path'
            output=$2
            shift 2
            ;;
        --no-header)
            with_header=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            inputs+=("$@")
            break
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            inputs+=("$1")
            shift
            ;;
    esac
done

((${#inputs[@]} > 0)) || {
    usage >&2
    exit 2
}

find_objdump() {
    local candidate
    local -a candidates=()

    [[ -n "$objdump_path" ]] && candidates+=("$objdump_path")
    [[ -n "${SFPI_OBJDUMP:-}" ]] && candidates+=("$SFPI_OBJDUMP")

    if command -v riscv-tt-elf-objdump >/dev/null 2>&1; then
        candidates+=("$(command -v riscv-tt-elf-objdump)")
    fi

    if [[ -n "${SFPI_HOME:-}" ]]; then
        candidates+=(
            "$SFPI_HOME/compiler/bin/riscv-tt-elf-objdump"
            "$SFPI_HOME/bin/riscv-tt-elf-objdump"
        )
    fi

    candidates+=(
        "$repo_root/tt_metal/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-objdump"
        "$repo_root/tt_metal/third_party/sfpi/compiler/bin/riscv-tt-elf-objdump"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            readlink -f -- "$candidate"
            return 0
        fi
    done

    die 'could not find riscv-tt-elf-objdump; pass --objdump or set SFPI_OBJDUMP'
}

resolve_elf() {
    local input=$1
    local -a matches=()

    [[ -e "$input" ]] || die "input does not exist: $input"
    if [[ -f "$input" ]]; then
        printf '%s\n' "$(readlink -f -- "$input")"
        return 0
    fi
    [[ -d "$input" ]] || die "input is neither a file nor directory: $input"

    while IFS= read -r -d '' match; do
        matches+=("$match")
    done < <(find "$input" -type f -name math.elf -print0)

    ((${#matches[@]} > 0)) || die "no math.elf found below: $input"
    if ((${#matches[@]} != 1)); then
        printf 'disassemble.sh: multiple math.elf files below %s; pass one explicitly:\n' "$input" >&2
        printf '  %s\n' "${matches[@]}" >&2
        exit 2
    fi
    readlink -f -- "${matches[0]}"
}

objdump_path=$(find_objdump)
declare -a elfs=()
for input in "${inputs[@]}"; do
    elfs+=("$(resolve_elf "$input")")
done

declare -a objdump_args=(-d -C -l --wide)
((with_source == 0)) || objdump_args+=(-S)
[[ -z "$section" ]] || objdump_args+=("--section=$section")

select_symbol() {
    local name=$1
    awk -v prefix="<$name" '
        /^[[:xdigit:]]+[[:space:]]+<.*>:/ {
            selected = index($0, prefix) > 0
            if (selected) {
                found = 1
            }
        }
        selected { print }
        END {
            if (!found) {
                exit 3
            }
        }
    '
}

disassemble_one() {
    local elf=$1
    if [[ -n "$symbol" ]]; then
        "$objdump_path" "${objdump_args[@]}" "$elf" | select_symbol "$symbol"
    else
        "$objdump_path" "${objdump_args[@]}" "$elf"
    fi
}

emit_one() {
    local elf=$1

    if ((with_header)); then
        printf '### ELF: %s\n' "$elf"
        printf '### ELF SHA256: %s\n' "$(sha256sum "$elf" | awk '{print $1}')"
        printf '### OBJDUMP: %s\n' "$objdump_path"
        printf '### OBJDUMP SHA256: %s\n' "$(sha256sum "$objdump_path" | awk '{print $1}')"
    fi

    if [[ -n "$match_regex" ]]; then
        disassemble_one "$elf" |
            grep --color=never -Ein -C "$context" -- "$match_regex"
    else
        disassemble_one "$elf"
    fi
}

emit_all() {
    local first=1
    local elf
    for elf in "${elfs[@]}"; do
        if ((first == 0)); then
            printf '\n'
        fi
        emit_one "$elf"
        first=0
    done
}

if [[ -n "$output" ]]; then
    emit_all >"$output"
else
    emit_all
fi
