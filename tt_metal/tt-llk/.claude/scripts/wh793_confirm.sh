#!/usr/bin/env bash
# Is the SFPI 7.93.0 Wormhole LoFi matmul slowdown real, or a code-placement effect?
#
# Both trees hold the SAME source: the stage-2 branch, which carries the TRISC code-shift
# support. Only tt_metal/sfpi-version differs, so a difference can only come from the
# toolchain. Each side measures the LoFi perf_math_matmul tests with the code linked at
# 5 positions. Slower at all 5 means the toolchain; slower at some means placement.
#
# Run it on a reserved Wormhole board. Put --work somewhere local: the container's home
# is near full, so /tmp is the right place.
#
# Usage: wh793_confirm.sh [--work DIR] [--jobs N]
#          [--step all|trees|nodes|measure|probe|elfs|pack]
#   probe: PROBE_N=<n>   tie the first n test ids to the configs they measure
#   elfs:  ELF_NODE=<id> build that one test with both toolchains and diff the disassembly
set -uo pipefail

W=/tmp/s2-793
JOBS=15
STEP=all
while [ $# -gt 0 ]; do
    case "$1" in
        --work) W=$2; shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        --step) STEP=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO_URL=https://github.com/tenstorrent/tt-metal
BRANCH=nstojictt/llk-perf-gate-placement
# The two commits are one SFPI bump apart. Only their sfpi-version file is taken.
SFPI_A=3f57b50ff00415889e9e1405df36d80f4d1f4cbe   # pins 7.92.0
SFPI_B=012c19beaae306bb2b5053c69e91892506d4061f   # pins 7.93.0

REPO=$W/repo
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
run() { say "+ $*"; "$@"; }

# --- python ------------------------------------------------------------------
venv() {
    [ -x "$W/venv/bin/pytest" ] && return 0
    say "building the venv"
    python3 -m venv "$W/venv"
    "$W/venv/bin/pip" -q install --upgrade pip
    "$W/venv/bin/pip" -q install -r "$W/t_a/tt_metal/tt-llk/tests/requirements.txt"
}

# --- the trees ---------------------------------------------------------------
# One clone, two worktrees of the same commit. The only edit is the toolchain pin,
# so the generated code is the toolchain's doing and nothing else.
#
# The clone is blobless, shallow and single-branch, and it checks out only the paths
# the tests build from. github over HTTP/2 drops the connection on this container
# ("RPC failed ... CANCEL"), so it pins HTTP/1.1 and retries.
SPARSE=(/tt_metal/tt-llk/ /tt_metal/hw/ /tt_metal/hostdevcommon/
        /ttnn/cpp/ttnn/operations/experimental/ /tt_metal/sfpi-info.sh /tt_metal/sfpi-version)
GIT=(git -c http.version=HTTP/1.1 -c http.postBuffer=524288000)

clone() {
    [ -d "$REPO/.git" ] && return 0
    local try
    for try in 1 2 3; do
        rm -rf "$REPO"
        say "cloning (attempt $try)"
        "${GIT[@]}" clone -q --filter=blob:none --no-checkout --depth=1 \
            --single-branch --branch "$BRANCH" "$REPO_URL" "$REPO" && return 0
        say "  clone failed; retrying in 15 s"
        sleep 15
    done
    echo "[ERROR] could not clone $REPO_URL" >&2
    return 1
}

trees() {
    clone || return 1
    local head; head=$("${GIT[@]}" -C "$REPO" rev-parse HEAD) || return 1
    say "source commit: $head"
    for pair in "a $SFPI_A" "b $SFPI_B"; do
        set -- $pair
        local tag=$1 sha=$2 t=$W/t_$1
        if [ ! -f "$t/.ready" ]; then
            rm -rf "$t"
            "${GIT[@]}" -C "$REPO" worktree prune
            run "${GIT[@]}" -C "$REPO" worktree add -q --no-checkout --detach "$t" "$head" || return 1
            run "${GIT[@]}" -C "$t" sparse-checkout set --no-cone "${SPARSE[@]}" || return 1
            run "${GIT[@]}" -C "$t" checkout -q || return 1
            # Only the toolchain pin is replaced. tests/sfpi-version is a symlink to it.
            run "${GIT[@]}" -C "$REPO" fetch -q --depth=1 origin "$sha" || return 1
            "${GIT[@]}" -C "$REPO" show "$sha:tt_metal/sfpi-version" > "$t/tt_metal/sfpi-version" || return 1
            [ -s "$t/tt_metal/sfpi-version" ] || { echo "[ERROR] empty sfpi-version for $tag" >&2; return 1; }
            touch "$t/.ready"
        fi
        say "  tree $tag: $(grep -m1 '^sfpi_version=' "$t/tt_metal/sfpi-version")"
        sfpi "$t/tt_metal/tt-llk/tests" || return 1
    done
}

# The download and hash check setup_testing_env.sh does, without the pre-commit hooks.
sfpi() {
    ( cd "$1" || exit 1
      eval "$(./sfpi-info.sh SHELL txz)"
      if [ -f sfpi/sfpi.version ] && [ "$sfpi_version" = "$(cat sfpi/sfpi.version)" ]; then
          say "    sfpi $sfpi_version already here"; exit 0
      fi
      say "    fetching sfpi $sfpi_version"
      tmp=$(mktemp -d)
      curl -sSfL --retry 3 -o "$tmp/$sfpi_filename" "$sfpi_url/$sfpi_filename" || exit 1
      got=$("${sfpi_hashtype}sum" -b "$tmp/$sfpi_filename" | cut -d' ' -f1)
      [ "$got" = "$sfpi_hash" ] || { echo "[ERROR] $sfpi_filename hash $got != $sfpi_hash" >&2; exit 1; }
      rm -rf sfpi; tar -xJf "$tmp/$sfpi_filename" -C .; rm -rf "$tmp"
      echo "$sfpi_version" > sfpi/sfpi.version
      say "    sfpi $sfpi_version installed" )
}

# --- the tests to measure ----------------------------------------------------
# Only the LoFi matmul tests. They hold the cluster the CI artifacts flagged, and the
# HiFi ones stay out so the run fits the board time.
nodes() {
    local t=$W/t_a/tt_metal/tt-llk/tests/python_tests
    ( cd "$t" && PATH="$W/venv/bin:$PATH" CHIP_ARCH=wormhole TT_LLK_DISABLE_ASSERTS=1 \
        pytest --collect-only -q -m "perf and not accuracy" perf_math_matmul.py ) \
        > "$W/collect.txt" 2>&1
    grep -oE '^perf_math_matmul\.py::[^ ]*MathFidelity\.LoFi[^ ]*' "$W/collect.txt" | sort -u > "$W/nodes.txt"
    say "node list: $(wc -l < "$W/nodes.txt") LoFi tests of $(grep -c '^perf_math_matmul\.py::' "$W/collect.txt") collected"
    [ -s "$W/nodes.txt" ] || { echo "[ERROR] no LoFi nodes collected; see $W/collect.txt" >&2; return 1; }
}

# --- measure -----------------------------------------------------------------
measure() {
    for tag in a b; do
        local llk=$W/t_$tag/tt_metal/tt-llk
        say "measuring tree $tag at 5 code positions"
        ( cd "$llk" && PATH="$W/venv/bin:$PATH" CHIP_ARCH=wormhole \
            LLK_PERF_NODE_LIST="$W/nodes.txt" RUNNER_TEMP="$W/tmp-$tag" \
            TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_RUN_TYPES=L1_TO_L1 \
            ./tests/run_llk_perf_placement.sh ) > "$W/measure-$tag.log" 2>&1
        say "  exit $? ; csv: $(find "$llk/tests/python_tests/perf_data/runs" -name '*.csv' 2>/dev/null | wc -l)"
    done
}

# --- one node at a time, to learn which test id carries which config -------------
# The perf CSV names every sweep parameter but not the test that produced it. Measuring
# one node alone ties the two together, so a config the five-position run confirmed can be
# traced back to a test id and then to its ELF.
probe() {
    local n=${PROBE_N:-40} llk=$W/t_a/tt_metal/tt-llk i=0
    : > "$W/probe.tsv"
    while IFS= read -r node && [ "$i" -lt "$n" ]; do
        i=$((i+1))
        rm -rf "$llk/perf_data/runs/probe-$i"
        ( cd "$llk/tests/python_tests" && PATH="$W/venv/bin:$PATH" CHIP_ARCH=wormhole \
            TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_RUN_TYPES=L1_TO_L1 PERF_RUN_TAG="probe-$i" \
            RUNNER_TEMP="$W/probe-tmp" \
            pytest -q --override-ini=log_cli=false -m "perf and not accuracy" --timeout=120 "$node" \
        ) > /dev/null 2>&1
        local csv=$llk/perf_data/runs/probe-$i/perf_math_matmul/perf_math_matmul.csv
        if [ -f "$csv" ]; then
            [ -s "$W/probe.header" ] || head -1 "$csv" > "$W/probe.header"
            awk -v id="$node" 'NR>1 {print id "\t" $0}' "$csv" >> "$W/probe.tsv"
        fi
        rm -rf "$llk/perf_data/runs/probe-$i"
    done < "$W/nodes.txt"
    say "probed $i nodes, $(wc -l < "$W/probe.tsv") rows -> $W/probe.tsv"
}

# --- the ELFs of one test, from both toolchains ----------------------------------
# Compiling a single node leaves exactly that test's kernels in the build root, so the
# pair can be named without guessing which hash belongs to which config. Disassembly is
# what gets compared: an ELF hash embeds the build path.
elfs() {
    local node=${ELF_NODE:?ELF_NODE must name a test id}
    local out=$W/elfs; rm -rf "$out"; mkdir -p "$out"
    for tag in a b; do
        local llk=$W/t_$tag/tt_metal/tt-llk root=$W/elf-tmp-$tag
        rm -rf "$root"; mkdir -p "$root"
        say "compiling the node with tree $tag"
        ( cd "$llk/tests/python_tests" && PATH="$W/venv/bin:$PATH" CHIP_ARCH=wormhole \
            TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_RUN_TYPES=L1_TO_L1 RUNNER_TEMP="$root" \
            pytest -q --override-ini=log_cli=false --compile-producer -m "perf and not accuracy" \
            --timeout=120 "$node" ) > "$W/elf-$tag.log" 2>&1
        local ver; ver=$(cat "$llk/tests/sfpi/sfpi.version")
        local objdump; objdump=$(find "$llk/tests/sfpi" -name '*objdump' -type f | head -1)
        local n=0 e
        while IFS= read -r e; do
            local name; name=${e#"$root"/}; name=${name//\//_}
            cp "$e" "$out/${name%.elf}.sfpi-$ver.elf"
            "$objdump" -d --no-show-raw-insn "$e" \
              | sed -e '1,/Disassembly of section/d' -e 's/^[[:space:]]*[0-9a-f]*:[[:space:]]*//' \
              > "$out/${name%.elf}.sfpi-$ver.asm"
            n=$((n+1))
        done < <(find "$root" -name '*.elf' | sort)
        say "  tree $tag (sfpi $ver): $n ELF(s)"
        find "$root" -name 'build.h' -exec cp {} "$out/build.sfpi-$ver.h" \; 2>/dev/null
    done
    ( cd "$out" && for f in *.sfpi-7.92.0.asm; do
        b=${f%.sfpi-7.92.0.asm}
        [ -f "$b.sfpi-7.93.0.asm" ] || continue
        diff -u "$f" "$b.sfpi-7.93.0.asm" > "$b.diff" || true
        echo "$b: $(grep -c '^[-+][^-+]' "$b.diff") changed instruction line(s)"
      done )
    echo "$node" > "$out/NODE.txt"
    say "wrote $out"
}

pack() {
    rm -f "$W/results.tgz"
    ( cd "$W" && tar czf results.tgz \
        t_a/tt_metal/tt-llk/tests/python_tests/perf_data/runs \
        t_b/tt_metal/tt-llk/tests/python_tests/perf_data/runs \
        nodes.txt measure-a.log measure-b.log 2>/dev/null )
    say "packed $(du -h "$W/results.tgz" | cut -f1) into $W/results.tgz"
}

mkdir -p "$W"
case "$STEP" in
    trees)   trees ;;
    nodes)   venv && nodes ;;
    probe)   probe ;;
    elfs)    elfs ;;
    measure) measure ;;
    pack)    pack ;;
    all)     trees && venv && nodes && measure && pack ;;
    *) echo "unknown step: $STEP" >&2; exit 2 ;;
esac
