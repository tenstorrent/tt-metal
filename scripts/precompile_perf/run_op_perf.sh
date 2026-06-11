#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Per-op A/B perf runner for up-front precompile. For each op in perf_set.sh it
# runs two arms over the SAME selection, on the SAME device, from COLD caches:
#
#   Arm B (precompile, "the new thing")  — runs FIRST. Warms the JIT cache
#         up-front (collect + parallel compile) on the real device, then runs
#         the suite warm. Phases (collect / compile / warm-run) are split out.
#   Arm A (cold, baseline)               — runs SECOND. Old inline path; kernels
#         compile serially during the run. One e2e number.
#
# Both arms start from a freshly DELETED JIT cache and a freshly DELETED ccache
# (both isolated to throwaway dirs so your shared caches are never touched).
# Speedup = A_total / B_total. See METHODOLOGY.md for the full rationale and the
# phase definitions; OP_PERF_CATALOG.md for the op table and result template.
#
# RUN THIS INSIDE tmux. The full sweep is long (cold compiles + warm runs over
# every op, x main and nightly) and must survive SSH disconnects:
#     tmux new -s perf
#     scripts/precompile_perf/run_op_perf.sh both 2>&1 | tee /tmp/perf_run.log
#     # detach: Ctrl-b d   reattach: tmux attach -t perf
#
# Usage:
#   scripts/precompile_perf/run_op_perf.sh [main|nightly|both] [op_name ...]
#     no args         -> main suite, all ops
#     "both"          -> main then nightly, all ops
#     trailing names  -> restrict to those ops (e.g. "main conv2d matmul")

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/perf_set.sh"

# --- args ---
SUITES_ARG="${1:-main}"; shift || true
OP_FILTER=("$@")   # empty = all
case "$SUITES_ARG" in
    main)    SUITES=(main) ;;
    nightly) SUITES=(nightly) ;;
    both)    SUITES=(main nightly) ;;
    *) echo "first arg must be main|nightly|both (got '$SUITES_ARG')" >&2; exit 2 ;;
esac

# --- tmux nudge (not fatal; the sweep is long) ---
if [[ -z "${TMUX:-}" && -z "${PERF_ALLOW_NO_TMUX:-}" ]]; then
    echo "WARNING: not inside tmux. This sweep is long and will die on disconnect." >&2
    echo "         Start 'tmux new -s perf' first, or set PERF_ALLOW_NO_TMUX=1 to override." >&2
    exit 3
fi

# --- isolated, always-cold caches (throwaway dirs; your real caches untouched) ---
export TT_METAL_CACHE="${TT_METAL_CACHE:-/tmp/perf_jit_cache}"
export CCACHE_DIR="${CCACHE_DIR:-/tmp/perf_ccache}"
export TT_METAL_CCACHE_KERNEL_SUPPORT=1   # ccache ON (matches deployment) ...
reset_caches() {                          # ... but DELETED before each arm -> cold start
    rm -rf "$TT_METAL_CACHE"
    ccache -C >/dev/null 2>&1 || true
}

STAMP="$(date +%Y%m%d_%H%M%S 2>/dev/null || echo run)"
OUT_DIR="/tmp/perf_${STAMP}"
mkdir -p "$OUT_DIR"
RESULTS_TSV="$OUT_DIR/results.tsv"
printf 'op\tsuite\tprograms\tcold_total_s\tprecompile_total_s\tcollect_s\tcompile_s\twarmrun_s\tspeedup\tnote\n' > "$RESULTS_TSV"
echo "Results dir: $OUT_DIR" >&2

want_op() {  # honor the optional op filter
    [[ ${#OP_FILTER[@]} -eq 0 ]] && return 0
    local o; for o in "${OP_FILTER[@]}"; do [[ "$o" == "$1" ]] && return 0; done
    return 1
}

# extract "(<N>s total" from SAFE_PYTEST_TOTAL_RUNTIME
grab_total()   { grep -oE '\(([0-9]+)s total' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9]+'; }
# extract warmup (collect+compile) seconds from run_safe's "✓ warmup complete in Ns"
grab_warmup()  { grep -oE 'warmup complete in ([0-9]+)s' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9]+'; }
# the collect log path run_safe prints ("Log: /tmp/precompile_collect_NNN.log")
grab_clog()    { grep -oE '/tmp/precompile_collect_[0-9]+\.log' "$1" 2>/dev/null | tail -1; }

run_arm() {  # $1=label $2=logfile ; rest=run_safe args. Resets caches first.
    local label="$1" log="$2"; shift 2
    reset_caches
    echo ">>> [$label] scripts/run_safe_pytest.sh $*" >&2
    scripts/run_safe_pytest.sh "$@" > "$log" 2>&1
}

preflight_trace() {  # warn if a selected path carries trace markers not covered by the -k filter
    local kf="$1"; shift
    [[ "$kf" != "-" ]] && return 0   # an explicit filter is assumed to handle it
    local hits
    hits="$(grep -rlE "$PERF_TRACE_MARKERS" "$@" 2>/dev/null | head -5)"
    [[ -n "$hits" ]] && echo "  ! pre-flight: trace markers found in selection (no -k filter set):" >&2 \
        && echo "$hits" | sed 's/^/      /' >&2
}

for suite in "${SUITES[@]}"; do
    base="$PERF_MAIN_BASE"; [[ "$suite" == nightly ]] && base="$PERF_NIGHTLY_BASE"
    while IFS=$'\t' read -r op main_t nightly_t kfilter; do
        [[ -z "$op" || "$op" == \#* ]] && continue
        want_op "$op" || continue
        tgt="$main_t"; [[ "$suite" == nightly ]] && tgt="$nightly_t"
        if [[ "$tgt" == "-" ]]; then
            echo "=== $op [$suite] — no target in this tree, skipping ===" >&2
            printf '%s\t%s\t-\t-\t-\t-\t-\t-\t-\tno target in tree\n' "$op" "$suite" >> "$RESULTS_TSV"
            continue
        fi
        # resolve paths relative to the tree base
        paths=(); for p in $tgt; do paths+=("$base/$p"); done
        kargs=(); [[ "$kfilter" != "-" ]] && kargs=(-k "$kfilter")
        echo "=== $op [$suite] -> ${paths[*]} ${kargs[*]:-} ===" >&2
        preflight_trace "$kfilter" "${paths[@]}"

        blog="$OUT_DIR/${op}_${suite}_precompile.log"
        alog="$OUT_DIR/${op}_${suite}_cold.log"

        # Arm B (precompile) FIRST
        run_arm "B precompile" "$blog" --precompile --run-all "${paths[@]}" "${kargs[@]}"
        # Arm A (cold) SECOND
        run_arm "A cold" "$alog" --run-all "${paths[@]}" "${kargs[@]}"

        # --- attribute phases ---
        b_total="$(grab_total "$blog")"; a_total="$(grab_total "$alog")"
        warmup="$(grab_warmup "$blog")"
        note="-"
        compile_s="-"; programs="-"; collect_s="-"; warmrun_s="-"
        if grep -q '✗ warmup' "$blog" 2>/dev/null; then
            note="precompile degraded to COLD (warmup failed) — see log"
        else
            clog="$(grab_clog "$blog")"
            if [[ -n "$clog" && -f "$clog" ]]; then
                read -r programs compile_s < <(grep -oE 'compiled [0-9]+ programs in [0-9.]+s' "$clog" 2>/dev/null | tail -1 \
                    | sed -E 's/compiled ([0-9]+) programs in ([0-9.]+)s/\1 \2/')
            fi
            [[ -n "${warmup:-}" && -n "${compile_s:-}" && "$compile_s" != "-" ]] && \
                collect_s="$(awk -v w="$warmup" -v c="$compile_s" 'BEGIN{printf "%.1f", (w-c<0?0:w-c)}')"
            [[ -n "${b_total:-}" && -n "${warmup:-}" ]] && \
                warmrun_s="$(awk -v t="$b_total" -v w="$warmup" 'BEGIN{printf "%d", (t-w<0?0:t-w)}')"
        fi
        speedup="-"
        [[ -n "${a_total:-}" && -n "${b_total:-}" && "${b_total:-0}" -gt 0 ]] && \
            speedup="$(awk -v a="$a_total" -v b="$b_total" 'BEGIN{printf "%.2f", a/b}')"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$op" "$suite" "${programs:--}" "${a_total:--}" "${b_total:--}" \
            "${collect_s:--}" "${compile_s:--}" "${warmrun_s:--}" "$speedup" "$note" >> "$RESULTS_TSV"
        echo "    -> programs=${programs:--} cold=${a_total:--}s precompile=${b_total:--}s (collect=${collect_s:--} compile=${compile_s:--} warmrun=${warmrun_s:--}) speedup=${speedup}" >&2
    done < <(perf_ops)
done

echo "" >&2
echo "============ RESULTS ($RESULTS_TSV) ============" >&2
column -t -s $'\t' "$RESULTS_TSV" >&2
echo "" >&2
echo "Per-op logs in $OUT_DIR/ (one *_precompile.log + *_cold.log per op/suite)." >&2
