#!/bin/bash
# Run the clang-tidy cmake build with a time budget and one automatic retry.
#
# Usage: clang_tidy_build.sh <attempt1-min> <attempt2-min> <margin-min> [cmake --build args...]
#
# Each attempt gets a deadline of (window - margin) minutes. Past it the clang-tidy
# wrapper (cmake/clang-tidy-export-fixes-wrapper.sh) fails fast, so ninja drains while
# in-flight TUs finish and land in ctcache. If an attempt hit its deadline, the build
# is retried once on the same runner: ninja resumes incrementally and ctcache is warm,
# so the retry only pays for what is left. Deadlines only apply when ctcache S3 is
# reachable (CTCACHE_S3_OK=1); otherwise there is nothing to preserve and a single
# attempt runs as before.
#
# Exits with the last cmake exit code and appends CLANG_TIDY_DEADLINE_HIT=<0|1> to
# $GITHUB_ENV (1 = the final attempt still ran out of time).
set -uo pipefail

windows=("$1" "$2")
margin="$3"
shift 3

marker=/tmp/clang-tidy-deadline-hit
summary="${GITHUB_STEP_SUMMARY:-/dev/null}"
rc=0
hit=0

for attempt in 1 2; do
    window="${windows[attempt - 1]}"
    rm -f "$marker"
    if [ "${CTCACHE_S3_OK:-0}" = "1" ]; then
        export CLANG_TIDY_DEADLINE_EPOCH=$(($(date +%s) + (window - margin) * 60))
        export CLANG_TIDY_DEADLINE_MARKER="$marker"
    else
        unset CLANG_TIDY_DEADLINE_EPOCH CLANG_TIDY_DEADLINE_MARKER
    fi

    echo "clang-tidy build attempt ${attempt}/2 (window ${window}m, drain margin ${margin}m)"
    rc=0
    cmake --build --preset clang-tidy-fix-parallel "$@" || rc=$?

    hit=0
    [ -e "$marker" ] && hit=1
    [ "$hit" = 1 ] || break

    if [ "$attempt" = 1 ]; then
        echo "::warning::clang-tidy ran out of time budget on attempt 1. Completed translation units are saved in ctcache; retrying once automatically."
        {
            echo "### ⏱️ Clang Tidy hit its time budget on attempt 1"
            echo "Completed translation units are saved in ctcache. Retrying once automatically."
        } >>"$summary"
    fi
done

if [ "$hit" = 1 ]; then
    echo "::error::clang-tidy ran out of time budget again on the automatic retry. Completed translation units are saved in ctcache; re-run this job to resume."
    {
        echo "### ⏱️ Clang Tidy hit its time budget on the automatic retry"
        echo "Completed translation units are saved in ctcache. Re-run this job to resume from where it stopped."
    } >>"$summary"
fi

echo "CLANG_TIDY_DEADLINE_HIT=${hit}" >>"${GITHUB_ENV:-/dev/null}"
exit "$rc"
