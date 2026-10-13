#!/bin/bash

# Output-tagging helpers shared by the exabox cluster scripts.

# Tag each line with [hostname], adding [HH:MM:SS] only when the line has no
# timestamp of its own so tool logs aren't stamped twice. Ranks prepend a bare
# "[host] " prefix at the source; this keeps that host, adds the time, and passes
# already-fully-tagged lines through unchanged (idempotent under a second pass).
tag_stream() {
    local line host rest
    local esc=$'\x1b'
    local done_re='^\[[^][]*\]\[[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\] '   # already [host][time]
    local rank_re='^\[([^][]*)\] (.*)$'                                # rank's bare [host] prefix
    local ts_re="^(${esc}\[[0-9;]*[a-zA-Z])*[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}"  # leading timestamp, ANSI-tolerant
    local self="${HOSTNAME:-$(hostname)}"
    while IFS= read -r line; do
        if [[ "$line" =~ $done_re ]]; then
            printf '%s\n' "$line"
            continue
        fi
        if [[ "$line" =~ $rank_re ]]; then
            host="${BASH_REMATCH[1]}"
            rest="${BASH_REMATCH[2]}"
        else
            host="$self"
            rest="$line"
        fi
        if [[ "$rest" =~ $ts_re ]]; then
            printf '[%s] %s\n' "$host" "$rest"
        else
            printf '[%s][%(%H:%M:%S)T] %s\n' "$host" -1 "$rest"
        fi
    done
}
