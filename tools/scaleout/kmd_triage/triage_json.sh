#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# triage_json.sh -- shared machine-readable output for the triage scripts.
#
# Sourced by host_side.sh and device_side.sh; not runnable on its own.  Both
# write the same shape, which the exabox health check ingests (see
# HEALTH_CHECK.md, "Triage phase"):
#
#   {"tool": "...", "version": N,
#    "checks": [{"name", "status", "details", "ip", "data"}]}
#
# status is PASS / WARN / FAIL / SKIP.  ip is one of the health check's IP
# groups -- board, pcie, gddr, eth, asic, fw, thermal, other -- and anything
# it does not recognise is folded into "other" on ingest.
#
# This lives in one file rather than being copied into both scripts because
# it is a cross-repo contract: two copies would let the shape drift, and the
# consumer would have no way to tell which one it was looking at.  The rollup
# to a single verdict is deliberately not here -- the health check computes
# that from the checks, so the precedence lives in one place.

# Accumulated by json_add, emitted by json_write.
json_checks=()

# Escape a string for use as a JSON string body.
json_escape() {
	local s=$1
	s=${s//\\/\\\\}
	s=${s//\"/\\\"}
	s=${s//$'\t'/\\t}
	s=${s//$'\n'/\\n}
	s=${s//$'\r'/\\r}
	printf '%s' "$s"
}

# json_add NAME STATUS DETAILS IP [DATA_JSON]
#
# DATA_JSON is inserted verbatim and must be a JSON object; it defaults to an
# empty one.  Callers build it with printf, so keep it simple.
json_add() {
	local data=${5:-}
	[[ -n $data ]] || data="{}"

	json_checks+=("$(printf '{"name": "%s", "status": "%s", "details": "%s", "ip": "%s", "data": %s}' \
		"$(json_escape "$1")" "$2" "$(json_escape "$3")" "$4" "$data")")
}

# A JSON array of strings from the remaining arguments.  No arguments is an
# empty array, which is why callers can pass a possibly-unset array.
json_str_array() {
	local out="[" i=0 v
	for v in "$@"; do
		(( i++ )) && out+=", "
		out+="\"$(json_escape "$v")\""
	done
	printf '%s]' "$out"
}

# Join the arguments with ", " for a human-readable details line.  Not via
# IFS: "${arr[*]}" joins on the first character of IFS only, so IFS=", "
# silently produces a comma with no space.
json_join() {
	local out="" v
	for v in "$@"; do
		out+="${out:+, }$v"
	done
	printf '%s' "$out"
}

# json_write TOOL VERSION OUTFILE
json_write() {
	local tool=$1 version=$2 out=$3
	local i last=$(( ${#json_checks[@]} - 1 ))

	{
		printf '{\n'
		printf '  "tool": "%s",\n' "$(json_escape "$tool")"
		printf '  "version": %s,\n' "$version"
		printf '  "checks": [\n'
		# json_checks is always assigned at file scope, so it is set even
		# when empty and this is safe under set -u.
		for i in "${!json_checks[@]}"; do
			printf '    %s%s\n' "${json_checks[$i]}" \
				"$( (( i < last )) && printf ',' )"
		done
		printf '  ]\n'
		printf '}\n'
	} > "$out"
}
