#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# device_side.sh -- first-step triage, device side: liveness and NOC integrity.
#
# host_side.sh stops at the PCIe link; this opens each device and talks to the
# chip.  Five probes per device, cheapest and most passive first:
#
#   1. hung        Driver sysfs ladder: PCI config space, every tt_* attribute,
#                  a double heartbeat sample, then one NOC read to ARC.  Needs
#                  no TLB, so it still works when another process holds them
#                  all.
#   2. info        One-screen inventory: PCI link, IOMMU mode, board type and
#                  id, DRAM training, firmware version, vitals.  Host-side
#                  facts print even when the chip is hung.
#   3. scratch     ARC reset-unit scratch registers.  Raw evidence for later
#                  analysis: postcodes, boot status, telemetry pointers.
#   4. telemetry   Walks the ARC firmware telemetry table and decodes every
#                  tag.  Its structural reads are the liveness check: all ones
#                  means the chip is not answering.
#   5. test noc_sanity
#                  Asks all 204 (Blackhole) or 120 (Wormhole) NOC0 nodes who
#                  they are, across every tile type.  The only probe that
#                  touches nodes other than ARC, so it runs last and is
#                  skipped for a chip that already failed telemetry.
#
# All five are subcommands of the kmd_triage binary, built from
# kmd_triage.cpp by the `kmd_triage` CMake target (see ../CMakeLists.txt).
#
# Everything here opens the device power-aware (O_RDWR | O_APPEND) and never
# issues SET_POWER_STATE, so device power state is left exactly as found.
# Nothing here writes to the device, and no destructive subcommand of the
# kmd_triage binary (reset, nuke, write32) is invoked.
#
# ---------------------------------------------------------------------------
# A NOTE ON ALL ONES, WHICH IS THE SIGNAL EVERYTHING HERE TURNS ON
#
# A 32-bit read of 0xFFFFFFFF has two causes that we currently cannot tell
# apart, and they mean opposite things:
#
#   (a) Someone answered, and the value really is 0xFFFFFFFF.  Takes ~1us.
#       Blackhole firmware stores exactly this in TAG_FAN_SPEED and
#       TAG_FAN_RPM when fan control is disabled, which is every UBB tray in
#       a Galaxy, on healthy silicon.  This is a firmware/state fact.
#
#   (b) Nobody answered.  The root port's Completion Timeout fires (65-210ms
#       on the machines measured so far) and synthesises a completion that
#       reads back as all ones.  This is a link, routing or endpoint fact.
#
# Both leave the same value in the register, so the tools distinguish them by
# *position* rather than by what happened: a structural read (a pointer
# register, a table header, a directory entry) that reads all ones is treated
# as fatal, while a tag value that reads all ones is reported and the dump
# continues.  That inference is right in the common cases and wrong in one:
# a chip that is alive but has garbage in its telemetry pointer register is
# reported as "not answering".
#
# Measuring the latency of each read would settle it directly -- 1us versus
# 100ms is three orders of magnitude -- but that is not implemented yet.  If
# this ambiguity ever produces a misdiagnosis in the field, that is the fix.
# ---------------------------------------------------------------------------
#
# Needs the kmd_triage binary and read-write access to /dev/tenstorrent/*.
#
# Usage:
#   ./device_side.sh -o deviceside-$(hostname -s)-$(date +%Y%m%d-%H%M%S).txt

set -u
set -o pipefail
shopt -s nullglob

DEVICE_SIDE_VERSION=1

# Wall-clock limit per tool run.  A healthy chip finishes each tool in well
# under a second; hitting this limit is itself a finding.
#
# The bound that matters is reads x completion timeout.  scratch issues 72
# reads, info about 100, telemetry about 165; at the 210ms worst case those
# are 15s, 21s and 35s, all inside this limit.  noc_sanity issues 612 on
# Blackhole, which would not fit -- but it stops at the first node that does
# not answer, and it is skipped entirely for a chip telemetry already found
# silent, so the full 612 only happens on a chip that is answering fast.
# Raise this if you run noc_sanity with -k on a badly damaged chip.
TOOL_TIMEOUT=60

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Only --json needs this, so a copy of this script on its own still runs the
# text report; write_json checks before using it.
if [[ -r $script_dir/triage_json.sh ]]; then
	# shellcheck source=triage_json.sh
	source "$script_dir/triage_json.sh"
fi

outfile=""
jsonfile=""
do_noc=1
devices=()

# Absolute path to the kmd_triage binary; filled in by resolve_tool().
TOOL=""

usage() {
	cat <<EOF
Usage: $0 [options] [/dev/tenstorrent/N ...]

With no device arguments, all of /dev/tenstorrent/* is probed.

  -o FILE               write the report to FILE (default: stdout)
  --json FILE           also write machine-readable findings to FILE, for the
                        exabox health check
  --no-noc              skip the NOC sweep, the only probe that touches
                        nodes other than ARC
  -h, --help            this text

The kmd_triage binary is found via \$KMD_TRIAGE_BIN, then \$TT_METAL_HOME's
build tree, then a build tree above this script, then \$PATH.

Exit status: 0 nothing wrong at this level, 1 degraded, 2 hopeless,
3 the script itself failed.
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	-o) outfile=${2:-}; shift 2 || { usage >&2; exit 3; } ;;
	--json) jsonfile=${2:-}; shift 2 || { usage >&2; exit 3; } ;;
	--no-noc) do_noc=0; shift ;;
	-h|--help) usage; exit 0 ;;
	-*) echo "$0: unknown argument '$1'" >&2; usage >&2; exit 3 ;;
	*) devices+=("$1"); shift ;;
	esac
done

# ---------------------------------------------------------------- helpers

# Read a sysfs file, collapsing newlines.  Never fails; missing or unreadable
# files come back as "-".
rd() {
	local v
	v=$(timeout 5 cat -- "$1" 2>/dev/null) || { printf -- '-'; return; }
	[[ -n $v ]] || { printf -- '-'; return; }
	printf '%s' "${v//$'\n'/ }"
}

section() {
	printf '\n'
	printf '================================================================================\n'
	printf ' %s\n' "$1"
	printf '================================================================================\n'
}

subsection() {
	if [[ -n ${2:-} ]]; then
		printf '\n-- %s %s\n' "$1" "$2"
	else
		printf '\n-- %s\n' "$1"
	fi
}

problems=()
problem() { problems+=("$1"); }

arch_of() {
	case "$1" in
	0xb140) printf 'Blackhole' ;;
	0x401e) printf 'Wormhole' ;;
	0xfaca) printf 'Grayskull' ;;
	-)      printf -- '-' ;;
	*)      printf 'unknown(%s)' "$1" ;;
	esac
}

# ---------------------------------------------------------------- the tools

workdir=""

# Locate the kmd_triage binary.  It used to be compiled here at run time;
# it is now the `kmd_triage` CMake target, so this only has to find it.
#
# Not found is exit 3 -- the script's own failure -- and deliberately not a
# finding about the hardware: we learned nothing about the chips.  Build it
# with build_metal.sh, or point KMD_TRIAGE_BIN at one.
resolve_tool() {
	local c
	local -a candidates=()

	[[ -n ${KMD_TRIAGE_BIN:-} ]] && candidates+=("$KMD_TRIAGE_BIN")

	# The health check exports TT_METAL_HOME; honour every build dir name
	# build_metal.sh can produce, newest-typical first.
	if [[ -n ${TT_METAL_HOME:-} ]]; then
		for c in build build_Release build_RelWithDebInfo build_Debug; do
			candidates+=("$TT_METAL_HOME/$c/tools/scaleout/kmd_triage")
		done
	fi

	# Run straight out of a checkout: tools/scaleout/kmd_triage -> root.
	local repo_root
	repo_root=$(cd -- "$script_dir/../../.." && pwd)
	for c in build build_Release build_RelWithDebInfo build_Debug; do
		candidates+=("$repo_root/$c/tools/scaleout/kmd_triage")
	done

	for c in "${candidates[@]}"; do
		if [[ -x $c ]]; then
			TOOL=$(cd -- "$(dirname -- "$c")" && pwd)/$(basename -- "$c")
			return 0
		fi
	done

	# Installed alongside the other scaleout tools.
	if TOOL=$(command -v kmd_triage 2>/dev/null); then
		return 0
	fi

	TOOL=""
	echo "$0: cannot find the kmd_triage binary" >&2
	printf '%s\n' "$0: looked at:" "${candidates[@]/#/  }" >&2
	echo "$0: build it (./build_metal.sh) or set KMD_TRIAGE_BIN" >&2
	return 1
}

# ---------------------------------------------------------------- collection

# Parallel arrays, one entry per probed device.
d_path=(); d_ord=(); d_bdf=(); d_arch=()
d_hung_rc=(); d_info_rc=(); d_scratch_rc=(); d_telem_rc=(); d_noc_rc=()
d_loc=(); d_board=(); d_fw=(); d_aiclk=(); d_temp=(); d_dram=()
d_noc_verdict=(); d_status=()

collect_devices() {
	local d
	if (( ${#devices[@]} == 0 )); then
		# Character devices only: /dev/tenstorrent also contains the
		# by-bdf and by-id symlink directories.
		while IFS= read -r d; do
			[[ -n $d && -c $d ]] && devices+=("$d")
		done < <(printf '%s\n' /dev/tenstorrent/* | sort -V)
	fi
}

# Runs one tool under the timeout, output to a file.  Echoes the exit status:
# the tools' own codes pass through, 124/137 mean the timeout fired.
run_tool() {
	local out=$1
	shift
	timeout "$TOOL_TIMEOUT" "$@" > "$out" 2>&1
	echo $?
}

# One "key: value" line from the info subcommand's output.  The key is the
# whole of the first field including its colon, so a value containing a colon
# (a PCI location, say) comes back intact.
info_field() {
	awk -v k="$2:" '$1 == k { $1 = ""; sub(/^[ \t]+/, ""); print; exit }' "$1" 2>/dev/null
}

probe_devices() {
	local i path ord sys bdf devid v noc_rc nv

	for i in "${!devices[@]}"; do
		path=${devices[$i]}
		ord=${path##*/}
		d_path+=("$path"); d_ord+=("$ord")

		# PCI identity from sysfs, so it is known even for a chip that
		# answers nothing.
		sys="/sys/class/tenstorrent/tenstorrent!$ord/device"
		bdf="-"
		[[ -e $sys ]] && bdf=$(basename "$(readlink -f "$sys")")
		d_bdf+=("$bdf")
		devid=$(rd "$sys/device")
		d_arch+=("$(arch_of "$devid")")

		d_hung_rc+=("$(run_tool "$workdir/$ord.hung"    "$TOOL" hung      -d "$path")")
		d_info_rc+=("$(run_tool "$workdir/$ord.info"    "$TOOL" info      -d "$path")")
		d_scratch_rc+=("$(run_tool "$workdir/$ord.scratch" "$TOOL" scratch -d "$path")")
		d_telem_rc+=("$(run_tool "$workdir/$ord.telem"  "$TOOL" telemetry -d "$path")")

		# The summary columns come from info, which prints one stable
		# "key: value" line per fact.
		# location is a Galaxy-only line; absent on anything else.
		v=$(info_field "$workdir/$ord.info" location);    d_loc+=("${v:--}")
		v=$(info_field "$workdir/$ord.info" board_type);  d_board+=("${v:--}")
		v=$(info_field "$workdir/$ord.info" fw_bundle);   d_fw+=("${v:--}")
		v=$(info_field "$workdir/$ord.info" aiclk_mhz);   d_aiclk+=("${v:--}")
		v=$(info_field "$workdir/$ord.info" asic_temp_c); d_temp+=("${v:--}")
		v=$(info_field "$workdir/$ord.info" dram_status); d_dram+=("${v:--}")

		# The NOC sweep is gated on telemetry: a chip whose structural
		# telemetry reads returned all ones is not answering, and sweeping
		# 200+ nodes of a dead chip can only make things worse.  Telemetry
		# exit 3 means the chip answers but its telemetry is unpublished
		# or malformed, so the sweep is still worth running there.
		noc_rc="-"; nv="-"
		if (( do_noc )); then
			case ${d_telem_rc[$i]} in
			0|3)
				noc_rc=$(run_tool "$workdir/$ord.noc" "$TOOL" test noc_sanity -d "$path")
				nv=$(grep -E '^\[(PASS|FAIL)\]' "$workdir/$ord.noc" | tail -n 1)
				[[ -n $nv ]] || nv="-"
				;;
			esac
		fi
		d_noc_rc+=("$noc_rc")
		d_noc_verdict+=("$nv")
	done
}

# Everything that lands in the problem list is decided here.
#
# The binary's exit codes are per-subcommand; 0 is always success and 1 is
# always "the tool, its invocation or its environment is broken", never a
# finding about the chip.
evaluate() {
	local i status loc

	for i in "${!d_path[@]}"; do
		status=OK
		loc="${d_path[$i]} (${d_bdf[$i]})"

		case ${d_hung_rc[$i]} in
		0) ;;
		2)
			status=DEAD
			problem "[DEAD]    $loc: $(hung_reason "$i"), see section 3" ;;
		3)
			[[ $status == OK ]] && status=FW-SICK
			problem "[FW]      $loc: $(hung_reason "$i"), see section 3" ;;
		124|137)
			status=HUNG
			problem "[HUNG]    $loc: the liveness check did not finish within ${TOOL_TIMEOUT}s; the chip may have wedged the link" ;;
		*)
			[[ $status == OK ]] && status=TOOL-FAIL
			problem "[TOOL]    $loc: hung failed (exit ${d_hung_rc[$i]}), see section 3" ;;
		esac

		case ${d_info_rc[$i]} in
		0) ;;
		2)
			status=DEAD ;;
		3)
			[[ $status == OK ]] && status=FW-BAD ;;
		124|137)
			status=HUNG
			problem "[HUNG]    $loc: info did not finish within ${TOOL_TIMEOUT}s" ;;
		*)
			[[ $status == OK ]] && status=TOOL-FAIL
			problem "[TOOL]    $loc: info failed (exit ${d_info_rc[$i]}), see section 4" ;;
		esac

		case ${d_scratch_rc[$i]} in
		0) ;;
		124|137)
			status=HUNG
			problem "[HUNG]    $loc: scratch did not finish within ${TOOL_TIMEOUT}s; the chip may have wedged the link" ;;
		*)
			[[ $status == OK ]] && status=TOOL-FAIL
			problem "[TOOL]    $loc: scratch failed (exit ${d_scratch_rc[$i]}), see section 5" ;;
		esac

		case ${d_telem_rc[$i]} in
		0) ;;
		2)
			status=DEAD
			problem "[DEAD]    $loc: telemetry read all ones; the chip is not answering and the NOC sweep was skipped" ;;
		3)
			[[ $status == OK ]] && status=FW-BAD
			problem "[FW]      $loc: chip answers but its telemetry is unpublished or malformed, see section 6" ;;
		124|137)
			status=HUNG
			problem "[HUNG]    $loc: telemetry did not finish within ${TOOL_TIMEOUT}s; the chip may have wedged the link" ;;
		*)
			[[ $status == OK ]] && status=TOOL-FAIL
			problem "[TOOL]    $loc: telemetry failed (exit ${d_telem_rc[$i]}), see section 6" ;;
		esac

		case ${d_noc_rc[$i]} in
		-|0) ;;
		2)
			[[ $status == OK ]] && status=NOC-BAD
			problem "[NOC]     $loc: ${d_noc_verdict[$i]#\[FAIL\] }; the chip is alive but misconfigured, see section 7" ;;
		3)
			[[ $status == OK ]] && status=NOC-SILENT
			problem "[NOC]     $loc: ${d_noc_verdict[$i]#\[FAIL\] }; a node stopped answering mid-sweep, see section 7" ;;
		124|137)
			status=HUNG
			problem "[HUNG]    $loc: noc_sanity did not finish within ${TOOL_TIMEOUT}s; the chip may have wedged the link" ;;
		*)
			[[ $status == OK ]] && status=TOOL-FAIL
			problem "[TOOL]    $loc: noc_sanity failed (exit ${d_noc_rc[$i]}), see section 7" ;;
		esac

		d_status+=("$status")
	done
}

# The [FAIL] line the hung subcommand ends on, which says which rung of the
# liveness ladder gave way.
hung_reason() {
	local line
	line=$(grep -E '^\[FAIL\] ' "$workdir/${d_ord[$1]}.hung" | tail -n 1)
	if [[ -n $line ]]; then
		printf '%s' "${line#\[FAIL\] }"
	else
		printf 'liveness check failed'
	fi
}

# ---------------------------------------------------------------- report

report_header() {
	printf '================================================================================\n'
	printf ' TENSTORRENT DEVICE-SIDE TRIAGE -- liveness and NOC integrity\n'
	printf '================================================================================\n'
	printf ' generated    : %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
	printf ' version      : %s\n' "$DEVICE_SIDE_VERSION"
	printf ' host         : %s\n' "$(uname -n)"
	printf ' kernel       : %s\n' "$(uname -srvm)"
	printf ' running as   : uid %s\n' "$(id -u)"
	printf ' tt-kmd       : %s\n' "$(rd /sys/module/tenstorrent/version)"
	printf ' tool         : %s\n' "$TOOL"
	printf ' tool timeout : %ss\n' "$TOOL_TIMEOUT"
	printf ' noc sweep    : %s\n' "$( (( do_noc )) && echo enabled || echo 'skipped (--no-noc)' )"
	printf '\n'
	printf ' Sections, in order:\n'
	printf '   1. VERDICT\n'
	printf '   2. DEVICE SUMMARY\n'
	printf '   3. LIVENESS\n'
	printf '   4. DEVICE INFO\n'
	printf '   5. ARC SCRATCH REGISTERS\n'
	printf '   6. TELEMETRY\n'
	printf '   7. NOC SANITY\n'
}

report_verdict() {
	local i a verdict alive=0

	section "1. VERDICT"

	printf ' devices      : %s\n' "${#d_path[@]}"

	local -A tally=()
	for i in "${!d_path[@]}"; do
		a=${d_arch[$i]}
		tally[$a]=$(( ${tally[$a]:-0} + 1 ))
	done
	for a in "${!tally[@]}"; do
		printf ' %-13s %s\n' "$a" "${tally[$a]}"
	done

	for i in "${!d_path[@]}"; do
		case ${d_status[$i]} in
		DEAD|HUNG) ;;
		*) alive=$((alive + 1)) ;;
		esac
	done

	if (( ${#d_path[@]} == 0 )); then
		verdict=HOPELESS
		problem "[DRIVER]  no /dev/tenstorrent devices; run host_side.sh for the PCIe and driver picture"
	elif (( alive == 0 )); then
		verdict=HOPELESS
	elif (( ${#problems[@]} > 0 )); then
		verdict=DEGRADED
	else
		verdict=PASS
	fi

	printf '\n'
	printf ' DEVICESIDE-VERDICT: %s problems=%s\n' "$verdict" "${#problems[@]}"
	printf '\n'
	if (( ${#problems[@]} == 0 )); then
		if (( do_noc )); then
			printf ' No problems found at this level.  Every chip answered its sysfs\n'
			printf ' telemetry, ARC scratch and telemetry table reads, and every NOC0 node\n'
			printf ' reported correct coordinates.\n'
		else
			printf ' No problems found at this level.  Every chip answered its sysfs\n'
			printf ' telemetry, ARC scratch and telemetry table reads.  The NOC sweep was\n'
			printf ' skipped (--no-noc).\n'
		fi
	else
		printf ' PROBLEMS (%s):\n' "${#problems[@]}"
		printf '   %s\n' "${problems[@]}"
	fi

	case $verdict in
	PASS) exit_status=0 ;;
	DEGRADED) exit_status=1 ;;
	*) exit_status=2 ;;
	esac
}

report_summary() {
	local i noc any_loc=0
	# Field widths carry a precision as well, so that a garbage value cannot
	# push the columns out of alignment for every other row.
	local fmt=' %-4.4s %-12.12s %-9.9s %-16.16s %-12.12s %-6.6s %-7.7s %-14.14s %-4.4s %s\n'
	local locfmt=' %-4.4s %-5.5s %-12.12s %-9.9s %-16.16s %-12.12s %-6.6s %-7.7s %-14.14s %-4.4s %s\n'

	section "2. DEVICE SUMMARY"
	printf '\n Values are decoded from the ARC firmware telemetry table, read over the\n'
	printf ' NOC.  A chip marked DEAD answered nothing, so its telemetry columns are\n'
	printf ' blank.  noc is the node sweep: every NOC0 node is asked who it is.\n'

	# The loc column only appears when something can fill it, so a single-card
	# host does not get a column of dashes.
	for i in "${!d_path[@]}"; do
		[[ ${d_loc[$i]} != - ]] && any_loc=1
	done
	if (( any_loc )); then
		printf '\n loc is the Galaxy physical position, u<ubb>c<chip>, decoded from the PCI\n'
		printf ' bus number.  Device ordinals come from driver probe order and mean nothing\n'
		printf ' physical; loc is what tells you which tray to pull.\n'
	fi
	printf '\n'

	if (( any_loc )); then
		printf "$locfmt" dev loc BDF arch board fw_bundle aiclk temp dram noc status
		printf "$locfmt" ---- ----- ------------ --------- ---------------- ------------ ------ ------- -------------- ---- ------
	else
		printf "$fmt" dev BDF arch board fw_bundle aiclk temp dram noc status
		printf "$fmt" ---- ------------ --------- ---------------- ------------ ------ ------- -------------- ---- ------
	fi

	for i in "${!d_path[@]}"; do
		noc=${d_noc_verdict[$i]}
		case $noc in
		"[PASS]"*) noc=PASS ;;
		"[FAIL]"*) noc=FAIL ;;
		esac
		if (( any_loc )); then
			printf "$locfmt" \
				"${d_ord[$i]}" "${d_loc[$i]}" "${d_bdf[$i]}" "${d_arch[$i]}" \
				"${d_board[$i]}" "${d_fw[$i]}" "${d_aiclk[$i]}" "${d_temp[$i]}" \
				"${d_dram[$i]}" "$noc" "${d_status[$i]}"
		else
			printf "$fmt" \
				"${d_ord[$i]}" "${d_bdf[$i]}" "${d_arch[$i]}" \
				"${d_board[$i]}" "${d_fw[$i]}" "${d_aiclk[$i]}" "${d_temp[$i]}" \
				"${d_dram[$i]}" "$noc" "${d_status[$i]}"
		fi
	done
}

report_tool_outputs() {
	local sec=$1 title=$2 suffix=$3 skipnote=$4
	local i f

	section "$sec. $title"

	for i in "${!d_path[@]}"; do
		subsection "device ${d_ord[$i]}" "(${d_path[$i]}, ${d_bdf[$i]})"
		f="$workdir/${d_ord[$i]}.$suffix"
		if [[ -s $f ]]; then
			cat "$f"
		elif [[ -e $f ]]; then
			printf ' (no output)\n'
		else
			printf ' %s\n' "$skipnote"
		fi
	done
}

# ------------------------------------------------------------------- json
#
# The text report above is for a human reading a ticket; this is for the
# exabox health check, which folds these checks into its own report.  The
# shape and the helpers live in triage_json.sh.
#
# One check per probe rather than one per device, so the check names are the
# same on a 1-chip host and a 32-chip Galaxy -- the dashboard keys its
# routing on the check name, so a name that varies with device count would
# fragment the history.  The offending devices go in details and data.

# Roll one probe's per-device exit codes into a single check.
#
# The severity split is the same judgement evaluate() makes for the text
# report: what the chip says is a finding, what the tool says about itself is
# lost coverage.  So a chip that wedges the link (124/137) is a FAIL, while
# an unexpected exit code is only a WARN -- it tells us the probe broke, not
# that the hardware is bad.
#
# Exit 2 and 3 are per-subcommand, so KIND says which reading applies:
#
#   chip  liveness, info, arc_scratch, telemetry.  2 is a chip that answers
#         nothing (FAIL); 3 is a chip that answers but whose firmware state
#         is wrong (WARN).
#   noc   noc_sanity.  Both codes are the sweep's findings about the grid, so
#         both are a FAIL: 2 is a node that answered wrongly, 3 is a node
#         that went silent.  3 is the worse of the two, which is why this
#         cannot share the chip reading -- that would file NOC silence as a
#         firmware warning and let it pass the health check.
#
# json_add_probe NAME IP KIND RC...
json_add_probe() {
	local name=$1 ip=$2 kind=$3
	shift 3

	case $kind in
	chip|noc) ;;
	*)
		echo "$0: json_add_probe: unknown kind '$kind'" >&2
		return 1 ;;
	esac
	local rcs=("$@")
	local i rc ok=0 skipped=0 status details
	local -a fail=() warn=()

	for i in "${!rcs[@]}"; do
		rc=${rcs[$i]}
		case $rc in
		-)       (( skipped++ )) ;;
		0)       (( ok++ )) ;;
		124|137) fail+=("${d_bdf[$i]} (timeout after ${TOOL_TIMEOUT}s)") ;;
		2)
			if [[ $kind == noc ]]; then
				fail+=("${d_bdf[$i]} (node answered wrongly)")
			else
				fail+=("${d_bdf[$i]} (not answering)")
			fi ;;
		3)
			if [[ $kind == noc ]]; then
				fail+=("${d_bdf[$i]} (node stopped answering mid-sweep)")
			else
				warn+=("${d_bdf[$i]} (firmware state)")
			fi ;;
		*)       warn+=("${d_bdf[$i]} (probe exit $rc)") ;;
		esac
	done

	if (( ${#rcs[@]} == 0 )); then
		json_add "$name" SKIP "no devices probed" "$ip"
		return
	fi
	if (( skipped == ${#rcs[@]} )); then
		json_add "$name" SKIP "skipped on all ${#rcs[@]} device(s)" "$ip"
		return
	fi

	if (( ${#fail[@]} )); then
		status=FAIL
	elif (( ${#warn[@]} )); then
		status=WARN
	else
		status=PASS
	fi

	details="$ok/${#rcs[@]} ok"
	(( skipped )) && details+=", $skipped skipped"
	(( ${#fail[@]} )) && details+="; FAIL: $(json_join "${fail[@]}")"
	(( ${#warn[@]} )) && details+="; WARN: $(json_join "${warn[@]}")"

	json_add "$name" "$status" "$details" "$ip" \
		"$(printf '{"devices": %s, "ok": %s, "skipped": %s, "failed": %s, "warned": %s}' \
			"${#rcs[@]}" "$ok" "$skipped" \
			"$(json_str_array "${fail[@]+"${fail[@]}"}")" \
			"$(json_str_array "${warn[@]+"${warn[@]}"}")")"
}

build_json() {
	local n=${#d_path[@]}

	# Nothing to probe is the one case the per-probe checks cannot express:
	# they would all be SKIP, which reads as "not asked for" rather than
	# "asked, and there was nothing there".
	if (( n == 0 )); then
		json_add deviceside_devices_present FAIL \
			"no /dev/tenstorrent devices; see host_side.sh for the PCIe and driver picture" \
			other
	else
		json_add deviceside_devices_present PASS "$n device(s) probed" other \
			"$(printf '{"devices": %s}' "$n")"
	fi

	json_add_probe deviceside_liveness    asic  chip "${d_hung_rc[@]+"${d_hung_rc[@]}"}"
	json_add_probe deviceside_info        board chip "${d_info_rc[@]+"${d_info_rc[@]}"}"
	json_add_probe deviceside_arc_scratch fw    chip "${d_scratch_rc[@]+"${d_scratch_rc[@]}"}"
	json_add_probe deviceside_telemetry   fw    chip "${d_telem_rc[@]+"${d_telem_rc[@]}"}"
	json_add_probe deviceside_noc_sanity  asic  noc  "${d_noc_rc[@]+"${d_noc_rc[@]}"}"
}

write_json() {
	if ! declare -F json_write >/dev/null; then
		echo "$0: --json needs triage_json.sh next to this script" >&2
		return 1
	fi
	build_json
	json_write device_side "$DEVICE_SIDE_VERSION" "$1"
}

# ---------------------------------------------------------------- main

main() {
	collect_devices
	probe_devices
	evaluate

	report_header
	report_verdict
	report_summary
	report_tool_outputs 3 "LIVENESS (hung)" hung "not run"
	report_tool_outputs 4 "DEVICE INFO (info)" info "not run"
	report_tool_outputs 5 "ARC SCRATCH REGISTERS (scratch)" scratch "not run"
	report_tool_outputs 6 "TELEMETRY (telemetry)" telem "not run"
	report_tool_outputs 7 "NOC SANITY (noc_sanity)" noc \
		"skipped: telemetry says this chip is not answering (or --no-noc)"
	printf '\n-- end of report --\n'
}

exit_status=3

# Sourcing the script defines the functions without running anything, which
# is how the pieces get tested away from hardware.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
	return 0
fi

workdir=$(mktemp -d) || exit 3
trap 'rm -rf "$workdir"' EXIT

resolve_tool || exit 3

if [[ -n $outfile ]]; then
	main > "$outfile" 2>&1
	# The verdict is buried in the file, so repeat it for whoever ran this.
	grep -E '^ DEVICESIDE-VERDICT:' "$outfile"
	sed -n '/^ PROBLEMS (/,/^$/p' "$outfile"
	printf 'report written to %s\n' "$outfile"
else
	main
fi

# After main, so the probe results are in the arrays. main is a function, not
# a subshell, so its redirection above does not hide them.
if [[ -n $jsonfile ]]; then
	write_json "$jsonfile" || {
		echo "$0: could not write $jsonfile" >&2
		exit 3
	}
fi

exit "$exit_status"
