#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# host_side.sh -- first-step triage, host side: host, PCIe and driver state.
#
# Passive.  Nothing here opens /dev/tenstorrent, allocates a TLB, or touches
# the NOC, so it is safe to run against a wedged system.  The only device
# contact is reading the driver's read-only telemetry sysfs attributes;
# --no-device-reads suppresses even that.
#
# The report is ordered so that the interesting material is at the top: a
# verdict, a list of specific problems, then per-device tables.  The 32
# lspci -vvv dumps are last.
#
# Run as root.  Without root, lspci output is truncated, the debugfs mappings
# are unavailable, and the kernel log may be unreadable.
#
# Usage:
#   sudo ./host_side.sh -o hostside-$(hostname -s)-$(date +%Y%m%d-%H%M%S).txt

set -u
set -o pipefail
shopt -s nullglob

HOST_SIDE_VERSION=1

TT_VENDOR=0x1e52
PCI_DEVICE_ID_GRAYSKULL=0xfaca
PCI_DEVICE_ID_WORMHOLE=0x401e
PCI_DEVICE_ID_BLACKHOLE=0xb140

# See enumerate.c: a Galaxy is identified by subsystem device ID, and the PCI
# bus number encodes physical position.  The high nibble selects the UBB, the
# low nibble is the 1-based chip index on that UBB.
PCI_SUBSYSTEM_ID_GALAXY_WH=0x0035
PCI_SUBSYSTEM_ID_GALAXY_BH=0x0047
GALAXY_CHIP_COUNT=32
WH_GALAXY_UBB_PREFIX=(0xC 0x8 0x0 0x4)
BH_GALAXY_UBB_PREFIX=(0x0 0x4 0xC 0x8)

# On Galaxy, exactly one chip per UBB is wired x8; the rest are x1.  That chip
# is the one with a 6 in the low nibble of its bus number.
GALAXY_X8_CHIP=6

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Only --json needs this, so a copy of this script on its own still runs the
# text report; write_json checks before using it.
if [[ -r $script_dir/triage_json.sh ]]; then
	# shellcheck source=triage_json.sh
	source "$script_dir/triage_json.sh"
fi

outfile=""
jsonfile=""
expect_chips=""
do_device_reads=1
do_lspci=1

usage() {
	cat <<EOF
Usage: $0 [options]

  -o FILE             write the report to FILE (default: stdout)
  --json FILE         also write machine-readable findings to FILE, for the
                      exabox health check
  --expect N          expect N Tenstorrent chips (default: 32 on Galaxy,
                      otherwise no expectation)
  --no-device-reads   skip the telemetry sysfs attributes, which are the only
                      part of this script that contacts the device
  --no-lspci          skip the full lspci -vvv dumps
  -h, --help          this text

Exit status: 0 nothing wrong at this level, 1 degraded, 2 hopeless,
3 the script itself failed.
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	-o) outfile=${2:-}; shift 2 || { usage >&2; exit 3; } ;;
	--json) jsonfile=${2:-}; shift 2 || { usage >&2; exit 3; } ;;
	--expect) expect_chips=${2:-}; shift 2 || { usage >&2; exit 3; } ;;
	--no-device-reads) do_device_reads=0; shift ;;
	--no-lspci) do_lspci=0; shift ;;
	-h|--help) usage; exit 0 ;;
	*) echo "$0: unknown argument '$1'" >&2; usage >&2; exit 3 ;;
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

# PCIe link speed string ("32.0 GT/s PCIe") to generation number.
gen_of() {
	case "$1" in
	*64.0*) echo 6 ;;
	*32.0*) echo 5 ;;
	*16.0*) echo 4 ;;
	*8.0*)  echo 3 ;;
	*5.0*)  echo 2 ;;
	*2.5*)  echo 1 ;;
	*)      echo 0 ;;
	esac
}

gen_str() { [[ $1 == 0 ]] && printf -- '-' || printf 'Gen%s' "$1"; }

arch_of() {
	case "$1" in
	"$PCI_DEVICE_ID_BLACKHOLE") printf 'Blackhole' ;;
	"$PCI_DEVICE_ID_WORMHOLE")  printf 'Wormhole' ;;
	"$PCI_DEVICE_ID_GRAYSKULL") printf 'Grayskull' ;;
	*) printf 'unknown(%s)' "$1" ;;
	esac
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

# ---------------------------------------------------------------- collection

# Parallel arrays, one entry per Tenstorrent PCI function, in BDF order.
d_bdf=(); d_path=(); d_devid=(); d_ssid=(); d_ord=(); d_driver=()
d_curgen=(); d_maxgen=(); d_curwidth=(); d_maxwidth=()
d_capgen=(); d_expwidth=(); d_loc=(); d_bridge=(); d_badbars=()
d_status=(); d_pwr=()
d_card=(); d_serial=(); d_fw=(); d_aiclk=(); d_hb1=(); d_hb2=(); d_trips=()
d_unreachable=(); d_onesfield=()
d_aer_c=(); d_aer_n=(); d_aer_f=()

is_galaxy=0
galaxy_arch=""

# Upstream bridges, deduplicated: on Galaxy many chips share one.
tt_bridges=()

collect_devices() {
	local path bdf vendor devid ssid c

	for path in /sys/bus/pci/devices/*; do
		[[ -r $path/vendor ]] || continue
		vendor=$(rd "$path/vendor")
		[[ $vendor == "$TT_VENDOR" ]] || continue

		bdf=${path##*/}
		devid=$(rd "$path/device")
		ssid=$(rd "$path/subsystem_device")

		d_bdf+=("$bdf")
		d_path+=("$path")
		d_devid+=("$devid")
		d_ssid+=("$ssid")

		case "$ssid" in
		"$PCI_SUBSYSTEM_ID_GALAXY_BH") is_galaxy=1; galaxy_arch=bh ;;
		"$PCI_SUBSYSTEM_ID_GALAXY_WH") is_galaxy=1; galaxy_arch=wh ;;
		esac

		local ord="-"
		for c in "$path"/tenstorrent/tenstorrent!*; do
			ord=${c##*!}
		done
		d_ord+=("$ord")

		local drv="none"
		[[ -L $path/driver ]] && drv=$(basename "$(readlink -f "$path/driver")")
		d_driver+=("$drv")

		d_pwr+=("$(rd "$path/power_state")")

		local cs ms cw mw
		cs=$(rd "$path/current_link_speed"); ms=$(rd "$path/max_link_speed")
		cw=$(rd "$path/current_link_width"); mw=$(rd "$path/max_link_width")
		d_curgen+=("$(gen_of "$cs")"); d_maxgen+=("$(gen_of "$ms")")
		[[ $cw == -  ]] && cw=0
		[[ $mw == -  ]] && mw=0
		d_curwidth+=("$cw"); d_maxwidth+=("$mw")

		# The upstream bridge is the other end of the link; the achievable
		# speed is the lesser of the two ends' capabilities.  This is what
		# lets us tell a Gen4 system from a Gen5 system that trained down.
		local parent="" bridge="-"
		parent=$(readlink -f "$path/..")
		if [[ -n $parent && -r $parent/vendor ]]; then
			bridge=${parent##*/}
		fi
		d_bridge+=("$bridge")

		if [[ $bridge != - ]]; then
			local known=0 b
			for b in ${tt_bridges[@]+"${tt_bridges[@]}"}; do
				[[ $b == "$bridge" ]] && known=1
			done
			(( known )) || tt_bridges+=("$bridge")
		fi

		local capgen bgen
		capgen=$(gen_of "$ms")
		if [[ $bridge != - ]]; then
			bgen=$(gen_of "$(rd "/sys/bus/pci/devices/$bridge/max_link_speed")")
			(( bgen != 0 && bgen < capgen )) && capgen=$bgen
		fi
		d_capgen+=("$capgen")

		d_badbars+=("$(unassigned_bars "$path")")
		d_aer_c+=("$(aer_total "$path/aer_dev_correctable" TOTAL_ERR_COR)")
		d_aer_n+=("$(aer_total "$path/aer_dev_nonfatal" TOTAL_ERR_NONFATAL)")
		d_aer_f+=("$(aer_total "$path/aer_dev_fatal" TOTAL_ERR_FATAL)")
	done
}

# A BAR with a size but no assigned address means the kernel could not place
# it.  On Galaxy this shows up after hotplug; see tools/fix-tt-hotplug-bars.
unassigned_bars() {
	local i=0 start end flags out=""
	[[ -r $1/resource ]] || return
	while read -r start end flags; do
		(( i >= 6 )) && break
		if [[ $flags != 0x0000000000000000 && $start == 0x0000000000000000 ]]; then
			out+="BAR$i "
		fi
		i=$((i + 1))
	done < "$1/resource"
	printf '%s' "${out% }"
}

aer_total() {
	[[ -r $1 ]] || { printf -- '-'; return; }
	awk -v k="$2" '$1==k { print $2; found=1 } END { if (!found) print "-" }' "$1" 2>/dev/null
}

# Galaxy physical location from the bus number.
galaxy_loc() {
	local bdf=$1 bus high low ubb prefix
	bus=$(( 16#${bdf:5:2} ))
	high=$(( bus >> 4 ))
	low=$(( bus & 0x0f ))
	(( low >= 1 && low <= 8 )) || { printf -- '-'; return; }

	local -a table
	if [[ $galaxy_arch == bh ]]; then
		table=("${BH_GALAXY_UBB_PREFIX[@]}")
	else
		table=("${WH_GALAXY_UBB_PREFIX[@]}")
	fi

	for ubb in 0 1 2 3; do
		prefix=$(( ${table[$ubb]} ))
		if (( prefix == high )); then
			printf 'u%dc%d' "$((ubb + 1))" "$low"
			return
		fi
	done
	printf -- '-'
}

galaxy_expected_width() {
	local bdf=$1 bus low
	bus=$(( 16#${bdf:5:2} ))
	low=$(( bus & 0x0f ))
	if (( low == GALAXY_X8_CHIP )); then echo 8; else echo 1; fi
}

# A chip that has stopped answering does not fail the read, it returns all
# ones, and each attribute then decodes that into its own flavour of garbage:
# FFFFFFFFFFFFFFFF for the serial, 255.255.255.255 for a version, 4294967295
# for anything counted.  Bare 255 is deliberately not on this list, because an
# idle AICLK of 255 MHz is a real reading.
is_all_ones() {
	local v=$1
	case "$v" in
	65535|4294967295|18446744073709551615) return 0 ;;
	esac
	[[ $v =~ ^(0[xX])?[fF]{4,16}$ ]] && return 0
	[[ $v =~ ^255(\.255)+$ ]] && return 0
	return 1
}

collect_telemetry() {
	local i sys attr val

	for i in "${!d_bdf[@]}"; do
		d_card+=("-"); d_serial+=("-"); d_fw+=("-"); d_aiclk+=("-")
		d_hb1+=("-"); d_hb2+=("-"); d_trips+=("-")
		d_unreachable+=(0); d_onesfield+=("")

		[[ ${d_ord[$i]} == - ]] && continue
		(( do_device_reads )) || continue

		sys="/sys/class/tenstorrent/tenstorrent!${d_ord[$i]}"
		[[ -d $sys ]] || continue

		# tt_serial goes first because it is the one attribute whose
		# all-ones value cannot be mistaken for a real reading.  Once
		# anything reads all ones, stop: the rest would be garbage dressed
		# up as plausible values.
		for attr in tt_serial tt_card_type tt_fw_bundle_ver tt_aiclk \
			    tt_therm_trip_count tt_heartbeat; do
			val=$(rd "$sys/$attr")
			if is_all_ones "$val"; then
				d_unreachable[$i]=1
				d_onesfield[$i]="$attr=$val"
				break
			fi
			case $attr in
			tt_serial)           d_serial[$i]=$val ;;
			tt_card_type)        d_card[$i]=$val ;;
			tt_fw_bundle_ver)    d_fw[$i]=$val ;;
			tt_aiclk)            d_aiclk[$i]=$val ;;
			tt_therm_trip_count) d_trips[$i]=$val ;;
			tt_heartbeat)        d_hb1[$i]=$val ;;
			esac
		done
	done
}

# Second heartbeat sample.  Whether the counter advances is worth more than
# its value, but only for chips we could read in the first place.
resample_heartbeat() {
	local i sys
	local -a candidates=()

	(( do_device_reads )) || return
	for i in "${!d_bdf[@]}"; do
		(( d_unreachable[i] )) && continue
		[[ ${d_hb1[$i]} == - ]] && continue
		candidates+=("$i")
	done
	(( ${#candidates[@]} )) || return

	sleep 0.5
	for i in "${candidates[@]}"; do
		sys="/sys/class/tenstorrent/tenstorrent!${d_ord[$i]}"
		d_hb2[$i]=$(rd "$sys/tt_heartbeat")
	done
}

aer_problem() {
	local what=$1 kind=$2 count=$3
	[[ $count =~ ^[0-9]+$ ]] || return 0
	(( count > 0 )) || return 0
	problem "[AER]     $what: $count $kind PCIe error(s) logged"
}

# Errors on the link frequently land on the bridge rather than the endpoint,
# so a chip can look clean while the port feeding it is counting failures.
evaluate_bridges() {
	local b
	for b in ${tt_bridges[@]+"${tt_bridges[@]}"}; do
		aer_problem "bridge $b" correctable \
			"$(aer_total "/sys/bus/pci/devices/$b/aer_dev_correctable" TOTAL_ERR_COR)"
		aer_problem "bridge $b" non-fatal \
			"$(aer_total "/sys/bus/pci/devices/$b/aer_dev_nonfatal" TOTAL_ERR_NONFATAL)"
		aer_problem "bridge $b" fatal \
			"$(aer_total "/sys/bus/pci/devices/$b/aer_dev_fatal" TOTAL_ERR_FATAL)"
	done
}

# Work out what each link should look like and flag the ones that do not
# match.  Everything that lands in the problem list is decided here.
evaluate() {
	local i loc expw status

	for i in "${!d_bdf[@]}"; do
		if (( is_galaxy )); then
			loc=$(galaxy_loc "${d_bdf[$i]}")
			expw=$(galaxy_expected_width "${d_bdf[$i]}")
		else
			loc="-"
			expw=${d_maxwidth[$i]}
			if [[ ${d_bridge[$i]} != - ]]; then
				local bw
				bw=$(rd "/sys/bus/pci/devices/${d_bridge[$i]}/max_link_width")
				[[ $bw =~ ^[0-9]+$ ]] && (( bw > 0 && bw < expw )) && expw=$bw
			fi
		fi
		d_loc+=("$loc")
		d_expwidth+=("$expw")

		status="OK"

		if [[ ${d_driver[$i]} != tenstorrent ]]; then
			status="NO-DRIVER"
			problem "[DRIVER]  ${d_bdf[$i]} ($loc): tt-kmd is not bound, driver is '${d_driver[$i]}'"
		fi

		if [[ ${d_curgen[$i]} == 0 ]]; then
			# No point comparing widths; there is nothing to compare against.
			status="LINK-DOWN"
			problem "[LINK]    ${d_bdf[$i]} ($loc): link speed unreadable, the device may be gone"
		else
			if (( d_curgen[i] < d_capgen[i] )); then
				[[ $status == OK ]] && status="LINK-SLOW"
				problem "[LINK]    ${d_bdf[$i]} ($loc): trained at $(gen_str "${d_curgen[$i]}"), link is capable of $(gen_str "${d_capgen[$i]}")"
			fi
			if (( d_curwidth[i] < expw )); then
				[[ $status == OK ]] && status="LINK-NARROW"
				problem "[LINK]    ${d_bdf[$i]} ($loc): trained at x${d_curwidth[$i]}, expected x$expw"
			elif (( d_curwidth[i] > expw )); then
				problem "[NOTE]    ${d_bdf[$i]} ($loc): trained at x${d_curwidth[$i]}, wider than the expected x$expw"
			fi
		fi

		if [[ -n ${d_badbars[$i]} ]]; then
			[[ $status == OK ]] && status="BAR-UNASSIGNED"
			problem "[BAR]     ${d_bdf[$i]} ($loc): unassigned ${d_badbars[$i]}, the kernel could not place the window"
		fi

		aer_problem "${d_bdf[$i]} ($loc)" correctable "${d_aer_c[$i]}"
		aer_problem "${d_bdf[$i]} ($loc)" non-fatal   "${d_aer_n[$i]}"
		aer_problem "${d_bdf[$i]} ($loc)" fatal       "${d_aer_f[$i]}"

		# All ones means the read did not reach the chip.  That is all it
		# means: it is not evidence about the ARC firmware, which we simply
		# cannot see from here.  Section 3 spells out the distinction.
		if (( d_unreachable[i] )); then
			status="ALL-ONES"
			problem "[DEAD]    ${d_bdf[$i]} ($loc): telemetry reads all ones (${d_onesfield[$i]}); the chip is not answering and no further telemetry was read"
		elif [[ ${d_hb1[$i]} != - && ${d_hb1[$i]} == "${d_hb2[$i]}" ]]; then
			[[ $status == OK ]] && status="FW-STALLED"
			problem "[FW]      ${d_bdf[$i]} ($loc): tt_heartbeat did not advance over 0.5s (still ${d_hb1[$i]}); the chip answers but ARC may be stalled"
		fi

		d_status+=("$status")
	done
}

# ---------------------------------------------------------------- report

report_header() {
	printf '================================================================================\n'
	printf ' TENSTORRENT HOST-SIDE TRIAGE -- host, PCIe and driver state\n'
	printf '================================================================================\n'
	printf ' generated    : %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
	printf ' version      : %s\n' "$HOST_SIDE_VERSION"
	printf ' host         : %s\n' "$(uname -n)"
	printf ' uptime       : %s\n' "$(uptime -p 2>/dev/null || rd /proc/uptime)"
	printf ' kernel       : %s\n' "$(uname -srvm)"
	printf ' distro       : %s\n' "$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
	printf ' running as   : uid %s%s\n' "$(id -u)" \
		"$( (( EUID == 0 )) || echo '  *** NOT ROOT: lspci is truncated, debugfs and dmesg may be missing ***')"
	printf ' tt-kmd       : %s\n' "$(rd /sys/module/tenstorrent/version)"
	printf ' tt-kmd loaded: %s\n' "$([[ -d /sys/module/tenstorrent ]] && echo yes || echo NO)"
	printf '\n'
	printf ' Sections, in order:\n'
	printf '   1. VERDICT\n'
	printf '   2. PCIe LINK STATUS\n'
	printf '   3. DEVICE AND FIRMWARE STATE\n'
	printf '   4. PCIe ERROR COUNTERS\n'
	printf '   5. OPEN FILE DESCRIPTORS\n'
	printf '   6. DRIVER MAPPINGS\n'
	printf '   7. HOST CONFIGURATION\n'
	printf '   8. KERNEL LOG\n'
	printf '   9. LSPCI DETAIL\n'
}

report_verdict() {
	local found=${#d_bdf[@]} bound=0 i verdict

	for i in "${!d_bdf[@]}"; do
		[[ ${d_ord[$i]} != - ]] && bound=$((bound + 1))
	done

	section "1. VERDICT"

	printf ' system       : %s\n' \
		"$( (( is_galaxy )) && echo "Galaxy (${galaxy_arch^^})" || echo "not a Galaxy" )"
	printf ' chips on bus : %s' "$found"
	if [[ -n $expect_chips ]]; then
		printf '  (expected %s)' "$expect_chips"
	fi
	printf '\n'

	local -A tally=()
	for i in "${!d_bdf[@]}"; do
		local a
		a=$(arch_of "${d_devid[$i]}")
		tally[$a]=$(( ${tally[$a]:-0} + 1 ))
	done
	local a
	for a in "${!tally[@]}"; do
		printf ' %-13s %s\n' "$a" "${tally[$a]}"
	done

	printf ' chips bound  : %s  (have a /dev/tenstorrent node)\n' "$bound"

	if [[ -n $expect_chips && $found -lt $expect_chips ]]; then
		problem "[MISSING] only $found of $expect_chips expected chips are present on the PCI bus"
	fi

	if (( found == 0 )); then
		verdict=HOPELESS
	elif [[ ! -d /sys/module/tenstorrent ]]; then
		verdict=HOPELESS
		problem "[DRIVER]  the tenstorrent module is not loaded"
	elif (( bound == 0 )); then
		verdict=HOPELESS
	elif (( ${#problems[@]} > 0 )); then
		verdict=DEGRADED
	else
		verdict=PASS
	fi

	printf '\n'
	printf ' HOSTSIDE-VERDICT: %s problems=%s\n' "$verdict" "${#problems[@]}"
	printf '\n'
	if (( ${#problems[@]} == 0 )); then
		printf ' No problems found at this level.  This script only inspects host, PCIe\n'
		printf ' and driver state; it does not prove that the chips can do any work.\n'
		printf ' Run device_side.sh next.\n'
	else
		printf ' PROBLEMS (%s):\n' "${#problems[@]}"
		printf '   %s\n' "${problems[@]}"
	fi

	# Coverage we did not get, kept out of the problem list and out of the
	# verdict: it says nothing about the hardware, only about the invocation.
	if (( klog_records == 0 )); then
		printf '\n NOT CHECKED: the kernel log was unreadable, so PCIe, IOMMU and\n'
		printf ' machine-check faults were not examined.\n'
		if (( EUID == 0 )); then
			# Already root, so do not send the reader after sudo.
			printf ' Running as root, so this is not a permission problem:\n'
			printf ' neither dmesg nor journalctl returned any records.\n'
		else
			printf ' Re-run with sudo for those.\n'
		fi
	fi

	case $verdict in
	PASS) exit_status=0 ;;
	DEGRADED) exit_status=1 ;;
	*) exit_status=2 ;;
	esac
}

report_links() {
	local i fmt=' %-4.4s %-12.12s %-6.6s %-12.12s %-11.11s %-11.11s %-11.11s %s\n'

	section "2. PCIe LINK STATUS"
	printf '\n'
	printf ' current is what the link trained to.  capable is the lesser of what the\n'
	printf ' endpoint and its upstream bridge advertise, so a Gen4 host reads Gen4 here\n'
	printf ' and is not flagged.  On Galaxy the expected width comes from the bus number:\n'
	printf ' the chip with a %s in the low nibble is x8, the rest are x1.\n\n' "$GALAXY_X8_CHIP"

	printf "$fmt" dev BDF loc bridge current capable expected status
	printf "$fmt" ---- ------------ ------ ------------ ----------- ----------- ----------- ------

	for i in "${!d_bdf[@]}"; do
		printf "$fmt" \
			"${d_ord[$i]}" "${d_bdf[$i]}" "${d_loc[$i]}" "${d_bridge[$i]}" \
			"$(gen_str "${d_curgen[$i]}") x${d_curwidth[$i]}" \
			"$(gen_str "${d_capgen[$i]}") x${d_maxwidth[$i]}" \
			"$(gen_str "${d_capgen[$i]}") x${d_expwidth[$i]}" \
			"${d_status[$i]}"
	done
}

report_devices() {
	# Field widths carry a precision as well, so that a garbage value cannot
	# push the columns out of alignment for every other row.
	local i hb
	local fmt=' %-4.4s %-12.12s %-8.8s %-18.18s %-15.15s %-10.10s %-16.16s %-6.6s %s\n'

	section "3. DEVICE AND FIRMWARE STATE"
	if (( ! do_device_reads )); then
		printf '\n Skipped: --no-device-reads.\n'
		return
	fi

	printf '\n Telemetry is produced by the ARC firmware and read through a BAR.  A chip\n'
	printf ' that has stopped answering returns all ones instead of failing the read, so\n'
	printf ' the first attribute that reads all ones stops the rest for that device.\n'
	printf '\n'
	printf ' Be careful what you conclude from that.  Config space is answered by the\n'
	printf ' PCIe controller, not the chip, so a device can enumerate, train a full\n'
	printf ' width link and show clean AER counters while everything behind the BAR\n'
	printf ' reads all ones.  It means the read did not reach the chip.  It is not\n'
	printf ' evidence about the state of the ARC firmware, which cannot be seen at all\n'
	printf ' in that condition.\n'
	printf '\n heartbeat is sampled twice, 0.5s apart; a value that does not advance is\n'
	printf ' reported as STUCK.\n\n'

	printf "$fmt" dev BDF card serial fw_bundle aiclk heartbeat trips pwr
	printf "$fmt" ---- ------------ -------- ------------------ --------------- ---------- ---------------- ------ ---

	for i in "${!d_bdf[@]}"; do
		if (( d_unreachable[i] )); then
			printf ' %-4.4s %-12.12s %s\n' "${d_ord[$i]}" "${d_bdf[$i]}" \
				"ALL ONES from ${d_onesfield[$i]}, remaining telemetry not read"
			continue
		fi
		if [[ ${d_hb1[$i]} == - ]]; then
			hb="-"
		elif [[ ${d_hb1[$i]} == "${d_hb2[$i]}" ]]; then
			hb="${d_hb2[$i]} STUCK"
		else
			hb="${d_hb2[$i]}"
		fi
		printf "$fmt" \
			"${d_ord[$i]}" "${d_bdf[$i]}" "${d_card[$i]}" "${d_serial[$i]}" \
			"${d_fw[$i]}" "${d_aiclk[$i]}" "$hb" "${d_trips[$i]}" "${d_pwr[$i]}"
	done
}

report_aer() {
	local i f b

	section "4. PCIe ERROR COUNTERS"
	printf '\n Only nonzero counters are listed.  Errors are frequently logged on the\n'
	printf ' upstream bridge rather than the endpoint, so bridges are included.\n'

	subsection "endpoints"
	local any=0
	for i in "${!d_bdf[@]}"; do
		for f in correctable nonfatal fatal; do
			local path="${d_path[$i]}/aer_dev_$f" line
			[[ -r $path ]] || continue
			line=$(awk '{ for (j = 1; j < NF; j += 2) if ($(j+1) != 0) printf "%s=%s ", $j, $(j+1) }' "$path" 2>/dev/null)
			[[ -n $line ]] || continue
			printf ' %-12s %-11s %s\n' "${d_bdf[$i]}" "$f" "$line"
			any=1
		done
	done
	(( any )) || printf ' all zero\n'

	subsection "upstream bridges"
	any=0
	for b in ${tt_bridges[@]+"${tt_bridges[@]}"}; do
		for f in correctable nonfatal fatal; do
			local path="/sys/bus/pci/devices/$b/aer_dev_$f" line
			[[ -r $path ]] || continue
			line=$(awk '{ for (j = 1; j < NF; j += 2) if ($(j+1) != 0) printf "%s=%s ", $j, $(j+1) }' "$path" 2>/dev/null)
			[[ -n $line ]] || continue
			printf ' %-12s %-11s %s\n' "$b" "$f" "$line"
			any=1
		done
	done
	(( any )) || printf ' all zero\n'
}

report_pids() {
	local i ord pid comm

	section "5. OPEN FILE DESCRIPTORS"
	printf '\n /proc/driver/tenstorrent/<N>/pids: who currently holds each device.\n'
	printf ' A device with an open fd must not be reset.\n'

	for i in "${!d_bdf[@]}"; do
		ord=${d_ord[$i]}
		[[ $ord == - ]] && continue
		subsection "device $ord" "(${d_bdf[$i]})"
		local pidfile="/proc/driver/tenstorrent/$ord/pids"
		if [[ ! -r $pidfile ]]; then
			printf ' %s: unreadable\n' "$pidfile"
			continue
		fi
		local n=0
		while read -r pid; do
			[[ -n $pid ]] || continue
			comm=$(rd "/proc/$pid/comm")
			printf ' pid %-8s %-20s %s\n' "$pid" "$comm" "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
			n=$((n + 1))
		done < "$pidfile"
		(( n )) || printf ' no open file descriptors\n'
	done
}

report_mappings() {
	local i ord path

	section "6. DRIVER MAPPINGS"
	printf '\n /sys/kernel/debug/tenstorrent/<N>/mappings: pinned pages, DMA buffers,\n'
	printf ' iATU regions, TLB windows and BAR mappings, per open fd.\n'

	if [[ ! -d /sys/kernel/debug/tenstorrent ]]; then
		printf '\n debugfs is not mounted or the driver has no debugfs directory.\n'
		return
	fi

	for i in "${!d_bdf[@]}"; do
		ord=${d_ord[$i]}
		[[ $ord == - ]] && continue
		path="/sys/kernel/debug/tenstorrent/$ord/mappings"
		subsection "device $ord" "(${d_bdf[$i]})"
		if [[ -r $path ]]; then
			timeout 10 cat "$path" 2>&1
		else
			printf ' %s: unreadable (root required)\n' "$path"
		fi
	done
}

report_host() {
	section "7. HOST CONFIGURATION"

	subsection "kernel command line"
	rd /proc/cmdline; printf '\n'

	subsection "IOMMU"
	local -a g=(/sys/kernel/iommu_groups/*)
	local -a u=(/sys/class/iommu/*)
	printf ' iommu groups : %s\n' "${#g[@]}"
	printf ' iommu units  : %s\n' "${#u[@]}"
	local i
	for i in "${!d_bdf[@]}"; do
		local grp
		grp=$(readlink -f "${d_path[$i]}/iommu_group" 2>/dev/null)
		printf ' %-12s group %s  dma_mask=%s\n' "${d_bdf[$i]}" "${grp##*/}" "$(rd "${d_path[$i]}/dma_mask_bits")"
	done

	subsection "module parameters"
	local p
	for p in /sys/module/tenstorrent/parameters/*; do
		printf ' %-24s %s\n' "${p##*/}" "$(rd "$p")"
	done

	subsection "hugepages and memory"
	grep -E '^(MemTotal|MemAvailable|HugePages_|Hugepagesize)' /proc/meminfo 2>/dev/null

	subsection "kernel taint"
	printf ' /proc/sys/kernel/tainted = %s\n' "$(rd /proc/sys/kernel/tainted)"

	subsection "loaded tenstorrent modules"
	grep -E '^tenstorrent' /proc/modules 2>/dev/null || printf ' none\n'
}

# Lines worth waking someone up for.  Deliberately narrow: matching a bare
# "AER" picks up "AER: enabled with IRQ 125" on every boot, and every root
# port logs one of those.
KLOG_FAULT_RE='AER:.*(Corrected|Uncorrected|Multiple|Device recovery|can.t recover|aer_status|error status)'
KLOG_FAULT_RE+='|\(First\)'
KLOG_FAULT_RE+='|PCIe Bus Error|[Mm]achine [Cc]heck|\bmce:|Hardware Error'
KLOG_FAULT_RE+='|DMAR:.*[Ff]ault|AMD-Vi:.*IO_PAGE_FAULT'
KLOG_FAULT_RE+='|DPC: *containment|Link Down'
KLOG_FAULT_RE+='|Completion Timeout|Unsupported Request|reset_link'
# Matched case-sensitively: "SATA link down" is not our problem, but the
# PCIe hotplug driver's "Slot(0): Link Down" is.

klog_cmd=(dmesg -T)

klog_fault_count=0
klog_records=0

# How many actual records a log source yields: lines that are neither blank
# nor a journalctl "-- ... --" placeholder.  Counting records rather than
# trusting an exit status is the whole point; see select_klog.
count_klog_records() {
	local n
	n=$(timeout 30 "$@" 2>/dev/null | grep -cvE '^(-- |$)')
	[[ $n =~ ^[0-9]+$ ]] || n=0
	printf '%s' "$n"
}

# Choose the log source, and keep the record count it gave us so the report
# can tell "the log is clean" from "the log was not readable".
#
# The choice is not made on journalctl's exit status: it answers a caller that
# cannot read the kernel journal with "-- No entries --" on stdout and status
# 0.  Selecting on that swapped a working dmesg for a source that yields
# nothing whenever journalctl is installed without a journal to read -- a
# container being the usual case -- and the script then told a root user the
# kernel log was unreadable and to re-run under sudo.  So require records, and
# keep dmesg when journalctl cannot produce any.
select_klog() {
	local candidate=(journalctl -k --no-pager -o short-precise)

	if command -v journalctl >/dev/null 2>&1; then
		klog_records=$(count_klog_records "${candidate[@]}")
		if (( klog_records > 0 )); then
			klog_cmd=("${candidate[@]}")
			return 0
		fi
	fi

	klog_cmd=(dmesg -T)
	klog_records=$(count_klog_records "${klog_cmd[@]}")
	return 0
}

check_klog() {
	# No records means nothing was read, which is not the same as nothing
	# being wrong, so it is reported as coverage we did not get rather than
	# as "none".  It is not a problem, though: it says nothing about the
	# hardware, only about how the script was invoked, so it must not move
	# the verdict.
	(( klog_records > 0 )) || return 0

	klog_fault_count=$(timeout 30 "${klog_cmd[@]}" 2>/dev/null | grep -Ec "$KLOG_FAULT_RE")
	[[ $klog_fault_count =~ ^[0-9]+$ ]] || klog_fault_count=0
	(( klog_fault_count > 0 )) &&
		problem "[KLOG]    $klog_fault_count PCIe/IOMMU/MCE fault line(s) in the kernel log, see section 8"
	return 0
}

report_klog() {
	section "8. KERNEL LOG"
	printf '\n source: %s\n' "${klog_cmd[*]}"

	subsection "PCIe, IOMMU and machine check faults ($klog_fault_count lines)"
	if (( klog_records == 0 )); then
		printf ' No records read, so nothing was checked -- not evidence of a clean\n'
		if (( EUID == 0 )); then
			printf ' log.  Already root, so not a permission problem: neither\n'
			printf ' dmesg nor journalctl returned records.\n'
		else
			printf ' log.  Re-run as root.\n'
		fi
	elif (( klog_fault_count > 0 )); then
		timeout 30 "${klog_cmd[@]}" 2>/dev/null | grep -E "$KLOG_FAULT_RE" | tail -n 400
	else
		printf ' none\n'
	fi

	subsection "tenstorrent driver"
	timeout 30 "${klog_cmd[@]}" 2>/dev/null |
		grep -Ei 'tenstorrent|tt-kmd' | tail -n 200 || printf ' unavailable\n'

	subsection "last 200 lines, unfiltered"
	timeout 30 "${klog_cmd[@]}" 2>/dev/null | tail -n 200 || printf ' unavailable\n'
}

report_lspci() {
	section "9. LSPCI DETAIL"

	if (( ! do_lspci )); then
		printf '\n Skipped: --no-lspci.\n'
		return
	fi
	if ! command -v lspci >/dev/null 2>&1; then
		printf '\n lspci is not installed.\n'
		return
	fi

	subsection "topology"
	timeout 30 lspci -tv 2>/dev/null

	local i b
	for i in "${!d_bdf[@]}"; do
		subsection "endpoint ${d_bdf[$i]}" "(device ${d_ord[$i]}, ${d_loc[$i]})"
		timeout 30 lspci -vvv -s "${d_bdf[$i]}" 2>&1
	done

	for b in ${tt_bridges[@]+"${tt_bridges[@]}"}; do
		subsection "upstream bridge $b"
		timeout 30 lspci -vvv -s "$b" 2>&1
	done
}

# ------------------------------------------------------------------- json
#
# The text report above is for a human reading a ticket; this is for the
# exabox health check.  The shape and the helpers live in triage_json.sh.
#
# One check per class of finding rather than one per device, so the check
# names do not change with chip count -- the dashboard keys its routing on
# the name.  The offending devices go in details and data.

# Roll a per-device predicate into one check.
#
# Every caller answers the same question for one device at a time -- is this
# one bad, and if so how do I say it -- so the counting, the worst-wins
# precedence and the details line are done once here.
#
# json_add_devices NAME IP FAIL_OR_WARN DESC_FN
#
# DESC_FN is called with a device index and echoes a description when that
# device is a finding, or nothing when it is fine.
json_add_devices() {
	local name=$1 ip=$2 severity=$3 desc_fn=$4
	local i d ok=0 status details
	local -a bad=()

	for i in "${!d_bdf[@]}"; do
		d=$($desc_fn "$i")
		if [[ -n $d ]]; then
			bad+=("$d")
		else
			(( ok++ ))
		fi
	done

	if (( ${#d_bdf[@]} == 0 )); then
		json_add "$name" SKIP "no Tenstorrent devices on the bus" "$ip"
		return
	fi

	if (( ${#bad[@]} )); then
		status=$severity
		details="$ok/${#d_bdf[@]} ok; $(json_join "${bad[@]}")"
	else
		status=PASS
		details="$ok/${#d_bdf[@]} ok"
	fi

	json_add "$name" "$status" "$details" "$ip" \
		"$(printf '{"devices": %s, "ok": %s, "findings": %s}' \
			"${#d_bdf[@]}" "$ok" "$(json_str_array "${bad[@]+"${bad[@]}"}")")"
}

# One predicate per class of finding.  Each mirrors the corresponding branch
# of evaluate(), which is what fills the human-readable problem list.
_json_desc_driver() {
	[[ ${d_driver[$1]} != tenstorrent ]] &&
		printf '%s: driver is "%s"' "${d_bdf[$1]}" "${d_driver[$1]}"
}

_json_desc_link() {
	local i=$1
	if [[ ${d_curgen[$i]} == 0 ]]; then
		printf '%s: link speed unreadable, the device may be gone' "${d_bdf[$i]}"
	elif (( d_curgen[i] < d_capgen[i] )); then
		printf '%s: trained at %s, capable of %s' "${d_bdf[$i]}" \
			"$(gen_str "${d_curgen[$i]}")" "$(gen_str "${d_capgen[$i]}")"
	elif (( d_curwidth[i] < d_expwidth[i] )); then
		printf '%s: trained at x%s, expected x%s' "${d_bdf[$i]}" \
			"${d_curwidth[$i]}" "${d_expwidth[$i]}"
	fi
}

_json_desc_bar() {
	[[ -n ${d_badbars[$1]} ]] &&
		printf '%s: unassigned %s' "${d_bdf[$1]}" "${d_badbars[$1]}"
}

# Endpoint AER only. The bridge counters evaluate_bridges() looks at are not
# per-device, so they are reported as their own check below.
_json_desc_aer() {
	local i=$1 parts=""
	local k v
	for k in c:correctable n:non-fatal f:fatal; do
		case ${k%%:*} in
		c) v=${d_aer_c[$i]} ;;
		n) v=${d_aer_n[$i]} ;;
		f) v=${d_aer_f[$i]} ;;
		esac
		[[ $v =~ ^[0-9]+$ ]] && (( v > 0 )) && parts+="${parts:+, }$v ${k#*:}"
	done
	[[ -n $parts ]] && printf '%s: %s' "${d_bdf[$i]}" "$parts"
}

_json_desc_reachable() {
	(( d_unreachable[$1] )) &&
		printf '%s: reads all ones (%s)' "${d_bdf[$1]}" "${d_onesfield[$1]}"
}

_json_desc_heartbeat() {
	local i=$1
	[[ ${d_hb1[$i]} != - && ${d_hb1[$i]} == "${d_hb2[$i]}" ]] &&
		printf '%s: tt_heartbeat stuck at %s over 0.5s' "${d_bdf[$i]}" "${d_hb1[$i]}"
}

# Bridge AER, which is not attributable to a single endpoint.
json_add_bridge_aer() {
	local b v kind ok=0 seen=0
	local -a bad=()

	for b in ${tt_bridges[@]+"${tt_bridges[@]}"}; do
		local parts=""
		(( seen++ ))
		for kind in correctable:aer_dev_correctable:TOTAL_ERR_COR \
			nonfatal:aer_dev_nonfatal:TOTAL_ERR_NONFATAL \
			fatal:aer_dev_fatal:TOTAL_ERR_FATAL; do
			local label=${kind%%:*} rest=${kind#*:}
			v=$(aer_total "/sys/bus/pci/devices/$b/${rest%%:*}" "${rest#*:}")
			[[ $v =~ ^[0-9]+$ ]] && (( v > 0 )) && parts+="${parts:+, }$v $label"
		done
		if [[ -n $parts ]]; then
			bad+=("bridge $b: $parts")
		else
			(( ok++ ))
		fi
	done

	if (( seen == 0 )); then
		json_add hostside_bridge_aer SKIP "no upstream bridges identified" pcie
	elif (( ${#bad[@]} )); then
		json_add hostside_bridge_aer WARN \
			"$ok/$seen clean; $(json_join "${bad[@]}")" pcie \
			"$(printf '{"bridges": %s, "ok": %s, "findings": %s}' \
				"$seen" "$ok" "$(json_str_array "${bad[@]}")")"
	else
		json_add hostside_bridge_aer PASS "$ok/$seen bridge(s) clean" pcie \
			"$(printf '{"bridges": %s, "ok": %s}' "$seen" "$ok")"
	fi
}

build_json() {
	local found=${#d_bdf[@]} bound=0 i

	for i in "${!d_bdf[@]}"; do
		[[ ${d_ord[$i]} != - ]] && (( bound++ ))
	done

	# Chip count first: everything else is per-device, so a missing chip
	# would otherwise show up only as a smaller denominator.
	if (( found == 0 )); then
		json_add hostside_chips_present FAIL "no Tenstorrent chips on the PCI bus" pcie
	elif [[ -n $expect_chips ]] && (( found < expect_chips )); then
		json_add hostside_chips_present FAIL \
			"only $found of $expect_chips expected chips on the bus" pcie \
			"$(printf '{"found": %s, "expected": %s, "bound": %s}' \
				"$found" "$expect_chips" "$bound")"
	else
		json_add hostside_chips_present PASS \
			"$found chip(s) on the bus, $bound bound to tt-kmd" pcie \
			"$(printf '{"found": %s, "expected": "%s", "bound": %s}' \
				"$found" "${expect_chips:-none}" "$bound")"
	fi

	if [[ ! -d /sys/module/tenstorrent ]]; then
		json_add hostside_driver_loaded FAIL "the tenstorrent module is not loaded" other
	else
		json_add hostside_driver_loaded PASS \
			"tt-kmd $(rd /sys/module/tenstorrent/version)" other
	fi

	json_add_devices hostside_driver_bound  other FAIL _json_desc_driver
	json_add_devices hostside_pcie_link     pcie  FAIL _json_desc_link
	json_add_devices hostside_pcie_bar      pcie  FAIL _json_desc_bar
	# AER counters are cumulative since boot and correctable errors are
	# recoverable by design, so they are a WARN: worth a look, not a reason
	# to fail a unit on their own.
	json_add_devices hostside_pcie_aer      pcie  WARN _json_desc_aer
	json_add_devices hostside_chip_reachable asic FAIL _json_desc_reachable
	json_add_devices hostside_arc_heartbeat fw    WARN _json_desc_heartbeat
	json_add_bridge_aer

	# Coverage we did not get. A SKIP rather than a PASS: reporting a clean
	# log we could not read would be worse than admitting we did not look.
	if (( klog_records == 0 )); then
		json_add hostside_kernel_log SKIP \
			"kernel log unreadable$( (( EUID == 0 )) || printf ' (not root)' ); PCIe, IOMMU and machine-check faults not examined" \
			other
	elif (( klog_fault_count > 0 )); then
 		json_add hostside_kernel_log WARN \
 			"$klog_fault_count PCIe/IOMMU/MCE fault line(s) found in $klog_records record(s)" other \
 			"$(printf '{"records": %s, "faults": %s}' "$klog_records" "$klog_fault_count")"
	else
		json_add hostside_kernel_log PASS \
			"$klog_records kernel log record(s) examined" other \
			"$(printf '{"records": %s}' "$klog_records")"
	fi
}

write_json() {
	if ! declare -F json_write >/dev/null; then
		echo "$0: --json needs triage_json.sh next to this script" >&2
		return 1
	fi
	build_json
	json_write host_side "$HOST_SIDE_VERSION" "$1"
}

# ---------------------------------------------------------------- main

main() {
	collect_devices

	if [[ -z $expect_chips ]] && (( is_galaxy )); then
		expect_chips=$GALAXY_CHIP_COUNT
	fi

	select_klog
	collect_telemetry
	resample_heartbeat
	evaluate
	evaluate_bridges
	check_klog

	report_header
	report_verdict
	report_links
	report_devices
	report_aer
	report_pids
	report_mappings
	report_host
	report_klog
	report_lspci

	printf '\n-- end of report --\n'
}

exit_status=3

# Sourcing the script defines the functions without running anything, which is
# how the Galaxy bus-number decoding gets tested away from Galaxy hardware.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
	return 0
fi

if [[ -n $outfile ]]; then
	main > "$outfile" 2>&1
	# The verdict is buried in the file, so repeat it for whoever ran this.
	grep -E '^ HOSTSIDE-VERDICT:' "$outfile"
	sed -n '/^ PROBLEMS (/,/^$/p' "$outfile"
	sed -n '/^ NOT CHECKED:/,/^$/p' "$outfile"
	printf 'report written to %s\n' "$outfile"
else
	main
fi

# After main, so the collected facts are in the arrays. main is a function,
# not a subshell, so its redirection above does not hide them.
if [[ -n $jsonfile ]]; then
	write_json "$jsonfile" || {
		echo "$0: could not write $jsonfile" >&2
		exit 3
	}
fi

exit "$exit_status"
