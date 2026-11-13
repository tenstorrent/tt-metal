# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/stubhost.sh: run.sh --stub. Sourced by run.sh before anything else prints. It builds
# a fake healthy 32-chip Blackhole galaxy in a temp dir (fake /sys tree, /dev/tenstorrent nodes, stub tt-smi, ipmitool,
# lspci, setpci, sudo, dmesg and a stub device python) and puts it first on PATH, so the whole run.sh flow runs with
# no device, no driver, no sudo and no network (the offline bundle's CI check, tools/bundle). Nothing real is opened,
# reset or power-cycled. It proves the code paths run, never the hardware: a loud WARNING is the first output line,
# it is repeated in the summary and the RESULT line says STUB (it can never pass for a real run).
# The stub sudo runs only stub tools and refuses every other command. The temp dir is removed on exit.
STUB_WARN="WARNING: STUB MODE (--stub): fake devices, nothing real was tested. This run proves the code paths only, NEVER the hardware."
echo "$STUB_WARN"
STUB_W=$(mktemp -d "${TMPDIR:-/tmp}/traytest-stub.XXXXXX")
st_exit_cmd() { local t; t=$(trap -p EXIT); [ -n "$t" ] || return 0; eval "set -- ${t#trap }"; [ "$1" = -- ] && shift; printf '%s; ' "$1"; }
# only this shell removes the tree: a subshell TERMed before its own trap - EXIT (hb_stop right after hb_start, as --stub
# skips setup) runs the inherited TERM trap, whose exit runs this EXIT trap there (bash 5.1; 9/10 loaded smokes
# lost the tree 18 s in: 0/32 chips answer, real ps and tt-smi on PATH, rc 6)
trap "$(st_exit_cmd)[ \"\$BASHPID\" != $BASHPID ] || rm -rf -- \"\${STUB_W:?}\"" EXIT
st_b=$STUB_W/bin
mkdir -p "$st_b" "$STUB_W/sys/module/tenstorrent" "$STUB_W/sys/class/tenstorrent" "$STUB_W/sys/bus/pci/devices" \
  "$STUB_W/dev/tenstorrent" "$STUB_W/dev/ipmi0" "$STUB_W/proc" "$STUB_W/state"
echo 2.11.0 > "$STUB_W/sys/module/tenstorrent/version"
st_i=0
for st_base in 0x00 0x40 0xc0 0x80; do for st_k in 1 2 3 4 5 6 7 8; do   # buses 01-08 tray 1, 41-48 tray 2, c1-c8 tray 3, 81-88 tray 4
  st_bus=$(printf '%02x' $(( st_base + st_k ))); st_rp=$STUB_W/sys/devices/pci0000:00/0000:00:$(printf '%02x' $st_i).0; st_d=$st_rp/0000:$st_bus:00.0
  mkdir -p "$st_d/tenstorrent/tenstorrent!$st_i" "$st_d/hwmon/hwmon$st_i" "$STUB_W/sys/class/tenstorrent/tenstorrent!$st_i"
  echo 0x1e52 > "$st_d/vendor"; printf '\x52\x1e' > "$st_d/config"; echo "32.0 GT/s PCIe" > "$st_d/current_link_speed"; echo 4 > "$st_d/current_link_width"
  printf 'RxErr 0\nBadTLP 0\nTOTAL_ERR_COR 0\n' > "$st_d/aer_dev_correctable"; cp "$st_d/aer_dev_correctable" "$st_rp/aer_dev_correctable"
  st_h=$st_d/hwmon/hwmon$st_i; echo 55000 > "$st_h/temp1_input"; echo 120000000 > "$st_h/power1_input"; echo 300000000 > "$st_h/power1_max"
  echo 125000000 > "$st_h/power1_cap"; echo 250000 > "$st_h/curr1_max"; echo 245000 > "$st_h/curr1_input"
  ln -s "$st_d" "$STUB_W/sys/bus/pci/devices/0000:$st_bus:00.0"; ln -s "$st_rp" "$STUB_W/sys/bus/pci/devices/${st_rp##*/}"
  ln -s "$st_d" "$STUB_W/sys/class/tenstorrent/tenstorrent!$st_i/device"; echo 800 > "$STUB_W/sys/class/tenstorrent/tenstorrent!$st_i/tt_aiclk"
  touch "$STUB_W/dev/tenstorrent/$st_i"; st_i=$((st_i + 1))
done; done
st_stub() { printf '#!/bin/bash\n%s\n' "$2" > "$st_b/$1"; chmod +x "$st_b/$1"; }
st_stub hostname 'echo stub-host'
st_stub tt-smi 'echo "stub tt-smi $*" >> "$W_STATE/calls"; exit 0'   # -glx_reset: every chip answers again
st_stub sudo '[ "$1" = -n ] && shift; [ "$1" = -l ] && { echo "    (ALL) NOPASSWD: ALL"; exit 0; }; [ "$1" = -u ] && shift 2; [ "$1" = -H ] && shift
[ "$1" = true ] && exit 0; c=$(command -v "$1"); case $c in "$W_BIN"/*) exec "$@" ;; esac; echo "stub sudo: refused $1 (stub mode runs no real privileged command)" >&2; exit 1'
st_stub ipmitool 'case "$*" in "sdr type Power Supply") for n in 0 1 2 3; do printf "PSU%s | %02xh | ok | 6.1%s | Device Present\\nPSU%s_Status | %02xh | ok | 6.9%s | Transition to OK\\nPSU%s_Input | %02xh | ok | 6.10%s | Device Present\\n" $n $n $n $n $((n + 8)) $n $n $((n + 208)) $n; done; echo "PSU_Redundancy | 0Bh | ok | 6.108 | Fully Redundant" ;;
"sensor reading Power_PSU"*) shift 2; for s; do echo "$s | 900"; done ;; *"power reading"*) echo "    Instantaneous power reading:                  4100 Watts" ;;
*sel*) echo "   1 | 10/21/2026 | 09:00:00 | System Event | Log area reset | Asserted" ;; *"raw 0x04 0x2d "*) echo " 2f c0" ;;
*"raw 0x06 0x52 "*" 0x80 0x01 0x19") echo " ff" ;; *) echo "stub ipmitool $*" ;; esac'
st_stub lspci 'case "$*" in *"-d 1e52"*) for d in "$TRAY_REPRO_SYS"/bus/pci/devices/*; do [ "$(cat "$d/vendor" 2>/dev/null)" = 0x1e52 ] && echo "${d##*/0000:} Processing accelerators: Tenstorrent Inc. Blackhole"; done | if [[ "$*" == *-D* ]]; then sed s/^/0000:/; else cat; fi ;; *) echo "stub lspci $*" ;; esac'
st_stub setpci 'echo f015'   # the root port rescan's LnkSta: DLL Link Active
st_stub dmesg 'echo "[  100.000000] tenstorrent: probe ok"'
for st_t in docker crictl journalctl; do st_stub "$st_t" 'exit 0'; done
st_stub ps 'case "$1" in -eo) exit 0 ;; esac; exec /bin/ps "$@"'   # no other /dev/tenstorrent holder on the fake host
cp "$TT_DIR/lib/stubhost/py" "$st_b/py"; cp "$TT_DIR/lib/stubhost/disc.json" "$STUB_W/disc.json"
export PATH="$st_b:$PATH" W_BIN=$st_b W_STATE=$STUB_W/state W_DEV=$STUB_W/dev/tenstorrent W_DISC=$STUB_W/disc.json TRAY_STUB=1
export UB0_PYTHON=$st_b/py TRAY_REPRO_SYS=$STUB_W/sys TRAY_REPRO_TT_SMI=$st_b/tt-smi TRAY_REPRO_LINK_DESIGN=32.0x4 TRAY_REPRO_RESET_SETTLE_S=1
export HEAVY_IPMID=$STUB_W/dev/ipmi HEAVY_BMC_ENV=$STUB_W/none HEAVY_DEVD=$STUB_W/dev/tenstorrent HEAVY_PROC=$STUB_W/proc
export FLEET_PV_SUDO=0 FLEET_PROC=$STUB_W/proc FLEET_DEVD=$STUB_W/dev/tenstorrent TRAY_RUN_LOCK=$STUB_W/run.lock FLEET_COMPILE_S=8 FLEET_PAIR_HANG_S=3
export FLEET_CHUNK_MIN=0.02,0.02 FLEET_BUSBAR_CHUNK_MIN=0.02,0.02 FLEET_GAP_S=1 FLEET_BUSBAR_HEAVY_S=1 FLEET_PAIR_S=1 FLEET_C7_S=2
export FLEET_C7_HANG_S=5 FLEET_TELEM_S=1 FLEET_POWER_POLL_S=0.5 FLEET_POLL_TAIL_S=0.6 FLEET_LSPCI_POLL_S=1 TRAY_MAXUTIL_DIST=$STUB_W/none
unset st_i st_base st_k st_bus st_rp st_d st_h st_t
