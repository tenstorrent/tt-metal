# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Per-card device lock, dirty flag and reset. Sourced by run_safe_pytest.sh, tt-probe.sh and
# tt_ops_code_gen's eval_test_runner.sh, so that runs pinned to different cards of one host run in parallel.
#
# TT_VISIBLE_DEVICES=<one id>   card mode: the run uses that one chip only. It holds the machine lock
#                               (/tmp/tt-device.lock) SHARED and its card lock (/tmp/tt-device-<id>.lock)
#                               EXCLUSIVE, marks and checks /tmp/tt-device-<id>.dirty, and resets only its chip.
#                               The id is UMD's: chips in PCI-address order, the numbering tt-metal and tt-smi use.
#                               It is not always the /dev/tenstorrent number (the two orders can differ).
# unset, or a list of ids       machine mode, as before: the machine lock EXCLUSIVE, /tmp/tt-device.dirty,
#                               every card reset.
#
# Scripts that predate this file take the machine lock exclusively too, so old and new runs exclude each other.
# Lock order is always machine lock, then card lock, so the two modes cannot deadlock.
#
# After sourcing: TT_DEV_LOCK and TT_DEV_DIRTY are the lock and dirty flag this run uses on fd 9 as before;
# call tt_dev_hold_machine_shared before taking fd 9, tt_dev_is_dirty instead of testing the dirty flag, and
# tt_dev_reset instead of `tt-smi -r`; add TT_DEV_TRIAGE_ARGS to tt-triage and TT_DEV_TRIAGE_SUFFIX to its
# report name. Card mode also exports TT_MESH_GRAPH_DESC_PATH (one chip of a p300) and
# TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS (one port per card), unless they are already set.

TT_DEV_MACHINE_LOCK="/tmp/tt-device.lock"
TT_DEV_MACHINE_DIRTY="/tmp/tt-device.dirty"
TT_DEV_CARD=""
if [[ "${TT_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
    TT_DEV_CARD="$TT_VISIBLE_DEVICES"
fi
if [[ -n "$TT_DEV_CARD" ]]; then
    TT_DEV_LOCK="/tmp/tt-device-${TT_DEV_CARD}.lock"
    TT_DEV_DIRTY="/tmp/tt-device-${TT_DEV_CARD}.dirty"
    TT_DEV_LABEL="card ${TT_DEV_CARD}"
    # The card's sysfs node: the UMD id is the index in PCI-address order.
    TT_DEV_SYSFS="$(for _d in /sys/class/tenstorrent/tenstorrent!*; do
        [[ -e "$_d" ]] && echo "$(basename "$(readlink -f "$_d/device")") $_d"; done | sort | sed -n "$((TT_DEV_CARD + 1))p" | cut -d' ' -f2)"
    # One chip of a multi-chip Blackhole board (p300) is a CUSTOM cluster to tt-metal, which then requires a
    # fabric mesh graph descriptor: give it the single-chip one (a p150's).
    if [[ -z "${TT_MESH_GRAPH_DESC_PATH:-}" && -n "$TT_DEV_SYSFS" && "$(cat "$TT_DEV_SYSFS/device/device" 2>/dev/null)" == 0xb140 ]]; then
        _tt_dev_mgd="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
        [[ -f "$_tt_dev_mgd" ]] && export TT_MESH_GRAPH_DESC_PATH="$_tt_dev_mgd"
    fi
    # Own Inspector RPC port: parallel runs would otherwise contend for localhost:50051, and a hang's tt-triage
    # would read another card's process. Pass TT_DEV_TRIAGE_ARGS to tt-triage; the report gets a per-card name.
    export TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS="${TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS:-localhost:$((50052 + TT_DEV_CARD))}"
    TT_DEV_TRIAGE_ARGS="--inspector-rpc-port=${TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS##*:}"
    TT_DEV_TRIAGE_SUFFIX="-card${TT_DEV_CARD}"
else
    TT_DEV_LOCK="$TT_DEV_MACHINE_LOCK"
    TT_DEV_DIRTY="$TT_DEV_MACHINE_DIRTY"
    TT_DEV_LABEL="all cards"
    TT_DEV_TRIAGE_ARGS=""
    TT_DEV_TRIAGE_SUFFIX=""
fi

# Card mode: hold the machine lock shared on fd 8 for the rest of the script (children inherit it, like fd 9).
# Blocks while a machine-mode run, or an old script, holds it exclusively. No-op in machine mode.
tt_dev_hold_machine_shared() {
    [[ -z "$TT_DEV_CARD" ]] && return 0
    exec 8>"$TT_DEV_MACHINE_LOCK"
    if ! flock -s -n 8; then
        echo "${1:-TT_DEVICE}: waiting for the machine lock (a whole-machine run holds it)..."
        flock -s 8
    fi
}

# A machine-mode run (or an old script) that left the machine dirty may have left any card dirty: hand one
# dirty flag to each card, once, under a small mutex so a card run checking its own flag sees the split.
_tt_dev_split_machine_dirty() {
    (
        flock 7
        if [[ -f "$TT_DEV_MACHINE_DIRTY" ]]; then
            local d i=0
            for d in /dev/tenstorrent/*; do
                [[ -e "$d" ]] && touch "/tmp/tt-device-$((i++)).dirty"
            done
            rm -f "$TT_DEV_MACHINE_DIRTY"
        fi
    ) 7>/tmp/tt-device.dirty.lock
}

# 0 when this run must reset before using the device. Call with the locks held.
tt_dev_is_dirty() {
    if [[ -n "$TT_DEV_CARD" ]]; then
        _tt_dev_split_machine_dirty
        [[ -f "$TT_DEV_DIRTY" ]]
        return
    fi
    [[ -f "$TT_DEV_MACHINE_DIRTY" ]] && return 0
    compgen -G "/tmp/tt-device-[0-9]*.dirty" >/dev/null
}

# Reset this run's card (card mode) or every card (machine mode). Returns tt-smi's status. On success in
# machine mode every card's dirty flag goes too; the caller still removes TT_DEV_DIRTY as before.
tt_dev_reset() {
    if [[ -n "$TT_DEV_CARD" ]]; then
        # By UMD id, the same numbering as TT_VISIBLE_DEVICES; tt-smi must see every chip for its ids to mean
        # that, so TT_VISIBLE_DEVICES is dropped. --no_reinit: after the reset tt-smi otherwise re-detects every
        # device, and touching a chip another run is using kills tt-smi with SIGBUS (seen on a p300 whose other
        # chip was busy). The next device open initialises this chip.
        # shellcheck disable=SC2086
        env -u TT_VISIBLE_DEVICES tt-smi -r "${TT_DEV_CARD}" ${TT_DEV_RESET_ARGS:---no_reinit}
        return
    fi
    tt-smi -r || return
    rm -f /tmp/tt-device-[0-9]*.dirty
}
