# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Checks the addresses and codes check_eth_status hardcodes against metal's base-firmware headers, so a layout change
# fails CI instead of being misread on hardware. Needs no hardware. Values those headers don't define are not checked.

import os
import shutil
import subprocess
import sys

import pytest

metal_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(metal_home, "tools", "triage"))

from check_eth_status import (
    BH_BOOT_RESULTS,
    BH_BOOT_RESULTS_WORDS,
    BH_CORRECTED_CODEWORDS,
    BH_ERR_STAT,
    BH_ETH_FW_VERSION,
    BH_HEARTBEAT,
    BH_MAILBOX,
    BH_MAILBOX_SLOT_SIZE,
    BH_MAILBOX_SLOTS,
    BH_PCS_STATUS,
    BH_PORT_STATUS,
    BH_PORT_STATUS_NAMES,
    BH_QUEUES,
    BH_RETRAIN_COUNT,
    BH_RX_BAD_FCS,
    BH_RX_LINK_UP,
    BH_RXQ_DROPS,
    BH_TXQ_RESENDS,
    BH_UNCORRECTED_CODEWORDS,
    MAILBOX_CALL,
    MAILBOX_DONE,
    MAILBOX_SLOT_NAMES,
    MAILBOX_TYPE_NAMES,
    WH_BOOT_RESULTS,
    WH_BOOT_RESULTS_WORDS,
    WH_CORRECTED_CODEWORDS,
    WH_CRC_ERRORS,
    WH_LINK_STATUS,
    WH_RETRAIN_COUNT,
    WH_RETRAIN_FORCE,
    WH_SHARED_HEARTBEAT,
    WH_UNCORRECTED_CODEWORDS,
)

HW_INC = os.path.join(metal_home, "tt_metal", "hw", "inc", "internal", "tt-1xx")

# u64 counters in eth_live_status_t: (triage address, field)
BH_U64_COUNTERS = [
    (BH_RX_BAD_FCS, "frames_rxd_badfcs"),
    (BH_CORRECTED_CODEWORDS, "corr_cw"),
    (BH_UNCORRECTED_CODEWORDS, "uncorr_cw"),
    *[(BH_TXQ_RESENDS + 8 * queue, f"txq{queue}_resend_cnt") for queue in range(BH_QUEUES)],
    *[(BH_RXQ_DROPS + 8 * queue, f"rxq{queue}_pkt_drop") for queue in range(BH_QUEUES)],
]

# (triage value, C expression over metal's base-firmware headers)
BLACKHOLE_LAYOUT = [
    (BH_BOOT_RESULTS, "MEM_SYSENG_BOOT_RESULTS_BASE"),
    (BH_BOOT_RESULTS_WORDS * 4, "sizeof(boot_results_t) + sizeof(all_eth_mailbox_t)"),
    (BH_PORT_STATUS, "MEM_SYSENG_ETH_STATUS + offsetof(eth_status_t, port_status)"),
    (BH_HEARTBEAT, "MEM_SYSENG_ETH_STATUS + offsetof(eth_status_t, heartbeat)"),
    (BH_RETRAIN_COUNT, "MEM_SYSENG_ETH_LIVE_STATUS + offsetof(eth_live_status_t, retrain_count)"),
    (BH_RX_LINK_UP, "MEM_SYSENG_ETH_LIVE_STATUS + offsetof(eth_live_status_t, rx_link_up)"),
    (BH_ETH_FW_VERSION, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, eth_fw_ver)"),
    (BH_MAILBOX, "MEM_SYSENG_ETH_MAILBOX_ADDR"),
    (BH_MAILBOX_SLOTS, "NUM_ETH_MAILBOX"),
    (BH_MAILBOX_SLOT_SIZE, "sizeof(eth_mailbox_t)"),
    (BH_PCS_STATUS, "ETH_CORE_A_ETH_CTRL_A_PCS_STATUS_REG_ADDR"),
    (BH_ERR_STAT, "ETH_CORE_A_ETH_CTRL_A_ERR_STAT_REG_ADDR"),
    (MAILBOX_CALL << 16, "MEM_SYSENG_ETH_MSG_CALL"),
    (MAILBOX_DONE << 16, "MEM_SYSENG_ETH_MSG_DONE"),
    *[(code, f"MEM_SYSENG_ETH_MSG_{name}") for code, name in MAILBOX_TYPE_NAMES.items()],
    *[(code, f"PORT_{name.upper()}") for code, name in BH_PORT_STATUS_NAMES.items()],
    *[(slot, f"MAILBOX_{name}") for slot, name in enumerate(MAILBOX_SLOT_NAMES)],
    *[
        (address, f"MEM_SYSENG_ETH_LIVE_STATUS + offsetof(eth_live_status_t, {field})")
        for address, field in BH_U64_COUNTERS
    ],
    *[(8, f"sizeof(eth_live_status_t::{field})") for _, field in BH_U64_COUNTERS],
]

WORMHOLE_LAYOUT = [
    (WH_BOOT_RESULTS, "MEM_SYSENG_BOOT_RESULTS_BASE"),
    (WH_BOOT_RESULTS_WORDS * 4, "sizeof(boot_results_t)"),
    (WH_LINK_STATUS, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, link_status)"),
    (WH_RETRAIN_COUNT, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, retrain_cnt)"),
    (WH_RETRAIN_COUNT, "eth_l1_mem::address_map::RETRAIN_COUNT_ADDR"),
    (WH_RETRAIN_FORCE, "eth_l1_mem::address_map::RETRAIN_FORCE_ADDR"),
    (WH_SHARED_HEARTBEAT, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, reserved_48)"),
    (WH_CRC_ERRORS, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, crc_err)"),
    (WH_CORRECTED_CODEWORDS, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, corr_cw_hi)"),
    (WH_CORRECTED_CODEWORDS + 4, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, corr_cw_lo)"),
    (WH_UNCORRECTED_CODEWORDS, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, uncorr_cw_hi)"),
    (WH_UNCORRECTED_CODEWORDS + 4, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, uncorr_cw_lo)"),
]


def find_riscv_compiler() -> str | None:
    # Where metal's JIT build looks for SFPI (tt_metal/jit_build/build.cpp), then PATH.
    for root in (os.path.join(metal_home, "runtime", "sfpi"), "/opt/tenstorrent/sfpi"):
        compiler = os.path.join(root, "compiler", "bin", "riscv-tt-elf-g++")
        if os.access(compiler, os.X_OK):
            return compiler
    return shutil.which("riscv-tt-elf-g++")


@pytest.mark.parametrize(
    "arch, mcpu, headers, layout",
    [
        ("blackhole", "tt-bh", ["eth_fw_api.h"], BLACKHOLE_LAYOUT),
        ("wormhole", "tt-wh", ["eth_fw_api.h", "eth_l1_address_map.h"], WORMHOLE_LAYOUT),
    ],
)
def test_addresses_match_firmware_headers(arch, mcpu, headers, layout):
    # Compiled for the RISC-V cores because x86-64 lays some of these structs out differently.
    compiler = find_riscv_compiler()
    if compiler is None:
        pytest.skip("SFPI RISC-V compiler not found")
    source = ["#include <cstddef>"] + [f'#include "{header}"' for header in headers]
    source += [f'static_assert(({expr}) == {value:#x}, "{expr}");' for value, expr in layout]
    command = [compiler, f"-mcpu={mcpu}", "-std=c++17", "-fsyntax-only", "-x", "c++", "-"]
    command.append(f"-I{os.path.join(HW_INC, arch)}")
    result = subprocess.run(command, input="\n".join(source), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
