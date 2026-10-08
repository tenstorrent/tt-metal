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

from check_eth_status import BlackholeEthCore as BH, WormholeEthCore as WH

HW_INC = os.path.join(metal_home, "tt_metal", "hw", "inc", "internal", "tt-1xx")

# u64 counters in eth_live_status_t: (triage address, field)
BH_U64_COUNTERS = [
    (BH.RX_BAD_FCS, "frames_rxd_badfcs"),
    (BH.CORRECTED_CODEWORDS, "corr_cw"),
    (BH.UNCORRECTED_CODEWORDS, "uncorr_cw"),
    *[(BH.TXQ_RESENDS + 8 * queue, f"txq{queue}_resend_cnt") for queue in range(BH.QUEUES)],
    *[(BH.RXQ_DROPS + 8 * queue, f"rxq{queue}_pkt_drop") for queue in range(BH.QUEUES)],
]

# (triage value, C expression over metal's base-firmware headers)
BLACKHOLE_LAYOUT = [
    (BH.BOOT_RESULTS, "MEM_SYSENG_BOOT_RESULTS_BASE"),
    (BH.BOOT_RESULTS_WORDS * 4, "sizeof(boot_results_t) + sizeof(all_eth_mailbox_t)"),
    (BH.PORT_STATUS, "MEM_SYSENG_ETH_STATUS + offsetof(eth_status_t, port_status)"),
    (BH.HEARTBEAT, "MEM_SYSENG_ETH_STATUS + offsetof(eth_status_t, heartbeat)"),
    (BH.RETRAIN_COUNT, "MEM_SYSENG_ETH_LIVE_STATUS + offsetof(eth_live_status_t, retrain_count)"),
    (BH.RX_LINK_UP, "MEM_SYSENG_ETH_LIVE_STATUS + offsetof(eth_live_status_t, rx_link_up)"),
    (BH.ETH_FW_VERSION, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, eth_fw_ver)"),
    (BH.MAILBOX, "MEM_SYSENG_ETH_MAILBOX_ADDR"),
    (BH.MAILBOX_SLOTS, "NUM_ETH_MAILBOX"),
    (BH.MAILBOX_SLOT_SIZE, "sizeof(eth_mailbox_t)"),
    (BH.PCS_STATUS, "ETH_CORE_A_ETH_CTRL_A_PCS_STATUS_REG_ADDR"),
    (BH.ERR_STAT, "ETH_CORE_A_ETH_CTRL_A_ERR_STAT_REG_ADDR"),
    (BH.MAILBOX_CALL << 16, "MEM_SYSENG_ETH_MSG_CALL"),
    (BH.MAILBOX_DONE << 16, "MEM_SYSENG_ETH_MSG_DONE"),
    *[(code, f"MEM_SYSENG_ETH_MSG_{name}") for code, name in BH.MAILBOX_TYPE_NAMES.items()],
    *[(code, f"PORT_{name.upper()}") for code, name in BH.PORT_STATUS_NAMES.items()],
    *[(slot, f"MAILBOX_{name}") for slot, name in enumerate(BH.MAILBOX_SLOT_NAMES)],
    *[
        (address, f"MEM_SYSENG_ETH_LIVE_STATUS + offsetof(eth_live_status_t, {field})")
        for address, field in BH_U64_COUNTERS
    ],
    *[(8, f"sizeof(eth_live_status_t::{field})") for _, field in BH_U64_COUNTERS],
]

WORMHOLE_LAYOUT = [
    (WH.BOOT_RESULTS, "MEM_SYSENG_BOOT_RESULTS_BASE"),
    (WH.BOOT_RESULTS_WORDS * 4, "sizeof(boot_results_t)"),
    (WH.LINK_STATUS, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, link_status)"),
    (WH.RETRAIN_COUNT, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, retrain_cnt)"),
    (WH.RETRAIN_COUNT, "eth_l1_mem::address_map::RETRAIN_COUNT_ADDR"),
    (WH.RETRAIN_FORCE, "eth_l1_mem::address_map::RETRAIN_FORCE_ADDR"),
    (WH.SHARED_HEARTBEAT, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, reserved_48)"),
    (WH.CRC_ERRORS, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, crc_err)"),
    (WH.CORRECTED_CODEWORDS, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, corr_cw_hi)"),
    (WH.CORRECTED_CODEWORDS + 4, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, corr_cw_lo)"),
    (WH.UNCORRECTED_CODEWORDS, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, uncorr_cw_hi)"),
    (WH.UNCORRECTED_CODEWORDS + 4, "MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, uncorr_cw_lo)"),
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
