#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    eth_status_sampling

Description:
    Data provider script that reads base-firmware state from every active ethernet core as early as possible.
    check_eth_status reads the same words again later and compares the two samples.

Owner:
    onenezicTT
"""

from dataclasses import dataclass
import struct
import time
from typing import cast

from run_checks import run as get_run_checks, RunChecks
from triage import ScriptConfig, ScriptPriority, run_script, triage_singleton
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device, read_word_from_device

script_config = ScriptConfig(
    data_provider=True,
    depends=["run_checks"],
    # Run as early as possible so more time passes before check_eth_status takes its second sample.
    priority=ScriptPriority.HIGH,
)

# Covers two Blackhole link-status refresh periods (about 1.25 s each) between the samples.
MINIMUM_SAMPLE_GAP_SECONDS = 2.0
# Extra heartbeat reads when both samples hold the same value, before calling the word frozen.
HEARTBEAT_BURST_READS = 4

# What a failed host read returns.
ALL_ONES = 0xFFFFFFFF

# Wormhole base firmware: wormhole/eth_fw_api.h, eth_l1_address_map.h, UMD wormhole_eth.hpp.
WH_HEARTBEAT = 0x1C  # base FW, once per loop pass
WH_PORT_DISABLE_MASK = 0x1008  # boot params, bit N = channel N disabled
WH_TRAIN_STATUS = 0x1104  # 0 in progress, 1 success, 2 fail
WH_LINK_ERROR_STATUS = 0x1440  # training error code, >= 11 means not connected
WH_LAUNCH_ERISC_APP_FLAG = 0x9004  # nonzero while metal firmware owns the core
WH_BOOT_RESULTS = 0x1EC0  # boot_results_t
WH_BOOT_RESULTS_WORDS = 80
WH_LINK_STATUS = 0x1ED4  # link is up only when it equals WH_LINK_UP
WH_LINK_UP = 6
WH_RETRAIN_COUNT = 0x1EDC
WH_RETRAIN_FORCE = 0x1EFC  # 1 while a requested retrain is pending
WH_SHARED_HEARTBEAT = 0x1F80  # base FW (0xABCD) and fabric router (0xDCBA)
WH_LOCAL_ETH_ID = 0x1FD0

# Blackhole base firmware: blackhole/eth_fw_api.h, 32-bit device layout.
BH_BOOT_RESULTS = 0x7CC00  # boot_results_t (1 KB), then the eth FW mailboxes (64 B)
BH_BOOT_RESULTS_WORDS = 272
BH_PORT_STATUS = 0x7CC04  # 0 unknown, 1 up, 2 down, 3 unused
BH_HEARTBEAT = 0x7CC70  # heartbeat[0]: base FW (0xABCD) and fabric router (0xDCBA)
BH_RETRAIN_COUNT = 0x7CE00
BH_RX_LINK_UP = 0x7CE04  # snapshot, up only when it equals 1
BH_LOCAL_INFO = 0x7CFC0  # chip_info_t, byte 2 is the physical eth channel
BH_MAILBOX = 0x7D000  # HOST, RISC1, CMFW, OTHER
BH_MAILBOX_SLOTS = 4
BH_MAILBOX_SLOT_SIZE = 16  # msg + 3 args
BH_PCS_STATUS = 0xFFB9800C  # live link status, up only when it equals 1


@dataclass(frozen=True)
class EthReadPlan:
    blocks: tuple[tuple[int, int], ...]  # (address, number of u32 words) read in one access
    words: tuple[int, ...]
    heartbeats: tuple[int, ...]


WORMHOLE_READ_PLAN = EthReadPlan(
    blocks=((WH_BOOT_RESULTS, WH_BOOT_RESULTS_WORDS),),
    words=(WH_HEARTBEAT, WH_PORT_DISABLE_MASK, WH_TRAIN_STATUS, WH_LINK_ERROR_STATUS, WH_LAUNCH_ERISC_APP_FLAG),
    heartbeats=(WH_HEARTBEAT, WH_SHARED_HEARTBEAT),
)

BLACKHOLE_READ_PLAN = EthReadPlan(
    blocks=((BH_BOOT_RESULTS, BH_BOOT_RESULTS_WORDS),),
    words=(BH_PCS_STATUS,),
    heartbeats=(BH_HEARTBEAT,),
)


@dataclass
class EthCoreSample:
    timestamp: float
    words: dict[int, int]


def get_read_plan(location: OnChipCoordinate) -> EthReadPlan | None:
    if location.device.is_wormhole():
        return WORMHOLE_READ_PLAN
    if location.device.is_blackhole():
        return BLACKHOLE_READ_PLAN
    return None


class EthStatusSampling:
    def __init__(self, run_checks: RunChecks, context: Context):
        self._context = context
        self.initial_samples: dict[OnChipCoordinate, EthCoreSample] = {}
        for result in run_checks.run_per_block_check(self._try_read_sample, block_filter=["active_eth"]) or []:
            self.initial_samples[result.location] = cast(EthCoreSample, result.result)

    def _try_read_sample(self, location: OnChipCoordinate) -> EthCoreSample | None:
        try:
            return self.read_sample(location)
        except Exception:
            # check_eth_status reports the core if its own read fails too.
            return None

    def read_sample(self, location: OnChipCoordinate) -> EthCoreSample | None:
        plan = get_read_plan(location)
        if plan is None:
            return None
        words: dict[int, int] = {}
        for address, count in plan.blocks:
            data = read_from_device(location, address, num_bytes=4 * count, context=self._context)
            for index, value in enumerate(struct.unpack(f"<{count}I", data)):
                words[address + 4 * index] = value
        for address in plan.words:
            words[address] = read_word_from_device(location, address, context=self._context)
        return EthCoreSample(timestamp=time.monotonic(), words=words)

    def read_heartbeat_burst(self, location: OnChipCoordinate) -> list[dict[int, int]]:
        plan = get_read_plan(location)
        if plan is None:
            return []
        return [
            {address: read_word_from_device(location, address, context=self._context) for address in plan.heartbeats}
            for _ in range(HEARTBEAT_BURST_READS)
        ]

    def get_initial_sample(self, location: OnChipCoordinate) -> EthCoreSample | None:
        return self.initial_samples.get(location)

    def wait_for_second_sample(self) -> None:
        if not self.initial_samples:
            return
        latest = max(sample.timestamp for sample in self.initial_samples.values())
        remaining = MINIMUM_SAMPLE_GAP_SECONDS - (time.monotonic() - latest)
        if remaining > 0:
            time.sleep(remaining)


@triage_singleton
def run(args, context: Context) -> EthStatusSampling:
    run_checks = get_run_checks(args, context)
    return EthStatusSampling(run_checks, context)


if __name__ == "__main__":
    run_script()
