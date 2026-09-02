#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_dram_health

Description:
    Per-GDDR-instance error counters from ARC telemetry: corrected EDC counts, uncorrected EDC
    flags, DRAM training and BIST status, temperature and speed. Reads only ARC telemetry, so it
    is safe against a live hung process. Blackhole only.

    An uncorrected error fails the check: corruption was detected and not fixed, so wrong data
    reached the consumer. Corrected counts are reported but do not fail, since the retry landed;
    they saturate at 255, so 255 means "at least 255" and no rate can be derived. Training, BIST,
    speed and temperature are informational; only an explicit training or BIST failure fails.

Owner:
    onenezicTT
"""

from collections import defaultdict
from dataclasses import dataclass

import tt_umd
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device
from ttexalens.util import FirmwareVersion
from ttexalens.tt_exalens_lib import read_arc_telemetry_entry

from run_checks import run as get_run_checks
from triage import (
    ScriptConfig,
    log_check_device,
    log_warning_device,
    run_script,
    triage_field,
)
from triage_hw_utils import decode_ddr_status, decode_gddr_module

script_config = ScriptConfig(depends=["run_checks"])

# BIST occupies the upper half of DDR_STATUS only on Blackhole from here on. Gate matches UMD.
BIST_MIN_FW = FirmwareVersion(19, 7, 0)

TAG_DDR_STATUS = int(tt_umd.TelemetryTag.DDR_STATUS)
TAG_DDR_SPEED = int(tt_umd.TelemetryTag.DDR_SPEED)
TAG_GDDR_TEMP_BASE = int(tt_umd.TelemetryTag.GDDR_0_1_TEMP)
TAG_GDDR_CORR_BASE = int(tt_umd.TelemetryTag.GDDR_0_1_CORR_ERRS)
TAG_GDDR_UNCORR = int(tt_umd.TelemetryTag.GDDR_UNCORR_ERRS)


# Values stay numeric so the sqlite output is queryable.
@dataclass
class DramHealthRow:
    # "Dev" comes from PerDeviceCheckResult; redeclaring it makes the sqlite serializer raise.
    instance: int = triage_field("GDDR Inst")
    corrected_rd: int = triage_field("Corrected Rd")
    corrected_wr: int = triage_field("Corrected Wr")
    uncorrected_rd: int = triage_field("Uncorrected Rd")
    uncorrected_wr: int = triage_field("Uncorrected Wr")
    training: str = triage_field("Train")
    bist: str = triage_field("BIST")
    endpoints: list[OnChipCoordinate] = triage_field("Endpoints")
    temp_top: int = triage_field("Temp Top")
    temp_bottom: int = triage_field("Temp Bot")
    speed_mts: int = triage_field("Speed MT/s")


# A DRAM core's logical coordinate is (channel, subchannel), so exalens' own block list gives the
# channel grouping directly - right arch by construction, and harvested cores are already excluded.
def dram_endpoints_by_channel(device: Device) -> list[list[OnChipCoordinate]]:
    by_channel: dict[int, list[tuple[int, OnChipCoordinate]]] = defaultdict(list)
    for location in device.get_block_locations("dram"):
        (channel, subchannel), _ = location.to("logical")
        by_channel[channel].append((subchannel, location))
    return [[loc for _, loc in sorted(by_channel.get(ch, []))] for ch in range(max(by_channel) + 1)]


def check_dram_telemetry(device: Device) -> list[DramHealthRow] | None:
    if not device.is_blackhole():
        # Wormhole does not provide GDDR telemetry.
        return None

    device_id = device.id
    fw = device.firmware_version
    endpoints = dram_endpoints_by_channel(device)
    modules = len(endpoints)

    try:
        speed = read_arc_telemetry_entry(device_id, TAG_DDR_SPEED)
        status_word = read_arc_telemetry_entry(device_id, TAG_DDR_STATUS)
        uncorr_bitmask = read_arc_telemetry_entry(device_id, TAG_GDDR_UNCORR)
        temp_words = [read_arc_telemetry_entry(device_id, TAG_GDDR_TEMP_BASE + p) for p in range(modules // 2)]
        corr_words = [read_arc_telemetry_entry(device_id, TAG_GDDR_CORR_BASE + p) for p in range(modules // 2)]
    except RuntimeError as e:
        log_warning_device(device, f"no GDDR telemetry on FW {fw}: {e}")
        return None

    check_bist = fw >= BIST_MIN_FW
    statuses = decode_ddr_status(status_word, modules, check_bist)

    rows: list[DramHealthRow] = []
    for i in range(modules):
        training, bist = statuses[i]
        m = decode_gddr_module(i, temp_words[i // 2], corr_words[i // 2], uncorr_bitmask)

        log_check_device(
            device,
            not (m.uncorr_rd or m.uncorr_wr),
            f"GDDR instance {i}: uncorrected EDC errors rd/wr {m.uncorr_rd}/{m.uncorr_wr} - "
            f"corruption was detected and not fixed, so wrong data reached the consumer.",
        )

        log_check_device(
            device,
            training == "SUCCESS" and bist == "SUCCESS",
            f"GDDR instance {i} did not come up cleanly: training={training} bist={bist}",
        )

        rows.append(
            DramHealthRow(
                instance=i,
                corrected_rd=m.corr_rd,
                corrected_wr=m.corr_wr,
                uncorrected_rd=m.uncorr_rd,
                uncorrected_wr=m.uncorr_wr,
                training=training,
                bist=bist,
                endpoints=endpoints[i],
                temp_top=m.temp_top,
                temp_bottom=m.temp_bottom,
                speed_mts=speed,
            )
        )

    return rows


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    return run_checks.run_per_device_check(lambda device: check_dram_telemetry(device))


if __name__ == "__main__":
    run_script()
