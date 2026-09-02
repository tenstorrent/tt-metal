# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for hardware triage scripts. Not a triage script itself."""

import importlib.metadata
import datetime
from dataclasses import dataclass
from pathlib import Path

from ttexalens.tt_exalens_lib import read_arc_telemetry_entry
from ttexalens.umd_device import TimeoutDeviceRegisterError


# ---------------------------------------------------------------------------
# Telemetry decoders
# ---------------------------------------------------------------------------


# ETH/DDR/L2CPU firmware: (major << 16) | (minor << 8) | patch
# Confirmed by SW_VERSION = 0x00020000 = 2.0.0 in tt_cluster.hpp
def _ver3(v):
    return f"{v >> 16}.{(v >> 8) & 0xFF}.{v & 0xFF}"


# BM/CM/Flash firmware: (major << 24) | (minor << 16) | (patch << 8) | build
def _ver4(v):
    return f"{(v >> 24) & 0xFF}.{(v >> 16) & 0xFF}.{(v >> 8) & 0xFF}.{v & 0xFF}"


# Temperature: raw / 65536 = degrees Celsius
def _temp(v):
    return f"{v / 65536:.1f} C"


# Uptime from ARC heartbeat (same logic as check_arc.py)
def _uptime(v):
    offset = 0xA5A5A5A5 if v >= 0xA5A5A5A5 else 0
    return str(datetime.timedelta(seconds=int((v - offset) * 0.1)))


def _mhz(v):
    return f"{v} MHz"


def _mts(v):
    return f"{v} MT/s"


def _hex(v):
    return hex(v)


def _eth_ports(v):
    return f"0x{v:04X} ({bin(v).count('1')} live)"


TELEMETRY_DECODERS = {
    "ETH_FW_VERSION": _ver3,
    "DDR_FW_VERSION": _ver3,
    "L2CPU_FW_VERSION": _ver3,
    "BM_APP_FW_VERSION": _ver4,
    "BM_BL_FW_VERSION": _ver4,
    "FLASH_BUNDLE_VERSION": _ver4,
    "CM_FW_VERSION": _ver4,
    "ASIC_TEMPERATURE": _temp,
    "BOARD_TEMPERATURE": _temp,
    "ARCCLK": _mhz,
    "AICLK": _mhz,
    "DDR_SPEED": _mts,
    "TIMER_HEARTBEAT": _uptime,
    "ETH_LIVE_STATUS": _eth_ports,
    "DDR_STATUS": _hex,
    "BOARD_ID_HIGH": _hex,
    "BOARD_ID_LOW": _hex,
    "ASIC_ID": _hex,
    "HARVESTING_STATE": _hex,
    "ASIC_ID_LOW": _hex,
}


def read_tag(device_id, tag: str) -> str:
    raw = None
    try:
        raw = read_arc_telemetry_entry(device_id, tag)
        decoder = TELEMETRY_DECODERS.get(tag)
        return decoder(raw) if decoder else str(raw)
    except TimeoutDeviceRegisterError:
        raise
    except Exception as e:
        return f"error: {e} {raw}"


# ---------------------------------------------------------------------------
# System / package helpers
# ---------------------------------------------------------------------------


def get_kmd_version() -> str:
    try:
        return Path("/sys/module/tenstorrent/version").read_text().strip()
    except Exception as e:
        return f"unavailable ({e})"


def get_pkg_version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


# ---------------------------------------------------------------------------
# GDDR telemetry decoders
# ---------------------------------------------------------------------------


@dataclass
class GddrModule:
    index: int
    corr_rd: int
    corr_wr: int
    uncorr_rd: int
    uncorr_wr: int
    temp_top: int
    temp_bottom: int


# Two modules per word: even in [15:0], odd in [31:16]; low byte read, high byte write.
# Transcribed from decode_gddr_module_telemetry in UMD's firmware_info_provider_implementation.cpp.
def decode_gddr_module(index: int, temp_word: int, corr_word: int, uncorr_bitmask: int) -> GddrModule:
    shift = 16 if index % 2 else 0
    return GddrModule(
        index=index,
        corr_rd=(corr_word >> shift) & 0xFF,
        corr_wr=(corr_word >> (shift + 8)) & 0xFF,
        uncorr_rd=(uncorr_bitmask >> (2 * index)) & 1,
        uncorr_wr=(uncorr_bitmask >> (2 * index + 1)) & 1,
        temp_bottom=(temp_word >> shift) & 0xFF,
        temp_top=(temp_word >> (shift + 8)) & 0xFF,
    )


# Low bit = complete, high bit = error. 0b00 is IN_PROGRESS, not FAIL.
def decode_status_bits(two_bits: int) -> str:
    return {0b01: "SUCCESS", 0b10: "FAIL", 0b11: "FAIL"}.get(two_bits & 0x3, "IN_PROGRESS")


# Per channel: (training, bist). BIST is the same layout shifted by 16.
def decode_ddr_status(status_word: int, num_channels: int, check_bist: bool) -> list[tuple[str, str]]:
    out = []
    for ch in range(num_channels):
        training = decode_status_bits((status_word >> (2 * ch)) & 0x3)
        bist = decode_status_bits((status_word >> (16 + 2 * ch)) & 0x3) if check_bist else "n/a"
        out.append((training, bist))
    return out
