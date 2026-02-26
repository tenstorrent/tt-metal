# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for hardware triage scripts. Not a triage script itself."""

import importlib.metadata
import datetime
from pathlib import Path

from ttexalens.tt_exalens_lib import read_arc_telemetry_entry


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
