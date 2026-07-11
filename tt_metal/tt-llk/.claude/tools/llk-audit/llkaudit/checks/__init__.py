# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Checker registry. Add a new check by importing it and listing it in ALL."""

from .cfg_word_overlap import CfgWordOverlap
from .mmio_race import MmioRace
from .reconfig_stall import ReconfigStall
from .semaphore_handshake import SemaphoreHandshake

#: All available checks, keyed by their CLI name.
ALL = {c.name: c for c in (MmioRace, CfgWordOverlap, SemaphoreHandshake, ReconfigStall)}
