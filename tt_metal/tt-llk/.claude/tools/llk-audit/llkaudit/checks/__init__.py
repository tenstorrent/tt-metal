# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Checker registry. Add a new check by importing it and listing it in ALL."""

from .cb_sync import CbSync
from .cfg_word_overlap import CfgWordOverlap
from .mailbox_sync import MailboxSync
from .mmio_race import MmioRace
from .noc_sync import NocSync
from .reconfig_stall import ReconfigStall
from .semaphore_handshake import SemaphoreHandshake
from .srcreg_bank import SrcRegBank

#: All available checks, keyed by their CLI name. cb-sync / noc-sync are
#: deterministic but their surface is JIT-compiled kernels OUTSIDE tt-llk, so
#: they only produce findings when fed a KERNEL fact base (the on-request
#: capture — run.sh --full-jit); over the tt-llk fact base they are trivially
#: empty. See the "Full-audit kernel tier" runbook in race-audit-all.
ALL = {
    c.name: c
    for c in (
        MmioRace,
        CfgWordOverlap,
        SemaphoreHandshake,
        ReconfigStall,
        SrcRegBank,
        MailboxSync,
        CbSync,
        NocSync,
    )
}
