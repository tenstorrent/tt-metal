# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# SFPU nodes run on TRISC3 and borrow Dest from the thread that owns the section (MATH or PACK),
# the same way unpack to dest borrows it for UNPACK. The lender side is emitted into that thread's
# kernel, the borrower side into the TRISC3 kernel.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig


def hw_configure_sfpu(config: "GlobalConfig") -> str:
    return f"_llk_sfpu_hw_configure_<{config.dest_acc.cpp_enum_value}>();\n"


def sfpu_lend_dest(lender: str) -> str:
    return (
        f"_llk_tile_regs_borrow_<DstClient::{lender}, DstClient::SFPU>();\n"
        f"_llk_tile_regs_return_<DstClient::{lender}, DstClient::SFPU>();\n"
    )


def sfpu_wait_for_dest_available(lender: str) -> str:
    return f"_llk_tile_regs_borrow_<DstClient::{lender}, DstClient::SFPU>();\n"


def sfpu_section_done(lender: str) -> str:
    return f"_llk_tile_regs_return_<DstClient::{lender}, DstClient::SFPU>();\n"
