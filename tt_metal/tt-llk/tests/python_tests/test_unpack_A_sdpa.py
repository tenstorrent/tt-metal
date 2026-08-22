# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ADVANCE TEST: covers demo-fork experimental LLK unpack_A_sdpa (tt-metal#47554 / tt-blaze#1971), pending promotion.
# Include path (shadow -I) repoint on promotion. Primitive verified vs tt-blaze main as of this writing.
#
# unpack_A_sdpa is init/mop-config + a dummy-SrcB-valid helper only; it has no per-tile execute of its own. This test
# drives all three of its symbols:
#   - _llk_unpack_A_sdpa_init_<num_tiles, BType>(...)      : programs the SrcA-only UNPACR MOP.
#   - the base llk_unpack_A execute                        : streams the operand tiles into SrcA under that MOP.
#   - _llk_unpack_A_sdpa_set_srcb_dummy_valid_()           : injects STALL_UNPACK + a UNPACR_NOP SET_DVALID on SrcB
#                                                            (ZEROSRC, no real data) so the downstream dual-source
#                                                            eltwise's math preamble STALLWAIT(SRCB_VLD) clears.
#
# To exercise unpack_A_sdpa with a validatable NUMERIC golden it is paired with the demo-fork math SDPA column-
# broadcast SrcB-reuse op (llk_math_sdpa_bcast_col_srcb_reuse.h), exactly as test_sdpa_bcast_col_srcb_reuse.py does --
# the two differ only in which primitive is nominally under test, so they share one driver
# (helpers/sdpa_bcast_utils.py) rather than cloning the stimuli/golden/compare body.
#
# The pairing is what makes the ordering constraint on the dummy-SrcB helper observable: it must be issued BEFORE the
# two operand unpacks. Issued after them, the unpacker blocks on SrcA banks that only the math execute frees while
# math sits in the preamble's STALLWAIT(SRCB_VLD) waiting for this very instruction.
#
# Blackhole-only (@blackhole_only): the primitive headers resolve through a Blackhole-only shadow -I.

from conftest import blackhole_only
from helpers.advance_llk_includes import (  # noqa: F401  (module-scoped autouse fixture)
    advance_llk_include_paths,
)
from helpers.device import BootMode
from helpers.param_config import parametrize
from helpers.sdpa_bcast_utils import (
    SDPA_BCAST_FORMATS,
    run_sdpa_bcast_col_srcb_reuse,
)


@blackhole_only
@parametrize(
    formats=SDPA_BCAST_FORMATS,
)
def test_unpack_A_sdpa(
    formats,
    boot_mode=BootMode.DEFAULT,
):
    run_sdpa_bcast_col_srcb_reuse("sources/unpack_A_sdpa_test.cpp", formats, boot_mode)
