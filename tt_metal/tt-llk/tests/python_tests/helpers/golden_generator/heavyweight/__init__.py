# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Heavyweight golden generators: block-by-block models of the hardware pipeline.

A golden here does not compute its operation. It declares the sequence of
transfers the hardware performs -- L1 to src register, src to Dest, Dest to L1
-- and the result is whatever falls out of the end, with precision lost at each
boundary because that is where silicon loses it.

Two subpackages, each with a README:

* :mod:`.operations` -- the pipelines, and the engine that runs them.
* :mod:`.data_transfer_blocks` -- what each transfer does to a buffer.

:mod:`.mismatch` sits alongside them and is for failures only. It adds what a
value dump lacks -- *how badly*, in lattice steps rather than absolute error,
and the golden's pre-pack Dest beside the packed result, which narrows where to
look though it cannot by itself say whether the device's math or its pack
diverged. ``passed_test`` already prints the failing tiles with the bad datums
highlighted, so use the two together.

Writing a test touches only the first. Pick the golden for your architecture and
hand it tensors::

    from helpers.golden_generator.heavyweight.operations.quasar_operations import (
        QuasarDataCopyGolden,
    )

    result = QuasarDataCopyGolden().run(stimuli, in_format, out_format)
"""
