# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Block-by-block models of the hardware pipeline.

A golden here does not compute its operation. It declares the transfers the
hardware performs and the result is whatever falls out of the end, losing
precision at each boundary because that is where silicon loses it.

:mod:`.operations` has the pipelines, :mod:`.data_transfer_blocks` what each
transfer does to a buffer, and :mod:`.mismatch` is for failures only. Each has
a README.

Writing a test touches only the first::

    from helpers.golden_generator.heavyweight.operations import (
        QUASAR,
        Op,
        get_golden,
    )

    result = get_golden(Op.DataCopy, QUASAR).run(stimuli, in_format, out_format)
"""
