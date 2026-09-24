# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn port of the Gemma4-31B DFlash speculative-decoding drafter
(z-lab/gemma-4-31B-it-DFlash). See models/demos/gemma4/docs/dflash_design.md
for the design and build plan. In progress -- real-weight loading (config.py,
weight_mapping.py, mlp.py, weights.py) is the only piece implemented and
validated so far (see tests/dflash/test_dflash_weights.py). Context
extraction, the drafter's own forward pass, and the verify/accept loop wiring
are not yet implemented.
"""
