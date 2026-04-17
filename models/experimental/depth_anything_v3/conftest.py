# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for Depth Anything V3. Pins TT_METAL_HOME to the source tree the
installed _ttnncpp.so was built against, so JIT kernel compilation finds a
consistent header + kernel-source pair."""

import os

# The editable ttnn install in ~/.tenstorrent-venv resolves to medgemma's
# tt-metal source. The runtime hard-codes medgemma's hw/inc paths into the
# kernel build command, so kernel sources MUST come from the same tree —
# otherwise newer kernels in da3's tree reference headers that don't exist
# in medgemma. The runtime reads `TT_METAL_RUNTIME_ROOT` (not _HOME) to
# resolve kernel source paths; without it the cwd fallback picks da3.
_MEDGEMMA_TREE = "/home/ttuser/experiments/medgemma/tt-metal"
os.environ.setdefault("TT_METAL_RUNTIME_ROOT", _MEDGEMMA_TREE)
os.environ.setdefault("TT_METAL_HOME", _MEDGEMMA_TREE)
