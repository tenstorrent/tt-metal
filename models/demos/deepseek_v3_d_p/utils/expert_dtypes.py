# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The default routed-expert weight dtype, in one place and overridable from the environment.

`routed_expert_weights_dtype=ttnn.bfloat4_b` used to be spelled as a literal default in six
places (TtMoe, TtPrefillBlock x2, TtPrefillTransformer, TtPrefillRuntime, and the layer-by-layer
loader). A/B-ing expert precision therefore meant editing six defaults and hoping none was missed
-- and missing one is not a crash, it is a cache built at one dtype and requested at another.

So the literal lives here, and `TT_PREFILL_ROUTED_EXPERT_WEIGHTS_DTYPE` overrides it for the whole
process. Resolved once at import: every default binds the same value, so the weight-cache BUILD and
the completeness CHECK cannot disagree.

Nothing else changes -- unset, this is exactly `ttnn.bfloat4_b`.
"""

import os

from loguru import logger

import ttnn

ENV_VAR = "TT_PREFILL_ROUTED_EXPERT_WEIGHTS_DTYPE"

# Only formats the expert matmuls actually accept. bfloat16 is here for cache/unit tests that
# already build experts at bf16.
SUPPORTED = {
    "bfloat4_b": ttnn.bfloat4_b,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat16": ttnn.bfloat16,
}

_FALLBACK = ttnn.bfloat4_b


def _resolve() -> ttnn.DataType:
    name = os.environ.get(ENV_VAR)
    if not name:
        return _FALLBACK
    key = name.strip().lower()
    if key not in SUPPORTED:
        raise ValueError(f"{ENV_VAR}={name!r} is not one of {sorted(SUPPORTED)}")
    dtype = SUPPORTED[key]
    if dtype is not _FALLBACK:
        # Loud on purpose: this changes both the numerics and the weight-cache filenames, and the
        # cache is keyed by path, not by dtype -- point the cache env at a different directory or
        # expect a full rebuild.
        logger.warning(f"{ENV_VAR}={key}: routed experts are NOT the default {_FALLBACK.name}")
    return dtype


DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE = _resolve()
