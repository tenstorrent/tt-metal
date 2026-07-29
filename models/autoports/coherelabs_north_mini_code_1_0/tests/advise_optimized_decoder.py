# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn-advise capture`` target for the North-Mini optimized dense block.

This is a build-time capture helper, not a pytest or hardware benchmark.  The
advisor supplies a mock 1x1 mesh device to ``make_inputs`` and traces one
batch-shaped decode step through layer 0 (dense attention + dense MLP).

Run from a fresh shell process after sourcing the repo-local shard-advisor
bootstrap:

    ttnn-advise capture \
        models/autoports/coherelabs_north_mini_code_1_0/tests/advise_optimized_decoder.py:decode \
        --out /tmp/north-mini-shard-advice

Set ``NORTH_MINI_SHARD_ADVISE_BATCH=1`` to capture the small-batch geometry;
the serving-batch default is 32.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Append, never prepend: the advisor environment's installed ``ttnn`` package
# must win over the tt-metal source directory named ``ttnn``.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (  # noqa: E402
    REAL_REVISION,
    _decode_inputs,
    _page_table,
    _synthetic_state,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import (  # noqa: E402
    MODEL_ID,
    OptimizationConfig,
    OptimizedDecoder,
)

LAYER_IDX = 0
BATCH = int(os.environ.get("NORTH_MINI_SHARD_ADVISE_BATCH", "32"))
MAX_CACHE_LEN = 32

_DECODER = None
_DECODE_KWARGS = None


def _build(device):
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=REAL_REVISION,
        local_files_only=True,
    )
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config, LAYER_IDX),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_cache_len=MAX_CACHE_LEN,
        optimization_config=OptimizationConfig(),
    )

    generator = torch.Generator().manual_seed(20260729 + BATCH)
    hidden = torch.randn(1, BATCH, 1, config.hidden_size, generator=generator).mul_(0.02).to(torch.bfloat16)
    hidden = _to_tt(hidden, device)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(
        _page_table(BATCH, 1),
        device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    current_positions, position_cos, position_sin = _decode_inputs(
        decoder,
        config,
        device,
        [0] * BATCH,
    )
    kwargs = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "page_table": page_table,
        "current_positions": current_positions,
        "position_cos": position_cos,
        "position_sin": position_sin,
    }
    return decoder, kwargs, hidden


def decode(hidden):
    """Trace one final-policy decode step through dense layer 0."""

    return _DECODER.decode_forward(hidden, **_DECODE_KWARGS)


def make_inputs(device):
    """Build the positional arguments expected by ``decode``."""

    global _DECODER, _DECODE_KWARGS
    _DECODER, _DECODE_KWARGS, hidden = _build(device)
    return (hidden,)
