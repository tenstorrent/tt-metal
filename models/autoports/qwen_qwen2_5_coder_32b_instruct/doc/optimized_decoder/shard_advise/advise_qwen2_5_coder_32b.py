# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn-advise capture`` target for the optimized dense Qwen decoder.

This deliberately exercises the same batch-32 decode graph and synthetic input
builders as ``tests/test_optimized_decoder.py``.  Values are synthetic because
the advisor consumes shapes, layouts, dtypes, and operation topology; real
checkpoint weights are used for the subsequent PCC and latency decisions.

Run from a separately bootstrapped advisor shell as documented in work_log.md.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

TT_METAL_ROOT = Path(__file__).resolve().parents[6]
# Append: prepending the checkout would let its source ttnn/ shadow the package
# installed in the advisor environment.
if str(TT_METAL_ROOT) not in sys.path:
    sys.path.append(str(TT_METAL_ROOT))

from models.autoports.qwen_qwen2_5_coder_32b_instruct.tests.test_optimized_decoder import (  # noqa: E402
    _config,
    _empty_caches,
    _synthetic_state,
    _tt_tensor,
)
from models.autoports.qwen_qwen2_5_coder_32b_instruct.tt.optimized_decoder import (  # noqa: E402
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    REPRESENTATIVE_LAYER,
    OptimizationConfig,
    OptimizedDecoder,
)

_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None


def _build(device):
    config = _config()
    # The installed advisor tracer models paged_update_cache but not the fused
    # two-cache wrapper.  Using the semantically identical split update keeps
    # the dense attention + MLP graph authoritative for sharding advice.
    profile = replace(
        OptimizationConfig.named("packed_mlp_bfp8_hifi2_dram_gate40c"),
        name="advisor_capture_split_cache_update",
        fused_kv_update=False,
    )
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=device,
        batch=EMITTED_BATCH,
        optimization_config=profile,
    )
    key_cache, value_cache = _empty_caches(
        config,
        device,
        batch=EMITTED_BATCH,
        max_cache_len=EMITTED_CACHE_LENGTH,
        dtype=profile.kv_cache_dtype,
    )
    hidden = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=torch.Generator().manual_seed(25032026),
        dtype=torch.bfloat16,
    )
    return decoder, key_cache, value_cache, _tt_tensor(hidden, device)


def decode(hidden):
    """Trace one representative dense attention + dense MLP decode step."""

    return _DECODER.decode_forward(
        hidden,
        _KEY_CACHE,
        _VALUE_CACHE,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )


def make_inputs(device):
    global _DECODER, _KEY_CACHE, _VALUE_CACHE
    _DECODER, _KEY_CACHE, _VALUE_CACHE, hidden = _build(device)
    return (hidden,)
