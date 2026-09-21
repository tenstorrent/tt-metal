# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The seam between the shared prefill runtime and Kimi-K3's own transformer.

`TtPrefillRuntime._build_model` calls `self.MODEL_CLS(...)` with a fixed keyword list. Kimi-K3's
transformer names the model-level knobs it understands and forwards the rest to `TtKimiK3Block` as
`**block_kwargs`, so a knob added upstream and NOT named here does not fail at the seam -- it
silently becomes a block argument and raises `TypeError` on the first layer, after mesh bringup and
weight load. That is a long way to travel for a signature mismatch, and it is only reachable through
`build_runtime`, which no eager test exercises.

`tp_shard_kv` did exactly this: it arrived with GLM-5.2's 2D KV sharding (#51968), after the branch's
merge-base, and broke every runner construction until it was named.

Hardware-free: this reads the call site's keyword list out of the source and binds it against the two
signatures. No mesh, no weights, no checkpoint.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from models.demos.deepseek_v3_d_p.tt.kimi_k3.block import TtKimiK3Block
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer

_RUNTIME_SRC = Path(__file__).parents[2] / "tt" / "tt_prefill_runtime.py"


def _model_cls_kwargs() -> list[str]:
    """The keywords `_build_model` passes to `MODEL_CLS`, read off the call site."""
    src = _RUNTIME_SRC.read_text(encoding="utf-8")
    start = src.index("self.MODEL_CLS(")
    call = src[start : src.index("\n        )\n", start)]
    names = re.findall(r"^\s{12}(\w+)=", call, re.M)
    assert names, "could not parse the MODEL_CLS call site; the test needs updating, not deleting"
    return names


def test_every_runtime_kwarg_binds_to_the_transformer_or_the_block():
    passed = _model_cls_kwargs()
    named = set(inspect.signature(TtKimiK3Transformer.__init__).parameters)
    accepted = set(inspect.signature(TtKimiK3Block.__init__).parameters)

    swept = [k for k in passed if k not in named]
    orphans = [k for k in swept if k not in accepted]
    assert not orphans, (
        f"{orphans} reach TtKimiK3Transformer from TtPrefillRuntime._build_model, are not named in "
        f"its signature, and are not accepted by TtKimiK3Block -- so build_runtime raises TypeError "
        f"on the first layer. Name them in TtKimiK3Transformer.__init__ (rejecting the values "
        f"Kimi-K3 cannot honour) rather than widening TtKimiK3Block."
    )


def test_the_block_kwargs_that_are_swept_through_are_genuinely_block_level():
    """A model-level knob landing in the block is a bug even when the block happens to accept it."""
    passed = _model_cls_kwargs()
    named = set(inspect.signature(TtKimiK3Transformer.__init__).parameters)
    swept = {k for k in passed if k not in named}
    # Expert dtypes, the dispatch capacity factor and the routing/semaphore placement are per-block
    # MoE construction arguments. Anything else appearing here is a model-level knob that has drifted
    # into the block path and should be named on the transformer instead.
    expected = {
        "dispatch_buffer_capacity_factor",
        "routed_expert_activations_dtype",
        "routed_expert_weights_dtype",
        "shared_expert_activations_dtype",
        "shared_expert_weights_dtype",
        "routing_use_l1_small_for_semaphores",
        "overlap_shared_expert_with_dispatch",
    }
    assert swept == expected, (
        f"the set of kwargs forwarded to TtKimiK3Block changed: +{sorted(swept - expected)} "
        f"-{sorted(expected - swept)}. If the addition is model-level, name it on the transformer."
    )
