# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks that substate() keeps the warm-cache placeholder marker the attention loader branches on."""

import torch

from models.common.weight_cache import CachedStateDict
from models.demos.gemma4_d_p.utils.substate import substate

LAYER = "model.language_model.layers.0"
MANIFEST = {f"{LAYER}.self_attn.q_proj.weight": ((8, 4), "torch.bfloat16")}


def test_substate_keeps_placeholder_marker():
    layer = substate(CachedStateDict(MANIFEST, {}), LAYER)
    attn = substate(layer, "self_attn")
    assert getattr(attn, "is_placeholder", False)
    assert attn["q_proj.weight"].shape == (8, 4)


def test_substate_of_real_weights_has_no_marker():
    attn = substate({f"{LAYER}.self_attn.q_proj.weight": torch.zeros(8, 4)}, f"{LAYER}.self_attn")
    assert not getattr(attn, "is_placeholder", False)
    assert set(attn) == {"q_proj.weight"}
