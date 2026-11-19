# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fixtures of the expert and action tests: the reference's teacher-forced prefix cache, loaded into a TT cache."""

import pytest
import torch

import ttnn
from models.experimental.alpamayo2_edge.reference import golden_cache
from models.experimental.alpamayo2_edge.reference.action import expert_mask
from models.experimental.alpamayo2_edge.reference.forced_cot import driving_text_cot
from models.experimental.alpamayo2_edge.reference.model import ReferenceModel
from models.experimental.alpamayo2_edge.reference.weights import load_weights
from models.experimental.alpamayo2_edge.tests._util import reference_cache
from models.experimental.alpamayo2_edge.tt.common import round_up
from models.experimental.alpamayo2_edge.tt.expert import MASKED, ExpertFrame, make_expert_frame
from models.experimental.alpamayo2_edge.tt.generate import CotResult
from models.experimental.alpamayo2_edge.tt.text import AlpamayoCache
from models.tt_dit.utils import tensor

# Fills the cache rows between the prefix and p_pad: any attention weight that leaks onto them breaks the PCC.
POISON = 1000.0


def _reference_prefix(cfg, inputs, forced: torch.Tensor) -> dict:
    """The teacher-forced CoT and its prefix cache, K and V of every layer in bf16 as the TT cache holds them."""
    model = ReferenceModel.from_pretrained(cfg)
    steps = []

    def progress(_m, _a, _out):
        steps.append(None)
        print(f"reference LM forward {len(steps)}", flush=True)

    model.lm.text.register_forward_hook(progress)
    _, cache = model.prefill(inputs)
    cot = model.decode(inputs, cache, forced=forced)
    return {
        "tokens": cot.tokens,
        "offsets": cot.offsets,
        "k": [k.to(torch.bfloat16) for k in cache.k],
        "v": [v.to(torch.bfloat16) for v in cache.v],
    }


@pytest.fixture(scope="module")
def prefix(cfg, sample_inputs):
    inputs = sample_inputs
    forced = driving_text_cot(cfg)[None]
    golden = golden_cache.cached(
        "expert_prefix_cache",
        {
            "input_ids": inputs.input_ids,
            "position_ids": inputs.position_ids,
            "patches": inputs.patches,
            "grid": inputs.image_grid_thw,
            "forced": forced,
        },
        lambda: _reference_prefix(cfg, inputs, forced),
        cfg=cfg,
    )
    assert torch.equal(golden["tokens"], forced)
    return golden


@pytest.fixture(scope="module")
def expert_weights(model_dir):
    return load_weights(model_dir, "expert.expert.")


@pytest.fixture
def expert_setup(mesh_device, cfg, sample_inputs, prefix) -> tuple[AlpamayoCache, ExpertFrame, int]:
    """The prefix in a TT cache with rows [prefix_len, p_pad) poisoned, the frame built over it, and prefix_len."""
    inputs = sample_inputs
    n = cfg.action.n_waypoints
    offsets = prefix["offsets"]
    prefix_len = int(offsets.max())
    p_pad = round_up(prefix_len, ttnn.TILE_SIZE)
    assert p_pad > prefix_len, "the sample must leave rows between the prefix and p_pad to poison"
    size = AlpamayoCache.size_for(inputs.input_ids.shape[1], max_new_tokens=256, action_rows=n)
    assert size >= p_pad + n

    kv = reference_cache(prefix)
    pad = (0, 0, 0, p_pad - prefix_len)
    kv.k = [torch.nn.functional.pad(k, pad, value=POISON) for k in kv.k]
    kv.v = [torch.nn.functional.pad(v, pad, value=POISON) for v in kv.v]
    cache = AlpamayoCache.from_reference(kv, size=size, mesh_device=mesh_device)
    cot = CotResult(tokens=prefix["tokens"], offsets=offsets, prefix_length=prefix_len, logits=None, cache=cache)
    frame = make_expert_frame(cot, inputs, ecfg=cfg.expert, n_waypoints=n, mesh_device=mesh_device)

    assert frame.p_pad == p_pad
    mask = tensor.to_torch(frame.mask).bfloat16()
    expected = expert_mask(offsets, inputs.attention_mask, p_pad, n).clamp(min=MASKED).bfloat16()
    assert mask.shape == (1, 1, n, p_pad + n)
    assert torch.equal(mask, expected)
    assert bool((mask[..., prefix_len:p_pad] < -1e8).all())
    assert bool((mask[..., p_pad:] == 0).all())
    return cache, frame, prefix_len
