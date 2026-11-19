# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TT action expert against the torch reference on one Blackhole P100.

The expert attends over the reference's teacher-forced prefix cache, loaded into the TT cache as is, so the TT LM
plays no part and both sides see the same prefix rows.
"""

from dataclasses import replace

import pytest
import torch
from loguru import logger

from models.experimental.alpamayo2_edge.reference import golden_cache
from models.experimental.alpamayo2_edge.reference.action import expert_mask, expert_position_ids
from models.experimental.alpamayo2_edge.reference.forced_cot import driving_text_cot
from models.experimental.alpamayo2_edge.reference.inputs import prepare_sample_inputs, synthetic_sample
from models.experimental.alpamayo2_edge.reference.model import ReferenceModel
from models.experimental.alpamayo2_edge.reference.text import (
    DecoderLayer,
    KVCache,
    TextModel,
    attention_bias,
    mrope_cos_sin,
)
from models.experimental.alpamayo2_edge.reference.weights import load_weights
from models.experimental.alpamayo2_edge.tests.conftest import P100_MESH
from models.experimental.alpamayo2_edge.tt.expert import (
    ACTION_TOKENS,
    TtExpert,
    TtExpertLayer,
    make_expert_frame,
)
from models.experimental.alpamayo2_edge.tt.generate import CotResult
from models.experimental.alpamayo2_edge.tt.text import AlpamayoCache
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality

PCC = 0.99


def _ceil(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


def _pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    stacked = torch.stack([golden.flatten().double(), actual.flatten().double()])
    return torch.corrcoef(stacked)[0, 1].item()


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
def sample_inputs(cfg):
    return prepare_sample_inputs(cfg, synthetic_sample(0))


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


def _reference_cache(prefix: dict) -> KVCache:
    kv = KVCache(len(prefix["k"]))
    kv.k = [k.float() for k in prefix["k"]]
    kv.v = [v.float() for v in prefix["v"]]
    return kv


def _reference_conditioning(cfg, inputs, offsets: torch.Tensor, prefix_len: int):
    """Positions [3, B, 64] and additive mask exactly as `ReferenceModel.expert_velocity` builds them (B = 1)."""
    positions = expert_position_ids(offsets, inputs.rope_deltas, ACTION_TOKENS)
    mask = expert_mask(offsets, inputs.attention_mask, prefix_len, ACTION_TOKENS)
    return positions, mask


def _tt_setup(mesh_device, cfg, inputs, prefix):
    seq = inputs.input_ids.shape[1]
    offsets = prefix["offsets"]
    prefix_len = int(offsets.max())
    size = AlpamayoCache.size_for(seq)
    assert size >= _ceil(prefix_len, 32) + ACTION_TOKENS
    cache = AlpamayoCache.from_reference(_reference_cache(prefix), size=size, mesh_device=mesh_device)
    cot = CotResult(tokens=prefix["tokens"], offsets=offsets, prefix_length=prefix_len, logits=None, cache=cache)
    frame = make_expert_frame(
        cot, inputs, mesh_device=mesh_device, rope_theta=cfg.expert.rope_theta, head_dim=cfg.expert.head_dim
    )
    assert frame.p_pad == _ceil(prefix_len, 32)
    assert tuple(frame.mask.shape) == (1, 1, ACTION_TOKENS, frame.p_pad + ACTION_TOKENS)
    return cache, frame, prefix_len


@P100_MESH
def test_expert_depth_mismatch_raises(mesh_device, cfg, expect_error):
    with expect_error(ValueError, "expert has 2 layers.*LM has 3"):
        TtExpert(replace(cfg.expert, layers=2), mesh_device=mesh_device, lm_num_layers=3)


@P100_MESH
@pytest.mark.timeout(1500)
def test_expert_layer(mesh_device, cfg, sample_inputs, prefix, expert_weights):
    inputs = sample_inputs
    ecfg = cfg.expert
    cache, frame, prefix_len = _tt_setup(mesh_device, cfg, inputs, prefix)
    logger.info(f"prefix {prefix_len} rows, p_pad {frame.p_pad}, cache {cache.size} rows")

    torch.manual_seed(0)
    # bf16-exact, so both sides start from the same input and `out - x` is the layer's update alone.
    x = torch.randn(1, ACTION_TOKENS, ecfg.hidden).bfloat16().float()

    kv = _reference_cache(prefix)
    positions, mask = _reference_conditioning(cfg, inputs, prefix["offsets"], prefix_len)
    cos, sin = mrope_cos_sin(positions, ecfg.head_dim, ecfg.rope_theta, ecfg.mrope_section)
    bias = attention_bias(ACTION_TOKENS, prefix_len, mask, False, torch.float32)
    reference = DecoderLayer(ecfg, expert_weights, 0, torch.float32)
    with torch.no_grad():
        golden = reference(x, cos, sin, bias, kv)
    golden_own_k, golden_own_v = kv.k[0][:, :, prefix_len:], kv.v[0][:, :, prefix_len:]

    layer = TtExpertLayer(ecfg, mesh_device=mesh_device)
    state = TtExpert.convert_state(expert_weights)
    layer.load_torch_state_dict({k[len("layers.0.") :]: v for k, v in state.items() if k.startswith("layers.0.")})

    k_before, v_before = (tensor.to_torch(t)[:, :, :prefix_len] for t in cache.layer_kv(0))
    tt_x = tensor.from_torch(x, device=mesh_device)
    first = tensor.to_torch(layer.forward(tt_x, layer_kv=cache.layer_kv(0), frame=frame))
    second = tensor.to_torch(layer.forward(tt_x, layer_kv=cache.layer_kv(0), frame=frame))
    k_after, v_after = (tensor.to_torch(t) for t in cache.layer_kv(0))

    rows = slice(frame.p_pad, frame.p_pad + ACTION_TOKENS)
    own_k, own_v = k_after[:, :, rows], v_after[:, :, rows]
    logger.info(
        f"layer 0 PCC {_pcc(golden, first) * 100:.4f} %, update PCC {_pcc(golden - x, first - x) * 100:.4f} %, "
        f"own K PCC {_pcc(golden_own_k, own_k) * 100:.4f} %, own V PCC {_pcc(golden_own_v, own_v) * 100:.4f} %"
    )
    assert first.shape == golden.shape
    # The second call overwrites the rows the first one appended, so it sees the same keys.
    assert torch.equal(first, second)
    assert torch.equal(k_after[:, :, :prefix_len], k_before)
    assert torch.equal(v_after[:, :, :prefix_len], v_before)
    assert_quality(golden, first, pcc=PCC)
    # The residual dominates the output; the layer's own update must match too.
    assert_quality(golden - x, first - x, pcc=PCC)
    assert_quality(golden_own_k, own_k, pcc=PCC)
    assert_quality(golden_own_v, own_v, pcc=PCC)


@P100_MESH
@pytest.mark.timeout(1500)
def test_expert_stack(mesh_device, cfg, sample_inputs, prefix, expert_weights):
    inputs = sample_inputs
    ecfg = cfg.expert
    cache, frame, prefix_len = _tt_setup(mesh_device, cfg, inputs, prefix)
    logger.info(f"prefix {prefix_len} rows, p_pad {frame.p_pad}, {ecfg.layers} expert layers")

    torch.manual_seed(0)
    # bf16-exact, so both sides start from the same input and `out - x` is the layer's update alone.
    x = torch.randn(1, ACTION_TOKENS, ecfg.hidden).bfloat16().float()

    kv = _reference_cache(prefix)
    positions, mask = _reference_conditioning(cfg, inputs, prefix["offsets"], prefix_len)
    reference = TextModel(ecfg, expert_weights, has_embedding=False)
    with torch.no_grad():
        golden = reference(x, positions, cache=kv, attn_mask=mask, causal=False)

    expert = TtExpert(ecfg, mesh_device=mesh_device, lm_num_layers=cfg.text.layers)
    expert.load_torch_state_dict(TtExpert.convert_state(expert_weights))
    tt_x = tensor.from_torch(x, device=mesh_device)
    first = tensor.to_torch(expert.forward(tt_x, cache=cache, frame=frame))
    second = tensor.to_torch(expert.forward(tt_x, cache=cache, frame=frame))

    logger.info(f"expert stack PCC {_pcc(golden, first) * 100:.4f} %")
    assert first.shape == golden.shape
    assert torch.equal(first, second)
    assert_quality(golden, first, pcc=PCC)
