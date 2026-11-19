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

from models.experimental.alpamayo2_edge.reference.action import expert_mask, expert_position_ids
from models.experimental.alpamayo2_edge.reference.text import DecoderLayer, TextModel, attention_bias, mrope_cos_sin
from models.experimental.alpamayo2_edge.tests._util import PCC, pcc, reference_cache
from models.experimental.alpamayo2_edge.tests.conftest import P100_MESH
from models.experimental.alpamayo2_edge.tt.expert import TtExpert, TtExpertLayer
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality


def _reference_conditioning(cfg, inputs, offsets: torch.Tensor, prefix_len: int):
    """Positions [3, B, N] and additive mask exactly as `ReferenceModel.expert_velocity` builds them (B = 1)."""
    n = cfg.action.n_waypoints
    positions = expert_position_ids(offsets, inputs.rope_deltas, n)
    mask = expert_mask(offsets, inputs.attention_mask, prefix_len, n)
    return positions, mask


def _layer0_rows(cache, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(tensor.to_torch(t)[:, :, :rows] for t in cache.layer_kv(0))


@P100_MESH
def test_expert_depth_mismatch_raises(mesh_device, cfg, expect_error):
    with expect_error(ValueError, "expert has 2 layers.*LM has 3"):
        TtExpert(replace(cfg.expert, layers=2), mesh_device=mesh_device, lm_num_layers=3)


@P100_MESH
@pytest.mark.timeout(1500)
def test_expert_layer(mesh_device, cfg, sample_inputs, prefix, expert_weights, expert_setup):
    inputs = sample_inputs
    ecfg = cfg.expert
    n = cfg.action.n_waypoints
    cache, frame, prefix_len = expert_setup
    logger.info(f"prefix {prefix_len} rows, p_pad {frame.p_pad}, cache {cache.size} rows")

    torch.manual_seed(0)
    # bf16-exact, so both sides start from the same input and `out - x` is the layer's update alone.
    x = torch.randn(1, n, ecfg.hidden).bfloat16().float()

    kv = reference_cache(prefix)
    positions, mask = _reference_conditioning(cfg, inputs, prefix["offsets"], prefix_len)
    cos, sin = mrope_cos_sin(positions, ecfg.head_dim, ecfg.rope_theta, ecfg.mrope_section)
    bias = attention_bias(n, prefix_len, mask, False, torch.float32)
    reference = DecoderLayer(ecfg, expert_weights, 0, torch.float32)
    with torch.no_grad():
        golden = reference(x, cos, sin, bias, kv)
    golden_own_k, golden_own_v = kv.k[0][:, :, prefix_len:], kv.v[0][:, :, prefix_len:]

    layer = TtExpertLayer(ecfg, mesh_device=mesh_device)
    state = TtExpert.convert_state(expert_weights)
    layer.load_torch_state_dict({k[len("layers.0.") :]: v for k, v in state.items() if k.startswith("layers.0.")})

    k_before, v_before = _layer0_rows(cache, frame.p_pad)
    tt_x = tensor.from_torch(x, device=mesh_device)
    first = tensor.to_torch(layer.forward(tt_x, layer_kv=cache.layer_kv(0), frame=frame))
    second = tensor.to_torch(layer.forward(tt_x, layer_kv=cache.layer_kv(0), frame=frame))
    k_after, v_after = (tensor.to_torch(t) for t in cache.layer_kv(0))

    rows = slice(frame.p_pad, frame.p_pad + n)
    own_k, own_v = k_after[:, :, rows], v_after[:, :, rows]
    logger.info(
        f"layer 0 PCC {pcc(golden, first) * 100:.4f} %, update PCC {pcc(golden - x, first - x) * 100:.4f} %, "
        f"own K PCC {pcc(golden_own_k, own_k) * 100:.4f} %, own V PCC {pcc(golden_own_v, own_v) * 100:.4f} %"
    )
    assert first.shape == golden.shape
    # The second call overwrites the rows the first one appended, so it sees the same keys.
    assert torch.equal(first, second)
    # The prefix and the poisoned rows after it are left as they were.
    assert torch.equal(k_after[:, :, : frame.p_pad], k_before)
    assert torch.equal(v_after[:, :, : frame.p_pad], v_before)
    assert_quality(golden, first, pcc=PCC)
    # The residual dominates the output; the layer's own update must match too.
    assert_quality(golden - x, first - x, pcc=PCC)
    assert_quality(golden_own_k, own_k, pcc=PCC)
    assert_quality(golden_own_v, own_v, pcc=PCC)


@P100_MESH
@pytest.mark.timeout(1500)
def test_expert_stack(mesh_device, cfg, sample_inputs, prefix, expert_weights, expert_setup):
    inputs = sample_inputs
    ecfg = cfg.expert
    n = cfg.action.n_waypoints
    cache, frame, prefix_len = expert_setup
    logger.info(f"prefix {prefix_len} rows, p_pad {frame.p_pad}, {ecfg.layers} expert layers")

    torch.manual_seed(0)
    # bf16-exact, so both sides start from the same input and `out - x` is the layer's update alone.
    x = torch.randn(1, n, ecfg.hidden).bfloat16().float()

    kv = reference_cache(prefix)
    positions, mask = _reference_conditioning(cfg, inputs, prefix["offsets"], prefix_len)
    reference = TextModel(ecfg, expert_weights, has_embedding=False)
    with torch.no_grad():
        golden = reference(x, positions, cache=kv, attn_mask=mask, causal=False)

    expert = TtExpert(ecfg, mesh_device=mesh_device, lm_num_layers=cfg.text.layers)
    expert.load_torch_state_dict(TtExpert.convert_state(expert_weights))
    k_before, v_before = _layer0_rows(cache, frame.p_pad)
    tt_x = tensor.from_torch(x, device=mesh_device)
    first = tensor.to_torch(expert.forward(tt_x, cache=cache, frame=frame))
    second = tensor.to_torch(expert.forward(tt_x, cache=cache, frame=frame))
    k_after, v_after = _layer0_rows(cache, frame.p_pad)

    logger.info(f"expert stack PCC {pcc(golden, first) * 100:.4f} %")
    assert first.shape == golden.shape
    assert torch.equal(first, second)
    assert torch.equal(k_after, k_before)
    assert torch.equal(v_after, v_before)
    assert_quality(golden, first, pcc=PCC, ccc=PCC)
