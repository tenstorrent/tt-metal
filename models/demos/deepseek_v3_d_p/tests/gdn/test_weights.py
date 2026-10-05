# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the GDN device weight layout and its device-free cache (mock cluster, no device fixture).

Run with a mock cluster so ttnn touches no hardware:
    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_8xP150.yaml> pytest models/demos/deepseek_v3_d_p/tests/gdn/test_weights.py
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.deepseek_v3_d_p.reference.gdn.head_slice import gdn_head_slice_config, slice_gdn_heads
from models.demos.deepseek_v3_d_p.reference.gdn.layer import GDNReferenceState, delta_rule_recurrence
from models.demos.deepseek_v3_d_p.reference.gdn.tests.helpers import TINY, random_weights
from models.demos.deepseek_v3_d_p.reference.gdn.tests.test_head_slice import (
    _nonzero_state,
    _rank_partial_reference,
    _restrict_state,
)
from models.demos.deepseek_v3_d_p.tt.gdn.weights import (
    GDNHostWeights,
    GDNWeights,
    gdn_input_projection_widths,
    prepare_gdn_host_weights,
)

# Four K heads / twelve V heads: TP2 gives 6 V heads per rank, TP4 three, so the `a` block carries tile padding.
_CONFIG = replace(TINY, num_key_heads=4, num_value_heads=12)
_SHARD_DIMS = {"input_projection": -1, "output_projection": -2, "decay_scale": -1, "decay_bias": -1, "norm": None}


def _rank_block(host: GDNHostWeights, name: str, tp: int, rank: int) -> torch.Tensor:
    tensor = host.tensor(name)
    shard_dim = _SHARD_DIMS.get(name, -1)  # convolution taps shard on the last dim
    return tensor if shard_dim is None else tensor.chunk(tp, dim=shard_dim)[rank]


@pytest.mark.parametrize("tp", [2, 4])
def test_rank_shard_equals_tp1_layout_of_its_head_slice(tp: int) -> None:
    """TP rank r's shard of the TPn layout is exactly the TP1 layout of K-head group slice r."""
    weights = random_weights(_CONFIG)
    packed = prepare_gdn_host_weights(weights, _CONFIG, tp)
    heads = _CONFIG.num_key_heads // tp
    names = list(_SHARD_DIMS) + [f"conv_tap_{tap}" for tap in range(_CONFIG.conv_kernel_size)]
    for rank in range(tp):
        sliced = prepare_gdn_host_weights(
            slice_gdn_heads(weights, _CONFIG, key_head_start=rank * heads, num_key_heads=heads),
            gdn_head_slice_config(_CONFIG, heads),
            1,
        )
        for name in names:
            assert torch.equal(_rank_block(packed, name, tp, rank), sliced.tensor(name)), f"rank {rank} {name}"


def _rank_forward(host: GDNHostWeights, config, tp: int, rank: int, hidden, state: GDNReferenceState):
    """Consume one rank's blocks exactly as the documented device layout says (module docstring of tt/gdn/weights)."""
    widths = gdn_input_projection_widths(config, tp)
    key_heads, value_heads = config.num_key_heads // tp, config.num_value_heads // tp
    q, k, v, z, a, b = (hidden @ _rank_block(host, "input_projection", tp, rank).float()).split(
        list(widths.values()), -1
    )
    assert torch.equal(a[:, value_heads:], torch.zeros_like(a[:, value_heads:])), "a padding columns must be zero"
    a = a[:, :value_heads]
    qkv = torch.cat([q, k, v], -1)
    taps = [
        _rank_block(host, f"conv_tap_{tap}", tp, rank).float().reshape(-1) for tap in range(config.conv_kernel_size)
    ]
    padded = torch.cat([state.conv, qkv])
    T = hidden.shape[0]
    conv = F.silu(sum(padded[j : j + T] * taps[j] for j in range(config.conv_kernel_size)))
    q, k, v = conv.split([widths["q"], widths["k"], widths["v"]], -1)
    g = _rank_block(host, "decay_scale", tp, rank).reshape(-1) * F.softplus(
        a + _rank_block(host, "decay_bias", tp, rank).reshape(-1)
    )
    beta = torch.sigmoid(b)

    def l2(x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    q = (l2(q.reshape(T, key_heads, -1)) * config.head_k_dim**-0.5).repeat_interleave(config.group, 1)
    k = l2(k.reshape(T, key_heads, -1)).repeat_interleave(config.group, 1)
    o, _ = delta_rule_recurrence(q, k, v.reshape(T, value_heads, -1), g, beta, state.recurrent)
    norm = host.norm.float() * o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + config.norm_eps)
    gate = F.silu if config.output_gate_activation == "silu" else torch.sigmoid
    gated = (norm * gate(z.reshape(T, value_heads, -1))).reshape(T, -1)
    return gated @ _rank_block(host, "output_projection", tp, rank).float()


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("tp", [1, 4])
def test_rank_layout_reproduces_reference_rank_partial(tp: int, activation: str) -> None:
    """Round trip: each rank's device-layout blocks, consumed as documented, give the reference's TP-rank partial
    (nonzero carried state); the partials sum to the full layer."""
    config = replace(_CONFIG, output_gate_activation=activation)
    weights = {name: tensor.float() for name, tensor in random_weights(config).items()}
    host = prepare_gdn_host_weights(weights, config, tp)
    hidden = torch.randn(40, config.hidden_size, generator=torch.Generator().manual_seed(11))
    state = _nonzero_state(config, seed=12)
    heads = config.num_key_heads // tp
    for rank in range(tp):
        rank_state = _restrict_state(state, config, rank * heads, heads)
        expected = _rank_partial_reference(hidden, weights, config, state, rank * heads, heads)
        actual = _rank_forward(host, config, tp, rank, hidden, rank_state)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5, msg=f"rank {rank}")


def test_input_projection_widths_are_tile_aligned_blocks() -> None:
    """27B at TP4: 4 K / 12 V heads per rank -> q, k 512; v, z 1536; a 12 padded to 32; b 12."""
    config = replace(_CONFIG, hidden_size=5120, num_key_heads=16, num_value_heads=48, head_k_dim=128, head_v_dim=128)
    widths = gdn_input_projection_widths(config, 4)
    assert widths == {"q": 512, "k": 512, "v": 1536, "z": 1536, "a": 32, "b": 12}
    starts = torch.tensor(list(widths.values())).cumsum(0)[:-1]
    assert all(int(s) % 32 == 0 for s in starts)


def test_decay_parameters_are_fp32_and_signed() -> None:
    weights = random_weights(_CONFIG)
    host = prepare_gdn_host_weights(weights, _CONFIG, 2)
    assert host.decay_scale.dtype == torch.float32 and host.decay_bias.dtype == torch.float32
    assert torch.equal(host.decay_scale.reshape(-1), -weights["A_log"].float().exp())
    assert torch.equal(host.decay_bias.reshape(-1), weights["dt_bias"].float())


def test_rejects_tp_that_splits_a_key_group(expect_error) -> None:
    with expect_error(ValueError, "divisible by tensor parallel size 3"):
        prepare_gdn_host_weights(random_weights(_CONFIG), _CONFIG, 3)


@pytest.mark.parametrize(
    "mesh_shape, tensor_parallel_axis",
    [pytest.param((2, 4), 1, id="LB-A-mesh2x4-tpaxis1"), pytest.param((8, 1), 1, id="LB-B-mesh8x1-tpaxis1")],
)
def test_device_free_cache_is_complete_and_keyed(tmp_path: Path, mesh_shape, tensor_parallel_axis: int) -> None:
    """The device-free build writes every tensorbin; completeness is per mesh, TP axis, config and prefix; each
    stored shard holds the rank's host block."""
    config = replace(_CONFIG, hidden_size=64, head_k_dim=32, head_v_dim=32)
    weights = random_weights(config)
    prefix = "layer_0.gdn"
    assert not GDNWeights.check_cache_complete(
        tmp_path, prefix, config, mesh_shape, tensor_parallel_axis=tensor_parallel_axis
    )
    GDNWeights.build_ttnn_cache(
        weights, tmp_path, prefix, config, mesh_shape, tensor_parallel_axis=tensor_parallel_axis
    )
    files = sorted(tmp_path.glob("*.tensorbin"))
    assert len(files) == 9
    assert GDNWeights.check_cache_complete(
        tmp_path, prefix, config, mesh_shape, tensor_parallel_axis=tensor_parallel_axis
    )
    other_axis = 1 - tensor_parallel_axis
    assert not GDNWeights.check_cache_complete(tmp_path, prefix, config, mesh_shape, tensor_parallel_axis=other_axis)
    sigmoid = replace(config, output_gate_activation="sigmoid")
    assert not GDNWeights.check_cache_complete(
        tmp_path, prefix, sigmoid, mesh_shape, tensor_parallel_axis=tensor_parallel_axis
    )
    assert not GDNWeights.check_cache_complete(
        tmp_path, "layer_1.gdn", config, mesh_shape, tensor_parallel_axis=tensor_parallel_axis
    )

    tp = mesh_shape[tensor_parallel_axis]
    host = prepare_gdn_host_weights(weights, config, tp)
    for name, dtype in (("input_projection", torch.bfloat16), ("decay_scale", torch.float32)):
        (path,) = [f for f in files if f".{name}." in f.name]
        shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(ttnn.load_tensor(path))]
        # TP1 stores one host tensor that loading replicates (as KDA); TP > 1 stores one shard per mesh device.
        assert len(shards) == (mesh_shape[0] * mesh_shape[1] if tp > 1 else 1)
        for index, shard in enumerate(shards):
            rank = (index // mesh_shape[1], index % mesh_shape[1])[tensor_parallel_axis] if tp > 1 else 0
            expected = _rank_block(host, name, tp, rank).to(dtype)
            assert shard.dtype == dtype and torch.equal(
                shard.reshape(expected.shape), expected
            ), f"{name} shard {index}"
    (tmp_path / files[0].name).unlink()
    assert not GDNWeights.check_cache_complete(
        tmp_path, prefix, config, mesh_shape, tensor_parallel_axis=tensor_parallel_axis
    )
