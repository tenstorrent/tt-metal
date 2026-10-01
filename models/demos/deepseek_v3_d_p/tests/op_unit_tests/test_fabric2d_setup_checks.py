# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the setup and input checks of TtDispatch2dModule and TtCombine2dModule, and for
the fabric payload the prefill runner opens the mesh with when a model uses them.

No device is opened: the fabric queries are replaced with stubs, and the mesh and tensors are plain
objects that only have what the checks read.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tt.moe.fabric2d_contract import check_fabric2d_setup, fabric2d_payload_size
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombine2dModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatch2dModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe

FC = ttnn.FabricConfig
EMB_DIM = 7168
# 7168 bf16 values (14336 B, already a multiple of 64) plus 64 B of forwarding metadata.
EXACT_PAYLOAD = 14400


@pytest.fixture
def fabric(monkeypatch):
    """Returns a setter for the fabric config and max payload the checks read. Defaults pass."""

    def set_fabric(config=FC.FABRIC_2D_TORUS_XY, max_payload=EXACT_PAYLOAD):
        monkeypatch.setattr(ttnn, "get_fabric_config", lambda: config)
        monkeypatch.setattr(ttnn, "get_tt_fabric_max_payload_size_bytes", lambda: max_payload)

    set_fabric()
    return set_fabric


def _mesh(rows=8, cols=4):
    return SimpleNamespace(shape=(rows, cols))


@pytest.mark.parametrize("rows", [4, 8])
def test_even_ring_of_four_or_more_passes(fabric, rows):
    check_fabric2d_setup(_mesh(rows), cluster_axis=0, emb_dim=EMB_DIM)


@pytest.mark.parametrize("rows", [2, 5])
def test_short_or_odd_ring_raises(expect_error, fabric, rows):
    with expect_error(ValueError, "even ring of at least 4 chips"):
        check_fabric2d_setup(_mesh(rows), cluster_axis=0, emb_dim=EMB_DIM)


@pytest.mark.parametrize(
    "cluster_axis, config",
    [(0, FC.FABRIC_2D_TORUS_Y), (0, FC.FABRIC_2D_TORUS_XY), (1, FC.FABRIC_2D_TORUS_X), (1, FC.FABRIC_2D_TORUS_XY)],
)
def test_fabric_that_wraps_the_axis_passes(fabric, cluster_axis, config):
    fabric(config=config)
    check_fabric2d_setup(_mesh(8, 4), cluster_axis=cluster_axis, emb_dim=EMB_DIM)


# FABRIC_1D_RING wraps axis 0 but is not a 2D fabric.
@pytest.mark.parametrize(
    "cluster_axis, config",
    [
        (0, FC.FABRIC_2D),
        (0, FC.FABRIC_2D_TORUS_X),
        (0, FC.FABRIC_1D_RING),
        (1, FC.FABRIC_2D_TORUS_Y),
    ],
)
def test_fabric_that_does_not_wrap_the_axis_raises(expect_error, fabric, cluster_axis, config):
    fabric(config=config)
    with expect_error(ValueError, "2D fabric that wraps"):
        check_fabric2d_setup(_mesh(8, 4), cluster_axis=cluster_axis, emb_dim=EMB_DIM)


# 500 bf16 values are 1000 B. dispatch_fabric2d pads that to a 1024 B page (a multiple of the 64 B DRAM
# alignment); combine_fabric2d would reject it, but the rounded-up size is right for both.
@pytest.mark.parametrize("emb_dim, exact_payload", [(EMB_DIM, EXACT_PAYLOAD), (500, 1024 + 64)])
def test_payload_exact_passes_one_byte_short_raises(expect_error, fabric, emb_dim, exact_payload):
    fabric(max_payload=exact_payload)
    check_fabric2d_setup(_mesh(), cluster_axis=0, emb_dim=emb_dim)
    fabric(max_payload=exact_payload - 1)
    with expect_error(ValueError, "fabric max payload"):
        check_fabric2d_setup(_mesh(), cluster_axis=0, emb_dim=emb_dim)


def _dispatch(mesh, num_experts_per_tok=2):
    return TtDispatch2dModule(
        mesh_device=mesh,
        experts_per_chip=2,
        num_routed_experts=64,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatch_buffer_token_size=1024,
        seq_len_per_chip=64,
        emb_dim=EMB_DIM,
    )


def _combine(mesh):
    return TtCombine2dModule(
        mesh_device=mesh,
        experts_per_chip=2,
        num_experts_per_tok=2,
        seq_len_per_chip=64,
        emb_dim=EMB_DIM,
    )


# The stub mesh has no device methods, so a ValueError here means the check ran before any device work.
@pytest.mark.parametrize("make_module", [_dispatch, _combine], ids=["dispatch", "combine"])
def test_wrappers_check_before_device_work(expect_error, fabric, make_module):
    make_module(_mesh(8, 4))
    with expect_error(ValueError, "even ring"):
        make_module(_mesh(2, 4))
    fabric(config=FC.FABRIC_2D)
    with expect_error(ValueError, "2D fabric that wraps"):
        make_module(_mesh(8, 4))


def test_dispatch_rejects_top_k_above_8(expect_error, fabric):
    _dispatch(_mesh(8, 4), num_experts_per_tok=8)
    with expect_error(ValueError, "top-k up to 8"):
        _dispatch(_mesh(8, 4), num_experts_per_tok=16)


def test_combine_has_no_top_k_limit(fabric):
    TtCombine2dModule(
        mesh_device=_mesh(8, 4), experts_per_chip=2, num_experts_per_tok=16, seq_len_per_chip=64, emb_dim=EMB_DIM
    )


def _tensor(memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return SimpleNamespace(memory_config=lambda: memory_config)


@pytest.mark.parametrize(
    "name",
    [
        "x",
        "indices",
        "expert_dispatch_table",
        "expert_token_counts",
        "expert_region_offsets",
        "all_expert_offsets",
        "padding_config",
    ],
)
def test_dispatch_rejects_input_not_in_dram(expect_error, fabric, name):
    inputs = {
        "x": _tensor(),
        "indices": _tensor(),
        "expert_dispatch_table": _tensor(),
        "expert_token_counts": _tensor(),
        "expert_region_offsets": _tensor(),
        "all_expert_offsets": _tensor(),
        "padding_config": _tensor(),
    }
    inputs[name] = _tensor(ttnn.L1_MEMORY_CONFIG)
    with expect_error(ValueError, f"{name} must be interleaved in DRAM"):
        _dispatch(_mesh(8, 4))(**inputs)


def _tt_moe(monkeypatch, **overrides):
    """TtMoe on the stub mesh. Its first device work is the CCL lookup, stubbed here, then the gate."""
    monkeypatch.setattr("models.demos.deepseek_v3_d_p.tt.moe.tt_moe.get_tt_ccl", lambda mesh_device: None)
    kwargs = dict(
        mesh_device=_mesh(8, 4),
        dispatch_group_size=8,
        num_dispatch_groups=4,
        experts_per_chip=8,
        num_routed_experts=256,
        num_experts_per_tok=8,
        metadata_len=3,
        max_dispatched_tokens_per_expert=5120,
        max_dispatch_buffer_token_size=5120,
        seq_len_per_chip=640,
        gate_weights=None,
        emb_dim=EMB_DIM,
        hidden_dim=2048,
        n_expert_groups=8,
        n_limited_groups=4,
        route_scale=2.5,
        topology=(ttnn.Topology.Ring, ttnn.Topology.Ring),
    )
    kwargs.update(overrides)
    return TtMoe(**kwargs)


@pytest.mark.parametrize("name", ["dispatch_impl", "combine_impl"])
def test_tt_moe_rejects_unknown_impl(expect_error, fabric, monkeypatch, name):
    with expect_error(ValueError, f"{name} must be one of"):
        _tt_moe(monkeypatch, **{name: "fabric_2d"})


@pytest.mark.parametrize("name", ["dispatch_impl", "combine_impl"])
def test_tt_moe_fabric2d_checks_before_device_work(expect_error, fabric, monkeypatch, name):
    with expect_error(ValueError, "axis-0 topology is"):
        _tt_moe(monkeypatch, topology=(ttnn.Topology.Linear, ttnn.Topology.Ring), **{name: "fabric2d"})
    with expect_error(ValueError, "3 metadata fields"):
        _tt_moe(monkeypatch, metadata_len=4, **{name: "fabric2d"})
    with expect_error(ValueError, "even ring"):
        _tt_moe(monkeypatch, mesh_device=_mesh(2, 4), **{name: "fabric2d"})
    fabric(max_payload=EXACT_PAYLOAD - 1)
    with expect_error(ValueError, "fabric max payload"):
        _tt_moe(monkeypatch, **{name: "fabric2d"})


def test_tt_moe_fabric2d_dispatch_rejects_top_k_above_8(expect_error, fabric, monkeypatch):
    with expect_error(ValueError, "top-k up to 8"):
        _tt_moe(monkeypatch, num_experts_per_tok=16, dispatch_impl="fabric2d")


@pytest.mark.parametrize("uses_fabric2d", [False, True])
def test_tt_moe_refuses_a_trace_controller_with_fabric2d(expect_error, uses_fabric2d):
    tt_moe = object.__new__(TtMoe)
    tt_moe.uses_fabric2d = uses_fabric2d
    tt_moe.set_trace_controller(None)
    if uses_fabric2d:
        with expect_error(ValueError, "cannot be traced"):
            tt_moe.set_trace_controller(object())
    else:
        tt_moe.set_trace_controller(object())


# A token of the routed width in bf16, rounded up to 64 B, plus 64 B of forwarding metadata. Kimi-K3 routes
# at its 3584 latent width, so its 7168 B own payload is too small by the 64 B of metadata.
@pytest.mark.parametrize(
    "model_cfg, expected",
    [
        (DeepSeekV3Config, 14400),
        (KimiK26Config, 14400),
        (KimiK27Config, 14400),
        (DeepSeekV4ProConfig, 14400),
        (GLM53Config, 12352),
        (KimiK3Config, 7232),
        (DeepSeekV4FlashConfig, 8256),
        (MistralSmall4Config, 8256),
    ],
    ids=lambda p: p.__name__ if isinstance(p, type) else str(p),
)
def test_fabric2d_payload_size(model_cfg, expected):
    assert fabric2d_payload_size(model_cfg) == expected


def test_fabric2d_payload_size_keeps_a_larger_model_payload():
    model_cfg = SimpleNamespace(EMB_SIZE=1024, FABRIC_PAYLOAD_SIZE=9000)
    assert fabric2d_payload_size(model_cfg) == 9000
