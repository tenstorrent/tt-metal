# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only regressions for expert-parallel / mesh-axis agreement.

``ttnn.moe_routing_remap`` requires ``expert_parallel_size == mesh_shape[cluster_axis]``.
Two places have to hold that up: ``MeshConfig`` must not accept an EP that disagrees with
its own ``ep_axis``, and the decode expert path must forward the configured EP and axis
instead of hardcoded literals. Neither needs a device to check.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models.demos.gpt_oss.config import MeshConfig, ModeConfig, mesh_1x8, mesh_2x4, mesh_4x4, mesh_4x8
from models.demos.gpt_oss.tt.experts.config import ExpertConfig
from models.demos.gpt_oss.tt.experts.decode import decode_forward


class _RemapCallCaptured(Exception):
    """Unwinds decode_forward once the call under test has been recorded."""

    def __init__(self, args, kwargs):
        super().__init__("captured moe_routing_remap call")
        self.captured_args = args
        self.captured_kwargs = kwargs


@pytest.mark.parametrize(
    "mesh_config_factory, expected_ep",
    [
        # 2x4 is the mesh the old hardcoded (nnz=4, ep=4, axis=0) call broke on: it declared
        # 4 partitions while running on 2 rows, and now trips the live mesh-axis TT_FATAL.
        (mesh_2x4, 2),
        # 4x8/4x4 are where the hardcoded literals happened to be right; asserting them here
        # pins that this change is a no-op on the meshes gpt_oss runs on today.
        (mesh_4x8, 4),
        (mesh_4x4, 4),
    ],
    ids=["mesh_2x4", "mesh_4x8", "mesh_4x4"],
)
def test_decode_forward_forwards_configured_ep_and_axis(mesh_config_factory, expected_ep):
    """decode_forward must pass the configured EP and ep_axis to moe_routing_remap.

    Reverting either argument to a literal leaves the operator-level tests green, so assert
    the arguments directly. ttnn is patched out, which stops the forward at the call we care
    about and keeps this runnable without a device.
    """
    mesh_config = mesh_config_factory()
    assert expected_ep == mesh_config.mesh_shape[mesh_config.ep_axis]

    config = ExpertConfig(
        intermediate_size=64,
        num_experts=32,
        hidden_size=64,
        num_experts_per_tok=4,
        swiglu_limit=7.0,
    )
    reshaped_sparsity = object()

    with patch("models.demos.gpt_oss.tt.experts.decode.ttnn") as mock_ttnn:
        mock_ttnn.reshape.return_value = reshaped_sparsity

        def capture(*args, **kwargs):
            raise _RemapCallCaptured(args, kwargs)

        mock_ttnn.moe_routing_remap.side_effect = capture

        with pytest.raises(_RemapCallCaptured) as exc_info:  # allow-pytest.raises: control-flow sentinel
            decode_forward(
                hidden_states=SimpleNamespace(shape=[1, 1, 1, config.hidden_size]),
                routing_weights=MagicMock(),
                weights=MagicMock(),
                config=config,
                mesh_config=mesh_config,
                mesh_device=MagicMock(),
                ccl_manager=MagicMock(),
                program_config=MagicMock(),
            )

    assert exc_info.value.captured_args == (
        reshaped_sparsity,
        config.num_experts_per_tok,
        expected_ep,
        mesh_config.ep_axis,
    )
    assert exc_info.value.captured_kwargs == {}


def test_mesh_config_rejects_ep_not_matching_ep_axis(expect_error):
    """EP must agree with its own axis at construction, not as a TT_FATAL mid-forward.

    tp x dp x ep == total_devices and tp <= mesh_shape[tp_axis] both pass here, so before
    the EP check this config built fine and then hard-failed inside moe_routing_remap on the
    first decode step.
    """
    with expect_error(ValueError, r"decode: EP\(2\) != mesh_0_size\(4\)"):
        MeshConfig((4, 8), decode=ModeConfig(tp=8, ep=2))


@pytest.mark.parametrize(
    "mesh_config_factory",
    [mesh_1x8, mesh_2x4, mesh_4x4, mesh_4x8],
    ids=["mesh_1x8", "mesh_2x4", "mesh_4x4", "mesh_4x8"],
)
def test_mesh_config_factories_satisfy_ep_axis_constraint(mesh_config_factory):
    """The EP check must not reject any shipped config.

    Prefill runs EP=1 on multi-row meshes and never reaches the remap, so EP=1 stays
    unconstrained; only EP>1 has to match the axis extent.
    """
    mesh_config = mesh_config_factory()
    ep_dim_size = mesh_config.mesh_shape[mesh_config.ep_axis]

    for mode_config in (mesh_config.decode, mesh_config.prefill):
        assert mode_config.ep == 1 or mode_config.ep == ep_dim_size
