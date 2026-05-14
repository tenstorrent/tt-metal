# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit test for the ``zero_centered`` kwarg on ``DistributedNorm``.

When ``zero_centered=True`` the constructor must transform the inner norm's
weight tensor by ``w' = w + 1`` and re-upload it to the device.  We exercise
this by mocking ``ttnn.to_torch`` / ``ttnn.from_torch`` / ``ttnn.deallocate``
and recording the source tensor passed to ``ttnn.from_torch``.

This test does not touch a real device.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.distributed_norm import DistributedNorm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(dim: int = 5120, qk_norm: bool = False) -> MagicMock:
    """Minimal mock of the ModelArgs surface ``DistributedNorm.__init__`` reads."""
    mesh_device = MagicMock()
    # Simulate an 8x4 mesh (rows, cols).
    mesh_device.shape = (8, 4)

    args = MagicMock()
    args.mesh_device = mesh_device
    args.dim = dim
    args.qk_norm = qk_norm
    return args


def _make_fake_norm(weight_torch: torch.Tensor, has_distributed: bool = True) -> MagicMock:
    """Return a mock RMSNorm whose ``weight`` / ``weight_distributed`` attributes
    are opaque tokens.  ``ttnn.to_torch`` is patched separately to return
    ``weight_torch`` when handed either of these tokens."""
    norm = MagicMock()

    weight_token = MagicMock(name="weight_on_device")
    weight_token.dtype = ttnn.bfloat16
    weight_token.layout = ttnn.ROW_MAJOR_LAYOUT
    weight_token.memory_config = MagicMock(return_value=ttnn.DRAM_MEMORY_CONFIG)
    norm.weight = weight_token

    if has_distributed:
        wd_token = MagicMock(name="weight_distributed_on_device")
        wd_token.dtype = ttnn.bfloat16
        wd_token.layout = ttnn.ROW_MAJOR_LAYOUT
        wd_token.memory_config = MagicMock(return_value=ttnn.DRAM_MEMORY_CONFIG)
        norm.weight_distributed = wd_token
    else:
        # Match the ``getattr(..., None)`` lookup in the production code.
        # MagicMock auto-creates attributes, so explicitly delete to make the
        # ``hasattr`` semantic behave like the real RMSNorm (which only sets
        # ``weight_distributed`` when ``is_distributed`` is True).
        del norm.weight_distributed

    norm.eps = 1e-6
    norm.output_mem_config = None
    return norm


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_distributed", [True, False])
def test_zero_centered_bakes_plus_one_into_weight(has_distributed):
    """When ``zero_centered=True``, the tensor handed to ``ttnn.from_torch``
    must equal the source weight + 1.0."""
    dim = 5120
    torch.manual_seed(0)
    source_weight = torch.randn(1, 1, dim // 32, 32, dtype=torch.float32)

    captured_from_torch_inputs = []

    def fake_from_torch(tensor, *args, **kwargs):
        # Clone so subsequent in-place ops in callers don't change what we
        # captured.
        captured_from_torch_inputs.append(tensor.clone())
        sentinel = MagicMock(name="new_weight_on_device")
        sentinel.dtype = kwargs.get("dtype")
        sentinel.layout = kwargs.get("layout")
        sentinel.memory_config = MagicMock(return_value=kwargs.get("memory_config"))
        return sentinel

    # The bake helper builds two different composers; each is identified by a
    # tagged sentinel so we can tell which call-site asked for the readback.
    def fake_replicate_composer(*a, **kw):
        s = MagicMock(name="ConcatMeshToTensor")
        s._kind = "replicated"
        return s

    def fake_sharded_composer(*a, **kw):
        s = MagicMock(name="ConcatMesh2dToTensor")
        s._kind = "sharded"
        return s

    def fake_replicate_mapper(*a, **kw):
        return MagicMock(name="ReplicateTensorToMesh")

    def fake_shard_mapper(*a, **kw):
        return MagicMock(name="ShardTensor2dMesh")

    def fake_to_torch(tt_tensor, *args, **kwargs):
        # All shards round-trip to ``source_weight``.  The bake helper slices
        # the result back to the un-replicated weight; we therefore stack the
        # source weight ``replication_factor`` times along dim 0 so the slice
        # ``[: shape0 // factor]`` recovers exactly ``source_weight``.
        composer = kwargs.get("mesh_composer")
        kind = getattr(composer, "_kind", None)
        if kind == "replicated":
            factor = 8 * 4  # replicated branch: divide by full mesh size
        else:
            factor = 8  # column-sharded branch: divide by num rows
        return source_weight.repeat(factor, 1, 1, 1)

    def fake_deallocate(_):
        return None

    args = _make_args(dim=dim)

    patches = [
        patch.object(ttnn, "from_torch", side_effect=fake_from_torch),
        patch.object(ttnn, "to_torch", side_effect=fake_to_torch),
        patch.object(ttnn, "deallocate", side_effect=fake_deallocate),
        patch.object(ttnn, "ConcatMeshToTensor", side_effect=fake_replicate_composer),
        patch.object(ttnn, "ConcatMesh2dToTensor", side_effect=fake_sharded_composer),
        patch.object(ttnn, "ReplicateTensorToMesh", side_effect=fake_replicate_mapper),
        patch.object(ttnn, "ShardTensor2dMesh", side_effect=fake_shard_mapper),
    ]

    def _enter_all():
        for p in patches:
            p.start()

    def _exit_all():
        for p in patches:
            p.stop()

    # --- Case 1: zero_centered=False — no re-upload should happen ---
    _enter_all()
    try:
        norm_plain = _make_fake_norm(source_weight, has_distributed=has_distributed)
        dn_plain = DistributedNorm(norm_plain, args, zero_centered=False)
        assert dn_plain.zero_centered is False
        # No re-upload on the plain path.
        assert len(captured_from_torch_inputs) == 0
    finally:
        _exit_all()

    # --- Case 2: zero_centered=True — every weight must be re-uploaded as +1 ---
    captured_from_torch_inputs.clear()
    _enter_all()
    try:
        norm_zc = _make_fake_norm(source_weight, has_distributed=has_distributed)
        dn_zc = DistributedNorm(norm_zc, args, zero_centered=True)
        assert dn_zc.zero_centered is True
    finally:
        _exit_all()

    expected_uploads = 2 if has_distributed else 1
    assert (
        len(captured_from_torch_inputs) == expected_uploads
    ), f"expected {expected_uploads} re-uploads, got {len(captured_from_torch_inputs)}"

    for uploaded in captured_from_torch_inputs:
        assert (
            uploaded.shape == source_weight.shape
        ), f"uploaded weight shape {uploaded.shape} != source {source_weight.shape}"
        assert torch.allclose(
            uploaded, source_weight + 1.0, atol=1e-6, rtol=0.0
        ), "uploaded weight tensor is not equal to source + 1.0"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
