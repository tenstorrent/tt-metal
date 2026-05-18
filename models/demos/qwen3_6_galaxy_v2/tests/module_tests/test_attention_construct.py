# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only construct test for ``TtLlamaAttention`` with ``is_qwen36=True``.

Exercises the V2-4 weight-build branch:

* The fused ``q_proj.weight`` (``[12288, 5120]``) is de-interleaved into Q
  (first 6144 channels post-deinterleave) and Gate (second 6144 channels).
* ``k_proj`` and ``v_proj`` keep their HF shapes (``[1024, 5120]``).
* ``o_proj.weight`` is ``[5120, 6144]`` (transposed once → ``[6144, 5120]``
  before upload).
* QKVG fused buffer is the concatenation per col of ``[Q | Gate | K | V]``,
  total per col 14 heads × 256 = 3584; across all 4 cols 14336 channels.
* QK-norm weights are baked with ``+1`` when ``zero_centered_norm=True``.

All ``ttnn`` device entry points are patched so the test never touches a real
mesh device.  We just verify the constructor runs end-to-end and exposes the
``q_proj_weight_shape`` / ``gate_proj_weight_shape`` / ``qkvg_total_width``
attributes the V2-7 cross-check needs.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import ttnn

# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


class _FakeMeshDevice:
    """Minimal MeshDevice stand-in.  ``ttnn.MeshDevice`` is checked via
    ``__class__.__name__`` in ``RMSNorm.__init__``; setting the class name here
    keeps any happens-to-not-be-mocked is-mesh-device check happy."""

    def __init__(self, shape=(8, 4)):
        self.shape = list(shape)

    def get_num_devices(self):
        return self.shape[0] * self.shape[1]

    def compute_with_storage_grid_size(self):
        g = SimpleNamespace(x=7, y=10)
        return g


def _make_qwen36_args():
    """Build a minimal ``args`` SimpleNamespace exposing every attribute the
    constructor reads on the ``is_qwen36`` path.  No real mesh / model_config
    is needed because the qwen3.6 branch skips the 70B-prefetcher mem-configs."""
    mesh = _FakeMeshDevice(shape=(8, 4))
    args = SimpleNamespace(
        num_devices=32,
        dim=5120,
        n_heads=24,
        head_dim=256,
        max_seq_len=4096,
        max_batch_size=1,
        n_kv_heads_unpadded=4,
        n_kv_heads=8,  # V2-TP: padded 4 → 8 for 2D-TP head split on rows
        is_qwen36=True,
        zero_centered_norm=True,
        rope_dim=64,
        partial_rotary_factor=0.25,
        norm_eps=1e-6,
        qk_norm=True,
        cluster_shape=[8, 4],
        # ``is_qwen36`` branch reads these via ``getattr`` with defaults; we
        # provide explicit values to mirror the real qwen36 config.
        ccl_dtype=ttnn.bfloat8_b,
        use_prefetcher=False,
        is_multichip=True,
        # ``model_config`` only needs to be a dict-like — the qwen36 branch
        # makes a copy and writes ``USE_PREFETCHER``.
        model_config={},
        # ``get_state_dict_prefix("TtLlamaAttention", layer_num)`` is called.
        get_state_dict_prefix=lambda module_name, layer_num: f"layers.{layer_num}.attention",
        # ``ccl_topology`` is callable on 70B, @property on qwen36 — pass a
        # plain value here (the qwen36 branch handles both).
        ccl_topology=ttnn.Topology.Linear,
        dummy_weights=True,
    )
    args.mesh_device = mesh
    return args, mesh


def _make_qwen36_state_dict(layer_num=0):
    """Build a state_dict with the four attention weights + qknorm weights at
    the right shapes for qwen3.6-27B (H=5120, n_q=24, n_kv=4, hd=256)."""
    prefix = f"layers.{layer_num}.attention"
    return {
        f"{prefix}.wq.weight": torch.zeros(12288, 5120),
        f"{prefix}.wk.weight": torch.zeros(1024, 5120),
        f"{prefix}.wv.weight": torch.zeros(1024, 5120),
        f"{prefix}.wo.weight": torch.zeros(5120, 6144),
        f"{prefix}.q_norm.weight": torch.zeros(256),
        f"{prefix}.k_norm.weight": torch.zeros(256),
    }


# ---------------------------------------------------------------------------
# Helpers — patch every ttnn allocator/mesh-mapper the constructor touches.
# ---------------------------------------------------------------------------


def _ttnn_patches():
    """Patch the device-facing ttnn primitives used by ``__init__``.  Each
    returns a MagicMock standing in for the allocated tensor."""

    def _sentinel(name):
        return MagicMock(name=name)

    return [
        patch.object(ttnn, "from_torch", lambda *a, **kw: _sentinel("from_torch")),
        patch.object(ttnn, "as_tensor", lambda *a, **kw: _sentinel("as_tensor")),
        patch.object(ttnn, "ReplicateTensorToMesh", lambda *a, **kw: _sentinel("ReplicateMapper")),
        patch.object(ttnn, "ShardTensor2dMesh", lambda *a, **kw: _sentinel("ShardMapper")),
        patch.object(ttnn, "deallocate", lambda *a, **kw: None),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_qwen36_attention_constructs():
    """The qwen3.6 path constructs without touching a real mesh device and
    exposes ``q_proj_weight_shape`` / ``gate_proj_weight_shape`` /
    ``qkvg_total_width`` attributes for V2-7 cross-checking."""
    args, _mesh = _make_qwen36_args()
    sd = _make_qwen36_state_dict(layer_num=0)

    patches = _ttnn_patches()
    for p in patches:
        p.start()
    try:
        from models.demos.qwen3_6_galaxy_v2.tt.llama_attention import TtLlamaAttention

        attn = TtLlamaAttention(
            mesh_device=args.mesh_device,
            state_dict=sd,
            weight_cache_path=None,
            layer_num=0,
            dtype=ttnn.bfloat16,
            transformation_mats=None,
            configuration=args,
            paged_attention_config=None,
            use_paged_kv_cache=True,  # skip layer_past upload to keep mocks simple
            prefetcher_setup=None,
            tt_ccl=None,  # qwen36 branch never reads tt_ccl.mode at __init__
        )
    finally:
        for p in patches:
            p.stop()

    # ----------------------------------------------------------------------
    # Acceptance assertions
    # ----------------------------------------------------------------------
    assert attn.is_qwen36 is True
    assert attn.zero_centered_norm is True
    assert attn.qk_norm is True
    assert attn.rope_dim == 64
    # V2-TP: 2D tensor-parallel layout.  Q/Gate are n_q*hd=6144; K/V padded
    # 4 → 8 heads → 2048 rows.
    assert attn.q_proj_weight_shape == (6144, 5120)
    assert attn.gate_proj_weight_shape == (6144, 5120)
    assert attn.k_proj_weight_shape == (2048, 5120)
    assert attn.v_proj_weight_shape == (2048, 5120)
    assert attn.wo_proj_weight_shape == (5120, 6144)
    # Per-chip fused QKVG width = (n_q + n_q + n_kv_padded + n_kv_padded) / 8 rows * hd
    #                          = (24 + 24 + 8 + 8) / 8 * 256
    #                          = 2048.
    assert attn.total_per_chip == 2048
    assert attn.n_q_per_chip == 3
    assert attn.n_kv_per_chip == 1
    # Total fused QKVG width across all 8 rows = 2048 * 8 = 16384.
    assert attn.qkvg_total_width == 16384
    # qwen36 path uses ``self.wqkvg``; ``self.wqkv_interleaved`` is aliased to
    # the same buffer so any downstream attribute lookup still resolves.
    assert hasattr(attn, "wqkvg")
    assert hasattr(attn, "wqkv_interleaved")
    assert attn.wqkv_interleaved is attn.wqkvg


@pytest.mark.cpu_only
def test_qwen36_attention_de_interleaves_q_and_gate_correctly():
    """Sanity check that the de-interleave math (HF ``q_proj[n_q,2,hd,H]`` →
    Q = ``[:, 0, :, :]``, Gate = ``[:, 1, :, :]``) actually picks out the
    intended halves.  We pre-fill ``q_proj.weight`` with a recognisable
    pattern and capture the tensor handed to ``ttnn.as_tensor`` for
    ``wqkvg`` to ensure the column-0 block opens with the Q heads (not Gate)."""
    args, _mesh = _make_qwen36_args()
    # ``cache_name`` returns None whenever ``dummy_weights=True`` — disable it
    # here so the as_tensor captures get a non-None cache_file_name key.
    args.dummy_weights = False
    sd = _make_qwen36_state_dict(layer_num=0)
    # Build a recognisable q_proj weight: per-head, dim=1 alternates Q / gate.
    n_q, hd, H = 24, 256, 5120
    q_proj = torch.zeros(n_q, 2, hd, H)
    q_proj[:, 0, :, :] = 1.0  # Q heads
    q_proj[:, 1, :, :] = 7.0  # Gate heads
    sd["layers.0.attention.wq.weight"] = q_proj.reshape(n_q * 2 * hd, H)
    sd["layers.0.attention.wk.weight"] = torch.full((1024, 5120), 2.0)
    sd["layers.0.attention.wv.weight"] = torch.full((1024, 5120), 3.0)

    captured = {}

    def _capture_as_tensor(tensor, *a, **kw):
        cache = kw.get("cache_file_name")
        # ttnn.as_tensor sees the host tensor pre-upload — record by cache key.
        if cache is None:
            return MagicMock(name="as_tensor")
        name = str(cache).rsplit("/", 1)[-1]
        captured.setdefault(name, []).append(tensor.detach().clone())
        return MagicMock(name=f"as_tensor[{name}]")

    patches = [
        patch.object(ttnn, "from_torch", lambda *a, **kw: MagicMock(name="from_torch")),
        patch.object(ttnn, "as_tensor", _capture_as_tensor),
        patch.object(ttnn, "ReplicateTensorToMesh", lambda *a, **kw: MagicMock(name="ReplicateMapper")),
        patch.object(ttnn, "ShardTensor2dMesh", lambda *a, **kw: MagicMock(name="ShardMapper")),
        patch.object(ttnn, "deallocate", lambda *a, **kw: None),
    ]
    for p in patches:
        p.start()
    try:
        from models.demos.qwen3_6_galaxy_v2.tt.llama_attention import TtLlamaAttention

        TtLlamaAttention(
            mesh_device=args.mesh_device,
            state_dict=sd,
            # Provide a Path-like cache root so the attention constructor's
            # ``cache_name`` lambda returns a non-None path with a basename we
            # can capture above.
            weight_cache_path=__import__("pathlib").Path("/tmp/_v24_attn_capture"),
            layer_num=0,
            dtype=ttnn.bfloat16,
            transformation_mats=None,
            configuration=args,
            paged_attention_config=None,
            use_paged_kv_cache=True,
            prefetcher_setup=None,
            tt_ccl=None,
        )
    finally:
        for p in patches:
            p.stop()

    # V2-TP: wqkvg upload now has shape [1, 1, H, qkvg_total_width=16384].
    qkvg_blob = captured.get("layers.0.attention.wqkvg_tp2d_row_col")
    assert (
        qkvg_blob is not None and len(qkvg_blob) == 1
    ), f"wqkvg upload not captured; captured keys: {list(captured.keys())}"
    qkvg_tensor = qkvg_blob[0]
    assert qkvg_tensor.shape == (1, 1, 5120, 16384), f"qkvg shape={qkvg_tensor.shape}"

    # Row 0 block spans output columns [0, 2048).  Within that block the
    # per-chip layout is [Q (3*256=768) | Gate (768) | K (256) | V (256)].
    row0 = qkvg_tensor[0, 0, :, :2048]
    assert torch.all(row0[:, 0:768] == 1.0), "Q block did not pick up the Q sentinel value"
    assert torch.all(row0[:, 768:1536] == 7.0), "Gate block did not pick up the Gate sentinel value"
    # K block (cols [1536, 1792)) and V block (cols [1792, 2048)).  K head 0
    # comes from the repeat_interleave(2, dim=0) padded layout, so K head 0
    # = original K head 0 with sentinel 2.0; V head 0 sentinel 3.0.
    assert torch.all(row0[:, 1536:1792] == 2.0), "K block sentinel mismatch"
    assert torch.all(row0[:, 1792:2048] == 3.0), "V block sentinel mismatch"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
