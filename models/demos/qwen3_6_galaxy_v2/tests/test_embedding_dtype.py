# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit test for ``TtLlamaEmbedding`` dtype selection.

Olmo3 hit a 64-layer prefill PCC bug because ``llama_embedding.py`` emitted
``bfloat8_b`` for prefill (via a ``seq_len > 32`` heuristic).  The residual
stream then accumulated bf8b quantization error layer over layer, dragging the
hidden-state PCC down and amplifying std by ~43%.

The fix on the ``is_qwen36`` branch: force ``ttnn.bfloat16`` unconditionally
for the embedding output, regardless of prefill / decode or seq_len.

This test mocks ``ttnn.embedding``, ``ttnn.reshape``, ``ttnn.as_tensor`` and
``ttnn.ShardTensor2dMesh`` and captures the ``dtype`` kwarg passed to
``ttnn.embedding``.  It asserts:

  * ``is_qwen36=True``  -> dtype is ``ttnn.bfloat16`` in BOTH prefill and decode
  * ``is_qwen36=False`` -> dtype is ``ttnn.bfloat8_b`` in prefill (legacy path
                           untouched), ``ttnn.bfloat16`` in decode.

No real device is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.llama_embedding import TtLlamaEmbedding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(is_qwen36: bool) -> MagicMock:
    """Minimal mock of the ModelArgs surface used by ``TtLlamaEmbedding``."""
    args = MagicMock()
    args.is_qwen36 = is_qwen36
    args.dummy_weights = True  # skip cache_file_name path
    args.cluster_shape = [8, 4]
    args.get_state_dict_prefix = MagicMock(return_value="")

    # ``get_model_config()`` is called in __init__ to fetch EMB_WEIGHTS_MEMCFG;
    # ``model_config["DECODE_RESIDUAL_MEMCFG"]`` is read in forward.
    model_config = {
        "EMB_WEIGHTS_MEMCFG": MagicMock(name="EMB_WEIGHTS_MEMCFG"),
        "DECODE_RESIDUAL_MEMCFG": MagicMock(name="DECODE_RESIDUAL_MEMCFG"),
    }
    args.get_model_config = MagicMock(return_value=model_config)
    args.model_config = model_config
    return args


def _make_input_token(seq_len: int) -> MagicMock:
    """Mock input tensor whose ``.shape[-2] * .shape[-1]`` equals ``seq_len`` so
    the forward's first reshape lands on shape ``(1, 1, 1, seq_len)``."""
    tok = MagicMock(name=f"input_seq_{seq_len}")
    # The forward reshapes once to (1, 1, 1, seq_len) — we don't actually run
    # the reshape; we control the post-reshape shape via our mock.
    tok.shape = (1, 1, 1, seq_len)
    return tok


def _fake_reshape(tensor, shape, *args, **kwargs):
    """Return a mock whose ``.shape`` reflects the requested reshape — the
    production code reads ``x.shape[-1]`` after the first reshape to gate
    prefill vs decode."""
    out = MagicMock(name="reshaped")
    # ``ttnn.Shape((1, 1, 1, N))`` is itself iterable / indexable.
    try:
        out.shape = tuple(shape)
    except TypeError:
        # ttnn.Shape may not be iterable in the mock; fall back to attribute access.
        out.shape = shape
    return out


def _fake_embedding(*args, **kwargs):
    """Return an opaque token; we only care about the kwargs (dtype) captured."""
    out = MagicMock(name="embedding_out")
    out.shape = (1, 32768, 5120)  # placeholder; not read after second reshape
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run_forward(args, seq_len_after_flatten: int):
    """Construct ``TtLlamaEmbedding`` and run ``forward`` once with all ttnn
    calls patched.  Returns the captured ``dtype`` kwarg passed to
    ``ttnn.embedding``."""
    state_dict = {"tok_embeddings.weight": torch.zeros(151936, 5120)}
    mesh_device = MagicMock(name="mesh_device")
    weight_cache_path = Path("/tmp/unused")

    captured = {}

    def capturing_embedding(*a, **kw):
        captured["dtype"] = kw.get("dtype")
        captured["memory_config"] = kw.get("memory_config")
        return _fake_embedding(*a, **kw)

    # We also need ttnn.Shape((..)) to be subscript-friendly; the production
    # code uses ``ttnn.reshape(x, ttnn.Shape((...)))`` and then reads
    # ``x.shape[-1]`` from the *result*.  Our ``_fake_reshape`` controls the
    # result's shape directly, so we can let ``ttnn.Shape`` be a passthrough.
    with patch.object(ttnn, "as_tensor", return_value=MagicMock(name="weights_on_device")), patch.object(
        ttnn, "ShardTensor2dMesh", return_value=MagicMock(name="shard_mapper")
    ), patch.object(ttnn, "reshape", side_effect=_fake_reshape), patch.object(
        ttnn, "embedding", side_effect=capturing_embedding
    ), patch.object(
        ttnn, "Shape", side_effect=lambda x: tuple(x)
    ):
        emb = TtLlamaEmbedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=weight_cache_path,
            state_dict=state_dict,
            dtype=ttnn.bfloat16,
        )

        # Input tensor whose shape[-2]*shape[-1] == seq_len_after_flatten.
        tok = MagicMock(name="input")
        tok.shape = (1, 1, 1, seq_len_after_flatten)
        emb.forward(tok)

    return captured


@pytest.mark.cpu_only
def test_qwen36_prefill_forces_bfloat16():
    """With ``is_qwen36=True`` and prefill seq_len=128 (> 32), dtype must be
    ``ttnn.bfloat16`` — NOT the legacy ``ttnn.bfloat8_b``."""
    args = _make_args(is_qwen36=True)
    captured = _run_forward(args, seq_len_after_flatten=128)
    assert captured["dtype"] == ttnn.bfloat16, (
        f"qwen3.6 prefill must use bfloat16 to keep the residual stream in "
        f"bfloat16 across 64 layers, got {captured['dtype']!r}"
    )


@pytest.mark.cpu_only
def test_qwen36_decode_forces_bfloat16():
    """With ``is_qwen36=True`` and decode seq_len=1 (<= 32), dtype must be
    ``ttnn.bfloat16`` (same as the legacy decode path)."""
    args = _make_args(is_qwen36=True)
    captured = _run_forward(args, seq_len_after_flatten=1)
    assert captured["dtype"] == ttnn.bfloat16, f"qwen3.6 decode must use bfloat16, got {captured['dtype']!r}"


@pytest.mark.cpu_only
def test_non_qwen36_prefill_keeps_legacy_bfloat8_b():
    """Regression guard: with ``is_qwen36=False`` and prefill seq_len=128, the
    legacy llama / qwen-32B path must still emit ``ttnn.bfloat8_b``."""
    args = _make_args(is_qwen36=False)
    captured = _run_forward(args, seq_len_after_flatten=128)
    assert captured["dtype"] == ttnn.bfloat8_b, (
        f"non-qwen36 prefill must take the legacy bfloat8_b path, got " f"{captured['dtype']!r}"
    )


@pytest.mark.cpu_only
def test_non_qwen36_decode_keeps_bfloat16():
    """Regression guard: with ``is_qwen36=False`` and decode seq_len=1, the
    legacy path uses ``ttnn.bfloat16``."""
    args = _make_args(is_qwen36=False)
    captured = _run_forward(args, seq_len_after_flatten=1)
    assert captured["dtype"] == ttnn.bfloat16, f"non-qwen36 decode must use bfloat16, got {captured['dtype']!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
