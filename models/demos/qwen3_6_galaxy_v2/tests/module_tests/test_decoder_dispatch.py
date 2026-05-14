# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only smoke test for the hybrid attention dispatch in
``TtTransformerBlock.__init__`` (qwen3.6 V2 decoder).

We patch ``TtLlamaAttention`` and ``TtQwen36DeltaAttention`` in the decoder's
import namespace with sentinel classes that record the kwargs they receive,
then assert:

  * ``is_qwen36=True`` + layer's pattern entry is ``"linear_attention"``  →
    DeltaNet sentinel is instantiated.
  * ``is_qwen36=True`` + layer's pattern entry is ``"full_attention"``    →
    LlamaAttention sentinel is instantiated.
  * ``is_qwen36=False`` (e.g. 70B / qwen3-32B / olmo path) on any layer  →
    LlamaAttention sentinel is instantiated regardless of pattern.

The DistributedNorm / RMSNorm / MLP / model_config plumbing is all mocked
out — this is purely a dispatch-conditional smoke test.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Sentinel attention classes
# ---------------------------------------------------------------------------


class _DeltaSentinel:
    """Stand-in for TtQwen36DeltaAttention."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        # Mimic the surface the decoder reads later (none of these matter for
        # the dispatch test but keep the attribute set finite).
        self.dn_state_buffer = MagicMock(name="dn_state_buffer")
        self.conv_state_buffer = MagicMock(name="conv_state_buffer")
        _DeltaSentinel.instances.append(self)


class _LlamaSentinel:
    """Stand-in for TtLlamaAttention."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _LlamaSentinel.instances.append(self)

    def prefetch(self, *a, **kw):
        pass


# ---------------------------------------------------------------------------
# Mock args / state_dict
# ---------------------------------------------------------------------------


def _make_args(is_qwen36: bool, pattern):
    """Mock the slice of ModelArgs that ``TtTransformerBlock.__init__`` reads."""
    args = MagicMock(name="TtQwen36ModelArgs")
    args.dim = 5120
    args.n_heads = 24
    args.n_kv_heads = 4
    args.max_seq_len = 4096
    args.max_batch_size = 32
    args.unfuse_res_add = False
    args.dummy_weights = True
    args.is_distributed_norm = True
    args.is_qwen36 = is_qwen36
    args.zero_centered_norm = is_qwen36
    args.linear_attention_pattern = pattern
    # get_state_dict_prefix is called when constructing the RMSNorm wrapper.
    args.get_state_dict_prefix = MagicMock(return_value="layers.X.")

    # model_config has to behave like a dict for the decoder's lookups, plus
    # be retrievable via args.get_model_config().
    model_config = {
        "SHARDED_NORM_ATTN_PRGM_CFG": MagicMock(),
        "SHARDED_ATTN_INPUT_MEMCFG": MagicMock(),
        "SHARDED_ATTN_INPUT_RING_MEMCFG": MagicMock(),
        "SHARDED_NORM_MLP_PRGM_CFG": MagicMock(),
        "SHARDED_MLP_INPUT_MEMCFG": MagicMock(),
        "SHARDED_FF12_RING_MEMCFG": MagicMock(),
        "CCL_TOPOLOGY": MagicMock(),
        "DECODE_RESIDUAL_MEMCFG": MagicMock(),
        "USE_PREFETCHER": False,
    }
    args.get_model_config = MagicMock(return_value=model_config)
    return args


def _make_state_dict(layer_num):
    """Tiny placeholder state-dict — only key membership is consulted by the
    decoder's DeltaNet weight slicer."""
    return {
        f"layers.{layer_num}.linear_attn.in_proj_qkv.weight": MagicMock(),
        f"layers.{layer_num}.linear_attn.conv1d.weight": MagicMock(),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_sentinels():
    _DeltaSentinel.instances.clear()
    _LlamaSentinel.instances.clear()
    yield
    _DeltaSentinel.instances.clear()
    _LlamaSentinel.instances.clear()


def _build_block(args, state_dict, layer_num):
    """Construct ``TtTransformerBlock`` with all heavyweight collaborators
    mocked out — the only real code paths exercised are the dispatch branch
    and the surrounding plumbing."""
    # Patch every collaborator that the decoder __init__ touches.
    # NOTE: We patch DeltaNet's symbol at its source module (the decoder uses
    # a late local import); the rest are top-level imports in the decoder.
    with patch(
        "models.demos.qwen3_6_galaxy_v2.tt.qwen36_delta_attention.TtQwen36DeltaAttention",
        _DeltaSentinel,
    ), patch(
        "models.demos.qwen3_6_galaxy_v2.tt.llama_decoder.TtLlamaAttention",
        _LlamaSentinel,
    ), patch(
        "models.demos.qwen3_6_galaxy_v2.tt.llama_decoder.TtLlamaMLP"
    ) as mlp_cls, patch(
        "models.demos.qwen3_6_galaxy_v2.tt.llama_decoder.DistributedNorm"
    ) as norm_cls, patch(
        "models.demos.qwen3_6_galaxy_v2.tt.llama_decoder.RMSNorm"
    ) as rmsnorm_cls:
        mlp_cls.return_value = MagicMock(name="mlp")
        norm_cls.return_value = MagicMock(name="norm")
        rmsnorm_cls.return_value = MagicMock(name="rmsnorm_inner")

        # Import locally so the patches above are in effect at construction
        # time. (Top-level import would resolve the symbols before patching.)
        from models.demos.qwen3_6_galaxy_v2.tt.llama_decoder import TtTransformerBlock

        mesh_device = MagicMock(name="mesh_device")
        block = TtTransformerBlock(
            args=args,
            mesh_device=mesh_device,
            dtype=MagicMock(name="bfloat16"),
            state_dict=state_dict,
            layer_num=layer_num,
            n_layers=64,
            weight_cache_path=None,
            transformation_mats=MagicMock(),
            paged_attention_config=None,
            use_paged_kv_cache=False,
            prefetcher_setup=None,
            tt_ccl=MagicMock(),
        )
        return block


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


_QWEN_PATTERN = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16


@pytest.mark.cpu_only
def test_qwen36_linear_layer_dispatches_to_deltanet():
    """layer_num=0 on the qwen3.6 pattern is "linear_attention" → DeltaNet."""
    args = _make_args(is_qwen36=True, pattern=_QWEN_PATTERN)
    block = _build_block(args, _make_state_dict(0), layer_num=0)

    assert isinstance(block.attention, _DeltaSentinel), (
        f"expected DeltaNet sentinel for linear_attention layer 0, " f"got {type(block.attention).__name__}"
    )
    assert block.is_linear_attention_layer is True
    assert block.is_qwen36 is True
    # DeltaNet must have received the layer-sliced weights dict with the
    # ``linear_attn.*`` prefix preserved.
    assert "linear_attn.in_proj_qkv.weight" in block.attention.kwargs["weights_dict"]
    assert block.attention.kwargs["layer_num"] == 0


@pytest.mark.cpu_only
def test_qwen36_full_attention_layer_dispatches_to_llama_attention():
    """layer_num=3 on the qwen3.6 pattern is "full_attention" → TtLlamaAttention."""
    args = _make_args(is_qwen36=True, pattern=_QWEN_PATTERN)
    block = _build_block(args, _make_state_dict(3), layer_num=3)

    assert isinstance(block.attention, _LlamaSentinel), (
        f"expected LlamaAttention sentinel for full_attention layer 3, " f"got {type(block.attention).__name__}"
    )
    assert block.is_linear_attention_layer is False
    assert block.is_qwen36 is True


@pytest.mark.cpu_only
def test_non_qwen36_always_dispatches_to_llama_attention():
    """is_qwen36=False (70B / qwen3-32B / olmo path) → TtLlamaAttention on any
    layer, regardless of any (hypothetical) layer pattern."""
    args = _make_args(is_qwen36=False, pattern=None)
    for layer_num in (0, 3, 17, 63):
        block = _build_block(args, _make_state_dict(layer_num), layer_num=layer_num)
        assert isinstance(block.attention, _LlamaSentinel), (
            f"non-qwen36 build must always pick TtLlamaAttention, "
            f"got {type(block.attention).__name__} at layer {layer_num}"
        )
        assert block.is_linear_attention_layer is False
        assert block.is_qwen36 is False


@pytest.mark.cpu_only
def test_non_qwen36_with_pattern_still_dispatches_to_llama_attention():
    """A safety check: even if some downstream code accidentally sets a
    ``linear_attention_pattern`` on a non-qwen36 build, the ``is_qwen36`` gate
    keeps the 70B path on TtLlamaAttention."""
    args = _make_args(is_qwen36=False, pattern=_QWEN_PATTERN)
    block = _build_block(args, _make_state_dict(0), layer_num=0)
    assert isinstance(block.attention, _LlamaSentinel)
    assert block.is_linear_attention_layer is False
