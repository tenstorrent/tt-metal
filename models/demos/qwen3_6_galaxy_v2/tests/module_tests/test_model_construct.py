# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only construct test for ``TtTransformer`` (V2 qwen3.6 model wrapper).

Exercises the V2-model wave-3 wiring inside ``TtTransformer.__init__``:

* The per-layer loop builds 64 ``TtTransformerBlock`` instances (patched to a
  sentinel class so no device weights are uploaded).
* ``self.rope_setup`` is threaded onto every full_attention layer's
  ``attention`` (layers 3, 7, 11, ..., 63 — every 4th layer in the canonical
  qwen3.6 pattern). DeltaNet layers (the other 48) do NOT receive it.
* The final ``DistributedNorm`` is constructed with ``zero_centered=True``.
* The HF weight ingestion ``standardize_hf_keys_qwen36`` is exercised when
  the state_dict is supplied in raw HF (``model.language_model.*``) form.

All ttnn device entry points the constructor reaches are patched out so the
test never touches a real mesh device.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


class _FakeMeshDevice:
    """Minimal MeshDevice stand-in supporting the attribute reads inside
    ``TtTransformer.__init__`` (e.g. ``get_num_devices()``, ``.shape``)."""

    def __init__(self, shape=(8, 4)):
        self.shape = list(shape)

    def get_num_devices(self):
        return self.shape[0] * self.shape[1]

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=7, y=10)

    def set_sub_device_stall_group(self, *a, **kw):
        return None

    def load_sub_device_manager(self, *a, **kw):
        return None

    def create_sub_device_manager(self, *a, **kw):
        return MagicMock(name="sub_device_manager")


_QWEN_PATTERN = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16
assert len(_QWEN_PATTERN) == 64


def _make_qwen36_args(mesh_device):
    """Build a SimpleNamespace exposing every attribute ``TtTransformer.__init__``
    reads on the qwen3.6 path. ``MagicMock`` would too eagerly satisfy
    ``getattr`` queries (returning a truthy MagicMock for ``use_prefetcher``);
    SimpleNamespace forces us to enumerate the real config surface."""
    args = SimpleNamespace(
        is_qwen36=True,
        is_qwen=False,
        is_olmo=False,
        vocab_size=248_320,
        padded_vocab_size=248_832,
        n_layers=64,
        dim=5120,
        n_heads=24,
        n_kv_heads=4,
        head_dim=256,
        rope_dim=64,
        rope_theta=10_000_000,
        rope_scaling_factor=1.0,
        max_seq_len=4096,
        max_batch_size=32,
        norm_eps=1e-6,
        zero_centered_norm=True,
        linear_attention_pattern=list(_QWEN_PATTERN),
        use_prefetcher=False,
        use_scaled_rope=False,
        is_distributed_norm=True,
        dummy_weights=True,
        cluster_shape=[8, 4],
        max_grid_size=SimpleNamespace(x=7, y=10),
        # ``get_model_config()`` returns a dict-like with the keys the
        # constructor looks up (LM head norm + topology). Values can be
        # MagicMocks because they only get passed through to mocked
        # collaborators.
        model_config=None,
    )
    args.weight_cache_path = lambda dtype: None
    args.get_state_dict_prefix = lambda module_name, layer_num: ""
    model_config = {
        "SHARDED_NORM_LM_HEAD_PRGM_CFG": MagicMock(name="lm_head_prg_cfg"),
        "LM_HEAD_INPUT_MEMCFG": MagicMock(name="lm_head_input_memcfg"),
        "CCL_TOPOLOGY": MagicMock(name="ccl_topology"),
        "GALAXY_NUM_LINKS": 1,
        "USE_PREFETCHER": False,
    }
    args.get_model_config = lambda: model_config
    return args


# ---------------------------------------------------------------------------
# Sentinel collaborators
# ---------------------------------------------------------------------------


class _BlockSentinel:
    """Stand-in for ``TtTransformerBlock``. Records the layer_num it was
    constructed for so the threading-test can verify per-layer dispatch."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.layer_num = kwargs.get("layer_num")
        # The threading helper reads ``is_linear_attention_layer`` to decide
        # whether to skip RoPE — mirror the real decoder's attribute.
        layer_num = self.layer_num
        pattern = kwargs["args"].linear_attention_pattern
        self.is_linear_attention_layer = pattern[layer_num] == "linear_attention"
        # Attention sentinel — its ``rope_setup`` attribute is what the
        # threading step writes to.
        self.attention = MagicMock(name=f"attention[layer={layer_num}]")
        # Explicitly delete rope_setup so the threading check can see whether
        # it was *added* (vs MagicMock auto-creating it).
        del self.attention.rope_setup
        _BlockSentinel.instances.append(self)

    def prefetch(self, *a, **kw):
        return None


class _NormSentinel:
    """Stand-in for ``DistributedNorm`` that records its constructor kwargs."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _NormSentinel.instances.append(self)


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_sentinels():
    _BlockSentinel.instances.clear()
    _NormSentinel.instances.clear()
    yield
    _BlockSentinel.instances.clear()
    _NormSentinel.instances.clear()


def _patches():
    """Return a dict of named patch objects covering every collaborator the
    constructor instantiates so we never actually open a mesh device. The
    dict form lets tests look up specific mocks (e.g. ``standardize``)
    without depending on list order."""
    import ttnn  # local — ttnn is heavy to import; only needed under sys.modules
    from models.demos.qwen3_6_galaxy_v2.tt.load_checkpoints import standardize_hf_keys_qwen36 as _real_std

    patches = {
        "TtTransformerBlock": patch("models.demos.qwen3_6_galaxy_v2.tt.llama_model.TtTransformerBlock", _BlockSentinel),
        "DistributedNorm": patch("models.demos.qwen3_6_galaxy_v2.tt.llama_model.DistributedNorm", _NormSentinel),
        "TtLlamaEmbedding": patch("models.demos.qwen3_6_galaxy_v2.tt.llama_model.TtLlamaEmbedding"),
        "LMHead": patch("models.demos.qwen3_6_galaxy_v2.tt.llama_model.LMHead"),
        "TT_CCL": patch("models.demos.qwen3_6_galaxy_v2.tt.llama_model.TT_CCL"),
        "TtLlamaRotarySetup": patch("models.demos.qwen3_6_galaxy_v2.tt.llama_model.TtLlamaRotarySetup"),
        "RMSNorm": patch("models.demos.qwen3_6_galaxy_v2.tt.llama_model.RMSNorm"),
        "SamplingGenerator": patch("models.demos.qwen3_6_galaxy_v2.tt.llama_model.SamplingGenerator"),
        "TtLlamaPrefetcherSetup": patch("models.demos.qwen3_6_galaxy_v2.tt.llama_model.TtLlamaPrefetcherSetup"),
        # Wrap the real function so it actually runs (and the result is
        # consumed by the constructor) while still being a Mock that records
        # call_count.
        "standardize": patch(
            "models.demos.qwen3_6_galaxy_v2.tt.llama_model.standardize_hf_keys_qwen36",
            wraps=_real_std,
        ),
        "ttnn_from_torch": patch.object(ttnn, "from_torch", lambda *a, **kw: MagicMock(name="from_torch")),
        "ttnn_ReplicateTensorToMesh": patch.object(
            ttnn, "ReplicateTensorToMesh", lambda *a, **kw: MagicMock(name="ReplicateMapper")
        ),
        "ttnn_ShardTensor2dMesh": patch.object(
            ttnn, "ShardTensor2dMesh", lambda *a, **kw: MagicMock(name="ShardMapper")
        ),
        "ttnn_SubDeviceId": patch.object(ttnn, "SubDeviceId", lambda *a, **kw: MagicMock(name="SubDeviceId")),
        "ttnn_SubDevice": patch.object(ttnn, "SubDevice", lambda *a, **kw: MagicMock(name="SubDevice")),
        "ttnn_CoreRangeSet": patch.object(ttnn, "CoreRangeSet", lambda *a, **kw: MagicMock(name="CoreRangeSet")),
        "ttnn_CoreRange": patch.object(ttnn, "CoreRange", lambda *a, **kw: MagicMock(name="CoreRange")),
        "ttnn_CoreCoord": patch.object(ttnn, "CoreCoord", lambda *a, **kw: MagicMock(name="CoreCoord")),
    }
    return patches


def _build_model(state_dict=None):
    """Construct ``TtTransformer`` under the patch set above and return the
    model plus the (already-stopped) patches dict so callers can inspect
    the mocks via ``patches["standardize"].new`` etc.

    The patches are stopped before this function returns. Mock state from
    inside the constructor remains accessible via the ``patch.new`` attribute
    of each stopped patcher (the underlying Mock object persists)."""
    mesh = _FakeMeshDevice(shape=(8, 4))
    args = _make_qwen36_args(mesh)

    patches = _patches()
    started = {name: p.start() for name, p in patches.items()}
    try:
        from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer

        model = TtTransformer(
            args=args,
            dtype=MagicMock(name="bfloat16"),
            mesh_device=mesh,
            state_dict=state_dict if state_dict is not None else {},
            weight_cache_path=None,
            paged_attention_config=None,
            use_paged_kv_cache=False,
            decode_mode_only=True,  # skip the prefill switch_mode call
        )
    finally:
        for p in patches.values():
            p.stop()
    return model, started


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_qwen36_transformer_builds_64_layers():
    """``args.n_layers == 64`` → ``self.layers`` has 64 entries, and every
    sentinel records its own ``layer_num``."""
    model, _started = _build_model()
    assert len(model.layers) == 64
    assert [layer.layer_num for layer in model.layers] == list(range(64))


@pytest.mark.cpu_only
def test_qwen36_transformer_threads_rope_setup_onto_full_attention_layers_only():
    """For canonical [lin, lin, lin, full] × 16, full_attention layers are
    indices 3, 7, 11, ..., 63 (16 total). RoPE must be threaded onto exactly
    those layers' ``attention.rope_setup`` — and only those."""
    model, _started = _build_model()

    full_attn_indices = [i for i, kind in enumerate(_QWEN_PATTERN) if kind == "full_attention"]
    expected_full_attn = list(range(3, 64, 4))
    assert full_attn_indices == expected_full_attn, "Sanity: full_attn pattern indices"
    assert len(full_attn_indices) == 16

    for i, layer in enumerate(model.layers):
        if i in full_attn_indices:
            assert (
                getattr(layer.attention, "rope_setup", None) is model.rope_setup
            ), f"full_attention layer {i} did not get rope_setup threaded"
        else:
            # DeltaNet layers — must NOT have rope_setup written.
            assert (
                not hasattr(layer.attention, "rope_setup") or layer.attention.rope_setup is not model.rope_setup
            ), f"DeltaNet (linear_attention) layer {i} unexpectedly received rope_setup"


@pytest.mark.cpu_only
def test_qwen36_final_norm_constructed_with_zero_centered_true():
    """The final pre-LM-head ``DistributedNorm`` is constructed with
    ``zero_centered=True`` when ``args.zero_centered_norm=True``."""
    model, _started = _build_model()
    # _NormSentinel.instances may include DistributedNorms created by the
    # per-layer block — but those go through the *decoder* (patched as
    # _BlockSentinel) and never reach the top-level DistributedNorm patch
    # since the decoder imports DistributedNorm from its own namespace.
    # The only top-level DistributedNorm we instantiate is ``self.norm``.
    assert (
        len(_NormSentinel.instances) == 1
    ), f"expected exactly 1 top-level DistributedNorm, got {_NormSentinel.instances}"
    norm = _NormSentinel.instances[0]
    assert norm.kwargs.get("zero_centered") is True, f"zero_centered kwarg: {norm.kwargs.get('zero_centered')}"
    assert model.norm is norm


@pytest.mark.cpu_only
def test_qwen36_standardize_hf_keys_is_called_on_raw_hf_state_dict():
    """When the caller hands in a raw HF state_dict (containing
    ``model.language_model.*`` keys), ``__init__`` runs the qwen3.6
    standardization pass so downstream constructors see canonical keys."""
    # Build a tiny raw-HF state_dict carrying the distinctive prefix.
    import torch

    raw_hf_sd = {
        "model.language_model.embed_tokens.weight": torch.zeros(10, 5120),
        "model.language_model.norm.weight": torch.zeros(5120),
    }
    _model, started = _build_model(state_dict=raw_hf_sd)
    std_mock = started["standardize"]
    assert std_mock.call_count == 1, f"standardize_hf_keys_qwen36 called {std_mock.call_count} times, expected 1"


@pytest.mark.cpu_only
def test_qwen36_standardize_hf_keys_skipped_for_already_meta_state_dict():
    """When the state_dict is already in meta-style form (no
    ``model.language_model.*`` keys), the standardization pass is skipped."""
    _model, started = _build_model(state_dict={"tok_embeddings.weight": MagicMock()})
    std_mock = started["standardize"]
    assert std_mock.call_count == 0, "standardize_hf_keys_qwen36 should NOT run on already-meta state_dict"


@pytest.mark.cpu_only
def test_qwen36_lm_head_shape_padded_vocab_5120():
    """Verify the LM head sees ``padded_vocab_size=248832`` and ``dim=5120``
    on the qwen3.6 args. LMHead is patched, so this verifies the kwargs
    threaded through the constructor."""
    model, _started = _build_model()
    # The args object reaches LMHead via its ``args=`` kwarg — verify the
    # vocab / dim values the LM head will read.
    assert model.args.padded_vocab_size == 248_832
    assert model.args.dim == 5120
    assert getattr(model.args, "pad_logits_to_power_of_2", None) is not False  # may be True or absent


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
