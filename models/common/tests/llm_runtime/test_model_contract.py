# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generic host-only tensor-model contract for migrated sibling models."""

import inspect
from dataclasses import dataclass, fields
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from models.common.models.deepseek_r1_distill_qwen_14b import model as deepseek_model
from models.common.models.llama32_1b import model as llama32_model
from models.common.models.llama32_3b import model as llama32_3b_model
from models.common.models.llama33_70b import model as llama33_70b_model
from models.common.models.mistral_7b import model as mistral_model
from models.common.models.phi4 import model as phi4_model
from models.common.models.qwen2_7b import model as qwen2_model
from models.common.models.qwen3_32b import model as qwen3_32b_model
from models.common.models.qwen25_7b import model as qwen25_model
from models.common.models.qwen25_72b import model as qwen25_72b_model
from models.common.models.qwen25_coder_32b import model as qwen25_coder_32b_model

MODEL_CONTRACTS = {
    "llama32_1b": SimpleNamespace(
        module=llama32_model,
        model_class=llama32_model.Llama32_1BTransformer1D,
        config_class=llama32_model.Llama32_1BTransformer1DConfig,
        attention_config_class=llama32_model.Attention1DConfig,
        make_attention_config=lambda **kwargs: _make_llama32_attention_config(**kwargs),
        make_config=lambda **kwargs: _make_llama32_config(**kwargs),
        make_layer=lambda attention_config=None: _make_llama32_layer(attention_config),
        construct_model=lambda monkeypatch: _construct_llama32_model(monkeypatch),
        expected_module_names=(
            "layer[0].attn_norm",
            "layer[0].attention",
            "layer[0].ff_norm",
            "layer[0].mlp",
            "layer[1].attn_norm",
            "layer[1].attention",
            "layer[1].ff_norm",
            "layer[1].mlp",
            "final_norm",
            "lm_head",
        ),
    ),
    "llama32_3b": SimpleNamespace(
        module=llama32_3b_model,
        model_class=llama32_3b_model.Llama32_3BTransformer1D,
        config_class=llama32_3b_model.Llama32_3BTransformer1DConfig,
        attention_config_class=llama32_3b_model.Attention1DConfig,
        make_attention_config=lambda **kwargs: _make_llama32_attention_config(**kwargs),
        make_config=lambda **kwargs: _make_llama32_config(module=llama32_3b_model, **kwargs),
        make_layer=lambda attention_config=None: _make_llama32_layer(attention_config),
        construct_model=lambda monkeypatch: _construct_llama32_model(monkeypatch, module=llama32_3b_model),
        expected_module_names=(
            "layer[0].attn_norm",
            "layer[0].attention",
            "layer[0].ff_norm",
            "layer[0].mlp",
            "layer[1].attn_norm",
            "layer[1].attention",
            "layer[1].ff_norm",
            "layer[1].mlp",
            "final_norm",
            "lm_head",
        ),
    ),
    "llama33_70b": SimpleNamespace(
        module=llama33_70b_model,
        model_class=llama33_70b_model.Llama33_70BTransformer1D,
        config_class=llama33_70b_model.Llama33_70BTransformer1DConfig,
        attention_config_class=llama33_70b_model.Attention1DConfig,
        make_attention_config=lambda **kwargs: _make_llama32_attention_config(**kwargs),
        make_config=lambda **kwargs: _make_llama32_config(module=llama33_70b_model, **kwargs),
        make_layer=lambda attention_config=None: _make_llama32_layer(attention_config),
        construct_model=lambda monkeypatch: _construct_llama32_model(monkeypatch, module=llama33_70b_model),
        expected_module_names=(
            "layer[0].attn_norm",
            "layer[0].attention",
            "layer[0].ff_norm",
            "layer[0].mlp",
            "layer[1].attn_norm",
            "layer[1].attention",
            "layer[1].ff_norm",
            "layer[1].mlp",
            "final_norm",
            "lm_head",
        ),
    ),
    "qwen2_7b": SimpleNamespace(
        module=qwen2_model,
        model_class=qwen2_model.Qwen2_7B,
        config_class=qwen2_model.Qwen2_7BTransformerConfig,
        attention_config_class=qwen2_model.Attention1DConfig,
        make_attention_config=lambda **kwargs: _make_llama32_attention_config(**kwargs),
        make_config=lambda **kwargs: _make_qwen2_config(**kwargs),
        make_layer=lambda attention_config=None: _make_llama32_layer(attention_config),
        construct_model=lambda monkeypatch: _construct_qwen2_model(monkeypatch),
        expected_module_names=(
            "layer[0].attn_norm",
            "layer[0].attention",
            "layer[0].ff_norm",
            "layer[0].mlp",
            "layer[1].attn_norm",
            "layer[1].attention",
            "layer[1].ff_norm",
            "layer[1].mlp",
            "final_norm",
            "lm_head",
        ),
    ),
    "qwen25_7b": SimpleNamespace(
        module=qwen25_model,
        model_class=qwen25_model.Qwen25_7B,
        config_class=qwen25_model.Qwen25_7BTransformerConfig,
        attention_config_class=qwen25_model.Attention1DConfig,
        make_attention_config=lambda **kwargs: _make_llama32_attention_config(**kwargs),
        make_config=lambda **kwargs: _make_qwen2_config(module=qwen25_model, **kwargs),
        make_layer=lambda attention_config=None: _make_llama32_layer(attention_config),
        construct_model=lambda monkeypatch: _construct_qwen2_model(monkeypatch, module=qwen25_model),
        expected_module_names=(
            "layer[0].attn_norm",
            "layer[0].attention",
            "layer[0].ff_norm",
            "layer[0].mlp",
            "layer[1].attn_norm",
            "layer[1].attention",
            "layer[1].ff_norm",
            "layer[1].mlp",
            "final_norm",
            "lm_head",
        ),
    ),
    "qwen25_72b": SimpleNamespace(
        module=qwen25_72b_model,
        model_class=qwen25_72b_model.Qwen25_72B,
        config_class=qwen25_72b_model.Qwen25_72BConfig,
        attention_config_class=qwen25_72b_model.Attention1DConfig,
        make_attention_config=lambda **kwargs: _make_llama32_attention_config(**kwargs),
        make_config=lambda **kwargs: _make_qwen25_72b_config(**kwargs),
        make_layer=lambda attention_config=None: _make_llama32_layer(attention_config),
        construct_model=lambda monkeypatch: _construct_qwen25_72b_model(monkeypatch),
        expected_module_names=(
            "layer[0].attn_norm",
            "layer[0].attention",
            "layer[0].ff_norm",
            "layer[0].mlp",
            "layer[1].attn_norm",
            "layer[1].attention",
            "layer[1].ff_norm",
            "layer[1].mlp",
            "final_norm",
            "lm_head",
        ),
    ),
    "qwen25_coder_32b": SimpleNamespace(
        module=qwen25_coder_32b_model,
        model_class=qwen25_coder_32b_model.Qwen25Coder32B,
        config_class=qwen25_coder_32b_model.Qwen25Coder32BConfig,
        attention_config_class=qwen25_coder_32b_model.Attention1DConfig,
        make_attention_config=lambda **kwargs: _make_llama32_attention_config(**kwargs),
        make_config=lambda **kwargs: _make_qwen25_coder_32b_config(**kwargs),
        make_layer=lambda attention_config=None: _make_llama32_layer(attention_config),
        construct_model=lambda monkeypatch: _construct_qwen25_coder_32b_model(monkeypatch),
        expected_module_names=(
            "layer[0].attn_norm",
            "layer[0].attention",
            "layer[0].ff_norm",
            "layer[0].mlp",
            "layer[1].attn_norm",
            "layer[1].attention",
            "layer[1].ff_norm",
            "layer[1].mlp",
            "final_norm",
            "lm_head",
        ),
    ),
    "qwen3_32b": SimpleNamespace(
        module=qwen3_32b_model,
        model_class=qwen3_32b_model.Qwen3_32B,
        config_class=qwen3_32b_model.Qwen3_32BConfig,
        attention_config_class=qwen3_32b_model.Attention1DConfig,
        make_attention_config=lambda **kwargs: _make_llama32_attention_config(**kwargs),
        make_config=lambda **kwargs: _make_qwen3_32b_config(**kwargs),
        make_layer=lambda attention_config=None: _make_llama32_layer(attention_config),
        construct_model=lambda monkeypatch: _construct_qwen3_32b_model(monkeypatch),
        expected_module_names=(
            "layer[0].attn_norm",
            "layer[0].attention",
            "layer[0].ff_norm",
            "layer[0].mlp",
            "layer[1].attn_norm",
            "layer[1].attention",
            "layer[1].ff_norm",
            "layer[1].mlp",
            "final_norm",
            "lm_head",
        ),
    ),
    "deepseek_r1_distill_qwen_14b": SimpleNamespace(
        module=deepseek_model,
        model_class=deepseek_model.DeepSeekR1Qwen14B,
        config_class=deepseek_model.DeepSeekR1Qwen14BTransformerConfig,
        attention_config_class=deepseek_model.Attention1DConfig,
        make_attention_config=lambda **kwargs: _make_llama32_attention_config(**kwargs),
        make_config=lambda **kwargs: _make_qwen2_config(module=deepseek_model, **kwargs),
        make_layer=lambda attention_config=None: _make_llama32_layer(attention_config),
        construct_model=lambda monkeypatch: _construct_qwen2_model(monkeypatch, module=deepseek_model),
        expected_module_names=(
            "layer[0].attn_norm",
            "layer[0].attention",
            "layer[0].ff_norm",
            "layer[0].mlp",
            "layer[1].attn_norm",
            "layer[1].attention",
            "layer[1].ff_norm",
            "layer[1].mlp",
            "final_norm",
            "lm_head",
        ),
    ),
    "mistral_7b": SimpleNamespace(
        module=mistral_model,
        model_class=mistral_model.Mistral7B,
        config_class=mistral_model.Mistral7BTransformerConfig,
        attention_config_class=mistral_model.Attention1DConfig,
        make_attention_config=lambda **kwargs: _make_llama32_attention_config(**kwargs),
        make_config=lambda **kwargs: _make_mistral_config(**kwargs),
        make_layer=lambda attention_config=None: _make_llama32_layer(attention_config),
        construct_model=lambda monkeypatch: _construct_mistral_model(monkeypatch),
        expected_module_names=(
            "layer[0].attn_norm",
            "layer[0].attention",
            "layer[0].ff_norm",
            "layer[0].mlp",
            "layer[1].attn_norm",
            "layer[1].attention",
            "layer[1].ff_norm",
            "layer[1].mlp",
            "final_norm",
            "lm_head",
        ),
    ),
    "phi4": SimpleNamespace(
        module=phi4_model,
        model_class=phi4_model.Phi4Transformer,
        config_class=phi4_model.Phi4TransformerConfig,
        attention_config_class=phi4_model.Attention1DConfig,
        make_attention_config=lambda **kwargs: _make_llama32_attention_config(**kwargs),
        make_config=lambda **kwargs: _make_phi4_config(**kwargs),
        make_layer=lambda attention_config=None: _make_llama32_layer(attention_config),
        construct_model=lambda monkeypatch: _construct_phi4_model(monkeypatch),
        expected_module_names=(
            "layer[0].attn_norm",
            "layer[0].attention",
            "layer[0].ff_norm",
            "layer[0].mlp",
            "layer[1].attn_norm",
            "layer[1].attention",
            "layer[1].ff_norm",
            "layer[1].mlp",
            "final_norm",
            "lm_head",
        ),
    ),
}


@pytest.fixture(params=MODEL_CONTRACTS.items(), ids=lambda item: item[0])
def contract(request):
    return request.param[1]


@dataclass(frozen=True)
class _PagedAttentionConfig:
    block_size: int
    max_num_blocks: int


def _make_llama32_attention_config(*, n_kv_heads=8, kv_cache=None):
    return SimpleNamespace(
        n_kv_heads=n_kv_heads,
        use_vllm_paged_kv_cache=True,
        paged_attention_config=_PagedAttentionConfig(block_size=32, max_num_blocks=128),
        kv_cache=kv_cache,
    )


def _make_llama32_layer(attention_config=None):
    attention_config = attention_config or _make_llama32_attention_config()
    attention = SimpleNamespace(config=attention_config, kv_cache=attention_config.kv_cache)
    return SimpleNamespace(
        attention_norm=object(),
        attention=attention,
        self_attn=attention,
        ff_norm=object(),
        feed_forward=object(),
    )


def _make_llama32_config(*, module=llama32_model, n_layers=1, num_devices=2, n_kv_heads=8, sampling_config=None):
    block_configs = [
        SimpleNamespace(attention_config=_make_llama32_attention_config(n_kv_heads=n_kv_heads)) for _ in range(n_layers)
    ]
    config_class = next(
        getattr(module, name)
        for name in (
            "Llama32_1BTransformer1DConfig",
            "Llama32_3BTransformer1DConfig",
            "Llama33_70BTransformer1DConfig",
        )
        if hasattr(module, name)
    )
    return config_class(
        n_layers=n_layers,
        vocab_size=128256,
        max_batch_size=4,
        max_seq_len=4096,
        dim=2048,
        num_devices=num_devices,
        mesh_device=SimpleNamespace(get_num_devices=lambda: num_devices),
        embedding_config=object(),
        rope_config=object(),
        block_configs=block_configs,
        norm_config=object(),
        lm_head_config=object(),
        sampling_config=sampling_config,
    )


def _construct_llama32_model(monkeypatch, *, module=llama32_model):
    sentinels = {
        "embedding": object(),
        "rope_setup": object(),
        "layer": _make_llama32_layer(),
        "norm": object(),
        "lm_head": object(),
        "sampling": object(),
    }
    for owner_name, sentinel_name in (
        ("Embedding1D", "embedding"),
        ("RotarySetup1D", "rope_setup"),
        ("TransformerBlock1D", "layer"),
        ("RMSNorm1D", "norm"),
        ("LMHead1D", "lm_head"),
        ("Sampling1D", "sampling"),
    ):
        monkeypatch.setattr(
            getattr(module, owner_name),
            "from_config",
            MagicMock(return_value=sentinels[sentinel_name]),
        )
    config = _make_llama32_config(module=module, num_devices=1, sampling_config=object())
    model_class = next(
        getattr(module, name)
        for name in (
            "Llama32_1BTransformer1D",
            "Llama32_3BTransformer1D",
            "Llama33_70BTransformer1D",
        )
        if hasattr(module, name)
    )
    return model_class(config), config, sentinels


def _make_qwen2_config(*, module=qwen2_model, n_layers=1, num_devices=2, n_kv_heads=4, sampling_config=None):
    block_configs = [
        SimpleNamespace(attention_config=_make_llama32_attention_config(n_kv_heads=n_kv_heads)) for _ in range(n_layers)
    ]
    config_class = (
        getattr(module, "Qwen2_7BTransformerConfig", None)
        or getattr(module, "Qwen25_7BTransformerConfig", None)
        or module.DeepSeekR1Qwen14BTransformerConfig
    )
    return config_class(
        n_layers=n_layers,
        vocab_size=152064,
        max_batch_size=4,
        max_seq_len=4096,
        dim=3584,
        num_devices=num_devices,
        mesh_device=SimpleNamespace(get_num_devices=lambda: num_devices),
        embedding_config=object(),
        rope_config=object(),
        block_configs=block_configs,
        norm_config=object(),
        lm_head_config=object(),
        sampling_config=sampling_config,
        tt_ccl=object(),
    )


def _construct_qwen2_model(monkeypatch, *, module=qwen2_model):
    sentinels = {
        "embedding": object(),
        "rope_setup": object(),
        "layer": _make_llama32_layer(),
        "norm": object(),
        "lm_head": object(),
        "sampling": object(),
    }
    layer_class = (
        "Qwen2_7BDecoderLayer"
        if module is qwen2_model
        else ("Qwen25_7BDecoderLayer" if module is qwen25_model else "DeepSeekR1Qwen14BDecoderLayer")
    )
    for owner_name, sentinel_name in (
        ("Embedding1D", "embedding"),
        ("RotarySetup1D", "rope_setup"),
        (layer_class, "layer"),
        ("RMSNorm1D", "norm"),
        ("LMHead1D", "lm_head"),
        ("Sampling1D", "sampling"),
    ):
        monkeypatch.setattr(
            getattr(module, owner_name),
            "from_config",
            MagicMock(return_value=sentinels[sentinel_name]),
        )
    config = _make_qwen2_config(module=module, sampling_config=object())
    model_class = getattr(module, "Qwen2_7B", None) or getattr(module, "Qwen25_7B", None) or module.DeepSeekR1Qwen14B
    return model_class(config), config, sentinels


def _make_qwen25_72b_config(*, n_layers=1, num_devices=8, n_kv_heads=8, sampling_config=None):
    del sampling_config
    block_configs = [
        SimpleNamespace(attention_config=_make_llama32_attention_config(n_kv_heads=n_kv_heads)) for _ in range(n_layers)
    ]
    return qwen25_72b_model.Qwen25_72BConfig(
        hf_model_id="Qwen/Qwen2.5-72B-Instruct",
        dim=8192,
        n_heads=64,
        n_kv_heads=n_kv_heads,
        head_dim=128,
        hidden_dim=29568,
        vocab_size=152064,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        num_hidden_layers=n_layers,
        max_batch_size=4,
        max_seq_len=4096,
        rope_table_len=8192,
        num_devices=num_devices,
        mesh_device=SimpleNamespace(get_num_devices=lambda: num_devices),
        block_configs=block_configs,
    )


def _construct_qwen25_72b_model(monkeypatch):
    sentinels = {
        "embedding": object(),
        "rope_setup": object(),
        "layer": _make_llama32_layer(),
        "norm": object(),
        "lm_head": object(),
        "sampling": object(),
    }
    monkeypatch.setattr(qwen25_72b_model, "get_tt_ccl", MagicMock(return_value=object()))
    monkeypatch.setattr(qwen25_72b_model, "Sampling1D", MagicMock(return_value=sentinels["sampling"]))
    config = _make_qwen25_72b_config()
    return (
        qwen25_72b_model.Qwen25_72B(
            config,
            sentinels["embedding"],
            sentinels["rope_setup"],
            [sentinels["layer"]],
            sentinels["norm"],
            sentinels["lm_head"],
            config.mesh_device,
        ),
        config,
        sentinels,
    )


def _make_qwen25_coder_32b_config(*, n_layers=1, num_devices=8, n_kv_heads=8, sampling_config=None):
    del sampling_config
    block_configs = [
        SimpleNamespace(attention_config=_make_llama32_attention_config(n_kv_heads=n_kv_heads)) for _ in range(n_layers)
    ]
    return qwen25_coder_32b_model.Qwen25Coder32BConfig(
        hf_model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        dim=5120,
        n_heads=40,
        n_kv_heads=n_kv_heads,
        head_dim=128,
        hidden_dim=27648,
        vocab_size=152064,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        num_hidden_layers=n_layers,
        max_batch_size=4,
        max_seq_len=4096,
        rope_table_len=8192,
        num_devices=num_devices,
        mesh_device=SimpleNamespace(get_num_devices=lambda: num_devices),
        block_configs=block_configs,
    )


def _construct_qwen25_coder_32b_model(monkeypatch):
    sentinels = {
        "embedding": object(),
        "rope_setup": object(),
        "layer": _make_llama32_layer(),
        "norm": object(),
        "lm_head": object(),
        "sampling": object(),
    }
    monkeypatch.setattr(qwen25_coder_32b_model, "get_tt_ccl", MagicMock(return_value=object()))
    monkeypatch.setattr(qwen25_coder_32b_model, "Sampling1D", MagicMock(return_value=sentinels["sampling"]))
    config = _make_qwen25_coder_32b_config()
    return (
        qwen25_coder_32b_model.Qwen25Coder32B(
            config,
            sentinels["embedding"],
            sentinels["rope_setup"],
            [sentinels["layer"]],
            sentinels["norm"],
            sentinels["lm_head"],
            config.mesh_device,
        ),
        config,
        sentinels,
    )


def _make_qwen3_32b_config(*, n_layers=1, num_devices=8, n_kv_heads=8, sampling_config=None):
    del sampling_config
    block_configs = [
        SimpleNamespace(attention_config=_make_llama32_attention_config(n_kv_heads=n_kv_heads)) for _ in range(n_layers)
    ]
    return qwen3_32b_model.Qwen3_32BConfig(
        hf_model_id="Qwen/Qwen3-32B",
        dim=5120,
        n_heads=64,
        n_kv_heads=n_kv_heads,
        head_dim=128,
        hidden_dim=27648,
        vocab_size=151936,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        num_hidden_layers=n_layers,
        max_batch_size=4,
        max_seq_len=4096,
        rope_table_len=8192,
        num_devices=num_devices,
        mesh_device=SimpleNamespace(get_num_devices=lambda: num_devices),
        block_configs=block_configs,
    )


def _construct_qwen3_32b_model(monkeypatch):
    sentinels = {
        "embedding": object(),
        "rope_setup": object(),
        "layer": _make_llama32_layer(),
        "norm": object(),
        "lm_head": object(),
        "sampling": object(),
    }
    monkeypatch.setattr(qwen3_32b_model, "get_tt_ccl", MagicMock(return_value=object()))
    monkeypatch.setattr(qwen3_32b_model, "Sampling1D", MagicMock(return_value=sentinels["sampling"]))
    monkeypatch.setattr(qwen3_32b_model.weight_utils, "lm_head_padded_vocab_size", MagicMock(return_value=152064))
    config = _make_qwen3_32b_config()
    return (
        qwen3_32b_model.Qwen3_32B(
            config,
            sentinels["embedding"],
            sentinels["rope_setup"],
            [sentinels["layer"]],
            sentinels["norm"],
            sentinels["lm_head"],
            config.mesh_device,
        ),
        config,
        sentinels,
    )


def _make_mistral_config(*, n_layers=1, num_devices=2, n_kv_heads=8, sampling_config=None):
    block_configs = [
        SimpleNamespace(attention_config=_make_llama32_attention_config(n_kv_heads=n_kv_heads)) for _ in range(n_layers)
    ]
    return mistral_model.Mistral7BTransformerConfig(
        n_layers=n_layers,
        vocab_size=32768,
        max_batch_size=4,
        max_seq_len=4096,
        dim=4096,
        num_devices=num_devices,
        mesh_device=object(),
        embedding_config=object(),
        rope_config=object(),
        block_configs=block_configs,
        norm_config=object(),
        lm_head_config=object(),
        sampling_config=sampling_config,
        tt_ccl=object(),
    )


def _construct_mistral_model(monkeypatch):
    sentinels = {
        "embedding": object(),
        "rope_setup": object(),
        "layer": _make_llama32_layer(),
        "norm": object(),
        "lm_head": object(),
        "sampling": object(),
    }
    for owner_name, sentinel_name in (
        ("Embedding1D", "embedding"),
        ("RotarySetup1D", "rope_setup"),
        ("Mistral7BDecoderLayer", "layer"),
        ("RMSNorm1D", "norm"),
        ("LMHead1D", "lm_head"),
        ("Sampling1D", "sampling"),
    ):
        monkeypatch.setattr(
            getattr(mistral_model, owner_name),
            "from_config",
            MagicMock(return_value=sentinels[sentinel_name]),
        )
    config = _make_mistral_config(sampling_config=object())
    return mistral_model.Mistral7B(config), config, sentinels


def _make_phi4_config(*, n_layers=1, num_devices=2, n_kv_heads=10, sampling_config=None):
    block_configs = [
        SimpleNamespace(attention_config=_make_llama32_attention_config(n_kv_heads=n_kv_heads)) for _ in range(n_layers)
    ]
    return phi4_model.Phi4TransformerConfig(
        n_layers=n_layers,
        vocab_size=100352,
        max_batch_size=4,
        max_seq_len=4096,
        dim=5120,
        num_devices=num_devices,
        mesh_device=object(),
        embedding_config=object(),
        rope_config=object(),
        block_configs=block_configs,
        norm_config=object(),
        lm_head_config=object(),
        sampling_config=sampling_config,
        tt_ccl=object(),
    )


def _construct_phi4_model(monkeypatch):
    sentinels = {
        "embedding": object(),
        "rope_setup": object(),
        "layer": _make_llama32_layer(),
        "norm": object(),
        "lm_head": object(),
        "sampling": object(),
    }
    for owner_name, sentinel_name in (
        ("Embedding1D", "embedding"),
        ("RotarySetup1D", "rope_setup"),
        ("Phi4DecoderLayer", "layer"),
        ("RMSNorm1D", "norm"),
        ("LMHead1D", "lm_head"),
        ("Sampling1D", "sampling"),
    ):
        monkeypatch.setattr(
            getattr(phi4_model, owner_name),
            "from_config",
            MagicMock(return_value=sentinels[sentinel_name]),
        )
    config = _make_phi4_config(sampling_config=object())
    return phi4_model.Phi4Transformer(config), config, sentinels


def test_config_exposes_complete_runtime_metadata(contract):
    names = {field.name for field in fields(contract.config_class)}
    assert {
        "dim",
        "mesh_device",
        "num_devices",
        "n_layers",
        "max_batch_size",
        "max_seq_len",
        "block_configs",
    } <= names
    assert "n_kv_heads" in {field.name for field in fields(contract.attention_config_class)}

    config = contract.make_config(n_layers=2, num_devices=2, n_kv_heads=8)
    assert len(config.block_configs) == config.n_layers
    for block in config.block_configs:
        assert hasattr(block.attention_config, "n_kv_heads")
        assert block.attention_config.n_kv_heads % config.num_devices == 0


@pytest.mark.parametrize(
    "method,names,keyword_only,defaults",
    [
        (
            "prefill_forward",
            (
                "self",
                "x_embed",
                "rot_mats",
                "user_id",
                "page_table",
                "chunk_page_table",
                "chunk_start_idx",
                "get_last_token",
                "batch_size",
                "chunk_start_idx_tensor",
                "last_token_slice",
                "last_token_index",
            ),
            (),
            {
                "user_id": 0,
                "page_table": None,
                "chunk_page_table": None,
                "chunk_start_idx": None,
                "get_last_token": -1,
                "batch_size": 1,
                "chunk_start_idx_tensor": None,
                "last_token_slice": None,
                "last_token_index": None,
            },
        ),
        (
            "post_process_prefill_output",
            ("self", "hidden_states", "last_token_idx", "last_token_slice", "last_token_index"),
            (),
            {"last_token_slice": None, "last_token_index": None},
        ),
        (
            "post_process_batched_prefill_output",
            (
                "self",
                "hidden_states",
                "last_token_idx_list",
                "padded_batch",
                "prefill_seq_len",
                "last_token_slice",
                "last_token_index",
            ),
            (),
            {"last_token_slice": None, "last_token_index": None},
        ),
        ("set_kv_cache", ("self", "kv_cache"), (), {}),
        (
            "configure_paged_attention",
            ("self", "block_size", "max_num_blocks"),
            ("block_size", "max_num_blocks"),
            {},
        ),
        ("prepare_prefill_rot_mats", ("self", "position_indices"), (), {}),
        ("iter_executor_named_modules", ("self",), (), {}),
    ],
)
def test_method_signatures_are_exact(contract, method, names, keyword_only, defaults):
    signature = inspect.signature(getattr(contract.model_class, method))
    parameters = signature.parameters
    assert tuple(parameters) == names
    for name, parameter in parameters.items():
        expected_kind = (
            inspect.Parameter.KEYWORD_ONLY if name in keyword_only else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        assert parameter.kind is expected_kind
        expected_default = defaults.get(name, inspect.Parameter.empty)
        assert parameter.default == expected_default

    annotations = {
        "prefill_forward": (
            (
                inspect.Parameter.empty,
                "ttnn.Tensor",
                "tuple[ttnn.Tensor, ttnn.Tensor]",
                "int",
                "ttnn.Tensor | None",
                "ttnn.Tensor | None",
                "int | None",
                "int",
                "int",
                "ttnn.Tensor | None",
                "tuple[ttnn.Tensor, ttnn.Tensor] | None",
                "ttnn.Tensor | None",
            ),
            "ttnn.Tensor",
        ),
        "post_process_prefill_output": (
            (
                inspect.Parameter.empty,
                "ttnn.Tensor",
                "int",
                "tuple[ttnn.Tensor, ttnn.Tensor] | None",
                "ttnn.Tensor | None",
            ),
            "ttnn.Tensor",
        ),
        "post_process_batched_prefill_output": (
            (
                inspect.Parameter.empty,
                "ttnn.Tensor",
                "list[int]",
                "int",
                "int",
                "tuple[ttnn.Tensor, ttnn.Tensor] | None",
                "ttnn.Tensor | None",
            ),
            "ttnn.Tensor",
        ),
        "set_kv_cache": ((inspect.Parameter.empty, "list | None"), "None"),
        "configure_paged_attention": ((inspect.Parameter.empty, "int", "int"), "None"),
        "prepare_prefill_rot_mats": (
            (inspect.Parameter.empty, "ttnn.Tensor"),
            "tuple[ttnn.Tensor, ttnn.Tensor]",
        ),
        "iter_executor_named_modules": (
            (inspect.Parameter.empty,),
            inspect.Signature.empty,
        ),
    }
    expected_parameter_annotations, expected_return_annotation = annotations[method]
    assert tuple(parameter.annotation for parameter in parameters.values()) == expected_parameter_annotations
    assert signature.return_annotation == expected_return_annotation


def test_required_methods_exist(contract):
    for method in (
        "set_kv_cache",
        "configure_paged_attention",
        "prepare_prefill_rot_mats",
        "iter_executor_named_modules",
        "embed_decode",
        "embed_prefill",
        "gather_and_untilize_logits",
        "increment_positions",
    ):
        assert callable(getattr(contract.model_class, method))


def test_constructed_model_resolves_runtime_surface(contract, monkeypatch):
    model, config, sentinels = contract.construct_model(monkeypatch)

    assert model.config is config
    assert model.embedding is sentinels["embedding"]
    assert model.rope_setup is sentinels["rope_setup"]
    assert model.layers == [sentinels["layer"]]
    assert model.norm is sentinels["norm"]
    assert model.lm_head is sentinels["lm_head"]
    assert model.sampling is sentinels["sampling"]
    assert model.supports_on_device_sampling
    assert model.mesh_device is config.mesh_device
    assert model.vocab_size == config.vocab_size
    assert model.n_layers == config.n_layers
    assert model.num_devices == config.num_devices
    assert model.model_args is None


def test_named_modules_are_complete_unique_and_ordered(contract):
    layers = [contract.make_layer(), contract.make_layer()]
    model = SimpleNamespace(layers=layers, norm=object(), lm_head=object())
    named = list(contract.model_class.iter_executor_named_modules(model))

    assert tuple(name for name, _ in named) == contract.expected_module_names
    assert len(named) == 4 * len(layers) + 2
    assert len({name for name, _ in named}) == len(named)
    assert tuple(module for _, module in named) == (
        layers[0].attention_norm,
        layers[0].attention,
        layers[0].ff_norm,
        layers[0].feed_forward,
        layers[1].attention_norm,
        layers[1].attention,
        layers[1].ff_norm,
        layers[1].feed_forward,
        model.norm,
        model.lm_head,
    )


def test_named_modules_without_layers_yields_nothing(contract):
    assert list(contract.model_class.iter_executor_named_modules(SimpleNamespace())) == []


def test_set_kv_cache_binds_identity_and_unbinds_idempotently(contract):
    layers = [contract.make_layer(), contract.make_layer()]
    model = SimpleNamespace(layers=layers)
    cache = [[object(), object()], [object(), object()]]

    contract.model_class.set_kv_cache(model, cache)
    for layer, expected in zip(layers, cache):
        bound = layer.attention.config.kv_cache
        assert bound == tuple(expected)
        assert bound[0] is expected[0]
        assert bound[1] is expected[1]
        assert layer.attention.kv_cache is bound

    contract.model_class.set_kv_cache(model, None)
    contract.model_class.set_kv_cache(model, None)
    assert all(layer.attention.config.kv_cache is None for layer in layers)
    assert all(layer.attention.kv_cache is None for layer in layers)


def test_set_kv_cache_rejects_wrong_layer_count_before_binding(contract, expect_error):
    layers = [contract.make_layer(), contract.make_layer()]
    model = SimpleNamespace(layers=layers)

    with expect_error(ValueError, "model has 2 layers"):
        contract.model_class.set_kv_cache(model, [[object(), object()]])

    assert all(layer.attention.config.kv_cache is None for layer in layers)
    assert all(layer.attention.kv_cache is None for layer in layers)


@pytest.mark.parametrize("bad_pair", [[object()], object()])
def test_set_kv_cache_validates_all_pairs_before_binding(contract, bad_pair, expect_error):
    layers = [contract.make_layer(), contract.make_layer()]
    model = SimpleNamespace(layers=layers)

    with expect_error((TypeError, ValueError), "layer 1.*K/V tensor"):
        contract.model_class.set_kv_cache(model, [[object(), object()], bad_pair])

    assert all(layer.attention.config.kv_cache is None for layer in layers)
    assert all(layer.attention.kv_cache is None for layer in layers)


def test_configure_paged_attention_updates_construction_and_live_configs_and_rejects_bound_cache(
    contract, expect_error
):
    construction = contract.make_attention_config()
    live = contract.make_attention_config()
    model = SimpleNamespace(
        config=SimpleNamespace(block_configs=(SimpleNamespace(attention_config=construction),)),
        layers=(contract.make_layer(live),),
    )

    contract.model_class.configure_paged_attention(model, block_size=16, max_num_blocks=200)
    assert construction.paged_attention_config.block_size == live.paged_attention_config.block_size == 16
    assert construction.paged_attention_config.max_num_blocks == live.paged_attention_config.max_num_blocks == 200

    bound = object()
    live.kv_cache = (bound, bound)
    with expect_error(RuntimeError, "already has a bound KV cache"):
        contract.model_class.configure_paged_attention(model, block_size=32, max_num_blocks=128)
    assert construction.paged_attention_config.block_size == live.paged_attention_config.block_size == 16


def test_prepare_prefill_rot_mats_gathers_device_rows(contract, monkeypatch):
    position_indices = object()
    cos_matrix, sin_matrix = object(), object()
    cos_rows, sin_rows = object(), object()
    cos_4d, sin_4d = object(), object()
    fake_ttnn = SimpleNamespace(
        TILE_LAYOUT=object(),
        embedding=MagicMock(side_effect=[cos_rows, sin_rows]),
        unsqueeze_to_4D=MagicMock(side_effect=[cos_4d, sin_4d]),
    )
    rope = SimpleNamespace(cos_matrix=cos_matrix, sin_matrix=sin_matrix, load_device_weights=MagicMock())
    monkeypatch.setattr(contract.module, "ttnn", fake_ttnn)

    result = contract.model_class.prepare_prefill_rot_mats(SimpleNamespace(rope_setup=rope), position_indices)

    rope.load_device_weights.assert_called_once_with()
    assert fake_ttnn.embedding.call_args_list == [
        call(position_indices, cos_matrix, layout=fake_ttnn.TILE_LAYOUT),
        call(position_indices, sin_matrix, layout=fake_ttnn.TILE_LAYOUT),
    ]
    assert result == (cos_4d, sin_4d)
