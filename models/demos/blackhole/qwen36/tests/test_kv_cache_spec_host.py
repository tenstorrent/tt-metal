# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the QWEN36_MAMBA_SPEC layer specs (tt/kv_cache_spec.py) through vLLM's own grouping and pool layout.

Needs vLLM (0.31) and no device; skipped where vLLM is not installed. Qwen3.6-27B / Qwen3.8-27B text config:
64 layers, 16 full attention (4 KV heads x 256) and 48 GDN (16 key heads, 48 value heads, 128 x 128, conv kernel 4).

    pytest --noconftest models/demos/blackhole/qwen36/tests/test_kv_cache_spec_host.py
"""

import pytest
import torch

kv_iface = pytest.importorskip("vllm.v1.kv_cache_interface")
kv_cache_utils = pytest.importorskip("vllm.v1.core.kv_cache_utils")

from models.demos.blackhole.qwen36.tt.kv_cache_spec import bf8_bytes, qwen36_kv_cache_spec  # noqa: E402

FullAttentionSpec, MambaSpec = kv_iface.FullAttentionSpec, kv_iface.MambaSpec

TEXT_CONFIG = {
    "num_hidden_layers": 64,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "hidden_size": 5120,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}
K = 3  # MTP draft length (QWEN36_SPEC_K default)


@pytest.fixture(scope="module")
def vllm_config():
    from vllm.config import DeviceConfig, ModelConfig, VllmConfig

    # ModelConfig's default model caps max_model_len; the grouping under test does not depend on it.
    config = VllmConfig(model_config=ModelConfig(max_model_len=32768), device_config=DeviceConfig(device="cpu"))
    # The TT platform reports support_hybrid_kv_cache(); the CPU platform used here does not.
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    # The engine core resolves the layout from the worker; the TT worker advertises only LBNHC.
    config.cache_config.kv_cache_layout = "LBNHC"
    return config


def _split(specs):
    attn = {n: s for n, s in specs.items() if isinstance(s, FullAttentionSpec)}
    gdn = {n: s for n, s in specs.items() if isinstance(s, MambaSpec)}
    return attn, gdn


def test_bf8_page_bytes():
    # One K or V tile row block of [block, 256] BF8: whole 32x32 tiles of 1088 bytes.
    assert bf8_bytes(2 * 256) * 1024 == 2 * (1024 * 256 // 1024) * 1088 == 557056
    with pytest.raises(AssertionError):
        bf8_bytes(8)


@pytest.mark.parametrize("block", [1024, 2048])
def test_tp8_specs(block):
    specs = qwen36_kv_cache_spec(TEXT_CONFIG, 8, block, kv_bf8=True, num_speculative_blocks=K)
    attn, gdn = _split(specs)
    assert len(specs) == 64 and len(attn) == 16 and len(gdn) == 48
    assert sorted(int(n.split(".")[2]) for n in attn) == list(range(3, 64, 4))
    a, g = next(iter(attn.values())), next(iter(gdn.values()))
    assert a.num_kv_heads == 4 and a.num_heads == 1 and a.head_size == 256
    assert a.page_size_bytes == block * 2 * 256 * 17 // 16  # BF8 K and V of one head per device
    assert g.shapes == ((3, 1280), (6, 128, 128)) and g.dtypes == (torch.bfloat16, torch.float32)
    assert g.real_page_size_bytes == 3 * 1280 * 2 + 6 * 128 * 128 * 4 == 400896
    assert g.page_size_bytes == a.page_size_bytes
    assert g.num_speculative_blocks == K and g.mamba_cache_mode == "none"


def test_gdn_state_must_fit_the_attention_page():
    # TP=4 holds 12 value heads per device (801792 B of state): a 1024-token BF8 page (557056 B) is too small.
    with pytest.raises(ValueError, match="does not fit"):
        qwen36_kv_cache_spec(TEXT_CONFIG, 4, 1024, kv_bf8=True)
    qwen36_kv_cache_spec(TEXT_CONFIG, 4, 2048, kv_bf8=True)
    qwen36_kv_cache_spec(TEXT_CONFIG, 4, 1024, kv_bf8=False)  # bf16 page: 1 MiB


@pytest.mark.parametrize(
    "tp, block, kv_bf8, mamba_block",
    [
        (8, 1024, True, None),
        (8, 2048, True, None),
        (8, 1024, False, None),
        (4, 2048, True, None),
        (4, 2048, False, 8192),
        (4, 4096, True, 8192),
        (8, 2048, True, 8192),
    ],
)
@pytest.mark.parametrize("spec_blocks", [0, K], ids=["no_mtp", "mtp_k3"])
def test_vllm_groups_and_pool(vllm_config, tp, block, kv_bf8, mamba_block, spec_blocks):
    """vLLM makes 1 attention group and 3 GDN groups of 16 layers with one page size, and lays layer j of every group
    at the same byte, so block id b names the same page in the attention layer and in its three GDN layers. The last
    case is the mode-none prototype layout: bf16 KV, GDN block = max_model_len."""
    specs = qwen36_kv_cache_spec(
        TEXT_CONFIG, tp, block, kv_bf8=kv_bf8, num_speculative_blocks=spec_blocks, mamba_block_size=mamba_block
    )
    page = next(s for s in specs.values() if isinstance(s, FullAttentionSpec)).page_size_bytes

    groups = kv_cache_utils.get_kv_cache_groups(vllm_config, specs)
    attn_groups = [g for g in groups if isinstance(g.kv_cache_spec, FullAttentionSpec)]
    gdn_groups = [g for g in groups if isinstance(g.kv_cache_spec, MambaSpec)]
    assert len(groups) == 4 and len(attn_groups) == 1 and len(gdn_groups) == 3
    assert [len(g.layer_names) for g in groups] == [16] * 4
    assert {g.kv_cache_spec.page_size_bytes for g in groups} == {page}
    assert {g.kv_cache_spec.block_size for g in attn_groups} == {block}
    assert {g.kv_cache_spec.block_size for g in gdn_groups} == {mamba_block or block}

    bytes_per_block = kv_cache_utils._pool_bytes_per_block(groups)
    assert bytes_per_block == 16 * page
    config = kv_cache_utils.get_kv_cache_config_from_groups(vllm_config, groups, bytes_per_block * 100 + 1)
    assert config.num_blocks == 100
    tensor_of = {name: t for t in config.kv_cache_tensors for name in t.layers}

    def start(name):
        t = tensor_of[name]
        return t.offset + t.layers.index(name) * t.layer_stride

    for j, attn_name in enumerate(attn_groups[0].layer_names):
        for g in gdn_groups:
            assert start(g.layer_names[j]) == start(attn_name), (j, attn_name, g.layer_names[j])
            assert tensor_of[g.layer_names[j]].block_stride == tensor_of[attn_name].block_stride == page
