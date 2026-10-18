# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from models.common.models.qwen3_32b import model as qwen_model


def _attention_config():
    return SimpleNamespace(
        n_kv_heads=8,
        head_dim=128,
        kv_cache_dtype=qwen_model.ttnn.bfloat8_b,
        use_vllm_paged_kv_cache=True,
        paged_attention_config=qwen_model.Qwen3_32BPagedAttentionConfig(block_size=32, max_num_blocks=128),
        kv_cache=None,
    )


def _layer(attention_config=None):
    attention = SimpleNamespace(config=attention_config or _attention_config(), kv_cache=None)
    return SimpleNamespace(
        input_layernorm=object(),
        self_attn=attention,
        post_attention_layernorm=object(),
        mlp=object(),
        attention_norm=object(),
        attention=attention,
        ff_norm=object(),
        feed_forward=object(),
    )


def test_named_modules_use_canonical_runtime_order_with_legacy_layer_names():
    layers = [_layer(), _layer()]
    model = SimpleNamespace(layers=layers, norm=object(), lm_head=object())

    named = list(qwen_model.Qwen3_32B.iter_executor_named_modules(model))

    assert tuple(name for name, _ in named) == (
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
    )


def test_set_kv_cache_binds_and_unbinds_self_attention_aliases():
    layers = [_layer(), _layer()]
    model = SimpleNamespace(layers=layers)
    cache = [[object(), object()], [object(), object()]]

    qwen_model.Qwen3_32B.set_kv_cache(model, cache)

    for layer, pair in zip(layers, cache):
        assert layer.self_attn.config.kv_cache == tuple(pair)
        assert layer.self_attn.kv_cache == tuple(pair)

    qwen_model.Qwen3_32B.set_kv_cache(model, None)
    assert all(layer.self_attn.config.kv_cache is None for layer in layers)
    assert all(layer.self_attn.kv_cache is None for layer in layers)


def test_configure_paged_attention_updates_live_and_construction_configs(expect_error):
    attention_config = _attention_config()
    model = SimpleNamespace(
        config=SimpleNamespace(block_configs=[SimpleNamespace(attention_config=attention_config)]),
        layers=[_layer(attention_config)],
    )

    qwen_model.Qwen3_32B.configure_paged_attention(model, block_size=16, max_num_blocks=200)

    assert attention_config.paged_attention_config.block_size == 16
    assert attention_config.paged_attention_config.max_num_blocks == 200

    attention_config.kv_cache = (object(), object())
    with expect_error(RuntimeError, "already has a bound KV cache"):
        qwen_model.Qwen3_32B.configure_paged_attention(model, block_size=32, max_num_blocks=128)


def test_all_gather_rmsnorm_honors_memory_config_when_tensor_is_already_full_width(monkeypatch):
    requested_memory_config = object()
    converted_tensor = object()
    x = SimpleNamespace(shape=(1, 1, 32, 5120))
    norm = SimpleNamespace(
        config=SimpleNamespace(
            mesh_device=SimpleNamespace(get_num_devices=lambda: 8),
            weight=SimpleNamespace(source=SimpleNamespace(numel=lambda: 5120)),
        )
    )
    calls = []

    def fake_to_memory_config(tensor, memory_config):
        calls.append((tensor, memory_config))
        return converted_tensor

    monkeypatch.setattr(qwen_model.ttnn, "to_memory_config", fake_to_memory_config)

    assert qwen_model._all_gather_rmsnorm_tensor(norm, x, memory_config=requested_memory_config) is converted_tensor
    assert calls == [(x, requested_memory_config)]


@pytest.mark.parametrize(
    "cluster_type",
    [qwen_model.ttnn.cluster.ClusterType.P150_X4, qwen_model.ttnn.cluster.ClusterType.P300_X2],
)
def test_qwen_rmsnorm_and_logits_all_gathers_pin_ring_for_bh_four_die_products(cluster_type, monkeypatch):
    mesh = SimpleNamespace(
        arch=lambda: qwen_model.ttnn.device.Arch.BLACKHOLE,
        get_num_devices=lambda: 4,
    )
    ccl = SimpleNamespace(
        get_and_cycle_ag_semaphore_handles=lambda: object(),
        get_and_cycle_barrier_semaphore_handle=lambda: object(),
    )
    memory_config = object()
    tensor = SimpleNamespace(shape=(1, 1, 32, 1280), memory_config=lambda: memory_config)
    norm = SimpleNamespace(
        config=SimpleNamespace(
            mesh_device=mesh,
            weight=SimpleNamespace(source=SimpleNamespace(numel=lambda: 5120)),
            tt_ccl=ccl,
        )
    )
    topologies = []

    def fake_all_gather(value, **kwargs):
        topologies.append(kwargs["topology"])
        return value

    monkeypatch.setattr(qwen_model.ttnn.cluster, "get_cluster_type", lambda: cluster_type)
    monkeypatch.setattr(qwen_model.ttnn.experimental, "all_gather_async", fake_all_gather)
    monkeypatch.setattr(qwen_model.ttnn, "untilize", lambda value, **_kwargs: value)

    assert qwen_model._all_gather_rmsnorm_tensor(norm, tensor) is tensor
    model = SimpleNamespace(num_devices=4, tt_ccl=ccl, mesh_device=mesh)
    assert qwen_model.Qwen3_32B.gather_and_untilize_logits(model, tensor) is tensor
    assert topologies == [qwen_model.ttnn.Topology.Ring, qwen_model.ttnn.Topology.Ring]


def test_qwen_ccl_topology_preserves_wormhole_t3k_ring(monkeypatch):
    mesh = SimpleNamespace(
        arch=lambda: qwen_model.ttnn.device.Arch.WORMHOLE_B0,
        get_num_devices=lambda: 8,
    )
    monkeypatch.setattr(
        qwen_model.ttnn.cluster,
        "get_cluster_type",
        lambda: qwen_model.ttnn.cluster.ClusterType.T3K,
    )

    assert qwen_model._qwen3_ccl_topology(mesh) == qwen_model.ttnn.Topology.Ring


@pytest.mark.parametrize(
    ("arch", "cluster_type", "num_devices"),
    [
        (qwen_model.ttnn.device.Arch.BLACKHOLE, qwen_model.ttnn.cluster.ClusterType.P150_X8, 4),
        (qwen_model.ttnn.device.Arch.BLACKHOLE, qwen_model.ttnn.cluster.ClusterType.P150_X4, 8),
        (qwen_model.ttnn.device.Arch.WORMHOLE_B0, qwen_model.ttnn.cluster.ClusterType.T3K, 4),
    ],
)
def test_qwen_ccl_topology_rejects_mismatched_product_identity(
    arch, cluster_type, num_devices, monkeypatch, expect_error
):
    mesh = SimpleNamespace(arch=lambda: arch, get_num_devices=lambda: num_devices)
    monkeypatch.setattr(qwen_model.ttnn.cluster, "get_cluster_type", lambda: cluster_type)

    with expect_error(ValueError, "Qwen3-32B CCL supports"):
        qwen_model._qwen3_ccl_topology(mesh)


def test_decode_reshards_final_norm_output_to_lm_head_input_memory_config(monkeypatch):
    """Guard the BH final-norm -> LMHead sharding boundary without opening hardware."""

    decode_norm_memcfg = object()
    lm_head_memcfg = SimpleNamespace(is_sharded=lambda: True)
    gathered = object()
    normalized = SimpleNamespace(memory_config=lambda: decode_norm_memcfg)
    resharded = object()
    logits = object()
    calls = []

    norm = SimpleNamespace(
        config=SimpleNamespace(decode_memory_config=decode_norm_memcfg),
        decode_forward=lambda x: calls.append(("norm", x)) or normalized,
    )
    lm_head = SimpleNamespace(
        config=SimpleNamespace(input_memcfg=lm_head_memcfg),
        forward=lambda x: calls.append(("lm_head", x)) or logits,
    )
    model = SimpleNamespace(layers=[], norm=norm, lm_head=lm_head)

    def fake_all_gather(final_norm, x, *, memory_config):
        calls.append(("all_gather", final_norm, x, memory_config))
        return gathered

    def fake_reshard(x, memory_config):
        calls.append(("reshard", x, memory_config))
        return resharded

    monkeypatch.setattr(qwen_model, "_all_gather_rmsnorm_tensor", fake_all_gather)
    monkeypatch.setattr(qwen_model.ttnn, "reshard", fake_reshard)

    x_embed = object()
    assert qwen_model.Qwen3_32B.decode_forward(model, x_embed, object(), (object(), object())) is logits
    assert calls == [
        ("all_gather", norm, x_embed, decode_norm_memcfg),
        ("norm", gathered),
        ("reshard", normalized, lm_head_memcfg),
        ("lm_head", resharded),
    ]


def test_decoder_layer_prefill_calls_chunk_capable_attention_entrypoint(monkeypatch):
    captured = {}
    attention_output = object()
    final_output = object()
    attention = SimpleNamespace(
        prefill_forward=lambda x, rot_mats, **kwargs: captured.update(attention=(x, rot_mats, kwargs))
        or attention_output
    )
    layer = qwen_model.Qwen3_32BDecoderLayer(
        input_layernorm=SimpleNamespace(prefill_forward=lambda x: x),
        self_attn=attention,
        post_attention_layernorm=SimpleNamespace(prefill_forward=lambda x: x),
        mlp=SimpleNamespace(prefill_forward=lambda x: x),
    )
    monkeypatch.setattr(qwen_model, "_all_gather_rmsnorm_tensor", lambda _norm, x, **_kwargs: x)
    monkeypatch.setattr(qwen_model.ttnn, "add", lambda *_args, **_kwargs: final_output)

    chunk_start_idx_tensor = object()
    rot_mats = (object(), object())
    assert (
        layer.prefill_forward(
            object(),
            rot_mats,
            user_id=[0, 1],
            page_table=object(),
            chunk_page_table=object(),
            chunk_start_idx=128,
            batch_size=2,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
        )
        is final_output
    )
    assert captured["attention"][1] is rot_mats
    assert captured["attention"][2]["chunk_start_idx_tensor"] is chunk_start_idx_tensor
    assert captured["attention"][2]["batch_size"] == 2
