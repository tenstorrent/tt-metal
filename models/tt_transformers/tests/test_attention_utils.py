# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import torch

from models.tt_transformers.tt import attention as attention_module


class _StopAfterBiasConstruction(Exception):
    pass


class _FakeDecoderOptimizations:
    def get_tensor_dtype(self, **_):
        return None

    def get_math_fidelity(self, **_):
        return None


def test_galaxy_qkv_bias_uses_weight_matching_2d_sharding_and_survives_column_reduce(monkeypatch, expect_error):
    mesh_device = object()
    mesh_mapper_calls = []
    as_tensor_calls = []

    monkeypatch.setattr(attention_module.ttnn, "ShardTensorToMesh", lambda *args, **kwargs: object())
    monkeypatch.setattr(attention_module.ttnn, "ReplicateTensorToMesh", lambda *args, **kwargs: object())
    monkeypatch.setattr(attention_module.ttnn, "from_torch", lambda *args, **kwargs: object())

    def shard_tensor_2d_mesh(device, *, dims, mesh_shape):
        mapper = object()
        mesh_mapper_calls.append((device, dims, mesh_shape, mapper))
        return mapper

    def as_tensor(tensor, **kwargs):
        as_tensor_calls.append((tensor, kwargs))
        return SimpleNamespace(shape=(1, tensor.shape[-1]))

    monkeypatch.setattr(attention_module.ttnn, "ShardTensor2dMesh", shard_tensor_2d_mesh)
    monkeypatch.setattr(attention_module.ttnn, "as_tensor", as_tensor)
    monkeypatch.setattr(
        attention_module.ttnn,
        "reshape",
        lambda *args, **kwargs: (_ for _ in ()).throw(_StopAfterBiasConstruction("bias captured")),
    )

    configuration = SimpleNamespace(
        num_devices=32,
        dim=8192,
        n_heads=64,
        head_dim=128,
        max_seq_len=32768,
        max_batch_size=32,
        n_kv_heads=8,
        min_kv_prefill_shard_seqlen=0,
        ccl_dtype=None,
        MAX_QKV_MM_SEQ_LEN=2048,
        tile_size=32,
        tile_padded_batch_rows=32,
        rms_norm_add_unit_offset=False,
        use_qk_fused=False,
        use_hf_rope=False,
        arch_name="wormhole_b0",
        max_grid_size=None,
        compute_kernel_config_hifi2=None,
        compute_kernel_config_hifi2_fp16=None,
        compute_kernel_config_hifi4=None,
        layer_types=None,
        cluster_shape=[8, 4],
        is_multichip=True,
        dummy_weights=False,
        qkv_size=10240,
        get_model_config=lambda: {},
        ccl_topology=lambda: None,
        get_state_dict_prefix=lambda *_: "layers.0.attention",
    )
    args = SimpleNamespace(decoders_optimizations=_FakeDecoderOptimizations())
    state_dict = {
        "layers.0.attention.wq.bias": torch.arange(16),
        "layers.0.attention.wk.bias": torch.arange(100, 108),
        "layers.0.attention.wv.bias": torch.arange(200, 208),
    }

    with expect_error(_StopAfterBiasConstruction, "bias captured"):
        attention_module.Attention(
            mesh_device=mesh_device,
            tt_ccl=None,
            args=args,
            state_dict=state_dict,
            weight_cache_path=Path("cache"),
            layer_num=0,
            dtype=None,
            transformation_mats=None,
            configuration=configuration,
        )

    assert len(mesh_mapper_calls) == 1
    assert mesh_mapper_calls[0][:3] == (mesh_device, (-1, None), [8, 4])
    qkv_bias, tensor_kwargs = as_tensor_calls[0]
    assert tensor_kwargs["mesh_mapper"] is mesh_mapper_calls[0][3]
    assert tensor_kwargs["cache_file_name"].name.endswith("wqkv_bias_prefill_sharded_2d_col_reduce_4")

    # Each row shard is replicated over all four mesh columns and added before
    # the column all-reduce. Their sum must restore one copy of the model bias,
    # rather than the four copies produced by an unscaled replicated bias.
    expected_device_shards = [
        torch.tensor([2 * i, 2 * i + 1, 100 + i, 200 + i], dtype=qkv_bias.dtype) for i in range(8)
    ]
    reduced_device_shards = [shard * configuration.cluster_shape[1] for shard in torch.chunk(qkv_bias, 8)]
    assert all(torch.equal(actual, expected) for actual, expected in zip(reduced_device_shards, expected_device_shards))
