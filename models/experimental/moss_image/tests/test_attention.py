# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fixed-input numerical regression for model-local accurate attention.

Two-card cases replicate the local attention problem and check both replicas;
full DiT TP collectives and independent generation are qualified separately.
"""
import math

import pytest
import torch
import ttnn

from models.experimental.moss_image.tt.attention import AccurateAttention
from models.experimental.moss_image.validation.accuracy_review import error_metrics


def attention_reference(q, k, v, *, mask=None, causal=False, round_probabilities=False):
    """Deterministic dense FP64 oracle, optionally with the eager BF16 P boundary."""
    repeat = q.shape[1] // k.shape[1]
    k = k.double().repeat_interleave(repeat, dim=1)
    v = v.double().repeat_interleave(repeat, dim=1)
    logits = q.double() @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        future = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).triu(1)
        logits = logits.masked_fill(future, -torch.inf)
    if mask is not None:
        logits = logits + mask.double()
    probability = logits.softmax(dim=-1)
    if round_probabilities:
        probability = probability.to(torch.bfloat16).double()
    return (probability @ v).to(torch.bfloat16)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param((1, 1), {"l1_small_size": 131072}, id="1-card"),
        pytest.param((1, 2), {"l1_small_size": 131072, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="2-cards"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "case,head_group_size",
    [
        pytest.param("causal-gqa", 2, id="causal-gqa-two-head-groups"),
        pytest.param("masked-mha", 6, id="masked-mha-head-group-tail-padded-keys"),
        pytest.param("masked-gqa", 6, id="masked-gqa-per-head-mask"),
    ],
)
def test_accurate_attention(mesh_device, case, head_group_size, record_property):
    generator = torch.Generator().manual_seed(20261018)
    heads, key_heads = (8, 2) if case == "causal-gqa" else ((8, 8) if case == "masked-mha" else (12, 3))
    queries, keys = (128, 128) if case == "causal-gqa" else (64, 96)
    # Match the encoder's 256-wide heads and DiT's 128-wide heads.
    dimension = 256 if case == "causal-gqa" else 128
    value_dimension = dimension
    q = (torch.randn(1, heads, queries, dimension, generator=generator) * 0.5).to(torch.bfloat16)
    k = (torch.randn(1, key_heads, keys, dimension, generator=generator) * 0.5).to(torch.bfloat16)
    v = torch.randn(1, key_heads, keys, value_dimension, generator=generator).to(torch.bfloat16)
    mask = None
    if case != "causal-gqa":
        mask_heads = heads if case == "masked-gqa" else 1
        mask = torch.zeros(1, mask_heads, queries, keys, dtype=torch.float32)
        mask[..., 79:] = -torch.inf
        if mask_heads > 1:
            # Unequal per-head valid lengths reveal accidental head broadcasting.
            for head in range(heads):
                mask[:, head, :, 79 - head :] = -torch.inf

    def upload(value):
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
        return ttnn.from_torch(
            value,
            dtype=ttnn.float32 if value.dtype == torch.float32 else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    attention = AccurateAttention(mesh_device, compute, head_group_size=head_group_size)
    q_device, k_device, v_device = upload(q), upload(k), upload(v)
    mask_device = upload(mask) if mask is not None else None
    actual = attention(q_device, k_device, v_device, attn_mask=mask_device, is_causal=case == "causal-gqa")
    exact = attention_reference(q, k, v, mask=mask, causal=case == "causal-gqa")
    policy = attention_reference(q, k, v, mask=mask, causal=case == "causal-gqa", round_probabilities=True)
    hosts = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(actual)]
    assert len(hosts) == mesh_device.get_num_devices()
    for replica, host in enumerate(hosts):
        assert host.shape == exact.shape
        for reference_name, reference in [("dense_fp64", exact), ("bf16_probability_policy", policy)]:
            metrics = error_metrics(host, reference)
            for key in [
                "pcc",
                "relative_rms",
                "std_normalized_rmse",
                "std_normalized_rmse_upstream",
                "max_absolute_error",
            ]:
                record_property(f"replica_{replica}.{reference_name}.{key}", metrics[key])
            assert metrics["pcc"] >= 0.9999, metrics
            assert metrics["relative_rms"] <= 0.01, metrics

    if mask is not None:
        changed = v.clone()
        changed[..., 79:, :] = 16384  # Every head masks these keys.
        perturbed = attention(q_device, k_device, upload(changed), attn_mask=mask_device, is_causal=False)
        for original, tensor in zip(hosts, ttnn.get_device_tensors(perturbed)):
            assert torch.equal(original, ttnn.to_torch(tensor)), "Masked values influenced attention output"
    else:
        changed = v.clone()
        changed[..., keys // 2 :, :] = 16384
        perturbed = attention(q_device, k_device, upload(changed), is_causal=True)
        for original, tensor in zip(hosts, ttnn.get_device_tensors(perturbed)):
            assert torch.equal(
                original[..., : keys // 2, :], ttnn.to_torch(tensor)[..., : keys // 2, :]
            ), "Future values influenced the causal prefix"
