# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DRAM footprint measurement tests for each Z-Image-Turbo component."""

import gc

import pytest
import torch

import ttnn

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CAP_TOKENS = 128
IMG_LATENT_H = 64
IMG_LATENT_W = 64
LATENT_CHANNELS = 16

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _dram_alloc_mb(mesh_device):
    ttnn.synchronize_device(mesh_device)
    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return v.num_banks * v.total_bytes_allocated_per_bank / 2**20


def _to_device_bf16(pt, mesh_device):
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_device_int32(pt, mesh_device):
    return ttnn.from_torch(
        pt.to(torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.fixture(scope="function")
def device_params(request):
    return {"l1_small_size": 1 << 15, "trace_region_size": 50_000_000}


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_measure_text_encoder_dram(mesh_device):
    mesh_device.enable_program_cache()
    from transformers import AutoTokenizer

    from models.demos.z_image_turbo.tt.text_encoder.model_ttnn import TextEncoderTTNN

    before = _dram_alloc_mb(mesh_device)

    te = TextEncoderTTNN(mesh_device)
    after_load = _dram_alloc_mb(mesh_device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    messages = [{"role": "user", "content": "a beautiful sunset over the ocean"}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted, padding="max_length", truncation=True, max_length=CAP_TOKENS, return_tensors="pt")[
        "input_ids"
    ]
    tt_input_ids = _to_device_int32(input_ids, mesh_device)

    tt_out = te(tt_input_ids)
    after_fwd = _dram_alloc_mb(mesh_device)

    ttnn.deallocate(tt_input_ids, True)
    ttnn.deallocate(tt_out, True)
    del te
    gc.collect()

    print(
        f"\nTE DRAM: weights={after_load:.1f} MB, after_fwd={after_fwd:.1f} MB, overhead={after_fwd - after_load:.1f} MB"
    )


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_measure_dit_dram(mesh_device):
    mesh_device.enable_program_cache()
    from models.demos.z_image_turbo.tt.dit.model_ttnn import ZImageTransformerTTNN

    before = _dram_alloc_mb(mesh_device)

    tr = ZImageTransformerTTNN(mesh_device)
    after_load = _dram_alloc_mb(mesh_device)

    cap_cpu = torch.randn(1, CAP_TOKENS, 2560).bfloat16()
    tr.set_cap_feats(cap_cpu)

    torch.manual_seed(42)
    latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
    lat_pt = latents.squeeze(0).unsqueeze(1).bfloat16()
    ts_pt = torch.tensor([0.5], dtype=torch.bfloat16)

    tt_lat = _to_device_bf16(lat_pt, mesh_device)
    tt_ts = _to_device_bf16(ts_pt, mesh_device)

    tt_out = tr._forward_impl([tt_lat], tt_ts)
    after_fwd = _dram_alloc_mb(mesh_device)

    for t in tt_out:
        ttnn.deallocate(t, True)
    ttnn.deallocate(tt_lat, True)
    ttnn.deallocate(tt_ts, True)
    del tr
    gc.collect()

    print(
        f"\nDIT DRAM: weights={after_load:.1f} MB, after_fwd={after_fwd:.1f} MB, overhead={after_fwd - after_load:.1f} MB"
    )


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_measure_vae_dram(mesh_device):
    mesh_device.enable_program_cache()
    from models.demos.z_image_turbo.tt.vae.model_ttnn import VaeDecoderTTNN

    before = _dram_alloc_mb(mesh_device)

    vae = VaeDecoderTTNN(mesh_device)
    after_load = _dram_alloc_mb(mesh_device)

    latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
    image_tensor = vae(latents)
    after_fwd = _dram_alloc_mb(mesh_device)

    del vae
    gc.collect()

    print(
        f"\nVAE DRAM: weights={after_load:.1f} MB, after_fwd={after_fwd:.1f} MB, overhead={after_fwd - after_load:.1f} MB"
    )
