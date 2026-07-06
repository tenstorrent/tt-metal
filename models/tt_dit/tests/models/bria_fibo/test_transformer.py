# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _load_ref_transformer(dtype=torch.bfloat16):
    try:
        from diffusers import BriaFiboTransformer2DModel
    except Exception:
        from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel
    try:
        # When running offline, resolve the HF repo ID to its local cache path.
        fibo_path = FIBO_PATH
        if not os.path.isdir(fibo_path):
            from huggingface_hub import snapshot_download

            fibo_path = snapshot_download(fibo_path, allow_patterns=["transformer/*"], local_files_only=True)
        return BriaFiboTransformer2DModel.from_pretrained(fibo_path, subfolder="transformer", torch_dtype=dtype).eval()
    except Exception as e:
        pytest.skip(f"FIBO transformer unavailable: {e}")


def test_fibo_transformer_reference_config():
    m = _load_ref_transformer()
    c = m.config
    assert c.num_layers == 8 and c.num_single_layers == 38
    assert c.num_attention_heads == 24 and c.attention_head_dim == 128
    assert c.in_channels == 48 and c.joint_attention_dim == 4096
    assert c.axes_dims_rope == [16, 56, 56]
    assert len(m.caption_projection) == c.num_layers + c.num_single_layers  # 46


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_fibo_text_projection(*, mesh_device):
    from models.tt_dit.models.transformers.transformer_bria_fibo import BriaFiboTextProjection
    from models.tt_dit.utils import tensor as tt_tensor
    from models.tt_dit.utils.check import assert_quality

    m = _load_ref_transformer()
    ref = m.caption_projection[0]  # HF BriaFiboTextProjection
    torch.manual_seed(0)
    x = torch.randn(1, 64, 2048)
    with torch.no_grad():
        r = ref(x.to(torch.bfloat16))
    tt = BriaFiboTextProjection(in_features=2048, hidden_size=1536, mesh_device=mesh_device)
    tt.load_torch_state_dict(ref.state_dict())
    out = tt.forward(tt_tensor.from_torch(x.to(torch.bfloat16), device=mesh_device))
    assert tuple(tt_tensor.to_torch(out).shape)[-1] == 1536
    assert_quality(r, tt_tensor.to_torch(out), pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_fibo_timestep_embed(*, mesh_device):
    from models.tt_dit.models.transformers.transformer_bria_fibo import BriaFiboTimestepEmbed
    from models.tt_dit.utils import tensor as tt_tensor
    from models.tt_dit.utils.check import assert_quality

    m = _load_ref_transformer()
    ref = m.time_embed  # HF BriaFiboTimestepProjEmbeddings
    inner_dim = m.config.num_attention_heads * m.config.attention_head_dim  # 3072
    torch.manual_seed(0)
    timestep = torch.tensor([500.0, 250.0])
    with torch.no_grad():
        r = ref(timestep, dtype=torch.bfloat16)
    tt = BriaFiboTimestepEmbed(inner_dim=inner_dim, mesh_device=mesh_device)
    tt.load_torch_state_dict(ref.state_dict())
    # Pass timestep as [batch, 1] bfloat16 tensor on device
    tt_timestep = tt_tensor.from_torch(timestep.unsqueeze(-1).to(torch.bfloat16), device=mesh_device)
    out = tt.forward(tt_timestep)
    out_torch = tt_tensor.to_torch(out)
    assert tuple(out_torch.shape)[-1] == inner_dim
    assert_quality(r, out_torch, pcc=0.99)
