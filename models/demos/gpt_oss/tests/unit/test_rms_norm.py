import os

import pytest
import torch
from transformers import AutoConfig

import ttnn


class ReferenceRMSNorm(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-05, device: torch.device | None = None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.randn(num_features, device=device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing"""
    path = os.getenv("HF_MODEL", "openai/gpt-oss-20b")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (4, 8)}.get(
            os.environ.get("MESH_DEVICE"), (1, ttnn.get_num_devices())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 32),
    ],
)
def test_rms_norm(
    mesh_device,
    device_params,
    mode,
    seq_len,
    hf_config,
):
    torch.manual_seed(0)
    breakpoint()

    print(hf_config.hidden_size)
    ref_rms_norm = ReferenceRMSNorm(hf_config.hidden_size)
