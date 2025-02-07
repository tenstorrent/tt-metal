import torch
import pytest
import ttnn
from models.utility_functions import comp_pcc
from models.common.rmsnorm import RMSNorm as TtRMSNorm


class Phi3RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Phi3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def test_phi3_rms_norm(mesh_device, use_program_cache, reset_seeds):
    hidden_size = 3072  # Taken from config

    ref_model = Phi3RMSNorm(hidden_size)
    input_tensor = torch.randn(1, 1, 32, hidden_size)
    ref_output = ref_model(input_tensor)

    ttnn_input = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Dummy weights
    state_dict = ref_model.state_dict()
    # Rename the keys to match the TtRMSNorm
    state_dict = {f"rmsnorm.{k}": v for k, v in state_dict.items()}

    ttnn_model = TtRMSNorm(
        device=mesh_device,
        dim=hidden_size,
        state_dict=state_dict,
        weight_key="rmsnorm",
    )

    ttnn_output = ttnn_model(ttnn_input, mode="decode")
    ttnn_output = ttnn.to_torch(ttnn_output)

    passing, pcc_message = comp_pcc(ref_output, ttnn_output, 0.99)
    print(f"PCC: {pcc_message}")
    assert passing, f"RMSNorm output does not meet PCC requirement {0.99}."
