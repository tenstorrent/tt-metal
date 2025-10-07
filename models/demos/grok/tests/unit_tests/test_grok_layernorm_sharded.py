import pytest
import torch

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.grok.reference.llama_clone import RMSNorm as RefRMSNorm
from models.demos.grok.tt.ccl import CCL_Manager
from models.demos.grok.tt.distributed_norm import DistributedNorm
from models.demos.grok.tt.model_config import TtModelArgs


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_grok_layernorm(mesh_device):
    tt_ccl = CCL_Manager(mesh_device)
    model_args = TtModelArgs(mesh_device)

    torch_gamma = torch.randn(8192) - 1

    torch_input = torch.randn(1, 1, 32, 8192) + 10

    torch_layernorm = RefRMSNorm(8192, 1e-5)
    torch_layernorm.weight = torch.nn.Parameter(torch_gamma)
    torch_layernorm_output = torch_layernorm(torch_input)

    tt_input = model_args.prepare_residual_tensor_decode(
        torch_input.squeeze(0).transpose(1, 0),
        model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        force_replicated=False,
    )
    norm_op = DistributedNorm(
        RMSNorm(
            device=mesh_device,
            dim=8192,
            eps=1e-5,
            state_dict={"norm.weight": torch_gamma},
            weight_dtype=ttnn.bfloat16,
            weight_key="norm",
            is_distributed=True,
            ccl_topology=ttnn.Topology.Ring,
            tt_ccl=tt_ccl,
        ),
        model_args,
        tt_ccl=tt_ccl,
    )

    tt_output = norm_op(tt_input, mode="decode")

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=(8, 4))
    )[:1, :, :, :]
    # comp_pcc(tt_output_torch, torch_layernorm_output)
    breakpoint()
