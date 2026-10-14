import torch, ttnn, pytest
import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/distributed_rms_norm")
from test_distributed_rms_norm import FABRIC_PARAMS, SYSTEM_MESH
from ttnn.operations.distributed_rms_norm import distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

ttnn.set_fabric_config(
    FABRIC_PARAMS["fabric_config"],
    FABRIC_PARAMS["reliability_mode"],
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    FABRIC_PARAMS["fabric_router_config"],
) if False else None
