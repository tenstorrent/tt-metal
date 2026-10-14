import ttnn, pytest
from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
from ttnn.operations.ccl import Topology
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2)) if False else None
