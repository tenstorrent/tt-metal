import ttnn
from tests.sweep_framework.sweeps.ccl.generality.reduce_scatter import (
    run as run_reduce_scatter_test,
    LEAD_MODEL_SHARD_SPECS,
)


# shard spec in this example on is:

# From DeepSeek on Galaxy

# input_shape=(32, 128),
# input_cores=(7, 8),
# input_strategy="w",
# output_shape=(32,112[128]),
# output_cores=(4,4),
# output_strategy="w"


PARAMETERS = {
    "mesh_shape": (2, 4),
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "input_shape": [1, 1, 32, 7168],
    "dim": 3,
    "cluster_axis": 1,
    "num_links": 1,
    "input_dtype": ttnn.bfloat16,
    "layout": ttnn.TILE_LAYOUT,
    "buffer_type": ttnn.BufferType.L1,
    "shard_specs": LEAD_MODEL_SHARD_SPECS[4],
    "num_iters": 1,
    "topology": ttnn.Topology.Linear,
    "device": None,
}


def test_reduce_scatter():
    run_reduce_scatter_test(**PARAMETERS)
