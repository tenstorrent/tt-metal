import ttnn
from tests.sweep_framework.sweeps.ccl.generality.all_reduce import (
    run as run_all_reduce_test,
    LEAD_MODEL_SHARD_SPECS,
)


# shard spec in this example on is:

# From DeepSeek on Galaxy

# input_shape=(32, 64),
# input_cores=(4, 6),
# input_strategy="w",
# output_shape=(32,128),
# output_cores=(2,5),
# output_strategy="w"


PARAMETERS = {
    "mesh_shape": (2, 4),
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "num_links": 1,
    "input_shape": [1, 1, 32, 1280],
    "cluster_axis": 0,
    "input_dtype": ttnn.bfloat16,
    "layout": ttnn.TILE_LAYOUT,
    "buffer_type": ttnn.BufferType.L1,
    # "shard_specs": LEAD_MODEL_SHARD_SPECS[0], # sharded is broken
    "shard_specs": None,  # and not sharded is also broken
    "num_iters": 1,
    "topology": ttnn.Topology.Linear,
    "math_op": ttnn.ReduceType.Sum,
    "device": None,
}


def test_all_reduce():
    assert run_all_reduce_test(**PARAMETERS)[0]
