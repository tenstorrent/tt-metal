# Probe only: lift the Phase 0 gate in-process to see whether the existing height path is right.
import copy, torch, ttnn
import importlib, ttnn.operations.g6_interleaved_to_sharded

op = importlib.import_module("ttnn.operations.g6_interleaved_to_sharded.g6_interleaved_to_sharded")
from eval.golden_tests.g6_interleaved_to_sharded.feature_spec import FLUX_57, FLUX_58
from eval.golden_tests.g6_interleaved_to_sharded.helpers import (
    build_memory_config,
    check_output,
    make_torch_input,
    create_ttnn_input_tensor,
)

op.SUPPORTED["rank"] = [2, 4]
op.SUPPORTED["out_scheme"] = ["width_sharded", "height_sharded"]


def variant(base, shape=None, **out):
    s = copy.deepcopy(base)
    if shape:
        s["shape"] = list(shape)
    s["out_memory"].update(out)
    return s


cases = {
    "flux57": FLUX_57,
    "h_multi_2x1x33x128_4cores": variant(
        FLUX_58, (2, 1, 33, 128), scheme="height_sharded", grid=[[0, 0], [3, 0]], shard_shape=[32, 128]
    ),
    "h_multi_colmajor_2x3x33x128_4x2": variant(
        FLUX_58,
        (2, 3, 33, 128),
        scheme="height_sharded",
        grid=[[0, 0], [3, 1]],
        shard_shape=[64, 128],
        orientation="col_major",
    ),
    "h_partial_last_3x1x1x256": variant(
        FLUX_58, (3, 1, 1, 256), scheme="height_sharded", grid=[[0, 0], [1, 0]], shard_shape=[64, 256]
    ),
    "h_wide_shard_1x1x1x128_w256": variant(
        FLUX_58, (1, 1, 1, 128), scheme="height_sharded", grid=[[0, 0], [0, 0]], shard_shape=[32, 256]
    ),
    "w_rank4_1x1x1x5120": variant(FLUX_58, (1, 1, 1, 5120)),
    "w_rank4_2x1x33x256": variant(FLUX_58, (2, 1, 33, 256), grid=[[0, 0], [3, 0]], shard_shape=[128, 64]),
}
dev = ttnn.open_device(device_id=0)
try:
    for name, s in cases.items():
        torch.manual_seed(0)
        t = make_torch_input(s["shape"], s["dtype"])
        x = create_ttnn_input_tensor(
            t, dev, dtype=s["dtype"], layout=s["layout"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x = ttnn.fill_implicit_tile_padding(x, float("nan"), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mc = build_memory_config(s["out_memory"], dev)
        try:
            y = op.g6_interleaved_to_sharded(x, mc)
            check_output(y, t, shape=s["shape"], dtype=s["dtype"], layout=s["layout"], memory_config=mc)
            print("PASS", name, op.describe_plan(x, mc)["num_shards"], "shards")
        except Exception as e:
            print("FAIL", name, type(e).__name__, str(e)[:200])
finally:
    ttnn.close_device(dev)
