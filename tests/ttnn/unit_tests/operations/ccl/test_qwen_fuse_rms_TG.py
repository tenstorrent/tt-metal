import ttnn
import os
from tracy import signpost

from models.common.utility_functions import skip_for_blackhole, skip_for_wormhole_b0


from conftest import is_6u
from models.demos.llama3_70b_galaxy.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

from tests.ttnn.unit_tests.operations.ccl.fusion_subtests.rms_test import (
    run_rms_trace,
    run_rms_fuse_impl,
)

from tests.ttnn.unit_tests.operations.ccl.fusion_subtests.concat_fuse_test import (
    run_concat_fuse_impl,
)

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from conftest import is_6u


# Enumerate the post-commit cases explicitly
@skip_for_blackhole("This is a wormhole test")
@pytest.mark.skipif(is_6u(), reason="This test is not for 6U devices")
@pytest.mark.parametrize(
    "num_devices, elements_per_batch, input_shard_grid, output_shard_grid",
    [
        # RMS NORM ALL GATHER FUSION No Reshard
        (
            4,
            5120,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1))}),
            None,
        ),
        (
            4,
            5120,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1))}),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in PREFETCHER_NOC1_GRID
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize("fused_add", [True, False])
@pytest.mark.parametrize("use_noc1_only", [True, False])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("residual_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_rms_fuse(
    mesh_device,
    num_devices,
    elements_per_batch,
    num_links,
    num_iters,
    function_level_defaults,
    input_shard_grid,
    output_shard_grid,
    fused_add,
    use_noc1_only,
    input_dtype,
    residual_dtype,
    output_dtype,
    topology,
):
    run_rms_fuse_impl(
        mesh_device,
        num_devices,
        elements_per_batch,
        num_links,
        function_level_defaults,
        input_shard_grid,
        output_shard_grid,
        topology,
        fused_add,
        use_noc1_only=use_noc1_only,
        output_dtype=output_dtype,
        num_iters=num_iters,
        input_dtype=input_dtype,
        residual_dtype=residual_dtype,
    )
