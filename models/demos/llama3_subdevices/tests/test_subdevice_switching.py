import pytest
import ttnn
from tests.ttnn.unit_tests.operations.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL


@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_switching(mesh_device):
    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    tt_ccl = TT_CCL(mesh_device, crs, prefetcher_setup.worker_sub_device_id, mode="prefill")
    tt_ccl.close()
    del prefetcher_setup

    print("Switching to decode")

    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="decode")
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    tt_ccl = TT_CCL(mesh_device, sub_core_grids, prefetcher_setup.worker_sub_device_id)
    tt_ccl.close()
    del prefetcher_setup

    print("Switching to prefill")

    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    tt_ccl = TT_CCL(mesh_device, crs, prefetcher_setup.worker_sub_device_id, mode="prefill")
    tt_ccl.close()
    del prefetcher_setup
