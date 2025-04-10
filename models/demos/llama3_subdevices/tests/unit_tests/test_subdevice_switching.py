# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.demos.llama3_subdevices.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs


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
    model_args = TtModelArgs(mesh_device, max_batch_size=1, max_seq_len=1024, dummy_weights=True)

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, mode="prefill")
    tt_ccl.close()
    del prefetcher_setup

    print("Switching to decode")

    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="decode")
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id)
    tt_ccl.close()
    del prefetcher_setup

    print("Switching to prefill")

    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, mode="prefill")
    tt_ccl.close()
    del prefetcher_setup
