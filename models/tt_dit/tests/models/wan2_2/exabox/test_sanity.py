import pytest
import torch

import ttnn


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
    ],
    indirect=True,
    ids=["fabric_1d_ring"],
)
@pytest.mark.parametrize("mesh_device", [(4, 32)], indirect=True)
def test_open_4x32_mesh(
    mesh_device,
):
    print(mesh_device.shape)
    print(f"{torch.get_num_threads()=}")
    print(f"{torch.get_num_interop_threads()=}")
    import os

    print(f"{os.cpu_count()=}")
    # print(mesh_device.shape)

    # INSERT_YOUR_CODE
    # Determine MPI rank and size using environment variables
    import os

    rank = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("MPI_RANK") or os.environ.get("PMI_RANK")
    size = os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("MPI_SIZE") or os.environ.get("PMI_SIZE")
    if rank is not None and size is not None:
        print(f"MPI rank: {rank} / {size}")
    else:
        print("MPI rank or size environment variables not set, cannot determine MPI rank")

    import os as _os
    import subprocess

    try:
        print(
            "taskset:",
            subprocess.check_output(["taskset", "-cp", str(_os.getpid())]).decode(),
        )
    except Exception as e:
        print("taskset check failed:", e)
