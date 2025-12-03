import pytest
import ttnn
import numpy as np
from diffusers.utils import export_to_video


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 32)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT}],
    indirect=True,
)
def test_pipeline_inference_exabox(
    mesh_device,
):
    import socket

    hostname = socket.gethostname()
    video_filename = f"{hostname}_crash_exabox.mp4"
    frames = np.zeros((81, 720, 1280, 3))
    export_to_video(frames, video_filename, fps=16)
    return
