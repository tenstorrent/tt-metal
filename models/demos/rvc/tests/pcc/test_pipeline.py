# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from models.demos.rvc.runner.performant_runner import RVCInferenceConfig, RVCRunner


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 65384, "trace_region_size": 679936, "num_command_queues": 2}], indirect=True
)
def test_pipeline_real_file_usage(device) -> None:
    from models.demos.rvc.torch_impl.vc.pipeline import Pipeline as TorchPipeline

    torch_pipe = TorchPipeline(if_f0=True, version="v1", num="48k", validation=True)
    runner = RVCRunner()
    runner.initialize_inference(device, {"inference": RVCInferenceConfig(num_secs=30.0)}, validation=True)

    audio_input = runner.ttnn_pipeline.prepare_audio_input()
    torch_out = np.asarray(torch_pipe.run(audio_input))
    tt_out = np.asarray(runner.run(audio_input))

    assert torch_out.size > 0
    assert tt_out.size > 0

    torch_mono = torch_out.astype(np.float32)
    tt_mono = tt_out.astype(np.float32)

    min_len = min(torch_mono.shape[0], tt_mono.shape[0])
    assert min_len > 0

    # Functional comparison for real-file smoke:
    # lengths should be close and TT output should be non-silent.
    assert abs(torch_mono.shape[0] - tt_mono.shape[0]) <= 2 * 160
    assert np.max(np.abs(tt_mono)) > 0
