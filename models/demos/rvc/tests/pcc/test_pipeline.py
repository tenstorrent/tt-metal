# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest


def _to_mono_1d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x[:, 0]
    raise AssertionError(f"Unexpected audio rank: {x.ndim}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65384}], indirect=True)
def test_pipeline_real_file_usage(device) -> None:
    from models.demos.rvc.torch_impl.vc.pipeline import Pipeline as TorchPipeline
    from models.demos.rvc.tt_impl.vc.pipeline import Pipeline as TTPipeline

    torch_pipe = TorchPipeline(if_f0=True, version="v1", num="48k")
    tt_pipe = TTPipeline(tt_device=device, if_f0=True, version="v1", num="48k")

    torch_out = np.asarray(torch_pipe.infer())
    tt_out = np.asarray(tt_pipe.infer())

    # assert torch_out.size > 0
    # assert tt_out.size > 0

    # torch_mono = _to_mono_1d(torch_out.astype(np.float32))
    # tt_mono = _to_mono_1d(tt_out.astype(np.float32))

    # min_len = min(torch_mono.shape[0], tt_mono.shape[0])
    # assert min_len > 0

    # # Functional comparison for real-file smoke:
    # # lengths should be close and TT output should be non-silent.
    # assert abs(torch_mono.shape[0] - tt_mono.shape[0]) <= 2 * 160
    # assert np.max(np.abs(tt_mono)) > 0
