# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import numpy as np
import pytest


def _require_real_file_env() -> Path:
    input_path = os.getenv("RVC_TEST_INPUT")
    if not input_path:
        pytest.skip("RVC_TEST_INPUT is not set; real-file pipeline test requires an existing audio file.")
    path = Path(input_path)
    if not path.exists():
        pytest.skip(f"RVC_TEST_INPUT does not exist: {path}")

    for env_key in ("RVC_CONFIGS_DIR", "RVC_ASSETS_DIR"):
        env_val = os.getenv(env_key)
        if not env_val:
            pytest.skip(f"{env_key} is not set; real-file pipeline test requires model/config assets.")
        if not Path(env_val).exists():
            pytest.skip(f"{env_key} path does not exist: {env_val}")
    return path


def _to_mono_1d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x[:, 0]
    raise AssertionError(f"Unexpected audio rank: {x.ndim}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65384}], indirect=True)
def test_pipeline_real_file_usage(device) -> None:
    input_path = _require_real_file_env()
    from models.demos.rvc.torch_impl.vc.pipeline import Pipeline as TorchPipeline
    from models.demos.rvc.tt_impl.vc.pipeline import Pipeline as TTPipeline

    torch_pipe = TorchPipeline(if_f0=True, version="v1", num="48k")
    tt_pipe = TTPipeline(tt_device=device, if_f0=True, version="v1", num="48k")

    # Keep infer invocation aligned with models/demos/rvc/scripts/infer.py.
    common_kwargs = dict(
        speaker_id=0,
        f0_up_key=0,
        f0_method="pm",
        index_rate=0.75,
        resample_sr=0,
        rms_mix_rate=0.25,
        protect=0.33,
    )

    torch_out = np.asarray(torch_pipe.infer(str(input_path), **common_kwargs))
    tt_out = np.asarray(tt_pipe.infer(str(input_path), **common_kwargs))

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
