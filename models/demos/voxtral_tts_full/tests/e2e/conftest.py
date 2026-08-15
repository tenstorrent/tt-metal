# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Module-scoped device / model / pipeline for the e2e suite.

The pipeline stages ~9 GB of weights and the HF reference is a 16 GB fp32 load, so both are built
ONCE for the whole module.  The device is opened here rather than through the repo's
function-scoped `device` fixture for the same reason -- and because the trace selftest needs a
`trace_region_size`, which the module-scoped path of that fixture cannot parametrise.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from models.demos.voxtral_tts_full.tt import pipeline as P
from models.demos.voxtral_tts_full.tt import reference as ref


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, trace_region_size=P.DEFAULT_TRACE_REGION_SIZE)
    try:
        yield dev
    finally:
        ttnn.close_device(dev)


@pytest.fixture(scope="module")
def hf_model():
    return ref.load_hf_model(dtype=torch.float32)


@pytest.fixture(scope="module")
def inputs():
    return ref.encode_inputs()


@pytest.fixture(scope="module")
def pipe(device, hf_model):
    return P.build_pipeline(device, model=hf_model)


@pytest.fixture(scope="module")
def horizon():
    """The frame budget for the gate: the horizon the bring-up capture itself ran at
    (`_captured/codec_decoder/args.pt` is [T, 37]).  Both sides use it, and both apply the same
    [END_AUDIO] stop rule inside it."""
    return P.captured_frame_count()


@pytest.fixture(scope="module")
def golden(hf_model, inputs, horizon):
    return ref.cached_reference_tts(inputs, max_frames=horizon, model=hf_model, verbose=False)


@pytest.fixture(scope="module")
def tt_run(pipe, inputs, horizon):
    """ONE run of the shared pipeline; every gate is asserted against this same run."""
    return pipe.run_tts(inputs, max_frames=horizon, verbose=True)
