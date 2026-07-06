# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""On-device residency gate for the coqui/XTTS-v2 pipeline (text -> speech).

This drives the SAME real forward the demo/e2e use — `tt/pipeline.py::forward_on_device`
— which runs the whole chain (speaker encode -> conditioning -> autoregressive GPT
decode with on-device sampling -> latents -> HiFi-GAN vocode) fully RESIDENT on the
device: every upload is `ttnn.as_tensor`, sampling + the token feed run on device, and
NO intermediate is copied back to host. The output waveform is a live device tensor.

It asserts residency structurally (device storage, expected shapes) WITHOUT reading
anything back to host — so the forward's op stream contains no host-transfer op
(from_torch / to_torch / from_device / to_device). The numeric correctness of this
exact forward (PCC vs the HF reference) is gated separately by `test_e2e_tts.py`; here
we only prove the forward is trace-capturable / everything-on-device.

Named to sort first in tests/e2e so the emit-e2e on-device op-stream probe targets this
host-free forward (the reference/PCC test necessarily copies back for comparison).
"""

from __future__ import annotations

import importlib.util as ilu
import os

import pytest
import torch

import ttnn
from models.demos.xtts_v2.tt import pipeline as P

HF_MODEL_ID = "coqui/XTTS-v2"
# small horizon: the probe caps decode via TT_PERF_MAX_NEW_TOKENS; default stays quick.
_N = int(os.environ.get("XTTS_FWD_N", os.environ.get("TT_PERF_MAX_NEW_TOKENS", "6")))


def _load_reference():
    here = os.path.dirname(os.path.abspath(__file__))
    rl = os.path.normpath(os.path.join(here, "..", "pcc", "_reference_loader.py"))
    spec = ilu.spec_from_file_location("_reference_loader", rl)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_reference_model(HF_MODEL_ID)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forward_on_device(device):
    torch.manual_seed(0)
    model = _load_reference()

    # Run the real forward, fully resident. Returns device tensors only.
    fo = P.forward_on_device(device, model, text="hello world.", language="en", N=_N)

    wav = fo["waveform"]
    codes = fo["codes"]
    lat = fo["latents"]

    # residency: every pipeline output lives ON DEVICE (checked via metadata only —
    # NO ttnn.to_torch / from_device, so the forward's op stream stays host-free).
    for name, t in [("waveform", wav), ("codes", codes), ("latents", lat),
                    ("g", fo["g"]), ("cond_lat", fo["cond_lat"])]:
        assert isinstance(t, ttnn.Tensor), f"{name} is not a ttnn.Tensor"
        assert t.storage_type() == ttnn.StorageType.DEVICE, f"{name} is not resident on device"

    # shapes: the AR decode produced exactly N codes; latents are [1, N, model_dim];
    # the vocoder produced a non-trivial waveform.
    assert list(codes.shape) == [1, _N], f"codes shape {list(codes.shape)} != [1, {_N}]"
    assert int(lat.shape[0]) == 1 and int(lat.shape[1]) == _N, f"latents shape {list(lat.shape)}"
    assert int(lat.shape[-1]) == int(model.gpt.model_dim)
    assert len(wav.shape) == 3 and int(wav.shape[0]) == 1, f"waveform shape {list(wav.shape)}"
    assert int(wav.shape[1]) > _N, "waveform time axis is degenerate"

    print(f"on-device forward OK: N={_N} codes{list(codes.shape)} "
          f"latents{list(lat.shape)} waveform{list(wav.shape)}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__ + "::test_forward_on_device", "-svv"]))
