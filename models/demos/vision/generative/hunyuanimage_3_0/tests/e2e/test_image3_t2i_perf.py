# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full end-to-end text->image PERF test for HunyuanImage-3.0 (hybrid).

Measures the TOTAL LATENCY of a complete prompt->image run at FULL model depth
(32 layers): the diffusion loop (all steps x CFG=2, transformer on TT) + VAE
decode + postprocess, excluding the one-time cold model load. Emits the parseable
headline

    E2E_T2I_TOTAL_LATENCY_S=<seconds>

(the real s/image wall-clock, replacing the earlier projection), plus the
loop/step/VAE breakdown. This is the number that ships.

Correctness is gated separately (test_image3_t2i_e2e_pcc.py + per-block PCC); this
test additionally sanity-checks the output (finite, non-degenerate) so a broken
render can't post a fast "win".

No hard latency gate by default (no measured baseline yet -- seed one from the
first green run, per the manual-track convention). Set HUNYUAN_T2I_MAX_LATENCY_S
to assert a ceiling once a baseline exists.

Run:  ./python_env/bin/python -m pytest \
        models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_image3_t2i_perf.py -s
Env:  HUNYUAN_T2I_NUM_LAYERS (32), HUNYUAN_T2I_STEPS (default = generation_config diff_infer_steps=50),
      HUNYUAN_T2I_SIZE (1024,1024), HUNYUAN_T2I_MAX_LATENCY_S (unset = report only), HUNYUAN_T2I_OUT
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0.tt import gen_image as gi

try:
    _MESH = tuple(int(x) for x in ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
except Exception:
    _MESH = (1, 8)

PROMPT = os.environ.get("HUNYUAN_T2I_PROMPT", "A serene mountain lake at sunrise, photorealistic, ultra detailed.")
NUM_LAYERS = int(os.environ.get("HUNYUAN_T2I_NUM_LAYERS", "32"))
STEPS = os.environ.get("HUNYUAN_T2I_STEPS")  # None => generation_config default (50)
STEPS = int(STEPS) if STEPS else None
_SZ = os.environ.get("HUNYUAN_T2I_SIZE", "1024,1024")
IMAGE_SIZE = tuple(int(x) for x in _SZ.replace("x", ",").split(","))
OUT = os.environ.get("HUNYUAN_T2I_OUT", "hunyuan_t2i_perf.png")
MAX_LATENCY_S = os.environ.get("HUNYUAN_T2I_MAX_LATENCY_S")
MAX_LATENCY_S = float(MAX_LATENCY_S) if MAX_LATENCY_S else None
USE_TRACE = os.environ.get("HUNYUAN_T2I_TRACE", "1") != "0"  # host-free traced replay (the ~10x lever)


@pytest.mark.parametrize(
    "device_params",
    # trace_region_size must hold the captured 32-layer forward (traced replay).
    [{"l1_small_size": 24576, "trace_region_size": 200000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_t2i_e2e_perf(device_params, mesh_device):
    torch.manual_seed(0)
    model, tt_pipe, _uninstall = gi.build_tt_backed_model(mesh_device, num_layers=NUM_LAYERS, use_trace=USE_TRACE)
    print(f"\nt2i e2e PERF: num_layers={NUM_LAYERS} use_trace={USE_TRACE}")
    img, timing = gi.generate_image(
        model,
        tt_pipe,
        prompt=PROMPT,
        image_size=IMAGE_SIZE,
        num_inference_steps=STEPS,
        seed=0,
        out_path=OUT,
    )

    # Headline already printed by generate_image as E2E_T2I_TOTAL_LATENCY_S=...
    print(
        f"\nt2i e2e PERF: total_latency_s={timing['total_latency_s']:.3f} "
        f"(loop {timing['loop_s']:.1f}s @ {timing['ms_per_step_mean']:.0f} ms/step x{timing['steps']}, "
        f"vae {timing['vae_decode_s']:.1f}s) num_layers={NUM_LAYERS} token_hw={timing['token_hw']} out={timing['out_path']}"
    )

    # tracy-free stage profile (HUNYUAN_STAGE_PROFILE=1): attn vs MoE vs other.
    from models.demos.vision.generative.hunyuanimage_3_0._stubs import image3_decoder_layer as _dl

    sp = _dl._STAGE_PROF
    if sp["on"] and sp["layers"]:
        n = sp["layers"]
        other = timing["loop_s"] * 1000.0 - sp["attn_ms"] - sp["moe_ms"]
        print(
            f"STAGE_PROFILE: attn={sp['attn_ms']:.0f}ms moe={sp['moe_ms']:.0f}ms "
            f"other(loop-attn-moe,incl.sync-inflation)={other:.0f}ms over {n} layer-calls | "
            f"per-layer attn={sp['attn_ms'] / n:.1f}ms moe={sp['moe_ms'] / n:.1f}ms "
            f"(attn:moe = {sp['attn_ms'] / max(sp['moe_ms'], 1e-9):.2f}:1)"
        )

    # Sanity: a broken render must not post a fast time.
    assert timing["total_latency_s"] > 0.0
    assert img.size[0] > 0 and img.size[1] > 0
    ext = torch.tensor(img.getextrema()).float()
    assert ext.std() > 0.0, "rendered image is degenerate (flat) — perf number is meaningless"

    if MAX_LATENCY_S is not None:
        assert (
            timing["total_latency_s"] <= MAX_LATENCY_S
        ), f"t2i e2e latency {timing['total_latency_s']:.1f}s > ceiling {MAX_LATENCY_S:.1f}s"
