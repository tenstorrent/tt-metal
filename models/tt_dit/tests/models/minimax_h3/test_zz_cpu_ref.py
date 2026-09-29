# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device side of tools/cpu_reference_forward.py: run the full-depth real-checkpoint transformer on the reference's
inputs with whatever MINIMAX_H3_* knobs are set, save the outputs, and print PSNR / PCC against the CPU reference.

    H3_CPU_REF=~/cpu_ref/ref_5s.npz H3_TAG=tip pytest test_zz_cpu_ref.py -k 4x8 -s
    MINIMAX_H3_FAST=1 MINIMAX_H3_SDPA_FIXED_SOFTMAX_BLOCKS=auto H3_CPU_REF=... H3_TAG=rec3 pytest ... -k 4x8 -s
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.quant_config import apply_env_quant_config
from ....models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ....utils.test import skip_if_unsupported_num_links
from .common import GALAXY_RING
from .test_transformer_minimax_h3 import (
    _CALLER_OWNED_CONFIG_KEYS,
    _checkpoint_dir,
    _load_reference_state_dict,
    _modality_metadata,
    _prepare_tt_inputs,
)
from .tools.cpu_reference_forward import pcc, psnr


@pytest.mark.timeout(5400)
@GALAXY_RING
def test_cpu_ref_forward(mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, reset_seeds) -> None:
    ref_path = os.environ.get("H3_CPU_REF")
    if not ref_path:
        pytest.skip("set H3_CPU_REF to a tools/cpu_reference_forward.py npz")
    tag = os.environ.get("H3_TAG", "tt")
    ref = np.load(ref_path)
    skip_if_unsupported_num_links(mesh_device, num_links)
    directory = _checkpoint_dir()

    config = {k: v for k, v in json.loads((directory / "config.json").read_text()).items() if not k.startswith("_")}
    model_kwargs = {k: v for k, v in config.items() if k not in _CALLER_OWNED_CONFIG_KEYS}
    model_kwargs["patch_size"] = tuple(model_kwargs["patch_size"])
    num_text, num_audio, num_video = int(ref["num_text"]), int(ref["num_audio"]), int(ref["num_video"])
    grid = tuple(int(g) for g in ref["grid"])
    per_modality = _modality_metadata(num_text, num_audio, num_video, grid, ())
    host_inputs = {k: torch.from_numpy(ref[k]) for k in ("video_input", "audio_input", "prompt_input", "timestep")}

    inputs = _prepare_tt_inputs(
        mesh_device,
        sp_axis,
        tp_axis,
        num_links,
        topology,
        per_modality,
        text_dim=model_kwargs["text_dim"],
        video_patch_dim=model_kwargs["in_channels"] * int(torch.tensor(model_kwargs["patch_size"]).prod()),
        audio_channels=model_kwargs["audio_in_channels"],
        head_dim=model_kwargs["attention_head_dim"],
        rope_freq_dim=config["rope_freq_dim"],
        rope_theta=config["rope_theta"],
        host_inputs=host_inputs,
    )
    # the same metadata as the reference run, or the comparison is meaningless
    assert np.array_equal(inputs.position_ids.numpy(), ref["position_ids"]) and np.array_equal(inputs.tags.numpy(), ref["tags"])

    tt_model = MiniMaxH3Transformer3DModel(
        **model_kwargs,
        mesh_device=mesh_device,
        ccl_manager=inputs.ccl_manager,
        parallel_config=inputs.parallel_config,
        is_fsdp=is_fsdp,
    )
    start = time.time()
    state_dict = _load_reference_state_dict(directory)
    tt_model.load_torch_state_dict(state_dict)
    del state_dict
    apply_env_quant_config(tt_model)
    logger.info(f"[{tag}] checkpoint on the mesh in {time.time() - start:.0f} s; knobs: " + " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("MINIMAX_H3_") and k != "MINIMAX_H3_MODEL_PATH"))

    tt_model.prepare_static_sources(**inputs.tt_static)
    tt_model(**inputs.tt)  # compile pass
    ttnn.synchronize_device(mesh_device)
    start = time.time()
    tt_video, tt_audio = tt_model(**inputs.tt)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[{tag}] warm forward {time.time() - start:.2f} s")

    def compose(t: ttnn.Tensor) -> np.ndarray:
        out = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape))
        )
        return out.reshape(-1, *out.shape[2:])[:1].float().numpy()

    tt_video, tt_audio = compose(tt_video)[:, :num_video], compose(tt_audio)[:, :num_audio]
    out_dir = Path(os.environ.get("H3_CPU_REF_OUT", Path(ref_path).parent))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tt_{tag}.npz"
    np.savez(out_path, tt_video=tt_video, tt_audio=tt_audio, tag=tag)
    if "ref_video" in ref:
        logger.info(
            f"[{tag}] vs CPU reference: video PSNR {psnr(ref['ref_video'], tt_video):.2f} dB PCC {pcc(ref['ref_video'], tt_video):.6f}; "
            f"audio PSNR {psnr(ref['ref_audio'], tt_audio):.2f} dB PCC {pcc(ref['ref_audio'], tt_audio):.6f}"
        )
    logger.info(f"[{tag}] wrote {out_path}")
