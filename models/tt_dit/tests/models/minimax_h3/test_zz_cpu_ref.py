# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device side of tools/cpu_reference_forward.py: run the full-depth real-checkpoint transformer on the reference's
inputs for one or more MINIMAX_H3_* knob sets, save each output, and print PSNR / PCC against the CPU reference.

All knob sets run in ONE process (one mesh open, the checkpoint read once): every fresh process start on the 4x8 risks
the pipeline start-up hang, and the knobs are read when a model is constructed, so each set builds its own model.

    H3_CPU_REF=~/cpu_ref/ref_5s.npz H3_TAG=tip pytest test_zz_cpu_ref.py -k 4x8 -s
    H3_CPU_REF=... H3_CPU_REF_CONFIGS="tip:;bf8:MINIMAX_H3_BF8_WEIGHTS=qkv,ff1;rec3:MINIMAX_H3_BF8_WEIGHTS=qkv,ff1|MINIMAX_H3_SDPA_PV_FIDELITY=LoFi" \
        pytest test_zz_cpu_ref.py -k 4x8 -s
(configs are separated by ';', a config is 'tag:ENV=VAL|ENV=VAL'; an empty knob list is the plain tip)
"""

import gc
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

_KEEP = ("MINIMAX_H3_MODEL_PATH",)


def _configs() -> list[tuple[str, dict[str, str]]]:
    spec = os.environ.get("H3_CPU_REF_CONFIGS")
    if not spec:
        return [(os.environ.get("H3_TAG", "tt"), {})]
    out = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        tag, _, knobs = item.partition(":")
        env = dict(kv.split("=", 1) for kv in knobs.split("|") if kv)
        out.append((tag.strip(), env))
    return out


def _set_knobs(base: dict[str, str], knobs: dict[str, str]) -> None:
    for k in [k for k in os.environ if k.startswith("MINIMAX_H3_") and k not in _KEEP]:
        del os.environ[k]
    os.environ.update({k: v for k, v in base.items() if k not in _KEEP or True})
    os.environ.update(knobs)


@pytest.mark.timeout(10800)
@GALAXY_RING
def test_cpu_ref_forward(mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, reset_seeds) -> None:
    ref_path = os.environ.get("H3_CPU_REF")
    if not ref_path:
        pytest.skip("set H3_CPU_REF to a tools/cpu_reference_forward.py npz")
    configs = _configs()
    base_env = {k: v for k, v in os.environ.items() if k.startswith("MINIMAX_H3_")}
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

    start = time.time()
    state_dict = _load_reference_state_dict(directory)  # once: ~62 GB of host RAM, shared by every knob set
    logger.info(f"checkpoint read from disk in {time.time() - start:.0f} s; {len(configs)} knob set(s): {[c[0] for c in configs]}")

    out_dir = Path(os.environ.get("H3_CPU_REF_OUT", Path(ref_path).parent))
    out_dir.mkdir(parents=True, exist_ok=True)

    def compose(t: ttnn.Tensor) -> np.ndarray:
        out = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape))
        )
        return out.reshape(-1, *out.shape[2:])[:1].float().numpy()

    results = []
    for tag, knobs in configs:
        _set_knobs(base_env, knobs)
        knob_str = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("MINIMAX_H3_") and k not in _KEEP) or "(tip)"
        tt_model = MiniMaxH3Transformer3DModel(
            **model_kwargs,
            mesh_device=mesh_device,
            ccl_manager=inputs.ccl_manager,
            parallel_config=inputs.parallel_config,
            is_fsdp=is_fsdp,
        )
        start = time.time()
        tt_model.load_torch_state_dict(dict(state_dict))
        apply_env_quant_config(tt_model)
        logger.info(f"[{tag}] weights on the mesh in {time.time() - start:.0f} s; knobs: {knob_str}")

        tt_model.prepare_static_sources(**inputs.tt_static)
        tt_model(**inputs.tt)  # compile pass
        ttnn.synchronize_device(mesh_device)
        start = time.time()
        tt_video, tt_audio = tt_model(**inputs.tt)
        ttnn.synchronize_device(mesh_device)
        warm = time.time() - start

        tt_video, tt_audio = compose(tt_video)[:, :num_video], compose(tt_audio)[:, :num_audio]
        out_path = out_dir / f"tt_{tag}.npz"
        np.savez(out_path, tt_video=tt_video, tt_audio=tt_audio, tag=tag, knobs=knob_str)
        row = (tag, warm, psnr(ref["ref_video"], tt_video), pcc(ref["ref_video"], tt_video), psnr(ref["ref_audio"], tt_audio), pcc(ref["ref_audio"], tt_audio))
        results.append(row)
        logger.info(
            f"[{tag}] warm forward {warm:.2f} s; vs CPU reference: video PSNR {row[2]:.2f} dB PCC {row[3]:.6f}; "
            f"audio PSNR {row[4]:.2f} dB PCC {row[5]:.6f}; wrote {out_path}"
        )

        tt_model.deallocate_weights()
        del tt_model
        gc.collect()
        ttnn.synchronize_device(mesh_device)

    _set_knobs(base_env, {})
    logger.info("SUMMARY vs CPU reference (5 s shape, full depth, fp32 CPU): tag | warm s | video dB | video PCC | audio dB | audio PCC")
    for tag, warm, vp, vc, ap, ac in results:
        logger.info(f"SUMMARY {tag:12s} | {warm:6.2f} | {vp:7.2f} | {vc:.6f} | {ap:7.2f} | {ac:.6f}")
