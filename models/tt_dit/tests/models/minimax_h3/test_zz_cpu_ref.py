# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device side of tools/cpu_reference_forward.py: run the full-depth real-checkpoint transformer on the reference's
inputs for one or more MINIMAX_H3_* knob sets, save each output, and print PSNR / PCC against the CPU reference.

One process, one model: the checkpoint goes onto the mesh once (~9 min), then every knob set is applied in place (the
SDPA knobs are attention attributes plus environment read when the program config is built) and scored. A set with
MINIMAX_H3_BF8_WEIGHTS typecasts the weights on the device, which cannot be undone, so list such sets last.

    H3_CPU_REF=~/cpu_ref/ref_5s.npz H3_TAG=tip pytest test_zz_cpu_ref.py -k 4x8 -s
    H3_CPU_REF=... H3_CPU_REF_CONFIGS="tip:;lofipv:MINIMAX_H3_SDPA_PV_FIDELITY=LoFi;bf8:MINIMAX_H3_BF8_WEIGHTS=qkv,ff1" \
        pytest test_zz_cpu_ref.py -k 4x8 -s
(configs are separated by ';', a config is 'tag:ENV=VAL|ENV=VAL'; an empty knob list is the plain tip)
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
from ....utils.tensor import typed_tensor
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


def _apply_knobs(tt_model, knobs: dict[str, str], mesh_device) -> str:
    """Put the model into the state a fresh process with exactly `knobs` in its environment would build."""
    for k in [k for k in os.environ if k.startswith("MINIMAX_H3_") and k not in _KEEP]:
        del os.environ[k]
    os.environ.update(knobs)
    fidelity = getattr(ttnn.MathFidelity, knobs.get("MINIMAX_H3_SDPA_FIDELITY", "HiFi2"))
    kv_dtype = getattr(ttnn, knobs["MINIMAX_H3_SDPA_KV_DTYPE"]) if "MINIMAX_H3_SDPA_KV_DTYPE" in knobs else None
    v_dtype = getattr(ttnn, knobs["MINIMAX_H3_SDPA_V_DTYPE"]) if "MINIMAX_H3_SDPA_V_DTYPE" in knobs else kv_dtype
    for block in tt_model.transformer_blocks:
        attn = block.attn
        attn.sdpa_fixed_offset = False
        attn.sdpa_fixed_offset_value = 0.0
        attn._sdpa_program_configs.clear()  # per-phase fidelity is read from the environment when a config is built
        attn.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        )

        def dummy(dtype):
            if dtype is None:
                return attn.dummy_joint_input
            return typed_tensor(torch.zeros((1, attn.n_local_heads, 0, attn.head_dim)), dtype, mesh_device)

        attn.sdpa_k_dtype, attn.sdpa_v_dtype = kv_dtype, v_dtype
        attn.dummy_joint_k, attn.dummy_joint_v = dummy(kv_dtype), dummy(v_dtype)
    spec = knobs.get("MINIMAX_H3_SDPA_FIXED_SOFTMAX_BLOCKS")
    if spec:
        tt_model._set_fixed_softmax_blocks(spec)
    # The bf8 typecast cannot be undone, so a config can only add linears to the set already cast; cast just those.
    applied = getattr(tt_model, "_cpu_ref_bf8_applied", set())
    wanted = {name for name in knobs.get("MINIMAX_H3_BF8_WEIGHTS", "").split(",") if name}
    if wanted - applied:
        os.environ["MINIMAX_H3_BF8_WEIGHTS"] = ",".join(sorted(wanted - applied))
        apply_env_quant_config(tt_model)
        os.environ["MINIMAX_H3_BF8_WEIGHTS"] = knobs["MINIMAX_H3_BF8_WEIGHTS"]
        tt_model._cpu_ref_bf8_applied = applied | wanted
    applied = getattr(tt_model, "_cpu_ref_bf8_applied", set())
    weights = "bf8:" + ",".join(sorted(applied)) if applied else "bf16"
    if hasattr(tt_model, "fused_heads"):
        tt_model.fused_heads = knobs.get("MINIMAX_H3_FUSED_HEADS", "0") == "1"
    return (
        " ".join(f"{k}={v}" for k, v in sorted(knobs.items())) + f" [weights {weights}]"
        if knobs
        else f"(tip) [weights {weights}]"
    )


@pytest.mark.timeout(10800)
@GALAXY_RING
def test_cpu_ref_forward(mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, reset_seeds) -> None:
    ref_path = os.environ.get("H3_CPU_REF")
    if not ref_path:
        pytest.skip("set H3_CPU_REF to a tools/cpu_reference_forward.py npz")
    configs = _configs()
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
    assert np.array_equal(inputs.position_ids.numpy(), ref["position_ids"]) and np.array_equal(
        inputs.tags.numpy(), ref["tags"]
    )

    # build the model in the plain-tip state; every knob set is applied in place afterwards
    _apply_knobs_env_only = [k for k in os.environ if k.startswith("MINIMAX_H3_") and k not in _KEEP]
    for k in _apply_knobs_env_only:
        del os.environ[k]
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
    logger.info(
        f"checkpoint on the mesh in {time.time() - start:.0f} s; {len(configs)} knob set(s): {[c[0] for c in configs]}"
    )
    tt_model.prepare_static_sources(**inputs.tt_static, prompt_cap=inputs.tt_static["prompt_1BLP"].shape[2])

    out_dir = Path(os.environ.get("H3_CPU_REF_OUT", Path(ref_path).parent))
    out_dir.mkdir(parents=True, exist_ok=True)

    def compose(t: ttnn.Tensor) -> np.ndarray:
        out = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape))
        )
        return out.reshape(-1, *out.shape[2:])[:1].float().numpy()

    results = []
    for tag, knobs in configs:
        knob_str = _apply_knobs(tt_model, knobs, mesh_device)
        logger.info(f"[{tag}] knobs: {knob_str}")
        tt_model(**inputs.tt)  # compile pass for this knob set
        ttnn.synchronize_device(mesh_device)
        start = time.time()
        tt_video, tt_audio = tt_model(**inputs.tt)
        ttnn.synchronize_device(mesh_device)
        warm = time.time() - start

        tt_video, tt_audio = compose(tt_video)[:, :num_video], compose(tt_audio)[:, :num_audio]
        out_path = out_dir / f"tt_{tag}.npz"
        np.savez(out_path, tt_video=tt_video, tt_audio=tt_audio, tag=tag, knobs=knob_str)
        row = (
            tag,
            warm,
            psnr(ref["ref_video"], tt_video),
            pcc(ref["ref_video"], tt_video),
            psnr(ref["ref_audio"], tt_audio),
            pcc(ref["ref_audio"], tt_audio),
        )
        results.append(row)
        logger.info(
            f"[{tag}] warm forward {warm:.2f} s; vs CPU reference: video PSNR {row[2]:.2f} dB PCC {row[3]:.6f}; "
            f"audio PSNR {row[4]:.2f} dB PCC {row[5]:.6f}; wrote {out_path}"
        )

    logger.info(
        "SUMMARY vs CPU reference (5 s shape, full depth, fp32 CPU): tag | warm s | video dB | video PCC | audio dB | audio PCC"
    )
    for tag, warm, vp, vc, ap, ac in results:
        logger.info(f"SUMMARY {tag:12s} | {warm:6.2f} | {vp:7.2f} | {vc:.6f} | {ap:7.2f} | {ac:.6f}")
