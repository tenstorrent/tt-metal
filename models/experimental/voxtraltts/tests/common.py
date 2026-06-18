# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch
import ttnn
from loguru import logger


from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_MODEL
from models.experimental.voxtraltts.tt.text_model import VoxtralTTTextModel
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_default_optimizations
from models.experimental.voxtraltts.utils.audio_tokenizer_optimizations import (
    voxtral_audio_tokenizer_default_optimizations,
)

VOXTRAL_STANDARD_CHAR_TEXT = (
    "Voxtral TTS is a frontier open weights text to speech model for production voice agents. It produces "
    "realistic expressive speech with natural prosody across English, French, Spanish, German, Italian, "
    "Portuguese, Dutch, Arabic, and Hindi. The system supports preset voices, low latency streaming, batch "
    "inference, and twenty four kilohertz audio output for customer support, real time translation, reading "
    "applications, call centers, and responsive multilingual assistant workflows. With clear speech."
)


def ensure_voxtral_device_available() -> None:
    """Skip the test when the host exposes no Tenstorrent devices."""
    if ttnn.get_num_devices() < 1:
        pytest.skip("No Tenstorrent device available on this host")


def voxtral_single_device_mesh_shape() -> ttnn.MeshShape:
    """1×1 compute mesh — default for acoustic/audio (replicated weights on multi-device)."""
    ensure_voxtral_device_available()
    return ttnn.MeshShape(1, 1)


def voxtral_parse_mesh_shape(env_value: str) -> tuple[int, int]:
    parts = [p.strip() for p in env_value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected mesh shape 'rows,cols', got {env_value!r}")
    return int(parts[0]), int(parts[1])


def voxtral_requested_compute_mesh_shape() -> tuple[int, int]:
    """Parse ``VOXTRAL_COMPUTE_MESH_SHAPE`` without probing PCIe (safe before device open)."""
    raw = os.getenv("VOXTRAL_COMPUTE_MESH_SHAPE", "1,1").strip()
    return voxtral_parse_mesh_shape(raw)


def voxtral_compute_mesh_shape() -> ttnn.MeshShape:
    """Compute mesh shape from ``VOXTRAL_COMPUTE_MESH_SHAPE`` (default ``1,1``).

    On P150 (no 1×4 host mesh) this is always 1×1 regardless of the environment.
    On BH QB2, ``1,4`` enables tensor-parallel text on the full host mesh; acoustic and
    audio tokenizer replicate weights onto every device.
    """
    ensure_voxtral_device_available()
    if voxtral_host_mesh_shape() is None:
        return ttnn.MeshShape(1, 1)
    raw = os.getenv("VOXTRAL_COMPUTE_MESH_SHAPE", "1,1").strip()
    rows, cols = voxtral_parse_mesh_shape(raw)
    return ttnn.MeshShape(rows, cols)


def voxtral_host_mesh_shape() -> ttnn.MeshShape | None:
    """Host mesh for BH QuietBox 2 (1×4). ``None`` on single-card P150."""
    ensure_voxtral_device_available()
    if ttnn.device.is_blackhole() and ttnn.get_num_devices() == 4:
        return ttnn.MeshShape(1, 4)
    return None


def voxtral_log_runtime_mesh(runtime: "VoxtralRuntimeMesh") -> None:
    """Log host/compute topology for demo and e2e diagnostics."""
    host = runtime.host_mesh_shape
    compute = runtime.compute_mesh_shape
    if host is not None:
        logger.info(
            "voxtral mesh: host={} compute={} arch={} pcie_ids={}",
            host,
            compute,
            ttnn.get_arch_name(),
            runtime.physical_device_ids,
        )
    else:
        logger.info(
            "voxtral mesh: compute={} arch={} device_id={} host_devices={}",
            compute,
            ttnn.get_arch_name(),
            runtime.physical_device_id,
            ttnn.get_num_devices(),
        )


@dataclass
class VoxtralRuntimeMesh:
    """Device handle returned by :func:`open_voxtral_runtime_mesh`."""

    compute_device: ttnn.Device | ttnn.MeshDevice
    fabric: dict
    previous_default: ttnn.Device | None
    physical_device_id: int
    parent_mesh: ttnn.MeshDevice | None = None
    host_mesh_shape: tuple[int, int] | None = None
    compute_mesh_shape: tuple[int, int] = (1, 1)
    compute_uses_submesh: bool = False

    @property
    def physical_device_ids(self) -> list[int]:
        if self.parent_mesh is not None:
            return list(ttnn.get_pcie_device_ids())
        return [ttnn.GetPCIeDeviceID(self.physical_device_id)]


def voxtral_default_fabric_config():
    """Fabric mode for multi-device Voxtral compute (CCL all-gather / reduce-scatter)."""
    cluster = ttnn.cluster.get_cluster_type()
    if cluster in (
        ttnn.cluster.ClusterType.P150_X4,
        ttnn.cluster.ClusterType.P150_X8,
    ):
        return ttnn.FabricConfig.FABRIC_1D_RING
    return ttnn.FabricConfig.FABRIC_1D


def prepare_voxtral_open_mesh_kwargs(device_params: dict | None) -> tuple[dict, dict]:
    """Host-aware ``open_mesh_device`` kwargs plus fabric settings for teardown.

    Applies ``get_updated_device_params`` (Blackhole/QB dispatch axis, etc.) and strips
    fabric keys that must be passed to ``set_fabric`` before opening the mesh.
    """
    from tests.scripts.common import get_updated_device_params

    updated = get_updated_device_params(dict(device_params or {}))
    fabric = {
        "fabric_config": updated.pop("fabric_config", None),
        "fabric_tensix_config": updated.pop("fabric_tensix_config", None),
        "reliability_mode": updated.pop("reliability_mode", None),
        "fabric_manager": updated.pop("fabric_manager", None),
        "fabric_router_config": updated.pop("fabric_router_config", None),
    }
    return updated, fabric


def voxtral_resolve_physical_device_id(device_id: int | None = None) -> int:
    """Pick one PCIe device for single-card Voxtral (P150 or one rank on BH QB2 1×4)."""
    if device_id is not None:
        return int(device_id)
    env_id = os.getenv("VOXTRAL_DEVICE_ID")
    if env_id is not None and env_id.strip() != "":
        return int(env_id)
    if ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.TG:
        from conftest import first_available_tg_device

        return first_available_tg_device()
    return 0


def open_voxtral_runtime_mesh(
    device_params: dict | None = None,
    *,
    device_id: int | None = None,
) -> VoxtralRuntimeMesh:
    """Open the Voxtral runtime device with host-aware mesh selection.

    * **P150 (1 card):** ``CreateDevice(0)`` — unchanged single-card path.
    * **BH QB2 (4 cards):** ``open_mesh_device(1×4)`` for the host fabric topology. By default a
      ``1×1`` submesh is used for compute (audio-safe, ``cluster_shape=(1,1)``). Set
      ``VOXTRAL_COMPUTE_MESH_SHAPE=1,4`` to run tensor-parallel text on the full 1×4 mesh;
      acoustic and audio tokenizer replicate weights on every device.
    """
    from conftest import set_fabric

    physical_device_id = voxtral_resolve_physical_device_id(device_id)
    params = dict(device_params or {})
    requested_compute = voxtral_requested_compute_mesh_shape()
    if int(requested_compute[0]) * int(requested_compute[1]) > 1 and params.get("fabric_config") is None:
        params["fabric_config"] = voxtral_default_fabric_config()
    host_shape = voxtral_host_mesh_shape()
    open_kwargs, fabric = prepare_voxtral_open_mesh_kwargs(params)
    previous_default = ttnn.GetDefaultDevice()

    if host_shape is not None:
        set_fabric(
            fabric["fabric_config"],
            fabric["reliability_mode"],
            fabric["fabric_tensix_config"],
            fabric["fabric_manager"],
            fabric["fabric_router_config"],
        )
        parent_mesh = ttnn.open_mesh_device(mesh_shape=host_shape, **open_kwargs)
        compute_shape = voxtral_compute_mesh_shape()
        host_devices = int(host_shape[0]) * int(host_shape[1])
        compute_devices = int(compute_shape[0]) * int(compute_shape[1])
        if compute_devices > host_devices:
            raise ValueError(
                f"VOXTRAL_COMPUTE_MESH_SHAPE={tuple(compute_shape)} exceeds host mesh "
                f"{tuple(host_shape)} ({host_devices} devices)"
            )
        if compute_devices > 1:
            if compute_devices != host_devices:
                raise ValueError(
                    f"Partial multi-device compute is unsupported: host={tuple(host_shape)} "
                    f"compute={tuple(compute_shape)}"
                )
            compute_device = parent_mesh
            compute_uses_submesh = False
        else:
            compute_device = parent_mesh.create_submesh(voxtral_single_device_mesh_shape())
            compute_uses_submesh = True
        ttnn.SetDefaultDevice(compute_device)
        runtime = VoxtralRuntimeMesh(
            compute_device=compute_device,
            fabric=fabric,
            previous_default=previous_default,
            physical_device_id=physical_device_id,
            parent_mesh=parent_mesh,
            host_mesh_shape=tuple(parent_mesh.shape),
            compute_mesh_shape=tuple(compute_device.shape),
            compute_uses_submesh=compute_uses_submesh,
        )
        voxtral_log_runtime_mesh(runtime)
        return runtime

    device = ttnn.CreateDevice(device_id=physical_device_id, **open_kwargs)
    ttnn.SetDefaultDevice(device)
    runtime = VoxtralRuntimeMesh(
        compute_device=device,
        fabric=fabric,
        previous_default=previous_default,
        physical_device_id=physical_device_id,
        parent_mesh=None,
        host_mesh_shape=None,
        compute_mesh_shape=(1, 1),
        compute_uses_submesh=False,
    )
    voxtral_log_runtime_mesh(runtime)
    return runtime


def close_voxtral_runtime_mesh(runtime: VoxtralRuntimeMesh) -> None:
    """Tear down resources opened by :func:`open_voxtral_runtime_mesh`."""
    from conftest import reset_fabric

    if runtime.previous_default is not None:
        ttnn.SetDefaultDevice(runtime.previous_default)

    if runtime.parent_mesh is not None:
        if runtime.compute_uses_submesh:
            for submesh in runtime.parent_mesh.get_submeshes():
                ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(runtime.parent_mesh)
        reset_fabric(runtime.fabric["fabric_config"])
    else:
        ttnn.close_device(runtime.compute_device)


VOXTRAL_PRESET_VOICES: tuple[str, ...] = (
    "casual_male",
    "casual_female",
    "cheerful_female",
    "neutral_male",
    "neutral_female",
    "ar_male",
    "de_male",
    "de_female",
    "es_male",
    "es_female",
    "fr_male",
    "fr_female",
    "hi_male",
    "hi_female",
    "it_male",
    "it_female",
    "nl_male",
    "nl_female",
    "pt_male",
    "pt_female",
)


def resolve_voxtral_model_name_or_skip() -> str:
    """Return Voxtral checkpoint id from env or skip."""
    model_name_or_path = os.getenv("VOXTRAL_TTS_MODEL") or os.getenv("HF_MODEL") or DEFAULT_VOXTRAL_MODEL
    if "voxtral" not in model_name_or_path.lower():
        pytest.skip(
            f"Expected a Voxtral checkpoint, got '{model_name_or_path}'. "
            "Set VOXTRAL_TTS_MODEL or HF_MODEL to a Voxtral model/repo."
        )
    return model_name_or_path


def create_real_voxtral_text_model_or_skip(
    device,
    *,
    max_seq_len: int = 256,
    max_batch_size: int = 1,
    dtype=ttnn.bfloat16,
    optimizations=voxtral_text_default_optimizations,
):
    """Build the TT text model with the production config by default."""
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        return VoxtralTTTextModel.create_from_model_name(
            mesh_device=device,
            model_name_or_path=model_name_or_path,
            dtype=dtype,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
        )
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTTextModel from real checkpoint: {exc}")


def create_voxtral_audio_tokenizer_or_skip(
    device,
    *,
    state_dict,
    tokenizer_cfg,
    full_checkpoint=None,
    optimizations=voxtral_audio_tokenizer_default_optimizations,
):
    """Build ``VoxtralTTAudioTokenizer`` with production optimizations by default."""
    from models.experimental.voxtraltts.tt.audio_tokenizer.model import VoxtralTTAudioTokenizer

    opt = optimizations() if callable(optimizations) else optimizations
    try:
        return VoxtralTTAudioTokenizer(
            device,
            state_dict=state_dict,
            tokenizer_cfg=tokenizer_cfg,
            full_checkpoint=full_checkpoint,
            optimizations=opt,
        )
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTAudioTokenizer: {exc}")


def log_per_step_code_match(ref_codes: torch.Tensor, tt_codes: torch.Tensor) -> None:
    """Log semantic/acoustic code agreement per AR step (one CPU + one TT E2E forward)."""
    n_frames = min(int(tt_codes.shape[2]), int(ref_codes.shape[2]))
    n_acoustic = ref_codes.shape[1] - 1
    first_diff_step: int | None = None

    logger.info("")
    logger.info("=" * 70)
    logger.info("PER-STEP CODE MATCH (prod path, single CPU + single TT forward)")
    logger.info("=" * 70)

    for t in range(n_frames):
        sem_ok = bool((ref_codes[0, 0, t] == tt_codes[0, 0, t]).item())
        ac_match = int((ref_codes[0, 1:, t] == tt_codes[0, 1:, t]).sum().item())
        ac_ok = ac_match == n_acoustic
        if not sem_ok or not ac_ok:
            if first_diff_step is None:
                first_diff_step = t
            if not ac_ok:
                bad_cb = (ref_codes[0, 1:, t] != tt_codes[0, 1:, t]).nonzero(as_tuple=False).reshape(-1)
                bad_preview = [
                    f"cb{int(i.item())}:{int(ref_codes[0, 1 + i, t].item())}->{int(tt_codes[0, 1 + i, t].item())}"
                    for i in bad_cb[:6]
                ]
                ac_detail = f" mismatches={bad_preview}"
                if bad_cb.numel() > 6:
                    ac_detail += f" ... (+{int(bad_cb.numel()) - 6} more)"
            else:
                ac_detail = ""
            logger.info(
                f"  step {t}: semantic={'OK' if sem_ok else 'DIFF'} "
                f"(cpu={int(ref_codes[0, 0, t].item())} tt={int(tt_codes[0, 0, t].item())}) "
                f"acoustic={ac_match}/{n_acoustic}{ac_detail}"
            )
        else:
            logger.info(f"  step {t}: semantic=OK acoustic={ac_match}/{n_acoustic} [all match]")

    if first_diff_step is None:
        logger.info(f"  summary: all {n_frames} steps match CPU codes exactly")
    else:
        logger.info(f"  summary: first divergence at step {first_diff_step}")
