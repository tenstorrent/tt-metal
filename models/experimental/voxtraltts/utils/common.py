# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import bz2
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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
from models.experimental.voxtraltts.utils.mesh import voxtral_mesh_device_compute_shape

VOXTRAL_GOLDEN_CODES_DEFAULT_PATH = (
    Path(__file__).resolve().parents[1] / "reference" / "reference_outputs" / "voxtral_golden_codes.refpt"
)

VOXTRAL_STANDARD_CHAR_TEXT = "Voxtral is a 4 billion parameter open-weight TTS model released by Mistral AI in 2026, designed for low-latency, multilingual voice generation across English, Spanish, French, Portuguese, Hindi, German, Dutch, and Italian. It builds on the Mistral 3 billion language backbone with a flow-matching acoustic decoder and produces audio at 12.5 Hz with high quality, suitable for streaming voice applications and real-time agent deployments which supports low latency."

# Same fixture as ``models/tt_transformers/tests/test_model_prefill.py``.
TALE_OF_TWO_CITIES_BZ2 = (
    Path(__file__).resolve().parents[3] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2"
)


@lru_cache(maxsize=1)
def tale_of_two_cities_text_or_skip() -> str:
    """Raw UTF-8 text of *A Tale of Two Cities* (shared PCC / perf fixture)."""
    if not TALE_OF_TWO_CITIES_BZ2.is_file():
        pytest.skip(f"Missing tale prompt fixture: {TALE_OF_TWO_CITIES_BZ2}")
    with bz2.open(TALE_OF_TWO_CITIES_BZ2, "rt", encoding="utf-8") as f:
        text = f.read()
    if not text:
        pytest.skip("Tale of Two Cities fixture is empty")
    return text


@lru_cache(maxsize=1)
def tale_of_two_cities_token_ids_or_skip() -> tuple[int, ...]:
    """Tokenize *A Tale of Two Cities* for PCC / perf prompts."""
    text = tale_of_two_cities_text_or_skip()
    from models.experimental.voxtraltts.reference.voxtral_request import encode_plain_text_to_token_ids

    model_name = resolve_voxtral_model_name_or_skip()
    try:
        token_ids = encode_plain_text_to_token_ids(text, model_name)
    except ImportError as exc:
        pytest.skip(f"mistral-common required for tale prompts: {exc}")
    except Exception as exc:
        pytest.skip(f"Voxtral tokenizer unavailable for tale prompts ({model_name}): {exc}")
    if not token_ids:
        pytest.skip("Tale of Two Cities tokenization returned an empty sequence")
    logger.info(f"Tale of Two Cities prompt: {len(token_ids)} tokens from {TALE_OF_TWO_CITIES_BZ2.name}")
    return tuple(int(t) for t in token_ids)


def tale_prompt_tokens(seq_len: int) -> torch.Tensor:
    ids = tale_of_two_cities_token_ids_or_skip()
    if len(ids) < seq_len:
        pytest.skip(f"Tale prompt has {len(ids)} tokens, need seq_len={seq_len}")
    return torch.tensor([list(ids[:seq_len])], dtype=torch.int64)


def tale_continuation_tokens(start: int, count: int) -> torch.Tensor:
    ids = tale_of_two_cities_token_ids_or_skip()
    end = start + count
    if len(ids) < end:
        pytest.skip(f"Tale prompt has {len(ids)} tokens, need continuation [{start}:{end})")
    return torch.tensor([list(ids[start:end])], dtype=torch.int64)


def _tale_speech_text_for_min_prompt_len_impl(min_prompt_len: int, model_name: str, voice: str) -> tuple[str, int]:
    from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request

    text = tale_of_two_cities_text_or_skip()
    lo, hi = 1, len(text)
    best_text = ""
    best_len = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        prefix = text[:mid]
        prompt_len = len(compose_speech_request(prefix, model_name, voice=voice)["prompt_token_ids"])
        if prompt_len >= min_prompt_len:
            best_text, best_len = prefix, prompt_len
            hi = mid - 1
        else:
            lo = mid + 1
    if best_len < min_prompt_len:
        pytest.skip(
            f"Tale speech prompt cannot reach min_prompt_len={min_prompt_len} "
            f"(best voice-wrapped length={best_len})"
        )
    return best_text, best_len


@lru_cache(maxsize=32)
def _tale_speech_text_for_min_prompt_len_cached(min_prompt_len: int, model_name: str, voice: str) -> tuple[str, int]:
    return _tale_speech_text_for_min_prompt_len_impl(min_prompt_len, model_name, voice)


def tale_speech_text_for_min_prompt_len(
    min_prompt_len: int,
    *,
    model_name: str,
    voice: str = "casual_male",
) -> tuple[str, int]:
    """Return ``(text_prefix, prompt_seq_len)`` with voice-wrapped speech prompt length ≥ ``min_prompt_len``.

    Uses binary search over the tale fixture so perf sweeps exercise production
    ``compose_speech_request`` tokenization (voice template + user text).
    """
    return _tale_speech_text_for_min_prompt_len_cached(min_prompt_len, model_name, voice)


def speech_prompt_seq_len(text: str, *, model_name: str, voice: str = "casual_male") -> int:
    """Voice-wrapped prompt length for ``compose_speech_request`` (production path)."""
    from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request

    return len(compose_speech_request(text, model_name, voice=voice)["prompt_token_ids"])


def ensure_voxtral_device_available() -> None:
    """Skip the test when the host exposes no Tenstorrent devices."""
    if ttnn.get_num_devices() < 1:
        pytest.skip("No Tenstorrent device available on this host")


def voxtral_single_device_mesh_shape() -> ttnn.MeshShape:
    """1×1 compute mesh — default for acoustic/audio (replicated weights on multi-device)."""
    ensure_voxtral_device_available()
    return ttnn.MeshShape(1, 1)


def voxtral_requested_compute_mesh_shape() -> tuple[int, int]:
    """Parse ``MESH_DEVICE`` without probing PCIe (safe before device open)."""
    return voxtral_mesh_device_compute_shape()


def voxtral_compute_mesh_shape() -> ttnn.MeshShape:
    """Compute mesh shape from ``MESH_DEVICE`` (default ``1,1`` when unset).

    On P150 (no 1×4 host mesh) this is always 1×1 regardless of ``MESH_DEVICE``.
    On BH QB2, ``P150x4`` enables tensor-parallel text on the full 1×4 mesh; acoustic and
    audio tokenizer replicate weights onto every device.
    """
    ensure_voxtral_device_available()
    if voxtral_host_mesh_shape() is None:
        return ttnn.MeshShape(1, 1)
    rows, cols = voxtral_mesh_device_compute_shape()
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
      ``MESH_DEVICE=P150x4`` to run tensor-parallel text on the full 1×4 mesh; acoustic and audio
      tokenizer replicate weights on every device.
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
                f"MESH_DEVICE requests compute mesh {tuple(compute_shape)}, which exceeds host mesh "
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


def load_voxtral_checkpoint_or_skip() -> dict[str, torch.Tensor]:
    """Load the full Voxtral safetensors checkpoint or skip when unavailable."""
    from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict

    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        return _load_safetensors_state_dict(model_name_or_path)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")


def load_acoustic_fm_layer_weights_or_skip(layer_idx: int = 0) -> dict[str, torch.Tensor]:
    """Per-layer weights from ``acoustic_transformer.layers.{layer_idx}``."""
    from models.experimental.voxtraltts.reference.functional import extract_acoustic_layer_weights

    full = load_voxtral_checkpoint_or_skip()
    weights = extract_acoustic_layer_weights(full, layer_idx)
    if not weights:
        pytest.skip(f"No acoustic_transformer.layers.{layer_idx} weights in checkpoint")
    return weights


def load_audio_tokenizer_state_dict_or_skip() -> dict[str, torch.Tensor]:
    """Audio tokenizer subset of the Voxtral checkpoint."""
    from models.experimental.voxtraltts.tt.audio_tokenizer.model import extract_audio_tokenizer_state_dict

    full = load_voxtral_checkpoint_or_skip()
    sd = extract_audio_tokenizer_state_dict(full)
    if not sd:
        pytest.skip("No audio tokenizer weights in checkpoint")
    return sd


def create_real_voxtral_text_model_or_skip(
    device,
    *,
    max_seq_len: int = 256,
    max_batch_size: int = 1,
    dtype=ttnn.bfloat16,
    optimizations=voxtral_text_default_optimizations,
    use_paged_kv_cache: bool = True,
    paged_block_size: int = 32,
):
    """Build the TT text model with the production config by default.

    ``use_paged_kv_cache=True`` (default) builds ``paged_attention_config`` for paged
    SDPA at all sequence lengths. Set ``False`` only to exercise the non-paged path.
    The text model always uses internal KV blocks (``layer_past``); ``use_paged_kv_cache``
    on ``VoxtralTTTextModel.create`` selects paged vs default attention layout.
    """
    import math

    from models.experimental.voxtraltts.tt.text_backbone.common import PagedAttentionConfig

    model_name_or_path = resolve_voxtral_model_name_or_skip()
    paged_cfg = None
    if use_paged_kv_cache:
        max_num_blocks = math.ceil(max_seq_len / paged_block_size)
        paged_cfg = PagedAttentionConfig(block_size=paged_block_size, max_num_blocks=max_num_blocks)
    try:
        return VoxtralTTTextModel.create_from_model_name(
            mesh_device=device,
            model_name_or_path=model_name_or_path,
            dtype=dtype,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
            paged_attention_config=paged_cfg,
            # Internal KV blocks; paged_attention_config selects paged SDPA layout.
            use_paged_kv_cache=False,
        )
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTTextModel from real checkpoint: {exc}")


def build_voxtral_text_page_table_host(*, max_seq_len: int, paged_block_size: int = 32) -> torch.Tensor:
    """Host page table ``[1, max_num_blocks]`` sized to the full KV block pool."""
    import math

    max_num_blocks = math.ceil(max_seq_len / paged_block_size)
    return torch.arange(max_num_blocks, dtype=torch.int32).unsqueeze(0)


def build_voxtral_text_page_table_tt(mesh_device, *, max_seq_len: int, paged_block_size: int = 32) -> ttnn.Tensor:
    """Device page table for paged KV prefill/decode (full block pool)."""
    from models.experimental.voxtraltts.utils.mesh import voxtral_replicate_mesh_mapper

    page_table_host = build_voxtral_text_page_table_host(
        max_seq_len=max_seq_len,
        paged_block_size=paged_block_size,
    )
    return ttnn.from_torch(
        page_table_host,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=voxtral_replicate_mesh_mapper(mesh_device),
    )


@lru_cache(maxsize=1)
def hf_voxtral_text_reference_or_skip():
    """Cached HF ``MistralForCausalLM`` text backbone loaded from the Voxtral checkpoint.

    Used for incremental-decode PCC: prefill with ``use_cache=True``, then thread
    ``past_key_values`` step by step (same contract as the TT decode path under test).
    """
    from transformers import MistralForCausalLM

    from models.experimental.voxtraltts.reference.cpu_reference import (
        _build_text_config,
        _load_text_state_dict,
        _resolve_model_file,
    )
    from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config

    name = resolve_voxtral_model_name_or_skip()
    try:
        cfg = load_voxtral_config(name)
        ckpt = _resolve_model_file(name, "consolidated.safetensors")
        hf = MistralForCausalLM(_build_text_config(name))
        hf.load_state_dict(_load_text_state_dict(ckpt, cfg), strict=False)
    except Exception as exc:
        pytest.skip(f"HF text reference unavailable: {exc}")
    return hf.to(dtype=torch.bfloat16).eval()


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
