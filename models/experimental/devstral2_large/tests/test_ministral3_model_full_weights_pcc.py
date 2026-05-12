# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-depth PCC: **text-only** HF ``Ministral3Model`` vs :class:`~models.experimental.devstral2_large.tt.tt_ministral3_model.TtMinistral3Model`.

The TT stack implements the Hugging Face **language** backbone only—``embed_tokens``, all
decoder ``layers``, final ``norm``, and HF-format RoPE tables—**not** ``lm_head`` and not
vision towers (see ``tt_ministral3_model.py`` module doc). Checkpoints are filtered to text
trunk tensors via :func:`_text_trunk_sd`; multimodal Hub layouts are flattened only so
``ModelArgs`` / ``Ministral3Config`` match the Devstral-2 snapshot.

Opt-in: ``DEVSTRAL2_FULL_MODEL_PCC=1``.

The full-weights PCC test persists HF inputs and reference hidden states to ``.pt`` files,
drops the Hugging Face module from RAM, and calls ``malloc_trim`` on Linux before the
TTNN model runs.

**Checkpoint resolution**

1. If ``DEVSTRAL2_WEIGHTS_DIR`` is set it must point at a directory that contains
   ``model.safetensors.index.json`` (same layout as an HF hub snapshot).
2. Otherwise ``snapshot_download(local_files_only=True)`` is used against the hub
   cache (default ``~/.cache/huggingface/hub``).

**ARC watchdog fix**

The Blackhole KMD programs the ARC management-processor watchdog to 10 s when
``open_mesh_device`` is called (``ARC_MSG_TYPE_SET_WDT_TIMEOUT = 10 000 ms`` in
``/usr/src/tenstorrent-<ver>/blackhole.c``).  The ARC WDT is reset only by
ARC-level messages — not by PCIe DMA.  Loading 128 GB of FP8 safetensors shards
takes ~2 minutes with zero ARC traffic, so the watchdog fires, the chip resets,
and the process receives SIGTERM at ~20 s.

The ``_preloaded_devstral2_weights`` fixture has ``scope="module"``, which means
pytest sets it up *before* the function-scoped ``mesh_device`` fixture ever opens
the device.  All slow disk I/O therefore runs with no device open at all.
"""

from __future__ import annotations

import ctypes
import gc
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tests.test_ministral3_model import (
    DEVSTRAL2_LARGE_REPO_ID,
    _build_meta_state_dict,
    _force_mlp_weight_dtypes_bf16,
    _force_replicated_rmsnorm,
    _text_cfg_from_hf_cfg,
    _to_bf16_host_if_fp8,
)
from models.experimental.devstral2_large.tt.tt_ministral3_model import TtMinistral3Model
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import DEVSTRAL2_LARGE_L1_SMALL_SIZE
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict, standardize_hf_keys_multimodal
from models.tt_transformers.tt.model_config import ModelArgs, PrecisionSetting, TensorGroup


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _torch_load_pt(path: Path):
    """Load a ``torch.save`` payload written by this test (trusted local file)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _release_pytorch_cpu_ram() -> None:
    """Best-effort return of freed CPU tensor memory to the OS (no CUDA involved)."""
    gc.collect()
    gc.collect()
    if sys.platform == "linux":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass


def _resolve_ckpt_dir(repo_id: str) -> Path:
    """Return the directory that holds ``model.safetensors.index.json``."""
    explicit = os.environ.get("DEVSTRAL2_WEIGHTS_DIR")
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not (p / "model.safetensors.index.json").is_file():
            pytest.skip(
                f"DEVSTRAL2_WEIGHTS_DIR={p} does not contain model.safetensors.index.json. "
                "Point it at a Hub snapshot root, or unset to use the hub cache."
            )
        return p

    from huggingface_hub import snapshot_download

    try:
        snap = snapshot_download(repo_id, local_files_only=False)
    except Exception as exc:
        pytest.skip(
            f"Devstral-2 weights not in HF cache ({exc}). "
            "Run `huggingface-cli download mistralai/Devstral-2-123B-Instruct-2512` "
            "or set DEVSTRAL2_WEIGHTS_DIR."
        )
    p = Path(snap).resolve()
    if not (p / "model.safetensors.index.json").is_file():
        pytest.skip(f"snapshot_download returned {p} but model.safetensors.index.json is missing.")
    return p


def _text_trunk_sd(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep only ``Ministral3Model`` HF keys; strip vision / projector tensors."""
    skip = (
        "model.visual.",
        "model.vision_tower.",
        "model.vision_model.",
        "model.multi_modal_projector.",
    )
    lang = "model.language_model."
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if any(k.startswith(p) for p in skip):
            continue
        if k.startswith(lang):
            out["model." + k[len(lang) :]] = v
        elif k.startswith("model."):
            out[k] = v
        elif k == "lm_head.weight":
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# module-scoped pre-load fixture  (runs BEFORE mesh_device opens the device)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _preloaded_devstral2_weights():
    """Load every shard from disk before any TT device is opened.

    Returns ``{hf_key: bf16_tensor}`` for the **text trunk** only (language-model / ``Ministral3Model``
    keys; vision and projector tensors removed), or skips the module if the checkpoint is
    unavailable or ``DEVSTRAL2_FULL_MODEL_PCC`` is not set.
    """
    if os.environ.get("DEVSTRAL2_FULL_MODEL_PCC") != "1":
        pytest.skip(
            "Set DEVSTRAL2_FULL_MODEL_PCC=1 to run full checkpoint PCC. "
            "Weights are read from the HF hub cache or DEVSTRAL2_WEIGHTS_DIR."
        )

    ckpt_dir = _resolve_ckpt_dir(DEVSTRAL2_LARGE_REPO_ID)
    logger.info(f"Pre-loading Devstral-2 shards from {ckpt_dir} (no device open yet)")

    try:
        raw_sd = load_hf_state_dict(str(ckpt_dir))
    except Exception as exc:
        pytest.skip(f"load_hf_state_dict failed: {exc}")

    # Normalise multimodal key layout if present
    if any("model.language_model." in k for k in raw_sd):
        raw_sd = standardize_hf_keys_multimodal(raw_sd)

    hf_sd = _text_trunk_sd(raw_sd)
    del raw_sd

    if not hf_sd:
        pytest.skip("No model.* text tensors found after filtering the checkpoint.")

    # Convert FP8 tensors to bf16 on CPU (no GPU/device needed)
    for k in list(hf_sd.keys()):
        hf_sd[k] = _to_bf16_host_if_fp8(hf_sd[k])

    logger.info(f"Pre-load complete: {len(hf_sd)} text-trunk tensors in bf16")
    return hf_sd


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    """``ModelArgs`` needs Devstral's multimodal Hub config; the PCC itself is **text trunk only**."""

    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    monkeypatch.setattr(mc.ModelArgs, "_set_hf_params", _set_hf_params_trust)

    def _get_hf_model_cls_devstral_safe(self):
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        if not self.is_multimodal:
            return AutoModelForCausalLM
        if type(self.hf_config) in AutoModelForImageTextToText._model_mapping:
            return AutoModelForImageTextToText
        raise ValueError(
            f"Test supports multimodal configs in AutoModelForImageTextToText only; got {type(self.hf_config)}"
        )

    monkeypatch.setattr(mc.ModelArgs, "get_hf_model_cls", _get_hf_model_cls_devstral_safe)


@pytest.fixture
def devstral2_123b_dummy_config_path(monkeypatch):
    from models.tt_transformers.tt import model_config as mc

    short_name = DEVSTRAL2_LARGE_REPO_ID.strip("/").split("/")[-1]
    monkeypatch.setitem(mc.ModelArgs.LOCAL_HF_PARAMS, short_name, DEVSTRAL2_LARGE_REPO_ID)


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "P150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
def test_ministral3_model_pcc_devstral2_large_full_weights_all_layers(
    mesh_device,
    seq_len,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
    devstral2_123b_dummy_config_path,
    _preloaded_devstral2_weights,
):
    """PCC: HF ``Ministral3Model`` **text backbone** vs ``TtMinistral3Model``.

    Matches :class:`~models.experimental.devstral2_large.tt.tt_ministral3_model.TtMinistral3Model`:
    ``embed_tokens`` → decoder ``layers`` → final ``norm`` (HF ``last_hidden_state``); no
    ``lm_head``, no vision. Reference uses an explicit causal ``attention_mask``; TT uses
    causal attention on device (``attention_mask`` ignored on TT).

    Weights are pre-loaded (module scope) as **text trunk** keys only. After the HF forward,
    ``inputs.pt`` / ``ref_out.pt`` are written, the HF module is freed, and the CPU heap is
    trimmed before ``TtMinistral3Model``. ``ref_out`` is reloaded from disk just before PCC.

    Device opens shortly before TT work so the ARC watchdog window stays safe. Peak RAM
    during the HF reference is ``hf_sd`` plus a full ``Ministral3Model`` copy.
    """
    # hf_sd: text-trunk HF keys only (vision stripped in ``_text_trunk_sd``)
    hf_sd = _preloaded_devstral2_weights

    monkeypatch.setenv("HF_MODEL", DEVSTRAL2_LARGE_REPO_ID)
    monkeypatch.setenv("MAX_PREFILL_CHUNK_SIZE", "128")

    # ---- device-configured ModelArgs (fast: reads config.json only, ~0.1 s) ----
    dtype = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max(512, seq_len),
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=False,
    )
    text_cfg = _text_cfg_from_hf_cfg(model_args.hf_config)

    # ---- build meta state dict (key renames; no large tensor copies) ----
    try:
        meta_sd = _build_meta_state_dict(hf_sd, text_cfg)
    except Exception as exc:
        pytest.skip(f"HF→meta key conversion failed: {exc}")
    logger.info("HF→meta state dict ready (still shares storage with hf_sd where possible)")

    # ---- set precision overrides for bf16 PCC comparison ----
    for _lid, dopt in model_args.decoders_optimizations.decoder_optimizations.items():
        dopt.tensor_dtype_settings[TensorGroup.WQKV] = PrecisionSetting.BF16
        dopt.tensor_dtype_settings[TensorGroup.WO] = PrecisionSetting.BF16
        dopt.tensor_dtype_settings[TensorGroup.ACTIVATION] = PrecisionSetting.BF16
        dopt.tensor_dtype_settings[TensorGroup.KV_CACHE] = PrecisionSetting.BF16
    _force_mlp_weight_dtypes_bf16(model_args)
    _force_replicated_rmsnorm(model_args)

    # ---- reference HF model (CPU bf16), persist I/O, then drop all HF-only RAM ----
    # ``hf_sd`` already holds one full bf16 copy; ``load_state_dict`` duplicates into
    # ``Ministral3Model``. After the forward we write ``inputs.pt`` / ``ref_out.pt``,
    # delete the HF model and in-memory ref tensors, and trim the CPU heap before TTNN.
    text_cfg._attn_implementation = "eager"
    ref_weights = {k[len("model.") :]: v for k, v in hf_sd.items() if k.startswith("model.")}
    ref_weights.pop("lm_head.weight", None)
    logger.info(
        "Allocating HF Ministral3Model + load_state_dict (peak host RAM ≈ hf_sd + full model copy; "
        "OOM kill here is not the TTNN path)"
    )
    hf_model = Ministral3Model(text_cfg).eval().to(torch.bfloat16)
    hf_model.load_state_dict(ref_weights, strict=True)
    logger.info("Reference Ministral3Model loaded")

    torch.manual_seed(42)
    gen = torch.Generator(device="cpu").manual_seed(42)
    input_ids = torch.randint(0, text_cfg.vocab_size, (batch_size, seq_len), dtype=torch.long, generator=gen)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    inputs_embeds = hf_model.embed_tokens(input_ids)
    causal_mask = create_causal_mask(
        config=text_cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    ref_out = hf_model(
        input_ids=input_ids,
        attention_mask=causal_mask,
        position_ids=position_ids,
        use_cache=False,
    ).last_hidden_state
    ref_out = ref_out.detach().to(dtype=torch.bfloat16, device="cpu", copy=True).contiguous()
    logger.info(f"Reference output shape: {ref_out.shape}")

    expected_shape = (batch_size, seq_len, text_cfg.hidden_size)
    assert ref_out.shape == expected_shape, f"HF text backbone output shape {ref_out.shape} != {expected_shape}"

    tmp_dir = Path(tempfile.mkdtemp(prefix="devstral2_full_pcc_"))
    inputs_pt = tmp_dir / "inputs.pt"
    ref_out_pt = tmp_dir / "ref_out.pt"
    try:
        torch.save(
            {"input_ids": input_ids.cpu().contiguous(), "position_ids": position_ids.cpu().contiguous()},
            inputs_pt,
        )
        torch.save({"ref_out": ref_out}, ref_out_pt)
        logger.info(f"Wrote reference cache to {tmp_dir} (inputs.pt, ref_out.pt)")

        del hf_model, ref_weights, inputs_embeds, causal_mask, ref_out, input_ids, position_ids
        _release_pytorch_cpu_ram()

        inputs_bundle = _torch_load_pt(inputs_pt)
        input_ids = inputs_bundle["input_ids"]
        position_ids_cpu = inputs_bundle["position_ids"]
        del inputs_bundle
        _release_pytorch_cpu_ram()

        logger.info("Starting TT_CCL / TtMinistral3Model (HF reference module already freed)")
        # ---- TT model construction (device ops start here — ARC WDT is reset by dispatch) ----
        tt_ccl = TT_CCL(mesh_device)
        transformation_mats = {"decode": None, "prefill": None}
        tt_model = TtMinistral3Model(
            model_args,
            mesh_device,
            tt_ccl,
            dtype,
            meta_sd,
            weight_cache_path=model_args.weight_cache_path(dtype),
            transformation_mats=transformation_mats,
        )
        logger.info("TtMinistral3Model constructed (text stack: embed → layers → norm; no lm_head)")

        # ---- TT forward pass (same ``input_ids`` / ``position_ids`` as HF text reference) ----
        tokens_4d = input_ids.reshape(1, 1, 1, -1).to(torch.int32)
        input_ids_tt = ttnn.from_torch(
            tokens_4d,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        position_ids_tt = ttnn.from_torch(
            position_ids_cpu.to(torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_out = tt_model(
            input_ids=input_ids_tt,
            position_ids=position_ids_tt,
            mode=Mode.PREFILL,
            batch_size=batch_size,
        ).last_hidden_state

        # ---- gather TT output ----
        out_last = int(tt_out.shape[-1])
        if out_last == model_args.dim:
            tt_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
        else:
            tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

        ref_bundle = _torch_load_pt(ref_out_pt)
        ref_out = ref_bundle["ref_out"]
        del ref_bundle
        _release_pytorch_cpu_ram()

        tt_torch = tt_torch.reshape(ref_out.shape)
        assert tt_torch.shape == expected_shape, f"TT text backbone output shape {tt_torch.shape} != {expected_shape}"

        # ---- PCC: HF ``Ministral3Model`` last_hidden_state vs ``TtMinistral3Model`` (text only) ----
        pcc_required = 0.99
        passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
        logger.info(comp_allclose(ref_out, tt_torch))
        logger.info(
            f"PCC (text Ministral3Model vs TtMinistral3Model, {text_cfg.num_hidden_layers} layers): {pcc_message}"
        )
        assert passing, f"PCC below {pcc_required}: {pcc_message}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
