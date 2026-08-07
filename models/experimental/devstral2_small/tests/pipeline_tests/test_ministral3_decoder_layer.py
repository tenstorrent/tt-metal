# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# PCC: HF Ministral3 decode logits vs TtDevstral2SmallModel decode logits.

from __future__ import annotations

import os
import re
import types

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from transformers.models.ministral3.modeling_ministral3 import Ministral3DecoderLayer

# Keep long decoder PCC runs focused; set TT_LOGGER_LEVEL=Warning/Debug externally when debugging TT-Metal.
os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_small.devstral_utils import apply_fp8_dequantize_compat
from models.experimental.devstral2_small.tt.pipeline.tt_devstral2_small_model import TtDevstral2SmallModel
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import ModelArgs

apply_fp8_dequantize_compat()

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"

PCC_TARGET = float(os.environ.get("MINISTRAL3_DECODER_STACK_PCC", "0.98"))
DECODE_STEPS = int(os.environ.get("MINISTRAL3_DECODER_DECODE_STEPS", "256"))
# Profile decode forward only: skip per-step to_torch (UntilizeDeviceOperation in Tracy).
PROFILE_DECODE = os.environ.get("MINISTRAL3_DECODER_PROFILE", "0").strip().lower() in ("1", "true", "yes")
_DECODE_PCC_RESULTS: dict[int, float | str] = {}


def _text_model_root(multimodal_inner):
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


def _hf_lm_head(hf_full, text_root):
    candidates = (
        hf_full,
        getattr(hf_full, "model", None),
        getattr(hf_full, "language_model", None),
        getattr(getattr(hf_full, "model", None), "language_model", None),
        text_root,
        getattr(text_root, "language_model", None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "lm_head"):
            return candidate.lm_head
    raise AttributeError("Could not find HF lm_head on Devstral model.")


def _logits_from_hidden(hf_full, text_root, hidden):
    normed = text_root.norm(hidden)
    return _hf_lm_head(hf_full, text_root)(normed).float()


def _tt_decode_logits_to_bsv(tt_logits, mesh_device, batch_size, vocab_size):
    logits_torch = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    while logits_torch.dim() > 4:
        logits_torch = logits_torch.squeeze(0)
    if logits_torch.dim() == 4:
        rows = logits_torch[:, :1, :batch_size, :vocab_size]
    elif logits_torch.dim() == 3:
        rows = logits_torch[0, :batch_size, :vocab_size]
    elif logits_torch.dim() == 2:
        rows = logits_torch[:batch_size, :vocab_size]
    else:
        raise RuntimeError(f"Unexpected TT logits shape after mesh compose: {tuple(logits_torch.shape)}")
    return rows.reshape(batch_size, 1, vocab_size)


def _pcc_value(pcc_msg) -> float | str:
    try:
        return float(pcc_msg)
    except (TypeError, ValueError):
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(pcc_msg))
        return float(match.group(0)) if match is not None else str(pcc_msg)


def _print_decode_pcc_summary(terminalreporter) -> None:
    if not _DECODE_PCC_RESULTS:
        return
    terminalreporter.write_sep("-", "Devstral text decode logits PCC")
    terminalreporter.write_line("decode_step\tPCC")
    for step, pcc in sorted(_DECODE_PCC_RESULTS.items()):
        pcc_str = f"{pcc:.6f}" if isinstance(pcc, float) else str(pcc)
        terminalreporter.write_line(f"{step}\t{pcc_str}")


@pytest.fixture(scope="module", autouse=True)
def decode_pcc_summary(pytestconfig):
    yield
    terminalreporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is not None:
        _print_decode_pcc_summary(terminalreporter)


def _hf_decoder_stack_forward(
    layers,
    rotary,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: DynamicCache,
) -> torch.Tensor:
    position_embeddings = rotary(hidden_states, position_ids=position_ids)
    cache_position = position_ids[0]
    hidden = hidden_states
    for layer in layers:
        layer_out = layer(
            hidden_states=hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
    return hidden


def _token_ids_to_tt(token_ids: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        token_ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.fixture
def trust_remote_ministral(monkeypatch):
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


def _mesh_device_param():
    mesh_env = os.environ.get("MESH_DEVICE")
    if mesh_env in {"P150": (1, 1), "BH-QB": (1, 4)}:
        return {"P150": (1, 1), "BH-QB": (1, 4)}[mesh_env]
    return int(os.environ.get("TT_MESH_WIDTH", "4"))


@torch.no_grad()
@pytest.mark.timeout(7200)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 60000000, "num_command_queues": 1}],
    indirect=True,
)
def test_ministral3_decoder_layer_decode_pcc_devstral_weights(
    mesh_device,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
):
    """Full decoder stack + final norm + LM head decode logits PCC vs HF."""
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=512,
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )
    model_args.is_distributed_norm = types.MethodType(
        lambda self, mode: model_args.is_multichip and mode == Mode.DECODE,
        model_args,
    )

    depth = int(model_args.full_model_n_layers)
    n_layers = depth
    model_args.n_layers = n_layers
    logger.info(
        f"Ministral3 decoder stack decode PCC: n_layers={n_layers}/{depth} "
        f"steps={DECODE_STEPS} pcc_required={PCC_TARGET}"
    )

    try:
        meta_state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    hf_full = model_args.cached_hf_model
    assert hf_full is not None

    text_root = _text_model_root(hf_full.model)
    rotary = text_root.rotary_emb
    rotary.eval()
    text_root.norm.eval()
    _hf_lm_head(hf_full, text_root).eval()
    hf_layers = text_root.layers[:n_layers]
    for layer in hf_layers:
        assert isinstance(layer, Ministral3DecoderLayer), type(layer)
        layer.eval()

    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtDevstral2SmallModel(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        model_args=model_args,
        meta_state_dict=meta_state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        transformation_mats={"decode": None, "prefill": None},
        configuration=model_args,
        vision_config=hf_full.config.vision_config,
    )

    hf_cache = DynamicCache()
    generation_start_pos = 0
    all_pass = True

    # Direct decode at pos=0 seeds the KV cache; PCC is checked on the next 32 decode steps.
    pcc_steps = [] if PROFILE_DECODE else list(range(1, DECODE_STEPS + 1))
    if PROFILE_DECODE and DECODE_STEPS > 0:
        pcc_steps = [DECODE_STEPS]

    for step in range(DECODE_STEPS + 1):
        pos = generation_start_pos + step
        token_ids = torch.randint(0, model_args.vocab_size, (batch_size, 1), dtype=torch.long)
        pt_decode_in = text_root.embed_tokens(token_ids)
        position_ids = torch.full((batch_size, 1), pos, dtype=torch.long)

        ref_hidden = _hf_decoder_stack_forward(hf_layers, rotary, pt_decode_in, position_ids, hf_cache)

        token_ids_tt = _token_ids_to_tt(token_ids, mesh_device)
        tt_logits = tt_model.forward_decode(token_ids_tt, pos)
        ttnn.deallocate(token_ids_tt)

        if step in pcc_steps:
            ttnn.synchronize_device(mesh_device)
            ref_logits = _logits_from_hidden(hf_full, text_root, ref_hidden)
            tt_logits_torch = _tt_decode_logits_to_bsv(tt_logits, mesh_device, batch_size, model_args.vocab_size)

            passing, pcc_value = comp_pcc(ref_logits, tt_logits_torch, PCC_TARGET)
            _DECODE_PCC_RESULTS[step - 1] = _pcc_value(pcc_value)
            logger.info(comp_allclose(ref_logits, tt_logits_torch))
            logger.info(f"Decode step {step - 1} pos={pos} n_layers={n_layers} logits PCC: {pcc_value}")
            if not passing:
                all_pass = False
        ttnn.deallocate(tt_logits)

    if PROFILE_DECODE:
        logger.info("MINISTRAL3_DECODER_PROFILE=1: PCC on final step only (no per-step to_torch in Tracy hot path).")

    assert all_pass, f"Decode logits PCC below {PCC_TARGET} for one or more steps"
