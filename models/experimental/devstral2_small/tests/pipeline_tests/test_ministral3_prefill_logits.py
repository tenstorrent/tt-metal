# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# PCC: HF Ministral3 prefill last-token logits vs TtDevstral2SmallModel prefill last-token logits.

from __future__ import annotations

import bz2
import contextlib
import gc
import os
import re
import types
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.modeling_ministral3 import Ministral3DecoderLayer

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_small.devstral_utils import (
    apply_fp8_dequantize_compat,
    host_input_ids_to_tt_replicated,
    tt_forward_prefill_from_device_ids,
)
from models.experimental.devstral2_small.tt.pipeline.tt_devstral2_small_model import TtDevstral2SmallModel
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import ModelArgs

apply_fp8_dequantize_compat()

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
PROMPT_FILE = Path("models/tt_transformers/tests/tale-of-two-cities.txt.bz2")
PCC_TARGET = float(os.environ.get("MINISTRAL3_PREFILL_LOGITS_PCC", "0.98"))
_PREFILL_PCC_RESULTS: dict[int, float | str] = {}


def _mesh_device_param():
    mesh_env = os.environ.get("MESH_DEVICE")
    if mesh_env in {"P150": (1, 1), "BH-QB": (1, 4)}:
        return {"P150": (1, 1), "BH-QB": (1, 4)}[mesh_env]
    try:
        return ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
    except Exception:
        return int(os.environ.get("TT_MESH_WIDTH", "4"))


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


def _prompt_text(path: Path) -> str:
    with bz2.open(path, "rt", encoding="utf-8") as f:
        return f.read()


def _prompt_ids(model_args, seq_len: int) -> torch.Tensor:
    ids = model_args.encode_prompt(_prompt_text(PROMPT_FILE), instruct=False)
    assert len(ids) >= seq_len, f"{PROMPT_FILE} tokenized to {len(ids)} tokens, shorter than seq_len={seq_len}"
    return torch.tensor(ids[:seq_len], dtype=torch.long).unsqueeze(0)


def _pad_token_id(model_args) -> int:
    tokenizer = model_args.tokenizer
    for attr in ("pad_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    return 0


def _hf_prefill_last_token_logits(hf_full, text_root, inputs_embeds, position_ids, text_cfg, n_layers):
    position_embeddings = text_root.rotary_emb(inputs_embeds, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=text_cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    hidden = inputs_embeds
    for layer in text_root.layers[:n_layers]:
        layer_out = layer(
            hidden_states=hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
    return _hf_lm_head(hf_full, text_root)(text_root.norm(hidden[:, -1:, :])).float()


def _pcc_value(pcc_msg) -> float | str:
    try:
        return float(pcc_msg)
    except (TypeError, ValueError):
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(pcc_msg))
        return float(match.group(0)) if match is not None else str(pcc_msg)


def _print_prefill_pcc_summary(terminalreporter) -> None:
    if not _PREFILL_PCC_RESULTS:
        return
    terminalreporter.write_sep("-", "Devstral text prefill logits PCC")
    terminalreporter.write_line("seq_len\tPCC")
    for seq_len, pcc in sorted(_PREFILL_PCC_RESULTS.items()):
        pcc_str = f"{pcc:.6f}" if isinstance(pcc, float) else str(pcc)
        terminalreporter.write_line(f"{seq_len}\t{pcc_str}")


@pytest.fixture(scope="module", autouse=True)
def prefill_pcc_summary(pytestconfig):
    yield
    terminalreporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is not None:
        _print_prefill_pcc_summary(terminalreporter)


def _tt_prefill_last_token_logits(tt_hidden, last_token_index, mesh_device, model_args, tt_lm_head):
    block_start = (last_token_index // 32) * 32
    h_block = ttnn.slice(
        tt_hidden,
        (0, 0, block_start, 0),
        (1, 1, block_start + 32, tt_hidden.shape[-1]),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    lm_head_input_mem_cfg = model_args.get_lm_head_input_mem_config(Mode.PREFILL, None)
    if lm_head_input_mem_cfg is not None and lm_head_input_mem_cfg.is_sharded():
        h_block_sharded = ttnn.interleaved_to_sharded(h_block, lm_head_input_mem_cfg)
        ttnn.deallocate(h_block)
        h_block = h_block_sharded

    logits = ttnn.to_memory_config(tt_lm_head(h_block), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    try:
        logits_torch = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
        while logits_torch.dim() > 4:
            logits_torch = logits_torch.squeeze(0)
        row = last_token_index % 32
        if logits_torch.dim() == 4:
            out = logits_torch[:, 0, row : row + 1, : model_args.vocab_size]
        elif logits_torch.dim() == 3:
            out = logits_torch[:, row : row + 1, : model_args.vocab_size]
        elif logits_torch.dim() == 2:
            out = logits_torch[row : row + 1, : model_args.vocab_size].unsqueeze(0)
        else:
            raise RuntimeError(f"Unexpected TT logits shape after mesh compose: {tuple(logits_torch.shape)}")
        return out.reshape(1, 1, model_args.vocab_size)
    finally:
        ttnn.deallocate(logits)


def _clear_memory_after_seq_len() -> None:
    for attr in vars(ModelArgs).values():
        cache_clear = getattr(attr, "cache_clear", None)
        if cache_clear is not None:
            with contextlib.suppress(Exception):
                cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@pytest.fixture(autouse=True)
def trim_host_memory_after_case():
    yield
    _clear_memory_after_seq_len()


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


@torch.no_grad()
@pytest.mark.timeout(7200)
@pytest.mark.models_performance_bare_metal
@pytest.mark.usefixtures("trust_remote_ministral", "ensure_gc")
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("seq_len", (128, 256, 512, 1024, 4096, 8192), ids=["128", "256", "512", "1k", "4k", "8k"])
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 60000000, "num_command_queues": 1}],
    indirect=True,
)
def test_ministral3_prefill_last_token_logits_pcc(mesh_device, seq_len, batch_size, monkeypatch):
    """Full text stack prefill last-token logits PCC (all layers + final norm + LM head)."""
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)
    monkeypatch.setenv("TT_MINISTRAL3_TEXT_REDUCED_PRECISION", "0")
    monkeypatch.setenv("TT_MINISTRAL3_SHORT_PREFILL_L1_WIDTH_MM", "0")

    dtype = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max(4096, seq_len),
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )
    model_args.is_distributed_norm = types.MethodType(lambda self, mode: False, model_args)
    depth = int(model_args.full_model_n_layers)
    model_args.n_layers = depth

    try:
        meta_state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    hf_full = model_args.cached_hf_model
    assert hf_full is not None
    text_root = _text_model_root(hf_full.model)
    text_cfg = model_args.hf_config.text_config
    text_root.rotary_emb.eval()
    text_root.norm.eval()
    _hf_lm_head(hf_full, text_root).eval()
    for layer in text_root.layers[:depth]:
        assert isinstance(layer, Ministral3DecoderLayer), type(layer)
        layer.eval()

    input_ids = _prompt_ids(model_args, seq_len)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    pt_prefill = text_root.embed_tokens(input_ids)

    logger.info("Running HF prefill logits reference...")
    ref_logits = _hf_prefill_last_token_logits(hf_full, text_root, pt_prefill, position_ids, text_cfg, depth)

    logger.info("Running TT prefill logits...")
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
    input_ids_tt = host_input_ids_to_tt_replicated(mesh_device, input_ids)
    try:
        tt_hidden = tt_forward_prefill_from_device_ids(
            input_ids_tt,
            seq_len,
            _pad_token_id(model_args),
            mesh_device,
            tt_model.language_model,
            model_args,
        )
    finally:
        ttnn.deallocate(input_ids_tt)

    try:
        tt_logits = _tt_prefill_last_token_logits(
            tt_hidden, seq_len - 1, mesh_device, model_args, tt_model.lm_head
        ).float()
    finally:
        ttnn.deallocate(tt_hidden)

    passing, pcc_msg = comp_pcc(ref_logits, tt_logits, PCC_TARGET)
    _PREFILL_PCC_RESULTS[int(seq_len)] = _pcc_value(pcc_msg)
    logger.info(comp_allclose(ref_logits, tt_logits))
    logger.info(f"Prefill last-token logits PCC (seq_len={seq_len}, n_layers={depth}): {pcc_msg}")

    assert passing, f"Prefill logits PCC below {PCC_TARGET} for seq_len={seq_len}: {pcc_msg}"
