# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

import ttnn
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from torch import nn
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.attention import TTNNQwen3NextGatedAttention
from models.experimental.tt_symbiote.modules.gated_deltanet import TTNNGatedDeltaNet
from models.experimental.tt_symbiote.modules.moe import TTNNQwen3MoE
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


QWEN3_MODEL_ID = "Qwen/Qwen3-Coder-Next"

# When > 0, stop after N identical generated token ids in a row (can truncate answers). Default off.
QWEN3_SAME_TOKEN_STREAK_STOP = 0
QWEN3_RNG_SEED = 42


def _model_max_context_length(model, tokenizer) -> int:
    """Best-effort maximum total sequence length for this checkpoint."""
    for attr in ("max_position_embeddings", "model_max_length"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    if tokenizer is not None:
        tml = getattr(tokenizer, "model_max_length", None)
        if isinstance(tml, int) and tml > 0 and tml < 10**9:
            return tml
    return 32768


def _compute_max_new_tokens(model, tokenizer, prompt_len: int) -> int:
    """Allow generation until EOS within remaining context (no fixed short cap)."""
    max_ctx = _model_max_context_length(model, tokenizer)
    reserve = 8
    available = max_ctx - int(prompt_len) - reserve
    if available < 1:
        pytest.fail(f"Prompt length {prompt_len} meets or exceeds model context {max_ctx}; cannot generate.")
    return available


# Run only on T3K (symbiote DeviceArch). Uses MESH_DEVICE env if set, else T3K.
MESH_DEVICE = (os.environ.get("MESH_DEVICE") or "T3K").upper()
MESH_SHAPE_T3K = (1, 8)


class StopOnConsecutiveSameToken(StoppingCriteria):
    """Stop when the same token id is generated ``streak`` times in a row (kills ``you you you`` tails)."""

    def __init__(self, streak: int = 6):
        self.streak = max(2, int(streak))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        if input_ids.shape[1] < self.streak:
            return torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)
        tail = input_ids[:, -self.streak :]
        done = (tail == tail[:, :1]).all(dim=1)
        return done


def _stopping_criteria_list() -> StoppingCriteriaList:
    """Optional degenerate-tail stop; disabled when ``QWEN3_SAME_TOKEN_STREAK_STOP <= 0``."""
    if QWEN3_SAME_TOKEN_STREAK_STOP <= 0:
        return StoppingCriteriaList([])
    return StoppingCriteriaList([StopOnConsecutiveSameToken(QWEN3_SAME_TOKEN_STREAK_STOP)])


def _merge_generate_kw(base: Dict[str, Any]) -> Dict[str, Any]:
    sc = _stopping_criteria_list()
    if len(sc) == 0:
        return base
    return {**base, "stopping_criteria": sc}


def _clone_model_inputs(inputs: dict) -> dict:
    """Deep-enough copy so ``generate`` / warmup cannot alias or resize shared prompt tensors."""
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def _seed_torch_for_generate() -> None:
    """Reset PyTorch RNG before ``generate`` when ``do_sample=True`` (no ``generator=`` kwarg on this model)."""
    torch.manual_seed(QWEN3_RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(QWEN3_RNG_SEED)


def _generate_extra_kw(model, tokenizer) -> Dict[str, Any]:
    """Kwargs for ``generate`` from ``model.generation_config`` / tokenizer (no env overrides)."""
    kw: Dict[str, Any] = {}
    gc = model.generation_config

    eos = getattr(model.config, "eos_token_id", None)
    if eos is None and tokenizer is not None:
        eos = tokenizer.eos_token_id
    if eos is not None:
        kw["eos_token_id"] = eos
    pad = getattr(model.config, "pad_token_id", None)
    if pad is None and tokenizer is not None:
        pad = tokenizer.pad_token_id
    if pad is not None:
        kw["pad_token_id"] = pad

    rp = getattr(gc, "repetition_penalty", None)
    if rp is not None and float(rp) != 1.0:
        kw["repetition_penalty"] = float(rp)

    ngram = getattr(gc, "no_repeat_ngram_size", None)
    if ngram is not None and int(ngram) > 0:
        kw["no_repeat_ngram_size"] = int(ngram)

    do_sample = bool(getattr(gc, "do_sample", False))
    kw["do_sample"] = do_sample
    if do_sample:
        for name, key in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
        ):
            val = getattr(gc, key, None)
            if val is not None:
                kw[name] = val

    return kw


def _get_cached_model_path():
    """Resolve HF cache snapshot path for Qwen3-Coder-Next (avoids network)."""
    if hub := os.environ.get("HF_HUB_CACHE"):
        cache_root = Path(hub) / f"models--{QWEN3_MODEL_ID.replace('/', '--')}"
    else:
        base = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        cache_root = Path(base) / "hub" / f"models--{QWEN3_MODEL_ID.replace('/', '--')}"
    snapshots = cache_root / "snapshots"
    if not snapshots.exists():
        return None
    dirs = sorted(snapshots.iterdir())
    return str(dirs[0]) if dirs else None


def _load_qwen3_model():
    """Load tokenizer and model. Uses HF cache if available to avoid network."""
    try:
        cached = _get_cached_model_path()
        load_kw = dict(trust_remote_code=True, torch_dtype="auto", device_map="auto")
        if cached:
            tokenizer = AutoTokenizer.from_pretrained(cached, trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(cached, local_files_only=True, **load_kw)
            return tokenizer, model
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(QWEN3_MODEL_ID, **load_kw)
        return tokenizer, model
    except ModuleNotFoundError as e:
        if "triton" in str(e).lower():
            pytest.skip("Qwen3-Coder-Next requires triton for loading")
        raise
    except Exception as e:
        msg = str(e).lower()
        if any(
            kw in msg
            for kw in (
                "network is unreachable",
                "name resolution",
                "connection",
                "protobuf",
                "does not appear to have",
            )
        ):
            pytest.skip(f"Model unavailable: {e}")
        raise


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_SHAPE_T3K],
    indirect=True,
)
def test_qwen3_coder_next(mesh_device):
    """Test Qwen3-Coder-Next model with TTNN acceleration (MoE + Gated Attention). Runs only on T3K."""
    if MESH_DEVICE != "T3K":
        pytest.skip(f"test_qwen3_coder_next runs only on T3K (MESH_DEVICE={os.environ.get('MESH_DEVICE')!r})")
    tokenizer, model = _load_qwen3_model()

    # All Linears → sharded TTNN except lm_head (always PyTorch for correct full-vocab logits).
    # Still PyTorch: embed_tokens, Qwen3NextRMSNorm, Qwen3NextRotaryEmbedding.
    nn_to_ttnn = {
        Qwen3NextSparseMoeBlock: TTNNQwen3MoE,
        Qwen3NextAttention: TTNNQwen3NextGatedAttention,
        Qwen3NextGatedDeltaNet: TTNNGatedDeltaNet,
        nn.Linear: TTNNLinearIColShardedWRowSharded,
    }
    exclude_replacement = {"lm_head"}

    messages = [
        {
            "role": "user",
            "content": "Write a Python function to calculate fibonacci numbers using dynamic programming.",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    def _to_device(v):
        if not isinstance(v, torch.Tensor):
            return v
        device = next(model.parameters()).device
        v = v.to(device)
        if v.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            return v
        return v.to(torch.bfloat16)

    inputs = {k: _to_device(v) for k, v in inputs.items()}

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.config.eos_token_id

    all_modules = register_module_replacement_dict(
        model, nn_to_ttnn, model_config=None, exclude_replacement=exclude_replacement
    )

    set_device(model, mesh_device, dump_visualization=False)

    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead

    warm_in = _clone_model_inputs(inputs)
    wk = _merge_generate_kw(_generate_extra_kw(model, tokenizer))
    if wk.get("do_sample", False):
        _seed_torch_for_generate()
    model.generate(**warm_in, max_new_tokens=2, use_cache=True, **wk)

    DispatchManager.clear_timings()
    prompt_len = int(inputs["input_ids"].shape[-1])
    max_new = _compute_max_new_tokens(model, tokenizer, prompt_len)
    main_in = _clone_model_inputs(inputs)
    mk = _merge_generate_kw(_generate_extra_kw(model, tokenizer))
    if mk.get("do_sample", False):
        _seed_torch_for_generate()
    outputs = model.generate(**main_in, max_new_tokens=max_new, use_cache=True, **mk)

    output_ids = outputs[0][prompt_len:].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Qwen3-Coder-Next output ({len(output_ids)} new tokens):\n{content}")
    DispatchManager.save_stats_to_file("qwen3_coder_next_timing_stats.csv")
