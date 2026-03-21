# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Coder-Next on TTNN (MoE + gated attention). Runs on T3K only."""

import os
import time
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

import ttnn
from torch import nn
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextRMSNorm,
    Qwen3NextSparseMoeBlock,
)

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.attention import TTNNQwen3NextGatedAttention
from models.experimental.tt_symbiote.modules.gated_deltanet import TTNNGatedDeltaNet
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.modules.moe import TTNNQwen3MoE
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedQwen3NextRMSNorm
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

QWEN3_MODEL_ID = "Qwen/Qwen3-Coder-Next"
REPETITION_PENALTY = 1.28

MESH_DEVICE = (os.environ.get("MESH_DEVICE") or "T3K").upper()
MESH_SHAPE_T3K = (1, 8)


class StopOnLowDiversityGeneratedTail(StoppingCriteria):
    """End generation when the last window of new tokens collapses to a 2-type loop (common under hybrid TTNN greedy decode)."""

    def __init__(self, prompt_len: int, window: int = 48, max_distinct: int = 2, min_new_tokens: int = 64):
        self.prompt_len = int(prompt_len)
        self.window = int(window)
        self.max_distinct = int(max_distinct)
        self.min_new_tokens = int(min_new_tokens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len < self.min_new_tokens:
            return torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)
        w = min(self.window, gen_len)
        tail = input_ids[:, -w:]
        out = [int(torch.unique(tail[b]).numel()) <= self.max_distinct for b in range(input_ids.shape[0])]
        return torch.tensor(out, device=input_ids.device, dtype=torch.bool)


def _merge_generate_kw(base: Dict[str, Any], prompt_len: int) -> Dict[str, Any]:
    return {
        **base,
        "stopping_criteria": StoppingCriteriaList([StopOnLowDiversityGeneratedTail(prompt_len)]),
    }


def _clone_model_inputs(inputs: dict) -> dict:
    out = {}
    for k, v in inputs.items():
        out[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return out


def _generate_kw(model, tokenizer) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"repetition_penalty": REPETITION_PENALTY, "do_sample": False}
    eos = getattr(model.config, "eos_token_id", None) or (tokenizer.eos_token_id if tokenizer else None)
    if eos is not None:
        kw["eos_token_id"] = eos
    pad = getattr(model.config, "pad_token_id", None) or (tokenizer.pad_token_id if tokenizer else None)
    if pad is not None:
        kw["pad_token_id"] = pad
    return kw


def _get_cached_model_path():
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
            x in msg
            for x in (
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
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE_T3K], indirect=True)
@pytest.mark.parametrize("max_new_tokens", [128])
@pytest.mark.slow
def test_qwen3_coder_next(mesh_device, max_new_tokens):
    if MESH_DEVICE != "T3K":
        pytest.skip(f"test_qwen3_coder_next runs only on T3K (MESH_DEVICE={os.environ.get('MESH_DEVICE')!r})")

    tokenizer, model = _load_qwen3_model()

    nn_to_ttnn = {
        Qwen3NextSparseMoeBlock: TTNNQwen3MoE,
        Qwen3NextAttention: TTNNQwen3NextGatedAttention,
        Qwen3NextGatedDeltaNet: TTNNGatedDeltaNet,
        Qwen3NextRMSNorm: TTNNDistributedQwen3NextRMSNorm,
        nn.Linear: TTNNLinearIColShardedWRowSharded,
    }

    messages = [
        {
            "role": "user",
            "content": "Write a quick sort algorithm.",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    dev = next(model.parameters()).device

    def to_dev(v):
        if not isinstance(v, torch.Tensor):
            return v
        v = v.to(dev)
        if v.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            return v
        return v.to(torch.bfloat16)

    inputs = {k: to_dev(v) for k, v in inputs.items()}
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.config.eos_token_id

    prompt_len = int(inputs["input_ids"].shape[-1])

    all_modules = register_module_replacement_dict(
        model, nn_to_ttnn, model_config=None, exclude_replacement={"lm_head"}
    )
    set_device(model, mesh_device, dump_visualization=False)

    t0 = time.perf_counter()
    for _, m in tqdm(all_modules.items(), desc="Weight prep (per TTNN module)"):
        m.preprocess_weights()
        m.move_weights_to_device()
    setup_s = time.perf_counter() - t0

    model.eval()
    torch.set_grad_enabled(False)

    DispatchManager.clear_timings()
    gen_kw = _merge_generate_kw(_generate_kw(model, tokenizer), prompt_len)
    t1 = time.perf_counter()
    outputs = model.generate(
        **_clone_model_inputs(inputs),
        max_new_tokens=max_new_tokens,
        use_cache=True,
        **gen_kw,
    )
    gen_s = time.perf_counter() - t1

    output_ids = outputs[0][prompt_len:].tolist()
    print(
        f"Qwen3-Coder-Next output ({len(output_ids)} new tokens):\n{tokenizer.decode(output_ids, skip_special_tokens=True)}"
    )

    e2e_tok_s = len(output_ids) / gen_s if gen_s > 0 else 0.0
    print("\n--- Generation performance ---")
    print(f"  Weight prep: {setup_s:.2f} s")
    print(f"  Generate ({len(output_ids)} new tokens): {gen_s:.3f} s")
    print(f"  Prompt: {prompt_len} tokens")
    print(f"  E2E: {e2e_tok_s:.3f} tok/s")
    if len(output_ids) >= max_new_tokens:
        print(f"  Note: reached max_new_tokens={max_new_tokens}")
    print("---\n")

    DispatchManager.save_stats_to_file("qwen3_coder_next_timing_stats.csv")
