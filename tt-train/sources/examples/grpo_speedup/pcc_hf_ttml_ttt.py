#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pairwise PCC across hf / ttml / tt-transformers on GSM8K prompts.

Single-process, sequential script. Runs the SAME ``NUM_PROMPTS`` prompts
from the GSM8K (``main`` / ``train``) split through three implementations
of the same Llama checkpoint and reports, per pair:

    * the per-prompt Pearson correlation (PCC) on the last-position logits
    * the minimum PCC observed across all prompts (the "worst case")

The three implementations are:

    1. hf   -- HuggingFace torch CPU forward, fp32 (gold reference)
    2. ttml -- tt-train Python bindings via LlamaGRPOCompleter._forward
    3. ttt  -- tt-transformers Transformer in DECODE mode

ttml and tt-transformers share one mesh device (opened once with
fabric_config=FABRIC_2D), built sequentially in the same process. No
subprocesses, no CLI arguments. Edit the constants below to change the
prompt count or model.
"""

from __future__ import annotations

import os

# Silence the noisy ttnn::tilize "Using input shard spec for output tensor
# because the legacy sharded optimized program factory is being used" warning
# (and other tt-metal log_warning lines). Must be set before any tt-metal /
# ttnn import so the C++ logger picks it up at init. Use setdefault so a
# shell-provided TT_LOGGER_LEVEL still wins.
os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

import sys
from pathlib import Path
from typing import List

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_PROMPTS = 7473
GSM8K_SPLIT = "train"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened arrays, in fp64."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_gsm8k_prompts(num_prompts: int, tokenizer) -> List[List[int]]:
    """Tokenize the first ``num_prompts`` GSM8K questions with ``tokenizer``."""
    import datasets

    print(f"[data] loading GSM8K split={GSM8K_SPLIT!r} (need {num_prompts} prompts)")
    ds = datasets.load_dataset("gsm8k", "main", split=GSM8K_SPLIT)
    questions = [ds[i]["question"] for i in range(num_prompts)]

    prompt_ids: List[List[int]] = []
    for q in questions:
        ids = tokenizer.encode(q, add_special_tokens=True)
        prompt_ids.append(ids)

    lens = [len(p) for p in prompt_ids]
    print(
        f"[data] tokenized {len(prompt_ids)} prompts  "
        f"len min/median/max = {min(lens)}/{int(np.median(lens))}/{max(lens)}"
    )
    return prompt_ids


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def run_hf_all(prompts: List[List[int]]) -> List[np.ndarray]:
    """HuggingFace torch CPU fp32 forward for every prompt. Loaded once."""
    import torch
    from transformers import AutoModelForCausalLM

    print(f"[hf] loading {MODEL_ID} (fp32, CPU)")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    out: List[np.ndarray] = []
    with torch.no_grad():
        for i, prompt_ids in enumerate(prompts):
            ids = torch.tensor([prompt_ids], dtype=torch.long)
            logits = model(ids).logits[0, -1, :].float().numpy().astype(np.float32)
            out.append(logits)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[hf]  {i + 1}/{len(prompts)}  N={len(prompt_ids)}  argmax={int(logits.argmax())}")

    print(f"[hf] done  {len(out)} prompts")
    return out


def run_ttml_all(prompts: List[List[int]], config_path: str):
    """ttml forward for every prompt. Builds the completer once, returns it.

    The returned completer keeps the AutoContext mesh device open; reuse it
    for the tt-transformers stage instead of opening a second one.
    """
    import ttnn

    from ttml.common.config import DeviceConfig, TrainingConfig, get_model_config, load_config
    from ttml.common.utils import no_grad

    from utils.llama_completer import LlamaCompletionCtx, LlamaGRPOCompleter

    print(f"[ttml] loading {MODEL_ID} via {config_path}")
    raw = load_config(config_path)
    training_config = TrainingConfig(raw)
    device_config = DeviceConfig(raw)
    transformer_config = get_model_config(training_config.model_config)

    completer = LlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(max_tokens_to_complete=1, temperature=0.0),
        transformer_config=transformer_config,
        device_config=device_config,
        model_source=MODEL_ID,
    )
    completer._model.eval()

    V = len(completer.tokenizer)
    out: List[np.ndarray] = []
    with no_grad():
        for i, prompt_ids in enumerate(prompts):
            N = len(prompt_ids)
            arr = np.array([prompt_ids], dtype=np.uint32)
            logits_tt = completer._forward(arr, [0], 1)
            logits_np = logits_tt.to_numpy(ttnn.DataType.FLOAT32)
            flat = logits_np.reshape(-1, logits_np.shape[-1])
            last = flat[N - 1, :V].astype(np.float32)
            out.append(last)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[ttml] {i + 1}/{len(prompts)}  N={N}  argmax={int(last.argmax())}")

    print(f"[ttml] done  {len(out)} prompts")
    return out, completer


def run_ttt_all(mesh_device, prompts: List[List[int]]) -> List[np.ndarray]:
    """tt-transformers Transformer in DECODE mode for every prompt.

    Builds the model once with ``max_seq_len`` large enough to fit the
    longest prompt, then DECODEs each prompt from position 0. Re-decoding
    from position 0 overwrites the per-prompt prefix in the paged KV cache;
    attention only reads positions <= current_pos so stale tail entries
    from longer earlier prompts are never read.
    """
    import torch

    os.environ["HF_MODEL"] = MODEL_ID

    import ttnn

    from models.tt_transformers.tt.common import Mode, PagedAttentionConfig
    from models.tt_transformers.tt.model import Transformer
    from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

    batch_size = 1
    longest = max(len(p) for p in prompts)
    max_seq_len = max(256, ((longest + 31) // 32) * 32)

    print(f"[ttt] building ModelArgs / loading state dict (max_seq_len={max_seq_len})")
    model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=batch_size,
        optimizations=lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name),
        max_seq_len=max_seq_len,
        cache_hf=True,
    )

    state_dict = model_args.load_state_dict()

    paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        batch_size,
        paged_attention_config.max_num_blocks // batch_size,
    )
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    print(f"[ttt] building Transformer (n_layers={model_args.n_layers})")
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
    )

    reference_model = model_args.reference_transformer(load_checkpoint=True)
    embd = model_args.reference_embedding(reference_model)
    prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{prefix}tok_embeddings.weight"]})
    del reference_model

    mesh_composer = ttnn.ConcatMesh2dToTensor(
        mesh_device,
        dims=(1, -1),
        mesh_shape=model_args.cluster_shape,
    )

    out: List[np.ndarray] = []
    for p_idx, prompt_ids in enumerate(prompts):
        N = len(prompt_ids)
        encoded = torch.tensor([prompt_ids])
        last_logits = None

        for i in range(N):
            decode_input = embd(encoded[:, i].unsqueeze(0)).view(batch_size, 1, -1)
            decode_input = model_args.prepare_residual_tensor_decode(
                decode_input,
                model_args.get_residual_mem_config(Mode.DECODE),
            )

            current_pos = torch.tensor([i] * batch_size)
            current_pos_tensor = ttnn.from_torch(
                current_pos,
                device=mesh_device,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device,
                    dims=(None, None),
                    mesh_shape=model_args.cluster_shape,
                ),
            )

            rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)

            tt_out = tt_model(
                decode_input,
                current_pos_tensor,
                rot_mats_global=rot_mats,
                mode=Mode.DECODE,
                page_table=page_table_tt,
            )

            out_torch = (
                ttnn.to_torch(tt_out, mesh_composer=mesh_composer)
                .permute(2, 1, 0, 3)
                .squeeze(2)[:batch_size, 0:1, : model_args.vocab_size]
            )
            ttnn.deallocate(tt_out)
            last_logits = out_torch[0, 0, :].float().numpy()

        last = last_logits.astype(np.float32)
        out.append(last)
        if (p_idx + 1) % 10 == 0 or p_idx == 0:
            print(f"[ttt]  {p_idx + 1}/{len(prompts)}  N={N}  argmax={int(last.argmax())}")

    print(f"[ttt] done  {len(out)} prompts")
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


PAIRS = [("hf", "ttml"), ("hf", "ttt"), ("ttml", "ttt")]


def report(prompts: List[List[int]], logits_by_backend: dict) -> None:
    P = len(prompts)
    backends = list(logits_by_backend.keys())
    assert all(len(logits_by_backend[b]) == P for b in backends), "mismatched prompt counts"

    per_prompt: dict = {pair: [] for pair in PAIRS}
    for i in range(P):
        V = min(logits_by_backend[b][i].shape[0] for b in backends)
        trimmed = {b: logits_by_backend[b][i][:V] for b in backends}
        for a, b in PAIRS:
            per_prompt[(a, b)].append(pcc(trimmed[a], trimmed[b]))

    print()
    print(f"=== Per-prompt pairwise PCC on last-position logits  (P={P}) ===")
    header = "  idx |    N |   PCC(hf,ttml) |    PCC(hf,ttt) |  PCC(ttml,ttt)"
    print(header)
    print("-" * len(header))
    for i in range(P):
        N = len(prompts[i])
        scores = [per_prompt[pair][i] for pair in PAIRS]
        print(f"  {i:3d} | {N:4d} |   {scores[0]:.6f}   |   {scores[1]:.6f}   |   {scores[2]:.6f}")

    print()
    print("=== Summary across prompts (per pair) ===")
    for pair in PAIRS:
        vals = np.asarray(per_prompt[pair], dtype=np.float64)
        worst_idx = int(np.argmin(vals))
        print(
            f"PCC({pair[0]}, {pair[1]}):  "
            f"min={vals.min():.6f} (prompt {worst_idx})  "
            f"median={np.median(vals):.6f}  "
            f"mean={vals.mean():.6f}  "
            f"max={vals.max():.6f}"
        )

    print()
    print("=== Worst-case pairwise PCC (minimum across all prompts) ===")
    for pair in PAIRS:
        vals = np.asarray(per_prompt[pair], dtype=np.float64)
        worst_idx = int(np.argmin(vals))
        tag = "  <-- noise floor for the bridge" if pair == ("ttml", "ttt") else ""
        print(f"min PCC({pair[0]}, {pair[1]}) = {vals.min():.6f}  (prompt {worst_idx}){tag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = load_gsm8k_prompts(NUM_PROMPTS, tokenizer)

    print("\n>>> stage hf")
    hf_logits = run_hf_all(prompts)

    # Open one mesh device; ttml + tt-transformers share it.
    # tt-transformers needs fabric_config set BEFORE any mesh open; ttml's
    # AutoContext doesn't do that for single device, so we set it ourselves
    # here. Both backends will then run on the same mesh device.
    import ttnn

    print("\n>>> stage ttml")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    config_path = os.path.join(REPO_ROOT, TTML_CONFIG_REL)

    completer = None
    try:
        ttml_logits, completer = run_ttml_all(prompts, config_path)
        mesh_device = completer._mesh_device

        print("\n>>> stage ttt")
        ttt_logits = run_ttt_all(mesh_device, prompts)
    finally:
        completer = None

    print("\n>>> stage compare")
    report(
        prompts,
        {"hf": hf_logits, "ttml": ttml_logits, "ttt": ttt_logits},
    )


if __name__ == "__main__":
    main()
