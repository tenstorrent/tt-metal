# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared full-depth, real-weight logits PCC harness for prefill and decode vs HuggingFace.
Prompt length must be a multiple of the GDN chunk size 128. Floors are regression detectors."""

import gc
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole, run_for_wormhole_b0_or_blackhole
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

# Mesh from MESH_DEVICE. A 9B on Wormhole needs the 2-chip N300.
MESH_SHAPE = {
    "P150": (1, 1),
    "P150x4": (1, 4),
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))
_MULTI = MESH_SHAPE != (1, 1)

# Multi-device needs FABRIC_1D and l1_small for GDN conv1d. Wormhole single-device must not reserve it.
_L1_SMALL = GDN_CONV1D_L1_SMALL_SIZE if (_MULTI or is_blackhole()) else 4096
DEVICE_PARAMS = [
    {
        "l1_small_size": _L1_SMALL,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {}),
    }
]

BLOCK_SIZE = 64
# 32 blocks, a multiple of 32 so chunked-SDPA page-table alignment holds.
NUM_BLOCKS = 32

PROMPT_LEN = int(os.environ.get("QWEN36_FULL_DEPTH_PROMPT_LEN", "128"))
DECODE_STEPS = int(os.environ.get("QWEN36_FULL_DEPTH_DECODE_STEPS", "5"))

# Deterministic text, long enough for any prompt these tests use.
_PROMPT_TEXT = (
    "The history of computing hardware spans several distinct eras. Mechanical calculators gave way to "
    "relay machines, relays to vacuum tubes, tubes to discrete transistors, and transistors to the "
    "integrated circuit. Each transition changed not only how fast a machine could compute but what kinds "
    "of problems people thought were worth computing at all. The stored-program architecture, in which "
    "instructions and data share one memory, made general-purpose machines practical and turned "
    "programming into a discipline of its own. Modern accelerators return to an older idea: many simple "
    "processing elements operating in parallel on regular data, fed by a memory hierarchy carefully "
    "arranged so that the arithmetic units are rarely idle. The capital of France is "
) * 4


def parametrize_full_depth():
    """Env-selected mesh and device params, shared so the two tests cannot diverge."""

    def decorator(fn):
        fn = pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)(fn)
        fn = pytest.mark.parametrize(
            "mesh_device", [pytest.param(MESH_SHAPE, id=f"{MESH_SHAPE[0]}x{MESH_SHAPE[1]}")], indirect=True
        )(fn)
        return run_for_wormhole_b0_or_blackhole()(fn)

    return decorator


def build_full_depth_model(mesh_device, *, max_seq_len=None, prompt_len=None):
    """Full-depth TT model. Depth is asserted so a truncated stack cannot pass quietly."""
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=1, max_seq_len=max_seq_len or NUM_BLOCKS * BLOCK_SIZE
    )
    args = model.args
    num_hidden_layers = args.hf_config.get_text_config().num_hidden_layers
    assert args.n_layers == num_hidden_layers == len(model.layers), (
        f"not a full-depth stack: n_layers={args.n_layers}, len(model.layers)={len(model.layers)}, "
        f"checkpoint has {num_hidden_layers}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.CKPT_DIR, trust_remote_code=True)
    token_ids = _build_prompt(tokenizer, prompt_len or PROMPT_LEN)
    logger.info(
        f"Full-depth harness: {args.CKPT_DIR} — {args.n_layers} layers, dim={args.dim}, vocab={args.vocab_size}, "
        f"mesh={tuple(mesh_device.shape)} ({model.num_devices} device(s)), prompt={token_ids.shape[1]}"
    )
    return model, tokenizer, token_ids


def _build_prompt(tokenizer, length):
    """Exactly ``length`` real tokens. prefill_paged reads the last row, so right-padding would score a pad."""
    ids = tokenizer(_PROMPT_TEXT, return_tensors="pt").input_ids
    assert ids.shape[1] >= length, f"prompt text tokenizes to {ids.shape[1]} tokens, need {length}"
    return ids[:, :length].to(torch.long)


def hf_reference(ckpt_dir, token_ids, decode_steps=0):
    """HF prefill plus greedy decode. Returned teacher tokens let TT replay the same inputs."""
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    ref_dtype = getattr(torch, os.environ.get("QWEN36_FULL_DEPTH_REF_DTYPE", "bfloat16"))
    logger.info(f"Loading HF reference ({ref_dtype}) from {ckpt_dir} ...")
    text_config = Qwen3_5TextConfig.from_pretrained(ckpt_dir)
    hf_model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        ckpt_dir, config=text_config, dtype=ref_dtype, output_loading_info=True
    )
    # Unexpected visual.*/mtp.* keys are fine; a missing key leaves a weight at random init.
    assert not loading_info["missing_keys"], f"HF reference has uninitialized weights: {loading_info['missing_keys']}"
    hf_model.eval()

    with torch.no_grad():
        out = hf_model(token_ids, use_cache=True)
        prefill_logits = out.logits[0, -1].float()
        cache = out.past_key_values

        decode_logits, teacher_tokens = [], []
        tok = int(prefill_logits.argmax())
        for _ in range(decode_steps):
            teacher_tokens.append(tok)
            out = hf_model(torch.tensor([[tok]], dtype=torch.long), past_key_values=cache, use_cache=True)
            step_logits = out.logits[0, -1].float()
            cache = out.past_key_values
            decode_logits.append(step_logits)
            tok = int(step_logits.argmax())

    del hf_model, cache, out
    gc.collect()
    return prefill_logits, decode_logits, teacher_tokens


def allocate_paged_kv(model, num_blocks=NUM_BLOCKS):
    """Paged KV plus GDN state. num_blocks must cover every position the caller will touch."""
    args = model.args
    n_kv = args.n_local_kv_heads if model.num_devices > 1 else args.n_kv_heads
    model.allocate_kv_caches([num_blocks, n_kv, BLOCK_SIZE, args.head_dim], ttnn.bfloat16, batch_size=1)
    return torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)


def tt_prefill_logits(model, token_ids, page_table):
    """Whole prompt via prefill_paged. Leaves the KV and GDN state decode reads."""
    vocab = model.args.vocab_size
    tt_logits = model.prefill_paged(token_ids, page_table, valid_len=token_ids.shape[1])
    if model.num_devices > 1:
        # The LM head leaves the logits replicated; read one replica.
        host = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    else:
        host = ttnn.to_torch(tt_logits)
    out = host.reshape(-1, host.shape[-1])[0, :vocab].float()
    assert not torch.isnan(out).any(), "NaN in full-depth prefill logits"
    return out


def tt_decode_logits(model, token, position, page_table):
    """One decode step through prepare_inputs_decode, ttnn_decode_forward, process_output_decode."""
    dev = model.prepare_inputs_decode(
        torch.tensor([[token]], dtype=torch.int32), torch.tensor([position], dtype=torch.int32), page_table
    )
    tt_out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
    out = model.process_output_decode(tt_out, 1).reshape(-1)[: model.args.vocab_size].float()
    assert not torch.isnan(out).any(), f"NaN in full-depth decode logits at position {position}"
    return out


def report(label, hf_logits, tt_logits, tokenizer):
    """PCC plus argmax/top-5 for one position. Near-tied logits can disagree without a defect."""
    _, pcc = comp_pcc(hf_logits, tt_logits, 0.0)
    hf_tok, tt_tok = int(hf_logits.argmax()), int(tt_logits.argmax())
    overlap = len(set(hf_logits.topk(5).indices.tolist()) & set(tt_logits.topk(5).indices.tolist()))
    logger.info(
        f"{label}: PCC={float(pcc):.6f} top5={overlap}/5 argmax HF={hf_tok} ({tokenizer.decode([hf_tok])!r}) "
        f"TT={tt_tok} ({tokenizer.decode([tt_tok])!r}) {'ok' if hf_tok == tt_tok else 'MISMATCH'}"
    )
    return float(pcc)
