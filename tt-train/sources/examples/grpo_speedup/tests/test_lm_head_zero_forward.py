#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Uniform-distribution test for ``LMHead.update``.

After zeroing every chunk of ``LMHead.output_weights_dram_sharded`` we
have ``logits = x @ 0 = 0``. Softmax over zero logits is uniform across
the entire vocab -- i.e. *every token has the exact same probability*.

We verify this by reading the LM-head output back to torch and checking
that the logits are identically zero. (Uniform softmax + greedy decoding
also implies the generated tokens are all the tie-broken-argmax of zero,
i.e. a single repeated token, which we sanity-check via ``generate``.)

Uses ``dummy_weights=True`` so this test runs without HF auth.
"""

from __future__ import annotations

import os

os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

import gc
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRPO_SPEEDUP = HERE.parent  # .../grpo_speedup
REPO_ROOT = HERE.parents[4]  # .../tt-metal
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GRPO_SPEEDUP))
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml"
MAX_SEQ_LEN = 2048
MAX_NEW_TOKENS = 8
TEMPERATURE = 0.0  # greedy -> uniform softmax collapses to a fixed argmax tie-break

PROMPT = "Explain a tensor in a paragraph."


def zero_lm_head(lm_head) -> None:
    """Zero out every chunk of ``self.output_weights_dram_sharded`` via ``LMHead.update``."""
    import torch

    V = lm_head.vocab_size
    H = lm_head.args.dim
    zero_weight = torch.zeros((V, H), dtype=torch.bfloat16)
    lm_head.update(zero_weight)


def _build_random_hidden_state(completer):
    """Construct a synthetic ``(1, 1, 32, dim)`` hidden state already in the
    DRAM-sharded layout that ``LMHead.forward`` expects in prefill mode.

    32 rows matches a single decode-mode tile and is the shape
    ``_apply_norm_and_lm_head`` itself documents
    (``models/tt_transformers/tt/model.py``). We mirror the resharding
    step that function does after the final RMSNorm: read the
    LM-head-input memory config from ``model_args`` and, if it's
    sharded, call ``interleaved_to_sharded`` before handing the tensor
    back. Without that, the program config inside ``LMHead.forward``'s
    ``ttnn.linear`` dereferences a shard spec the interleaved DRAM
    tensor doesn't carry, throwing ``RuntimeError: bad optional access``.
    """
    import torch
    import ttnn

    from models.tt_transformers.tt.common import Mode

    dim = completer.model_args.dim
    torch_x = torch.randn(1, 1, 32, dim, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        torch_x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=completer.mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(completer.mesh_device),
    )

    lm_head_input_mem_cfg = completer.model_args.get_lm_head_input_mem_config(Mode.PREFILL, None)
    if lm_head_input_mem_cfg.is_sharded():
        x = ttnn.interleaved_to_sharded(x, lm_head_input_mem_cfg)

    return x


def main() -> None:
    import torch
    import ttnn
    from ttml.common.config import DeviceConfig, load_config

    from utils.llama_completer_ttt import LlamaGRPOCompleter

    print(">>> set_fabric_config(FABRIC_2D)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    raw = load_config(os.path.join(REPO_ROOT, TTML_DEVICE_CONFIG_REL))
    device_config = DeviceConfig(raw)

    print(f">>> building LlamaGRPOCompleter ({MODEL_ID}, dummy_weights=True)")
    completer = LlamaGRPOCompleter(
        device_config=device_config,
        model_source=MODEL_ID,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        dummy_weights=True,
    )

    lm_head = completer.model.lm_head
    n_chunks = len(lm_head.output_weights_dram_sharded)
    print(
        f">>> lm_head: vocab={lm_head.vocab_size} padded={lm_head.padded_vocab_size} "
        f"dim={lm_head.args.dim} dram_sharded_chunks={n_chunks}"
    )

    # ---- Step 1: zero out the LM head ----
    print()
    print("=== Step 1: LMHead.update(torch.zeros(V, H)) ===")
    zero_lm_head(lm_head)

    # ---- Step 2: forward a random hidden state through LMHead and read logits ----
    print()
    print(f"=== Step 2: LMHead.forward(random hidden_state of shape (1, 1, 32, {completer.model_args.dim})) ===")
    x = _build_random_hidden_state(completer)
    logits = lm_head.forward(x)
    # The dram_sharded path concatenates chunks back via tt_all_reduce,
    # but the final tensor still has each device's column shard. For a
    # 1x1 mesh that's the full vocab.
    logits_torch = ttnn.to_torch(logits)
    print(f"  logits shape  = {tuple(logits_torch.shape)}")
    print(f"  logits dtype  = {logits_torch.dtype}")
    print(f"  logits max|.| = {float(logits_torch.abs().max()):.6g}")
    print(f"  logits mean|.|= {float(logits_torch.abs().mean()):.6g}")

    expected = torch.zeros_like(logits_torch)
    logits_zero_ok = torch.equal(logits_torch, expected)
    print(f"  logits == 0 elementwise: {logits_zero_ok}   [must be True]")

    # ---- Step 3: bonus sanity -- greedy generation collapses to a single repeated token ----
    print()
    print("=== Step 3: greedy generate -> tokens must all be identical ===")
    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)
    completions = completer.generate(
        [prompt_ids],
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    tokens = completions[0]
    print(f"  generated tokens = {tokens}")
    all_same = len(set(tokens)) <= 1
    print(f"  all tokens identical: {all_same}   [must be True]")

    # ---- Assertion ----
    print()
    print("=== assertions ===")
    ok = logits_zero_ok and all_same
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")

    import ttml

    del completer
    gc.collect()
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
