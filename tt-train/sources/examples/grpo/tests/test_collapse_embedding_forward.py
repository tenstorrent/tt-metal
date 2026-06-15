#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate ``Embedding.update`` directly via ``Embedding.forward()``.

Skips the generator entirely and compares the per-position embedding vectors
returned by the embedding layer alone. The invariant:

    A: original embeddings  x  all-TARGET_TOKEN_ID ids
    C: collapsed embeddings x  real prompt ids
    A == C

Both runs must produce ``E[TARGET_TOKEN_ID]`` at every position, so the two
``(1, 1, 1, S, H)`` tensors must be byte-identical.
"""

from __future__ import annotations

import os

os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]  # .../tt-metal
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MAX_SEQ_LEN = 2048

TARGET_TOKEN_ID = 16000

PROMPT = "Explain a tensor in a paragraph."


def _build_collapsed_embedding(completer):
    """Return the HF-format ``embed_tokens`` input for
    :meth:`Embedding.update` with every row replaced by row 16000.

    Shape: ``(1, 1, vocab_size, hidden_size)`` after ``as_update_input``
    wraps the natural HF ``(V, H)`` shape -- replicated, DRAM-interleaved,
    TILE_LAYOUT, bfloat16, as the new update contract requires.
    """
    from _completer_utils import as_update_input, to_torch_2d

    emb_hf_2d = to_torch_2d(completer.model.embd.weights)  # (V, H)
    target_row = emb_hf_2d[TARGET_TOKEN_ID, :].clone()
    collapsed_hf = target_row.unsqueeze(0).expand(emb_hf_2d.shape[0], -1).contiguous()  # (V, H)

    return as_update_input(collapsed_hf, completer.mesh_device)


def _ids_to_ttnn(completer, ids: List[int]):
    """Convert a python list of token ids to the ``(1, 1, 1, S)`` ttnn tensor
    that ``Embedding.forward`` expects (mirrors ``prepare_inputs_prefill``)."""
    import torch
    import ttnn

    tokens = torch.tensor(ids, dtype=torch.int32).reshape(1, 1, 1, -1)
    return ttnn.from_torch(
        tokens,
        device=completer.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(completer.mesh_device),
    )


def _embed(completer, ids: List[int]):
    """Run only ``Embedding.forward()`` and return the result as a torch tensor."""
    import ttnn

    ids_ttnn = _ids_to_ttnn(completer, ids)
    out = completer.model.embd(ids_ttnn)
    return ttnn.to_torch(out)


def main() -> None:
    import torch
    import ttnn

    from _completer_utils import open_completer

    print(">>> set_fabric_config(FABRIC_2D)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    print(f">>> building LlamaCompleterTtt ({MODEL_ID}, max_seq_len={MAX_SEQ_LEN})")
    with open_completer(
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        model_source=MODEL_ID,
    ) as completer:
        prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)
        all_16000_ids = [TARGET_TOKEN_ID] * len(prompt_ids)
        print(f">>> prompt_ids    = {prompt_ids}")
        print(f">>> all_16000_ids = {all_16000_ids}")

        print(">>> A: original embeddings  x  all-16000 ids")
        a_orig_16000 = _embed(completer, all_16000_ids)

        target_str = completer.tokenizer.decode([TARGET_TOKEN_ID], skip_special_tokens=False)
        print(f">>> collapsing embedding table: every row -> row {TARGET_TOKEN_ID} ({target_str!r})")
        new_weights = _build_collapsed_embedding(completer)
        completer.model.embd.update(embed_tokens=new_weights)

        print(">>> C: collapsed embeddings x  real prompt ids")
        c_coll_real = _embed(completer, prompt_ids)

        a_eq_c = torch.equal(a_orig_16000, c_coll_real)

        print()
        print(f"=== Embedding.forward() output shape: {tuple(a_orig_16000.shape)} ===")
        print(f"  A (orig x 16000s) == C (collapsed x real):  {a_eq_c}   [must be True]")

        if not a_eq_c:
            diff = (a_orig_16000.float() - c_coll_real.float()).abs()
            print(f"  max |A - C| = {float(diff.max()):.6g}, mean |A - C| = {float(diff.mean()):.6g}")

        print()
        print(f"  RESULT: {'PASS' if a_eq_c else 'FAIL'}")


if __name__ == "__main__":
    main()
