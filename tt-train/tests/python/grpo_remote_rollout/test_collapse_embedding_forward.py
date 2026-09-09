# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate ``Embedding.update`` via ``Embedding.forward()`` alone.

Invariant: original embeddings x all-TARGET_TOKEN_ID ids must equal collapsed
embeddings x real prompt ids (both yield ``E[TARGET_TOKEN_ID]`` at every pos).
"""

from __future__ import annotations

from typing import List

import pytest
import torch
import ttnn

from _completer_utils import as_update_input, open_completer, to_torch_2d

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_TOKEN_ID = 16000
PROMPT = "Explain a tensor in a paragraph."


@pytest.fixture(scope="module")
def completer():
    with open_completer(dummy_weights=False, model_source=MODEL_ID) as c:
        yield c


def _build_collapsed_embedding(completer):
    """Return the HF-format ``embed_tokens`` update input with every row
    replaced by row TARGET_TOKEN_ID."""
    model = completer.models[0]
    emb_hf_2d = to_torch_2d(model.embd.weights)  # (V, H)
    target_row = emb_hf_2d[TARGET_TOKEN_ID, :].clone()
    collapsed_hf = target_row.unsqueeze(0).expand(emb_hf_2d.shape[0], -1).contiguous()  # (V, H)

    return as_update_input(collapsed_hf, model.mesh_device)


def _ids_to_ttnn(completer, ids: List[int]):
    """Convert token ids to the ``(1, 1, 1, S)`` ttnn tensor ``Embedding.forward``
    expects."""
    model = completer.models[0]
    tokens = torch.tensor(ids, dtype=torch.int32).reshape(1, 1, 1, -1)
    return ttnn.from_torch(
        tokens,
        device=model.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
    )


def _embed(completer, ids: List[int]):
    """Run only ``Embedding.forward()`` and return the result as a torch tensor."""
    model = completer.models[0]
    ids_ttnn = _ids_to_ttnn(completer, ids)
    out = model.embd(ids_ttnn)
    return ttnn.to_torch(out)


def test_embedding_update_matches_direct_lookup(completer):
    """Invariant: after collapsing the table so every row = row TARGET_TOKEN_ID,
    embedding a real prompt must equal embedding all-TARGET_TOKEN_ID ids on the
    original table (both yield ``E[TARGET_TOKEN_ID]`` at every position)."""
    model = completer.models[0]
    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)
    all_target_ids = [TARGET_TOKEN_ID] * len(prompt_ids)

    a_orig = _embed(completer, all_target_ids)

    new_weights = _build_collapsed_embedding(completer)
    model.embd.update(embed_tokens=new_weights)

    c_collapsed = _embed(completer, prompt_ids)

    assert torch.equal(a_orig, c_collapsed), (
        "Embedding.update didn't reproduce a direct table lookup: "
        f"max|A-C|={float((a_orig.float() - c_collapsed.float()).abs().max()):.6g}, "
        f"mean|A-C|={float((a_orig.float() - c_collapsed.float()).abs().mean()):.6g}"
    )
