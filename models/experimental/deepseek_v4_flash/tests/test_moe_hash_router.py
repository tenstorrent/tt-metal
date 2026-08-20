# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Routing contract of :class:`DeepSeekV4HashRouter` (the ``hash_moe`` layers).

The hash router selects experts by a frozen ``tid2eid[input_ids]`` lookup and hands the
selection to ``fused_experts`` as the same ``(scores, indices)`` pair the learned router
emits. The ids are gathered on device with :func:`ttnn.embedding`, which only reads a
bfloat16 table, so they arrive as bf16 values rather than integers -- exact below 256
experts, and what the op's ``index_is_bf16`` decode expects.

Checked here: the gathered ids are the table's rows for the given tokens, the scores are
the reference ``sqrt(softplus(x @ Wᵀ))``, and the prefill (host ids) and traced (device
ids) entry points agree. What the op then does with the pair -- normalize and scale over
the selected scores -- is covered by ``test_fused_experts_bf16_indices``.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tt.moe import DeepSeekV4HashRouter

HIDDEN = 256
NUM_EXPERTS = 64
TOP_K = 6
VOCAB = 512


def _router(device, tid2eid: torch.Tensor, gate_w: torch.Tensor) -> DeepSeekV4HashRouter:
    cfg = types.SimpleNamespace(
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        routed_scaling_factor=1.5,
    )
    return DeepSeekV4HashRouter(cfg, {"gate.weight": gate_w, "gate.tid2eid": tid2eid}, device)


@torch.no_grad()
@pytest.mark.parametrize("tokens", (1, 8, 32), ids=lambda t: f"tokens{t}")
def test_hash_router_sparse_routing(device, reset_seeds, tokens: int) -> None:
    torch.manual_seed(0)
    tid2eid = torch.stack([torch.randperm(NUM_EXPERTS)[:TOP_K] for _ in range(VOCAB)])
    gate_w = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16)
    router = _router(device, tid2eid, gate_w)

    x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16)
    x_flat = ttnn.from_torch(
        x.reshape(1, 1, tokens, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    input_ids = torch.randint(0, VOCAB, (tokens,), dtype=torch.int64)

    routing = router.forward(x_flat, input_ids)

    # The ids are the table's rows, unreordered: the op pairs id j with the score it reads
    # for that expert, so only the set matters, but an exact match is the stronger check.
    ids = ttnn.to_torch(routing.indices).float().reshape(tokens, TOP_K)
    assert torch.equal(ids, tid2eid[input_ids].float()), "gathered expert ids do not match tid2eid"
    assert routing.indices.dtype == ttnn.bfloat16, "ids must stay bf16 (what ttnn.embedding gathers)"

    scores = ttnn.to_torch(routing.scores).float().reshape(tokens, NUM_EXPERTS)
    reference = torch.sqrt(F.softplus(x.float() @ gate_w.float().t()))
    passing, pcc_msg = comp_pcc(reference, scores, pcc=0.999)
    assert passing, f"gate scores vs reference: {pcc_msg}"

    # The traced decode path gathers from persistent on-device token ids instead of
    # uploading them, and must route identically.
    token_in = ttnn.from_torch(
        input_ids.reshape(1, tokens).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    static = router.forward_static(x_flat, token_in)
    assert torch.equal(ttnn.to_torch(static.indices).float().reshape(tokens, TOP_K), ids)
    assert torch.equal(ttnn.to_torch(static.scores).float().reshape(tokens, NUM_EXPERTS), scores)
