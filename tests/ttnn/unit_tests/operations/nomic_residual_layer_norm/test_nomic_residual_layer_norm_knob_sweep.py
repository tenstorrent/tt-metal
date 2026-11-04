# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-cost knob sweep driver (Refinement 3): exchange transport, row blocking, exchange overlap.

Each variant monkeypatches op / descriptor module knobs for one call pair. Run under
`scripts/run_safe_pytest.sh --profile` and read DEVICE KERNEL DURATION per case (two calls a case, in parametrize
order). Correctness is checked at the acceptance tolerance here too (PCC 0.99999), since a knob that changes the
schedule can break a block offset that a loose PCC would miss.
"""

import importlib
import os

import pytest
import torch
import ttnn

op_module = importlib.import_module("ttnn.operations.nomic_residual_layer_norm.nomic_residual_layer_norm")
pd_module = importlib.import_module(
    "ttnn.operations.nomic_residual_layer_norm.nomic_residual_layer_norm_program_descriptor"
)

HIDDEN = 768

SHAPES = [
    (8, 512, "op_layout"),
    (2, 512, "op_layout"),
    (4, 512, "op_layout"),
    (6, 512, "op_layout"),
    (7, 512, "op_layout"),
    (8, 512, "model_layout"),
    (4, 288, "model_layout"),
    (2, 512, "model_layout"),
]

# name -> [(module, attribute, value), ...]
VARIANTS = {
    "base": [],
    # The Refinement 2 defaults: one block a shard, serialized exchange.
    "r2_defaults": [(op_module, "PIPELINE_BLOCK_ROWS", None), (pd_module, "EXCHANGE_MODE", "serialized_chain")],
    "unpipelined": [(op_module, "PIPELINE_BLOCK_ROWS", None)],
    "serialized_chain": [(pd_module, "EXCHANGE_MODE", "serialized_chain")],
    "fused_recompute": [(pd_module, "FUSED_RECOMPUTE", True)],
    "concurrent_unicast": [(pd_module, "EXCHANGE_MODE", "concurrent_unicast")],
    "concurrent_mcast": [(pd_module, "EXCHANGE_MODE", "concurrent_mcast")],
    "block2": [(op_module, "MAX_BLOCK_ROWS", 2)],
    "block1": [(op_module, "MAX_BLOCK_ROWS", 1)],
    "g3": [(op_module, "HIDDEN_SPLIT_CANDIDATES", (3,))],
    "g3_block1": [(op_module, "HIDDEN_SPLIT_CANDIDATES", (3,)), (op_module, "MAX_BLOCK_ROWS", 1)],
    "g2": [(op_module, "HIDDEN_SPLIT_CANDIDATES", (2,))],
    "g2_block1": [(op_module, "HIDDEN_SPLIT_CANDIDATES", (2,)), (op_module, "MAX_BLOCK_ROWS", 1)],
    "model_concurrent_unicast": [(pd_module, "MODEL_EXCHANGE_MODE", "concurrent_unicast")],
    "model_concurrent_mcast": [(pd_module, "MODEL_EXCHANGE_MODE", "concurrent_mcast")],
}


# Optional comma-separated filters (node IDs / -k with spaces do not survive the --profile wrapper):
#   NRLN_VARIANTS=base,block2   NRLN_SHAPES=8x512-op_layout
def _filtered(env, names):
    keep = os.environ.get(env)
    return [n for n in names if not keep or n in keep.split(",")]


SHAPES = [
    s for s in SHAPES if f"{s[0]}x{s[1]}-{s[2]}" in _filtered("NRLN_SHAPES", [f"{b}x{q}-{p}" for b, q, p in SHAPES])
]


@pytest.mark.parametrize("variant", _filtered("NRLN_VARIANTS", list(VARIANTS)))
@pytest.mark.parametrize("batch, seq, placement", SHAPES, ids=[f"{b}x{s}-{p}" for b, s, p in SHAPES])
def test_knob_sweep(device, monkeypatch, variant, batch, seq, placement):
    for module, attr, value in VARIANTS[variant]:
        assert hasattr(module, attr), attr
        monkeypatch.setattr(module, attr, value)
    shape = (batch, 1, seq, HIDDEN)
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    r = torch.randn(shape).to(torch.bfloat16)
    gamma = 1.0 + 0.25 * torch.randn(HIDDEN)
    beta = (0.1 * torch.randn(HIDDEN)).to(torch.bfloat16)
    layout_of = op_module.model_memory_config if placement == "model_layout" else op_module.input_memory_config
    layout = layout_of(shape, device)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout)
    tt_r = ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout)
    tt_g = ttnn.from_torch(gamma.reshape(1, 1, 1, HIDDEN), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(beta.reshape(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(2):
        out = op_module.nomic_residual_layer_norm(
            tt_x,
            tt_g,
            tt_b,
            residual=tt_r,
            epsilon=1e-5,
            compute_kernel_config=op_module.default_compute_kernel_config(),
        )
    y = x.double() + r.double()
    expected = torch.nn.functional.layer_norm(y, (HIDDEN,), gamma.double(), beta.double(), eps=1e-5)
    got = ttnn.to_torch(out).double()
    assert torch.corrcoef(torch.stack([got.flatten(), expected.flatten()]))[0, 1] > 0.99999
