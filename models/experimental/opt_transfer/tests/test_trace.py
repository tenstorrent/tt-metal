import torch
from models.experimental.opt_transfer.trace import trace_module
from models.experimental.opt_transfer.references.seamless_m4t_v2 import SeamlessBlock


def test_trace_exposes_sibling_projections_sharing_input():
    blk = SeamlessBlock(embed=1024, num_heads=16)
    g = trace_module(blk, (torch.randn(1, 8, 1024),))
    linears = [n for n in g.nodes() if n.kind == "linear"]
    assert len(linears) >= 3
    q, k, v = (n for n in linears if n.name in ("q_proj", "k_proj", "v_proj"))
    assert q.inputs[0] == k.inputs[0] == v.inputs[0]


def test_trace_is_deterministic():
    blk = SeamlessBlock(embed=1024, num_heads=16)
    g1 = trace_module(blk, (torch.randn(1, 8, 1024),))
    g2 = trace_module(blk, (torch.randn(1, 8, 1024),))
    assert [n.name for n in g1.nodes()] == [n.name for n in g2.nodes()]
