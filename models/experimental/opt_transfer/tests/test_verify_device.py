import torch
from models.experimental.opt_transfer.references.seamless_m4t_v2 import SeamlessBlock
from models.experimental.opt_transfer.verify import pcc, golden_outputs


def test_pcc_identity_is_one():
    a = torch.randn(4, 4)
    assert pcc(a, a.clone()) > 0.999


def test_golden_outputs_runs_reference():
    blk = SeamlessBlock(1024, 16)
    x = torch.randn(1, 8, 1024)
    out = golden_outputs(blk, (x,))
    assert out.shape == (1, 8, 1024)
