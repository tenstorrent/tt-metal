"""single_layer CLI path: reproduces the chain dump for L3, and runs (finite, right shapes) on a random real-scale input for SWA L20/L46 and global L47."""
import os

import pytest
import torch

from models.demos.blackhole.mimo_v26.reference.single_layer import run

DUMP = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"


@pytest.mark.skipif(not os.path.isdir(DUMP), reason="dump missing")
def test_reproduces_chain_dump_layer3():
    d = torch.load(f"{DUMP}/layer_3.pt")
    out = run(3, d["prefill"]["h_in"][:4], torch.bfloat16)
    assert torch.equal(out["prefill"]["routing_idx"], d["prefill"]["routing_idx"][:512])
    assert (out["prefill"]["h_out"].float() - d["prefill"]["h_out"][:4].float()).abs().max() < 1e-2


@pytest.mark.parametrize("L", [20, 46, 47])
def test_random_real_scale_input(L):
    ref = torch.load(f"{DUMP}/layer_5.pt")["prefill"]["h_out"][:2].float()
    g = torch.Generator().manual_seed(1)
    h = torch.randn(2, 128, 4096, generator=g) * ref.std(dim=(0, 1)) + ref.mean(dim=(0, 1))
    out = run(L, h, torch.bfloat16, dec=torch.randn(2, 2, 4096, generator=g) * ref.std())
    assert all(torch.isfinite(v.float()).all() for v in out["prefill"].values())
    assert out["prefill"]["routing_idx"].shape == (256, 8) and len(out["decode"]) == 2
    assert ("k" in out["state"]) == (L == 47)
