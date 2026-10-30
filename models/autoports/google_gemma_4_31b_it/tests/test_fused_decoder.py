"""Real target shapes with deterministic weights drawn from checkpoint statistics."""

from argparse import Namespace

import pytest

from models.autoports.google_gemma_4_31b_it.tests.run_fused_decoder import run


@pytest.mark.parametrize("layer", [0, 5])
@pytest.mark.parametrize("batch,seq", [(1, 33), (8, 32)])
def test_fused_decoder(layer, batch, seq):
    run(Namespace(layer=layer, batch=batch, seq=seq, extended=False, synthetic=True, result=None))
