"""Normal CI uses real-config synthetic weights derived from checkpoint stats."""

from argparse import Namespace

import pytest

from .run_fused import run


@pytest.mark.parametrize("batch,split", [(1, None), (2, 31)])
def test_fused_dense_decoder(tmp_path, batch, split):
    run(
        Namespace(
            lengths=[257],
            batch=batch,
            decode_steps=2,
            synthetic=True,
            profile=False,
            split=split,
            remap=True,
            audit=True,
            unchunked_control=batch == 1,
            output=str(tmp_path / f"dense_{batch}.json"),
        )
    )
