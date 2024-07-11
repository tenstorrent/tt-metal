from loguru import logger
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.demos.mamba.tt.preprocessing import split_sequence_length


@pytest.mark.parametrize(
    "layout",
    [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
)
@pytest.mark.parametrize(
    "L, E, chunk_size, num_chunks",
    (
        (32, 32, 32, 1),
        (64, 32, 32, 2),
        (128, 32, 32, 4),
        (128, 64, 32, 2),
        (1024, 128, 32, 8),
    ),
)
def test_splitting_sequence_length(
    L: int, E: int, chunk_size: int, num_chunks: int, layout: ttnn.Layout, device: ttnn.Device
):
    expected = torch.randn((1, 1, L, E), dtype=torch.bfloat16)

    x = ttnn.from_torch(expected, dtype=ttnn.bfloat16, device=device, layout=layout)

    chunks = []
    for chunk in split_sequence_length(x, chunk_size=chunk_size):
        assert list(chunk.shape) == [1, 1, chunk_size, E]
        chunks.append(chunk)
    actual = ttnn.to_torch(ttnn.concat(chunks, dim=2))

    assert actual.shape == x.shape, "Expected input shape to match output shape"

    does_pass, output_pcc = comp_pcc(expected, actual, 1.0)
    logger.info(f"PCC value: {output_pcc}")
    assert does_pass, f"PCC value ({output_pcc}) is lower than 1.0"

    does_pass, output_allclose = comp_allclose(expected, actual)
    assert does_pass, "Allclose check failed: {output_allclose}"
