import torch
import pytest
import ttnn
import numpy as np
from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1024])),
        (torch.Size([4, 1, 1, 2048])),
    ),
)
def test_dopout(input_shapes, device):
    t = torch.ones(input_shapes)
    t_tt = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_ratios = []
    s = 124
    prob = 0.2
    for _ in range(1000):
        output = ttnn.dropout(t_tt, seed=s, probability=prob, scale=1.0 / (1.0 - prob))
        output_torch = ttnn.to_torch(output)
        r = 1.0 - (torch.count_nonzero(output_torch) / torch.count_nonzero(t)).item()
        tt_ratios.append(r)

    mean = np.mean(tt_ratios)
    std = np.std(tt_ratios)
    # current dropout has pretty high variance so we just checking with some reasonable nubmers
    assert np.allclose(mean, prob, rtol=0.02)
    assert std < prob
