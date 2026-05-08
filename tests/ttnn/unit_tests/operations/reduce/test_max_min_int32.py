import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device
INT32_INFO = torch.iinfo(torch.int32)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 32, 32),  # issue #21071 repro: single tile
        (1, 1, 64, 60),  # 2x2 tiles with partial tile
        (1, 1, 100, 120),  # 4x4 tiles with partial tile
        (1, 1, 30, 96),  # Ht=1, Wt=3
        (1, 1, 90, 32),  # Ht=3, Wt=1
        (2, 3, 64, 64),  # multi-batch, NC>1
        (1, 3, 17, 19),  # non-tile-aligned NC reduction
        (2, 4, 64, 60),  # multi-batch NC + partial W tile
    ],
)
@pytest.mark.parametrize("dim", [0, 1, -1, -2, (-1, -2), None])
@pytest.mark.parametrize("op", ["max", "min"])
def test_max_min_int32(device, input_shape, dim, op):
    torch.manual_seed(0)

    # Issue #21071 repro: keep deterministic input for that single case.
    # Exclude INT32_MIN (0x80000000) elsewhere -- it has no representable
    # negation under XOR-bit-31 and trips an SFPU column-reduce corner.
    if input_shape == (1, 1, 32, 32) and dim == -1 and op == "max":
        torch_input_tensor = torch.arange(32 * 32, dtype=torch.int32).reshape(input_shape)
    else:
        # ttnn.min(int32) is implemented as -MAX(-x) using the on-chip “negate” path;
        # the value INT32_MIN has no representable negation in int32 (it would overflow / behaves incorrectly
        # under the sign-bit-flip negate used on device)
        torch_input_tensor = torch.randint(INT32_INFO.min + 1, INT32_INFO.max, input_shape, dtype=torch.int32)

    torch_op = torch.amax if op == "max" else torch.amin
    ttnn_op = ttnn.max if op == "max" else ttnn.min

    torch_output_tensor = torch_op(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    output_tensor = ttnn_op(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.dtype == torch.int32, f"Expected int32 output, got {output_tensor.dtype}"
    assert_equal(output_tensor, torch_output_tensor)
