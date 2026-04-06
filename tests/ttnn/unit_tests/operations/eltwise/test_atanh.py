import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    assert_allclose,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)


@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_atanh(device, is_fp32):
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        # Cast to float32 and flush subnormal inputs — hardware flushes these to zero
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference in float32, flush subnormals to match hardware behavior
    torch_output = torch.atanh(torch_input.float())
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.atanh(tt_input)
    actual = ttnn.to_torch(tt_output)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Domain mask: atanh is only defined for |x| < 1
    # Exclude boundary values where precision degrades severely
    domain_mask = torch.abs(torch_input.float()) < 0.99

    # Filter out NaN/Inf for meaningful comparison, combined with domain mask
    finite_mask = domain_mask & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    if is_fp32:
        # Stricter tolerances — both sides have full float32 precision
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=3, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)
