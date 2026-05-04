import pytest
import torch
import ttnn
import ttml

_AVAILABLE = True


def _src_uint32(device):
    return ttnn.from_torch(
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


@pytest.mark.skipif(not _AVAILABLE, reason="ttml/ttnn not importable")
def test_typecast_uint32_to_bf16_directly_is_broken(device):
    """Regression marker: direct uint32→bf16 typecast does NOT recover [0, 1].
    Currently observed to emit [0.0, 2^31]."""
    for i in range(2):
        direct = ttnn.typecast(_src_uint32(device), ttnn.bfloat16)
        vals = ttnn.to_torch(direct).float().tolist()
        print(f"val = {vals}")
        assert vals == [0.0, 1.0, 2.0, 3.0], (
            f"ttnn::typecast(uint32→bf16) returned {vals} — the bug appears to be "
            "fixed. Drop the fp32 detour in moe_group_op.cpp and update "
            "memory/reference_ttnn_typecast_uint32_bf16_broken.md."
        )


@pytest.mark.skipif(not _AVAILABLE, reason="ttml/ttnn not importable")
def test_typecast_uint32_via_fp32_to_bf16_is_correct(device):
    """Workaround: uint32 → float32 → bf16 round-trip recovers [0, 1]."""
    for i in range(2):
        via_fp32 = ttnn.typecast(ttnn.typecast(_src_uint32(device), ttnn.float32), ttnn.bfloat16)
        vals = ttnn.to_torch(via_fp32).float().tolist()
        print(f"val = {vals}")
        assert vals == [0.0, 1.0, 2.0, 3.0], f"expected [0.0, 1.0, 2.0, 3.0], got {vals}"
