# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch unit tests for the Kimi K2.6 compressed-tensors INT4 dequant path.

These exercise the bit-manipulation-heavy unpacker (little-endian 4-bit lane extraction,
offset-binary decode, per-group scale broadcast) and the config-group parameter extraction
directly on the host -- no device or multi-GB checkpoint required. This is the INT4 counterpart
to the shared fp8 round-trip test
``models/demos/deepseek_v3/tests/test_hf_model_utils.py::test_dequantize_state_dict_compat_shim_handles_quantized_inputs``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.demos.deepseek_v3_d_p.utils.test_utils import (
    _dequantize_packed_int4_weight,
    _pack_quant_params,
    dequantize_state_dict,
    is_pack_quantized_int4,
)

pytestmark = pytest.mark.t3k_compat


def _pack_int4_offset_binary(levels: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_dequantize_packed_int4_weight``'s unpack step, for building test inputs.

    ``levels`` are signed INT4 values in ``[-8, 7]`` with shape ``[out, in]`` and ``in`` a multiple
    of the pack factor (8). Stores them offset-binary (``+8``) and packs 8 little-endian 4-bit lanes
    per int32 word along the input dim (lane ``j`` at bits ``[4j, 4j+4)``).
    """
    out_features, in_features = levels.shape
    assert in_features % 8 == 0, "test helper requires in_features divisible by the int4 pack factor (8)"
    stored = (levels + 8).to(torch.int64)  # offset-binary storage
    words = stored.reshape(out_features, in_features // 8, 8)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    return (words << shifts).sum(dim=-1).to(torch.int32)  # bit-31 may be set -> negative int32, as in real checkpoints


def _int4_quant_config(group_size: int, *, int4_group: str = "group_0", decoy: bool = False) -> dict:
    """A compressed-tensors quantization_config whose INT4 weights live in ``int4_group``.

    With ``decoy=True`` a non-INT4 (8-bit) ``group_0`` precedes the INT4 group, so the config
    reproduces the case where the INT4 group is *not* ``group_0``.
    """
    int4 = {"num_bits": 4, "type": "int", "strategy": "group", "group_size": group_size, "symmetric": True}
    groups: dict = {}
    if decoy:
        groups["group_0"] = {"weights": {**int4, "num_bits": 8}}
    groups[int4_group] = {"weights": int4}
    return {"quant_method": "compressed-tensors", "config_groups": groups}


def test_dequantize_packed_int4_weight_round_trips_known_values():
    """Pack known signed INT4 levels and assert the unpacker recovers ``levels * per-group scale``."""
    # Distinct value per lane (catches lane-order/endianness bugs) spanning the full [-8, 7] range,
    # across two 8-wide groups (catches the per-group scale broadcast).
    levels = torch.tensor(
        [
            [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8],
        ],
        dtype=torch.int64,
    )
    group_size = 8
    scale = torch.tensor([[0.5, 2.0], [1.0, 0.25]], dtype=torch.float32)

    packed = _pack_int4_offset_binary(levels)
    assert (packed < 0).any(), "test data should exercise the high (bit-31) lane"

    got = _dequantize_packed_int4_weight(
        packed,
        scale,
        torch.tensor([levels.shape[0], levels.shape[1]]),
        group_size=group_size,
        num_bits=4,
        symmetric=True,
        dtype=torch.float32,
    )
    expected = levels.to(torch.float32) * scale.repeat_interleave(group_size, dim=1)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_pack_quant_params_reads_the_matched_int4_group():
    """`is_pack_quantized_int4` and `_pack_quant_params` must agree when INT4 is not ``group_0``."""
    cfg = _int4_quant_config(group_size=32, int4_group="group_1", decoy=True)
    assert is_pack_quantized_int4(cfg) is True
    # Must read group_1 (num_bits=4), NOT the 8-bit decoy group_0.
    assert _pack_quant_params(cfg) == (4, 32, True)


def test_dequantize_state_dict_int4_pack_quant_round_trip():
    """End-to-end: a packed INT4 triplet is dequantized; fp32 passthrough tensors keep their dtype."""
    levels = torch.tensor([[-8, -1, 0, 7, 3, -4, 5, -2]], dtype=torch.int64)  # 1x8 == one group
    group_size = 8
    scale = torch.tensor([[0.25]], dtype=torch.float32)
    packed = _pack_int4_offset_binary(levels)

    weight = "model.layers.0.mlp.experts.0.gate_proj.weight"
    bias = "model.layers.0.mlp.gate.e_score_correction_bias"
    bias_val = torch.tensor([1.5, -2.5], dtype=torch.float32)  # router bias -- must stay fp32
    state_dict = {
        f"{weight}_packed": packed,
        f"{weight}_scale": scale,
        f"{weight}_shape": torch.tensor([levels.shape[0], levels.shape[1]]),
        bias: bias_val,
    }
    # INT4 group deliberately not group_0 -> also guards the is_pack_quantized_int4/_pack_quant_params fix.
    hf_config = SimpleNamespace(quantization_config=_int4_quant_config(group_size, int4_group="group_1", decoy=True))

    out = dequantize_state_dict(state_dict, hf_config, dtype=torch.bfloat16)

    assert set(out) == {weight, bias}  # _packed/_scale/_shape consumed into the single weight
    expected = (levels.to(torch.float32) * scale.repeat_interleave(group_size, dim=1)).to(torch.bfloat16)
    assert out[weight].dtype == torch.bfloat16
    torch.testing.assert_close(out[weight], expected, rtol=0, atol=0)
    # fp32 passthrough must NOT be downcast to bf16 (router-bias precision).
    assert out[bias].dtype == torch.float32
    torch.testing.assert_close(out[bias], bias_val, rtol=0, atol=0)
