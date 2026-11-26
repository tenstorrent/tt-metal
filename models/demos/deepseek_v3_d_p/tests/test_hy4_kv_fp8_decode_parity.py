# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device, 8x4 Galaxy: the rows decode's own KV append writes with ``append="nearest"`` are byte-identical to the
prefill pack of the same BF16 rows with ``per_token_cast_to_fp8(round_to_nearest=True)`` (the pack
``Hy4Adapter``'s KVPE cache uses by default).

It reuses decode's append test as it stands (``models/experimental/hy4_preview/fused/layer_attn/tests/test_append.py``,
rungs "solo", chip (0, 0) alone, and "galaxy", all 32 chips): ``AppendCase(mode="nearest")`` builds P1 with
``HY4A_FP8_NEAREST``, and ``check_append``'s bit-for-bit branch (the one decode runs for "truncate") compares every
owned row of the store with ``harness.pack_rows`` of the BF16 rows the append core finished, on the same chip; here
``pack_rows`` is the prefill pack with ``round_to_nearest=True``. Scales, RoPE bytes and every e4m3 byte must match; the truncating pack, as a control, must not.

Without decode's golden (``HY4_GOLDEN_DIR``) the residual streams are random: the append's input differs, the
comparison does not. Needs ``HY4_HF_MODEL`` (decode's layer setup reads the checkpoint's site parameters).
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

TA = pytest.importorskip("models.experimental.hy4_preview.fused.layer_attn.tests.test_append")
from models.experimental.hy4_preview.fused.layer_attn.tests import test_p1 as P1  # noqa: E402
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs  # noqa: E402


def _pack_rows_nearest(device, latent: torch.Tensor, rope: torch.Tensor):
    """``harness.pack_rows`` with the prefill pack's round-to-nearest (``MlaKvCache`` with fp8_rounding="nearest")."""
    tt_latent = hs.to_dev(latent.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_rope = hs.to_dev(rope.to(torch.bfloat16), device, ttnn.bfloat16)
    fp8, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
        tt_latent, round_scale_to_power_of_two=True, round_to_nearest=True
    )
    packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(fp8, scales, tt_rope)
    for t in (tt_latent, tt_rope, fp8, scales):
        ttnn.deallocate(t)
    return packed


def _streams(case):
    golden = P1.hf_harness.find_golden()
    if golden is not None:
        return golden["layers"][1]["streams_in"][slice(0, 8)].float()
    logger.warning("no decode golden (HY4_GOLDEN_DIR): random residual streams")
    g = torch.Generator().manual_seed(11)
    return torch.randn(case.a.rows, 4, case.a.cfg.dim, generator=g)


@P1.galaxy
@pytest.mark.parametrize("rung", ["solo", "galaxy"])
@pytest.mark.timeout(0)
def test_decode_nearest_append_is_the_prefill_nearest_pack(mesh_device, monkeypatch, rung, expect_error):
    case = TA.AppendCase(mesh_device, "nearest", chips=TA.RUNGS[rung])
    a = case.a
    streams = _streams(case)
    a.set_streams(streams)
    _, _, x = case.golden_x(streams)
    case.prefill_absent(x.to(torch.bfloat16).float())
    before = case.store_bytes()
    a.run_p1()
    ttnn.synchronize_device(mesh_device)
    case.mode = "truncate"  # selects check_append's byte-equality branch only; P1 was built (and ran) with "nearest"
    # control: the truncating prefill pack must NOT match (the comparison can see the rounding)
    with expect_error(AssertionError, "bytes differ from the prefill pack"):
        TA.check_append(case, before, f"{rung}: decode append nearest vs prefill pack truncate (control)")
    # check_append's bit-for-bit branch, against the prefill pack with round_to_nearest=True
    monkeypatch.setattr(hs, "pack_rows", _pack_rows_nearest)
    TA.check_append(case, before, f"{rung}: decode append nearest vs prefill pack nearest")
    logger.info(
        "HY4 KV FP8 parity: decode's append='nearest' rows == prefill per_token_cast_to_fp8(round_to_nearest=True)"
    )
