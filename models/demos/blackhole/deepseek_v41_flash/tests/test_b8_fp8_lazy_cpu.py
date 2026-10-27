# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the B=8 / 1M memory switches (branch ssinghal/dsv4p1-b8b):
  * DSV41_BUILD_LAZY_PF (``prefill_sparse.lazy_tables_on``): only with the single chunk trace; default off;
  * DSV41_FP8_SERVING: the single-trace default and the APC gate accept the fp8 pool only with it (defaults unchanged);
  * the e4m3 -> bf16 conversion of ``paged_kernels/row_gather.cpp`` (python mirror of the C++ bit arithmetic == torch.float8_e4m3fn for all 256 bytes);
  * the fp8 gather/scatter byte round trip (e4m3 value -> bf16 -> e4m3 is the identity: a restored snapshot equals the cold pool)."""

import pytest
import torch

from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as PS

ENV = (
    "DSV41_PF_SINGLE",
    "DSV41_PF_PAGED",
    "DSV41_PF_PAGED_FP8",
    "DSV41_PREFILL_DYN",
    "DSV41_FP8_SERVING",
    "DSV41_BUILD_LAZY_PF",
    "DSV41_POOL_DTYPE",
    "DSV41_APC",
)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(PS, "kv_len_tensor_available", lambda: True)


def test_single_trace_fp8_needs_the_switch(monkeypatch):
    monkeypatch.setenv("DSV41_PF_PAGED_FP8", "1")
    on, why = PS.resolve_server_default(512, 1048576)
    assert not on and "fp8" in why  # shipped behaviour
    monkeypatch.delenv("DSV41_PF_SINGLE")  # (the first call pinned it to 0)
    monkeypatch.setenv("DSV41_FP8_SERVING", "1")
    on, why = PS.resolve_server_default(512, 1048576)
    assert on and "fp8" in why


def test_lazy_tables_gate(monkeypatch):
    from models.demos.blackhole.deepseek_v41_flash.tt import prefill_sparse as SP

    assert not SP.lazy_tables_on()
    monkeypatch.setenv("DSV41_BUILD_LAZY_PF", "1")
    assert not SP.lazy_tables_on()  # no single trace: eager path needs the tables
    monkeypatch.setenv("DSV41_PF_SINGLE", "1")
    assert SP.lazy_tables_on()


def test_fp8_serving_env_setup(monkeypatch):
    import ttnn
    from models.demos.blackhole.deepseek_v41_flash.tt import common

    monkeypatch.setenv("DSV41_POOL_DTYPE", "fp8")
    assert common._env_setup(ttnn.bfloat16) == ttnn.bfloat16  # shipped: read mode forces the bf16 pool
    monkeypatch.setenv("DSV41_FP8_SERVING", "1")
    import os

    assert common._env_setup(ttnn.bfloat16) == ttnn.fp8_e4m3
    assert os.environ["DSV41_PF_PAGED_FP8"] == "1"


def _e4m3_to_bf16_bits(b):
    """line by line the arithmetic of e4m3_to_bf16 in paged_kernels/row_gather.cpp"""
    sign = (b & 0x80) << 8
    e, m = (b >> 3) & 0xF, b & 7
    if e == 0xF and m == 7:
        return sign | 0x7FC0
    if e == 0:
        if m == 0:
            return sign
        p = 2 if m >= 4 else (1 if m >= 2 else 0)
        frac = (m << (3 - p)) & 7
        return sign | ((p + 118) << 7) | (frac << 4)
    return sign | ((e + 120) << 7) | (m << 4)


def test_kernel_conversion_mirror_matches_torch():
    allb = torch.arange(256, dtype=torch.uint8)
    ref = allb.view(torch.float8_e4m3fn).to(torch.bfloat16)
    got = torch.tensor([_e4m3_to_bf16_bits(int(b)) for b in range(256)], dtype=torch.int32).to(torch.int16)
    got = got.view(torch.bfloat16) if got.dtype == torch.int16 else got
    nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(got), nan)
    assert torch.equal(got[~nan].float(), ref[~nan].float())
    # subnormals (|x| < 2^-6) are exact
    assert got[1].item() == 2**-9 and got[7].item() == 7 * 2**-9


def test_snapshot_roundtrip_is_identity():
    x = torch.randn(4096, 512) * 3
    q = x.to(torch.float8_e4m3fn)  # what paged_scatter_rows stores (bf16_to_e4m3 == torch cast, PAGED_NOTES.md)
    back = q.to(torch.bfloat16)  # what paged_gather_rows returns
    assert torch.equal(back.to(torch.float8_e4m3fn).view(torch.uint8), q.view(torch.uint8))
