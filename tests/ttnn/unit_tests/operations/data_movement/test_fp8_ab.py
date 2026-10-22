# Scratch, not for merge. Op level timing of the Fp8_e4m3 untilize kernels of test_untilize_fp8_row_split.py (generic_op on one core).
# "real" uses the test's reader and writer; "null" pushes and pops the pages without NoC traffic so the compute kernel is the op time.
import importlib.util
import os

import pytest
import torch
import ttnn

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_untilize_fp8_row_split.py")
_s = importlib.util.spec_from_file_location("fp8_split_mod", _p)
M = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M)

REAL_READER, REAL_WRITER = M.READER, M.WRITER
NULL_READER = REAL_READER.replace("noc_async_read(src.get_noc_addr(t), get_write_ptr(cb_in), tile_bytes);\n        noc_async_read_barrier();", "")
NULL_WRITER = REAL_WRITER.replace(
    "for (uint32_t r = 0; r < num_rows; ++r) {\n        noc_async_write(l1_addr + r * row_bytes, dst.get_noc_addr(r), row_bytes);\n    }", ""
)
assert NULL_READER != REAL_READER and NULL_WRITER != REAL_WRITER


def _set(mode):
    M.READER, M.WRITER = (NULL_READER, NULL_WRITER) if mode == "null" else (REAL_READER, REAL_WRITER)


def _vals(rows, width):
    g = torch.Generator().manual_seed(width)
    e = torch.randint(1, 15, (rows, width * 32), generator=g)
    m = torch.randint(0, 8, (rows, width * 32), generator=g)
    s = torch.randint(0, 2, (rows, width * 32), generator=g)
    return (((s << 7) | (e << 3) | m).to(torch.uint8)).view(torch.float8_e4m3fn).to(torch.bfloat16)


HELPER = [(5, False), (7, False), (11, False), (13, False), (11, True), (13, True), (17, True), (6, False), (8, False)]
TILE_ROWS = 8


@pytest.mark.parametrize("mode", ["null", "real"])
@pytest.mark.parametrize("width,full_sync", HELPER, ids=[f"w{w}{'F' if f else 'H'}" for w, f in HELPER])
def test_helper(device, mode, width, full_sync):
    _set(mode)
    n = TILE_ROWS * width
    M._launch(device, M._helper_compute(width, TILE_ROWS, full_sync), _vals(TILE_ROWS * 32, width), ttnn.fp8_e4m3,
              ttnn.tile_size(ttnn.fp8_e4m3), n, n)


def _bits(case, mode, ct, values, actual):
    # Real mode only: the output bytes against the Fp8_e4m3 conversion of the input; printed, never asserted, so every arm times the same ids.
    if mode != "real":
        return
    exp = values.to(torch.float8_e4m3fn).view(torch.uint8)
    bad = int((exp != actual).sum())
    print(f"RAB BITS {case} ct{ct} bad={bad} of {exp.numel()}", flush=True)


CTS = [9, 11, 13, 15, 17, 19, 12]


@pytest.mark.parametrize("mode", ["null", "real"])
@pytest.mark.parametrize("ct", CTS)
def test_dispatch(device, mode, ct):
    _set(mode)
    b = M._block_ct_dim(ct)
    c = M._deepseek_compute("dispatch/device/kernels/compute/untilize_dispatch.cpp", [M.CB_PRE, M.CB_OUT, M.CB_IN, ct * 32, M.ROWS, b])
    v = M._fp8_values(ct, ct)
    _bits("dispatch", mode, ct, v, M._launch(device, c, v, ttnn.fp8_e4m3, ct * 32, M.ROWS, 2 * b, pre_cb=(1, 0, M.SENTINEL)))


@pytest.mark.parametrize("mode", ["null", "real"])
@pytest.mark.parametrize("ct", CTS)
def test_combine(device, mode, ct):
    _set(mode)
    b = M._block_ct_dim(ct)
    c = M._deepseek_compute("combine/device/kernels/compute/untilize_combine.cpp",
                            [M.CB_OUT, M.CB_IN, M.CB_PRE, 1, 1, 0, M.ROWS, M.ROWS, ct, b, 2], runtime_args=[0, 1, 0, 1, 0, 1])
    v = M._fp8_values(ct, ct)
    _bits("combine", mode, ct, v, M._launch(device, c, v, ttnn.fp8_e4m3, ct * 32, M.ROWS, 2 * b, pre_cb=(2, M.ROWS, None)))
