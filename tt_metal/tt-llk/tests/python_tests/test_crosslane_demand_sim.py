# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cross-lane demand fixtures ON DEVICE/SIM (lane FK, X7).

Drives the fresh ema/cumsum ROW-CHAIN cores (fresh_cpp/{ema,cumsum}.h via
sources/sfpu_crosslane_demand_test.cpp) against the lane-FB demand goldens
(crosslane_fixtures/{ema,cumsum}.json), BIT-EXACT on 32-bit Dst.

The fixture "register chain" maps directly: fixture register i = the vector
at DEST address 2*i (dst_reg[i]); the flat L1 stimuli vector is the
face-major tile (the sfpu_blaze_test address-model convention), so fixture
row placement uses the silicon-pinned SFPU DEST address model.

ema pins the ARITHMETIC CONTRACT: the fixture records BOTH candidates
("fma" = inner mul rounded + one fused outer MAD; "mul_add" = every op
rounded).  Each device contract arm must reproduce ITS OWN golden exactly —
a third value is a reportable finding (doc=prior, sim=oracle discipline).
cumsum is exact: int32 mod 2^32; fp32 serial low->high, one rounding per
add (the order is the contract).
"""

import json
import os
import struct

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, format_dict
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    BLAZE_PARAMS,
    DEMAND_CHAIN,
    TILE_COUNT,
)

_FIXTURES = os.path.join(os.path.dirname(__file__), "crosslane_fixtures")


def _fixture(name):
    with open(os.path.join(_FIXTURES, f"{name}.json")) as f:
        return json.load(f)


# SFPU DEST address model (silicon-pinned; sfpu_blaze_test.py reading).
def _vec_flat_indices(addr: int) -> torch.Tensor:
    a = addr % 64
    tile = addr // 64
    face = a // 16
    lane = torch.arange(32)
    row = 4 * ((a % 16) // 4) + lane // 8
    col = 2 * (lane % 8) + ((addr >> 1) & 1)
    return tile * 1024 + face * 256 + row * 16 + col


def _place_rows(flat: torch.Tensor, rows_hex, base_reg: int = 0) -> None:
    for r, row in enumerate(rows_hex):
        vals = torch.tensor([int(x, 16) for x in row], dtype=torch.int64)
        flat[_vec_flat_indices(2 * (base_reg + r))] = vals


def _bits_to_i32(bits: torch.Tensor) -> torch.Tensor:
    """uint32 bit patterns (int64 tensor) -> torch.int32 with the same bits."""
    return bits.sub(torch.where(bits >= 2**31, 2**32, 0)).to(torch.int32)


# The harness Int32 path carries SIGN-MAGNITUDE patterns between L1 and Dst:
# the LREG two's-complement value v round-trips as L1 pattern
# sign(v) << 31 | |v|  (measured on the pinned sim, lane FK; the SFPLOAD
# INT32 arm decodes SM -> two's complement and the store re-encodes).  The
# fixture's register-chain contract speaks LREG values, so the probe encodes
# on the way in and decodes on the way out.  |v| = 2^31 is unrepresentable
# in SM32 — the fixtures avoid it.
def _sm_encode(bits: torch.Tensor) -> torch.Tensor:
    """uint32 two's-complement LREG pattern -> SM32 L1 pattern."""
    neg = bits >= 2**31
    mag = torch.where(neg, (2**32 - bits) % 2**32, bits)
    assert bool((mag < 2**31).all()), "SM32 cannot represent magnitude 2^31"
    return torch.where(neg, mag + 2**31, mag)


def _sm_decode(bits: torch.Tensor) -> torch.Tensor:
    """SM32 L1 pattern -> uint32 two's-complement LREG pattern."""
    neg = bits >= 2**31
    mag = bits % 2**31
    return torch.where(neg, (2**32 - mag) % 2**32, mag)


def _run_chain(demand_op, contract, bits_tile, is_int, param0=0, param1=0):
    """One tile of raw 32-bit patterns through the row-chain probe; returns
    the flat 1024-element result as uint32 bit patterns."""
    if is_int:
        formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
        src_A = _bits_to_i32(_sm_encode(bits_tile))
    else:
        formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
        src_A = _bits_to_i32(bits_tile).view(torch.float32)
    torch_format = format_dict[formats.input_format]
    src_B = torch.zeros(1024, dtype=torch_format)

    configuration = TestConfig(
        "sources/sfpu_crosslane_demand_test.cpp",
        formats,
        templates=[
            DEMAND_CHAIN(demand_op=demand_op, demand_contract=contract),
            BLAZE_PARAMS(param0_bits=param0, param1_bits=param1),
            APPROX_MODE(ApproximationMode.No),
        ],
        runtimes=[TILE_COUNT(1)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        unpack_to_dest=True,
        dest_acc=DestAccumulation.Yes,
        compile_time_formats=True,
    )
    res = configuration.run().result
    res_t = torch.tensor(res[:1024], dtype=torch_format)
    bits = res_t.view(torch.int32).to(torch.int64) % (2**32)
    if is_int:
        bits = _sm_decode(bits)
    return bits


_EMA_CONTRACTS = {1: "out_rows_fma", 2: "out_rows_mul_add"}


# ---------------------------------------------------------------------------
# THIRD-VALUE FINDING (lane FK, 2026-08-22, pinned sim 32489dda4fd6): the
# device matches NEITHER stored ema arithmetic contract exactly.  The real
# contract is the BlackHole FMA datapath (craq-sim src/fma.cpp
# fma_model_bh, the silicon-mirroring model): the 48-bit product mantissa is
# STICKY-TRUNCATED to 24+3 bits before the aligned add, then rounded once to
# nearest-even — between the fixture's "fma" (exact product, ~30/32 lanes
# agree) and "mul_add" (pre-rounded product) candidates.  The port below is
# the faithful python restatement; the sim gate asserts BIT-EXACT against
# it, and additionally reports both stored contracts' agreement so the
# fixture record stays adjudicated.  SECOND FINDING: the "mul_add" spelling
# (separate rounded mul then add) is NOT COMPILABLE as written — the rvtt
# combiner fuses mul+add back into one SFPMAD (device bits for both
# EMA contract arms are identical), so the only reachable device contract is
# the bh-fma one.
# ---------------------------------------------------------------------------
def _bh_fma(x: int, y: int, z: int) -> int:
    """Faithful port of craq-sim fma_model_bh (x*y + z on fp32 bits)."""

    def unpack(v):
        e = (v >> 23) & 255
        m = (v & 0x7FFFFF) ^ 0x800000
        if e == 0:
            m = 0  # flush denormals
        return e, m

    x_e, x_m = unpack(x)
    y_e, y_m = unpack(y)
    z_e, z_m = unpack(z)
    z_sign = z & 0x80000000

    p_sign = (x ^ y) & 0x80000000
    p_m = x_m * y_m
    p_e = x_e + y_e - 23 - 127

    p_m <<= 3
    z_m <<= 3

    # Realign p_m to match z_m (removing 23 bits, sticky).
    p_m = (p_m >> 23) | (1 if (p_m & 0x7FFFFF) else 0)
    p_e += 23

    if x_e == 255 or y_e == 255 or p_e >= 255 or z_e == 255:
        if (
            (x_e == 255 and (x_m != 0x800000 or y_m == 0))
            or (y_e == 255 and (y_m != 0x800000 or x_m == 0))
            or (z_e == 255 and z_m != 0x4000000)
            or (z_e == 255 and (x_e == 255 or y_e == 255) and (z_sign != p_sign))
        ):
            return 0x7FC00000
        if z_e == 255:
            return z
        return p_sign | 0x7F800000

    if p_m == 0 or p_e < 0:
        return z if z_m else (z_sign & p_sign)

    def semi_sticky(var, s, width):
        if s >= width:
            return 0
        v = var >> s
        if v and (v << s) != var:
            v |= 1
        return v

    r_e = max(p_e, z_e)
    if p_e < r_e:
        p_m = semi_sticky(p_m, r_e - p_e, 64)
    if z_e < r_e:
        z_m = semi_sticky(z_m, r_e - z_e, 32)
    r_sign = p_sign if p_m >= z_m else z_sign
    if z_sign != r_sign:
        z_m = (~z_m) & 0xFFFFFFFF
    if p_sign != r_sign:
        p_m = (~p_m) & 0xFFFFFFFFFFFFFFFF
    r_m = (z_m + p_m + (1 if p_sign != z_sign else 0)) & 0xFFFFFFFF

    if r_m == 0:
        return z_sign & p_sign

    n = 5 - (32 - r_m.bit_length())
    r_e += n
    if r_e >= 255:
        return r_sign | 0x7F800000
    if r_e <= 0:
        n += 1
        r_e = 0
    if n <= 0:
        r_m = (r_m << -n) & 0xFFFFFFFF
    else:
        r_m = (r_m >> n) | (1 if (r_m & (n | 1)) else 0)

    r = ((r_e << 23) + ((r_m >> 3) & 0x7FFFFF)) & 0xFFFFFFFF
    r += 1 if ((r_m & 7) + (r & 1)) > 4 else 0
    if not (r >> 23):
        r = 0
    return (r_sign | r) & 0xFFFFFFFF


def _bh_ema_chain(x_rows, y0, alpha, beta):
    """Device-contract ema chain: t = bh_fma(beta, y, +0); y = bh_fma(alpha,
    x, t) — the typed mul lowers to a MAD against the +0 constant."""
    y = [int(h, 16) for h in y0]
    rows = []
    for row in x_rows:
        ny = []
        for lane, xh in enumerate(row):
            t = _bh_fma(beta, y[lane], 0)
            ny.append(_bh_fma(alpha, int(xh, 16), t))
        y = ny
        rows.append(ny)
    return rows


@pytest.mark.parametrize("contract", [1, 2], ids=lambda c: _EMA_CONTRACTS[c])
def test_crosslane_demand_ema_chain(contract):
    fx = _fixture("ema")
    for case in fx["cases"]:
        bits_tile = torch.zeros(1024, dtype=torch.int64)
        _place_rows(bits_tile, case["x_rows"], base_reg=0)
        _place_rows(bits_tile, [case["y0"]], base_reg=8)
        alpha = int(case["alpha"], 16)
        # beta = f32_round(1 - alpha), the oracle's derivation.
        beta = struct.unpack(
            "<I", struct.pack("<f", 1.0 - struct.unpack("<f", struct.pack("<I", alpha))[0])
        )[0]
        out_bits = _run_chain(1, contract, bits_tile, is_int=False, param0=alpha, param1=beta)
        model_rows = _bh_ema_chain(case["x_rows"], case["y0"], alpha, beta)
        agree = {"out_rows_fma": 0, "out_rows_mul_add": 0, "total": 0}
        for r in range(8):
            want = model_rows[r]
            got = out_bits[_vec_flat_indices(2 * r)].tolist()
            for key in ("out_rows_fma", "out_rows_mul_add"):
                stored = [int(x, 16) for x in case[key][r]]
                agree[key] += sum(g == s for g, s in zip(got, stored))
            agree["total"] += len(got)
            bad = [(l, hex(g), hex(w)) for l, (g, w) in enumerate(zip(got, want)) if g != w]
            assert not bad, (
                f"ema chain contract-arm={_EMA_CONTRACTS[contract]} seed={case['seed']} "
                f"register {r}: device bits differ from the BH-FMA device-contract "
                f"model (fma_model_bh port): lanes {bad}"
            )
        print(
            f"ema chain seed={case['seed']}: device == bh-fma model on all "
            f"{agree['total']} lanes; stored-contract agreement "
            f"fma={agree['out_rows_fma']}/{agree['total']} "
            f"mul_add={agree['out_rows_mul_add']}/{agree['total']} "
            f"(third-value finding: the true contract is the sticky-truncated-"
            f"product BH FMA)"
        )


@pytest.mark.parametrize("arm", ["fp32", "int32"])
def test_crosslane_demand_cumsum_chain(arm):
    fx = _fixture("cumsum")
    is_int = arm == "int32"
    for case in fx["cases"]:
        rows_key = "int_rows" if is_int else "fp_rows"
        golden_key = "int_prefix" if is_int else "fp_prefix"
        bits_tile = torch.zeros(1024, dtype=torch.int64)
        _place_rows(bits_tile, case[rows_key], base_reg=0)
        out_bits = _run_chain(3 if is_int else 2, 1, bits_tile, is_int=is_int)
        for r, row in enumerate(case[golden_key]):
            want = [int(x, 16) for x in row]
            got = out_bits[_vec_flat_indices(2 * r)].tolist()
            assert got == want, (
                f"cumsum chain arm={arm} seed={case['seed']} register {r}: "
                f"device bits differ from the fixture golden\n"
                f"got  {[hex(g) for g in got[:8]]}...\n"
                f"want {[hex(w) for w in want[:8]]}..."
            )
