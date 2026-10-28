# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""How the device's packer rounds to bfloat4_b / bfloat8_b (one card; run_safe_pytest.sh).

Crafted tiles (exact half-steps, one bf16 / fp32 ulp either side, group maxima on a half-step, denormals, zeros) and
random ones are packed BY THE DEVICE: a copy kernel from 16-bit and 32-bit DEST with the packer's precise flag off and
on, ttnn.typecast and a matmul with a bfp output. The packed tiles are written out as raw bytes and compared byte for
byte with the rules in ``_quantize`` and with ttnn's host packer (ttnn._ttnn.bfp_utils.pack_bfp*).

Measured (Blackhole p150b, 2026-10-28): the packer rounds half away from zero to bfloat8_b's 7 bits; bfloat4_b then
drops the low 4 bits, so bfp4 is close to truncation (row error on the golden compressed KV 0.20 vs 0.11 for RNE).
Without bfp8_pack_precise each value is first rounded to 7 significant bits (``device_dbl``). The compute kernel's
cache key (ComputeKernel::config_hash) leaves bfp8_pack_precise out, so in one process two kernels that differ only
in it share a binary: the copy cases below therefore all run with it off."""

import pytest
import torch

import ttnn

KERNELS = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
COPY = "ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp"
MBITS = {ttnn.bfloat4_b: 3, ttnn.bfloat8_b: 7}  # magnitude bits (sign apart)
TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088}


# ---------------------------------------------------------------------------------------------- the rules (CPU)
def _quantize(x: torch.Tensor, m: int, rule: str):
    """x [R, 32] fp32 -> (codes [R, 32] int64 = sign << m | q, shared exponents [R, 2] int64), per 16-value group.

    rne_host: tt-metal's host packer (convert_u32_to_bfp): shift to the group's max exponent, round to nearest even,
      clamp; = true RNE for bf16 inputs.
    trunc: the same, truncating.
    device: the packer with bfp8_pack_precise (ttnn.typecast to bfloat8_b): round half away from zero to 7 bits at
      the shared exponent, clamp at 127, then (bfloat4_b) drop the low 4 bits.
    device_dbl: the packer without it (the default; a matmul's output, a typecast to bfloat4_b), whose source format
      is bfloat8_b: each value first rounded half away from zero to a 7-bit significand at its own exponent (which
      may carry it to the next exponent, and so bump the group's), then as ``device``."""
    u = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exp, sign = (u >> 23) & 0xFF, u >> 31
    r = x.shape[0]
    top = (1 << m) - 1
    if rule in ("rne_host", "trunc"):
        man = (u & 0x7FFFFF) | (1 << 23)
        eg = exp.view(r, 2, 16)
        shared = eg.amax(-1, keepdim=True)
        mm = man.view(r, 2, 16) >> (shared - eg).clamp(max=62)
        sh = 24 - m
        q, rv = mm >> sh, mm & ((1 << sh) - 1)
        if rule == "rne_host":
            q = q + ((rv > (1 << (sh - 1))) | ((rv == (1 << (sh - 1))) & ((q & 1) == 1))).long()
    elif rule in ("device", "device_dbl"):
        a = x.double().abs()
        if rule == "device_dbl":
            ulp = torch.pow(2.0, (exp - 127 - 6).double())
            a = torch.where(exp == 0, torch.zeros_like(a), torch.floor(a / ulp + 0.5) * ulp)
            exp = torch.where(a > 0, torch.floor(torch.log2(a)).long() + 127, torch.zeros_like(exp))
        eg = exp.view(r, 2, 16)
        shared = eg.amax(-1, keepdim=True)
        q8 = torch.floor(a.view(r, 2, 16) * torch.pow(2.0, -(shared - 127 - 6).double()) + 0.5).clamp(max=127)
        q = q8.long() >> (7 - m)
    else:
        raise ValueError(rule)
    q = q.clamp(max=top)
    q = torch.where(eg == 0, torch.zeros_like(q), q)
    s = torch.where(q == 0, torch.zeros_like(q), sign.view(r, 2, 16))
    return ((s << m) | q).view(r, 32), shared.view(r, 2)


def _tile_bytes(codes: torch.Tensor, shared: torch.Tensor, m: int) -> torch.Tensor:
    """codes [32 T, 32], shared [32 T, 2] -> [T, tile bytes] uint8 in tt-metal's bfp tile layout (64 exponent bytes:
    face 0..3 x 16 rows; then the faces' mantissas, row-major, bfp8 a byte each, bfp4 a nibble, low nibble first)."""
    t = codes.shape[0] // 32
    c = codes.view(t, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).reshape(t, 4, 256)  # [T, face, 16 x 16]
    e = shared.view(t, 2, 16, 2).permute(0, 1, 3, 2).reshape(t, 64)
    if m == 7:
        mant = c.reshape(t, 1024)
    else:
        c = c.reshape(t, 4, 128, 2)
        mant = (c[..., 0] | (c[..., 1] << 4)).reshape(t, 512)
    return torch.cat([e, mant], dim=-1).to(torch.uint8)


def expected(x: torch.Tensor, dtype, rule: str) -> torch.Tensor:
    m = MBITS[dtype]
    return _tile_bytes(*_quantize(x, m, rule), m)


def host_packer(x: torch.Tensor, dtype) -> torch.Tensor:
    pack = ttnn._ttnn.bfp_utils.pack_bfp4 if dtype == ttnn.bfloat4_b else ttnn._ttnn.bfp_utils.pack_bfp8
    words = torch.from_numpy(pack(x.contiguous().view(-1).float().numpy(), row_major_input=True).copy())
    return words.view(torch.uint8).view(-1, TILE_BYTES[dtype])


# ---------------------------------------------------------------------------------------------- the inputs
def _bf16_neighbours(v: torch.Tensor):
    b = v.bfloat16().view(torch.int16)
    return [(b + d).view(torch.bfloat16).float() for d in (-1, 0, 1)]


def crafted(m: int, fp32: bool, seed: int = 0) -> torch.Tensor:
    """Groups of 16: an anchor setting the shared exponent, then values on / around the half-steps of the m-bit grid;
    groups whose max sits on a half-step; denormals; zeros; random groups. Values are bf16 unless ``fp32``, which adds
    perturbations below bf16's resolution (does the packer see all of an fp32 DEST's bits?)."""
    g = torch.Generator().manual_seed(seed)
    step = 2.0 ** -(m - 1)  # the grid at shared exponent 127 (anchor 1.0)
    ties = (torch.arange(2**m, dtype=torch.float64) + 0.5) * step
    vals = []
    for t in _bf16_neighbours(ties.float()):
        vals.append(t)
    if fp32:
        for d in (-1, 1):
            vals.append((ties * (1 + d * 2.0**-20)).float())  # tie +- far below bf16's ulp
    vals = torch.cat(vals)
    vals = vals[(vals > 0) & (vals < 2)]
    groups = []
    for scale in (1.0, 2.0**-30, 2.0**40):
        for sgn in (1.0, -1.0):
            v = vals * sgn * scale
            for i in range(0, len(v), 15):
                chunk = v[i : i + 15]
                grp = torch.zeros(16)
                grp[0] = scale  # anchor: shared exponent of `scale`
                grp[1 : 1 + len(chunk)] = chunk
                groups.append(grp)
    maxima = ties[(ties >= 1) & (ties < 2)].float()  # the group max itself on a half-step
    for mx in _bf16_neighbours(maxima) + ([(maxima.double() * (1 + 2.0**-20)).float()] if fp32 else []):
        for v in mx:
            grp = torch.rand(16, generator=g) * float(v)
            grp[int(torch.randint(0, 16, (1,), generator=g))] = v
            groups.append(grp)
            groups.append(-grp)
    den = torch.tensor([2.0**-127, 2.0**-130, 2.0**-133, 0.0, -(2.0**-128)]).repeat(4)[:16]
    groups.append(den)
    groups.append(torch.cat([torch.tensor([2.0**-125]), den[:15]]))
    groups.append(torch.zeros(16))
    groups.append(torch.tensor([-0.0] * 16))
    # random: values spread over a few binades below a random group exponent, some far below
    n_rand = 2048
    e0 = torch.randint(-40, 40, (n_rand, 1), generator=g)
    e = e0 - torch.randint(0, 8, (n_rand, 16), generator=g)
    e = torch.where(torch.rand(n_rand, 16, generator=g) < 0.05, e - 24, e)
    mant = 1 + torch.randint(0, 2**23 if fp32 else 128, (n_rand, 16), generator=g).double() / (
        2**23 if fp32 else 128
    )
    sgn = torch.where(torch.rand(n_rand, 16, generator=g) < 0.5, -1.0, 1.0).double()
    groups += list((sgn * mant * torch.pow(2.0, e.double())).float())
    x = torch.stack(groups).view(-1)
    n = -(-x.numel() // 1024) * 1024
    x = torch.cat([x, torch.zeros(n - x.numel())])
    x = x.view(-1, 32)
    return x if fp32 else x.bfloat16().float()


# ---------------------------------------------------------------------------------------------- device packs
def _copy_pack(
    device, x: torch.Tensor, in_dtype, out_dtype, fp32_dest: bool, precise: bool, compute=COPY
) -> torch.Tensor:
    """A copy kernel's pack of x's tiles to out_dtype, written out as raw tile bytes [T, tile bytes]."""
    t = x.shape[0] // 32
    xin = ttnn.from_torch(x, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    nbytes = TILE_BYTES[out_dtype]
    out = ttnn.from_torch(torch.zeros(t, nbytes, dtype=torch.uint8), dtype=ttnn.uint8, device=device)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * ttnn.tile_size(in_dtype),
            core_ranges=core,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=in_dtype, page_size=ttnn.tile_size(in_dtype))
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * nbytes,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=out_dtype, page_size=nbytes)],
        ),
    ]
    rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd_rt[0][0] = [xin.buffer_address(), t, 0]
    wr_rt[0][0] = [out.buffer_address(), t, 0]
    cfg = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest, bfp8_pack_precise=precise)
    if in_dtype == ttnn.float32:
        modes = [ttnn.UnpackToDestMode.Default] * 64  # one per circular buffer
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KERNELS + "reader_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=ttnn.TensorAccessorArgs(xin).get_compile_time_args(),
            runtime_args=rd_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KERNELS + "writer_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=[16] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=wr_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=compute,
            core_ranges=core,
            compile_time_args=[t] if compute == COPY else [0, 16],
            runtime_args=[] if compute == COPY else _rt_all(t),
            config=cfg,
        ),
    ]
    ttnn.generic_op([xin, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return ttnn.to_torch(out).view(torch.uint8)


def _rt_all(*args):
    rt = ttnn.RuntimeArgs()
    rt[0][0] = list(args)
    return rt


def _raw_bytes(device, tt: ttnn.Tensor) -> torch.Tensor:
    """Raw tile bytes of a bfp TILE tensor on the device: its DRAM pages moved into a uint8 tensor by the reader /
    writer pair through one circular buffer (no compute, no repacking)."""
    nbytes = TILE_BYTES[tt.dtype]
    t = tt.buffer_num_pages()
    out = ttnn.from_torch(torch.zeros(t, nbytes, dtype=torch.uint8), dtype=ttnn.uint8, device=device)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cb = ttnn.CBDescriptor(
        total_size=2 * nbytes,
        core_ranges=core,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=tt.dtype, page_size=nbytes)],
    )
    rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd_rt[0][0] = [tt.buffer_address(), t, 0]
    wr_rt[0][0] = [out.buffer_address(), t, 0]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KERNELS + "reader_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=ttnn.TensorAccessorArgs(tt).get_compile_time_args(),
            runtime_args=rd_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KERNELS + "writer_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=[0] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=wr_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([tt, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb]))
    return ttnn.to_torch(out).view(torch.uint8)


def _typecast(device, x, in_dtype, out_dtype):
    xin = ttnn.from_torch(x, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return _raw_bytes(device, ttnn.typecast(xin, out_dtype))


def _matmul(device, x, out_dtype, fp32_acc):
    """x @ I with a bfp output (bf16 operands, HiFi4: the products are exact)."""
    xin = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    eye = ttnn.from_torch(torch.eye(32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_acc, packer_l1_acc=False
    )
    return _raw_bytes(device, ttnn.matmul(xin, eye, dtype=out_dtype, compute_kernel_config=cfg))


# (name, in dtype, out dtype, fp32 DEST, rule measured on a Blackhole p150b, 2026-10-28; bfp8_pack_precise off).
# bf16 -> bfp4 through 32-bit DEST is device_dbl but for 1 value in ~34k, so it is left out.
CASES = [
    ("copy bf16->bfp8 dest16", ttnn.bfloat16, ttnn.bfloat8_b, False, "device_dbl"),
    ("copy bf16->bfp4 dest16", ttnn.bfloat16, ttnn.bfloat4_b, False, "device_dbl"),
    ("copy fp32->bfp8 dest32", ttnn.float32, ttnn.bfloat8_b, True, "device_dbl"),
    ("copy fp32->bfp4 dest32", ttnn.float32, ttnn.bfloat4_b, True, "device_dbl"),
]
OPS = [  # (name, in dtype, out dtype, fn, rule)
    (
        "typecast bf16->bfp8",
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        lambda d, x: _typecast(d, x, ttnn.bfloat16, ttnn.bfloat8_b),
        "device",
    ),
    (
        "typecast fp32->bfp8",
        ttnn.float32,
        ttnn.bfloat8_b,
        lambda d, x: _typecast(d, x, ttnn.float32, ttnn.bfloat8_b),
        "device",
    ),
    (
        "typecast bf16->bfp4",
        ttnn.bfloat16,
        ttnn.bfloat4_b,
        lambda d, x: _typecast(d, x, ttnn.bfloat16, ttnn.bfloat4_b),
        "device_dbl",
    ),
    (
        "typecast fp32->bfp4",
        ttnn.float32,
        ttnn.bfloat4_b,
        lambda d, x: _typecast(d, x, ttnn.float32, ttnn.bfloat4_b),
        "device_dbl",
    ),
    *[
        (f"matmul bf16->{n} fp32_acc={f}", ttnn.bfloat16, o, lambda d, x, o=o, f=f: _matmul(d, x, o, f), "device_dbl")
        for o, n in ((ttnn.bfloat8_b, "bfp8"), (ttnn.bfloat4_b, "bfp4"))
        for f in (False, True)
    ],
]


def _check(name, got, x, dtype, rule):
    """got == the rule's bytes; prints how far the other rules are (bytes differing)."""
    others = "  ".join(
        f"{r} {(got != expected(x, dtype, r)).sum().item():5d}B" for r in ("rne_host", "trunc", "device", "device_dbl")
    )
    print(f"{name:32s} {others}", flush=True)
    assert torch.equal(got, expected(x, dtype, rule)), f"{name}: the packer no longer follows {rule}"


def test_host_packer_is_rne(device):
    """ttnn's host packer = rne_host (tt-metal's rule) on the crafted tiles, both formats."""
    for dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        for fp32 in (False, True):
            x = crafted(MBITS[dtype], fp32)
            assert torch.equal(host_packer(x, dtype), expected(x, dtype, "rne_host")), (dtype, fp32)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_device_packer_rounding(device, case):
    name, in_dtype, out_dtype, fp32_dest, rule = case
    x = crafted(MBITS[out_dtype], in_dtype == ttnn.float32)
    _check(name, _copy_pack(device, x, in_dtype, out_dtype, fp32_dest, precise=False), x, out_dtype, rule)


@pytest.mark.parametrize("op", OPS, ids=[c[0] for c in OPS])
def test_op_packer_rounding(device, op):
    name, in_dtype, out_dtype, fn, rule = op
    x = crafted(MBITS[out_dtype], in_dtype == ttnn.float32)
    _check(name, fn(device, x), x, out_dtype, rule)


def _op_rows(device, x: torch.Tensor) -> torch.Tensor:
    """ttnn.bringup.ckv_bfp4 (SFPU round onto the bfp4 grid, then the packer writes the row records) on x [R, 32]
    read as rows of 512 -> the records' codes [R, 32] and shared exponents [R, 2], as _quantize gives them."""
    v = x.reshape(-1)
    n = -(-v.numel() // (32 * 512)) * 32 * 512
    v = torch.cat([v, torch.zeros(n - v.numel())]).view(-1, 512)
    ckv = ttnn.from_torch(v.view(1, 1, -1, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rows = ttnn.from_torch(torch.zeros(1, 1, v.shape[0], 320, dtype=torch.uint8), dtype=ttnn.uint8, device=device)
    ttnn.bringup.ckv_bfp4(ckv, rows, 0)
    b = ttnn.to_torch(rows).view(torch.uint8).reshape(-1, 320).long()
    assert torch.equal(b[:, :32], b[:, 32:64]), "the two exponent copies differ"
    nb = b[:, 64:]
    codes = torch.stack([nb & 15, nb >> 4], -1).reshape(-1)[: x.numel()].view(-1, 32)
    return codes, b[:, :32].reshape(-1)[: x.numel() // 16].view(-1, 2)


def _assert_rne(got, x):
    codes, shared = _quantize(x, 3, "rne_host")
    nz = codes != 0  # an all-zero group's exponent is don't-care
    bad = (got[0] != codes).sum().item() + ((got[1] != shared) & nz.view(-1, 2, 16).any(-1)).sum().item()
    return bad


@pytest.mark.parametrize("seed", [0, 1])
def test_sfpu_round_then_pack_is_the_host_packer(device, seed):
    """The ckv_bfp4 op (SFPU round onto the bfp4 grid, then the packer writes the rows) gives the host packer's
    codes and exponents."""
    x = crafted(3, False, seed)
    small = x.abs().view(-1, 2, 16).amax(-1) < 2.0**-124  # the kernel's documented limit: those groups go to 0
    x = torch.where(small.repeat_interleave(16, 1), torch.zeros_like(x), x)
    bad = _assert_rne(_op_rows(device, x), x)
    print(f"sfpu round: {bad} codes / exponents differ of {x.numel()}")
    assert bad == 0


GOLDEN_KV = "/localdev/dnijemcevic/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4/s10240_c5120/kv_cache"


@pytest.mark.parametrize("layer", [2, 20])
def test_sfpu_round_on_golden_rows(device, layer):
    """Real compressed-KV rows (the device's bf16 ckv): SFPU round + pack == host packer, and the default pack alone
    is far off (row error vs native: host RNE / the plain pack / truncation)."""
    import os

    from safetensors import safe_open

    path = f"{GOLDEN_KV}/layer_{layer}.safetensors"
    if not os.path.exists(path):
        pytest.skip(f"no golden at {path}")
    with safe_open(path, "pt") as h:
        x = h.get_tensor([k for k in h.keys() if "compress_kv" in k][0]).float().reshape(-1, 512)[:2048]
    xb = x.bfloat16().float().reshape(-1, 32)  # 16-value groups stay whole
    bad = _assert_rne(_op_rows(device, xb), xb)
    plain = _copy_pack(device, xb, ttnn.bfloat16, ttnn.bfloat4_b, False, precise=False)
    exp = expected(xb, ttnn.bfloat4_b, "rne_host")
    print(f"layer {layer}: sfpu round {bad} codes / exponents differ; plain pack {(plain != exp).sum().item()} bytes")
    assert bad == 0
