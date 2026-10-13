# llk_analysis one-call check (#58743, ops/blackhole/sfpu_misc): device time of the ttnn ops that reach the misc entry
# points #58827 switches to one 32-row call, run under `python -m tracy -r -p --no-web-server -o <dir> prof_oc_misc.py
# <cases>`; a signpost precedes the R3_NMEAS measured calls of every case, after 2 warm-up calls. The generic-op cases
# (GLU test kernels, onecall_probe.cpp) run on one core.
import gc
import os
import sys

import torch
import ttnn
from tracy import signpost

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bits_oc_misc as B  # noqa: E402

if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")  # card rule of 2026-10-07 12:35 UTC; the mock cluster opens no card

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "6"))
BIG = (1, 1, 4096, 2048)


def hs(dev, dt, tdt=torch.bfloat16, tiles_h=8, tiles_w=8, kind="randn"):
    grid = dev.compute_with_storage_grid_size()
    ncores = grid.x * grid.y
    crs = ttnn.num_cores_to_corerangeset(ncores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    spec = ttnn.ShardSpec(crs, (32 * tiles_h, 32 * tiles_w), ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    shape = (1, 1, 32 * tiles_h * ncores, 32 * tiles_w)
    x = B.with_zeros(shape, 0, tdt) if kind == "zeros" else B.mixed(shape, 0).to(tdt)
    return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem), mem


CASES = {}
CASES["rand_bf16_4096x2048"] = (lambda d: (), lambda d: ttnn.rand(BIG, dtype=ttnn.bfloat16, device=d, seed=1234))
CASES["rand_fp32_4096x2048"] = (lambda d: (), lambda d: ttnn.rand(BIG, dtype=ttnn.float32, device=d, seed=1234))
CASES["rand_bf16_1024x1024"] = (lambda d: (), lambda d: ttnn.rand((1, 1, 1024, 1024), dtype=ttnn.bfloat16, device=d, seed=1234))
CASES["uniform_fp32_4096x2048"] = (lambda d: (B.tt(torch.zeros(BIG), d, ttnn.float32),), lambda d, x: ttnn.uniform(x, 0.0, 1.0, 1234))
CASES["bernoulli_fp32_4096x2048"] = (
    lambda d: (B.tt(torch.rand(BIG, generator=torch.Generator().manual_seed(0)), d, ttnn.float32),),
    lambda d, p: ttnn.bernoulli(p, 1234, dtype=ttnn.float32),
)
CASES["dropout_bf16_4096x2048"] = (
    lambda d: (B.tt(B.mixed(BIG, 1), d),),
    lambda d, x: ttnn.experimental.dropout(x, probability=0.5, scale=2.0, seed=1234),
)
CASES["logical_not_bf16_hs"] = (lambda d: hs(d, ttnn.bfloat16, kind="zeros"), lambda d, x, m: ttnn.logical_not(x, memory_config=m))
CASES["logical_not_int32_hs"] = (lambda d: hs(d, ttnn.int32, torch.int32, kind="zeros"), lambda d, x, m: ttnn.logical_not(x, memory_config=m))
CASES["softsign_bf16_hs"] = (lambda d: hs(d, ttnn.bfloat16), lambda d, x, m: ttnn.softsign(x, memory_config=m))
CASES["softsign_fp32_hs"] = (lambda d: hs(d, ttnn.float32, torch.float32, tiles_h=4), lambda d, x, m: ttnn.softsign(x, memory_config=m))
CASES["softshrink_bf16_hs"] = (lambda d: hs(d, ttnn.bfloat16), lambda d, x, m: ttnn.softshrink(x, lambd=0.5, memory_config=m))
CASES["softshrink_fp32_hs"] = (lambda d: hs(d, ttnn.float32, torch.float32, tiles_h=4), lambda d, x, m: ttnn.softshrink(x, lambd=0.5, memory_config=m))
CASES["gelu_bw_tanh_bf16_2048x2048"] = (
    lambda d: (B.tt(B.mixed((1, 1, 2048, 2048), 9), d), B.tt(B.mixed((1, 1, 2048, 2048), 10), d)),
    lambda d, g, x: ttnn.gelu_bw(g, x, variant=ttnn.GeluVariant.Tanh),
)
CASES["gelu_bw_tanh_fp32_2048x2048"] = (
    lambda d: (B.tt(B.mixed((1, 1, 2048, 2048), 9), d, ttnn.float32), B.tt(B.mixed((1, 1, 2048, 2048), 10), d, ttnn.float32)),
    lambda d, g, x: ttnn.gelu_bw(g, x, variant=ttnn.GeluVariant.Tanh),
)


def _mp_setup(d):
    n, h, w, c = 1, 256, 256, 64
    x = torch.randn((1, 1, n * h * w, c), generator=torch.Generator().manual_seed(11)).to(torch.bfloat16)
    return (B.tt(x, d, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), n, h, w, c)


def _mp_run(d, x, n, h, w, c):
    return ttnn.max_pool2d(
        input_tensor=x, batch_size=n, input_h=h, input_w=w, channels=c, kernel_size=[9, 9], stride=[4, 4],
        padding=[0, 0], dilation=[1, 1], deallocate_input=False, reallocate_halo_output=True, return_indices=True,
    )


CASES["max_pool2d_indices_9x9_256x256x64"] = (_mp_setup, _mp_run)


def _ln_setup(d, w=1000, h=4096):
    # Moreh layer norm over a width that is not a multiple of 32: its compute kernel masks the last tile column.
    from tests.ttnn.unit_tests.operations.test_utils import to_ttnn

    g = torch.Generator().manual_seed(12)
    x = torch.randn((1, 1, h, w), generator=g).to(torch.bfloat16)
    gamma = torch.randn((w,), generator=g).to(torch.bfloat16)
    beta = torch.randn((w,), generator=g).to(torch.bfloat16)
    out = to_ttnn(torch.empty_like(x), device=d, dtype=ttnn.bfloat16)
    return (to_ttnn(x, device=d, dtype=ttnn.bfloat16), to_ttnn(gamma, device=d, dtype=ttnn.bfloat16),
            to_ttnn(beta, device=d, dtype=ttnn.bfloat16), out)


def _ln_run(d, x, gamma, beta, out):
    return ttnn.operations.moreh.layer_norm(x, 1, 1e-5, gamma, beta, output=out, mean=None, rstd=None)


CASES["moreh_layer_norm_bf16_4096x1000"] = (_ln_setup, _ln_run)

# One-core generic-op cases: 256 tiles each.
NT1 = 256


def _glu_case(kind, kernel, fp32):
    import test_clamped_silu_glu_sfpu as C
    import test_situ_glu_sfpu as S

    def setup(d):
        if kind == "situ":
            g, u = S._coverage_inputs(NT1)
        else:
            g, u = C._coverage_inputs(NT1)
        return (g, u)

    def run(d, g, u):
        if kind == "situ":
            dt, pb = S.IN_DTYPES["bf16"]
            return S._run(d, g, u, dt, pb, fp32, 0) if kernel is None else S._run(d, g, u, dt, pb, fp32, 0, kernel)
        dt, pb = C.IN_DTYPES["bf16"]
        args = (d, g, u, dt, pb, fp32, 0, 1, 0)
        return C._run(*args) if kernel is None else C._run(*args, kernel)

    return setup, run


for fp32 in (False, True):
    CASES[f"situ_glu_math_{int(fp32)}"] = _glu_case("situ", None, fp32)
    CASES[f"situ_glu_pack_{int(fp32)}"] = _glu_case("situ", "situ_glu_pack.cpp", fp32)
    CASES[f"clamped_silu_glu_math_{int(fp32)}"] = _glu_case("clamped", None, fp32)
    CASES[f"clamped_silu_glu_pack_{int(fp32)}"] = _glu_case("clamped", "clamped_silu_glu_pack.cpp", fp32)


def _probe_case(op, dt, page, fp32, a_fn, b_fn):
    def setup(d):
        return (a_fn(), b_fn())

    def run(d, a, b):
        return B._probe(d, op, a, b, dt, page, fp32, NT1)

    return setup, run


T = B.TILE
_bfa = lambda s: B.mixed((NT1 * T,), s).to(torch.bfloat16)  # noqa: E731
_m01 = lambda s: (torch.rand(NT1 * T, generator=torch.Generator().manual_seed(s)) < 0.5).float()  # noqa: E731
_i32 = lambda s: torch.randint(-(2**31), 2**31 - 1, (NT1 * T,), generator=torch.Generator().manual_seed(s), dtype=torch.int64).to(torch.int32)  # noqa: E731
CASES["probe_identity_bf16"] = _probe_case(0, ttnn.bfloat16, T * 2, False, lambda: _bfa(20), lambda: _bfa(21))
CASES["probe_identity_uint32"] = _probe_case(1, ttnn.uint32, T * 4, True, lambda: _i32(24) & 0x7FFFFFFF, lambda: _i32(25) & 0x7FFFFFFF)
CASES["probe_mask_bf16"] = _probe_case(2, ttnn.bfloat16, T * 2, False, lambda: _bfa(26), lambda: _m01(27).to(torch.bfloat16))
CASES["probe_mask_int32"] = _probe_case(3, ttnn.int32, T * 4, True, lambda: _i32(28), lambda: _m01(29).to(torch.int32))
CASES["probe_mask_posinf_bf16"] = _probe_case(4, ttnn.bfloat16, T * 2, False, lambda: _bfa(30), lambda: _m01(31).to(torch.bfloat16))
CASES["probe_copy_dest_bf16"] = _probe_case(5, ttnn.bfloat16, T * 2, False, lambda: _bfa(32), lambda: _bfa(33))
CASES["probe_copy_dest_int32"] = _probe_case(7, ttnn.int32, T * 4, True, lambda: _i32(36), lambda: _i32(37))
CASES["probe_copy_dest_deprecated_bf16"] = _probe_case(8, ttnn.bfloat16, T * 2, False, lambda: _bfa(38), lambda: _bfa(39))


def main():
    names = sys.argv[1:] or list(CASES)
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        for name in names:
            setup, run = CASES[name]
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                ttnn.ReadDeviceProfiler(dev)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                ttnn.synchronize_device(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
