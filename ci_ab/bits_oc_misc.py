# llk_analysis one-call check (#58743, ops/blackhole/sfpu_misc): raw output words of every ttnn op that reaches a misc
# entry point #58827 switches to one 32-row call, on the tree under test; two trees are compared word for word offline.
# Each case writes $R3_BITS_OUT/<case>.npz. A case whose op is missing on the tree is skipped. The probe cases drive the
# entry points without a ttnn op (identity_tile, identity_tile_uint32, mask_tile, mask_posinf_tile, copy_dest_values)
# through ttnn.generic_op and onecall_probe.cpp.
import os
import sys

import numpy as np
import pytest
import torch
import ttnn

if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")  # card rule of 2026-10-07 12:35 UTC; the mock cluster opens no card

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("R3_BITS_OUT", os.path.join(HERE, "bits_out"))
MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))
sys.path.insert(0, os.path.join(os.environ.get("TT_METAL_HOME", "."), "tests/ttnn/unit_tests/operations/eltwise"))


@pytest.fixture(scope="module")
def dev():
    d = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield d
    ttnn.close_device(d)


def words(t):
    t = t.contiguous()
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        return t.view(torch.int16).numpy()
    if t.dtype == torch.float32:
        return t.view(torch.int32).numpy()
    return t.numpy()


def tt(x, d, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=None):
    return ttnn.from_torch(x, dtype=dt, layout=layout, device=d, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG)


def mixed(shape, seed=0, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * scale
    flat = x.view(-1)
    special = torch.tensor([0.0, -0.0, 0.5, -0.5, 1.0, -1.0, float("inf"), float("-inf"), 1e-40, -1e-40, 3.0e38, -3.0e38])
    flat[: special.numel() * 64] = special.repeat(64)[: min(flat.numel(), special.numel() * 64)]
    return x[..., :].reshape(shape)


def with_zeros(shape, seed, dt_torch):
    g = torch.Generator().manual_seed(seed)
    if dt_torch.is_floating_point:
        x = torch.randn(shape, generator=g)
    else:
        x = torch.randint(-1000, 1000, shape, generator=g, dtype=torch.int64)
    keep = torch.rand(shape, generator=g) < 0.5
    return (x * keep).to(dt_torch)


CASES = {}


def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn

    return deco


SHAPE = (1, 1, 1024, 1024)

# rand_tile: ttnn.rand, ttnn.uniform, ttnn.bernoulli, ttnn.randn, ttnn.sampling
for dt_name, dt in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
    case(f"rand_{dt_name}")(lambda d, dt=dt: [ttnn.rand(SHAPE, dtype=dt, device=d, seed=1234)])
    case(f"rand_{dt_name}_lowhigh")(lambda d, dt=dt: [ttnn.rand(SHAPE, dtype=dt, device=d, low=-3.0, high=5.0, seed=99)])
    case(f"rand_{dt_name}_tiny")(lambda d, dt=dt: [ttnn.rand(SHAPE, dtype=dt, device=d, low=0.0, high=1e-30, seed=7)])
    case(f"uniform_{dt_name}")(lambda d, dt=dt: [ttnn.uniform(tt(torch.zeros(SHAPE), d, dt), 0.0, 1.0, 1234)])
case("bernoulli_fp32")(lambda d: [ttnn.bernoulli(tt(torch.rand(SHAPE, generator=torch.Generator().manual_seed(0)), d, ttnn.float32), 1234, dtype=ttnn.float32)])


@case("randn_bf16")
def _randn(d):
    if not hasattr(ttnn, "randn"):
        pytest.skip("no ttnn.randn")
    return [ttnn.randn(SHAPE, device=d, dtype=ttnn.bfloat16, seed=1234)]


@case("sampling")
def _sampling(d):
    users, width = 32, 256
    shape = [1, 1, users, width]
    g = torch.Generator().manual_seed(0)
    vals = ttnn.from_torch(torch.randn(shape, generator=g), device=d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    idx = ttnn.from_torch(torch.arange(0, width, dtype=torch.int32).expand(shape), device=d, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    k = ttnn.from_torch(torch.tensor([10, 15, 20, 25, 30] * 6 + [10, 20]), device=d, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p = ttnn.from_torch(torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]), device=d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    temp = ttnn.from_torch(torch.ones(32), device=d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    return [ttnn.sampling(vals, idx, k=k, p=p, temp=temp, seed=2024)]


# dropout_tile: ttnn.experimental.dropout (bfloat16 only)
for dt_name, dt in (("bf16", ttnn.bfloat16),):
    case(f"dropout_{dt_name}")(lambda d, dt=dt: [ttnn.experimental.dropout(tt(mixed(SHAPE, 1), d, dt), probability=0.5, scale=2.0, seed=1234)])
    case(f"dropout_{dt_name}_p01")(lambda d, dt=dt: [ttnn.experimental.dropout(tt(mixed(SHAPE, 2), d, dt), probability=0.1, scale=1.0 / 0.9, seed=77)])

# logical_not_tile
for dt_name, dt, tdt in (
    ("bf16", ttnn.bfloat16, torch.bfloat16),
    ("fp32", ttnn.float32, torch.float32),
    ("int32", ttnn.int32, torch.int32),
    ("uint32", ttnn.uint32, torch.int32),
    ("uint16", ttnn.uint16, torch.int32),
):
    def _ln(d, dt=dt, tdt=tdt, dt_name=dt_name):
        x = with_zeros(SHAPE, 3, tdt)
        if dt_name in ("uint32", "uint16"):
            x = x.abs()
        return [ttnn.logical_not(tt(x, d, dt))]

    case(f"logical_not_{dt_name}")(_ln)
    if dt_name in ("bf16", "fp32"):
        case(f"logical_not_{dt_name}_chain")(
            lambda d, dt=dt, tdt=tdt: [ttnn.unary_chain(tt(with_zeros(SHAPE, 3, tdt), d, dt), [ttnn.UnaryWithParam(ttnn.UnaryOpType.LOGICAL_NOT_UNARY)])]
        )

# softsign_tile, softshrink_tile
for dt_name, dt in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
    case(f"softsign_{dt_name}")(lambda d, dt=dt: [ttnn.softsign(tt(mixed(SHAPE, 4), d, dt))])
    case(f"softsign_{dt_name}_chain")(lambda d, dt=dt: [ttnn.unary_chain(tt(mixed(SHAPE, 4), d, dt), [ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTSIGN)])])
    case(f"softshrink_{dt_name}")(lambda d, dt=dt: [ttnn.softshrink(tt(mixed(SHAPE, 5), d, dt), lambd=0.5)])
    case(f"softshrink_{dt_name}_chain")(lambda d, dt=dt: [ttnn.unary_chain(tt(mixed(SHAPE, 5), d, dt), [ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTSHRINK, 0.5)])])


# mask_tile: moreh layer_norm (Float16_b mask on a width that is not a multiple of 32), moreh sum on Int32 (Int32 mask)
def _moreh_ln(d, dims):
    shape = (2, 3, 50, 70)
    g = torch.Generator().manual_seed(6)
    x = torch.randn(shape, generator=g)
    nshape = shape[-dims:]
    gamma = torch.rand(nshape, generator=g) * 2 - 1
    beta = torch.rand(nshape, generator=g) * 2 - 1
    out = ttnn.operations.moreh.layer_norm(tt(x, d), dims, 1e-5, tt(gamma, d), tt(beta, d))
    return [o for o in out if o is not None]


case("moreh_layer_norm_w")(lambda d: _moreh_ln(d, 1))
case("moreh_layer_norm_hw")(lambda d: _moreh_ln(d, 2))


@case("moreh_sum_int32_w")
def _moreh_sum(d):
    g = torch.Generator().manual_seed(8)
    x = torch.randint(-100, 100, (2, 3, 50, 70), generator=g, dtype=torch.int32)
    return [ttnn.operations.moreh.sum(tt(x, d, ttnn.int32), 3, keepdim=True)]


# copy_dest_values: gelu_bw tanh (eltwise_bw_gelu_tanh*.cpp), max_pool2d with indices (compute_mpwi.cpp)
for dt_name, dt in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
    case(f"gelu_bw_tanh_{dt_name}")(
        lambda d, dt=dt: list(ttnn.gelu_bw(tt(mixed((1, 1, 512, 512), 9), d, dt), tt(mixed((1, 1, 512, 512), 10), d, dt), variant=ttnn.GeluVariant.Tanh))
    )


@case("max_pool2d_indices")
def _mp(d):
    # A 9x9 window is a large kernel for compute_mpwi.cpp, the form that calls copy_dest_values.
    n, h, w, c = 1, 64, 64, 64
    g = torch.Generator().manual_seed(11)
    x = torch.randn((1, 1, n * h * w, c), generator=g).to(torch.bfloat16)
    out, idx = ttnn.max_pool2d(
        input_tensor=tt(x, d, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        batch_size=n, input_h=h, input_w=w, channels=c, kernel_size=[9, 9], stride=[4, 4], padding=[0, 0],
        dilation=[1, 1], deallocate_input=False, reallocate_halo_output=True, return_indices=True,
    )
    return [out, idx]


# situ_glu_tile, clamped_silu_glu_tile and their pack forms: the test kernels of test_situ_glu_sfpu.py and
# test_clamped_silu_glu_sfpu.py (64 tiles)
def _glu_cases():
    try:
        import test_situ_glu_sfpu as S
        import test_clamped_silu_glu_sfpu as C
    except Exception:  # noqa: BLE001
        return
    have_pack = os.path.exists(os.path.join(os.environ.get("TT_METAL_HOME", "."), "tests/tt_metal/tt_metal/test_kernels/compute/situ_glu_pack.cpp"))
    for in_name in S.IN_DTYPES:
        for fp32 in (False, True):
            for dst_out in (0, 2):
                kernels = ["situ_glu.cpp"] + (["situ_glu_pack.cpp"] if have_pack else [])
                for k in kernels:
                    def f(d, in_name=in_name, fp32=fp32, dst_out=dst_out, k=k):
                        dt, pb = S.IN_DTYPES[in_name]
                        g, u = S._coverage_inputs(64)
                        if k == "situ_glu.cpp":
                            return [S._run(d, g, u, dt, pb, fp32, dst_out)]
                        return [S._run(d, g, u, dt, pb, fp32, dst_out, k)]

                    CASES[f"situ_glu_{in_name}_{int(fp32)}_{dst_out}_{k.split('.')[0]}"] = f
    for in_name in C.IN_DTYPES:
        for fp32 in (False, True):
            for dg, du, do in ((0, 1, 0), (0, 1, 2), (1, 3, 1)):
                kernels = [None] + (["clamped_silu_glu_pack.cpp"] if have_pack else [])
                for k in kernels:
                    def f(d, in_name=in_name, fp32=fp32, dg=dg, du=du, do=do, k=k):
                        dt, pb = C.IN_DTYPES[in_name]
                        g, u = C._coverage_inputs(64)
                        args = (d, g, u, dt, pb, fp32, dg, du, do)
                        return [C._run(*args) if k is None else C._run(*args, k)]

                    CASES[f"clamped_silu_glu_{in_name}_{int(fp32)}_{dg}{du}{do}_{'pack' if k else 'math'}"] = f


_glu_cases()

# Probe cases (onecall_probe.cpp through ttnn.generic_op)
PROBE = os.path.join(HERE, "onecall_probe.cpp")
TILE = 32 * 32


def _probe(d, op, a, b, dt, page, fp32, ntiles):
    shape = [1, ntiles, 32, 32]
    ta = tt(a.reshape(shape), d, dt)
    tb = tt(b.reshape(shape), d, dt)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, d, ttnn.DRAM_MEMORY_CONFIG)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(idx):
        return ttnn.CBDescriptor(
            total_size=2 * page,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
        )

    rrt = ttnn.RuntimeArgs()
    rrt[0][0] = [ta.buffer_address(), tb.buffer_address(), ntiles, 0]
    wrt = ttnn.RuntimeArgs()
    wrt[0][0] = [out.buffer_address(), ntiles, 0]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source="tests/tt_metal/tt_metal/test_kernels/dataflow/reader_situ_glu.cpp",
            core_ranges=core,
            compile_time_args=ttnn.TensorAccessorArgs(ta).get_compile_time_args() + ttnn.TensorAccessorArgs(tb).get_compile_time_args(),
            runtime_args=rrt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=[16] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=wrt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=PROBE,
            core_ranges=core,
            compile_time_args=[ntiles, op],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32),
        ),
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb(0), cb(1), cb(16)])
    return [ttnn.generic_op([ta, tb, out], prog)]


NT = 16


def _bf(seed):
    return mixed((NT * TILE,), seed).to(torch.bfloat16)


def _mask01(seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(NT * TILE, generator=g) < 0.5).float()


def _i32(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-(2**31), 2**31 - 1, (NT * TILE,), generator=g, dtype=torch.int64).to(torch.int32)


case("probe_identity_bf16")(lambda d: _probe(d, 0, _bf(20), _bf(21), ttnn.bfloat16, TILE * 2, False, NT))
case("probe_identity_bf16_fp32dest")(lambda d: _probe(d, 0, _bf(20), _bf(21), ttnn.bfloat16, TILE * 2, True, NT))
case("probe_identity_fp32")(lambda d: _probe(d, 0, mixed((NT * TILE,), 22), mixed((NT * TILE,), 23), ttnn.float32, TILE * 4, True, NT))
case("probe_identity_uint32")(lambda d: _probe(d, 1, _i32(24) & 0x7FFFFFFF, _i32(25) & 0x7FFFFFFF, ttnn.uint32, TILE * 4, True, NT))
case("probe_mask_bf16")(lambda d: _probe(d, 2, _bf(26), _mask01(27).to(torch.bfloat16), ttnn.bfloat16, TILE * 2, False, NT))
case("probe_mask_bf16_fp32dest")(lambda d: _probe(d, 2, _bf(26), _mask01(27).to(torch.bfloat16), ttnn.bfloat16, TILE * 2, True, NT))
case("probe_mask_int32")(lambda d: _probe(d, 3, _i32(28), _mask01(29).to(torch.int32), ttnn.int32, TILE * 4, True, NT))
case("probe_mask_posinf_bf16")(lambda d: _probe(d, 4, _bf(30), _mask01(31).to(torch.bfloat16), ttnn.bfloat16, TILE * 2, False, NT))
case("probe_copy_dest_bf16")(lambda d: _probe(d, 5, _bf(32), _bf(33), ttnn.bfloat16, TILE * 2, False, NT))
case("probe_copy_dest_fp32")(lambda d: _probe(d, 6, mixed((NT * TILE,), 34), mixed((NT * TILE,), 35), ttnn.float32, TILE * 4, True, NT))
case("probe_copy_dest_int32")(lambda d: _probe(d, 7, _i32(36), _i32(37), ttnn.int32, TILE * 4, True, NT))
case("probe_copy_dest_deprecated_bf16")(lambda d: _probe(d, 8, _bf(38), _bf(39), ttnn.bfloat16, TILE * 2, False, NT))


# Exhaustive bfloat16 inputs (rule 3 of 07_ISSUE_RESOLUTION.md): all 65536 patterns, NaNs, infinities, zeros and
# denormals included, in one 256x256 tensor (64 tiles); a second operand takes the same patterns in a fixed shuffle.
def all_bf16(shape=(1, 1, 256, 256)):
    w = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.int16)
    return torch.from_numpy(w.copy()).view(torch.bfloat16).reshape(shape)


def all_bf16_shuffled(seed, shape=(1, 1, 256, 256)):
    perm = torch.randperm(65536, generator=torch.Generator().manual_seed(seed))
    return all_bf16().reshape(-1)[perm].reshape(shape)


A16 = (1, 1, 256, 256)
case("softsign_bf16_all")(lambda d: [ttnn.softsign(tt(all_bf16(), d))])
case("softshrink_bf16_all")(lambda d: [ttnn.softshrink(tt(all_bf16(), d), lambd=0.5)])
case("logical_not_bf16_all")(lambda d: [ttnn.logical_not(tt(all_bf16(), d))])
case("dropout_bf16_all")(lambda d: [ttnn.experimental.dropout(tt(all_bf16(), d), probability=0.5, scale=2.0, seed=4321)])
case("gelu_bw_tanh_bf16_all")(
    lambda d: list(ttnn.gelu_bw(tt(all_bf16_shuffled(40), d), tt(all_bf16(), d), variant=ttnn.GeluVariant.Tanh))
)
NA = 64
case("probe_identity_bf16_all")(lambda d: _probe(d, 0, all_bf16().reshape(-1), all_bf16_shuffled(41).reshape(-1), ttnn.bfloat16, TILE * 2, False, NA))
case("probe_mask_bf16_all")(
    lambda d: _probe(d, 2, all_bf16().reshape(-1), (torch.rand(65536, generator=torch.Generator().manual_seed(42)) < 0.5).to(torch.bfloat16), ttnn.bfloat16, TILE * 2, False, NA)
)
case("probe_mask_posinf_bf16_all")(
    lambda d: _probe(d, 4, all_bf16().reshape(-1), (torch.rand(65536, generator=torch.Generator().manual_seed(43)) < 0.5).to(torch.bfloat16), ttnn.bfloat16, TILE * 2, False, NA)
)
case("probe_copy_dest_bf16_all")(lambda d: _probe(d, 5, all_bf16_shuffled(44).reshape(-1), all_bf16().reshape(-1), ttnn.bfloat16, TILE * 2, False, NA))
case("probe_copy_dest_deprecated_bf16_all")(lambda d: _probe(d, 8, all_bf16_shuffled(45).reshape(-1), all_bf16().reshape(-1), ttnn.bfloat16, TILE * 2, False, NA))


def _glu_all():
    try:
        import test_clamped_silu_glu_sfpu as C
        import test_situ_glu_sfpu as S
    except Exception:  # noqa: BLE001
        return
    g = all_bf16().reshape(-1)
    u = all_bf16_shuffled(46).reshape(-1)
    for fp32 in (False, True):
        def fs(d, fp32=fp32):
            dt, pb = S.IN_DTYPES["bf16"]
            return [S._run(d, g, u, dt, pb, fp32, 0)]

        def fc(d, fp32=fp32):
            dt, pb = C.IN_DTYPES["bf16"]
            return [C._run(d, g, u, dt, pb, fp32, 0, 1, 0)]

        CASES[f"situ_glu_bf16_all_{int(fp32)}"] = fs
        CASES[f"clamped_silu_glu_bf16_all_{int(fp32)}"] = fc


_glu_all()


# Single-core twins of the PRNG callers ttnn does not reach (07 rule 1): the gtest kernels that tt-train's AdamW
# (init_prng_seed, then stochastic_round_tile per tile) and the dropout gtest run, through ttnn.generic_op.
def _twin(d, kernel, x, in_dt, in_page, out_dt, out_page, ct_args, fp32, defines=()):
    ntiles = x.numel() // TILE
    shape = [1, ntiles, 32, 32]
    tx = tt(x.reshape(shape), d, in_dt)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dt, ttnn.TILE_LAYOUT, d, ttnn.DRAM_MEMORY_CONFIG)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(idx, dt, page):
        return ttnn.CBDescriptor(
            total_size=2 * page,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
        )

    rrt = ttnn.RuntimeArgs()
    rrt[0][0] = [tx.buffer_address(), ntiles, 0]
    wrt = ttnn.RuntimeArgs()
    wrt[0][0] = [out.buffer_address(), ntiles, 0]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=ttnn.TensorAccessorArgs(tx).get_compile_time_args(),
            runtime_args=rrt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=[16] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=wrt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kernel,
            core_ranges=core,
            compile_time_args=ct_args,
            defines=list(defines),
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32),
        ),
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb(0, in_dt, in_page), cb(16, out_dt, out_page)])
    return [ttnn.generic_op([tx, out], prog)]


def _sr_input():
    g = torch.Generator().manual_seed(47)
    return torch.randn(NA * TILE, generator=g) * torch.pow(2.0, torch.randint(-20, 20, (NA * TILE,), generator=g).float())


case("twin_stochastic_round")(
    lambda d: _twin(d, "tests/tt_metal/tt_metal/test_kernels/compute/stochastic_rounding_sfpu.cpp", _sr_input(), ttnn.float32, TILE * 4, ttnn.bfloat16, TILE * 2, [NA, 0x1234ABCD], True)
)
case("twin_stochastic_round_math_only")(
    lambda d: _twin(d, os.path.join(HERE, "stochastic_rounding_sfpu_math_only.cpp"), _sr_input(), ttnn.float32, TILE * 4, ttnn.bfloat16, TILE * 2, [NA, 0x1234ABCD], True)
)
case("twin_dropout_sfpu_bf16_all")(
    lambda d: _twin(
        d, "tests/tt_metal/tt_metal/test_kernels/compute/dropout_sfpu.cpp", all_bf16().reshape(-1), ttnn.bfloat16, TILE * 2, ttnn.bfloat16, TILE * 2,
        [NA, 1, 0x5EED, 0x3FFFFFFF, 0x40000000], False, (("SFPU_OP_DROPOUT_INCLUDE", "1"),),
    )
)


@pytest.mark.parametrize("name", list(CASES))
def test_bits(dev, name):
    outs = CASES[name](dev)
    if MOCK:
        return
    os.makedirs(OUT, exist_ok=True)
    arrs = {f"o{i}": words(ttnn.to_torch(o) if not isinstance(o, torch.Tensor) else o) for i, o in enumerate(outs)}
    np.savez(os.path.join(OUT, f"{name}.npz"), **arrs)
