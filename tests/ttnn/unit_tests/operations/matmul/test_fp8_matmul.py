# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Native FP8_E4M3 x FP8_E4M3 matmul through ttnn.matmul on Blackhole.

This test stacks the three pieces this experiment branch carries:
  1. PR #43775   - FP8_E4M3 as a real ttnn DataType.
  2. PR #44307   - tilize accepts FP8_E4M3 input (extended in this branch:
                   also allow FP8_E4M3 -> FP8_E4M3 TILE output).
  3. PR #43481   - FP8 LLK matmul plus the tt_metal jit_build wiring that
                   teaches kernel build about FP8 CBs.

Plus one branch-local relaxation that ties them together:
  - tt_metal/impl/tensor/spec/tensor_spec.cpp:
      DataType::FP8_E4M3 is allowed in TILE layout in addition to ROW_MAJOR.

Per-operand pipeline:
    torch.float32 (host, RM)
      -> ttnn.from_torch(dtype=fp8_e4m3, layout=ROW_MAJOR)   # FP8 RM on device
      -> ttnn.tilize(dtype=fp8_e4m3)                          # FP8 TILE
    -> ttnn.matmul(...)                                       # FP8 x FP8 -> {bf16, bfp8_b, fp8} TILE
    -> ttnn.to_torch()

Reference: torch matmul on the *FP8-quantized* inputs (input tensors are quantized
to torch.float8_e4m3fn first, then dequantized to float32 for the reference matmul).
This keeps the reference honest about FP8 input quantization while letting the
accumulator/output quantization show up as the PCC delta we're trying to measure.
"""

import time

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


def _to_fp8_tile(torch_input_f32, device):
    """torch.float32 (host, RM) -> FP8_E4M3 TILE on device."""
    tt_rm = ttnn.from_torch(
        torch_input_f32,
        device=device,
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.tilize(tt_rm, dtype=ttnn.fp8_e4m3)


def _fp8_quantize(x_f32):
    """Quantize a float32 torch tensor to FP8 (e4m3) and back to float32. Used to build the reference."""
    return x_f32.to(torch.float8_e4m3fn).to(torch.float32)


@pytest.mark.parametrize(
    "M, K, N",
    [
        (32, 32, 32),  # single-tile sanity; matches the smallest LLK gtest scale
        (64, 64, 64),  # 2x2 tile grid
        (128, 128, 128),  # 4x4 tile grid
        (32, 256, 32),  # K-major: exercises multi-block K accumulation
        (256, 64, 128),  # asymmetric M/N
    ],
    ids=[
        "1tile",
        "2x2tiles",
        "4x4tiles",
        "longK",
        "asymmetric",
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,  # easiest path; readback is plain bf16
        ttnn.fp8_e4m3,  # native FP8 output (matches rtawfik's gtest config)
    ],
    ids=["out_bf16", "out_fp8"],
)
def test_fp8_matmul(device, M, K, N, output_dtype):
    if not is_blackhole():
        pytest.skip("FP8_E4M3 requires Blackhole hardware")

    torch.manual_seed(0)
    # Uniform in [-1, 1) stays well within fp8 e4m3 range (~448 max abs) and gives the
    # accumulator more dynamic range to exercise than pure [0, 1).
    a_f32 = torch.empty(M, K, dtype=torch.float32).uniform_(-1.0, 1.0)
    b_f32 = torch.empty(K, N, dtype=torch.float32).uniform_(-1.0, 1.0)

    tt_a = _to_fp8_tile(a_f32, device)
    tt_b = _to_fp8_tile(b_f32, device)
    assert tt_a.dtype == ttnn.fp8_e4m3 and tt_a.layout == ttnn.TILE_LAYOUT
    assert tt_b.dtype == ttnn.fp8_e4m3 and tt_b.layout == ttnn.TILE_LAYOUT

    # Match the rtawfik LLK gtest's compute config: FP32 accumulator in Dest, no PACKER_L1_ACC.
    # Once basic correctness is established, sweep these to see the perf/accuracy tradeoff.
    compute_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    tt_c = ttnn.matmul(
        tt_a,
        tt_b,
        dtype=output_dtype,
        compute_kernel_config=compute_cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert tt_c.dtype == output_dtype
    assert tt_c.layout == ttnn.TILE_LAYOUT

    # Reference: matmul of the FP8-quantized inputs, computed in float32 on host.
    a_ref = _fp8_quantize(a_f32)
    b_ref = _fp8_quantize(b_f32)
    torch_ref = a_ref @ b_ref

    torch_out = ttnn.to_torch(tt_c).to(torch.float32)

    # PCC threshold accounts for FP8 accumulator -> packer quantization. The exact achievable
    # number depends on K (more K = more accumulator noise). Start conservative; tighten once
    # we see the actual numbers.
    expected_pcc = 0.97 if output_dtype == ttnn.fp8_e4m3 else 0.985

    # Print a small window of values when we're on the smallest case so a failure is human-readable.
    # This is the fastest way to triage "wrong by a constant" vs "wrong by a stride" vs "wrong everywhere".
    if M == 32 and N == 32:
        print(f"\n[fp8_matmul debug] M={M} K={K} N={N} output_dtype={output_dtype}")
        print(f"  torch_ref[0, :8] = {torch_ref[0, :8].tolist()}")
        print(f"  torch_out[0, :8] = {torch_out[0, :8].tolist()}")
        print(f"  diff[0, :8]      = {(torch_ref[0, :8] - torch_out[0, :8]).tolist()}")
        print(
            f"  ref mean/std={torch_ref.mean().item():.4f}/{torch_ref.std().item():.4f} "
            f"out mean/std={torch_out.mean().item():.4f}/{torch_out.std().item():.4f}"
        )

    assert_with_pcc(torch_ref, torch_out, expected_pcc)


# ---------------------------------------------------------------------------
# Diagnostic tests for controlled FP8 matmul stimuli.
#
# Use these to isolate where the 0.48 PCC on `test_fp8_matmul[out_bf16-1tile]`
# comes from. The 0.48 figure is structural-noise-like (not quantization-like),
# and these probes pin down which layer of the stack is responsible.
#
# Reading guide:
#   - ones / identity work but random fails   -> stimulus / reference issue
#   - identity works, ones fails              -> accumulator config
#   - identity fails (output != input)        -> FP8 TILE data layout mismatch
#                                                between tilize output and matmul input
#   - bf16 control fails                      -> ttnn.matmul plumbing is the problem,
#                                                not FP8-specific
# ---------------------------------------------------------------------------


def _default_compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def test_fp8_matmul_ones(device):
    """A and B are all 1.0. Output should be K (= 32) at every position.

    1.0 is exactly representable in fp8 e4m3, so there is no input quantization.
    32 is exactly representable too. If we see anything other than ~32 everywhere,
    matmul math/accumulation is wrong; if we see e.g. ~16 we are losing one face of K.
    """
    if not is_blackhole():
        pytest.skip("FP8_E4M3 requires Blackhole hardware")

    M, K, N = 32, 32, 32
    a_f32 = torch.ones(M, K, dtype=torch.float32)
    b_f32 = torch.ones(K, N, dtype=torch.float32)

    tt_a = _to_fp8_tile(a_f32, device)
    tt_b = _to_fp8_tile(b_f32, device)
    tt_c = ttnn.matmul(
        tt_a,
        tt_b,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_default_compute_cfg(device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(tt_c).to(torch.float32)

    print(
        f"\n[ones] expected all {K}.0; got "
        f"min={out.min().item()} max={out.max().item()} mean={out.mean().item()} "
        f"unique={torch.unique(out).tolist()[:8]}"
    )
    assert torch.allclose(
        out, torch.full_like(out, float(K)), atol=1e-3
    ), f"ones-matmul: expected uniform {K}, got min={out.min()} max={out.max()}"


def test_fp8_matmul_identity_passthrough(device):
    """A = random fp8 (quantized on host), B = identity. Output should equal A.

    With B = I, matmul reads A through the inner-product axis with a single
    nonzero contribution per (m, n). If the FP8 tile layout for A is correct
    on the matmul-input side, the output equals A (within fp8 packer quant
    on the output side). Mismatches here mean the tilize FP8 RM -> FP8 TILE
    path is producing bytes that the matmul reads with the wrong addressing.
    """
    if not is_blackhole():
        pytest.skip("FP8_E4M3 requires Blackhole hardware")

    M, K, N = 32, 32, 32
    torch.manual_seed(0)
    a_f32 = torch.empty(M, K, dtype=torch.float32).uniform_(-1.0, 1.0)
    a_quant = _fp8_quantize(a_f32)  # what we expect to see come back out
    b_f32 = torch.eye(K, N, dtype=torch.float32)

    tt_a = _to_fp8_tile(a_f32, device)
    tt_b = _to_fp8_tile(b_f32, device)
    tt_c = ttnn.matmul(
        tt_a,
        tt_b,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_default_compute_cfg(device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(tt_c).to(torch.float32)

    print(
        f"\n[identity] expected ~A; |A-out| stats: "
        f"mean={(a_quant - out).abs().mean().item():.4f} "
        f"max={(a_quant - out).abs().max().item():.4f}"
    )
    print(f"  a_quant[0, :8] = {a_quant[0, :8].tolist()}")
    print(f"  out    [0, :8] = {out[0, :8].tolist()}")
    assert_with_pcc(a_quant, out, 0.995)


def test_bf16_matmul_control(device):
    """Control: same shape, all bf16, no FP8 anywhere. Validates the test harness.

    If this fails we have a ttnn.matmul wiring issue independent of FP8, and
    the FP8 PCC noise is downstream of that. If this passes we know the
    harness is correct and we're chasing an FP8-specific bug.
    """
    if not is_blackhole():
        pytest.skip("Matches the FP8 test environment")

    M, K, N = 32, 32, 32
    torch.manual_seed(0)
    a_f32 = torch.empty(M, K, dtype=torch.float32).uniform_(-1.0, 1.0)
    b_f32 = torch.empty(K, N, dtype=torch.float32).uniform_(-1.0, 1.0)

    tt_a = ttnn.from_torch(a_f32, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_b = ttnn.from_torch(b_f32, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_c = ttnn.matmul(
        tt_a,
        tt_b,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_default_compute_cfg(device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(tt_c).to(torch.float32)
    ref = a_f32.to(torch.bfloat16).to(torch.float32) @ b_f32.to(torch.bfloat16).to(torch.float32)
    assert_with_pcc(ref, out, 0.99)


# ---------------------------------------------------------------------------
# Perf benchmark.
#
# The all-ones diagnostic proved the kernel that runs through this path is the
# real FP8 LLK matmul:
#   - tt_a / tt_b have DataType::FP8_E4M3 -> CBs come up as DataFormat::Fp8_e4m3,
#     unpacker decodes FP8, math runs FP8 -> FP32 dest, packer emits bf16/fp8.
#   - ttnn/cpp/ttnn/operations/matmul/ has zero FP8-specific branches, so there
#     is no fallback path. The same bmm.cpp kernel that runs for bf16 runs here,
#     just with different unpacker config.
# Tilize FP8 RM -> FP8 TILE writes some bytes into the wrong tile positions,
# so the numerical output is garbage, but kernel dispatch and timing are
# unaffected. For perf measurement that's fine.
#
# Note on hackier alternatives (host-built FP8 tile + dtype reinterpret, etc.):
# they don't change the perf number because they would feed the *same* kernel
# the *same-sized* CBs configured the *same* way. Skip them.
# ---------------------------------------------------------------------------


def _to_tile_input(torch_input_f32, device, in_dtype):
    """torch.float32 (host, RM) -> TILE on device with the requested input dtype.

    bf16 takes the standard host-side TILE packing path inside ttnn.from_torch.
    FP8 has no host-side TILE packing instantiation (see encode_tensor_data in
    tt_metal/impl/tensor/tensor_impl.cpp - explicit instantiations don't include
    float8_e4m3), so it has to go RM-then-tilize. Both end up as the same matmul
    input from the kernel's POV: a TILE-layout tensor whose DataType drives the
    CB DataFormat in the matmul program factory.
    """
    if in_dtype == ttnn.fp8_e4m3:
        tt_rm = ttnn.from_torch(
            torch_input_f32,
            device=device,
            dtype=ttnn.fp8_e4m3,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.tilize(tt_rm, dtype=ttnn.fp8_e4m3)
    return ttnn.from_torch(
        torch_input_f32,
        device=device,
        dtype=in_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# Shape selection rationale (5 total, covering small -> large, square -> narrow,
# and one model-specific projection from DeepSeek-V3):
#   - small_square      : tiny shape so dispatch overhead dominates; shows the
#                          per-call cost floor that FP8 can't help with.
#   - large_square      : canonical compute-bound shape; closest to peak TFLOP/s.
#                          FP8 vs bf16 here is roughly a math-throughput comparison.
#   - decode_narrow_M   : M=32 (single decode token, tile-aligned), K and N large.
#                          Dominated by weight read bandwidth; this is where the
#                          FP8 win (1 B/element vs 2 B/element on the weight read)
#                          shows up most cleanly.
#   - ds_v3_wo          : DeepSeek-V3 output projection at seq_len=2048. K=16384
#                          is the extreme-K case in the codebase (see
#                          test_matmul_deepseek.py "wo"), good for stressing the
#                          K-loop / accumulator path.
#   - ds_v3_qkv_a       : DeepSeek-V3 QKV-down projection shape (K=896, N=2112)
#                          at seq_len=2048 (jvega's tilize PR uses M up to 2048
#                          for the same hidden=7168 family). Production-ish
#                          prefill projection with moderate K, narrow-ish N.
@pytest.mark.parametrize(
    "M, K, N",
    [
        (512, 512, 512),  # small_square
        (4096, 4096, 4096),  # large_square
        (32, 8192, 8192),  # decode_narrow_M
        (2048, 16384, 896),  # ds_v3_wo (long K, narrow N)
        (2048, 896, 2112),  # ds_v3_qkv_a-like (prefill projection)
        # (2, 2, 2),  # 2x2 tile grid
        # (32, 32, 32),
    ],
    # ids=[
    #     "small_square",
    #     "large_square",
    #     "decode_narrow_M",
    #     "ds_v3_wo",
    #     "ds_v3_qkv_a",
    # ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        # bf16 baseline: feeds the same bmm.cpp kernel with DataFormat::Float16_b
        # CBs (and FP32 dest because we set fp32_dest_acc_en=True for parity with
        # the FP8 run). This is the reference number FP8 is being compared against.
        # (ttnn.bfloat16, ttnn.bfloat16),
        # FP8 -> bf16: FP8 unpacker, FP32 dest, bf16 packer. Isolates the input-
        # side bandwidth & math win; output writeback is the same size as bf16.
        (ttnn.fp8_e4m3, ttnn.bfloat16),
        # FP8 -> FP8: full FP8 path including 1 B/element output writeback.
        # (ttnn.fp8_e4m3, ttnn.fp8_e4m3),
    ],
    # ids=["bf16_to_bf16", "fp8_to_bf16", "fp8_to_fp8"],
)
def test_fp8_matmul_perf(device, M, K, N, in_dtype, out_dtype):
    """Time matmul on Blackhole, FP8 vs bf16 inputs at matched shapes.

    Output values are correct for bf16 and garbage for FP8 (the tilize FP8 RM ->
    FP8 TILE packer puts bytes in the wrong tile slots); kernel dispatch and
    timing are unaffected by that. See the diagnostic tests above for the
    layout proof.

    Reports per-iter latency and a sustained TFLOP/s number (Python dispatch +
    DRAM I/O + compute included). To isolate compute-only cycles run with
    TT_METAL_DEVICE_PROFILER=1 and read the perf CSV.
    """
    if not is_blackhole():
        pytest.skip("FP8_E4M3 requires Blackhole hardware")

    warmup_iters = 1
    measure_iters = 2

    torch.manual_seed(0)
    # a_f32 = torch.eye(M, K, dtype=torch.float32)
    # b_f32 = torch.eye(K, N, dtype=torch.float32)

    a_i8 = torch.ones(M, K, dtype=torch.uint8) * 56
    b_i8 = torch.randint(0, 256, (K, N), dtype=torch.uint8)

    # print(b_i8)

    tt_a = _to_tile_input(a_i8, device, ttnn.uint8)
    tt_b = _to_tile_input(b_i8, device, ttnn.uint8)
    # assert tt_a.dtype == in_dtype and tt_a.layout == ttnn.TILE_LAYOUT
    # assert tt_b.dtype == in_dtype and tt_b.layout == ttnn.TILE_LAYOUT

    compute_cfg = _default_compute_cfg(device)

    for _ in range(warmup_iters):
        out = ttnn.matmul(
            tt_a,
            tt_b,
            dtype=out_dtype,
            compute_kernel_config=compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Grab the output, convert to torch, print full tensor
        # out_torch = ttnn.to_torch(out)
        # torch.set_printoptions(profile='full')
        # print("\nOutput tensor:")
        # print(out_torch)
        # torch.set_printoptions(profile='default')

        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(measure_iters):
        out = ttnn.matmul(
            tt_a,
            tt_b,
            dtype=out_dtype,
            compute_kernel_config=compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()

    total_s = t1 - t0
    per_iter_ms = total_s / measure_iters * 1e3
    flops_per_iter = 2.0 * M * K * N
    tflops = (flops_per_iter * measure_iters) / total_s / 1e12

    print(
        f"\n[matmul_perf] M={M:>5d} K={K:>5d} N={N:>5d} "
        f"in={str(in_dtype):>20s} out={str(out_dtype):>20s} "
        f"iters={measure_iters}  per_iter={per_iter_ms:8.3f} ms  "
        f"sustained={tflops:7.2f} TFLOP/s"
    )
