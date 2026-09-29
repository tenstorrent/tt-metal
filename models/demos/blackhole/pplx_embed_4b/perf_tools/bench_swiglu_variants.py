# Cheaper SiLU for the fused-SwiGLU minimal_matmul: variants of the pack-thread SFPU pass (swiglu_sfpu.hpp's
# calculate_minimal_matmul_swiglu loop body, patched in place and restored afterwards; each variant in its own process
# with a fresh JIT cache) timed at the model's blocks, and their error against the ideal SwiGLU of the device's own
# pre-activations: the same operands through the plain packed matmul with a bf16 output (what DST holds), then torch
# fp32 silu(gate) * up. The fused output is bfp8, so no variant gets below its ~1.2% relative error. 'landed' is the
# header as committed (Schraudolph exp, bare SFPARECIP, NEGATIVE_RESULTS 62); 'silu_tile + mul' the pass before it.
# Usage: bench_swiglu_variants.py <batch> [xscale ...]   blocks default to the model's (bs8 8,40,6 1x6, else 4,40,8 1x8);
#        MM_BLOCKS=M,K,N,sh,sw overrides; SW_ONLY="landed|silu_tile + mul" picks variants; xscale scales the activation (gate std
#        ~xscale with the 0.02-scaled weight) to probe the tails; SW_REPEAT=3 runs the SFPU pass 3 times (timing only) so
#        a variant's cost shows although one pass is mostly hidden; SW_SKIP=first|last|all skips the pass on the first /
#        last subblock of every output block, or on all of them (timing only; errors meaningless).
import os
import statistics
import subprocess
import sys
import tempfile
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
KDIR = os.path.join(REPO, "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels")
HDR = os.path.join(KDIR, "swiglu_sfpu.hpp")
COMPUTE = os.path.join(KDIR, "compute_metal2.cpp")
SFPU_CALL = "PACK((llk_minimal_matmul_swiglu(gate, gate + 1, gate)));"
SKIP = {  # SW_SKIP: which subblocks of each output block run without the pass
    "first": "!(M_start == 0 && N_start == 0)",
    "last": "!(M_start + subblock_h >= M_block_tiles && N_start + subblock_w >= N_block_tiles)",
    "all": "false",
}
# a variant's body replaces everything after the up load up to the DST store
BODY_START = "        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];\n"
BODY_END = "        sfpi::dst_reg[out_tile_idx * dst_tile_size] = "
LOOP = "    for (int d = 0; d < ITERATIONS; d++) {\n"
RND = "        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);\n"
# exp(-gate) pieces (the sfpi of _sfpu_exp_21f_bf16_): xlog2 = -gate / ln2 + 127, clamped to [0, 255]; z is 2**(xlog2 -
# 127) with a linear mantissa (Schraudolph, <6% error); the 21f path refines the mantissa with a degree-2 polynomial.
XLOG2 = "        sfpi::vFloat xlog2 = sfpi::clamp(gate * -1.4426950216293334961f + 127.f, 0.0f, 255.0f);\n"
Z = "        sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));\n"
# Schraudolph with the bias shifted to centre the linear mantissa's error (max ~3%): 2**f ~ 1 + f - 0.0430
XLOG2_C = "        sfpi::vFloat xlog2 = sfpi::clamp(gate * -1.4426950216293334961f + 126.9570f, 0.0f, 255.0f);\n"
PROD = "        sfpi::vFloat result = up * (gate * sig);\n"
STORE = "        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;\n"
# name: loop body computing `result` from `gate` / `up` (the landed loop is unrolled 8x, so every variant is)
VARIANTS = {
    "landed": None,
    "silu_tile + mul": (
        "        sfpi::vFloat silu = sfpi::convert<sfpi::vFloat16b>(\n"
        "            gate * _sfpu_sigmoid_<false>(gate), sfpi::RoundMode::Nearest);\n"
        "        sfpi::vFloat result = silu * up;\n" + RND
    ),
    "exp21f, recip 1 NR, no rnd": (
        "        sfpi::vFloat sig = sfpu_reciprocal_iter<1>(1.0f + _sfpu_exp_21f_bf16_<true>(-gate));\n" + PROD
    ),
    "exp21f, recip no NR": (
        "        sfpi::vFloat sig = sfpi::approx_recip(1.0f + _sfpu_exp_21f_bf16_<true>(-gate));\n" + PROD
    ),
    "schraudolph, recip 1 NR": (XLOG2 + Z + "        sfpi::vFloat sig = sfpu_reciprocal_iter<1>(1.0f + z);\n" + PROD),
    "schraudolph, recip no NR": (XLOG2 + Z + "        sfpi::vFloat sig = sfpi::approx_recip(1.0f + z);\n" + PROD),
    "schraudolph c, recip no NR": (XLOG2_C + Z + "        sfpi::vFloat sig = sfpi::approx_recip(1.0f + z);\n" + PROD),
}
VARIANTS = {k: v for k, v in VARIANTS.items() if not os.getenv("SW_ONLY") or k in os.getenv("SW_ONLY").split("|")}


def patch(src, body):
    assert src.count(BODY_START) == 1 and src.count(BODY_END) == 1
    if body is None:
        return src
    a, b = src.index(BODY_START) + len(BODY_START), src.index(BODY_END)
    b = src.index("\n", b) + 1
    return src[:a] + body + STORE + src[b:]


def child(B, xscale, out_path):
    import torch

    import ttnn

    M, K, N = B * 512, 2560, 19456
    blocks = tuple(int(v) for v in os.environ["MM_BLOCKS"].split(","))
    torch.manual_seed(0)
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
    try:
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=blocks[0], K_block_size=blocks[1], N_block_size=blocks[2], subblock_h=blocks[3],
            subblock_w=blocks[4], compute_with_storage_grid_size=ttnn.CoreCoord(12, 10),
        )  # fmt: skip
        ckc = ttnn.init_device_compute_kernel_config(
            D.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.02, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=D)
        x = ttnn.from_torch(
            torch.randn(1, 1, M, K) * xscale, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )  # fmt: skip
        kw = dict(compute_kernel_config=ckc, config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pre = ttnn.to_torch(ttnn.experimental.minimal_matmul(x, w, dtype=ttnn.bfloat16, **kw))[0, 0, :64].float()
        fn = lambda: ttnn.experimental.minimal_matmul(x, w, dtype=ttnn.bfloat8_b, fuse_swiglu=True, **kw)
        out = ttnn.to_torch(fn())[0, 0, :64].float()
        torch.save((pre.clone(), out.clone()), out_path)
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(D)
        n = 4
        tid = ttnn.begin_trace_capture(D, cq_id=0)
        outs = [fn() for _ in range(n)]
        ttnn.end_trace_capture(D, tid, cq_id=0)
        ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) / n * 1e6)
        ttnn.release_trace(D, tid)
        print(f"CHILD {statistics.median(ts):.1f}", flush=True)
    finally:
        ttnn.close_device(D)


def main():
    import torch

    B = int(sys.argv[1])
    scales = [float(s) for s in sys.argv[2:]] or [1.0]
    blocks = os.getenv("MM_BLOCKS") or ("8,40,6,1,6" if B == 8 else "4,40,8,1,8")
    print(
        f"RES swiglu variants B{B} blocks={blocks} xscale={scales} repeat={os.getenv('SW_REPEAT', '1')} skip={os.getenv('SW_SKIP')}",
        flush=True,
    )
    orig, orig_c = open(HDR).read(), open(COMPUTE).read()
    rep, skip = int(os.getenv("SW_REPEAT", "1")), os.getenv("SW_SKIP")
    if rep != 1 or skip:  # timing only: repeated or skipped passes leave wrong values, so the errors are meaningless
        assert orig_c.count(SFPU_CALL) == 1
        call = " ".join([SFPU_CALL] * rep)
        open(COMPUTE, "w").write(orig_c.replace(SFPU_CALL, f"if ({SKIP[skip]}) {{ {call} }}" if skip else call))
    try:
        for vname, body in VARIANTS.items():
            open(HDR, "w").write(patch(orig, body))
            for xs in scales:
                d = tempfile.mkdtemp(prefix="swiglu_var_")
                env = dict(os.environ, TT_METAL_CACHE=os.path.join(d, "cache"), MM_BLOCKS=blocks)
                out_path = os.path.join(d, "out.pt")
                p = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--child", str(B), str(xs), out_path],
                    env=env, capture_output=True, text=True, timeout=1200,
                )  # fmt: skip
                res = [l for l in p.stdout.splitlines() if l.startswith("CHILD")]
                if not res:
                    err = [l for l in (p.stdout + p.stderr).splitlines() if "TT_THROW" in l or "rror" in l][:4]
                    print(f"RES {vname:28s} x{xs:<4g} FAILED {err}", flush=True)
                    continue
                pre, out = torch.load(out_path)
                y = pre.reshape(64, -1, 2, 32)  # [rows, output tile, gate / up, 32 columns]
                ideal = (torch.nn.functional.silu(y[:, :, 0]) * y[:, :, 1]).reshape(64, -1)
                e = out - ideal
                rel = (e.norm() / ideal.norm()).item()
                pcc = torch.corrcoef(torch.stack([out.flatten(), ideal.flatten()]))[0, 1].item()
                print(
                    f"RES {vname:28s} x{xs:<4g} {float(res[0].split()[1]):8.1f} us  rel err {rel:.5f}  "
                    f"max abs err {e.abs().max().item():.4f} (|ideal| max {ideal.abs().max().item():.2f})  "
                    f"pcc {pcc:.6f}",
                    flush=True,
                )
    finally:
        open(HDR, "w").write(orig)
        open(COMPUTE, "w").write(orig_c)


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(int(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    else:
        main()
