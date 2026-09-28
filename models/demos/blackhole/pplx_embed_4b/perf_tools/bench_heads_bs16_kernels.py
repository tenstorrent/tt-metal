# bs16 fused heads op: compute-kernel variants side by side, traced time per call and output PCC / max error vs v1.
# cos / sin in L1 like the model (QWEN_ROPE_PREFILL_L1=1); QKV input in L1 (the fused-QKV case: no DRAM read) and
# in DRAM (today's unfused op). Each variant runs in its own process (the kernel choice is read from env at call time,
# but the program cache would otherwise mix them). Usage: bench_heads_bs16_kernels.py [batch] [variant ...]
#   variants: v1 v2 v3 or path/to/compute.cpp (a v1-contract kernel); default v1 v2 v3
import os
import statistics
import subprocess
import sys
import tempfile
import time

NH, NKV, DH, EPS, S = 32, 8, 128, 1e-6, 512
ENVS = {"v1": {}, "v2": {"QWEN_FUSED_COMPUTE_V2": "1"}, "v3": {"QWEN_FUSED_COMPUTE_V3": "1"}}


def child(B, variant, out_dir):
    import torch

    import ttnn
    from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm import op as hop
    from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm.constants import make_norm_constants
    from models.tt_transformers.tt.common import get_rot_transformation_mat

    if variant not in ENVS:
        hop.COMPUTE_KERNEL = os.path.abspath(variant)
    torch.manual_seed(0)
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
    try:
        GQ, GK, SC, EP = make_norm_constants(torch.rand(DH) + 0.5, torch.rand(DH) + 0.5, EPS, D)
        ang = torch.rand(1, 1, S, DH) * 6.28
        L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
        mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
        cos, sin, T = mk(torch.cos(ang)), mk(torch.sin(ang)), mk(get_rot_transformation_mat(32))
        xt = torch.randn(B, 1, S, (NH + 2 * NKV) * DH)
        for pname, mc in (("L1 in", L1), ("DRAM in", DR)):
            x = ttnn.from_torch(xt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=mc)

            def fn():
                return hop.nlp_create_qkv_heads_norm_headsplit(
                    x, GQ, GK, SC, EP, num_heads=NH, num_kv_heads=NKV, memory_config=DR, rot_cos=cos,
                    rot_sin=sin, trans_mat=T, q_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat8_b, norm_eps=EPS,
                )  # fmt: skip

            outs = fn()
            if pname == "L1 in":
                torch.save([ttnn.to_torch(t).float() for t in outs], os.path.join(out_dir, "out.pt"))
            [ttnn.deallocate(t) for t in outs]
            for _ in range(2):
                [ttnn.deallocate(t) for t in fn()]
            ttnn.synchronize_device(D)
            n = 8
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
            [ttnn.deallocate(t) for o in outs for t in o]
            ttnn.deallocate(x)
            print(f"CHILD {pname.split()[0]} {statistics.median(ts):.1f}", flush=True)
    finally:
        ttnn.close_device(D)


def compare(ref, got):
    import torch

    res = []
    for name, a, b in zip("QKV", ref, got):
        a, b = a.flatten(), b.flatten()
        pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        res.append(f"{name} pcc={pcc:.6f} maxerr={(a - b).abs().max().item():.4f} exact={torch.equal(a, b)}")
    return "  ".join(res)


def main():
    import torch

    B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    variants = sys.argv[2:] or ["v1", "v2", "v3"]
    if "v1" not in variants:
        variants = ["v1"] + variants
    outs = {}
    for v in variants:
        d = tempfile.mkdtemp(prefix="heads_k_")
        env = dict(os.environ, **ENVS.get(v, {}))
        p = subprocess.run(
            [sys.executable, __file__, "--child", str(B), v, d], env=env, capture_output=True, text=True, timeout=900
        )
        res = dict(l.split()[1:] for l in p.stdout.splitlines() if l.startswith("CHILD"))
        if not res:
            err = [l for l in (p.stdout + p.stderr).splitlines() if "TT_THROW" in l or "Error:" in l or "error:" in l]
            print(f"RES B{B} {v}: FAILED {err[:4]}", flush=True)
            continue
        outs[v] = torch.load(os.path.join(d, "out.pt"))
        acc = compare(outs["v1"], outs[v]) if v != "v1" and "v1" in outs else ""
        print(
            f"RES B{B} {os.path.basename(v):28s} L1 in {res['L1']:>7s} us  DRAM in {res['DRAM']:>7s} us  {acc}",
            flush=True,
        )


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        main()
