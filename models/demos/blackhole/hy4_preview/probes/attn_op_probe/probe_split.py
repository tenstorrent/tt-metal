"""attn-op-probe S1b: DSV4 key-split decode kernel (tt/sparse_split.py) at Hy4 shapes. Unmodified file first (records the assert), then a probe-local copy with ONLY the python shape assert relaxed
(kernel sources under tt/sparse_split_kernels are used unmodified, mounted read-only). Not a rewrite: shows what else blocks."""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/probe")
import probe_sparse_sdpa as P

MD = "/home/container_app_user/tt-metal/models/demos/blackhole/deepseek_v41_flash/tt"
SRC = open(f"{MD}/sparse_split.py").read()


def load(relax):
    s = SRC
    if relax:
        a = "assert H == 32 and D == v_dim == 512 and sink is not None"
        assert a in s
        s = s.replace(a, "assert H == 32 and sink is not None  # PROBE: D==v_dim==512 assert dropped")
    s = s.replace(
        'KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_split_kernels")',
        f'KDIR = "{MD}/sparse_split_kernels"',
    )
    m = type(sys)("ss_probe")
    m.__file__ = f"{MD}/sparse_split.py"
    exec(compile(s, m.__file__, "exec"), m.__dict__)
    return m


def run(md, name, mod, D, S, topk, G, kc, T=65536, fmt="bf16", fid="hifi4"):
    fp8 = fmt == "fp8"
    torch.manual_seed(0)
    H = 32
    Hreal = 8
    VDv = 512
    kv = torch.randn(T, D).to(torch.bfloat16)
    per = T // S
    idx = P.make_idx(S, topk, [(s * per, (s + 1) * per) for s in range(S)], topk)
    q = torch.randn(Hreal, S, D).to(torch.bfloat16)
    sink_logit = torch.linspace(-2.0, 2.0, Hreal)
    qpad = torch.zeros(H, S, D, dtype=torch.bfloat16)
    qpad[:Hreal] = q
    sp = torch.zeros(H)
    sp[:Hreal] = sink_logit
    q_dev = P.to_dev(md, qpad.reshape(1, H, S, D), ttnn.bfloat16)
    if fp8:
        kv = (
            kv.float().to(torch.float8_e4m3fn).float().to(torch.bfloat16)
        )  # reference sees the plain-fp8 (no scale) rounding
        kv_dev = P.to_dev(md, kv.float().reshape(1, 1, T, D), ttnn.fp8_e4m3)
    else:
        kv_dev = P.to_dev(md, kv.reshape(1, 1, T, D), ttnn.bfloat16)
    ix32 = torch.where(idx < 0, torch.full_like(idx, -1), idx).to(torch.int32)
    idx_dev = P.to_dev(md, ix32.reshape(1, 1, S, topk), ttnn.uint32)
    sink_dev = P.to_dev(md, (sp / P.SCALE).reshape(1, 1, 1, H).to(torch.bfloat16), ttnn.bfloat16)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity={"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2}[fid],
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    res = {"case": name, "D": D, "S": S, "topk": topk, "G": G, "kc": kc, "fid": fid}
    if G == 0:  # stock op
        call = lambda: ttnn.transformer.sparse_sdpa(
            q_dev,
            kv_dev,
            idx_dev,
            VDv,
            kv_format=(ttnn.transformer.SparseKVFormat.FP8_E4M3 if fp8 else ttnn.transformer.SparseKVFormat.BF16),
            scale=P.SCALE,
            k_chunk_size=kc,
            compute_kernel_config=ckc,
            attention_sink=sink_dev,
        )
    else:
        call = lambda: mod.sparse_sdpa_split(q_dev, kv_dev, idx_dev, VDv, P.SCALE, kc, G, ckc, sink_dev)
    out = call()
    ttnn.synchronize_device(md)
    o = ttnn.to_torch(out)[0, :Hreal].float()
    P.VD = VDv
    ref = P.ref_sparse(q, kv, idx, sink_logit)
    res["pcc"] = round(P.pcc(o, ref), 6)
    for _ in range(3):
        ttnn.deallocate(call())
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(30):
        ttnn.deallocate(call())
    ttnn.synchronize_device(md)
    res["us_per_call"] = round((time.perf_counter() - t0) / 30 * 1e6, 1)
    try:
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(10):
            call()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        res["us_per_call_traced"] = round((time.perf_counter() - t0) / 30 * 1e6, 1)
        ttnn.release_trace(md, tid)
    except Exception as e:
        res["trace_err"] = str(e)[:200]
    print("RES " + json.dumps(res), flush=True)


CASES = [
    ("stock_D576_top2048_S4_plainfp8", False, dict(D=576, S=4, topk=2048, G=0, kc=128, fmt="fp8")),
    ("hy4_D576_top2048_S4_G8_plainfp8_relaxed", True, dict(D=576, S=4, topk=2048, G=8, kc=128, fmt="fp8")),
    ("hy4_D576_top2048_S4_G4_kc256_plainfp8_relaxed", True, dict(D=576, S=4, topk=2048, G=4, kc=256, fmt="fp8")),
    ("D512_top2048_S16_G4_kc256", False, dict(D=512, S=16, topk=2048, G=4, kc=256)),
    ("hy4_D576_top2048_S12_G4_kc256_relaxed", True, dict(D=576, S=12, topk=2048, G=4, kc=256)),
    ("hy4_D576_top2048_S16_G8_kc256_relaxed", True, dict(D=576, S=16, topk=2048, G=8, kc=256)),
    ("hy4_D576_top2048_S16_G4_kc256_T262k_relaxed", True, dict(D=576, S=16, topk=2048, G=4, kc=256, T=262144)),
    ("hy4_D576_top2048_S2_G8_kc256_relaxed", True, dict(D=576, S=2, topk=2048, G=8, kc=256)),
    ("hy4_D576_top2048_S4_G8_kc256_relaxed", True, dict(D=576, S=4, topk=2048, G=8, kc=256)),
    ("hy4_D576_top2048_S1_G4_kc256_relaxed", True, dict(D=576, S=1, topk=2048, G=4, kc=256)),
    ("hy4_D576_top2048_S4_G8_kc128_T1M_relaxed", True, dict(D=576, S=4, topk=2048, G=8, kc=128, T=1048576)),
    ("stock_D512_top640_S4", False, dict(D=512, S=4, topk=640, G=0, kc=128)),
    ("stock_D512_top2048_S4", False, dict(D=512, S=4, topk=2048, G=0, kc=128)),
    ("stock_D576_top2048_S4", False, dict(D=576, S=4, topk=2048, G=0, kc=128)),
    ("hy4_D576_top2048_S4_G2_kc512_relaxed", True, dict(D=576, S=4, topk=2048, G=2, kc=512)),
    ("hy4_D576_top2048_S4_G4_kc512_relaxed", True, dict(D=576, S=4, topk=2048, G=4, kc=512)),
    ("hy4_D576_top2048_S4_G8_kc512_relaxed", True, dict(D=576, S=4, topk=2048, G=8, kc=512)),
    ("hy4_D576_top2048_S1_G8_relaxed", True, dict(D=576, S=1, topk=2048, G=8, kc=128)),
    ("hy4_D576_top2048_S4_G4_kc256_hifi2_relaxed", True, dict(D=576, S=4, topk=2048, G=4, kc=256, fid="hifi2")),
    ("hy4_D576_top2048_S8_G4_kc256_relaxed", True, dict(D=576, S=8, topk=2048, G=4, kc=256)),
    ("hy4_D576_top2048_S16_G4_kc256_relaxed", True, dict(D=576, S=16, topk=2048, G=4, kc=256)),
    ("hy4_D576_top2048_S16_G2_kc512_relaxed", True, dict(D=576, S=16, topk=2048, G=2, kc=512)),
    ("native_D512_top640_S4_G4", False, dict(D=512, S=4, topk=640, G=4, kc=128)),
    ("hy4_D576_top2048_S4_G8_unmodified", False, dict(D=576, S=4, topk=2048, G=8, kc=128)),
    ("D512_top2048_S4_G8", False, dict(D=512, S=4, topk=2048, G=8, kc=128)),
    ("D512_top2048_S4_G8_hifi2", False, dict(D=512, S=4, topk=2048, G=8, kc=128, fid="hifi2")),
    ("D512_top2048_S4_G4_kc256", False, dict(D=512, S=4, topk=2048, G=4, kc=256)),
    ("D512_top2048_S4_G8_kc256", False, dict(D=512, S=4, topk=2048, G=8, kc=256)),
    ("D512_top2048_S16_G8", False, dict(D=512, S=16, topk=2048, G=8, kc=128)),
    ("D512_top2048_S4_G16_kc128", False, dict(D=512, S=4, topk=2048, G=16, kc=128)),
    ("hy4_D576_top2048_S4_G8_relaxed", True, dict(D=576, S=4, topk=2048, G=8, kc=128)),
    ("hy4_D576_top2048_S4_G4_kc256_relaxed", True, dict(D=576, S=4, topk=2048, G=4, kc=256)),
    ("hy4_D576_top2048_S4_G8_hifi2_relaxed", True, dict(D=576, S=4, topk=2048, G=8, kc=128, fid="hifi2")),
    ("hy4_D576_top2048_S16_G8_relaxed", True, dict(D=576, S=16, topk=2048, G=8, kc=128)),
]
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mods = {False: load(False), True: load(True)}
sel = os.environ.get("CASES")
try:
    for n, relax, kw in CASES:
        if sel and n not in sel.split(","):
            continue
        try:
            run(md, n, mods[relax], **kw)
        except Exception as e:
            print(
                "RES " + json.dumps({"case": n, "error": (str(e).strip().splitlines() or [repr(e)])[0][:300]}),
                flush=True,
            )
            print("ERRDETAIL", n, str(e)[:1200].replace("\n", " | "), flush=True)
finally:
    ttnn.close_mesh_device(md)
