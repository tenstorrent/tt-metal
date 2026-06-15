"""matmul_decode SHARDED-CHAIN v2: reuse MatmulDecodeLinear's own shard-spec
helpers (which satisfy the full-WS inputB==output FATAL guard), but cut the
intermediate sharded->interleaved->sharded round-trips between gate/up/mul/down.

The wrapper's _call_full does: I2S(a) -> mmd -> S2I(y) per projection. In the MLP
block that is 3x [I2S + S2I] + the eltwise reshards. This bench tests whether we
can keep the activation SHARDED across gate->gelu->up->mul->down, paying ONE I2S
(entry) + ONE S2I (exit), using:
  - gate/up output: width-sharded along N (the wrapper's _b_l1_mc_full grid)
  - sharded gelu + sharded multiply (same shard spec -> no reshard)
  - down: needs A sharded along K=mlp. gate's N-shard grid != down's K-shard grid
    in general, so we test ttnn.reshard (sharded->sharded, 1 op) vs S2I+I2S (2 ops).

Compares device-synced eager min-of-N:
  (a) native interleaved chain (linear x3)
  (b) wrapper chain (3x MatmulDecodeLinear calls, the in-pipeline matmul_decode)
  (c) fused sharded chain (1 I2S + reshard-hinge + 1 S2I)

    python_env/bin/python models/experimental/pi0_5/tests/perf/bench_mmd_sharded_chain.py
"""

import os
import time
import importlib.util as ilu
import torch
import ttnn

BF16 = ttnn.bfloat16
BF8 = ttnn.bfloat8_b
M, K, MLP = 32, 1024, 4096
NIT = int(os.environ.get("NIT", "60"))
WARM = int(os.environ.get("WARM", "10"))

_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tt", "mmdecode", "matmul_decode_linear.py")
_s = ilu.spec_from_file_location("mdl", _P)
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
MDL = _m.MatmulDecodeLinear


def _pcc(a, b):
    t1, t2 = a.flatten().float(), b.flatten().float()
    if t1.numel() != t2.numel():
        return -1.0
    m1, m2, s1, s2 = t1.mean(), t2.mean(), t1.std(), t2.std()
    if s1 < 1e-9 or s2 < 1e-9:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (((t1 - m1) * (t2 - m2)).mean() / (s1 * s2)).item()


def _time(fn, dev):
    for _ in range(WARM):
        o = fn()
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    best = 1e18
    for _ in range(NIT):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        best = min(best, (time.perf_counter() - t0) * 1e6)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    return best


def main():
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768)
    ck = ttnn.init_device_compute_kernel_config(
        d.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    torch.manual_seed(0)
    a = torch.randn(M, K) * 0.1
    wg = torch.randn(K, MLP) * 0.05
    wu = torch.randn(K, MLP) * 0.05
    wd = torch.randn(MLP, K) * 0.05
    g = torch.nn.functional.gelu(a.float() @ wg.float())
    u = a.float() @ wu.float()
    ref = (g * u) @ wd.float()

    a_bf16 = ttnn.from_torch(a, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=d)
    a4 = ttnn.from_torch(a.reshape(1, 1, M, K), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=d)
    wg8 = ttnn.from_torch(wg, dtype=BF8, layout=ttnn.TILE_LAYOUT, device=d)
    wu8 = ttnn.from_torch(wu, dtype=BF8, layout=ttnn.TILE_LAYOUT, device=d)
    wd8 = ttnn.from_torch(wd, dtype=BF8, layout=ttnn.TILE_LAYOUT, device=d)

    # ---- (a) native interleaved ----
    def native():
        gg = ttnn.gelu(
            ttnn.linear(a_bf16, wg8, dtype=BF16, compute_kernel_config=ck), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        uu = ttnn.linear(a_bf16, wu8, dtype=BF16, compute_kernel_config=ck)
        hh = ttnn.multiply(gg, uu, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gg)
        ttnn.deallocate(uu)
        oo = ttnn.linear(hh, wd8, dtype=BF16, compute_kernel_config=ck)
        ttnn.deallocate(hh)
        return oo

    t_nat = _time(native, d)
    pcc_nat = _pcc(ref, ttnn.to_torch(native()).reshape(M, K))

    # ---- build the 3 wrapper instances ----
    mmd_g = MDL(d, wg8, weight_dtype=BF8, out_dtype=BF16, role="gate")
    mmd_u = MDL(d, wu8, weight_dtype=BF8, out_dtype=BF16, role="up")
    mmd_d = MDL(d, wd8, weight_dtype=BF8, out_dtype=BF16, role="down")

    # ---- (b) wrapper chain (each call: I2S + mmd + S2I) ----
    def wrapper_chain():
        gg = ttnn.gelu(mmd_g(a4), memory_config=ttnn.L1_MEMORY_CONFIG)
        uu = mmd_u(a4)
        hh = ttnn.multiply(gg, uu, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gg)
        ttnn.deallocate(uu)
        oo = mmd_d(hh)
        ttnn.deallocate(hh)
        return oo

    t_wrap = _time(wrapper_chain, d)
    pcc_wrap = _pcc(ref, ttnn.to_torch(wrapper_chain()).reshape(M, K))

    # ---- (c) fused sharded chain: 1 I2S entry, sharded gelu/mul, reshard hinge, 1 S2I exit ----
    diag = {}
    pcc_chain = -1
    try:
        # gate/up: single-group full-WS expected (N=4096). use the wrapper helpers.
        assert mmd_g.k_split_G == 1 and mmd_u.k_split_G == 1, f"G_g={mmd_g.k_split_G} G_u={mmd_u.k_split_G}"
        a_shmc = mmd_g._a_shard_mc(M, mmd_g.Kc_call)
        bg = ttnn.to_memory_config(mmd_g.b_groups[0], mmd_g._b_l1_mc_full(mmd_g.Kc_call, mmd_g.N))
        bu = ttnn.to_memory_config(mmd_u.b_groups[0], mmd_u._b_l1_mc_full(mmd_u.Kc_call, mmd_u.N))

        def chain():
            a_s = ttnn.interleaved_to_sharded(a_bf16, a_shmc)
            gs = ttnn.matmul_decode(a_s, bg, partial_width_sharded=False, dtype=BF16, compute_kernel_config=ck)
            gs = ttnn.gelu(gs, memory_config=gs.memory_config())
            us = ttnn.matmul_decode(a_s, bu, partial_width_sharded=False, dtype=BF16, compute_kernel_config=ck)
            hs = ttnn.multiply(gs, us, memory_config=gs.memory_config())
            ttnn.deallocate(gs)
            ttnn.deallocate(us)
            ttnn.deallocate(a_s)
            # hinge: hs is N=4096 width-sharded; down needs A K=4096 width-sharded.
            # reshard to down's A spec (1 sharded->sharded op).
            hs2 = ttnn.reshard(hs, mmd_d._a_shard_mc(M, mmd_d.Kc_call)) if mmd_d.k_split_G == 1 else hs
            ttnn.deallocate(hs)
            bd = ttnn.to_memory_config(mmd_d.b_groups[0], mmd_d._b_l1_mc_full(mmd_d.Kc_call, mmd_d.N))
            os_ = ttnn.matmul_decode(hs2, bd, partial_width_sharded=False, dtype=BF16, compute_kernel_config=ck)
            ttnn.deallocate(hs2)
            ttnn.deallocate(bd)
            oi = ttnn.sharded_to_interleaved(os_, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(os_)
            return oi

        out = chain()
        pcc_chain = _pcc(ref, ttnn.to_torch(out).reshape(M, K))
        diag["chain_pcc"] = round(pcc_chain, 5)
        ttnn.deallocate(out)
        t_chain = _time(chain, d) if pcc_chain > 0.9 else -1
    except Exception as e:
        diag["ERROR"] = repr(e)[:200]
        t_chain = -1

    print("=== sharded-chain diag ===")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    print(f"\n=== MLP block M={M} K={K} mlp={MLP} (us, min-of-{NIT}) ===")
    print(f"  (a) native interleaved : {t_nat:9.2f}  PCC {pcc_nat:.5f}")
    print(f"  (b) wrapper mmd chain  : {t_wrap:9.2f}  PCC {pcc_wrap:.5f}  ({t_wrap/t_nat:.2f}x native)")
    if t_chain > 0:
        print(f"  (c) fused sharded chain: {t_chain:9.2f}  PCC {pcc_chain:.5f}  ({t_chain/t_nat:.2f}x native)")
        print(f"METRIC chain_ratio={t_chain/t_nat:.3f}")
        print(f"METRIC wrapper_ratio={t_wrap/t_nat:.3f}")
    else:
        print(f"  (c) fused sharded chain: FAILED")
    ttnn.close_mesh_device(d)


if __name__ == "__main__":
    main()
