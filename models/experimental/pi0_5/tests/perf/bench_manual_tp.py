"""Manual TP (non-fused CCL) for one FFN layer: dodge the tt_dit fused-AG/RS hang.

Megatron MLP split across TP chips WITHOUT the fused all_gather_minimal_matmul:
  - ff1 weight column-sharded on inner (each chip owns inner/TP cols) -> local gelu
  - ff2 weight row-sharded on inner -> local partial [M, dim] per chip
  - ttnn.reduce_scatter or all_reduce to sum the partials -> output
Activation is REPLICATED (each chip has full [M, dim]); no input all-gather needed
(that's the op that hung). This is the cleaner TP primitive for decode M=32.

Keeps ONE parent 8x4 mesh open; carves a single 1xTP submesh (no churn/teardown
loop that destabilized eth cores). PP single-chip baseline carved from the same
parent. Device-synced eager min-of-N. PCC vs torch.

Gate G4(manual): TP layer wall-clock < PP single-chip (235us) => TP worth building.

    TP=2 python_env/bin/python models/experimental/pi0_5/tests/perf/bench_manual_tp.py
"""

import os
import time
import torch
import ttnn

DIM, INNER, M = 1024, 4096, 32
TP = int(os.environ.get("TP", "2"))
NIT = int(os.environ.get("NIT", "50"))
WARM = int(os.environ.get("WARM", "10"))
BF16, BF8 = ttnn.bfloat16, ttnn.bfloat8_b


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
    torch.manual_seed(0)
    a = torch.randn(M, DIM) * 0.1
    w1 = torch.randn(DIM, INNER) * 0.05  # ff1 [dim, inner]
    w2 = torch.randn(INNER, DIM) * 0.05  # ff2 [inner, dim]
    ref = torch.nn.functional.gelu(a.float() @ w1.float()) @ w2.float()

    diag = {}
    parent = None
    t_pp, t_tp, pcc_pp, pcc_tp = -1, -1, 0.0, 0.0
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), l1_small_size=32768)
        ck = ttnn.init_device_compute_kernel_config(
            parent.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # ---- PP single chip ----
        d1 = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        a1 = ttnn.from_torch(a, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=d1)
        w1_1 = ttnn.from_torch(w1, dtype=BF8, layout=ttnn.TILE_LAYOUT, device=d1)
        w2_1 = ttnn.from_torch(w2, dtype=BF8, layout=ttnn.TILE_LAYOUT, device=d1)

        def pp():
            h = ttnn.gelu(
                ttnn.linear(a1, w1_1, dtype=BF16, compute_kernel_config=ck), memory_config=ttnn.L1_MEMORY_CONFIG
            )
            o = ttnn.linear(h, w2_1, dtype=BF16, compute_kernel_config=ck)
            ttnn.deallocate(h)
            return o

        pcc_pp = _pcc(ref, ttnn.to_torch(pp()).reshape(M, DIM))
        t_pp = _time(pp, d1)
        diag["pp_pcc"] = round(pcc_pp, 5)

        # ---- manual TP across 1xTP submesh ----
        dT = parent.create_submesh(ttnn.MeshShape(1, TP), ttnn.MeshCoordinate(1, 0))
        # replicate activation; shard w1 on inner (dim1), shard w2 on inner (dim0)
        aT = ttnn.from_torch(
            a, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=dT, mesh_mapper=ttnn.ReplicateTensorToMesh(dT)
        )
        w1T = ttnn.from_torch(
            w1,
            dtype=BF8,
            layout=ttnn.TILE_LAYOUT,
            device=dT,
            mesh_mapper=ttnn.ShardTensor2dMesh(dT, mesh_shape=tuple(dT.shape), dims=[None, -1]),
        )
        w2T = ttnn.from_torch(
            w2,
            dtype=BF8,
            layout=ttnn.TILE_LAYOUT,
            device=dT,
            mesh_mapper=ttnn.ShardTensor2dMesh(dT, mesh_shape=tuple(dT.shape), dims=[None, -2]),
        )

        def tp():
            # each chip: [M,dim]@[dim,inner/TP] -> [M,inner/TP] +gelu
            h = ttnn.gelu(
                ttnn.linear(aT, w1T, dtype=BF16, compute_kernel_config=ck), memory_config=ttnn.L1_MEMORY_CONFIG
            )
            # each chip: [M,inner/TP]@[inner/TP,dim] -> partial [M,dim]
            part = ttnn.linear(h, w2T, dtype=BF16, compute_kernel_config=ck)
            ttnn.deallocate(h)
            # sum partials across TP chips (all_reduce along axis 1)
            out = ttnn.experimental.all_reduce_async(
                part,
                cluster_axis=1,
                mesh_device=dT,
                num_links=1,
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )
            ttnn.deallocate(part)
            return out

        out = tp()
        ot = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(M, DIM)
        pcc_tp = _pcc(ref, ot)
        diag["tp_pcc"] = round(pcc_tp, 5)
        ttnn.deallocate(out)
        t_tp = _time(tp, dT) if pcc_tp > 0.9 else -1
    except Exception as e:
        import traceback

        diag["ERROR"] = repr(e)[:200]
        diag["TRACE"] = traceback.format_exc()[-500:]
    finally:
        if parent is not None:
            ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    print("=== manual-TP diag ===")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    print(f"\n=== FFN dim={DIM} inner={INNER} M={M}  TP={TP} (us, min-of-{NIT}) ===")
    if t_pp > 0:
        print(f"  PP single-chip : {t_pp:9.2f}  PCC {pcc_pp:.5f}")
    if t_tp > 0 and t_pp > 0:
        print(f"  manual TP {TP}ch : {t_tp:9.2f}  PCC {pcc_tp:.5f}  ({t_tp/t_pp:.2f}x PP)")
        print(f"METRIC tp_ratio={t_tp/t_pp:.3f}")
        print(f"GATE G4manual: TP {'<' if t_tp < t_pp else '>='} PP => {'PASS' if t_tp < t_pp else 'FAIL'}")
    else:
        print("  TP: FAILED (see diag)")


if __name__ == "__main__":
    main()
