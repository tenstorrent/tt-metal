"""Gate G4: TP vs snake-PP for ONE FFN layer (denoise expert MLP volume).

The prefill/denoise pipelines are degenerate snake-PP: each layer runs on ONE
chip while the other 17 idle (bs=1). TP instead splits a single layer across N
chips (ColParallel ff1 shards inner, RowParallel ff2 + reduce-scatter), so all N
chips do useful work on the same layer.

This bench uses tt_dit's ParallelFeedForward (the production TP FFN: ColParallel
ff1 + gelu, RowParallel ff2 with fused all-gather-matmul / matmul-reduce-scatter)
vs the single-chip FeedForward, at the expert MLP volume (dim=1024, inner=4096).
Both validated against their own torch ref; the GATE is the wall-clock RATIO.
NOTE: this is a standard 2-matmul FFN (ff1->gelu->ff2), used as the TP-cost proxy.
The pi0.5 expert is GeGLU (gate*up), ~1.5x the ff1 volume, but the TP-vs-PP
structural verdict (does CCL overhead beat the idle-chip waste) is what G4 needs.

Device-synced eager min-of-N. Gate G4: TP layer wall-clock < PP single-chip layer.

    TP=4 python_env/bin/python models/experimental/pi0_5/tests/perf/bench_tp_vs_pp_layer.py
"""

import os
import time
import torch
import ttnn

from models.tt_dit.layers.feedforward import FeedForward, ParallelFeedForward
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor

DIM, INNER = 1024, 4096
SEQ = 32  # M (denoise action tile)
TP = int(os.environ.get("TP", "4"))
NIT = int(os.environ.get("NIT", "50"))
WARM = int(os.environ.get("WARM", "10"))


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


class TorchFFN(torch.nn.Module):
    def __init__(self, dim, inner):
        super().__init__()
        self.ff1 = torch.nn.Linear(dim, inner, bias=True)
        self.ff2 = torch.nn.Linear(inner, dim, bias=True)

    def forward(self, x):
        return self.ff2(torch.nn.functional.gelu(self.ff1(x)))


def main():
    torch.manual_seed(0)
    tmodel = TorchFFN(DIM, INNER).to(torch.bfloat16).eval()
    x = torch.randn(1, 1, SEQ, DIM, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = tmodel(x)

    diag = {}
    parent = None
    t_pp, t_tp, pcc_pp, pcc_tp = -1, -1, 0.0, 0.0
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), l1_small_size=32768)

        # ---- PP: single chip submesh ----
        d1 = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        pp_model = FeedForward(DIM, DIM, inner_dim=INNER, bias=True, activation_fn="gelu", mesh_device=d1)
        pp_model.load_torch_state_dict(tmodel.state_dict())
        from models.tt_dit.utils.tensor import bf16_tensor

        x1 = bf16_tensor(x, device=d1)

        def pp():
            return pp_model(x1)

        o = pp()
        pcc_pp = _pcc(ref.squeeze(), ttnn.to_torch(ttnn.get_device_tensors(o)[0]).squeeze())
        ttnn.deallocate(o)
        t_pp = _time(pp, d1)
        diag["pp_pcc"] = round(pcc_pp, 5)

        # ---- TP: 1xTP submesh ----
        dT = parent.create_submesh(ttnn.MeshShape(1, TP), ttnn.MeshCoordinate(1, 0))
        ccl = CCLManager(dT, topology=ttnn.Topology.Linear)
        pcfg = DiTParallelConfig(
            cfg_parallel=ParallelFactor(1, 0),
            tensor_parallel=ParallelFactor(TP, 1),
            sequence_parallel=ParallelFactor(1, 0),
        )
        tp_model = ParallelFeedForward(
            DIM,
            DIM,
            inner_dim=INNER,
            bias=True,
            activation_fn="gelu",
            mesh_device=dT,
            mesh_axis=1,
            fsdp_mesh_axis=None,
            ccl_manager=ccl,
        )
        tp_model.load_torch_state_dict(tmodel.state_dict())
        xT = bf16_tensor(x, device=dT)

        def tp():
            return tp_model(xT, parallel_config=pcfg)

        o = tp()
        # ff2 RowParallel output is replicated after reduce-scatter; take device 0
        ot = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).squeeze()
        pcc_tp = _pcc(ref.squeeze(), ot)
        diag["tp_pcc"] = round(pcc_tp, 5)
        ttnn.deallocate(o)
        t_tp = _time(tp, dT) if pcc_tp > 0.9 else -1
    except Exception as e:
        import traceback

        diag["ERROR"] = repr(e)[:200]
        diag["TRACE"] = traceback.format_exc()[-600:]
    finally:
        if parent is not None:
            ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    print("=== TP/PP diag ===")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    print(f"\n=== FFN layer dim={DIM} inner={INNER} M={SEQ}  TP={TP} (us, min-of-{NIT}) ===")
    if t_pp > 0:
        print(f"  PP single-chip : {t_pp:9.2f}  PCC {pcc_pp:.5f}")
    if t_tp > 0 and t_pp > 0:
        print(f"  TP {TP}-chip    : {t_tp:9.2f}  PCC {pcc_tp:.5f}  ({t_tp/t_pp:.2f}x PP)")
        print(f"METRIC tp_ratio={t_tp/t_pp:.3f}")
        print(f"GATE G4: TP {'<' if t_tp < t_pp else '>='} PP  => {'PASS' if t_tp < t_pp else 'FAIL'}")
    else:
        print("  TP: FAILED (see diag)")


if __name__ == "__main__":
    main()
