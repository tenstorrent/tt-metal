"""Minimal repro: capture ttnn trace of a SINGLE CCA layer's trace_decode.
Pinpoints the op that triggers 'Writes are not supported during trace capture'.
Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_trace_repro.py
"""
import os
import traceback
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaConfig, ZayaWeights
from models.demos.zaya1_8b.tt.cca import CCAAttention
from models.demos.zaya1_8b.tt.trace import TraceState
from models.demos.zaya1_8b.tt.standard import to_dev

C = ZayaConfig


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)), trace_region_size=200000000)
    device.enable_program_cache()
    w = ZayaWeights()
    print("[repro] building 1 CCA layer (layer 0)...", flush=True)
    cca = CCAAttention(device, w, 0)
    MAX = 64
    ts = TraceState(device, MAX)
    h1 = to_dev(torch.randn(1, 1, C.dim), device)

    pos = 1
    # populate per-pos inputs (host writes — OK, before capture)
    from models.demos.zaya1_8b.tt.standard import compute_cos_sin
    cos, sin = compute_cos_sin(pos + 1)
    ttnn.copy(to_dev(cos[pos].reshape(1, 1, 1, C.rotary_dim), device), ts.cos)
    ttnn.copy(to_dev(sin[pos].reshape(1, 1, 1, C.rotary_dim), device), ts.sin)
    am = torch.zeros(1, 1, 1, MAX); am[..., pos + 1:] = -1e30
    ttnn.copy(to_dev(am, device, dtype=ttnn.float32), ts.amask)
    oh = torch.zeros(1, C.n_kv_heads, MAX, C.head_dim); oh[:, :, pos, :] = 1.0
    ttnn.copy(to_dev(oh, device), ts.onehot)
    ttnn.copy(to_dev(1.0 - oh, device), ts.inv)

    WARM = os.environ.get("REPRO_WARM", "1") == "1"
    if WARM:
        print("[repro] eager warm-up run of trace_decode...", flush=True)
        try:
            _ = cca.trace_decode(h1, ts, 0)
            ttnn.synchronize_device(device)
            print("[repro] eager run OK", flush=True)
        except Exception:
            print("[repro] EAGER run failed:", flush=True)
            traceback.print_exc()
            os._exit(2)
    else:
        print("[repro] SKIPPING warm-up (cold capture)", flush=True)

    print("[repro] begin_trace_capture...", flush=True)
    try:
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        cca.trace_decode(h1, ts, 0)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        print("[repro] CAPTURE OK", flush=True)
    except Exception:
        print("[repro] CAPTURE failed:", flush=True)
        traceback.print_exc()
        os._exit(1)

    os._exit(0)


if __name__ == "__main__":
    main()
