"""Standalone test: SDPA that writes the concat-heads layout, against stock SDPA + concat.

The model runs stock SDPA ([B, 16, S, 64]) and then a concat-heads op
([B, 1, S, 1024]) in every layer. The model-local encoder SDPA can write the
concat layout directly (direct_concat_heads), which removes one op per layer.

For one S512 nomask shape this reports:
  stock   stock SDPA (model config) + bge_concat_heads_headsplit
  fused   bge_encoder_sdpa_experimental(direct_concat_heads=True)
Accuracy: cosine and PCC of each against a float32 torch reference, and fused
against stock. Time: traced, REPS calls in one trace, median per call.

Run (tt-metal root):
  python models/demos/wormhole/bge_m3/tests/perf/encoder_sdpa_concat.py --batch 1
"""

import argparse
import time

import torch

import ttnn

HEADS, SEQ, HEAD_DIM = 16, 512, 64
REPS, TIMED = 24, 30

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--q-chunk", type=int, default=64)
parser.add_argument("--k-chunk", type=int, default=512)
parser.add_argument("--grid-x", type=int, default=8, help="fused op grid")
parser.add_argument("--grid-y", type=int, default=8, help="fused op grid")
parser.add_argument("--stock-grid-x", type=int, default=None, help="stock SDPA grid (default: fused grid)")
parser.add_argument("--stock-grid-y", type=int, default=None)
parser.add_argument("--stock-max-cores-per-head-batch", type=int, default=None)
parser.add_argument("--stock-q-chunk", type=int, default=None)
parser.add_argument("--stock-k-chunk", type=int, default=None)
parser.add_argument("--streaming", action="store_true", help="fused op uses the streaming compute pipeline")
parser.add_argument("--concat-groups", type=int, default=4)
parser.add_argument(
    "--no-concat", action="store_true", help="fused op writes [B, H, S, D] (isolates the concat writer)"
)
parser.add_argument("--no-timing", action="store_true", help="accuracy only")
parser.add_argument("--fused-fp32-dest", action="store_true", help="fused op accumulates in fp32 DEST")
args = parser.parse_args()

from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa import EncoderSDPAConfig
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa.op import bge_encoder_sdpa_experimental
from models.demos.wormhole.bge_m3.tt.custom_ops.fused_concat_heads.op import bge_concat_heads_headsplit

B = args.batch
scale = 1.0 / (HEAD_DIM**0.5)
device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=20_000_000)
torch.manual_seed(0)
q_h, k_h, v_h = (torch.randn((B, HEADS, SEQ, HEAD_DIM), dtype=torch.bfloat16) for _ in range(3))
mem = ttnn.L1_MEMORY_CONFIG


def dev(t):
    return ttnn.from_torch(t, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=mem)


q, k, v = dev(q_h), dev(k_h), dev(v_h)
# Reference in the concat layout [B, 1, S, H*D], from the bf8-rounded inputs.
qr, kr, vr = (ttnn.to_torch(t).float() for t in (q, k, v))
ref = torch.softmax(qr @ kr.transpose(-1, -2) * scale, dim=-1) @ vr
ref = ref.permute(0, 2, 1, 3).reshape(B, 1, SEQ, HEADS * HEAD_DIM)

ckc = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
stock_kwargs = {
    "compute_with_storage_grid_size": ttnn.CoreCoord(
        args.stock_grid_x or args.grid_x, args.stock_grid_y or args.grid_y
    ),
    "q_chunk_size": args.stock_q_chunk or args.q_chunk,
    "k_chunk_size": args.stock_k_chunk or args.k_chunk,
    "exp_approx_mode": False,
}
if args.stock_max_cores_per_head_batch:
    stock_kwargs["max_cores_per_head_batch"] = args.stock_max_cores_per_head_batch
pcfg = ttnn.SDPAProgramConfig(**stock_kwargs)
fcfg = EncoderSDPAConfig(
    batch=B,
    num_q_heads=HEADS,
    num_kv_heads=HEADS,
    q_seq_len=SEQ,
    kv_seq_len=SEQ,
    head_dim=HEAD_DIM,
    q_chunk_size=args.q_chunk,
    k_chunk_size=args.k_chunk,
    grid_x=args.grid_x,
    grid_y=args.grid_y,
    scale=scale,
    use_streaming=args.streaming,
    fp32_dest_acc_en=args.fused_fp32_dest,
    direct_concat_heads=not args.no_concat,
)


# The model writes the stock SDPA output to DRAM for B > 1 (score_memcfg) and to L1 at B1.
sdpa_mem = mem if B == 1 else ttnn.DRAM_MEMORY_CONFIG


def stock_sdpa():
    return ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=scale, program_config=pcfg, compute_kernel_config=ckc, memory_config=sdpa_mem
    )


def stock():
    ctx = stock_sdpa()
    out = bge_concat_heads_headsplit(ctx, head_groups=args.concat_groups, out_memcfg=mem)
    ttnn.deallocate(ctx)
    return out


def fused():
    return bge_encoder_sdpa_experimental(q, k, v, config=fcfg, output_mem_config=mem)


def stats(name, out):
    t = ttnn.to_torch(out).float()
    if t.shape[1] == HEADS:
        t = t.permute(0, 2, 1, 3)
    t = t.reshape(ref.shape)
    cos = torch.nn.functional.cosine_similarity(t.flatten(), ref.flatten(), dim=0).item()
    pcc = torch.corrcoef(torch.stack([t.flatten(), ref.flatten()]))[0, 1].item()
    print("ACC %-6s shape %s cos %.6f pcc %.6f maxabs %.4f" % (name, tuple(out.shape), cos, pcc, (t - ref).abs().max()))
    ttnn.deallocate(out)
    return t


def traced_us(fn):
    # Free each output at once, as the model does, so L1 holds one output at a time.
    for _ in range(REPS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(REPS):
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ts = []
    for _ in range(TIMED):
        s = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        ts.append((time.perf_counter() - s) * 1e6 / REPS)
    ttnn.release_trace(device, tid)
    return sorted(ts)[len(ts) // 2]


print(
    "SHAPE B%d H%d S%d D%d | fused q%d k%d grid %dx%d streaming=%s | stock %s"
    % (B, HEADS, SEQ, HEAD_DIM, args.q_chunk, args.k_chunk, args.grid_x, args.grid_y, args.streaming, stock_kwargs),
    flush=True,
)
s_out = stats("stock", stock())
try:
    f_out = stats("fused", fused())
    cos = torch.nn.functional.cosine_similarity(f_out.flatten(), s_out.flatten(), dim=0).item()
    print("ACC fused-vs-stock cos %.6f" % cos, flush=True)
except Exception as exc:
    print("FUSED FAIL %s" % str(exc).split("\n")[0][:200], flush=True)
    ttnn.close_mesh_device(device)
    raise SystemExit(1)

for name, fn in () if args.no_timing else (("stock_sdpa", stock_sdpa), ("stock", stock), ("fused", fused)):
    print("TIME %-10s %.2f us/call" % (name, traced_us(fn)), flush=True)
ttnn.close_mesh_device(device)
