# Round 3 matmul (ts2 in the candidate): compile a broad sample of non-bmm compute kernels on the UMD mock cluster, for the
# ELF identity of kernels the change does not touch (the TRISC wrapper and the matmul headers are shared by all of them).
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import torch
import ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
print("arch", dev.arch(), flush=True)
T = ttnn.TILE_LAYOUT
ok = fail = 0


def run(name, fn):
    global ok, fail
    try:
        fn()
        ttnn.synchronize_device(dev)
        ok += 1
    except Exception as e:  # keep going: the point is the set of kernels compiled
        fail += 1
        print("FAIL", name, str(e).splitlines()[0][:200] if str(e) else type(e).__name__, flush=True)


def t(shape, dtype=ttnn.bfloat16, layout=T):
    return ttnn.from_torch(torch.rand(shape) - 0.5, dtype=dtype, layout=layout, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


for dtn, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b), ("fp32", ttnn.float32)):
    a, b = t((1, 1, 256, 512), dt), t((1, 1, 256, 512), dt)
    for op in ("add", "subtract", "multiply"):
        run(f"{op} {dtn}", lambda: getattr(ttnn, op)(a, b))
    run(f"add bcast {dtn}", lambda: ttnn.add(a, t((1, 1, 1, 512), dt)))
    for op in ("exp", "gelu", "relu", "sqrt", "reciprocal", "sigmoid", "silu", "neg", "abs", "log"):
        run(f"{op} {dtn}", lambda: getattr(ttnn, op)(a))
    for op in ("sum", "max", "mean"):
        for dim in (-1, -2):
            run(f"{op} dim {dim} {dtn}", lambda: getattr(ttnn, op)(a, dim=dim))
    run(f"softmax {dtn}", lambda: ttnn.softmax(a, dim=-1))
    run(f"transpose {dtn}", lambda: ttnn.transpose(a, -2, -1))
    run(f"layer_norm {dtn}", lambda: ttnn.layer_norm(a, weight=t((1, 1, 32, 512), ttnn.bfloat16)))
    run(f"rms_norm {dtn}", lambda: ttnn.rms_norm(a, weight=t((1, 1, 32, 512), ttnn.bfloat16)))
    run(f"typecast {dtn}", lambda: ttnn.typecast(a, ttnn.bfloat16 if dt != ttnn.bfloat16 else ttnn.float32))
    if dt != ttnn.bfloat8_b:
        r = t((1, 1, 256, 512), dt, ttnn.ROW_MAJOR_LAYOUT)
        run(f"tilize {dtn}", lambda: ttnn.to_layout(r, ttnn.TILE_LAYOUT))
        run(f"untilize {dtn}", lambda: ttnn.to_layout(a, ttnn.ROW_MAJOR_LAYOUT))
x = t((1, 1, 1024, 1024))
run("argmax", lambda: ttnn.argmax(ttnn.to_layout(t((1, 1, 32, 1024)), ttnn.ROW_MAJOR_LAYOUT), dim=-1))
run("topk", lambda: ttnn.topk(t((1, 1, 32, 1024)), 32))
q, k, v = t((1, 8, 256, 64)), t((1, 8, 256, 64)), t((1, 8, 256, 64))
run("sdpa", lambda: ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True))
qd, kd, vd = t((1, 1, 32, 64)), t((1, 8, 256, 64)), t((1, 8, 256, 64))
run("sdpa decode", lambda: ttnn.transformer.scaled_dot_product_attention_decode(
    qd, kd, vd, cur_pos=[255], scale=0.125))
w = ttnn.from_torch(torch.rand(64, 32, 3, 3) - 0.5, dtype=ttnn.bfloat16)
inp = ttnn.from_torch(torch.rand(1, 32, 32, 32) - 0.5, dtype=ttnn.bfloat16, device=dev)
run("conv2d", lambda: ttnn.conv2d(
    input_tensor=inp, weight_tensor=w, in_channels=32, out_channels=64, device=dev, kernel_size=(3, 3), stride=(1, 1),
    padding=(1, 1), batch_size=1, input_height=32, input_width=32))
run("max_pool2d", lambda: ttnn.max_pool2d(
    input_tensor=ttnn.reshape(ttnn.to_layout(t((1, 1, 1024, 32)), ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1024, 32)), batch_size=1,
    input_h=32, input_w=32, channels=32, kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1]))
try:
    run("minimal_matmul", lambda: ttnn.experimental.minimal_matmul(x, t((1, 1, 1024, 1024))))
except AttributeError:
    pass
print(f"done ok={ok} fail={fail}", flush=True)
ttnn.close_device(dev)
