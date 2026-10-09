# In-process alternation (05, 17:40 UTC): one process runs each production case with the exact-fidelity define on (forms
# AA1 and AA2, identical code under two no-op defines, the A/A copy) and off (OFF), call by call. The measurement build
# adds RC_ALT_<value of the RC_ALT environment variable> to every kernel's defines (tt_metal/impl/kernels/kernel.cpp), and
# the four kernels read "#ifndef RC_ALT_OFF" around their define; the program cache is disabled, so each call creates its
# program with the current form. usage: python -m tracy -r -p --no-web-server -o OUT alt.py <run> <case,...>
import os, sys
import torch, ttnn
from tracy import signpost

run = sys.argv[1]
want = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
dev.disable_and_clear_program_cache()
hifi4 = lambda fp32: ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32)

def t(shape, dtype=ttnn.bfloat16, seed=0, mem=None, rand=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, generator=g) if rand else torch.randn(shape, generator=g)
    kw = {"memory_config": mem} if mem is not None else {}
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, **kw)

def cases():
    c = {}
    x = t((1, 16, 512, 512), ttnn.float32, 1); cc = hifi4(True)
    c["t5_softmax_fp32"] = lambda: ttnn.softmax(x, dim=-1, compute_kernel_config=cc)
    y = t((1, 12, 77, 77), ttnn.bfloat16, 2)
    c["clip_softmax"] = lambda: ttnn.softmax(y, dim=-1, numeric_stable=True, compute_kernel_config=cc)
    z = t((1, 1, 2048, 5376), ttnn.bfloat16, 3)
    c["gemma4_rms_5376"] = lambda: ttnn.rms_norm(z, compute_kernel_config=cc)
    w = t((1, 1, 2048, 2880), ttnn.bfloat16, 4)
    c["gptoss_rms_2880"] = lambda: ttnn.rms_norm(w)
    ry, rdy = t((32, 1, 256, 8), seed=5, rand=True), t((32, 1, 256, 8), seed=6)
    c["router_softmax_bw"] = lambda: ttnn.operations.moreh.softmax_backward(ry, rdy, 3)
    ay, ady = t((8, 32, 256, 256), seed=7, rand=True), t((8, 32, 256, 256), seed=8)
    c["attn_softmax_bw"] = lambda: ttnn.operations.moreh.softmax_backward(ay, ady, 3, compute_kernel_config=cc)
    grid = dev.compute_with_storage_grid_size()
    if grid.x >= 12 and grid.y >= 10:
        mem = ttnn.create_sharded_memory_config((224, 224), core_grid=ttnn.CoreGrid(y=10, x=12), strategy=ttnn.ShardStrategy.HEIGHT,
                                                orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        v = t((10, 12, 224, 224), ttnn.bfloat8_b, 9, mem)
        pc = ttnn.SoftmaxShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(12, 10), subblock_w=7, block_h=7, block_w=7)
        c["vit_softmax_sharded"] = lambda: ttnn.softmax_in_place(v, program_config=pc)
    return c

forms = ["AA1", "OFF", "AA2", "OFF"]
K = int(os.environ.get("ALT_K", "8"))
for name, fn in cases().items():
    if want and name not in want:
        continue
    for f in ("AA1", "AA2", "OFF"):  # build and warm each form
        os.environ["RC_ALT"] = f
        fn(); ttnn.synchronize_device(dev)
    for k in range(K):
        for f in (forms if k % 2 == 0 else forms[2:] + forms[:2]):
            os.environ["RC_ALT"] = f
            ttnn.synchronize_device(dev)
            signpost(header=f"{name}|{f}|{k}")
            fn()
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    print(f"CASE {name} ok", flush=True)
ttnn.close_device(dev)
