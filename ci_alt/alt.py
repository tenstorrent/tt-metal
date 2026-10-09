# In-process alternation (05, 17:40 UTC): one process runs each production case with the exact-fidelity define on (forms
# AA1 and AA2) and off (OFF1 and OFF2), call by call; each pair is identical code under two no-op defines (the A/A copy
# on each side). The measurement build adds RC_ALT_<value of RC_ALT> to every kernel's defines
# (tt_metal/impl/kernels/kernel.cpp), and the four kernels drop their define under RC_ALT_OFF1 or RC_ALT_OFF2; the
# program cache is disabled, so each call creates its program with the current form. usage: python -m tracy -r -p --no-web-server -o OUT alt.py <run> <case,...>
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
    # Sentence BERT on Blackhole (models/demos/blackhole/sentence_bert): attention_softmax_ ignores its program config and
    # runs the attention-optimized softmax.cpp (scale, mask, numeric stable) on the 6x8 height-sharded bfp8 scores.
    smem = ttnn.create_sharded_memory_config((768, 384), core_grid=ttnn.CoreGrid(y=8, x=6), strategy=ttnn.ShardStrategy.HEIGHT,
                                             orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    sb = t((8, 12, 384, 384), ttnn.bfloat8_b, 10, smem)
    sm = ttnn.from_torch(torch.zeros((8, 1, 1, 384)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    scfg = ttnn.SoftmaxShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(6, 8), subblock_w=6, block_h=24, block_w=12)
    c["sbert_attention_softmax"] = lambda: ttnn.transformer.attention_softmax_(sb, attention_mask=sm, head_size=64, program_config=scfg)
    grid = dev.compute_with_storage_grid_size()
    print(f"GRID {grid.x}x{grid.y}", flush=True)
    if grid.x >= 12 and grid.y >= 10:
        mem = ttnn.create_sharded_memory_config((224, 224), core_grid=ttnn.CoreGrid(y=10, x=12), strategy=ttnn.ShardStrategy.HEIGHT,
                                                orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        v = t((10, 12, 224, 224), ttnn.bfloat8_b, 9, mem)
        pc = ttnn.SoftmaxShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(12, 10), subblock_w=7, block_h=7, block_w=7)
        c["vit_softmax_sharded"] = lambda: ttnn.softmax_in_place(v, program_config=pc)
    return c

forms = ["AA1", "OFF1", "AA2", "OFF2"]
K = int(os.environ.get("ALT_K", "8"))
for name, fn in cases().items():
    if want and name not in want:
        continue
    try:
        for f in forms:  # build and warm each form
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
    except Exception as e:
        print(f"FAILED {name}: {str(e)[:300]}", flush=True)
ttnn.close_device(dev)
