# 12B uses bfp8 activations (precision_overrides.json). Test bf8 vs bf16, and
# default vs explicitly-matched compute config. Metric: relative L2 error vs fp32 torch.
import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    DIM, eps = 3840, 1e-6
    torch.manual_seed(0)
    x_t = torch.randn(1, 1, 32, DIM)
    w_t = torch.randn(1, 1, DIM // 32, 32)
    ref = (x_t / torch.sqrt(x_t.pow(2).mean(-1, keepdim=True) + eps)) * w_t.reshape(1, 1, 1, DIM)
    def relerr(y):
        return (y.float() - ref).norm().item() / ref.norm().item()

    cfgs = {
        "default": None,
        "HiFi4+fp32acc": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
            math_approx_mode=False, packer_l1_acc=False),
    }
    for xdt_name, xdt in (("bf16", ttnn.bfloat16), ("bf8b", ttnn.bfloat8_b)):
        for cname, ckc in cfgs.items():
            x = ttnn.from_torch(x_t, xdt, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            w = ttnn.from_torch(w_t, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = {}
            for side, fn in (("gen", ttnn.rms_norm), ("nat", ttnn._native_rms_norm)):
                kw = {"weight": w, "epsilon": eps}
                if ckc is not None: kw["compute_kernel_config"] = ckc
                try:
                    out[side] = relerr(ttnn.to_torch(fn(x, **kw)))
                except Exception as e:
                    out[side] = str(e)[:50]
            f = lambda v: f"{v:.3e}" if isinstance(v, float) else v
            ratio = (f"{out['gen']/out['nat']:.2f}x"
                     if all(isinstance(v, float) for v in out.values()) else "-")
            print(f"x={xdt_name:5s} cfg={cname:14s} gen={f(out['gen']):>10s} "
                  f"nat={f(out['nat']):>10s}  gen/nat={ratio}", flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
