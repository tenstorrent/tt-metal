# Per-head norm shape, PLAIN interleaved path, on the 1x4 mesh: check ALL FOUR devices.
import torch, ttnn
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=8192)
try:
    eps = 1e-6
    for shape, gamma in (((1,1,128,256), True), ((1,1,64,256), True), ((1,1,64,256), False)):
        torch.manual_seed(0)
        x_t = torch.randn(*shape, dtype=torch.bfloat16)
        w_t = torch.randn(1,1,shape[-1]//32,32, dtype=torch.bfloat16) if gamma else None
        xd = x_t.double()
        ref = xd/torch.sqrt(xd.pow(2).mean(-1,keepdim=True)+eps)
        if gamma: ref = ref*w_t.double().reshape(1,1,1,-1)
        # DIFFERENT data per device, so a wrong-device read is detectable
        xs = [x_t*(1+0.5*i) for i in range(4)]
        refs = [(xx.double()/torch.sqrt(xx.double().pow(2).mean(-1,keepdim=True)+eps))*(w_t.double().reshape(1,1,1,-1) if gamma else 1) for xx in xs]
        x = ttnn.from_torch(torch.cat(xs, 0), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0))
        w = None if not gamma else ttnn.from_torch(w_t, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md))
        for side, fn in (("gen", ttnn.rms_norm), ("nat", ttnn._native_rms_norm)):
            kw = {"epsilon": eps, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
            if gamma: kw["weight"] = w
            y = fn(x, **kw)
            per = []
            for i, t in enumerate(ttnn.get_device_tensors(y)):
                d = ttnn.to_torch(t).double()
                per.append(f"d{i}:{((d-refs[i]).norm()/refs[i].norm()).item():.2e}")
            print(f"{shape} gamma={gamma} {side}: " + " ".join(per), flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_mesh_device(md)
