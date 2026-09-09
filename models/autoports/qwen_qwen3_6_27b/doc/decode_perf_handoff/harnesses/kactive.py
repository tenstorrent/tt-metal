"""Does the vLLM stall reproduce standalone at k active rows?"""
import argparse, time, torch, ttnn
from pathlib import Path
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=32)
p.add_argument("--active", type=int, default=31)
p.add_argument("--length", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--all-layers", action="store_true")
p.add_argument("--max-context", type=int, default=512)
p.add_argument("--scatter", action="store_true", help="shuffle the page table like vLLM's block manager does")
p.add_argument("--device-logits", action="store_true", help="read_from_device=False, the path vLLM uses with sampling")
a = p.parse_args()

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
try:
    kw = dict(max_context=a.max_context, batch=a.batch)
    if not a.all_layers:
        kw["num_layers"] = a.layers
    g = build_generator(Path("models/autoports/qwen_qwen3_6_27b"), mesh, **kw)
    if a.scatter:
        # vLLM's block manager hands out blocks from a large pool, so a slot's
        # blocks are neither contiguous nor ordered. allocate_page_table gives
        # a tidy arange, which is the one prefill input never reproduced
        # outside the server.
        import ttnn as _t
        flat = g.page_table_host.flatten()
        perm = torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(11))
        g.page_table_host = flat[perm].reshape(g.page_table_host.shape).contiguous()
        g._page_table = g._upload(g.page_table_host, dtype=_t.int32)
        print(f"  scattered page table: {g.page_table_host.shape} "
              f"row0[:6]={g.page_table_host[0][:6].tolist()}", flush=True)
    v = g.model.vocab_size
    tokens = torch.zeros((a.batch, a.length), dtype=torch.long)
    lens = [0] * a.batch
    for s in range(a.active):
        tokens[s, :a.length] = (torch.arange(a.length) * 7 + s * 101 + 3) % v
        lens[s] = a.length
    print(f"  batch={a.batch} active={a.active} length={a.length} layers={a.layers} max_context={a.max_context}", flush=True)
    t0 = time.perf_counter()
    g.prefill_forward(tokens, page_table=g._page_table, kv_cache=g.kv_cache, prompt_lens=lens,
                      read_from_device=not a.device_logits)
    ttnn.synchronize_device(mesh)
    print(f"  COMPLETED in {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
