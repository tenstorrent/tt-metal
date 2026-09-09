"""Is the batch-32 LM head paying 32 sequential slot projections?"""
import time, torch, ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import _to_device

HIDDEN, VOCAB, BATCH = 5120, 248320, 32
CHUNKS = 8  # the LM head is split into DRAM-sharded chunks

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
try:
    per = VOCAB // CHUNKS
    g = torch.Generator().manual_seed(3)
    weights = [
        _to_device(torch.randn(HIDDEN, per, generator=g, dtype=torch.bfloat16) * 0.02, mesh_device=mesh)
        for _ in range(CHUNKS)
    ]

    def run(rows):
        x = _to_device(torch.randn(1, 1, rows, HIDDEN, generator=g, dtype=torch.bfloat16), mesh_device=mesh)
        outs = [ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG) for w in weights]
        ttnn.synchronize_device(mesh)
        for o in outs:
            ttnn.deallocate(o)
        ttnn.deallocate(x)

    for label, rows, reps in (("one 32-row tile (single slot)", 32, 1), ("batched 32 rows", 32, 1)):
        run(rows)  # warm

    # A: 32 sequential per-slot projections, as the model does at batch 32
    t0 = time.perf_counter()
    for _ in range(BATCH):
        run(32)
    a = (time.perf_counter() - t0) * 1000.0

    # B: one projection of the same total rows
    t0 = time.perf_counter()
    run(32)
    b = (time.perf_counter() - t0) * 1000.0

    print(f"  A: {BATCH} sequential slot projections = {a:.1f} ms", flush=True)
    print(f"  B: 1 projection (32 rows)              = {b:.1f} ms", flush=True)
    print(f"  ratio A/B = {a / b:.1f}x", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
