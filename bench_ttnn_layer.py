import os, sys, time

sys.path.insert(0, ".")
os.environ.setdefault("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
import torch, ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, get_rot_transformation_mat
from models.tt_transformers.tt.rope import get_rot_mats

SEQ = int(os.environ.get("SEQ", "512"))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    args = ModelArgs(mesh, max_batch_size=1, max_seq_len=SEQ, dummy_weights=True, cache_hf=True, use_hf_rope=False)
    args.n_layers = 1
    print(
        f"  CONFIG dim={args.dim} heads={args.n_heads} kv={args.n_kv_heads} head_dim={args.head_dim} "
        f"ffn={args.hidden_dim} seq={SEQ}",
        flush=True,
    )
    sd = args.load_state_dict()
    dtype = ttnn.bfloat8_b
    rot = get_rot_mats(
        head_dim=args.head_dim, device=mesh, seq_len=SEQ, theta=args.rope_theta, rope_scaling=args.rope_scaling
    )
    tmat = ttnn.as_tensor(
        get_rot_transformation_mat(args.head_dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    blk = TransformerBlock(
        mesh_device=mesh,
        tt_ccl=TT_CCL(mesh),
        state_dict=sd,
        weight_cache_path=args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats={"prefill": tmat},
        args=args,
        paged_attention_config=None,
    )
    x = args.prepare_residual_tensor_prefill((torch.rand(1, SEQ, args.dim) * 2 - 1))

    def once():
        return blk(x, None, rot_mats_global=rot, rot_mats_local=None, user_id=0, mode=Mode.PREFILL, page_table=None)

    once()
    ttnn.synchronize_device(mesh)
    n, t0 = 5, time.time()
    for _ in range(n):
        once()
    ttnn.synchronize_device(mesh)
    print(f"  RESULT ttnn TransformerBlock prefill S={SEQ}: {(time.time()-t0)/n*1e3:.2f}ms wall per layer", flush=True)
    # Sum every op's device time and divide by iterations: median_us from bench() is the
    # median of ONE op's duration, which for a ~20-op layer is not the layer's time.
    from unified_bench import Bench, _us

    ITERS = 6
    with Bench() as b:
        for _ in range(ITERS):
            once()
        ttnn.synchronize_device(mesh)
        ttnn.device.ReadDeviceProfiler(mesh)
        rows = list(b.collector.rows)
    per_iter = sum(_us(st, en, fr) for st, en, fr, _ in rows) / ITERS
    print(
        f"  RESULT device time summed over ops: {per_iter:.1f}us per layer "
        f"({len(rows) / ITERS:.0f} ops per forward)",
        flush=True,
    )
    from collections import defaultdict

    agg = defaultdict(float)
    for st, en, fr, src in rows:
        key = ",".join(sorted({s.split("/")[-1] for s in src}))[:58]
        agg[key] += _us(st, en, fr) / ITERS
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {v:8.1f}us  {k}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
