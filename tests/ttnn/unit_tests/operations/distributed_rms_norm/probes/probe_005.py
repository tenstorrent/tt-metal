import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
from eval.sharding import auto_shard_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get(None)
TSL = ttnn.TensorMemoryLayout
orig = pd.select_plan


def spy(inp, gamma, fp32, avail, G, budget):
    try:
        p = orig(inp, gamma, fp32, avail, G, budget)
        print(
            "PROBE plan",
            p.regime,
            "R",
            p.block_rows,
            "rows_max",
            p.rows_max,
            "budget",
            budget,
            "fp",
            pd.footprint(p.geometry, p.block_rows, p.x_depth),
            "S",
            p.geometry.S,
            "Wp",
            p.geometry.Wp,
        )
        return p
    except Exception as e:
        T = pd.tensor_tile_rows_of(inp)
        Wt = -(-inp.shape[-1] // 32)
        work, S, rows_max = pd._sharded_work(inp, T, Wt)
        G_w = max(w.index for w in work) + 1
        g = pd.compute_geometry(inp, gamma, pd.W_CHUNK_TILES, fp32, S, True, G_w > 1, G_w, False, G)
        print(
            "PROBE refused budget",
            budget,
            "fp(R=1)",
            pd.footprint(g, 1, 1),
            "shard",
            list(inp.memory_config().shard_spec.shape),
            "rows_max",
            rows_max,
            [(i, p * n) for i, _, p, n in pd.cb_inventory(g, 1, 1, None, None, None)],
        )
        raise


pd.select_plan = spy
for gd in (ttnn.float32, ttnn.bfloat16):
    try:
        run_distributed_rms_norm(
            ((1, 1, 2048, 2048),),
            mesh_device=mesh,
            memory_layout=TSL.HEIGHT_SHARDED,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            fp32_dest_acc_en=True,
            gamma_mode="gamma",
            gamma_dtype=gd,
            gamma_layout=ttnn.TILE_LAYOUT,
            cluster_axis=0,
            topology=ttnn.Topology.Linear,
            num_links=1,
        )
        print("PROBE ok", gd)
    except Exception as e:
        print("PROBE fail", gd, str(e)[:80])
cache.close()
