"""CFG batch-2 SDPA/KV BYTE-IDENTITY probe (ISOLATED — touches no production files).

Companion to cfg_batch2_byteident_probe.py (which proved every LM decode MATMUL is byte-identical
B=1 vs B=2 row-0).  This probe closes the remaining gate question: are the NON-matmul batched ops
in a B=2 LM decode forward — paged_update_cache, sdpa_decode, rms_norm — also byte-identical for
row 0 vs a plain B=1 forward?

  maxabsdiff == 0 on row-0 for every op  => the WHOLE B=2 forward is byte-identical per-row
                                            (matmuls proven elsewhere; rope/eltwise are per-row)
                                            => CFG batch-2 fusion is Tier-0 (long-form-safe by
                                               construction)
  any nonzero                            => math-changing => Tier-2 (must pass full 100-min render;
                                            real long-form-collapse risk — decide with the user)

The earlier cfg_batch2_probe.py checked these at PCC 0.99 only, which is far below the byte-identity
bar the acceptance gate requires.  This uses the real in-model configs (_SDPA_DECODE_CFG, _HIFI4,
the height-sharded KV-update memcfg) at VibeVoice's exact decode dims, a DEEP cache, and DIVERGENT
per-row positions (row0=pos-LM, row1=neg-LM differ by a fixed offset each frame) so any batch-driven
change in sdpa's online-softmax chunk-reduction order for row 0 would show up.

Run: python models/experimental/vibevoice/tests/perf/cfg_batch2_sdpa_byteident_probe.py
"""
import sys
import numpy as np
import torch
import ttnn

ROOT = "/home/ubuntu/vibe-voice/tt-metal"
sys.path.insert(0, ROOT)
from models.experimental.vibevoice.tt.ttnn_vibevoice_lm import _SDPA_DECODE_CFG, _HIFI4

H, N_HEADS, N_KV, HD = 1536, 12, 2, 128
MAX_SEQ = 8192  # deep, tile-aligned — mirrors long-context decode (chunking = f(cur_pos))
SCALE = 1.0 / np.sqrt(HD)

dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
torch.manual_seed(0)


def tt(a, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.as_tensor(a, device=dev, dtype=dt, layout=lay, memory_config=mc)


def maxdiff(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.abs(a - b).max())


_grid = dev.compute_with_storage_grid_size()


def kv_update_shard_mc(ncores):
    # [B,1,n_kv,hd] height-sharded L1: n_kv tile-pads to 32 rows => one core per batch row.
    sg = ttnn.num_cores_to_corerangeset(ncores, _grid, True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sg, [32, HD], ttnn.ShardOrientation.ROW_MAJOR),
    )


def new_cache(B):
    return ttnn.zeros(
        [B, N_KV, MAX_SEQ, HD],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


print(f"[probe] dims H={H} heads={N_HEADS} kv={N_KV} hd={HD} max_seq={MAX_SEQ}")
print("=== B=1 vs B=2 row-0 byte-identity for the non-matmul batched decode ops ===\n")

results = {}

# ── rms_norm: [2,1,1,H] row0 vs [1,1,1,H] (per-row reduction — expect identical) ──
w = tt(torch.randn(1, 1, 1, H, dtype=torch.bfloat16))
x_r0 = torch.randn(1, 1, 1, H, dtype=torch.bfloat16)
x_r1 = torch.randn(1, 1, 1, H, dtype=torch.bfloat16)
n1 = ttnn.to_torch(ttnn.rms_norm(tt(x_r0), weight=w, epsilon=1e-6, compute_kernel_config=_HIFI4)).float()
n2 = ttnn.to_torch(
    ttnn.rms_norm(tt(torch.cat([x_r0, x_r1], dim=0)), weight=w, epsilon=1e-6, compute_kernel_config=_HIFI4)
).float()
results["rms_norm"] = maxdiff(n1[0, 0, 0], n2[0, 0, 0])
print(
    f"  {'rms_norm':22s} row0 maxabsdiff={results['rms_norm']:.6e}  "
    f"{'IDENTICAL' if results['rms_norm'] == 0 else 'DIFFERS'}"
)

# ── paged_update_cache: batched write at [p0,p1] — row0 @p0 must equal the input, and the
#    row1 write at a different pos must not perturb row0 (byte-exact, not just PCC) ──
P0, P1 = 4000, 1500  # divergent, both mid-cache
k_r0 = torch.randn(1, 1, N_KV, HD, dtype=torch.bfloat16)
k_r1 = torch.randn(1, 1, N_KV, HD, dtype=torch.bfloat16)
cache2 = new_cache(2)
k_in2 = ttnn.to_memory_config(tt(torch.cat([k_r0, k_r1], dim=1)), kv_update_shard_mc(2))  # [1,2,n_kv,hd]
cur2_wr = tt(torch.tensor([P0, P1], dtype=torch.int32), dt=ttnn.int32, lay=ttnn.ROW_MAJOR_LAYOUT)
ttnn.experimental.paged_update_cache(cache2, k_in2, update_idxs_tensor=cur2_wr, page_table=None)
got2 = ttnn.to_torch(cache2).float()  # [2,n_kv,max,hd]
cache1 = new_cache(1)
k_in1 = ttnn.to_memory_config(tt(k_r0), kv_update_shard_mc(1))
cur1_wr = tt(torch.tensor([P0], dtype=torch.int32), dt=ttnn.int32, lay=ttnn.ROW_MAJOR_LAYOUT)
ttnn.experimental.paged_update_cache(cache1, k_in1, update_idxs_tensor=cur1_wr, page_table=None)
got1 = ttnn.to_torch(cache1).float()
results["paged_update_cache"] = maxdiff(got1[0], got2[0])  # full row-0 cache slab, all positions
print(
    f"  {'paged_update_cache':22s} row0 maxabsdiff={results['paged_update_cache']:.6e}  "
    f"{'IDENTICAL' if results['paged_update_cache'] == 0 else 'DIFFERS'}"
)


# ── sdpa_decode: batched KV [2,..] + cur=[p0,p1] vs B=1 KV=row0 + cur=[p0].  Test several
#    position pairs, incl. ones that put row0/row1 in different chunk counts, to expose any
#    batch-driven change in row-0's chunk-reduction order ──
def sdpa(K, V, Q, cur):
    return ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention_decode(
            Q,
            K,
            V,
            cur_pos_tensor=cur,
            scale=SCALE,
            program_config=_SDPA_DECODE_CFG,
            compute_kernel_config=_HIFI4,
        )
    ).float()  # [1,B,n_heads,hd]


worst_sdpa = 0.0
for pa, pb in [(4000, 1500), (300, 40), (6000, 511), (1024, 1023), (33, 4097)]:
    Kt = torch.randn(2, N_KV, MAX_SEQ, HD, dtype=torch.bfloat16)
    Vt = torch.randn(2, N_KV, MAX_SEQ, HD, dtype=torch.bfloat16)
    Qt = torch.randn(1, 2, N_HEADS, HD, dtype=torch.bfloat16)
    cur = tt(torch.tensor([pa, pb], dtype=torch.int32), dt=ttnn.int32, lay=ttnn.ROW_MAJOR_LAYOUT)
    a2 = sdpa(tt(Kt), tt(Vt), tt(Qt), cur)  # [1,2,n_heads,hd]
    cur0 = tt(torch.tensor([pa], dtype=torch.int32), dt=ttnn.int32, lay=ttnn.ROW_MAJOR_LAYOUT)
    a1 = sdpa(tt(Kt[0:1].contiguous()), tt(Vt[0:1].contiguous()), tt(Qt[:, 0:1].contiguous()), cur0)
    d = maxdiff(a1[0, 0], a2[0, 0])
    worst_sdpa = max(worst_sdpa, d)
    print(
        f"  {'sdpa_decode':22s} pos=({pa:5d},{pb:5d}) row0 maxabsdiff={d:.6e}  "
        f"{'IDENTICAL' if d == 0 else 'DIFFERS'}"
    )
results["sdpa_decode"] = worst_sdpa

print("\n=== SUMMARY ===")
for k, v in results.items():
    print(f"  {k:22s} {'IDENTICAL' if v == 0 else f'DIFFERS ({v:.2e})'}")
allid = all(v == 0 for v in results.values())
print(f"\n  ==> non-matmul B=2 ops byte-identical (row0): {allid}")
print(
    f"  ==> combined with matmul probe, WHOLE B=2 forward is Tier-0: "
    f"{'YES (integrate)' if allid else 'NO (Tier-2 — full-render gate, ask user)'}"
)
ttnn.close_mesh_device(dev)
