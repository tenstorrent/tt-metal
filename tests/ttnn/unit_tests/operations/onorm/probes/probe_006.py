# Record the derived per-core block surface + L1 footprint at the R3 defaults.
import ttnn, torch
import ttnn.operations.onorm.onorm_program_descriptor as pd

print(
    f"KNOBS: TPB={pd.TOKENS_PER_BLOCK} NORM={pd.NORM_CHUNK_TOKENS} GATE={pd.GATE_CHUNK_TILES} "
    f"DEST={pd.GATE_DEST_TILES} DM={pd.DM_BLOCK_TILES}x{pd.DM_DEPTH} O={pd.O_DEPTH} "
    f"RECONFIG={pd.RECONFIG_MODE} k={pd.EXCHANGE_COST_PER_BLOCK}"
)
V, HV, FLAT, TILE = 128, 32, 4096, 32
v_tiles, flat_tiles = V // TILE, FLAT // TILE
for B, T in [(1, 32), (1, 64), (1, 128), (1, 640), (8, 640)]:
    blocks = B * ((T + pd.TOKENS_PER_BLOCK - 1) // pd.TOKENS_PER_BLOCK)
    G = pd._retile_group_cores_probe if False else None
    # replicate the policy exactly through the module function needs a device; use math
    total = 110

    def work(g):
        ng = min(blocks, total // g)
        return -(-blocks // ng) * (1 / g + pd.EXCHANGE_COST_PER_BLOCK)

    G, g = 1, 2
    while g <= pd.MAX_RETILE_GROUP_CORES:
        if pd.TOKENS_PER_BLOCK % g == 0 and flat_tiles % g == 0 and work(g) < work(G):
            G = g
        g *= 2
    tpc, cpc = pd.TOKENS_PER_BLOCK // G, flat_tiles // G
    fpc = (pd.TOKENS_PER_BLOCK // TILE) * cpc
    nb, gc = min(pd.NORM_CHUNK_TOKENS, tpc), min(pd.GATE_CHUNK_TILES, fpc)
    pages = (
        v_tiles * nb * pd.O_DEPTH
        + pd.DM_BLOCK_TILES * pd.DM_DEPTH * 2
        + v_tiles
        + 1
        + nb * 2
        + 2 * v_tiles * nb
        + fpc * 2
        + gc
        + (v_tiles * nb * pd.RM_LOCAL_DEPTH if G > 1 else 0)
    )
    ng = min(blocks, total // G)
    print(
        f"B={B:<2d}T={T:<4d} blocks={blocks:<4d} G={G:<3d} cores={ng*G:<4d} tok/core={tpc:<3d} cols/core={cpc:<4d} "
        f"norm_chunk={nb} gate_chunk={gc} chunks={tpc//nb}/{fpc//gc} L1={pages} pages = {pages*2048/1024:.0f} KB"
    )
