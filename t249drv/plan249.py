import ttnn

f = ttnn.transformer.neighborhood_plan


def run(volume, window, brick, chunk, resident, qe, qo, so):
    p = f(
        volume,
        window,
        (1, 1, 1),
        brick,
        query_chunk_bricks=chunk,
        shard_extent=resident,
        shard_origin=so,
        query_extent=qe,
        query_origin=qo,
    )
    return p["gather_brick_count"], p["bricks_per_query_chunk"], p["chunk_count"]


for vol, win, brick, res in [
    ((21, 68, 120), (3, 7, 7), (8, 4, 1), (21, 68, 21)),
    ((41, 68, 120), (3, 5, 5), (16, 2, 1), (41, 68, 19)),
    ((81, 136, 240), (3, 5, 5), (8, 2, 2), (81, 136, 34)),
]:
    halo = (res[2] - vol[2] // 8) // 2
    for ch in [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (4, 1, 1), (1, 2, 2)]:
        try:
            g, b, c = run(vol, win, brick, ch, res, (res[0], res[1], vol[2] // 8), (0, 0, halo), (0, 0, -halo))
            print(vol, brick, ch, "gather", g, "bricks/chunk", b, "chunks", c, "pairs", c * b * g, flush=True)
        except Exception as e:
            print(vol, ch, "ERR", str(e)[:150], flush=True)
