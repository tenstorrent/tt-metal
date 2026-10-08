import ttnn

f = ttnn.transformer.neighborhood_plan
vol = (145, 272, 480)
win = (11, 11, 11)
brick = (2, 4, 4)
res = (146, 80, 72)
for ch in [(1, 1, 1), (2, 1, 1), (1, 1, 2), (1, 2, 1), (4, 1, 1)]:
    try:
        p = f(
            vol,
            win,
            (1, 1, 1),
            brick,
            query_chunk_bricks=ch,
            shard_extent=res,
            shard_origin=(-1, 68 - 5, 60 - 5),
            query_extent=(145, 68, 60),
            query_origin=(1, 5, 5),
        )
        g, b, c = p["gather_brick_count"], p["bricks_per_query_chunk"], p["chunk_count"]
        print(ch, "gather", g, "b/chunk", b, "chunks", c, "pairs", c * b * g, flush=True)
    except Exception as e:
        print(ch, "ERR", str(e)[:200], flush=True)
