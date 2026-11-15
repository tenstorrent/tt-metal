"""Embedding variants on the 1x8 row: dshard / vshard lookups vs torch (exact bf16 rows), traced latency, streams = hidden repeated into the 4 iHC channels."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "head_device.py")).read().split("for v in VARIANTS:")[0])
E = torch.load(f"{LMH}/embed.pt")["embed"]
g = torch.Generator().manual_seed(0)
for mode in os.environ.get("EMODES", "dshard,vshard").split(","):
    emb = Hy4DeviceEmbedding(md, mc, ccl, E, mode=mode)
    for T in TS:
        ids = torch.randint(0, VOCAB, (T,), generator=g)
        ids[0] = 0
        ids[-1] = VOCAB - 1
        if T > 2:
            ids[1] = 15104 * 3 + 5  # a shard boundary neighbour
        tk = emb.upload_tokens(ids)
        out = emb.forward(tk)
        h = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[:T]  # device 0's copy [T,1,4,D]
        ref = E[ids].float()
        exact = all(bool((h[:, 0, s, :] == ref).all()) for s in range(4))
        # all 8 devices identical?
        allh = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).reshape(8, T, 1, 4, DIM)
        ident = bool((allh == allh[0:1]).all())
        us = timed_trace(lambda: emb.forward(tk))
        us_l = timed_trace(lambda: emb.lookup(tk))
        res(
            event="emb",
            mode=mode,
            T=T,
            exact=exact,
            identical_on_all=ident,
            forward_us=round(us, 1),
            lookup_us=round(us_l, 1),
        )
    res(event="emb_bytes", mode=mode, per_chip_gib=VOCAB * DIM * 2 / 8 / 2**30)
ttnn.close_mesh_device(md)
