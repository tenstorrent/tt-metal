import torch, ttnn
from eval.memory import Sharded, place
from ttnn.operations.concat.concat import _dispatch, classify
from ttnn.operations.concat.concat_program_artifacts import select_regime

device = ttnn.open_device(device_id=0)
ENTRIES = [
    (-1, (1, 1, 64, 64), (1, 1, 64, 32)),          # neck join
    (-1, (1, 1, 64, 64), (1, 1, 64, 80)),
    (-2, (1, 64, 144), (1, 16, 144), (1, 4, 144)),
    (-2, (1, 2, 400), (1, 2, 400)),
    (-1, (1, 1, 32, 50), (1, 1, 32, 14)),
    (-2, (1, 1, 17, 64), (1, 1, 15, 64)),
    (-2, (2, 4, 32, 256), (2, 4, 96, 256), (2, 4, 32, 256)),
    (-1, (2, 1, 100, 32), (2, 1, 100, 96)),
    (-1, (4, 8, 32, 47), (4, 8, 32, 100), (4, 8, 32, 50)),
]
for layout in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
    for entry in ENTRIES:
        dim, *shapes = entry
        torch.manual_seed(0)
        host = [torch.randn(*s, dtype=torch.bfloat16) for s in shapes]
        dev = []
        for t in host:
            mc = place(Sharded("L1", "height"), t.shape, layout, device)
            dev.append(ttnn.from_torch(t, layout=layout, device=device, dtype=ttnn.bfloat16, memory_config=mc))
        axes = classify(dev, dim)
        ncores = [d.memory_config().shard_spec.grid.num_cores() for d in dev]
        shard = [tuple(d.memory_config().shard_spec.shape) for d in dev]
        try:
            out = _dispatch(dev, axes["dim"])
            got = ttnn.to_torch(out)
            ok = torch.equal(got, torch.cat(host, dim=dim))
            mc = out.memory_config()
            print("RES", layout, entry, select_regime([tuple(s) for s in shapes], axes["dim"], layout), ncores, shard, "OK" if ok else "MISMATCH", mc.buffer_type, mc.memory_layout, axes["sharding"])
        except Exception as e:
            print("RES", layout, entry, ncores, shard, "EXC", str(e)[:300])
ttnn.close_device(device)
