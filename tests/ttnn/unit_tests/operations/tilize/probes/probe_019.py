import math
import torch, ttnn


def info(name, t):
    mc = t.memory_config()
    print(f"=== {name} ===")
    print("  memory_layout:", mc.memory_layout, "buffer_type:", mc.buffer_type)
    try:
        print("  buffer_page_size:", t.buffer_page_size(), "buffer_num_pages:", t.buffer_num_pages())
    except Exception as e:
        print("  page/num err:", e)
    try:
        is_nd = mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED
        ss = mc.nd_shard_spec if is_nd else mc.shard_spec
        shape = ss.shard_shape if is_nd else ss.shape
        print("  shard_shape:", list(shape), "grid num_cores:", ss.grid.num_cores(), "orient:", ss.orientation)
    except Exception as e:
        print("  shard err:", e)


device = ttnn.open_device(device_id=0)
try:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})  # 4 cores

    # --- test_tilize_nd_sharded shapes (input nd) ---
    cases = [
        ([4, 128, 128], [2, 64, 64], None),
        ([3, 160, 160], [2, 64, 64], None),
        ([5, 4, 160, 160], [2, 3, 64, 96], None),
        ([23, 96, 160], [4, 64, 96], None),
        ([4, 128, 128], [2, 64, 64], [1, 64, 128]),
        ([3, 160, 160], [2, 64, 64], [1, 64, 96]),
        ([5, 4, 160, 160], [2, 3, 64, 96], [3, 2, 96, 64]),
        ([23, 96, 160], [4, 64, 96], [6, 64, 64]),
    ]
    for shape, sh, osh in cases:
        nd = ttnn.NdShardSpec(ttnn.Shape(sh), grid, ttnn.ShardOrientation.ROW_MAJOR)
        mc = ttnn.MemoryConfig(ttnn.BufferType.L1, nd)
        n = int(torch.tensor(shape).prod())
        x = torch.arange(n).reshape(shape).float().bfloat16()
        try:
            t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
            # compute n_shards
            nsh = 1
            for td, s in zip(shape, sh[-len(shape) :] if len(sh) >= len(shape) else sh):
                nsh *= -(-td // s)
            info(f"in-nd shape={shape} shard={sh} n_shards~={nsh}", t)
            if osh is not None:
                ond = ttnn.NdShardSpec(ttnn.Shape(osh), grid, ttnn.ShardOrientation.ROW_MAJOR)
                omc = ttnn.MemoryConfig(ttnn.BufferType.L1, ond)
                ot = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, omc)
                info(f"out-nd shape={shape} oshard={osh}", ot)
        except Exception as e:
            print(f"  ERR shape={shape} shard={sh}: {e}")

    # --- nd_sharded_to_legacy shapes ---
    print("\n#### nd -> legacy ####")
    leg_cases = [
        ([4, 128, 128], [2, 64, 64]),
        ([8, 64, 128], [2, 32, 64]),
        ([7, 128, 128], [2, 64, 96]),
    ]
    for shape, sh in leg_cases:
        nd = ttnn.NdShardSpec(ttnn.Shape(sh), grid, ttnn.ShardOrientation.ROW_MAJOR)
        mc = ttnn.MemoryConfig(ttnn.BufferType.L1, nd)
        n = int(torch.tensor(shape).prod())
        x = torch.arange(n).reshape(shape).float().bfloat16()
        try:
            t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
            info(f"leg-in shape={shape} shard={sh}", t)
        except Exception as e:
            print(f"  ERR {shape}: {e}")
        # legacy outputs
        H = 1
        for d in shape[:-1]:
            H *= d
        Wd = shape[-1]
        nc = 4
        for lay, oshape in [
            (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (H // nc, Wd)),
            (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (H, Wd // nc)),
            (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (H // int(math.sqrt(nc)), Wd // int(math.sqrt(nc)))),
        ]:
            try:
                osp = ttnn.ShardSpec(grid, list(oshape), ttnn.ShardOrientation.ROW_MAJOR)
                omc = ttnn.MemoryConfig(lay, ttnn.BufferType.L1, osp)
                ot = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, omc)
                info(f"leg-out {lay} shape={shape} oshard={list(oshape)}", ot)
            except Exception as e:
                print(f"  ERR leg-out {lay} {shape} {oshape}: {e}")
finally:
    ttnn.close_device(device)
