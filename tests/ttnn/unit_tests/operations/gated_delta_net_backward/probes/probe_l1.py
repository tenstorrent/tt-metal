import ttnn


def test_probe(device):
    print("L1 unreserved:", ttnn.get_max_worker_l1_unreserved_size())
    print("grid:", device.compute_with_storage_grid_size())
    print("arch:", device.arch())
    print("dram align", ttnn.get_dram_alignment(), "l1 align", ttnn.get_l1_alignment())
    print("tile f32", ttnn.tile_size(ttnn.float32), "bf16", ttnn.tile_size(ttnn.bfloat16))
