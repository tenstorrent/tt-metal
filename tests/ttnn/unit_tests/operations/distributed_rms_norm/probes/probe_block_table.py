# L1 block_table format + allocator placement policy (does a new buffer take the highest free block?).
import torch, ttnn, pytest

FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}]


def show(m, tag):
    v = ttnn.get_memory_view(m, ttnn.BufferType.L1)
    print(tag, "largest", v.largest_contiguous_bytes_free_per_bank, "total", v.total_bytes_per_bank)
    for b in v.block_table:
        print("   ", b)


def mk(m, nbytes, cores=((0, 0), (0, 0))):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*cores[0]), ttnn.CoreCoord(*cores[1]))])
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, (1, nbytes // 4), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.allocate_tensor_on_device(ttnn.Shape([1, nbytes // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, m, mc)


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
def test_bt(mesh_device):
    m = mesh_device
    show(m, "empty")
    big = mk(m, 512 * 1024)
    small = mk(m, 4096)
    show(m, "big+small")
    big.deallocate()
    mid = mk(m, 64 * 1024)
    show(m, "after free big, alloc mid")
    s2 = mk(m, 4096)
    print("s2 addr", s2.buffer_address(), "small", small.buffer_address(), "mid", mid.buffer_address())
