import ttnn

from models.experimental.deepseek_v4_flash.tt.decode_prefetch import DECODE_GCB_GROUP, DECODE_LAYOUTS
from models.experimental.deepseek_v4_flash.tt.l1_placement import placement_for
from models.experimental.deepseek_v4_flash.tt.layers import LinearDecode, fused_rms_norm_gamma_memory_config


def test_fused_rms_norm_gamma_is_width_sharded_on_the_weight_grid():
    n, num_cores = 1024, 32
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
    assert grid.num_cores() == num_cores

    mem = fused_rms_norm_gamma_memory_config(n, grid)

    assert mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert mem.buffer_type == ttnn.BufferType.L1
    assert tuple(mem.shard_spec.shape) == (1, n // num_cores)
    assert mem.shard_spec.grid == grid
    assert mem.shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR


def test_linear_decode_forwards_fused_rms_norm_group_size():
    layer = LinearDecode.__new__(LinearDecode)
    layer.output_core_grid = None
    layer.fused_rms_norm_eps = 1e-6
    layer.fused_rms_norm_gamma = 1.0
    layer.fused_rms_norm_group_size = 512
    output_memory_config = object()

    assert layer._epilogue_kwargs(output_memory_config) == {
        "output_mem_config": output_memory_config,
        "rms_norm": True,
        "rms_norm_gamma": 1.0,
        "rms_norm_epsilon": 1e-6,
        "rms_norm_group_size": 512,
    }


def test_linear_decode_accepts_scalar_fused_rms_norm_gamma():
    layer = LinearDecode.__new__(LinearDecode)
    layer.can_fuse_rms_norm = lambda: True

    assert layer.enable_fused_rms_norm(1e-6, 1.0, group_size=512)
    assert layer.fused_rms_norm_eps == 1e-6
    assert layer.fused_rms_norm_gamma == 1.0
    assert layer.fused_rms_norm_group_size == 512


def test_q_a_uses_full_width_32_core_layout():
    assert DECODE_LAYOUTS["q_a_proj"] == {"K": 4096, "N": 1024, "n_blocks": 32}


def test_q_a_uses_a_private_prefetch_ring():
    assert "q_a_proj" not in DECODE_GCB_GROUP


def test_kv_uses_full_width_16_core_layout():
    assert DECODE_LAYOUTS["kv_proj"] == {"K": 4096, "N": 512, "n_blocks": 16}


def test_kv_uses_a_private_prefetch_ring():
    assert "kv_proj" not in DECODE_GCB_GROUP


def test_packed_kv_matches_full_width_layout():
    placement = placement_for("kv_proj")
    assert placement.zone == "Z2"
    assert placement.k_blocks is None
    assert placement.n_blocks == 16
    assert placement.shard_shape == (4096, 32)


def test_packed_q_a_matches_full_width_layout():
    placement = placement_for("q_a_proj")
    assert placement.zone == "Z1"
    assert placement.k_blocks is None
    assert placement.n_blocks == 32
    assert placement.shard_shape == (4096, 32)
