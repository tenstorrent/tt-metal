import pytest
import math
from ttnn.operations.matmul_auto_config import (
    MatmulShapeProfile,
    MatmulConfigHeuristic,
    MatmulAutoConfig,
    MatmulConfigDatabase,
)


@pytest.mark.parametrize(
    "M, K, N, expected_type",
    [
        (32, 4096, 128, "1d"),
        (128, 4096, 32, "1d"),
        (1024, 4096, 1024, "2d"),
        (4096, 4096, 4096, "2d"),
        (64, 4096, 4096, "2d"),
        (32, 1024, 1024, "1d"),
        (8192, 4096, 4096, "2d"),
    ],
)
def test_heuristic_selects_valid_config(M, K, N, expected_type):
    heuristic = MatmulConfigHeuristic()
    config_name, params = heuristic.select(M=M, K=K, N=N, device_grid=(8, 8))
    assert config_name is not None
    assert len(params) > 0
    if expected_type == "1d":
        assert "1D" in config_name or "1d" in config_name
    else:
        assert "MultiCast" in config_name or "2d" in config_name.lower()


def test_heuristic_width_sharded():
    heuristic = MatmulConfigHeuristic()
    config_name, params = heuristic.select(
        M=1024, K=4096, N=4096, is_sharded=True, shard_layout="width_sharded",
        device_grid=(8, 8),
    )
    assert "1D" in config_name
    assert params.get("mcast_in0") is True


def test_heuristic_height_sharded():
    heuristic = MatmulConfigHeuristic()
    config_name, params = heuristic.select(
        M=1024, K=4096, N=4096, is_sharded=True, shard_layout="height_sharded",
        device_grid=(8, 8),
    )
    assert "1D" in config_name
    assert params.get("mcast_in0") is False


def test_heuristic_block_sharded():
    heuristic = MatmulConfigHeuristic()
    config_name, params = heuristic.select(
        M=1024, K=4096, N=1024, is_sharded=True, shard_layout="block_sharded",
        device_grid=(8, 8),
    )
    assert "MultiCast" in config_name


def test_heuristic_multi_device():
    heuristic = MatmulConfigHeuristic()
    config_name, params = heuristic.select(
        M=1024, K=4096, N=4096, num_devices=8, device_grid=(8, 8),
    )
    assert config_name is not None
    assert params.get("per_core_N", 0) >= 1


def test_shape_profile_compatibility():
    s1 = MatmulShapeProfile(M=1024, K=4096, N=4096)
    s2 = MatmulShapeProfile(M=1024, K=4096, N=4096)
    s3 = MatmulShapeProfile(M=2048, K=4096, N=4096)
    assert s1.is_compatible(s2)
    assert not s1.is_compatible(s3)


def test_database_store_and_query():
    db = MatmulConfigDatabase()
    shape = MatmulShapeProfile(M=1024, K=4096, N=4096)

    db.register(shape, "MatmulMultiCoreReuseMultiCastProgramConfig",
                {"in0_block_w": 2, "per_core_M": 4, "per_core_N": 8}, 500.0)

    db.register(shape, "MatmulMultiCoreReuseMultiCast1DProgramConfig",
                {"in0_block_w": 2, "mcast_in0": True}, 450.0)

    best = db.query(shape)
    assert best is not None
    assert best.config_name == "MatmulMultiCoreReuseMultiCast1DProgramConfig"
    assert best.runtime_us == 450.0


def test_database_closest_match():
    db = MatmulConfigDatabase()
    db.register(
        MatmulShapeProfile(M=1024, K=4096, N=4096),
        "MatmulMultiCoreReuseMultiCastProgramConfig",
        {"in0_block_w": 2, "per_core_M": 4, "per_core_N": 8},
        500.0,
    )
    close_shape = MatmulShapeProfile(M=1024, K=4096, N=4096)
    best = db.query(close_shape)
    assert best is not None


def test_auto_config_shape_extraction():
    pytest.importorskip("mock")
    import mock
    mock_tensor = mock.MagicMock()
    mock_tensor.shape = [1, 1, 1024, 4096]
    mock_tensor.dtype = "bfloat16"

    mock_b = mock.MagicMock()
    mock_b.shape = [1, 1, 4096, 4096]
    mock_b.dtype = "bfloat16"

    shape = MatmulAutoConfig._extract_shape(mock_tensor, mock_b)
    assert shape.M == 1024
    assert shape.K == 4096
    assert shape.N == 4096


def test_heuristic_returns_valid_all_shapes():
    heuristic = MatmulConfigHeuristic()
    shapes = [
        (1, 4096, 4096),
        (128, 64, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 128),
        (128, 4096, 4096),
    ]
    for M, K, N in shapes:
        config_name, params = heuristic.select(M=M, K=K, N=N, device_grid=(8, 8))
        assert config_name is not None, f"Failed for M={M}, K={K}, N={N}"
        assert len(params) > 0


def test_database_persistence(tmp_path):
    db_path = tmp_path / "test_matmul_db.pkl"
    db = MatmulConfigDatabase(db_path=str(db_path))
    shape = MatmulShapeProfile(M=1024, K=4096, N=4096)
    db.register(shape, "test_config", {"param": 1}, 100.0)
    assert db_path.exists()

    db2 = MatmulConfigDatabase(db_path=str(db_path))
    assert db2.profiles is not None
    assert len(db2.profiles) > 0


def test_database_to_json(tmp_path):
    db = MatmulConfigDatabase()
    shape = MatmulShapeProfile(M=1024, K=4096, N=4096)
    db.register(shape, "test_config", {"param": 1}, 100.0)
    json_path = tmp_path / "test_matmul_db.json"
    db.to_json(str(json_path))
    assert json_path.exists()


def test_all_config_names_valid():
    heuristic = MatmulConfigHeuristic()
    valid_configs = {
        "MatmulMultiCoreReuseProgramConfig",
        "MatmulMultiCoreReuseMultiCastProgramConfig",
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
        "MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig",
    }
    config_name, _ = heuristic.select(M=1024, K=4096, N=4096, device_grid=(8, 8))
    assert config_name in valid_configs, f"Unknown config: {config_name}"
