# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import enum
from pathlib import Path

from models.demos.deepseek_v3.utils.serialize_configs import to_jsonable, from_jsonable
from models.demos.deepseek_v3.utils.config_dataclass import (
    MeshDeviceStub,
    FromWeightConfig,
    SavedWeight,
    LinearConfig,
    EmbeddingConfig,
    MulConfig,
    ReshardConfig,
    RMSNormConfig,
)


class DummyMeshDevice:
    def __init__(self, shape):
        self.shape = tuple(shape)


class DummyEnum(enum.Enum):
    SOME = 1


def assert_roundtrip_equal(obj, mesh_device=None):
    jsonable = to_jsonable(obj)
    rebuilt = from_jsonable(jsonable, mesh_device=mesh_device)
    assert rebuilt == obj
    return jsonable, rebuilt


def test_roundtrip_mesh_device_stub_and_from_weight_config():
    mds = MeshDeviceStub((1, 8))
    assert_roundtrip_equal(mds)
    fw = FromWeightConfig(mds)
    assert_roundtrip_equal(fw)


def test_roundtrip_saved_weight_with_path():
    sw = SavedWeight(path=Path("/tmp/foo.input_tensor_b"), memory_config=None)
    assert_roundtrip_equal(sw)


def test_roundtrip_linear_config_with_specs():
    # Use spec dicts for MemoryConfig and ProgramConfig to avoid requiring ttnn
    mem_spec = {
        "__type__": "ttnn.MemoryConfig",
        "memory_layout": "TILE",
        "buffer_type": "DRAM",
        "shard_spec": {"shape": [1, 8], "dims": [0, -1]},
    }
    pc_spec = {
        "__type__": "ttnn.MatmulMultiCoreReuseMultiCastProgramConfig",
        "compute_with_storage_grid_size": {"x": 8, "y": 8},
        "in0_block_w": 4,
        "per_core_M": 1,
    }
    lc = LinearConfig(
        input_tensor_b=FromWeightConfig(MeshDeviceStub((1, 8))),
        memory_config=mem_spec,  # spec dict
        compute_kernel_config=None,
        program_config=pc_spec,  # spec dict
    )
    jsonable = to_jsonable(lc)
    rebuilt = from_jsonable(jsonable)
    assert isinstance(rebuilt, LinearConfig)
    assert rebuilt.input_tensor_b == lc.input_tensor_b
    assert rebuilt.memory_config == mem_spec
    assert rebuilt.program_config == pc_spec


def test_roundtrip_embedding_and_mul_and_reshard_configs():
    mem_spec = {
        "__type__": "ttnn.MemoryConfig",
        "memory_layout": "TILE",
        "buffer_type": "DRAM",
    }
    ec = EmbeddingConfig(weight=FromWeightConfig(MeshDeviceStub((1, 8))), memory_config=mem_spec, layout="TILE_LAYOUT")
    mc = MulConfig(memory_config=mem_spec, input_tensor_a_activations=None)
    rc = ReshardConfig(memory_config=mem_spec, dtype="bfloat16")

    for obj in (ec, mc, rc):
        jsonable = to_jsonable(obj)
        rebuilt = from_jsonable(jsonable)
        assert obj == rebuilt


def test_roundtrip_rmsnorm_config_partial():
    mem_spec = {"__type__": "ttnn.MemoryConfig", "memory_layout": "L1_SHARDED", "buffer_type": "L1"}
    rn = RMSNormConfig(epsilon=1e-6, weight=None, bias=None, residual_input_tensor=None, memory_config=mem_spec)
    jsonable = to_jsonable(rn)
    rebuilt = from_jsonable(jsonable)
    assert rebuilt == rn


def test_enum_and_path_and_tensorref_specs():
    # Enum
    e_json = to_jsonable(DummyEnum.SOME)
    e_obj = from_jsonable(e_json)
    assert e_obj == DummyEnum.SOME

    # Path
    p = Path("foo/bar.txt")
    p_json = to_jsonable(p)
    p_obj = from_jsonable(p_json)
    assert p_obj == p

    # TensorRef spec remains a spec
    t_spec = {"__type__": "ttnn.TensorRef", "shape": [1, 1, 32, 4096], "dtype": "bfloat16"}
    jsonable = to_jsonable(t_spec)
    rebuilt = from_jsonable(jsonable)
    assert rebuilt == t_spec


def test_mesh_device_placeholder_injection():
    spec = {"__type__": "ttnn.MeshDevice", "mesh_shape": [1, 8]}
    injected = from_jsonable(spec, mesh_device=DummyMeshDevice((1, 8)))
    assert isinstance(injected, DummyMeshDevice)
    assert injected.shape == (1, 8)

