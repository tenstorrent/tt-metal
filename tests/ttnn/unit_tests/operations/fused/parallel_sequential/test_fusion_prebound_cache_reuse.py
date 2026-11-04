# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.device_utils import is_blackhole
from models.experimental.ops.descriptors.fusion import Parallel, Sequential, clear_build_cache
from models.experimental.ops.descriptors.op_descriptor import LazyOutputList, OpDescriptor
from tests.ttnn.unit_tests.operations.fused.parallel_sequential.test_parallel_sequential_global_cb import (
    build_identity_op,
)


pytestmark = pytest.mark.skipif(not is_blackhole(), reason="not yet run on Wormhole")


def _tensor(value, device):
    return ttnn.from_torch(
        torch.full((1, 1, 32, 128), value, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )


def _identity(source, core_x):
    outputs = LazyOutputList([None], lambda slots: slots.__setitem__(0, _tensor(0, source.device())))
    core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(core_x, 0), ttnn.CoreCoord(core_x, 0))})
    return OpDescriptor(
        factory_fn=lambda: build_identity_op(source, outputs[0], core, 4).descriptor,
        input_tensors=[source],
        output_tensors=outputs,
        program_cache_key=("prebound_cache_reuse", core_x),
    )


@pytest.mark.parametrize("container_type", [Parallel, Sequential])
@pytest.mark.parametrize("entry_point", ["run", "build"])
def test_fresh_prebound_containers_compile_once(device, monkeypatch, container_type, entry_point):
    clear_build_cache()
    builds = []
    original = container_type._build_internal

    def counted_build(self, *args, **kwargs):
        builds.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(container_type, "_build_internal", counted_build)
    retained = []
    count = 2 if container_type is Parallel else 1
    try:
        for iteration in range(3):
            value = 3 + iteration
            ops = [_identity(_tensor(value + core, device), core) for core in range(count)]
            for op in ops:
                op.output_tensors[0] = _tensor(0, device)
            fused = container_type(*ops)
            if entry_point == "run":
                results = fused.run()
            else:
                built = fused.build()
                built.launch()
                results = list(built.output_tensors)
            assert len(results) == count
            for core, result in enumerate(results):
                expected = torch.full((1, 1, 32, 128), value + core, dtype=torch.bfloat16)
                assert torch.equal(ttnn.to_torch(result), expected)
                retained.append((result, expected))
            for result, expected in retained:
                assert torch.equal(ttnn.to_torch(result), expected)
        assert len(builds) == 1
    finally:
        clear_build_cache()
