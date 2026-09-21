# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Cold/warm converted-weight cache validation, including every mesh shard."""

import hashlib
import json
import os
from pathlib import Path
import time
from typing import NamedTuple

import pytest


@pytest.fixture
def device_params():
    import ttnn

    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["bh_lb"], indirect=True)
def test_weight_cache(mesh_device):
    import torch
    import ttnn

    from models.tt_dit.layers.module import Module, Parameter
    from models.tt_dit.utils import cache

    class Parallel(NamedTuple):
        pass

    class Weights(Module):
        def __init__(self):
            super().__init__()
            for name, axes in (("replicated", (None, None)), ("column", (None, 1)), ("row", (1, None))):
                setattr(self, name, Parameter(total_shape=(4096, 4096), device=mesh_device, mesh_axes=axes))

        def forward(self):
            raise NotImplementedError

    phase = os.environ["FLUX2_CACHE_PHASE"]
    assert phase in ("cold", "warm")
    output = Path(os.environ["FLUX2_CACHE_PROBE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    calls = []

    def original_weights():
        assert phase == "cold", "Warm load unexpectedly attempted conversion"
        calls.append(1)
        generator = torch.Generator().manual_seed(42)
        return {
            name: torch.randn((4096, 4096), generator=generator, dtype=torch.bfloat16)
            for name in ("replicated", "column", "row")
        }

    model = Weights()
    start = time.perf_counter()
    cache.load_model(
        model,
        model_name="cache-validation-v1",
        subfolder="weights",
        parallel_config=Parallel(),
        mesh_shape=(2, 4),
        mesh_device=mesh_device,
        get_torch_state_dict=original_weights,
    )
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    hashes = {}
    reference_generator = torch.Generator().manual_seed(42)
    for name, parameter in model.named_parameters():
        original = torch.randn((4096, 4096), generator=reference_generator, dtype=torch.bfloat16)
        expected = (
            [original] * 8 if name == "replicated" else list(original.chunk(4, dim=1 if name == "column" else 0)) * 2
        )
        hashes[name] = []
        for shard, reference in zip(ttnn.get_device_tensors(parameter.data), expected, strict=True):
            actual = ttnn.to_torch(shard).contiguous()
            assert torch.equal(actual, reference), f"Cached {name} differs from the original BF16 source"
            hashes[name].append(hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest())
        assert len(hashes[name]) == 8
    record = {
        "phase": phase,
        "load_and_sync_seconds": elapsed,
        "conversion_calls": len(calls),
        "exact_vs_original_torch": True,
        "hashes": hashes,
    }
    if phase == "warm":
        assert hashes == json.loads((output / "cold.json").read_text())["hashes"]
        assert not calls
    else:
        assert len(calls) == 1
    (output / f"{phase}.json").write_text(json.dumps(record, indent=2) + "\n")
    print("CACHE_VALIDATION", json.dumps(record), flush=True)
