# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

import models.common.models.llama32_3b.model as model_module
from models.common.models.llama32_3b.model import Llama32_3BTransformer1D


def test_batched_prefill_postprocess_gathers_each_slots_last_token_before_norm(monkeypatch):
    calls = []
    hidden = object()
    selector_tt = object()
    gathered = object()
    normalized = object()
    all_gathered = object()
    logits = object()
    output = object()
    mesh = SimpleNamespace(arch=lambda: "wormhole")

    class FakeTTNN:
        bfloat16 = "bfloat16"
        TILE_LAYOUT = "tile"
        DRAM_MEMORY_CONFIG = "dram"
        MathFidelity = SimpleNamespace(HiFi4="hifi4")

        @staticmethod
        def ReplicateTensorToMesh(device):
            assert device is mesh
            return "replicate"

        @staticmethod
        def from_torch(selector, **kwargs):
            calls.append(("from_torch", selector.clone(), kwargs))
            return selector_tt

        @staticmethod
        def init_device_compute_kernel_config(arch, **kwargs):
            assert arch == "wormhole"
            return kwargs

        @staticmethod
        def matmul(lhs, rhs, **kwargs):
            calls.append(("matmul", lhs, rhs, kwargs))
            return gathered

        @staticmethod
        def deallocate(tensor):
            calls.append(("deallocate", tensor))

        @staticmethod
        def to_memory_config(tensor, memory_config):
            calls.append(("to_memory_config", tensor, memory_config))
            return output

    fake_norm = SimpleNamespace(prefill_forward=lambda tensor: calls.append(("norm", tensor)) or normalized)
    fake_lm_head = SimpleNamespace(
        config=SimpleNamespace(input_memcfg=None),
        forward=lambda tensor: calls.append(("lm_head", tensor)) or logits,
    )
    model = SimpleNamespace(mesh_device=mesh, norm=fake_norm, lm_head=fake_lm_head)
    monkeypatch.setattr(model_module, "ttnn", FakeTTNN)
    monkeypatch.setattr(
        model_module,
        "_all_gather_rmsnorm_tensor",
        lambda norm, tensor: calls.append(("all_gather", norm, tensor)) or all_gathered,
    )

    result = Llama32_3BTransformer1D.post_process_batched_prefill_output(
        model,
        hidden,
        last_token_idx_list=[3, 7, 11, 0],
        padded_batch=4,
        prefill_seq_len=32,
    )

    assert result is output
    selector = calls[0][1]
    assert selector.shape == (1, 1, 32, 128)
    assert selector.dtype == torch.bfloat16
    assert torch.count_nonzero(selector).item() == 4
    assert [selector[0, 0, row].nonzero().item() for row in range(4)] == [3, 39, 75, 96]
    assert calls[0][2] == {
        "device": mesh,
        "dtype": "bfloat16",
        "layout": "tile",
        "mesh_mapper": "replicate",
    }
    assert [call[0] for call in calls] == [
        "from_torch",
        "matmul",
        "deallocate",
        "norm",
        "all_gather",
        "lm_head",
        "to_memory_config",
    ]
    assert calls[1][1:3] == (selector_tt, hidden)
    assert calls[2] == ("deallocate", selector_tt)
    assert calls[3] == ("norm", gathered)
    assert calls[4] == ("all_gather", fake_norm, normalized)
    assert calls[5] == ("lm_head", all_gathered)
