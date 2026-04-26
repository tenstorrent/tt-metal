# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3.tt import generator as generator_module


class _FakeMesh:
    shape = (8, 8)


class _FakeMtpStateDict(dict):
    def __init__(self):
        super().__init__({"eh_proj.weight": object()})
        self.closed = False

    def close(self):
        self.closed = True


def test_generator_weight_cache_falls_back_to_base_cache_plus_mtp_conversion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    mtp_state_dict = _FakeMtpStateDict()

    class FakeLazyStateDict:
        def __init__(self, model_path: Path):
            self.model_path = model_path

        def view_with_prefix(self, prefix: str):
            calls.append({"lazy_prefix": prefix, "model_path": self.model_path})
            return mtp_state_dict

    def fake_get_weight_config(**kwargs):
        calls.append(kwargs)
        if (
            kwargs["ModuleClass"] is generator_module.RowBatchedModel
            and kwargs.get("use_weight_cache") is True
            and kwargs["cache_subdir_name"] == "61_layers_mtp"
        ):
            raise FileNotFoundError("combined MTP cache missing")
        if kwargs["ModuleClass"] is generator_module.RowBatchedModel:
            assert kwargs["cache_subdir_name"] == "61_layers"
            assert kwargs.get("use_weight_cache") is True
            return {"base": True}
        if kwargs["ModuleClass"] is generator_module.MTP2D:
            assert kwargs["state_dicts"] == (mtp_state_dict,)
            assert kwargs["cache_subdir_name"] == "61_layers_mtp_module"
            return {"mtp": True}
        raise AssertionError(f"Unexpected ModuleClass: {kwargs['ModuleClass']}")

    monkeypatch.setattr(generator_module, "LazyStateDict", FakeLazyStateDict)
    monkeypatch.setattr(generator_module, "get_weight_config", fake_get_weight_config)

    gen = object.__new__(generator_module.DeepseekGenerator)
    gen.hf_config = SimpleNamespace(num_hidden_layers=61)
    gen.mesh_device = _FakeMesh()
    gen.use_weight_cache = True
    gen.enable_mtp = True
    gen.force_recalculate = False
    gen.random_weights = False
    gen.model_path = str(tmp_path / "model")
    gen.single_layer = None

    gen._prepare_weight_configs(tmp_path / "cache")

    assert gen.model_weight_config == {"base": True, "mtp": {"mtp": True}}
    assert mtp_state_dict.closed is True
    assert calls[0]["cache_subdir_name"] == "61_layers_mtp"
    assert calls[1]["cache_subdir_name"] == "61_layers"
    assert calls[2] == {"lazy_prefix": "model.layers.61.", "model_path": tmp_path / "model"}
    assert calls[3]["cache_subdir_name"] == "61_layers_mtp_module"
