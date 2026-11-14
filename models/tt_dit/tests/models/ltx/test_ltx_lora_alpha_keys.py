# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only test: LoRA ``.alpha`` values must reach every target the LTX loader resolves."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_LOADER = Path(__file__).resolve().parents[3] / "experimental" / "lora" / "ltx_adapter_loader.py"


def _load_loader():
    """Import the loader's pure key-parsing code without ttnn (its device-side imports are stubbed)."""
    pkg = "models.tt_dit.experimental.lora"
    stubs = {
        "models.tt_dit.layers.lora": types.SimpleNamespace(LoRAMixin=type("LoRAMixin", (), {})),
        f"{pkg}.promote": types.SimpleNamespace(promote_to_lora=lambda *a, **k: None),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(f"{pkg}.ltx_adapter_loader", _LOADER)
        mod = importlib.util.module_from_spec(spec)
        for name in ("models", "models.tt_dit", "models.tt_dit.experimental", pkg):
            sys.modules.setdefault(name, types.ModuleType(name)).__path__ = []
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


_loader = _load_loader()
_alpha_key, _collect_pairs, _parse_target = _loader._alpha_key, _loader._collect_pairs, _loader._parse_target

RANK = 4
ALPHA = 8.0

TARGETS = [
    "attn1.to_q",
    "attn1.to_out.0",
    "attn2.to_k",
    "ff.net.0.proj",
    "ff.net.2",
]


@pytest.mark.parametrize("block_prefix", ["transformer_blocks", "blocks"])
@pytest.mark.parametrize("outer", ["", "diffusion_model."])
def test_alpha_reaches_every_target(block_prefix, outer):
    raw = {}
    for t in TARGETS:
        base = f"{outer}{block_prefix}.3.{t}"
        raw[f"{base}.lora_A.weight"] = torch.zeros(RANK, 8)
        raw[f"{base}.lora_B.weight"] = torch.zeros(8, RANK)
        raw[f"{base}.alpha"] = torch.tensor(ALPHA)
    pairs, alphas, _, unrecognized = _collect_pairs(raw)
    assert not unrecognized
    assert len(pairs) == len(TARGETS)
    for base in pairs:
        parsed = _parse_target(base)
        assert parsed is not None
        assert alphas.get(_alpha_key(*parsed), float(RANK)) / RANK == ALPHA / RANK == 2.0, base


def test_missing_alpha_defaults_to_rank():
    raw = {
        "transformer_blocks.0.ff.net.2.lora_A.weight": torch.zeros(RANK, 8),
        "transformer_blocks.0.ff.net.2.lora_B.weight": torch.zeros(8, RANK),
    }
    _, alphas, _, _ = _collect_pairs(raw)
    assert alphas.get(_alpha_key(0, "ff", "ff2"), float(RANK)) == float(RANK)
