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


@pytest.fixture(scope="module")
def loader():
    """Import the loader's pure key-parsing code without ttnn (its device-side imports are stubbed).

    The stubs and the loaded module live in sys.modules only while this fixture is active, under a
    private module name, and no existing module is modified.
    """
    pkg = "models.tt_dit.experimental.lora"
    stubs = {
        "models.tt_dit.layers.lora": types.SimpleNamespace(LoRAMixin=type("LoRAMixin", (), {})),
        f"{pkg}.promote": types.SimpleNamespace(promote_to_lora=lambda *a, **k: None),
    }
    name = f"{pkg}._ltx_adapter_loader_under_test"
    with pytest.MonkeyPatch.context() as mp:
        for key, stub in stubs.items():
            mp.setitem(sys.modules, key, stub)
        spec = importlib.util.spec_from_file_location(name, _LOADER)
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        yield mod


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
def test_alpha_reaches_every_target(loader, block_prefix, outer):
    _alpha_key, _collect_pairs, _parse_target = loader._alpha_key, loader._collect_pairs, loader._parse_target
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


def test_missing_alpha_defaults_to_rank(loader):
    _alpha_key, _collect_pairs = loader._alpha_key, loader._collect_pairs
    raw = {
        "transformer_blocks.0.ff.net.2.lora_A.weight": torch.zeros(RANK, 8),
        "transformer_blocks.0.ff.net.2.lora_B.weight": torch.zeros(8, RANK),
    }
    _, alphas, _, _ = _collect_pairs(raw)
    assert alphas.get(_alpha_key(0, "ff", "ff2"), float(RANK)) == float(RANK)


def _fused_delta(_loader, alphas):
    _alpha_key = _loader._alpha_key
    D, IN = 8, 6
    gen = torch.Generator().manual_seed(0)
    qkvs = {x: {"A": torch.randn(RANK, IN, generator=gen), "B": torch.randn(D, RANK, generator=gen)} for x in "qkv"}
    captured = {}

    class _Linear:
        def register_lora(self, A, B, scale, name):
            captured.update(A=A, B=B, scale=scale)
            return 0

    attn = types.SimpleNamespace(
        is_self=True,
        num_heads=1,
        head_dim=D,
        n_local_heads=1,
        dim=D,
        parallel_config=types.SimpleNamespace(tensor_parallel=types.SimpleNamespace(factor=1)),
        to_qkv=_Linear(),
    )
    keyed = {_alpha_key(0, "attn1", f"to_{x}"): a for x, a in alphas.items()}
    _loader._register_fused(attn, qkvs, keyed, 1.0, "t", 0, "attn1")
    fused = captured["scale"] * (captured["B"] @ captured["A"])
    expected = []
    for x in "qkv":
        B = _loader._permute_qk_rows(qkvs[x]["B"], 1, D) if x in "qk" else qkvs[x]["B"]
        expected.append(alphas.get(x, float(RANK)) / RANK * (B @ qkvs[x]["A"]))
    return fused, torch.cat(expected, dim=0)


def test_fused_qkv_uses_each_projections_alpha(loader):
    fused, expected = _fused_delta(loader, {"q": 4.0, "k": 8.0, "v": 16.0})
    torch.testing.assert_close(fused, expected)


def test_fused_qkv_equal_and_missing_alpha_unchanged(loader):
    for alphas in ({"q": 8.0, "k": 8.0, "v": 8.0}, {}):
        fused, expected = _fused_delta(loader, alphas)
        torch.testing.assert_close(fused, expected)
