# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-host tests for the trace loader in host_io_decoder_harness.

Exercises both the legacy ``.pt`` path and the bit_sculpt safetensors per-layer
layout, including ``--trace-format=auto`` resolution and the symlink convention
that bit_sculpt's ``DebugTracer`` uses for non-zero-layer input files. No device
required — all tensors stay on CPU.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as safetensors_save_file

from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions as D
from models.demos.deepseek_v3_b1.tests.unit_tests.host_io_decoder_harness import (
    _load_reference_trace,
    _resolve_trace_format,
)


def _make_pt_trace(trace_dir: Path, prompt: str, seq_len: int = 4) -> None:
    torch.save(
        {
            "input": torch.randn(seq_len, D.HIDDEN_SIZE, dtype=torch.bfloat16),
            "output": torch.randn(seq_len, D.HIDDEN_SIZE, dtype=torch.bfloat16),
        },
        trace_dir / f"{prompt}.pt",
    )


def _make_safetensors_trace(
    trace_dir: Path,
    prompt: str,
    *,
    layers: tuple[int, ...] = (0, 1, 2),
    seq_len: int = 4,
    write_metadata: bool = True,
) -> dict[int, dict[str, torch.Tensor]]:
    """Create a fake bit_sculpt-style safetensors trace and return the written tensors.

    Mirrors ``analysis.debug_trace.DebugTracer._save_step``'s group-A layout:
      - decoder_input_layer_0.safetensors      (real file)
      - decoder_output_layer_{L}.safetensors   (real, every layer)
      - decoder_input_layer_{L}.safetensors    (symlink to decoder_output_layer_{L-1}, L >= 1)
    The returned dict[layer_idx][kind] lets the test assert exact tensor equality.
    """
    prompt_dir = trace_dir / prompt
    prompt_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, dict[str, torch.Tensor]] = {}

    layer0_input = torch.randn(seq_len, D.HIDDEN_SIZE, dtype=torch.bfloat16)
    safetensors_save_file(
        {"decoder_input_layer_0": layer0_input},
        str(prompt_dir / "decoder_input_layer_0.safetensors"),
    )
    written[0] = {"input": layer0_input}

    for layer in layers:
        out = torch.randn(seq_len, D.HIDDEN_SIZE, dtype=torch.bfloat16)
        safetensors_save_file(
            {f"decoder_output_layer_{layer}": out},
            str(prompt_dir / f"decoder_output_layer_{layer}.safetensors"),
        )
        written.setdefault(layer, {})["output"] = out
        # The next layer's input file is a relative symlink to this output file.
        # bit_sculpt does the same — see DebugTracer._save_step around line 583.
        if layer + 1 in layers:
            link = prompt_dir / f"decoder_input_layer_{layer + 1}.safetensors"
            link.symlink_to(f"decoder_output_layer_{layer}.safetensors")
            written.setdefault(layer + 1, {})["input"] = out

    if write_metadata:
        (prompt_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "n_layers": len(layers),
                    "hidden_dim": D.HIDDEN_SIZE,
                    "kv_lora_rank": 512,
                    "qk_rope_head_dim": 64,
                }
            )
        )

    return written


# ----- format resolution -----


def test_resolve_format_explicit_passthrough(tmp_path: Path) -> None:
    assert _resolve_trace_format(tmp_path, "x", "pt") == "pt"
    assert _resolve_trace_format(tmp_path, "x", "safetensors") == "safetensors"


def test_resolve_format_auto_prefers_pt(tmp_path: Path) -> None:
    """When both formats exist, .pt wins (back-compat with original DeepSeek pipeclean)."""
    (tmp_path / "foo.pt").touch()
    (tmp_path / "foo").mkdir()
    assert _resolve_trace_format(tmp_path, "foo", "auto") == "pt"


def test_resolve_format_auto_falls_back_to_safetensors(tmp_path: Path) -> None:
    (tmp_path / "foo").mkdir()
    assert _resolve_trace_format(tmp_path, "foo", "auto") == "safetensors"


def test_resolve_format_auto_raises_when_neither_exists(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Probed:"):
        _resolve_trace_format(tmp_path, "missing", "auto")


# ----- loading: .pt path -----


def test_load_pt_trace_returns_expected_dict(tmp_path: Path) -> None:
    _make_pt_trace(tmp_path, "p")
    trace = _load_reference_trace(tmp_path, "p", decoder_layer_idx=4, trace_format="pt")
    assert set(trace.keys()) == {"input", "output"}
    assert trace["input"].shape == (4, D.HIDDEN_SIZE)
    assert trace["output"].shape == (4, D.HIDDEN_SIZE)
    assert trace["input"].dtype == torch.bfloat16


# ----- loading: safetensors path -----


def test_load_safetensors_layer_0(tmp_path: Path) -> None:
    """Layer 0: real decoder_input_layer_0 + real decoder_output_layer_0."""
    written = _make_safetensors_trace(tmp_path, "p")
    trace = _load_reference_trace(tmp_path, "p", decoder_layer_idx=0, trace_format="safetensors")
    assert torch.equal(trace["input"], written[0]["input"])
    assert torch.equal(trace["output"], written[0]["output"])


def test_load_safetensors_layer_via_symlink(tmp_path: Path) -> None:
    """Layer L>=1: decoder_input_layer_{L} is a symlink → decoder_output_layer_{L-1}.

    The tensor key inside the file is the *target's* canonical key
    (decoder_output_layer_{L-1}), not the symlink name. The loader must accept
    whatever single key is there.
    """
    written = _make_safetensors_trace(tmp_path, "p")
    trace = _load_reference_trace(tmp_path, "p", decoder_layer_idx=2, trace_format="safetensors")
    # input at layer 2 == output at layer 1 (via symlink)
    assert torch.equal(trace["input"], written[1]["output"])
    assert torch.equal(trace["output"], written[2]["output"])


def test_load_safetensors_auto_resolves_when_pt_missing(tmp_path: Path) -> None:
    _make_safetensors_trace(tmp_path, "p")
    trace = _load_reference_trace(tmp_path, "p", decoder_layer_idx=0, trace_format="auto")
    assert trace["input"].shape == (4, D.HIDDEN_SIZE)


def test_load_safetensors_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Safetensors trace directory"):
        _load_reference_trace(tmp_path, "absent", decoder_layer_idx=0, trace_format="safetensors")


def test_load_safetensors_missing_layer_raises(tmp_path: Path) -> None:
    """Flat layout where some layers exist but the requested one doesn't.

    The loader probes flat first then falls through to step_* — when neither
    matches, the error mentions both probed layouts so the caller can tell
    whether the trace is incomplete or the layout is wrong.
    """
    _make_safetensors_trace(tmp_path, "p", layers=(0, 1))
    with pytest.raises(FileNotFoundError, match="decoder_input/output_layer_99"):
        _load_reference_trace(tmp_path, "p", decoder_layer_idx=99, trace_format="safetensors")


# ----- loading: per-step decode-mode layout -----


def _make_per_step_safetensors_trace(
    trace_dir: Path,
    prompt: str,
    *,
    decoder_layer_idx: int,
    prefill_len: int = 4,
    decode_steps: int = 3,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Create a fake bit_sculpt decode-mode trace (step_0 prefill + step_k deltas).

    Returns (inputs_per_step, outputs_per_step) so the test can assert
    concat-along-dim-0 against what the loader returns.
    """
    prompt_dir = trace_dir / prompt
    prompt_dir.mkdir(parents=True, exist_ok=True)
    inputs_per_step: list[torch.Tensor] = []
    outputs_per_step: list[torch.Tensor] = []
    for step in range(decode_steps + 1):  # step_0 is prefill
        step_dir = prompt_dir / f"step_{step}"
        step_dir.mkdir()
        rows = prefill_len if step == 0 else 1
        inp = torch.randn(rows, D.HIDDEN_SIZE, dtype=torch.bfloat16)
        out = torch.randn(rows, D.HIDDEN_SIZE, dtype=torch.bfloat16)
        safetensors_save_file(
            {f"decoder_input_layer_{decoder_layer_idx}": inp},
            str(step_dir / f"decoder_input_layer_{decoder_layer_idx}.safetensors"),
        )
        safetensors_save_file(
            {f"decoder_output_layer_{decoder_layer_idx}": out},
            str(step_dir / f"decoder_output_layer_{decoder_layer_idx}.safetensors"),
        )
        inputs_per_step.append(inp)
        outputs_per_step.append(out)
    return inputs_per_step, outputs_per_step


def test_load_safetensors_per_step_concatenates(tmp_path: Path) -> None:
    inputs, outputs = _make_per_step_safetensors_trace(
        tmp_path, "p", decoder_layer_idx=3, prefill_len=5, decode_steps=4
    )
    trace = _load_reference_trace(tmp_path, "p", decoder_layer_idx=3, trace_format="safetensors")
    # T_total = prefill_len + decode_steps = 5 + 4 = 9
    assert trace["input"].shape == (9, D.HIDDEN_SIZE)
    assert trace["output"].shape == (9, D.HIDDEN_SIZE)
    assert torch.equal(trace["input"], torch.cat(inputs, dim=0))
    assert torch.equal(trace["output"], torch.cat(outputs, dim=0))


def test_load_safetensors_per_step_numeric_sort(tmp_path: Path) -> None:
    """step_10 must come after step_2 — confirms numeric (not lexicographic) sort."""
    inputs, outputs = _make_per_step_safetensors_trace(
        tmp_path, "p", decoder_layer_idx=0, prefill_len=2, decode_steps=15
    )
    trace = _load_reference_trace(tmp_path, "p", decoder_layer_idx=0, trace_format="safetensors")
    # T_total = 2 + 15 = 17
    assert trace["input"].shape == (17, D.HIDDEN_SIZE)
    # Concat order is step_0 (2 rows), step_1..step_15 (1 row each).
    # Row 2 must equal step_1's single row; row 16 must equal step_15's row.
    assert torch.equal(trace["input"][2], inputs[1].squeeze(0))
    assert torch.equal(trace["input"][16], inputs[15].squeeze(0))


def test_load_safetensors_empty_dir_raises(tmp_path: Path) -> None:
    """No flat files AND no step_* subdirs → actionable error."""
    (tmp_path / "p").mkdir()
    with pytest.raises(FileNotFoundError, match="step_"):
        _load_reference_trace(tmp_path, "p", decoder_layer_idx=0, trace_format="safetensors")


# ----- shape / dtype contract -----


def test_load_safetensors_wrong_dtype_raises(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "p"
    prompt_dir.mkdir()
    bad = torch.randn(4, D.HIDDEN_SIZE, dtype=torch.float32)  # not bf16
    safetensors_save_file({"decoder_input_layer_0": bad}, str(prompt_dir / "decoder_input_layer_0.safetensors"))
    safetensors_save_file(
        {"decoder_output_layer_0": bad},
        str(prompt_dir / "decoder_output_layer_0.safetensors"),
    )
    with pytest.raises(ValueError, match="bfloat16"):
        _load_reference_trace(tmp_path, "p", decoder_layer_idx=0, trace_format="safetensors")


def test_load_safetensors_wrong_hidden_dim_raises(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "p"
    prompt_dir.mkdir()
    bad = torch.randn(4, 1234, dtype=torch.bfloat16)  # wrong last dim
    safetensors_save_file({"decoder_input_layer_0": bad}, str(prompt_dir / "decoder_input_layer_0.safetensors"))
    safetensors_save_file(
        {"decoder_output_layer_0": bad},
        str(prompt_dir / "decoder_output_layer_0.safetensors"),
    )
    with pytest.raises(ValueError, match="last dim must equal"):
        _load_reference_trace(tmp_path, "p", decoder_layer_idx=0, trace_format="safetensors")
