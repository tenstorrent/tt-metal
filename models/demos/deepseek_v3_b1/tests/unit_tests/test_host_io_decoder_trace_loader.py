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
from loguru import logger as loguru_logger
from safetensors.torch import save_file as safetensors_save_file

from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions as D
from models.demos.deepseek_v3_b1.tests.unit_tests.host_io_decoder_harness import (
    NUM_DENSE_LAYERS,
    HostIoDecoderSweepConfig,
    MultiTurnSchedule,
    _check_metadata_consistency,
    _load_reference_trace,
    _resolve_trace_format,
    _write_dumps,
)


@pytest.fixture
def loguru_warnings() -> list[str]:
    """Collect ``logger.warning(...)`` messages from the harness's loguru logger.

    pytest's stock ``caplog`` only intercepts the stdlib ``logging`` module, but
    the harness uses ``loguru``. The standard recipe is to attach a custom sink
    for the test, append messages to a list, then remove the sink at teardown.
    """
    messages: list[str] = []
    sink_id = loguru_logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        yield messages
    finally:
        loguru_logger.remove(sink_id)


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


# ----- metadata.json consistency -----


def _write_metadata(prompt_dir: Path, **overrides) -> Path:
    """Write a minimal bit_sculpt-style metadata.json with optional overrides."""
    prompt_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "n_layers": 61,
        "hidden_dim": D.HIDDEN_SIZE,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "moe_layer_offset": NUM_DENSE_LAYERS,
    }
    base.update(overrides)
    path = prompt_dir / "metadata.json"
    path.write_text(json.dumps(base))
    return path


def test_metadata_consistency_matching_offset_silent(tmp_path: Path, loguru_warnings) -> None:
    """DeepSeek trace (moe_layer_offset=3) against DeepSeek harness — no warning."""
    path = _write_metadata(tmp_path, moe_layer_offset=NUM_DENSE_LAYERS)
    _check_metadata_consistency(path, decoder_layer_idx=4)
    assert not any("moe_layer_offset" in m for m in loguru_warnings)


def test_metadata_consistency_kimi_layer_in_danger_zone(tmp_path: Path, loguru_warnings) -> None:
    """Kimi trace (offset=1) loaded against DeepSeek harness (=3) — layer 1 is the bug.

    The harness will route layer_idx=1 to the *dense* path (1 < NUM_DENSE_LAYERS=3),
    but the trace was produced treating layer 1 as MoE (1 >= moe_layer_offset=1).
    The warning must name the conflict and the specific layer.
    """
    path = _write_metadata(tmp_path, moe_layer_offset=1)
    _check_metadata_consistency(path, decoder_layer_idx=1)
    joined = "\n".join(loguru_warnings)
    assert "moe_layer_offset=1" in joined
    assert "ambiguous" in joined
    assert "decoder_layer_idx=1" in joined
    assert "MoE" in joined and "dense" in joined


def test_metadata_consistency_kimi_layer_outside_danger_zone(tmp_path: Path, loguru_warnings) -> None:
    """Kimi trace, layer 30 — both Kimi (offset=1) and DeepSeek (=3) agree it's MoE.

    A warning still fires (the offsets disagree at all), but it explicitly notes
    the chosen layer is *outside* the ambiguous range and dense/MoE dispatch
    agrees for this case.
    """
    path = _write_metadata(tmp_path, moe_layer_offset=1)
    _check_metadata_consistency(path, decoder_layer_idx=30)
    joined = "\n".join(loguru_warnings)
    assert "outside the ambiguous range" in joined


def test_metadata_consistency_hidden_dim_mismatch_warns(tmp_path: Path, loguru_warnings) -> None:
    path = _write_metadata(tmp_path, hidden_dim=9999)
    _check_metadata_consistency(path, decoder_layer_idx=4)
    assert any("hidden_dim=9999" in m for m in loguru_warnings)


def test_metadata_consistency_missing_file_silent(tmp_path: Path, loguru_warnings) -> None:
    """Loader must not warn when metadata.json is simply absent."""
    _check_metadata_consistency(tmp_path / "nonexistent_metadata.json", decoder_layer_idx=4)
    assert loguru_warnings == []


# ----- dump path -----
#
# Pure-host tests for _write_dumps: fabricate a sweep result (collected dict +
# fake KV cache + schedule), call _write_dumps, verify each output file exists
# with the right name, tensor key, and shape. Covers both formats and the two
# independent toggles (dump_hidden_states / dump_kv_cache).
#
# KVPE_DIM constant is fixed at 576 in the harness (kv_lora_rank 512 + qk_rope_head_dim 64);
# matches bit_sculpt's kv_post_transform layout.
_KVPE_DIM = 576


def _build_dump_config(
    *,
    dump_dir: Path,
    dump_format: str,
    decoder_layer_idx: int = 4,
    prompt_names: tuple[str, ...] = ("smoke",),
    num_replication_slots: int = 1,
    dump_hidden_states: bool = True,
    dump_kv_cache: bool = True,
) -> HostIoDecoderSweepConfig:
    """Build a sweep config configured for dumping; non-dump knobs are placeholders."""
    return HostIoDecoderSweepConfig(
        decoder_layer_idx=decoder_layer_idx,
        hidden_states_dir=dump_dir,  # not consumed by _write_dumps; placeholder
        prompt_names=prompt_names,
        num_replication_slots=num_replication_slots,
        dump_hidden_states=dump_hidden_states,
        dump_kv_cache=dump_kv_cache,
        dump_dir=dump_dir,
        dump_format=dump_format,
    )


def _build_fake_collected(prompt_lengths: dict[str, int], num_slots: int) -> dict[str, dict[int, torch.Tensor]]:
    return {
        prompt: {slot: torch.randn(L, D.HIDDEN_SIZE, dtype=torch.bfloat16) for slot in range(num_slots)}
        for prompt, L in prompt_lengths.items()
    }


def _build_fake_kv_cache(total_length: int, num_slots_total: int, max_seq_len: int = 64) -> torch.Tensor:
    """Mimic the harness's get_kv_cache_host shape: (num_slots, 1, max_seq_len, kvpe_dim)."""
    assert total_length <= max_seq_len
    return torch.randn(num_slots_total, 1, max_seq_len, _KVPE_DIM, dtype=torch.bfloat16)


def test_write_dumps_pt_format_back_compat(tmp_path: Path) -> None:
    """Default dump_format='pt': filenames + tensor shapes match the pre-PR contract."""
    config = _build_dump_config(dump_dir=tmp_path, dump_format="pt")
    collected = _build_fake_collected({"smoke": 4}, num_slots=1)
    kv_cache = _build_fake_kv_cache(total_length=4, num_slots_total=2)
    schedule = MultiTurnSchedule(prompt_lengths=(4,))

    _write_dumps(config, collected=collected, kv_cache_torch=kv_cache, schedule=schedule)

    # Hidden state dump: torch.save dict with raw tensor at canonical name.
    hs_path = tmp_path / "output_hidden_states_slot_00_smoke.pt"
    assert hs_path.exists()
    loaded_hs = torch.load(hs_path, map_location="cpu")
    assert torch.equal(loaded_hs, collected["smoke"][0])

    # KV dump retains the leading (1,) under .pt format.
    kv_path = tmp_path / "kv_cache_slot_00_smoke.pt"
    assert kv_path.exists()
    loaded_kv = torch.load(kv_path, map_location="cpu")
    assert loaded_kv.shape == (1, 4, _KVPE_DIM)
    assert torch.equal(loaded_kv, kv_cache[0, :, 0:4, :])


def test_write_dumps_safetensors_format_matches_bit_sculpt(tmp_path: Path) -> None:
    """dump_format='safetensors': per-prompt dir, bit_sculpt-canonical tensor keys, (T, 576) KV."""
    from safetensors.torch import load_file as safetensors_load_file

    config = _build_dump_config(dump_dir=tmp_path, dump_format="safetensors", decoder_layer_idx=4)
    collected = _build_fake_collected({"smoke": 4}, num_slots=1)
    kv_cache = _build_fake_kv_cache(total_length=4, num_slots_total=2)
    schedule = MultiTurnSchedule(prompt_lengths=(4,))

    _write_dumps(config, collected=collected, kv_cache_torch=kv_cache, schedule=schedule)

    hs_path = tmp_path / "smoke" / "decoder_output_layer_4_slot_00.safetensors"
    assert hs_path.exists()
    hs = safetensors_load_file(str(hs_path))
    assert set(hs.keys()) == {"decoder_output_layer_4"}
    assert torch.equal(hs["decoder_output_layer_4"], collected["smoke"][0])

    kv_path = tmp_path / "smoke" / "kv_cache_layer_4_slot_00.safetensors"
    assert kv_path.exists()
    kv = safetensors_load_file(str(kv_path))
    assert set(kv.keys()) == {"kv_post_transform_layer_4"}
    # Critical: (1, L_p, 576) -> (L_p, 576). The leading dim must be squeezed.
    assert kv["kv_post_transform_layer_4"].shape == (4, _KVPE_DIM)
    assert torch.equal(kv["kv_post_transform_layer_4"], kv_cache[0, 0, 0:4, :])


def test_write_dumps_mode_b_writes_per_slot_files(tmp_path: Path) -> None:
    """num_replication_slots=4: produces 4 hidden-state + 4 KV files per prompt."""
    config = _build_dump_config(dump_dir=tmp_path, dump_format="safetensors", num_replication_slots=4)
    collected = _build_fake_collected({"smoke": 4}, num_slots=4)
    kv_cache = _build_fake_kv_cache(total_length=4, num_slots_total=8)
    schedule = MultiTurnSchedule(prompt_lengths=(4,))

    _write_dumps(config, collected=collected, kv_cache_torch=kv_cache, schedule=schedule)

    prompt_dir = tmp_path / "smoke"
    hs_files = sorted(prompt_dir.glob("decoder_output_layer_4_slot_*.safetensors"))
    kv_files = sorted(prompt_dir.glob("kv_cache_layer_4_slot_*.safetensors"))
    assert [f.name for f in hs_files] == [f"decoder_output_layer_4_slot_{i:02d}.safetensors" for i in range(4)]
    assert [f.name for f in kv_files] == [f"kv_cache_layer_4_slot_{i:02d}.safetensors" for i in range(4)]


def test_write_dumps_multi_prompt_slices_kv_correctly(tmp_path: Path) -> None:
    """Two prompts at disjoint position ranges: each KV dump is the prompt's slice only."""
    from safetensors.torch import load_file as safetensors_load_file

    config = _build_dump_config(dump_dir=tmp_path, dump_format="safetensors", prompt_names=("a", "b"))
    collected = _build_fake_collected({"a": 3, "b": 5}, num_slots=1)
    kv_cache = _build_fake_kv_cache(total_length=8, num_slots_total=2)
    schedule = MultiTurnSchedule(prompt_lengths=(3, 5))

    _write_dumps(config, collected=collected, kv_cache_torch=kv_cache, schedule=schedule)

    kv_a = safetensors_load_file(str(tmp_path / "a" / "kv_cache_layer_4_slot_00.safetensors"))
    kv_b = safetensors_load_file(str(tmp_path / "b" / "kv_cache_layer_4_slot_00.safetensors"))
    # prompt "a" occupies positions [0, 3); prompt "b" occupies [3, 8).
    assert kv_a["kv_post_transform_layer_4"].shape == (3, _KVPE_DIM)
    assert kv_b["kv_post_transform_layer_4"].shape == (5, _KVPE_DIM)
    assert torch.equal(kv_a["kv_post_transform_layer_4"], kv_cache[0, 0, 0:3, :])
    assert torch.equal(kv_b["kv_post_transform_layer_4"], kv_cache[0, 0, 3:8, :])


def test_write_dumps_only_hidden_states(tmp_path: Path) -> None:
    """dump_hidden_states=True, dump_kv_cache=False: no KV files written, kv_cache_torch=None OK."""
    config = _build_dump_config(
        dump_dir=tmp_path,
        dump_format="safetensors",
        dump_hidden_states=True,
        dump_kv_cache=False,
    )
    collected = _build_fake_collected({"smoke": 4}, num_slots=1)
    schedule = MultiTurnSchedule(prompt_lengths=(4,))

    _write_dumps(config, collected=collected, kv_cache_torch=None, schedule=schedule)

    assert (tmp_path / "smoke" / "decoder_output_layer_4_slot_00.safetensors").exists()
    assert not list((tmp_path / "smoke").glob("kv_cache_layer_*"))


def test_write_dumps_only_kv_cache(tmp_path: Path) -> None:
    """dump_kv_cache=True, dump_hidden_states=False: no hidden-state files written."""
    config = _build_dump_config(
        dump_dir=tmp_path,
        dump_format="safetensors",
        dump_hidden_states=False,
        dump_kv_cache=True,
    )
    collected = _build_fake_collected({"smoke": 4}, num_slots=1)
    kv_cache = _build_fake_kv_cache(total_length=4, num_slots_total=2)
    schedule = MultiTurnSchedule(prompt_lengths=(4,))

    _write_dumps(config, collected=collected, kv_cache_torch=kv_cache, schedule=schedule)

    assert (tmp_path / "smoke" / "kv_cache_layer_4_slot_00.safetensors").exists()
    assert not list((tmp_path / "smoke").glob("decoder_output_layer_*"))


def test_write_dumps_both_off_is_noop(tmp_path: Path) -> None:
    """Both knobs off: _write_dumps writes nothing and tolerates kv_cache_torch=None."""
    # Bypass __post_init__ validation by setting dump_dir=None (allowed when both knobs off).
    config = HostIoDecoderSweepConfig(
        decoder_layer_idx=4,
        hidden_states_dir=tmp_path,
        prompt_names=("smoke",),
        dump_hidden_states=False,
        dump_kv_cache=False,
        dump_dir=None,
    )
    schedule = MultiTurnSchedule(prompt_lengths=(4,))
    _write_dumps(config, collected={}, kv_cache_torch=None, schedule=schedule)
    assert list(tmp_path.iterdir()) == []


# ----- weight_key_prefix wiring -----
#
# Smoke tests that confirm CacheWeightProvider's weight_key_prefix kwarg
# threads correctly into LazyStateDict.base_prefix and the right HF keys are
# in/out of the resulting Mapping view. Doesn't load any real model weights —
# just checks the index resolution layer, which is what Kimi needs.


def _make_fake_hf_model_dir(tmp_path: Path, weight_map: dict[str, str]) -> Path:
    """Create a minimal HF snapshot dir: model.safetensors.index.json + sharded names.

    Doesn't write actual safetensors shards (LazyStateDict reads the index
    eagerly but tensor files only on __getitem__ access). Sufficient for
    testing the prefix-resolution path.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    return model_dir


def test_cache_weight_provider_default_prefix_is_empty(tmp_path: Path) -> None:
    """Default behavior unchanged: weight_key_prefix='' means HF keys used as-is."""
    from models.demos.deepseek_v3_b1.demo.weight_provider import CacheWeightProvider

    model_dir = _make_fake_hf_model_dir(tmp_path, {"model.layers.0.input_layernorm.weight": "dummy.safetensors"})
    provider = CacheWeightProvider(cache_path=tmp_path / "cache", model_path=model_dir)
    assert provider._weight_key_prefix == ""
    assert provider._state_dict._base_prefix == ""
    assert "model.layers.0.input_layernorm.weight" in provider._state_dict


def test_cache_weight_provider_kimi_prefix_resolves_keys(tmp_path: Path) -> None:
    """With weight_key_prefix='language_model.', Kimi-style keys are resolvable.

    Kimi's HF state dict has keys like "language_model.model.layers.0.…" because
    the DeepSeek-V3 backbone is wrapped under that prefix by the multimodal
    KimiK25ForConditionalGeneration architecture. With the prefix set, code
    that looks up "model.layers.0.…" against the provider's state dict resolves
    to the right physical key.
    """
    from models.demos.deepseek_v3_b1.demo.weight_provider import CacheWeightProvider

    model_dir = _make_fake_hf_model_dir(
        tmp_path,
        {
            "language_model.model.layers.0.input_layernorm.weight": "dummy.safetensors",
            "language_model.model.layers.5.self_attn.q_a_proj.weight": "dummy.safetensors",
        },
    )
    provider = CacheWeightProvider(
        cache_path=tmp_path / "cache",
        model_path=model_dir,
        weight_key_prefix="language_model.",
    )
    # Provider records the prefix and forwards it to LazyStateDict.
    assert provider._weight_key_prefix == "language_model."
    assert provider._state_dict._base_prefix == "language_model."
    # Caller-facing keys are stripped of the prefix.
    assert "model.layers.0.input_layernorm.weight" in provider._state_dict
    assert "model.layers.5.self_attn.q_a_proj.weight" in provider._state_dict
    # Caller-facing key collision with the *full* key must miss — the prefix
    # is prepended *to* lookups, not stripped from them.
    assert "language_model.model.layers.0.input_layernorm.weight" not in provider._state_dict


def test_cache_weight_provider_prefix_mismatch_fails_lookup(tmp_path: Path) -> None:
    """Wrong prefix → physical key never resolves → contains check is False."""
    from models.demos.deepseek_v3_b1.demo.weight_provider import CacheWeightProvider

    model_dir = _make_fake_hf_model_dir(
        tmp_path,
        # Stored under "language_model." but provider is configured with no prefix.
        {"language_model.model.layers.0.input_layernorm.weight": "dummy.safetensors"},
    )
    provider = CacheWeightProvider(cache_path=tmp_path / "cache", model_path=model_dir)
    assert "model.layers.0.input_layernorm.weight" not in provider._state_dict
