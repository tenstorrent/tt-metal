# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LM Head (final norm + vocab projection) parity tests.

Validates ``Mistral4LMHead`` against a torch reference (``Mistral4RMSNorm`` + ``nn.Linear``).
Uses the root ``device`` fixture for single-chip runs.

Synthetic-config tests use tiny shapes. Checkpoint tests load ``norm`` + ``lm_head`` (or tied
``embed_tokens``) from ``models/mistral_small_4/`` when a Hugging Face sharded snapshot is present
(see ``download_model.py``); they are skipped otherwise.

**Activations vs weights (checkpoint tests)**

- **Weights** in ``safetensors`` are parameters only; they do **not** contain hidden states at the LM
  head. True last-layer activations require either (1) a full (or partial) model **forward** on
  some tokens, or (2) a **pre-exported** tensor file you place next to the snapshot.
- **Optional real activations**: if ``lm_head_parity_activations.pt`` exists under the snapshot
  directory (or the path in env ``MISTRAL4_LM_HEAD_ACTIVATIONS_PT``), decode/prefill tests load
  tensors from that file when keys and shapes match; otherwise they keep ``torch.randn`` as today.

The file must be a ``torch.save`` dict with optional keys ``decode`` and/or ``prefill``:

- ``decode``: ``torch.bfloat16`` tensor of shape ``[1, 1, 1, hidden_size]`` (decode batch 1).
- ``prefill``: ``torch.bfloat16`` (or float32) of shape ``[1, 1, 32, hidden_size]``.

You can build this offline (e.g. HF forward with ``output_hidden_states=True`` on a machine that
fits the model, then slice the last hidden state before the final norm — not automated here due to
119B size and multimodal layout).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_small_4_119B.tt.decoder_checkpoint import read_lm_head_checkpoint_tensors
from models.demos.mistral_small_4_119B.tt.lm_head import Mistral4LMHead
from models.demos.mistral_small_4_119B.tt.moe.moe import mistral4_text_config_from_snapshot
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.run_config import create_run_config


def _tiny_mistral4_config():
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=4096,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        rms_norm_eps=1e-6,
    )


def _assert_pcc(tt_output: torch.Tensor, reference_output: torch.Tensor, *, pcc_required: float) -> None:
    tt_out = tt_output.cpu().float()
    ref_out = reference_output.cpu().float()

    while tt_out.ndim < ref_out.ndim:
        tt_out = tt_out.unsqueeze(0)
    while ref_out.ndim < tt_out.ndim:
        ref_out = ref_out.unsqueeze(0)

    # Align batch/seq dims
    seq_or_batch = min(tt_out.shape[-2], ref_out.shape[-2])
    vocab = min(tt_out.shape[-1], ref_out.shape[-1])
    tt_out = tt_out[..., :seq_or_batch, :vocab]
    ref_out = ref_out[..., :seq_or_batch, :vocab]

    passing, pcc = comp_pcc(tt_out, ref_out, pcc_required)
    logger.info(f"lm_head PCC: {pcc}")
    assert passing, f"PCC {pcc} < required {pcc_required}"


def _iter_weight_tensors(weight_config):
    stack = [weight_config]
    while stack:
        node = stack.pop()
        if isinstance(node, ttnn.Tensor):
            yield node
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)


def _mistral4_snapshot_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "mistral_small_4"


_LM_HEAD_PARITY_ACTIVATIONS_FILENAME = "lm_head_parity_activations.pt"
_DECODE_ACT_SPATIAL = (1, 1, 1)
_PREFILL_SEQ_LEN = 32
_PREFILL_ACT_SPATIAL = (1, 1, _PREFILL_SEQ_LEN)


def _torch_load_compat(path: Path):
    """Load activations dict; fall back to full unpickler if ``weights_only`` rejects the blob."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu")


def _coerce_lm_head_activation(
    raw: torch.Tensor | None,
    spatial_shape: tuple[int, ...],
    hidden_size: int,
    field: str,
) -> torch.Tensor | None:
    """Return bf16 tensor [1,1,...,H] or None if missing/invalid."""
    if raw is None:
        return None
    if not isinstance(raw, torch.Tensor):
        logger.warning(f"lm_head parity activations[{field!r}] is not a Tensor (got {type(raw)}); ignoring")
        return None
    t = raw.detach().to(torch.bfloat16).contiguous()
    expected = (*spatial_shape, hidden_size)
    if tuple(t.shape) != expected:
        logger.warning(
            f"lm_head parity activations[{field!r}] has shape {tuple(t.shape)}, expected {expected}; "
            "falling back to random for that test"
        )
        return None
    return t


def _load_optional_checkpoint_lm_head_activations(
    snapshot_dir: Path,
    hidden_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    """Load optional decode/prefill inputs; returns (decode, prefill, provenance string)."""
    env_path = os.environ.get("MISTRAL4_LM_HEAD_ACTIVATIONS_PT", "").strip()
    path = Path(env_path).expanduser() if env_path else snapshot_dir / _LM_HEAD_PARITY_ACTIVATIONS_FILENAME
    if not path.is_file():
        return None, None, "random (no activations file)"

    try:
        blob = _torch_load_compat(path)
    except Exception as exc:
        logger.warning(f"Failed to load LM head activations from {path}: {exc}; using random inputs")
        return None, None, "random (load error)"

    if not isinstance(blob, dict):
        logger.warning(f"LM head activations file must contain a dict, got {type(blob)}; using random inputs")
        return None, None, "random (invalid format)"

    decode_raw = blob.get("decode") if isinstance(blob.get("decode"), torch.Tensor) else blob.get("decode_input")
    prefill_raw = blob.get("prefill") if isinstance(blob.get("prefill"), torch.Tensor) else blob.get("prefill_input")

    dec = _coerce_lm_head_activation(decode_raw, _DECODE_ACT_SPATIAL, hidden_size, "decode")
    pre = _coerce_lm_head_activation(prefill_raw, _PREFILL_ACT_SPATIAL, hidden_size, "prefill")
    if dec is None and pre is None:
        return None, None, f"random (no valid tensors in {path.name})"

    parts: list[str] = []
    if dec is not None:
        parts.append("decode=file")
    if pre is not None:
        parts.append("prefill=file")
    if dec is None or pre is None:
        parts.append("random_fill_missing")
    return dec, pre, f"{'+'.join(parts)} @ {path}"


def _hf_config_and_lm_head_tensors_from_snapshot() -> tuple[object, dict[str, torch.Tensor]]:
    """``Mistral4Config`` + ``norm.weight`` / ``lm_head.weight`` tensors from local snapshot, or skip."""
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")

    snapshot_dir = _mistral4_snapshot_dir()
    if not (snapshot_dir / "config.json").is_file():
        pytest.skip("No config.json under models/mistral_small_4/ (install snapshot per download_model.py)")
    if not (snapshot_dir / "model.safetensors.index.json").is_file():
        pytest.skip("No model.safetensors.index.json (snapshot incomplete)")

    try:
        tensors = read_lm_head_checkpoint_tensors(snapshot_dir)
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        pytest.skip(f"Could not read LM head tensors from snapshot: {exc}")

    hf_config = mistral4_text_config_from_snapshot(snapshot_dir)
    nw, lw = tensors["norm.weight"], tensors["lm_head.weight"]
    assert nw.shape == (hf_config.hidden_size,), (nw.shape, hf_config.hidden_size)
    assert lw.shape == (hf_config.vocab_size, hf_config.hidden_size), (
        lw.shape,
        hf_config.vocab_size,
        hf_config.hidden_size,
    )
    return hf_config, tensors


def test_mistral4_lm_head_decode_matches_torch(device, tmp_path):
    """``Mistral4LMHead.forward_decode`` vs torch RMSNorm + Linear (decode batch=1)."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm

    hf_config = _tiny_mistral4_config()
    hidden_size = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    batch_size = 1

    # Reference modules
    torch.manual_seed(42)
    reference_norm = Mistral4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps).eval().to(torch.bfloat16)
    reference_linear = torch.nn.Linear(hidden_size, vocab_size, bias=False).eval().to(torch.bfloat16)

    # Build state dict matching Mistral4LMHead.convert_weights expectations
    state_dict = {
        "norm.weight": reference_norm.weight.detach().clone(),
        "lm_head.weight": reference_linear.weight.detach().clone(),
    }

    # Reference forward
    torch_input = torch.randn(1, 1, batch_size, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        normed = reference_norm(torch_input)
        reference_output = reference_linear(normed)

    # TT path
    ccl = CCL(device)
    weight_config = Mistral4LMHead.convert_weights(hf_config, (state_dict,), tmp_path / "lm_head", device)
    model_config = Mistral4LMHead.decode_model_config(hf_config, device, batch_size_per_row=batch_size)
    model_state = Mistral4LMHead.create_state(hf_config, device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state, {})

    tt_input = None
    tt_output = None
    try:
        tt_input = ttnn.from_torch(
            torch_input,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        tt_output = Mistral4LMHead.forward_decode(tt_input, run_config)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=tuple(device.shape)),
        )
        tt_output_torch = tt_output_torch[:1]

        logger.info(f"TT output shape: {tt_output_torch.shape}, Reference shape: {reference_output.shape}")
        _assert_pcc(tt_output_torch, reference_output, pcc_required=0.97)
    finally:
        if tt_input is not None:
            ttnn.deallocate(tt_input)
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        for tensor in _iter_weight_tensors(weight_config):
            ttnn.deallocate(tensor)


def test_mistral4_lm_head_prefill_matches_torch(device, tmp_path):
    """``Mistral4LMHead.forward_prefill`` vs torch RMSNorm + Linear (prefill seq_len=32)."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm

    hf_config = _tiny_mistral4_config()
    hidden_size = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    seq_len = 32

    # Reference modules
    torch.manual_seed(42)
    reference_norm = Mistral4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps).eval().to(torch.bfloat16)
    reference_linear = torch.nn.Linear(hidden_size, vocab_size, bias=False).eval().to(torch.bfloat16)

    state_dict = {
        "norm.weight": reference_norm.weight.detach().clone(),
        "lm_head.weight": reference_linear.weight.detach().clone(),
    }

    # Reference forward
    torch_input = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        normed = reference_norm(torch_input)
        reference_output = reference_linear(normed)

    # TT path
    ccl = CCL(device)
    weight_config = Mistral4LMHead.convert_weights(hf_config, (state_dict,), tmp_path / "lm_head_prefill", device)
    model_config = Mistral4LMHead.prefill_model_config(hf_config, device)
    model_state = Mistral4LMHead.create_state(hf_config, device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state, {})

    tt_input = None
    tt_output = None
    try:
        tt_input = ttnn.from_torch(
            torch_input,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        tt_output = Mistral4LMHead.forward_prefill(tt_input, run_config)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=tuple(device.shape)),
        )
        tt_output_torch = tt_output_torch[:1]

        logger.info(f"TT output shape: {tt_output_torch.shape}, Reference shape: {reference_output.shape}")
        _assert_pcc(tt_output_torch, reference_output, pcc_required=0.97)
    finally:
        if tt_input is not None:
            ttnn.deallocate(tt_input)
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        for tensor in _iter_weight_tensors(weight_config):
            ttnn.deallocate(tensor)


def test_mistral4_lm_head_decode_checkpoint_matches_torch(device, tmp_path):
    """``forward_decode`` with snapshot LM-head weights vs torch (batch=1).

    Activations: optional ``lm_head_parity_activations.pt`` / ``MISTRAL4_LM_HEAD_ACTIVATIONS_PT``;
    else ``torch.randn`` with ``manual_seed(0)``.
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm

    hf_config, tensors = _hf_config_and_lm_head_tensors_from_snapshot()
    hidden_size = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    batch_size = 1
    snapshot_dir = _mistral4_snapshot_dir()

    state_dict = {
        "norm.weight": tensors["norm.weight"].to(torch.bfloat16).contiguous(),
        "lm_head.weight": tensors["lm_head.weight"].to(torch.bfloat16).contiguous(),
    }

    reference_norm = Mistral4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps).eval().to(torch.bfloat16)
    reference_norm.weight.data.copy_(state_dict["norm.weight"])
    reference_linear = torch.nn.Linear(hidden_size, vocab_size, bias=False).eval().to(torch.bfloat16)
    reference_linear.weight.data.copy_(state_dict["lm_head.weight"])

    opt_dec, _opt_pre, act_src = _load_optional_checkpoint_lm_head_activations(snapshot_dir, hidden_size)
    if opt_dec is not None:
        torch_input = opt_dec.clone()
        logger.info(f"[checkpoint decode] activations from file; meta={act_src}")
    else:
        torch.manual_seed(0)
        torch_input = torch.randn(1, 1, batch_size, hidden_size, dtype=torch.bfloat16)
        logger.info(f"[checkpoint decode] activations random; meta={act_src}")
    with torch.no_grad():
        normed = reference_norm(torch_input)
        reference_output = reference_linear(normed)

    ccl = CCL(device)
    weight_config = Mistral4LMHead.convert_weights(hf_config, (state_dict,), tmp_path / "lm_head_ckpt_decode", device)
    model_config = Mistral4LMHead.decode_model_config(hf_config, device, batch_size_per_row=batch_size)
    model_state = Mistral4LMHead.create_state(hf_config, device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state, {})

    tt_input = None
    tt_output = None
    try:
        tt_input = ttnn.from_torch(
            torch_input,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        tt_output = Mistral4LMHead.forward_decode(tt_input, run_config)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=tuple(device.shape)),
        )
        tt_output_torch = tt_output_torch[:1]

        logger.info(f"[checkpoint decode] TT shape {tt_output_torch.shape}, ref {reference_output.shape}")
        _assert_pcc(tt_output_torch, reference_output, pcc_required=0.97)
    finally:
        if tt_input is not None:
            ttnn.deallocate(tt_input)
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        for tensor in _iter_weight_tensors(weight_config):
            ttnn.deallocate(tensor)


def test_mistral4_lm_head_prefill_checkpoint_matches_torch(device, tmp_path):
    """``forward_prefill`` with snapshot LM-head weights vs torch (seq_len=32).

    Activations: optional ``lm_head_parity_activations.pt`` / ``MISTRAL4_LM_HEAD_ACTIVATIONS_PT``;
    else ``torch.randn`` with ``manual_seed(1)``.
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm

    hf_config, tensors = _hf_config_and_lm_head_tensors_from_snapshot()
    hidden_size = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    seq_len = _PREFILL_SEQ_LEN
    snapshot_dir = _mistral4_snapshot_dir()

    state_dict = {
        "norm.weight": tensors["norm.weight"].to(torch.bfloat16).contiguous(),
        "lm_head.weight": tensors["lm_head.weight"].to(torch.bfloat16).contiguous(),
    }

    reference_norm = Mistral4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps).eval().to(torch.bfloat16)
    reference_norm.weight.data.copy_(state_dict["norm.weight"])
    reference_linear = torch.nn.Linear(hidden_size, vocab_size, bias=False).eval().to(torch.bfloat16)
    reference_linear.weight.data.copy_(state_dict["lm_head.weight"])

    _opt_dec, opt_pre, act_src = _load_optional_checkpoint_lm_head_activations(snapshot_dir, hidden_size)
    if opt_pre is not None:
        torch_input = opt_pre.clone()
        logger.info(f"[checkpoint prefill] activations from file; meta={act_src}")
    else:
        torch.manual_seed(1)
        torch_input = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
        logger.info(f"[checkpoint prefill] activations random; meta={act_src}")
    with torch.no_grad():
        normed = reference_norm(torch_input)
        reference_output = reference_linear(normed)

    ccl = CCL(device)
    weight_config = Mistral4LMHead.convert_weights(hf_config, (state_dict,), tmp_path / "lm_head_ckpt_prefill", device)
    model_config = Mistral4LMHead.prefill_model_config(hf_config, device)
    model_state = Mistral4LMHead.create_state(hf_config, device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state, {})

    tt_input = None
    tt_output = None
    try:
        tt_input = ttnn.from_torch(
            torch_input,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        tt_output = Mistral4LMHead.forward_prefill(tt_input, run_config)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=tuple(device.shape)),
        )
        tt_output_torch = tt_output_torch[:1]

        logger.info(f"[checkpoint prefill] TT shape {tt_output_torch.shape}, ref {reference_output.shape}")
        _assert_pcc(tt_output_torch, reference_output, pcc_required=0.97)
    finally:
        if tt_input is not None:
            ttnn.deallocate(tt_input)
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        for tensor in _iter_weight_tensors(weight_config):
            ttnn.deallocate(tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
