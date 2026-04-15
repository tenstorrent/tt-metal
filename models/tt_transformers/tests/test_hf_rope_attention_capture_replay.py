# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Replay ``rotary_embedding_hf`` decode inputs captured from ``test_attention`` / ``Attention._hf_rope_new_decode``.

Capture: run ``pytest models/tt_transformers/tests/test_attention.py -k hf_rope`` with
``TT_HF_ROPE_DECODE_CAPTURE_DIR=/path/to/run`` (and ``use_hf_rope_new`` path).

Replay: ``TT_HF_ROPE_DECODE_CAPTURE_REPLAY_DIR=/path/to/run pytest ... this file``.

Requires the same mesh topology class as capture (see manifest ``cluster_shape``).

``test_hf_rope_attention_capture_replay_vs_captured_cos_sin_golden`` checks the op against
torch HF RoPE using the captured ``cos``/``sin`` tensors only (no ``position_indices.pt``).

Sweep parity: set ``TT_HF_ROPE_SWEEP_PARITY_LOG=1`` to log the live ``compute_kernel_config``
object and, per step after ``ttnn.load_tensor``, each tensor's shape (including logical /
padded when available), ``dtype``, ``layout``, and ``memory_config`` digest — suitable for
aligning ``rotary_embedding_hf`` sweep vectors without hardcoded shape tables
(see sweep suites ``hf_rope_capture_run3_{prefill,decode}`` and ``nightly_{prefill,decode}`` in
``tests/sweep_framework/sweeps/transformer/rotary_embedding_hf/rotary_embedding_hf.py``).
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.tt_transformers.tt.hf_rope_decode_capture import (
    list_capture_step_dirs,
    load_manifest,
    tensor_device_layout_digest,
    tensor_memory_config_shard_digest,
    torch_golden_hf_rope_decode_1b32d,
    torch_golden_hf_rope_decode_from_cos_sin,
)


def _rope_hf_hifi4_compute_kernel_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _sweep_parity_log_enabled() -> bool:
    return os.environ.get("TT_HF_ROPE_SWEEP_PARITY_LOG", "0") in ("1", "true", "True")


def _compute_kernel_config_digest(cfg) -> dict:
    """Serialize the same object passed to ``rotary_embedding_hf`` (no hardcoded HiFi4 table)."""
    return {
        "type": type(cfg).__name__,
        "math_fidelity": str(getattr(cfg, "math_fidelity", None)),
        "math_approx_mode": getattr(cfg, "math_approx_mode", None),
        "fp32_dest_acc_en": getattr(cfg, "fp32_dest_acc_en", None),
        "packer_l1_acc": getattr(cfg, "packer_l1_acc", None),
    }


def _sweep_decode_input_shape_from_tensor(tt: ttnn.Tensor) -> list[int]:
    """``rotary_embedding_hf`` decode sweep uses ``input_shape`` = logical shape when exposed."""
    fn = getattr(tt, "logical_shape", None)
    if callable(fn):
        try:
            return list(fn())
        except Exception:
            pass
    return list(tt.shape)


def _log_sweep_parity_compute_banner(compute_cfg) -> None:
    if not _sweep_parity_log_enabled():
        return
    block = json.dumps({"compute_kernel_config": _compute_kernel_config_digest(compute_cfg)}, indent=2)
    msg = f"[hf_rope_sweep_parity] Live compute_kernel_config (passed to rotary_embedding_hf):\n{block}"
    print(msg)
    logger.info(msg)


def _log_sweep_parity_from_loaded_tensors(
    step_dir: Path,
    manifest: dict,
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    cos_tt: ttnn.Tensor,
    sin_tt: ttnn.Tensor,
) -> None:
    """Log shapes, dtype, layout, memory_config from device tensors after load (not manifest tables)."""
    if not _sweep_parity_log_enabled():
        return
    payload = {
        "step_dir": step_dir.name,
        "model_name": manifest.get("model_name"),
        "Q": tensor_device_layout_digest(q),
        "K": tensor_device_layout_digest(k),
        "cos": tensor_device_layout_digest(cos_tt),
        "sin": tensor_device_layout_digest(sin_tt),
        "suggested_sweep_decode_spec_q": {
            "input_shape": _sweep_decode_input_shape_from_tensor(q),
            "is_decode": True,
        },
        "suggested_sweep_decode_spec_k": {
            "input_shape": _sweep_decode_input_shape_from_tensor(k),
            "is_decode": True,
        },
    }
    shape_q = _sweep_decode_input_shape_from_tensor(q)
    max_sl = manifest.get("max_seq_len")
    if len(shape_q) == 4 and max_sl is not None:
        _, _, n_heads, head_d = shape_q
        payload["suggested_sweep_prefill_spec_if_same_heads_seq"] = {
            "input_shape": [1, n_heads, int(max_sl), head_d],
            "cache_size": int(max_sl),
            "note": "decode capture only; seq/cache_size from manifest max_seq_len, head_dim/heads from loaded Q",
        }
    line = "[hf_rope_sweep_parity] loaded tensors → sweep hints: " + json.dumps(payload, default=str)
    print(line)
    logger.info(line)


def _digest_subset(d: dict) -> dict:
    """Stable fields for post-load sanity (full string repr can vary slightly)."""
    keys = ("memory_layout", "buffer_type", "shard_shape", "shard_orientation", "grid_repr")
    return {k: d.get(k) for k in keys}


def _print_pcc_stats_table(log_prefix: str, series: dict[str, list[float]], row_order: tuple[str, ...]) -> None:
    """Print and log a fixed-width id / mean / min / max table (one row per metric)."""
    lines: list[str] = [f"{'id':<16} {'mean':>12} {'min':>12} {'max':>12}"]
    for name in row_order:
        vals = series.get(name, [])
        if not vals:
            lines.append(f"{name:<16} {'(no data)':>12} {'':>12} {'':>12}")
            continue
        lines.append(f"{name:<16} {statistics.mean(vals):>12.6f} {min(vals):>12.6f} {max(vals):>12.6f}")
    block = "\n".join(lines)
    print(block)
    logger.info(f"{log_prefix} PCC summary (completed steps):\n{block}")


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_hf_rope_attention_capture_replay(mesh_device, reset_seeds):
    """Load each ``step_*`` / ``manifest.json`` / ``*.tensorbin`` and compare RoPE to torch golden."""
    if is_blackhole():
        pytest.skip("Capture/replay uses Wormhole HiFi4 kernel config (match Attention on WH).")

    root = os.environ.get("TT_HF_ROPE_DECODE_CAPTURE_REPLAY_DIR")
    if not root:
        pytest.skip("Set TT_HF_ROPE_DECODE_CAPTURE_REPLAY_DIR to the directory containing step_* capture folders.")

    replay_root = Path(root)
    step_dirs = list_capture_step_dirs(replay_root)
    if not step_dirs:
        pytest.skip(f"No step_* directories under {replay_root}")

    pcc_floor = float(os.environ.get("TT_HF_ROPE_DECODE_REPLAY_PCC", "0.95"))
    compute_cfg = _rope_hf_hifi4_compute_kernel_config()
    _log_sweep_parity_compute_banner(compute_cfg)

    pcc_series = {"layer_pcc": [], "k_cache_pcc": []}
    pcc_assert_msgs: list[str] = []
    other_failures: list[str] = []

    for step_dir in step_dirs:
        manifest = load_manifest(step_dir)
        tf = manifest["tensor_files"]
        q = ttnn.load_tensor(step_dir / tf["q_pre_rot"], device=mesh_device)
        k = ttnn.load_tensor(step_dir / tf["k_pre_rot"], device=mesh_device)
        cos_tt = ttnn.load_tensor(step_dir / tf["cos"], device=mesh_device)
        sin_tt = ttnn.load_tensor(step_dir / tf["sin"], device=mesh_device)
        _log_sweep_parity_from_loaded_tensors(step_dir, manifest, q, k, cos_tt, sin_tt)

        digest_mismatch = False
        for name, tensor, expected in (
            ("q", q, manifest.get("q_digest")),
            ("k", k, manifest.get("k_digest")),
            ("cos", cos_tt, manifest.get("cos_digest")),
            ("sin", sin_tt, manifest.get("sin_digest")),
        ):
            if not expected:
                continue
            got = tensor_memory_config_shard_digest(tensor)
            if _digest_subset(expected) != _digest_subset(got):
                other_failures.append(
                    f"{step_dir.name} {name} memory_config digest mismatch after load_tensor:\n"
                    f"expected={_digest_subset(expected)}\nactual={_digest_subset(got)}"
                )
                digest_mismatch = True
                break

        if digest_mismatch:
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)
            continue

        ctc = manifest.get("to_torch_composer") or {}
        dims = tuple(ctc.get("dims", (1, 3)))
        mesh_shape = list(ctc.get("mesh_shape", manifest.get("cluster_shape", [1, 1])))
        composer_cfg = ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=mesh_shape)

        q_out = ttnn.experimental.rotary_embedding_hf(
            q, cos_tt, sin_tt, is_decode=True, compute_kernel_config=compute_cfg
        )
        k_out = ttnn.experimental.rotary_embedding_hf(
            k, cos_tt, sin_tt, is_decode=True, compute_kernel_config=compute_cfg
        )

        pos_path = step_dir / "position_indices.pt"
        if not pos_path.is_file():
            ttnn.deallocate(q_out)
            ttnn.deallocate(k_out)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)
            other_failures.append(
                f"{step_dir.name}: missing position_indices.pt (set capture from test_attention with capture env)."
            )
            continue

        pos = torch.load(pos_path, map_location="cpu")

        q_pre = ttnn.to_torch(q, mesh_composer=composer_cfg).to(torch.bfloat16)
        k_pre = ttnn.to_torch(k, mesh_composer=composer_cfg).to(torch.bfloat16)
        q_tt = ttnn.to_torch(q_out, mesh_composer=composer_cfg).float()
        k_tt = ttnn.to_torch(k_out, mesh_composer=composer_cfg).float()

        q_ref, k_ref = torch_golden_hf_rope_decode_1b32d(
            q_pre,
            k_pre,
            pos,
            int(manifest["head_dim"]),
            int(manifest["max_seq_len"]),
            float(manifest["rope_theta"]),
            manifest.get("rope_scaling"),
        )

        q_tt = q_tt[:, :, : q_ref.shape[2], :]
        k_tt = k_tt[:, :, : k_ref.shape[2], :]
        ok_q, pcc_q = comp_pcc(q_ref, q_tt, pcc=pcc_floor)
        ok_k, pcc_k = comp_pcc(k_ref, k_tt, pcc=pcc_floor)
        pcc_series["layer_pcc"].append(pcc_q)
        pcc_series["k_cache_pcc"].append(pcc_k)
        logger.info(
            f"[hf_rope_capture_replay] step={step_dir.name} Q vs torch PCC={pcc_q} floor={pcc_floor} pass={ok_q}"
        )
        logger.info(
            f"[hf_rope_capture_replay] step={step_dir.name} K vs torch PCC={pcc_k} floor={pcc_floor} pass={ok_k}"
        )
        if not ok_q:
            pcc_assert_msgs.append(f"{step_dir.name} Q rotary_embedding_hf vs torch PCC={pcc_q} (need >= {pcc_floor})")
        if not ok_k:
            pcc_assert_msgs.append(f"{step_dir.name} K rotary_embedding_hf vs torch PCC={pcc_k} (need >= {pcc_floor})")

        ttnn.deallocate(q_out)
        ttnn.deallocate(k_out)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

    _print_pcc_stats_table("[hf_rope_capture_replay]", pcc_series, ("layer_pcc", "k_cache_pcc"))
    if other_failures:
        pytest.fail(
            "Non-PCC failures after running all steps:\n"
            + "\n---\n".join(other_failures)
            + ("\n\nPCC failures (also below floor):\n" + "\n".join(pcc_assert_msgs) if pcc_assert_msgs else "")
        )
    if pcc_assert_msgs:
        pytest.fail(
            f"PCC floor={pcc_floor} not met for one or more steps:\n"
            + "\n".join(pcc_assert_msgs)
            + f"\n\nCompleted steps with PCC: layer={len(pcc_series['layer_pcc'])}, k_cache={len(pcc_series['k_cache_pcc'])}"
        )


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_hf_rope_attention_capture_replay_vs_captured_cos_sin_golden(mesh_device, reset_seeds):
    """Replay capture: compare ``rotary_embedding_hf`` to torch HF RoPE using the same dumped cos/sin tensors."""
    if is_blackhole():
        pytest.skip("Capture/replay uses Wormhole HiFi4 kernel config (match Attention on WH).")

    root = os.environ.get("TT_HF_ROPE_DECODE_CAPTURE_REPLAY_DIR")
    if not root:
        pytest.skip("Set TT_HF_ROPE_DECODE_CAPTURE_REPLAY_DIR to the directory containing step_* capture folders.")

    replay_root = Path(root)
    step_dirs = list_capture_step_dirs(replay_root)
    if not step_dirs:
        pytest.skip(f"No step_* directories under {replay_root}")

    pcc_floor = float(os.environ.get("TT_HF_ROPE_DECODE_REPLAY_PCC", "0.95"))
    compute_cfg = _rope_hf_hifi4_compute_kernel_config()
    _log_sweep_parity_compute_banner(compute_cfg)

    pcc_series = {"layer_pcc": [], "k_cache_pcc": []}
    pcc_assert_msgs: list[str] = []
    other_failures: list[str] = []

    for step_dir in step_dirs:
        manifest = load_manifest(step_dir)
        tf = manifest["tensor_files"]
        q = ttnn.load_tensor(step_dir / tf["q_pre_rot"], device=mesh_device)
        k = ttnn.load_tensor(step_dir / tf["k_pre_rot"], device=mesh_device)
        cos_tt = ttnn.load_tensor(step_dir / tf["cos"], device=mesh_device)
        sin_tt = ttnn.load_tensor(step_dir / tf["sin"], device=mesh_device)
        _log_sweep_parity_from_loaded_tensors(step_dir, manifest, q, k, cos_tt, sin_tt)

        digest_mismatch = False
        for name, tensor, expected in (
            ("q", q, manifest.get("q_digest")),
            ("k", k, manifest.get("k_digest")),
            ("cos", cos_tt, manifest.get("cos_digest")),
            ("sin", sin_tt, manifest.get("sin_digest")),
        ):
            if not expected:
                continue
            got = tensor_memory_config_shard_digest(tensor)
            if _digest_subset(expected) != _digest_subset(got):
                other_failures.append(
                    f"{step_dir.name} {name} memory_config digest mismatch after load_tensor:\n"
                    f"expected={_digest_subset(expected)}\nactual={_digest_subset(got)}"
                )
                digest_mismatch = True
                break

        if digest_mismatch:
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)
            continue

        ctc = manifest.get("to_torch_composer") or {}
        dims = tuple(ctc.get("dims", (1, 3)))
        mesh_shape = list(ctc.get("mesh_shape", manifest.get("cluster_shape", [1, 1])))
        composer_cfg = ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=mesh_shape)

        q_out = ttnn.experimental.rotary_embedding_hf(
            q, cos_tt, sin_tt, is_decode=True, compute_kernel_config=compute_cfg
        )
        k_out = ttnn.experimental.rotary_embedding_hf(
            k, cos_tt, sin_tt, is_decode=True, compute_kernel_config=compute_cfg
        )

        q_pre = ttnn.to_torch(q, mesh_composer=composer_cfg).to(torch.bfloat16)
        k_pre = ttnn.to_torch(k, mesh_composer=composer_cfg).to(torch.bfloat16)
        cos_torch = ttnn.to_torch(cos_tt, mesh_composer=composer_cfg)
        sin_torch = ttnn.to_torch(sin_tt, mesh_composer=composer_cfg)
        q_tt = ttnn.to_torch(q_out, mesh_composer=composer_cfg).float()
        k_tt = ttnn.to_torch(k_out, mesh_composer=composer_cfg).float()

        q_ref, k_ref = torch_golden_hf_rope_decode_from_cos_sin(q_pre, k_pre, cos_torch, sin_torch)

        q_tt = q_tt[:, :, : q_ref.shape[2], :]
        k_tt = k_tt[:, :, : k_ref.shape[2], :]
        ok_q, pcc_q = comp_pcc(q_ref, q_tt, pcc=pcc_floor)
        ok_k, pcc_k = comp_pcc(k_ref, k_tt, pcc=pcc_floor)
        pcc_series["layer_pcc"].append(pcc_q)
        pcc_series["k_cache_pcc"].append(pcc_k)
        logger.info(
            f"[hf_rope_capture_replay_cos_golden] step={step_dir.name} Q PCC={pcc_q} floor={pcc_floor} pass={ok_q}"
        )
        logger.info(
            f"[hf_rope_capture_replay_cos_golden] step={step_dir.name} K PCC={pcc_k} floor={pcc_floor} pass={ok_k}"
        )
        if not ok_q:
            pcc_assert_msgs.append(
                f"{step_dir.name} Q rotary_embedding_hf vs torch(captured cos/sin) PCC={pcc_q} (need >= {pcc_floor})"
            )
        if not ok_k:
            pcc_assert_msgs.append(
                f"{step_dir.name} K rotary_embedding_hf vs torch(captured cos/sin) PCC={pcc_k} (need >= {pcc_floor})"
            )

        ttnn.deallocate(q_out)
        ttnn.deallocate(k_out)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

    _print_pcc_stats_table("[hf_rope_capture_replay_cos_golden]", pcc_series, ("layer_pcc", "k_cache_pcc"))
    if other_failures:
        pytest.fail(
            "Non-PCC failures after running all steps:\n"
            + "\n---\n".join(other_failures)
            + ("\n\nPCC failures (also below floor):\n" + "\n".join(pcc_assert_msgs) if pcc_assert_msgs else "")
        )
    if pcc_assert_msgs:
        pytest.fail(
            f"PCC floor={pcc_floor} not met for one or more steps:\n"
            + "\n".join(pcc_assert_msgs)
            + f"\n\nCompleted steps with PCC: layer={len(pcc_series['layer_pcc'])}, k_cache={len(pcc_series['k_cache_pcc'])}"
        )
