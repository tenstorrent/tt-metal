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
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.tt_transformers.tt.hf_rope_decode_capture import (
    list_capture_step_dirs,
    load_manifest,
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


def _digest_subset(d: dict) -> dict:
    """Stable fields for post-load sanity (full string repr can vary slightly)."""
    keys = ("memory_layout", "buffer_type", "shard_shape", "shard_orientation", "grid_repr")
    return {k: d.get(k) for k in keys}


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
    # pcc_floor = float(os.environ.get("TT_HF_ROPE_DECODE_REPLAY_PCC", "0.95"))
    compute_cfg = _rope_hf_hifi4_compute_kernel_config()

    for step_dir in step_dirs:
        manifest = load_manifest(step_dir)
        tf = manifest["tensor_files"]
        q = ttnn.load_tensor(step_dir / tf["q_pre_rot"], device=mesh_device)
        k = ttnn.load_tensor(step_dir / tf["k_pre_rot"], device=mesh_device)
        cos_tt = ttnn.load_tensor(step_dir / tf["cos"], device=mesh_device)
        sin_tt = ttnn.load_tensor(step_dir / tf["sin"], device=mesh_device)

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
                pytest.fail(
                    f"{step_dir.name} {name} memory_config digest mismatch after load_tensor:\n"
                    f"expected={_digest_subset(expected)}\nactual={_digest_subset(got)}"
                )

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
            pytest.fail(
                f"{step_dir.name}: missing position_indices.pt (set capture from test_attention with capture env)."
            )

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
        logger.info(
            f"[hf_rope_capture_replay] step={step_dir.name} Q vs torch PCC={pcc_q} floor={pcc_floor} pass={ok_q}"
        )
        logger.info(
            f"[hf_rope_capture_replay] step={step_dir.name} K vs torch PCC={pcc_k} floor={pcc_floor} pass={ok_k}"
        )
        assert ok_q, f"{step_dir.name} Q rotary_embedding_hf vs torch PCC={pcc_q} (need >= {pcc_floor})"
        assert ok_k, f"{step_dir.name} K rotary_embedding_hf vs torch PCC={pcc_k} (need >= {pcc_floor})"

        ttnn.deallocate(q_out)
        ttnn.deallocate(k_out)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)


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

    for step_dir in step_dirs:
        manifest = load_manifest(step_dir)
        tf = manifest["tensor_files"]
        q = ttnn.load_tensor(step_dir / tf["q_pre_rot"], device=mesh_device)
        k = ttnn.load_tensor(step_dir / tf["k_pre_rot"], device=mesh_device)
        cos_tt = ttnn.load_tensor(step_dir / tf["cos"], device=mesh_device)
        sin_tt = ttnn.load_tensor(step_dir / tf["sin"], device=mesh_device)

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
                pytest.fail(
                    f"{step_dir.name} {name} memory_config digest mismatch after load_tensor:\n"
                    f"expected={_digest_subset(expected)}\nactual={_digest_subset(got)}"
                )

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
        logger.info(
            f"[hf_rope_capture_replay_cos_golden] step={step_dir.name} Q PCC={pcc_q} floor={pcc_floor} pass={ok_q}"
        )
        logger.info(
            f"[hf_rope_capture_replay_cos_golden] step={step_dir.name} K PCC={pcc_k} floor={pcc_floor} pass={ok_k}"
        )
        assert (
            ok_q
        ), f"{step_dir.name} Q rotary_embedding_hf vs torch(captured cos/sin) PCC={pcc_q} (need >= {pcc_floor})"
        assert (
            ok_k
        ), f"{step_dir.name} K rotary_embedding_hf vs torch(captured cos/sin) PCC={pcc_k} (need >= {pcc_floor})"

        ttnn.deallocate(q_out)
        ttnn.deallocate(k_out)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)
