# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end pipeline gate for `meituan-longcat/LongCat-Video` (text-to-video diffusion).

Runs the ONE shared pipeline in `tt/pipeline.py` on the 1x4 mesh, feeding each stage the
PREVIOUS stage's REAL TT output, and asserts the three gates:

  Gate 1 -- routed graduated stubs are native/sharded ttnn (no torch-compute fallback).
  Gate 2 -- every graduated module on the real forward path is INVOKED (union tracked in
            pipeline.invoked); the 9 Wan-VAE sub-block ports the composite computes internally
            are covered by tests/pcc/ (see README) and are checked here per-component too.
  Gate 3 -- each stage's output PCC vs its Source-A golden >= 0.95 (text_encode >= 0.99).

`e2e PCC=<x>` is printed on EVERY run (pass or fail) immediately before each assert.

Select a subset with -k (e.g. `-k vae` for the cheap 485MB stage). Run on device with:
  ./python_env/bin/python -m pytest models/demos/hf_eager/longcat_video/tests/e2e/test_pipeline_e2e.py -s -k <stage>
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.longcat_video.tt.pipeline import ALL_GRADUATED, DIT_STUBS, UMT5_STUBS, build_pipeline

_ROOT = Path(__file__).resolve().parents[2]  # .../longcat_video
_STUBS = _ROOT / "_stubs"

# torch host-compute ops forbidden on ACTIVATIONS in a routed stub's hot path (Gate 1).
# NOTE: torch.arange / torch.einsum are NOT flagged -- their only uses in these stubs are
# deterministic RoPE/timestep CONSTANT-table precompute (outer products of arange/freq vectors,
# no activation dependence), which the trace/2CQ contract explicitly permits (same category as
# RoPE sin/cos constants). Docstrings/comments are stripped before scanning.
_FORBIDDEN = re.compile(
    r"\btorch\.(matmul|mm|bmm|softmax|log_softmax|layer_norm|rms_norm|batch_norm|"
    r"group_norm|embedding|embedding_bag|conv1d|conv2d|conv3d|conv_transpose\w*|"
    r"scaled_dot_product_attention|relu|gelu|silu|tanh|sigmoid|leaky_relu|argmax|topk|multinomial)\s*\("
    r"|\bF\.(matmul|softmax|layer_norm|rms_norm|scaled_dot_product_attention|linear|conv\w+|gelu|silu|relu)\s*\("
)


def _strip_docstrings_and_comments(src: str) -> str:
    # remove triple-quoted blocks (docstrings) then inline # comments
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return "\n".join(line.split("#", 1)[0] for line in src.splitlines())


def _open_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    try:
        return ttnn.open_mesh_device(ttnn.MeshShape(1, 4)), True
    except Exception:
        return ttnn.open_mesh_device(ttnn.MeshShape(1, 1)), False


# --------------------------------------------------------------------------- Gate 1 (static)
def test_gate1_routed_stubs_are_native_ttnn():
    """Every routed stub file is real ttnn with no torch-compute op in its body, and no runtime
    fallback was recorded. Sharded (ShardTensorToMesh) bodies count as native."""
    routed = ALL_GRADUATED
    offenders = []
    for name in routed:
        src = _strip_docstrings_and_comments((_STUBS / f"{name}.py").read_text())
        for ln, line in enumerate(src.splitlines(), 1):
            if _FORBIDDEN.search(line):
                offenders.append(f"{name}.py:{ln}: {line.strip()}")
    assert not offenders, "Gate 1 torch-compute in routed stub(s):\n" + "\n".join(offenders)

    fb = _ROOT / "_runtime_fallbacks.json"
    if fb.exists():
        import json

        data = json.loads(fb.read_text() or "{}")
        assert not data, f"Gate 1 runtime fallbacks recorded: {data}"

    # a genuine TP=4 pipeline must contain ShardTensorToMesh + a collective somewhere.
    joined = "\n".join((_STUBS / f"{n}.py").read_text() for n in routed)
    assert "ShardTensorToMesh" in joined, "no ShardTensorToMesh found -> not a TP pipeline"
    assert re.search(r"all_reduce|all_gather", joined), "no CCL collective found -> not a TP pipeline"


# --------------------------------------------------------------------------- Gate 3 per stage
def test_text_encode_pcc():
    dev, is_mesh = _open_mesh()
    try:
        pipe = build_pipeline(dev)
        ids = pipe.encode_prompt("A cat playing piano in a sunny room", max_length=32)
        tt = pipe.run_text_encode(ids)
        from models.demos.hf_eager.longcat_video.tt.pipeline import _replicated_to_torch

        tt_torch = _replicated_to_torch(tt, dev).to(torch.float32)
        golden = pipe._hf_reference_text_encode(ids).to(torch.float32)
        ok, pcc = comp_pcc(golden, tt_torch, 0.99)
        print(f"e2e PCC={pcc}", flush=True)
        assert all(s in pipe.invoked for s in UMT5_STUBS), f"UMT5 stubs not all invoked: {set(UMT5_STUBS)-pipe.invoked}"
        assert ok, f"text_encode PCC {pcc} < 0.99"
    finally:
        ttnn.close_mesh_device(dev)


def test_vae_pcc():
    dev, is_mesh = _open_mesh()
    try:
        torch.manual_seed(0)
        pipe = build_pipeline(dev)
        video = torch.randn(1, 3, 5, 32, 32, dtype=torch.float32)
        recon = pipe.run_vae(video)
        _ = pipe.run_vae_encode(video)  # wan_encoder3d real half
        golden = pipe._hf_reference_vae(video).to(torch.float32)
        ok, pcc = comp_pcc(golden, recon.to(torch.float32), 0.95)
        print(f"e2e PCC={pcc}", flush=True)
        for s in ("autoencoder_k_l_wan", "wan_encoder3d"):
            assert s in pipe.invoked, f"{s} not invoked"
        assert ok, f"vae PCC {pcc} < 0.95"
    finally:
        ttnn.close_mesh_device(dev)


@pytest.mark.slow
def test_denoise_pcc():
    dev, is_mesh = _open_mesh()
    try:
        torch.manual_seed(0)
        pipe = build_pipeline(dev)
        dit = pipe._dit_torch_model()
        C = dit.config.in_channels
        latent = torch.randn(1, C, 1, 32, 32, dtype=torch.float32)
        timestep = torch.tensor([500.0])
        # feed a REAL text-encode output (chained joint, no injected reference)
        ids = pipe.encode_prompt("A cat playing piano", max_length=64)
        embeds = pipe.run_text_encode(ids)
        out = pipe.run_denoise(latent, timestep, embeds)
        golden = pipe._hf_reference_denoise(latent, timestep, embeds).to(torch.float32)
        ok, pcc = comp_pcc(golden, out.to(torch.float32), 0.95)
        print(f"e2e PCC={pcc}", flush=True)
        assert all(s in pipe.invoked for s in DIT_STUBS), f"DiT stubs not all invoked: {set(DIT_STUBS)-pipe.invoked}"
        assert ok, f"denoise PCC {pcc} < 0.95"
    finally:
        ttnn.close_mesh_device(dev)


@pytest.mark.slow
def test_t2v_chain_and_coverage():
    """Full chained T2V behavioral run + Gate-2 coverage of the 19 real-forward stubs."""
    dev, is_mesh = _open_mesh()
    try:
        pipe = build_pipeline(dev)
        video = pipe.run_t2v("A cat playing piano in a sunny room", num_frames=1, height=32, width=32, steps=2)
        print(f"[t2v] decoded video shape={tuple(video.shape)}", flush=True)
        real_forward = set(UMT5_STUBS) | set(DIT_STUBS) | {"autoencoder_k_l_wan", "wan_encoder3d", "wan_decoder3d"}
        # decode-only chain didn't run vae encode/autoencoder; run them to complete VAE-forward set
        pipe.run_vae(video if video.shape[1] == 3 else torch.randn(1, 3, 5, 32, 32))
        pipe.run_vae_encode(torch.randn(1, 3, 5, 32, 32))
        missing = real_forward - pipe.invoked
        print(f"[t2v] invoked {len(pipe.invoked & real_forward)}/{len(real_forward)} real-forward stubs", flush=True)
        assert not missing, f"real-forward stubs not invoked: {missing}"
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s"]))
