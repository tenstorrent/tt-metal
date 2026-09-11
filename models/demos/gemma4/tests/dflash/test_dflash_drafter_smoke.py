# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small, focused smoke test for JUST the DFlash drafter (``DFlashDrafter`` in
tt/dflash_drafter.py) -- no target model, no prefill, no KV cache, no verify,
no trace capture. Loads only the drafter's own (small, ~5-layer) checkpoint
plus the target's tied embed_tokens weight (a single small tensor read, not a
full target model construction), feeds it SYNTHETIC random context via
``append_taps_torch`` (the module's own documented "host path, v1/tests" entry
point), and checks that ``draft()`` runs end to end and returns sane output.

This does NOT validate acceptance rate or generation quality -- synthetic
random context can't produce meaningful drafts. It only checks that the
drafter model itself (weight loading, attention, MLP, RoPE, tied lm_head,
context-append) runs correctly on device and returns well-formed output. For
a real, coherent generation check, see demo/dflash_fused_decoder_demo.py.

Requires HF_MODEL (target, for the tied lm_head's embed_tokens weight -- only
that one tensor is read, not the full checkpoint) and the z-lab dFlash drafter
snapshot in the HF cache (auto-discovered) or GEMMA4_DFLASH_DRAFTER.

Run: HF_MODEL=google/gemma-4-31B-it pytest \
    models/demos/gemma4/tests/dflash/test_dflash_drafter_smoke.py -k 1x8 -s
"""

import os

import pytest
import torch

from ...tests.test_factory import parametrize_mesh_with_fabric


def _dflash_default_snapshot():
    """Locate the z-lab drafter snapshot in the HF cache (mirrors
    generator_vllm.py's own helper -- reimplemented here so this test doesn't
    need to import generator_vllm.py, which pulls in tt_transformers'
    generator_vllm.py -> vllm, not installed in every dev environment)."""
    import glob

    hits = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--z-lab--gemma-4-31B-it-DFlash/snapshots/*/"))
    return hits[0] if hits else None


def _target_embed_weight(model_path):
    """Read just the target's embed_tokens weight tensor -- not a full model
    load. Tries the local HF cache snapshot for ``model_path`` first, then
    falls back to a plain AutoModel-free safetensors scan via huggingface_hub
    if the path itself is a loadable local/hub id."""
    import glob
    import json

    from safetensors import safe_open

    snapshot_dir = model_path if os.path.isdir(model_path) else None
    if snapshot_dir is None:
        hits = glob.glob(
            os.path.expanduser(f"~/.cache/huggingface/hub/models--*--{model_path.split('/')[-1]}/snapshots/*/")
        )
        snapshot_dir = hits[0].rstrip("/") if hits else None
    if not snapshot_dir or not os.path.isdir(snapshot_dir):
        pytest.skip(f"could not resolve a local snapshot dir for HF_MODEL={model_path!r}")

    idx = json.load(open(f"{snapshot_dir}/model.safetensors.index.json"))
    key = next(
        k
        for k in idx["weight_map"]
        if k.endswith("language_model.embed_tokens.weight") or k.endswith("model.embed_tokens.weight")
    )
    with safe_open(f"{snapshot_dir}/{idx['weight_map'][key]}", framework="pt") as f:
        return f.get_tensor(key)


@parametrize_mesh_with_fabric([(1, 8)])
def test_dflash_drafter_smoke(mesh_device, device_params, reset_seeds):
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager
    from models.demos.gemma4.tt.dflash_drafter import DFlashDrafter

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target, for the tied lm_head's embed weight) to run")

    snap = os.environ.get("GEMMA4_DFLASH_DRAFTER") or _dflash_default_snapshot()
    if not snap:
        pytest.skip("dFlash drafter snapshot not found; set GEMMA4_DFLASH_DRAFTER")

    embed_w = _target_embed_weight(model_path)

    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=mesh_device.shape[1]))
    ccl_manager = CCLManager(mesh_device)

    drafter = DFlashDrafter(
        mesh_device=mesh_device,
        drafter_path=snap,
        target_embed_weight_loader=lambda: embed_w,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        tensor_cache_path=None,
    )

    # Sanity: the loaded checkpoint's own basic shape expectations.
    assert drafter.vocab == embed_w.shape[0]
    assert drafter.block_size > 1
    assert len(drafter.target_layer_ids) > 0

    # Synthetic context: `fc`'s stored (transposed) shape is [1,1,IN,OUT] with
    # IN = len(target_layer_ids) * target_hidden -- derive target_hidden from
    # it instead of hardcoding a checkpoint-specific number, so this test
    # doesn't silently break if either config changes.
    n_taps = len(drafter.target_layer_ids)
    fc_in = drafter.fc.shape[-2]
    assert fc_in % n_taps == 0, f"fc input width {fc_in} not divisible by {n_taps} taps"
    target_hidden = fc_in // n_taps

    ctx_rows = 32  # a plausible small prompt length
    synthetic_taps = torch.randn(ctx_rows, n_taps * target_hidden, dtype=torch.float32) * 0.02
    drafter.append_taps_torch(synthetic_taps)
    assert drafter._ctx_len == ctx_rows

    anchor_id = int(torch.randint(0, drafter.vocab, (1,)).item())
    start_pos = ctx_rows
    draft_ids = drafter.draft(anchor_id, start_pos)

    K = drafter.block_size - 1
    assert len(draft_ids) == K, f"expected {K} draft ids, got {len(draft_ids)}"
    assert all(
        isinstance(i, int) and 0 <= i < drafter.vocab for i in draft_ids
    ), f"draft ids out of vocab range [0, {drafter.vocab}): {draft_ids}"

    # A second draft call at an advanced position, reusing the same
    # accumulated context, exercises the growing-context-buffer path
    # (append -> draft -> append -> draft) without re-loading anything.
    more_taps = torch.randn(K + 1, n_taps * target_hidden, dtype=torch.float32) * 0.02
    drafter.append_taps_torch(more_taps)
    assert drafter._ctx_len == ctx_rows + K + 1
    draft_ids_2 = drafter.draft(draft_ids[-1], start_pos + K + 1)
    assert len(draft_ids_2) == K
    assert all(0 <= i < drafter.vocab for i in draft_ids_2)
