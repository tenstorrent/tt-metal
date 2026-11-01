# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Parity of ModelArgs.load_state_dict's per-layer safetensors path with the full HF loader.

With cache_hf=False and n_layers below the checkpoint's layer count, load_state_dict reads only
the configured layers (plus embeddings, final norm and lm_head) straight from the safetensors
shards. This test loads the same one-layer model through that path and through the
AutoModelForCausalLM.from_pretrained path (cache_hf=True) and requires identical keys and
tensors, and it fails loudly if the per-layer path was skipped or fell back.

Runs against whatever HF_MODEL the job sets: a dense model, a tied-embedding model (the
lm_head alias) and Mixtral (per-expert weights the HF loader fuses and expand_fused_moe_experts
splits again) all go through here in CI.
"""

import pytest
import torch

from models.tt_transformers.tt import model_config as model_config_module
from models.tt_transformers.tt.model_config import ModelArgs

N_LAYERS = 1
MAX_SEQ_LEN = 1024


def test_configured_layers_state_dict_matches_hf_loader(mesh_device, monkeypatch, ensure_gc):
    per_layer_calls = []
    real_loader = model_config_module.load_hf_state_dict_for_layers

    def recording_loader(*args, **kwargs):
        per_layer_calls.append(args)
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(model_config_module, "load_hf_state_dict_for_layers", recording_loader)

    fast_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=MAX_SEQ_LEN)
    if fast_args.dummy_weights:
        pytest.skip("dummy weights: nothing to compare")
    if fast_args.full_model_n_layers <= N_LAYERS:
        pytest.skip(f"{fast_args.model_name} has {fast_args.full_model_n_layers} layer(s); no layers to leave out")
    fast_args.n_layers = N_LAYERS
    # The fallback path starts with get_hf_model_cls(); make it fail instead of silently loading everything.
    monkeypatch.setattr(
        fast_args,
        "get_hf_model_cls",
        lambda: pytest.fail("load_state_dict fell back to from_pretrained; the per-layer safetensors path was skipped"),
    )
    fast_state_dict = fast_args.load_state_dict()
    assert len(per_layer_calls) == 1, f"expected one per-layer safetensors read, saw {len(per_layer_calls)}"

    ref_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=MAX_SEQ_LEN, cache_hf=True)
    ref_args.n_layers = N_LAYERS
    ref_state_dict = ref_args.load_state_dict()

    missing = sorted(set(ref_state_dict) - set(fast_state_dict))
    extra = sorted(set(fast_state_dict) - set(ref_state_dict))
    assert not missing and not extra, f"key sets differ: missing from per-layer path {missing[:10]}, extra {extra[:10]}"

    mismatched = []
    for key, ref_tensor in ref_state_dict.items():
        fast_tensor = fast_state_dict[key]
        if (
            fast_tensor.dtype != ref_tensor.dtype
            or fast_tensor.shape != ref_tensor.shape
            or not torch.equal(fast_tensor, ref_tensor)
        ):
            mismatched.append(key)
    assert not mismatched, f"{len(mismatched)} tensors differ between the two paths, e.g. {mismatched[:10]}"
