# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch

from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import FalconForCausalLM


def load_falcon_reference_model(model_version, num_hidden_layers=None):
    """Load the vendored Falcon CPU reference with weights correctly populated, as float32.

    transformers 5.x silently fails to load the pretrained weights into the vendored
    ``hf_modeling_falcon`` model: layer params are left at their init values (LayerNorm = 1.0,
    Linear = random) while ``from_pretrained`` reports ``missing_keys=0`` (no error), which makes
    the reference — and the TT model, which is built from the same ``state_dict`` — produce
    NaN / ~0-PCC output. See issue #47924. The upstream ``transformers.FalconForCausalLM`` loads
    the *identical* checkpoint and key names correctly, so we copy its ``state_dict`` into the
    vendored model. The copy happens in the checkpoint dtype (bf16) and the upstream model is
    freed before the float32 upcast, so peak host memory is unchanged.

    The final ``.to(torch.float32)`` matches the float32 test inputs (avoids "mixed dtype (CPU)"
    LayerNorm errors) and mirrors the dtype fix from #47218 / #47929.
    """
    from transformers import FalconForCausalLM as UpstreamFalconForCausalLM

    kwargs = dict(local_files_only=os.getenv("CI") == "true", low_cpu_mem_usage=True)
    if num_hidden_layers is not None:
        kwargs["num_hidden_layers"] = num_hidden_layers

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_version, **kwargs)

    # Repair the silently-unloaded weights (#47924) by copying them from the upstream Falcon
    # class, which loads the same checkpoint correctly under transformers 5.x.
    upstream_model = UpstreamFalconForCausalLM.from_pretrained(model_version, **kwargs)
    hugging_face_reference_model.load_state_dict(upstream_model.state_dict())
    del upstream_model

    hugging_face_reference_model = hugging_face_reference_model.to(torch.float32)
    hugging_face_reference_model.eval()
    return hugging_face_reference_model


def load_hf_model(model_version):
    hugging_face_reference_model = load_falcon_reference_model(model_version)
    state_dict = hugging_face_reference_model.state_dict()

    return hugging_face_reference_model, state_dict
