# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import gc
import json
import os
import re
from functools import lru_cache

import torch
from safetensors import safe_open
from transformers.utils import cached_file

from models.demos.t3000.falcon40b.reference.hf_configuration_falcon import FalconConfig
from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import FalconForCausalLM


@lru_cache(maxsize=1)
def _native_falcon_state_dict(model_version, num_hidden_layers):
    """Read the pretrained Falcon weights straight from the checkpoint shards, in their native
    layout, keeping only the tensors for the requested number of layers.

    Why not ``from_pretrained`` (see issue #47924):
      * Loading the checkpoint through the *vendored* ``hf_modeling_falcon.FalconForCausalLM``
        silently leaves every parameter at its init value under transformers 5.x —
        ``from_pretrained`` reports ``missing_keys=0`` but copies nothing.
      * Loading through the *upstream* ``transformers.FalconForCausalLM`` does populate the
        weights, but it applies a rotary-layout transform to the Q/K rows of the fused
        ``query_key_value`` weight (the V rows are left untouched) to match upstream's rotary.
        Copying that ``state_dict`` into the vendored reference — whose ``rotate_half`` expects
        the *native* checkpoint layout — produces a correct output and V-cache but a wrong
        K-cache (K-cache PCC collapses while V-cache stays ~1.0).

    Reading the raw checkpoint tensors avoids both problems and reproduces the
    pre-transformers-5.x (native) behavior the TT model is validated against. Only the tensors
    for ``num_hidden_layers`` layers are read, so the single-layer decoder test stays cheap.
    """
    ci = os.getenv("CI") == "true"

    def want(key):
        m = re.match(r"transformer\.h\.(\d+)\.", key)
        if m:
            return num_hidden_layers is None or int(m.group(1)) < num_hidden_layers
        return True

    state_dict = {}

    def _resolve(filename):
        try:
            return cached_file(model_version, filename, local_files_only=ci)
        except Exception:
            return None

    def _add_safetensors(path, keys):
        with safe_open(path, framework="pt") as f:
            available = set(f.keys())
            for k in available if keys is None else keys:
                if k in available and want(k):
                    state_dict[k] = f.get_tensor(k)

    # Prefer safetensors (sharded, then single); fall back to pickled .bin.
    index_path = _resolve("model.safetensors.index.json")
    if index_path is not None:
        weight_map = json.load(open(index_path))["weight_map"]
        shard_to_keys = {}
        for k in weight_map:
            if want(k):
                shard_to_keys.setdefault(weight_map[k], []).append(k)
        for shard, keys in shard_to_keys.items():
            _add_safetensors(cached_file(model_version, shard, local_files_only=ci), keys)
        return state_dict

    single = _resolve("model.safetensors")
    if single is not None:
        _add_safetensors(single, None)
        return state_dict

    bin_index = _resolve("pytorch_model.bin.index.json")
    if bin_index is not None:
        weight_map = json.load(open(bin_index))["weight_map"]
        for shard in {weight_map[k] for k in weight_map if want(k)}:
            shard_sd = torch.load(
                cached_file(model_version, shard, local_files_only=ci), map_location="cpu", weights_only=True
            )
            state_dict.update({k: v for k, v in shard_sd.items() if want(k)})
        return state_dict

    bin_sd = torch.load(
        cached_file(model_version, "pytorch_model.bin", local_files_only=ci), map_location="cpu", weights_only=True
    )
    state_dict.update({k: v for k, v in bin_sd.items() if want(k)})
    return state_dict


@lru_cache(maxsize=1)
def load_falcon_reference_model(model_version, num_hidden_layers=None):
    """Load the vendored Falcon CPU reference with weights correctly populated, as float32.

    Weights are read straight from the checkpoint in their native layout (see
    ``_native_falcon_state_dict`` and #47924) and copied into a freshly constructed vendored
    model via ``load_state_dict``, since transformers 5.x ``from_pretrained`` silently fails to
    populate the vendored model and the upstream model rewrites the Q/K rotary layout.

    ``tie_weights`` restores ``lm_head`` if the checkpoint ties it to the input embeddings; it is
    a no-op for the (untied) falcon-40b checkpoint. The final ``.to(torch.float32)`` matches the
    float32 test inputs (avoids "mixed dtype (CPU)" LayerNorm errors).

    Cached (the reference is used read-only): constructing the model re-initialises the large
    embedding/lm_head tensors, so rebuilding it for every parametrization made the decoder sweep
    exceed its step timeout. Building once and reusing it keeps the sweep within budget.
    """
    ci = os.getenv("CI") == "true"
    config_kwargs = {"local_files_only": ci}
    if num_hidden_layers is not None:
        config_kwargs["num_hidden_layers"] = num_hidden_layers
    config = FalconConfig.from_pretrained(model_version, **config_kwargs)

    hugging_face_reference_model = FalconForCausalLM(config)
    # strict=False: lm_head may be absent from the checkpoint when tied to the input embeddings.
    hugging_face_reference_model.load_state_dict(
        _native_falcon_state_dict(model_version, num_hidden_layers), strict=False
    )
    hugging_face_reference_model.tie_weights()

    hugging_face_reference_model = hugging_face_reference_model.to(torch.float32)
    hugging_face_reference_model.eval()
    gc.collect()
    return hugging_face_reference_model


def load_hf_model(model_version):
    hugging_face_reference_model = load_falcon_reference_model(model_version)
    state_dict = hugging_face_reference_model.state_dict()

    return hugging_face_reference_model, state_dict
