# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared loader for the vendored Falcon CPU reference model.

Both the falcon40b tests (``tests/test_utils.py``) and the demo (``demo/demo.py``) need a
correctly-populated vendored ``hf_modeling_falcon.FalconForCausalLM``. This lives outside
``tests/`` so non-test code (the demo) can import it without depending on the test package.
See issue #47924 for why ``from_pretrained`` cannot be used under transformers 5.x.
"""

import gc
import json
import os
import re
from contextlib import nullcontext
from functools import lru_cache

import torch
from safetensors import safe_open
from transformers.utils import cached_file

# Skip the (expensive) random weight init at construction — every parameter is overwritten by the
# checkpoint load below, and for the full 60-layer model the init alone is minutes of wasted work
# (it pushed the e2e demo over its step timeout). Location varies across transformers 5.x; degrade
# to a no-op context if unavailable (correct, just slower).
try:
    from transformers.initialization import no_init_weights
except ImportError:  # pragma: no cover - layout differs across transformers versions
    try:
        from transformers.modeling_utils import no_init_weights
    except ImportError:  # pragma: no cover
        no_init_weights = nullcontext

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
def load_falcon_reference_model(model_version, num_hidden_layers=None, dtype=torch.float32):
    """Load the vendored Falcon CPU reference with weights correctly populated.

    Weights are read straight from the checkpoint in their native layout (see
    ``_native_falcon_state_dict`` and #47924) and copied into a freshly constructed vendored
    model via ``load_state_dict``, since transformers 5.x ``from_pretrained`` silently fails to
    populate the vendored model and the upstream model rewrites the Q/K rotary layout.

    ``tie_weights`` restores ``lm_head`` if the checkpoint ties it to the input embeddings; it is
    a no-op for the (untied) falcon-40b checkpoint.

    ``dtype`` defaults to float32, which the PCC tests need (they feed float32 inputs to a decoder
    layer, and a bf16 model would raise "mixed dtype (CPU)" in LayerNorm). The demo passes bfloat16
    — it only feeds token ids (no dtype mismatch) and runs the full 60-layer model, where float32
    would double host memory and the conversion is slow enough to exceed the demo's step timeout.

    Cached (the reference is used read-only): constructing the model re-initialises the large
    embedding/lm_head tensors, so rebuilding it for every parametrization made the decoder sweep
    exceed its step timeout. Building once and reusing it keeps the sweep within budget.
    """
    ci = os.getenv("CI") == "true"
    config_kwargs = {"local_files_only": ci}
    if num_hidden_layers is not None:
        config_kwargs["num_hidden_layers"] = num_hidden_layers
    config = FalconConfig.from_pretrained(model_version, **config_kwargs)

    with no_init_weights():
        hugging_face_reference_model = FalconForCausalLM(config)
    # strict=False: lm_head may be absent from the checkpoint when tied to the input embeddings.
    hugging_face_reference_model.load_state_dict(
        _native_falcon_state_dict(model_version, num_hidden_layers), strict=False
    )
    hugging_face_reference_model.tie_weights()

    hugging_face_reference_model = hugging_face_reference_model.to(dtype)
    hugging_face_reference_model.eval()
    gc.collect()
    return hugging_face_reference_model
