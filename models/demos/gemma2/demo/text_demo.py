# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Gemma-2 text demo.

Gemma-2 is a plain dense text decoder, so this reuses the generic
``tt_transformers`` text demo (``simple_text_demo.test_demo_text``) verbatim -
including all prefill/decode tracing, sampling and perf reporting - and only
swaps the model construction to the dedicated Gemma-2 ``ModelArgs`` subclass
(which adds the RMSNorm unit offset and the sqrt(hidden) embedding scale). No
shared ``tt_transformers`` source is modified.

Run exactly like the generic demo, e.g.:

    export HF_MODEL=google/gemma-2-9b-it   # or google/gemma-2-2b-it
    export MESH_DEVICE=P150
    pytest models/demos/gemma2/demo/text_demo.py -k "batch-1 and performance" -s
"""

import models.tt_transformers.demo.simple_text_demo as _demo
import ttnn
from models.demos.gemma2.tt.model_config import ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig


def _create_tt_model_gemma2(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    paged_attention_config: PagedAttentionConfig = None,
    dtype=ttnn.bfloat8_b,
    state_dict=None,
    num_layers=None,
    use_prefetcher=False,
    use_hf_rope=False,
):
    """Mirror of ``common.create_tt_model`` but using the Gemma-2 ``ModelArgs``."""
    from models.tt_transformers.tt.model import Transformer
    from models.tt_transformers.tt.prefetcher import Prefetcher

    num_tensors = 5 if use_prefetcher else 0
    prefetcher = Prefetcher(mesh_device, num_tensors, num_layers) if use_prefetcher else None

    tt_model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        prefetcher=prefetcher,
        use_hf_rope=use_hf_rope,
    )

    if num_layers is not None:
        tt_model_args.n_layers = num_layers
    if prefetcher is not None:
        prefetcher.num_layers = tt_model_args.n_layers
    if not state_dict:
        state_dict = tt_model_args.load_state_dict()

    model = Transformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
        prefetcher=prefetcher,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers] if paged_attention_config else None
    return tt_model_args, model, tt_kv_cache, state_dict


# Route the generic demo's model construction through the Gemma-2 ModelArgs.
# `test_demo_text` resolves `create_tt_model` from its module globals at call
# time, so patching the module attribute is sufficient (no source edits).
_demo.create_tt_model = _create_tt_model_gemma2

# Re-export the generic demo entrypoint so pytest collects it (parametrize marks
# travel with the function object).
test_demo_text = _demo.test_demo_text
