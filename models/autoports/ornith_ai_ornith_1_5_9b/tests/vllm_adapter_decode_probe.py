# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the existing adversarial decode battery through the serving adapter.

Keeps the battery's exact stale-object, output ownership, page-table, inactive
state and remap assertions. Only its builder and public boundaries change.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.autoports.ornith_ai_ornith_1_5_9b.tests import full_model_serving_decode_probe
from models.autoports.ornith_ai_ornith_1_5_9b.tt import generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm import OrnithServingDiagnostic


def build_adapter(model_dir, mesh_device, **kwargs):
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained("ornith-ai/Ornith-1.5-9B", local_files_only=True)
    batch, context = kwargs["max_batch_size"], kwargs["max_seq_len"]
    adapter = OrnithServingDiagnostic.initialize_vllm_model(hf_config, mesh_device, batch, context)
    pages = (context + 31) // 32
    cache = adapter.allocate_kv_cache_per_layer([((batch * pages, 1, 32, 256), torch.bfloat16, i) for i in range(8)])
    gen = adapter.generator
    gen.page_table = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
    gen.bind_serving_cache(cache, page_table=gen.page_table)
    # Keep original bound methods behind the adapter, so the battery can still
    # inspect/patch canonical state without bypassing the serving call boundary.
    names = ("prefill_forward", "decode_forward", "read_decode_output", "process_decode_output_host")
    adapter.generator = SimpleNamespace(
        cache=cache, counters=gen.counters, **{name: getattr(gen, name) for name in names}
    )
    for name in names:
        setattr(gen, name, getattr(adapter, name))

    def prefill_with_metadata(tokens, *args, **kwargs):
        # This legacy adversarial battery admits ordinary prompts only. Its
        # fixture supplies the original lengths explicitly at the serving seam;
        # resumed-request probes must pass their own original boundary.
        if "original_prompt_lens" not in kwargs:
            kwargs["original_prompt_lens"] = [int(length) for length in kwargs["prompt_lens"]]
        return adapter.prefill_forward(tokens, *args, **kwargs)

    gen.prefill_forward = prefill_with_metadata
    return gen


if __name__ == "__main__":
    with patch.object(generator, "build_generator", build_adapter):
        full_model_serving_decode_probe.main()
