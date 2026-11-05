# SPDX-License-Identifier: Apache-2.0
"""Reduced real-layer adapter check; not a full-model quality/performance result."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig

import ttnn

from ..tt.generator_vllm import Gemma4ForCausalLM
from ..tt.weights import HF_MODEL, HF_REVISION


def main():
    torch.set_num_threads(8)
    os.environ["GEMMA4_SERVING_TEST_LAYERS"] = "0,5"
    config = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    try:
        adapter = Gemma4ForCausalLM.initialize_vllm_model(config, mesh, 1, max_seq_len=4096)
        specs = [None] * 60
        for index, layer in zip(adapter.model.layer_indices, adapter.model.layers):
            specs[index] = ((256, layer.nkv, 32 if layer.sliding_window else 64, layer.head_dim), torch.bfloat16, 0)
        cache = adapter.allocate_kv_cache_per_layer(specs)
        assert len(adapter.cache_storage) == 1
        for kind in range(2):
            for left, right in zip(ttnn.get_device_tensors(cache[0][kind]), ttnn.get_device_tensors(cache[1][kind])):
                assert left.buffer_address() == right.buffer_address()
        adapter.warmup_model_prefill(cache, True, True)
        pages = torch.arange(128, dtype=torch.int32).reshape(1, -1)
        per_layer_pages = [pages] * 60
        per_layer_pages[5] = torch.arange(128, 192, dtype=torch.int32).reshape(1, -1)
        params = SimpleNamespace(temperature=[0.0], top_k=[1], top_p=[1.0], seed=[42])
        prompt = torch.tensor([[2, 125, 89] * 43], dtype=torch.int32)

        seed_audits = []

        def generate(stale, page_refresh=False, remap=False, reset=False, audit_seed=False):
            first = adapter.prefill_forward(
                prompt,
                start_pos=[0],
                prompt_lens=[129],
                page_table=pages,
                page_tables_per_layer=per_layer_pages,
                kv_cache=cache,
                sampling_params=params,
            )
            result = first.reshape(-1).tolist()

            def audit():
                if audit_seed:
                    actual = int(
                        ttnn.to_torch(ttnn.get_device_tensors(adapter.generator.sampler.seeds_tt_tensor)[0]).reshape(
                            -1
                        )[0]
                    )
                    expected = adapter.sampler_seeds[0]
                    seed_audits.append(dict(device=actual, host=expected))
                    assert actual == expected, (actual, expected)

            audit()
            for step in range(4):
                output = adapter.decode_forward(
                    torch.tensor([[result[-1] if not stale or step == 0 else 7]]),
                    torch.tensor([129 + step if not stale or step == 0 else 0]),
                    page_table=pages,
                    page_tables_per_layer=per_layer_pages,
                    kv_cache=cache,
                    reload_inputs=(step == 0 or (reset and step == 2)),
                    reload_page_table=page_refresh and step == 2,
                    reload_sampling_params=False,
                    reset_sampling_state=reset and step == 2,
                    read_from_device=False,
                    sampling_params=params,
                    slot_remap=[0] if remap and step == 2 else None,
                )
                host, events = adapter.read_decode_output(output, async_read=True)
                for event in events:
                    ttnn.event_synchronize(event)
                result.append(int(adapter.process_decode_output_host(host, is_tokens=True)[0, 0]))
                audit()
            return result

        # First decode establishes authoritative inputs in both runs; later
        # host values are deliberately stale and must not overwrite residency.
        control = generate(False)
        resident = generate(True)
        assert resident == control, (resident, control)
        refreshed = generate(True, page_refresh=True)
        assert refreshed == control, (refreshed, control)
        # First prefill draw lands at INT32_MAX; subsequent device draws cross
        # the signed boundary. Remap and absolute-position reset must preserve it.
        params = SimpleNamespace(temperature=[0.7], top_k=[32], top_p=[0.9], seed=[2**31 - 130])
        boundary_control = generate(False, audit_seed=True)
        boundary_remap = generate(False, remap=True, audit_seed=True)
        boundary_reset = generate(False, reset=True, audit_seed=True)
        assert boundary_control == boundary_remap == boundary_reset
        result = dict(
            scope="real layers0/5, shared vLLM storage, scheduler32/64 blocks, split-readback residency",
            shared_storage_verified=True,
            control=control,
            stale_host=resident,
            page_refresh_stale_host=refreshed,
            page_refresh_scope="same mapping reload preserves resident token/position; allocator growth tested in serving",
            seed_boundary=dict(
                control=boundary_control, remap=boundary_remap, reset=boundary_reset, seed_states=seed_audits
            ),
            counters=adapter.generator.counters,
            traces=[str(adapter.generator.model_trace), str(adapter.generator.sample_trace)],
        )
        Path("models/autoports/google_gemma_4_31b_it/readiness_vllm/adapter_state.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        adapter.release_persistent_capture()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
