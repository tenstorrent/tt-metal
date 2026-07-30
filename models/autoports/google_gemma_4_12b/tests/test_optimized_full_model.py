# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

import ttnn


MODEL_ID = "google/gemma-4-12B"
ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tt" / "generator.py"
PROMPT_FILE = ROOT / "doc" / "full_model" / "artifacts" / "aime24_prompt_0_plain.txt"

PROFILE_LAYERS = int(os.getenv("GEMMA4_12B_FULL_MODEL_PROFILE_LAYERS", "2"))
TRACE_REFRESH_LAYERS = int(os.getenv("GEMMA4_12B_TRACE_REFRESH_LAYERS", "1"))
TRACE_REGION_SIZE = int(os.getenv("GEMMA4_12B_FULL_MODEL_TRACE_REGION_SIZE", "100000000"))


def _require_t3k():
    if ttnn.get_num_devices() < 8:
        pytest.skip("google/gemma-4-12B optimized full model target requires an 8-device T3K mesh")


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("gemma4_12b_optimized_full_model_generator", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _signpost(name: str) -> None:
    try:
        from tracy import signpost
    except ImportError:
        return
    signpost(name)


def _prompt_token_ids():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return tokenizer.encode(PROMPT_FILE.read_text(encoding="utf-8").strip(), add_special_tokens=True)


def _first_mesh_tensor_to_torch(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


@pytest.mark.parametrize("num_layers", [PROFILE_LAYERS], ids=lambda value: f"{value}_layers")
def test_optimized_full_model_prefill_and_traced_decode_profile(num_layers):
    _require_t3k()
    generator_module = _load_generator_module()
    prompt_tokens = _prompt_token_ids()
    prompt = torch.tensor([prompt_tokens], dtype=torch.long)
    prompt_len = len(prompt_tokens)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), trace_region_size=TRACE_REGION_SIZE)
    try:
        generator = generator_module.build_generator(
            model_dir=ROOT,
            mesh_device=mesh_device,
            num_layers=num_layers,
            use_on_device_sampling=True,
        )
        try:
            generator.reset()
            generator.prefill_forward(
                prompt,
                page_table=generator.page_table_tt,
                kv_cache=generator.kv_cache,
                prompt_lens=[prompt_len],
                return_all_logits=False,
            )
            ttnn.synchronize_device(mesh_device)

            _signpost("PERF_PREFILL")
            generator.reset()
            prefill_logits = generator.prefill_forward(
                prompt,
                page_table=generator.page_table_tt,
                kv_cache=generator.kv_cache,
                prompt_lens=[prompt_len],
                return_all_logits=False,
            )
            ttnn.synchronize_device(mesh_device)
            _signpost("PERF_PREFILL_END")

            feed_token = int(torch.argmax(generator._mask_generation_logits(prefill_logits)[0, 0, :], dim=-1).item())
            token = torch.tensor([[feed_token]], dtype=torch.long)
            start_pos = torch.tensor([prompt_len], dtype=torch.long)

            generator.decode_forward(
                token,
                start_pos,
                page_table=generator.page_table_tt,
                kv_cache=generator.kv_cache,
                sample_on_device=True,
                enable_trace=True,
                return_ttnn=True,
            )
            ttnn.synchronize_device(mesh_device)

            _signpost("PERF_DECODE")
            generator.decode_forward(
                token,
                start_pos,
                page_table=generator.page_table_tt,
                kv_cache=generator.kv_cache,
                sample_on_device=True,
                enable_trace=True,
                return_ttnn=True,
            )
            ttnn.synchronize_device(mesh_device)
            _signpost("PERF_DECODE_END")
        finally:
            teardown = getattr(generator, "teardown", None)
            if callable(teardown):
                teardown()
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        gc.collect()


def test_decode_trace_refreshes_token_position_and_page_table_inputs():
    _require_t3k()
    generator_module = _load_generator_module()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), trace_region_size=TRACE_REGION_SIZE)
    try:
        generator = generator_module.build_generator(
            model_dir=ROOT,
            mesh_device=mesh_device,
            num_layers=TRACE_REFRESH_LAYERS,
            use_on_device_sampling=True,
        )
        try:
            generator.reset()
            page_table_a = torch.arange(generator.max_num_blocks, dtype=torch.int32).reshape(1, -1)
            page_table_b = page_table_a.clone()
            if page_table_b.shape[1] > 1:
                page_table_b[0, 0], page_table_b[0, 1] = page_table_b[0, 1].clone(), page_table_b[0, 0].clone()

            generator.decode_forward(
                torch.tensor([[123]], dtype=torch.long),
                torch.tensor([64], dtype=torch.long),
                page_table=page_table_a,
                kv_cache=generator.kv_cache,
                sample_on_device=True,
                enable_trace=True,
                return_ttnn=True,
            )
            trace_state = next(iter(generator._decode_traces.values()))

            generator.decode_forward(
                torch.tensor([[321]], dtype=torch.long),
                torch.tensor([65], dtype=torch.long),
                page_table=page_table_a,
                kv_cache=generator.kv_cache,
                sample_on_device=True,
                enable_trace=True,
                return_ttnn=True,
                async_decode=True,
            )
            ttnn.synchronize_device(mesh_device)

            token_input, position_input, cache_position_input = trace_state["inputs"]
            assert int(_first_mesh_tensor_to_torch(token_input).reshape(-1)[0].item()) == 321
            assert int(_first_mesh_tensor_to_torch(position_input).reshape(-1)[0].item()) == 65
            assert int(_first_mesh_tensor_to_torch(cache_position_input).reshape(-1)[0].item()) == 65
            page_table_host = _first_mesh_tensor_to_torch(trace_state["page_table"]).reshape_as(page_table_a)
            torch.testing.assert_close(page_table_host, page_table_a)

            generator.decode_forward(
                torch.tensor([[654]], dtype=torch.long),
                torch.tensor([66], dtype=torch.long),
                page_table=page_table_b,
                kv_cache=generator.kv_cache,
                sample_on_device=True,
                enable_trace=True,
                return_ttnn=True,
                async_decode=True,
            )
            ttnn.synchronize_device(mesh_device)

            assert int(_first_mesh_tensor_to_torch(token_input).reshape(-1)[0].item()) == 654
            assert int(_first_mesh_tensor_to_torch(position_input).reshape(-1)[0].item()) == 66
            assert int(_first_mesh_tensor_to_torch(cache_position_input).reshape(-1)[0].item()) == 66
            page_table_host = _first_mesh_tensor_to_torch(trace_state["page_table"]).reshape_as(page_table_b)
            torch.testing.assert_close(page_table_host, page_table_b)
        finally:
            teardown = getattr(generator, "teardown", None)
            if callable(teardown):
                teardown()
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        gc.collect()
