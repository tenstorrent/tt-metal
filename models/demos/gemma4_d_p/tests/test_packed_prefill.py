# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Short CP8/TP4 runner integration using a two-layer Gemma4 with synthetic weights."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.ccl import CCLManager
from models.demos.gemma4_d_p.tt.model import Gemma4Model
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.precision import Gemma4Precision
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.packing import PrefillSegment
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def _natural_output(runtime):
    tensors = ttnn.get_device_tensors(runtime.output)
    rows = torch.stack([ttnn.to_torch(tensors[rank * 4]).float().squeeze(0).squeeze(0) for rank in range(8)])
    return rows.reshape(8, -1, 32, rows.shape[-1]).permute(1, 0, 2, 3).reshape(-1, rows.shape[-1])


@torch.no_grad()
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 32 * 1024 * 1024})
@pytest.mark.timeout(600)
def test_packed_prefill_runner(mesh_device, monkeypatch):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    import models.demos.gemma4_d_p.tt.runners.runtime as runtime_module

    torch.manual_seed(771)
    torch.set_num_threads(8)
    config = Gemma4TextConfig(
        hidden_size=768,
        intermediate_size=3072,
        vocab_size=256,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_global_key_value_heads=4,
        head_dim=256,
        global_head_dim=512,
        sliding_window=1024,
        layer_types=["sliding_attention", "full_attention"],
        attention_k_eq_v=True,
        hidden_size_per_layer_input=0,
        max_position_embeddings=16384,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        },
    )
    config._attn_implementation = "sdpa"
    reference = Gemma4TextModel(config).eval()
    # Gemma uses scale=1; keep synthetic QK logits at unit variance.
    for layer in reference.layers:
        layer.self_attn.q_norm.weight.fill_(layer.self_attn.head_dim**-0.5)
    state = {f"model.language_model.{k}": v.bfloat16() for k, v in reference.state_dict().items()}
    reference.load_state_dict({k.removeprefix("model.language_model."): v.float() for k, v in state.items()})
    reference.norm = torch.nn.Identity()
    args = Gemma4ModelArgs(
        hidden_size=768,
        intermediate_size=3072,
        vocab_size=256,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_global_key_value_heads=4,
        layer_types=tuple(config.layer_types),
    )
    args._hf_text_config = config
    mesh = MeshConfig(mesh_device)
    caches = allocate_ring_kv_caches(mesh, args, num_users=6, max_seq_len=16384)

    def create_model(**kwargs):
        model = Gemma4Model(
            mesh,
            args,
            state,
            CCLManager(mesh),
            prefill_chunk_size=kwargs["prefill_chunk_size"],
            precision=Gemma4Precision({"kv_cache": ttnn.bfloat8_b}),
            max_seq_len=16384,
            ring_kv_caches=caches,
            stripe_size=32,
        )
        return args, model, caches, state

    monkeypatch.setattr(runtime_module, "create_tt_model", create_model)
    monkeypatch.setenv("GEMMA4_PREFILL_MAX_TRACES", "2")
    params = SimpleNamespace(
        mesh_shape=(8, 4), sp_axis=0, chunk_size=8192, max_seq_len=16384, num_users=6, num_layers=2, use_trace=True
    )
    runtime = Gemma4PrefillRuntime(mesh_device=mesh_device, hf_model_id="synthetic", tt_cache_path=None, config=params)
    completions = []
    runtime.set_layer_completion_sink(lambda layer, request: completions.append((layer, request)))
    runtime.compile(caches)
    runtime.capture_trace(caches)
    histories = [[] for _ in range(6)]
    first_trace = None
    schedules = [
        [(i, 0, length) for i, length in enumerate((256, 512, 768, 1024, 1536, 2048))],
        [(0, 256, 512), (2, 768, 256), (5, 2048, 768)],
        [(5 - i, 0, length) for i, length in enumerate((256, 512, 768, 1024, 1536, 2048))],
        [(1, 0, 1600), (4, 0, 257)],
    ]
    total_requests = 0
    try:
        for iteration, schedule in enumerate(schedules):
            segments = tuple(PrefillSegment(slot, start, start + length) for slot, start, length in schedule)
            token_lists = [
                torch.randint(1, 256, (end - start,)).tolist()
                for _, start, end in [(s.slot_id, s.actual_start, s.actual_end) for s in segments]
            ]
            for segment, tokens in zip(segments, token_lists):
                histories[segment.slot_id] = histories[segment.slot_id][: segment.actual_start] + tokens
            inp = runtime.make_batch_input(token_lists, segments)
            runtime.prefill_batch(inp, caches, segments=segments)
            if iteration == 0:
                first_trace = runtime.trace_id
            elif iteration == 2:
                assert (
                    runtime.trace_id == first_trace
                ), "Same sizes must reuse the trace across different slots/prefixes"
            assert len(runtime.traces) <= 2
            output = _natural_output(runtime)
            offset = 0
            for segment in segments:
                expected = reference(torch.tensor([histories[segment.slot_id]]), use_cache=False).last_hidden_state[0]
                expected = expected[segment.actual_start : segment.actual_end]
                actual = output[offset : offset + len(expected)]
                assert torch.isfinite(actual).all()
                passed, pcc = comp_pcc(expected, actual, 0.98)
                assert passed, f"iteration={iteration} request={segment}: {pcc}"
                offset += segment.padded_length()
                assert runtime.slot_ends[segment.slot_id] == segment.actual_end
            total_requests += len(segments)
            assert completions == [(layer, request) for request in range(total_requests) for layer in range(2)]
    finally:
        runtime.release_trace()
