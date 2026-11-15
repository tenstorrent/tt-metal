# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from models.demos.llama31_8b_p100_spec.demo.demo import create_tt_model
from models.demos.llama31_8b_p100_spec.tt import eagle_draft, eagle_head
from models.ttt_compat.tt.common import PagedAttentionConfig, preprocess_inputs_prefill
from models.ttt_compat.tt.generator import Generator, SamplingParams
from models.ttt_compat.tt.model_config import DecodersPrecision

PROMPT = "List three facts about the planet Mars and explain why each one matters for a future crewed mission."


def _pcc(a, b):
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def _host(mesh_device, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0, 0].float()


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": True, "num_command_queues": 1, "trace_region_size": 150000000}], indirect=True
)
def test_feature_taps_and_map(mesh_device):
    paged = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
    args, model, kv, _ = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        optimizations=lambda a: DecodersPrecision.performance(a.n_layers, a.model_name),
        max_seq_len=8192,
        paged_attention_config=paged,
        dtype=ttnn.bfloat8_b,
    )
    taps = eagle_draft.FeatureTaps(mesh_device, model, args.dim)
    weights, _ = eagle_head.read_checkpoint(eagle_head.head_dir())
    feature_map = eagle_draft.FeatureMap(mesh_device, args, weights["fc.weight"])

    g = torch.Generator().manual_seed(7)
    reference = []
    for t in taps.step:
        v = torch.randn(1, 1, eagle_draft.STEP_ROWS, args.dim, generator=g) * 0.5
        reference.append(v[0, 0])
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
            ),
            t,
        )
    mapped = feature_map(taps.step)
    expected = torch.cat(reference, -1).to(torch.bfloat16).float() @ weights["fc.weight"].float().T
    map_pcc = _pcc(_host(mesh_device, mapped), expected)
    assert map_pcc > 0.9999, f"feature map on device against the host product: correlation {map_pcc:.6f}"
    ttnn.deallocate(mapped)

    generator = Generator([model], [args], mesh_device, tokenizer=args.tokenizer)
    sampling = SamplingParams(temperature=0.0, top_k=32, top_p=0.08)
    page_table = torch.arange(1024, dtype=torch.int32).reshape(1, -1)
    tokens, _, _, lengths = preprocess_inputs_prefill([PROMPT], args.tokenizer, [args], True, 32, max_prefill_len=8192)
    tokens = torch.stack(tokens).view(1, -1)
    length = int(lengths[0])
    assert 2 <= length <= eagle_draft.PROMPT_ROWS
    generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=[kv],
        prompt_lens=[length],
        sampling_params=sampling,
        warmup_prefill=True,
        enable_trace=True,
    )
    from_prefill = [_host(mesh_device, t)[:length].clone() for t in taps.prompt]
    assert all(float(rows.abs().sum()) > 0 for rows in from_prefill)

    worst = 1.0
    fresh = True
    for p in range(length):
        generator.decode_forward(
            tokens[:, p : p + 1],
            torch.tensor([p], dtype=torch.int64),
            page_table=page_table,
            kv_cache=[kv],
            enable_trace=True,
            sampling_params=sampling,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=fresh,
        )
        fresh = False
        for layer, t in enumerate(taps.step):
            worst = min(worst, _pcc(_host(mesh_device, t)[0], from_prefill[layer][p]))
    assert worst > 0.98, f"prefill and decode hidden states differ: correlation {worst:.4f}"
