# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4.tt import spec_decode

from ..test_factory import parametrize_mesh_with_fabric


class FakeTensor:
    def __init__(self, name, owner=None):
        self.name = name
        self.owner = owner
        self.deallocated = False

    def deallocate(self, force=False):
        self.deallocated = True


def test_fused_verify_reshape_preserves_alias_owner(monkeypatch):
    permuted = []

    def reshape(tensor, shape):
        return FakeTensor("reshape", owner=tensor)

    def concat(tensors, dim):
        return FakeTensor("stacked")

    def permute(tensor, order):
        result = FakeTensor("user-major", owner=tensor)
        permuted.append(result)
        return result

    monkeypatch.setattr(spec_decode.ttnn, "reshape", reshape)
    monkeypatch.setattr(spec_decode.ttnn, "concat", concat)
    monkeypatch.setattr(spec_decode.ttnn, "permute", permute)

    assistant = SimpleNamespace(step=lambda *args: (FakeTensor("draft-logits"), FakeTensor("hidden")))
    target = SimpleNamespace(
        ttnn_packed_verify_forward=lambda **kwargs: (FakeTensor("verify-logits"), FakeTensor("verify-hidden"))
    )
    decoder = object.__new__(spec_decode.SpeculativeDecoder)
    decoder.draft_len = 1
    decoder.assistant = assistant
    decoder.target = target
    decoder._shared_kv = []
    # A target without per-layer inputs; later speculative changes read this in the fused body.
    decoder.target_has_pli = False
    decoder.tt_kv_cache = []
    decoder._argmax_last = lambda logits, rows: FakeTensor("argmax")
    trace = {
        "B": 1,
        "anchor_tok": FakeTensor("anchor"),
        "h": FakeTensor("hidden"),
        "d_pt": FakeTensor("draft-page-table"),
        "d_pu": FakeTensor("draft-position"),
        "d_pi": FakeTensor("draft-position-index"),
        "v_pos": FakeTensor("verify-position"),
        "mask_full": FakeTensor("full-mask"),
        "mask_slide": FakeTensor("sliding-mask"),
        "v_pt": FakeTensor("verify-page-table"),
        "write_idxs": [FakeTensor("write-index")],
    }

    verify_x, _, _ = decoder._fused_body_batched(trace)

    assert verify_x.owner is permuted[0]
    assert not permuted[0].deallocated


@pytest.mark.skipif(
    not (os.getenv("HF_MODEL") and os.getenv("GEMMA4_ASSISTANT_MODEL")),
    reason="set HF_MODEL and GEMMA4_ASSISTANT_MODEL to run",
)
@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_fused_verify_ids_survive_replays_under_scratch_pressure(mesh_device, reset_seeds, monkeypatch):
    """The packed verify IDs a replay reads back must be the IDs the replay verified.

    After every replay the verify row starts with the anchor token the host wrote
    into the trace input, followed by in-vocabulary drafts. Device allocations
    between replays reuse any buffer the trace no longer owns.
    """
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    # K+1 must keep the per-chip packed head dimension tile aligned (E2B: K=3/7 at TP=1/2).
    draft_len = int(os.getenv("GEMMA4_SPEC_DRAFT_LEN", "3"))
    monkeypatch.setenv("GEMMA4_SPEC_TRACE", "1")
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", str(draft_len))
    max_seq_len, block_size = 1024, 64
    paged = PagedAttentionConfig(block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size))
    generator, kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=os.environ["HF_MODEL"],
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=os.environ["GEMMA4_ASSISTANT_MODEL"],
    )
    page_table = create_tt_page_table(1, paged)
    prepared, encoded, positions, lengths = preprocess_inputs_prefill(
        ["The capital of France is"], tokenizer, generator.model_args, False, 24, max_prefill_len=max_seq_len
    )
    generator.prefill_forward_text(
        torch.stack(prepared).view(1, -1),
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=positions,
        warmup_prefill=False,
    )
    decoder = spec_decode.SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=kv_cache,
        page_table_torch=page_table,
        stop_tokens=set(),
        draft_len=draft_len,
    )

    P = draft_len + 1
    replicated = mesh_device.get_num_devices() > 1

    def host(tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0] if replicated else tensor).reshape(-1)

    replays = []
    original_ids_to_host = decoder._ids_to_host

    def recording_ids_to_host(ids_tt, n):
        trace = getattr(decoder, "_fused_trace_batched", None)
        if trace is not None and ids_tt is trace["vidx"]:
            verify_ids = host(trace["verify_x"])[:P].tolist()
            anchor = int(host(trace["anchor_tok"])[0])
            replays.append((anchor, [int(token) for token in verify_ids]))
            scratch = [
                ttnn.from_torch(
                    torch.full((1, 1, 32, 4096), -7.0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if replicated else None,
                )
                for _ in range(8)
            ]
            for tensor in scratch:
                tensor.deallocate(True)
        return original_ids_to_host(ids_tt, n)

    decoder._ids_to_host = recording_ids_to_host
    anchor_pos = lengths[0] - 1
    anchor_token = int(encoded[0][anchor_pos])
    outs, accepts = decoder.generate_batched([anchor_token], [anchor_pos], 24, max_seq_len)

    assert len(replays) >= 4
    assert replays[0][0] == anchor_token
    vocab = target.vocab_size
    for index, (anchor, verify_ids) in enumerate(replays):
        assert verify_ids[0] == anchor, f"replay {index}: verify row {verify_ids} does not start with anchor {anchor}"
        assert all(0 <= token < vocab for token in verify_ids), f"replay {index}: {verify_ids}"
    assert outs[0]
    assert all(0 <= count <= draft_len for count in accepts[0])
