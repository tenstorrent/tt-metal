# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.prefetcher import Prefetcher


@pytest.mark.skipif(
    not trace_allocation_tracker.TRACE_ALLOC_TRACKING, reason="requires TT_METAL_TRACE_ALLOC_TRACKING=1 at startup"
)
@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True, "trace_region_size": 16000000}], indirect=True)
def test_blackhole_prefetcher_prefill_then_decode_trace(mesh_device, reset_seeds, monkeypatch, tmp_path):
    if "blackhole" not in str(mesh_device.arch()).lower():
        pytest.skip("Requires the Blackhole Llama-8B DRAM prefetcher")
    monkeypatch.setenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))
    prefetcher = Prefetcher(mesh_device, num_tensors=5, num_layers=1)
    args = ModelArgs(
        mesh_device, instruct=True, dummy_weights=True, max_batch_size=1, max_seq_len=512, prefetcher=prefetcher
    )
    args.n_layers = 1
    args.trace_prefill_supported_seq_lens = [128]
    paging = PagedAttentionConfig(block_size=32, max_num_blocks=16)
    model = Transformer(
        args=args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=args.load_state_dict(),
        weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paging,
        prefetcher=prefetcher,
    )
    generator = Generator([model], [args], mesh_device)
    kv_cache = [[layer.attention.layer_past for layer in model.layers]]
    page_table = torch.arange(16, dtype=torch.int32).reshape(1, 16)
    prompt = torch.ones((1, 128), dtype=torch.int32)
    tokens = torch.ones((1, 1), dtype=torch.int32)
    positions = torch.tensor([128], dtype=torch.int32)

    def prefill(traced):
        return generator.prefill_forward_text(
            prompt,
            prompt_lens=torch.tensor([128]),
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=traced,
            warmup_prefill=False,
        )

    def decode(traced):
        return generator.decode_forward(
            tokens,
            positions,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=traced,
            reload_inputs=True,
            reload_page_table=True,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )[0]

    try:
        # Compile eager readback and obtain a reference before any trace exists.
        expected_prefill = prefill(False)
        expected_decode = decode(False)
        assert prefetcher._global_cb_addresses is None
        saved_ids = None
        saved_addresses = None
        for _ in range(3):
            actual_prefill = prefill(True)
            assert generator._trace_io_buffers._frozen
            assert prefetcher.mode == Mode.PREFILL
            assert prefetcher.global_cb is None
            assert any(generator.trace_id_prefill.values())
            actual_decode = decode(True)
            torch.testing.assert_close(actual_prefill, expected_prefill, rtol=0, atol=0)
            torch.testing.assert_close(actual_decode, expected_decode, rtol=0, atol=0)
            assert torch.isfinite(actual_decode).all()
            addresses = (prefetcher.global_cb.buffer_address(), prefetcher.global_cb.config_address())
            assert addresses == prefetcher._global_cb_addresses
            ids = (dict(generator.trace_id_prefill), dict(generator.trace_ids_decode[False]))
            if saved_ids is not None:
                assert ids == saved_ids
                assert addresses == saved_addresses
            saved_ids, saved_addresses = ids, addresses
    finally:
        generator.release_persistent_capture()
