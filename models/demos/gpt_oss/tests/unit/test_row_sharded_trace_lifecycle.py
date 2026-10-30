# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Row-sharded GPT-OSS trace lifecycle on Wormhole Galaxy, without a checkpoint."""

import pytest
import torch
from transformers import AutoConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

import ttnn
from models.common.sampling import SamplingParams
from models.common.utility_functions import is_wormhole_b0
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.model import Model
from models.demos.gpt_oss.tt.model_config import ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format


@pytest.mark.skipif(not is_wormhole_b0() or ttnn.get_num_devices() < 32, reason="Requires Wormhole Galaxy")
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "l1_small_size": 16384,
            "trace_region_size": 64000000,
        }
    ],
    indirect=True,
)
def test_row_sharded_prefill_decode_reuses_prepared_io(mesh_device, reset_seeds, monkeypatch):
    monkeypatch.setenv("HF_MODEL", "models/demos/gpt_oss/configs/gpt-oss-20b")
    config = AutoConfig.from_pretrained("models/demos/gpt_oss/configs/gpt-oss-20b")
    config.num_hidden_layers = 1
    config.max_position_embeddings = 512
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        reference = GptOssForCausalLM(config).eval()
    finally:
        torch.set_default_dtype(original_dtype)
    weights = convert_hf_qkv_to_meta_format(reference.state_dict(), config.head_dim)
    args = ModelArgs(mesh_device, dummy_weights=True, max_batch_size=128, max_seq_len=512)
    args.hf_config, args.vocab_size, args.n_layers = config, config.vocab_size, 1
    model = Model.create_transformer_compatible(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh_device,
        state_dict=weights,
        tensor_cache_path=None,
        paged_attention_config=PagedAttentionConfig(block_size=64, max_num_blocks=256),
        mesh_config=MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=8, ep=4)),
        create_kv_cache=True,
        users_row_sharded=True,
        use_throughput_experts=True,
    )
    del reference, weights
    generator = Generator([model], [args], mesh_device)
    pages = torch.arange(128, dtype=torch.int32).reshape(32, 4).repeat(4, 1)
    caches = [[layer.self_attn.layer_past for layer in model.layers]]
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    tokens = torch.ones((128, 128), dtype=torch.int32)
    tokens[:, 0] = tokens[:, 30] = 200006
    tokens[:, 31] = 173781
    try:
        expected_tokens = []
        expected_state = None
        for iteration in range(2):
            out = generator.prefill_forward_text(
                tokens,
                page_table=pages,
                kv_cache=caches,
                prompt_lens=[32] * 128,
                sampling_params=params,
                enable_trace=True,
                warmup_prefill=False,
            )
            for step in range(2):
                out = generator.decode_forward(
                    out[0].reshape(128, 1),
                    torch.full((128,), 32 + step, dtype=torch.int32),
                    page_table=pages,
                    kv_cache=caches,
                    sampling_params=params,
                    enable_trace=True,
                    reload_inputs=True,
                    reload_page_table=True,
                    reload_sampling_params=True,
                    reset_sampling_state=True,
                )
                ttnn.synchronize_device(mesh_device)
                assert out[0].numel() == 128
                if iteration == 0:
                    expected_tokens.append(out[0].clone())
                else:
                    torch.testing.assert_close(out[0], expected_tokens[step], rtol=0, atol=0)
            state = (
                dict(generator.trace_id_prefill),
                {key: dict(ids) for key, ids in generator.trace_ids_decode.items() if ids},
                [(t.buffer_unique_id(), t.buffer_address()) for *_, t in generator._trace_io_buffers._buffers],
                mesh_device.num_program_cache_entries(),
            )
            if expected_state is None:
                expected_state = state
            else:
                assert state == expected_state
    finally:
        generator.release_persistent_capture()
