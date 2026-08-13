# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Nonzero paged-cache regression across the 64-token page boundary."""

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import LAYER, _hf_layer, _state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, FunctionalDecoder, _to_device
from models.common.utility_functions import comp_pcc


@torch.no_grad()
def run():
    torch.manual_seed(20260729)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    config._attn_implementation = "eager"
    state = _state(config)
    hf_layer = _hf_layer(config, state)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    prompt = (torch.randn(1, 65, config.hidden_size) * 0.2).bfloat16()
    token = (torch.randn(1, 1, config.hidden_size) * 0.2 + 0.03).bfloat16()
    cache = DynamicCache(config=config)

    prompt_positions = torch.arange(65, dtype=torch.long).reshape(1, -1)
    prompt_rotary = rotary(prompt, prompt_positions.unsqueeze(0).expand(3, -1, -1))
    causal_mask = torch.full((1, 1, 65, 65), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    hf_layer(
        prompt,
        position_embeddings=prompt_rotary,
        position_ids=prompt_positions,
        attention_mask=causal_mask,
        past_key_values=cache,
    )
    decode_positions = torch.tensor([[65]], dtype=torch.long)
    decode_rotary = rotary(token, decode_positions.unsqueeze(0).expand(3, -1, -1))
    reference = hf_layer(
        token,
        position_embeddings=decode_rotary,
        position_ids=decode_positions,
        attention_mask=None,
        past_key_values=cache,
    )

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = FunctionalDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=LAYER,
            mesh_device=mesh,
            batch=1,
            max_context=128,
            page_size=64,
        )
        page_table = _to_device(
            torch.tensor([[1, 0]], dtype=torch.int32),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        decoder.prefill_forward(
            hidden_states=_to_device(prompt.unsqueeze(0), mesh_device=mesh),
            page_table=page_table,
            current_positions=_to_device(
                prompt_positions.to(torch.uint32),
                mesh_device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            ),
        )
        output = decoder.decode_forward(
            hidden_states=_to_device(token.reshape(1, 1, 1, -1), mesh_device=mesh),
            page_table=page_table,
            current_positions=_to_device(
                torch.tensor([65], dtype=torch.uint32),
                mesh_device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            ),
        )
        ttnn.synchronize_device(mesh)
        actual = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape_as(reference)
        passed, message = comp_pcc(reference.float(), actual.float(), 0.995)
        print("FULL_ATTENTION_PREFILL_DECODE_CACHE_PCC", message)
        assert passed, message

        # Logical page 0 is physical page 1 and is completely occupied by
        # prompt positions 0..63. Logical page 1 is physical page 0 and only
        # slots 0/1 (positions 64/65) may be occupied. These forbidden-slot
        # checks fail if the page table is ignored and identity routing is used.
        for name in ("key", "value"):
            cache_tensor = ttnn.to_torch(ttnn.get_device_tensors(decoder.caches[name])[0])
            occupancy = [
                [torch.count_nonzero(cache_tensor[block, :, slot]).item() for slot in (0, 1, 2, 63)] for block in (0, 1)
            ]
            print("CACHE_OCCUPANCY", name, occupancy)
            assert torch.count_nonzero(cache_tensor[1, :, 63]).item() > 0
            assert torch.count_nonzero(cache_tensor[0, :, :2]).item() > 0
            # Slot 63 is tile-aligned storage, unlike the partially padded
            # slots immediately after the two decode writes. Under ignored
            # identity routing it would contain logical token 63.
            assert torch.count_nonzero(cache_tensor[0, :, 63]).item() == 0
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    run()
