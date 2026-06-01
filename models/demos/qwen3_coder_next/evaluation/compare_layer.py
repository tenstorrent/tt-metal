# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare full_fused vs old fused output at DeltaNet layer 0 in the full model."""

import torch
import ttnn
from transformers import AutoTokenizer
from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.load_weights import load_state_dict
from models.demos.qwen3_coder_next.tt.model import TtQwen3CoderNextModel
from models.demos.qwen3_coder_next.tt.generator import Qwen3CoderNextGenerator
import models.demos.qwen3_coder_next.tt.deltanet as dn

MODEL_PATH = "/tmp/qwen36_model/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def main():
    config = Qwen3CoderNextConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    state_dict = load_state_dict(config, model_path=MODEL_PATH)
    device = ttnn.open_device(device_id=0)

    try:
        # Build model
        dn.USE_FULL_FUSED_KERNEL = False
        dn.USE_FUSED_KERNEL = True
        model = TtQwen3CoderNextModel(device, state_dict, config)
        gen = Qwen3CoderNextGenerator(model, config, tokenizer=tokenizer)

        # Prefill
        gen.reset()
        prompt = "Hello world"
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long)
        logits = gen.prefill(ids)
        l = ttnn.to_torch(logits).float().reshape(-1)
        first_tok = torch.argmax(l[:config.vocab_size]).item()
        print(f"First decode token: {first_tok}", flush=True)

        # Monkey-patch the DeltaNet forward to capture pre-o_proj output
        layer0_deltanet = model.layers[0].token_mixer  # TtGatedDeltaNet
        captured = {}

        # Save original methods
        orig_decode_fused = layer0_deltanet._decode_step_fused
        orig_decode_full_fused = layer0_deltanet._decode_step_full_fused

        def patched_decode_fused(hidden_states, deltanet_state):
            # Call original
            result = orig_decode_fused(hidden_states, deltanet_state)
            captured['old_output'] = ttnn.to_torch(result).clone()
            return result

        def patched_decode_full_fused(hidden_states, deltanet_state):
            result = orig_decode_full_fused(hidden_states, deltanet_state)
            captured['fused_output'] = ttnn.to_torch(result).clone()
            return result

        # Run old fused decode
        layer0_deltanet._decode_step_fused = patched_decode_fused
        dn.USE_FULL_FUSED_KERNEL = False
        dn.USE_FUSED_KERNEL = True

        # Save all states
        ds = gen.deltanet_state
        all_states = {}
        for i in range(config.num_hidden_layers):
            if config.layer_types[i] == "linear_attention":
                all_states[i] = {
                    'rec': ttnn.to_torch(ds.get_recurrent_state(i)).clone(),
                    'conv': ttnn.to_torch(ds.get_conv_state(i)).clone(),
                }

        _, tok_old = gen.decode_one_token(torch.tensor([[first_tok]], dtype=torch.long))
        print(f"Old fused token: {tok_old.item()}", flush=True)
        print(f"Old output: shape={captured['old_output'].shape} "
              f"min={captured['old_output'].min():.6f} max={captured['old_output'].max():.6f}", flush=True)

        # Restore all states
        for i in all_states:
            ds.set_recurrent_state(
                i, ttnn.from_torch(all_states[i]['rec'], dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=device))
            ds.set_conv_state(
                i, ttnn.from_torch(all_states[i]['conv'], dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=device))
        gen.position -= 1

        # Also restore attention KV caches
        gen.kv_caches = {}

        # Re-prefill to get clean attention state
        gen.deltanet_state = model.create_deltanet_state()
        gen.position = 0
        logits = gen.prefill(ids)

        # Run full fused decode
        layer0_deltanet._decode_step_full_fused = patched_decode_full_fused
        dn.USE_FULL_FUSED_KERNEL = True

        _, tok_fused = gen.decode_one_token(torch.tensor([[first_tok]], dtype=torch.long))
        print(f"Full fused token: {tok_fused.item()}", flush=True)
        print(f"Fused output: shape={captured['fused_output'].shape} "
              f"min={captured['fused_output'].min():.6f} max={captured['fused_output'].max():.6f}", flush=True)

        # Compare
        old_out = captured['old_output'].float()
        fused_out = captured['fused_output'].float()
        pcc = torch.corrcoef(torch.stack([old_out.flatten(), fused_out.flatten()]))[0, 1].item()
        mse = ((old_out - fused_out) ** 2).mean().item()
        max_diff = (old_out - fused_out).abs().max().item()
        print(f"\nLayer 0 DeltaNet output comparison (after o_proj):")
        print(f"  PCC: {pcc:.6f}")
        print(f"  MSE: {mse:.8f}")
        print(f"  Max abs diff: {max_diff:.6f}")
        print(f"  Old range: [{old_out.min():.6f}, {old_out.max():.6f}]")
        print(f"  Fused range: [{fused_out.min():.6f}, {fused_out.max():.6f}]")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
