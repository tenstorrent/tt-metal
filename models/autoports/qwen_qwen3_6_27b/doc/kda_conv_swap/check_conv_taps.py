# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exercise the causal conv with all four taps nonzero.

The shared synthetic harness sets ``conv[:, 0, -1] = 0.5`` and leaves taps 0-2
at zero, which degenerates the convolution to ``silu(0.5 * x_t)``.  Under that
weight any tap-history bug -- a stale window, a wrong shift, a mis-seeded state
-- is invisible.  This runs multi-step decode and a multi-chunk prefill with a
dense tap kernel and compares the fused path against both the composite path
and the Hugging Face layer.
"""

import argparse

import torch
from transformers import AutoConfig, DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import LAYER, _hf_layer, _state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import TARGET_FABRIC, MultichipDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder
from models.common.utility_functions import comp_pcc


def dense_state(config, seed=4242):
    """The shared synthetic weights, but with a genuinely 4-tap convolution."""
    state = _state(config)
    generator = torch.Generator().manual_seed(seed)
    key_width = config.linear_num_key_heads * config.linear_key_head_dim
    conv_width = 2 * key_width + config.linear_num_value_heads * config.linear_value_head_dim
    taps = torch.randn(conv_width, 1, config.linear_conv_kernel_dim, generator=generator) * 0.4
    state[f"model.language_model.layers.{LAYER}.linear_attn.conv1d.weight"] = taps.bfloat16()
    return state


def build(mesh, config, state, candidate, batch, max_context, multichip=False):
    cls = MultichipDecoder if multichip else OptimizedDecoder
    return cls.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER,
        mesh_device=mesh,
        batch=batch,
        max_context=max_context,
        page_size=64,
        candidate=candidate,
    )


def tt_decode_steps(mesh, config, state, candidate, tokens, batch, multichip=False, active_mask=None):
    decoder = build(mesh, config, state, candidate, batch, 64, multichip)
    page_table = _to_device(
        torch.arange(batch, dtype=torch.int32).reshape(batch, 1),
        mesh_device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    outputs = []
    for step, token in enumerate(tokens):
        positions = _to_device(
            torch.full((batch,), step, dtype=torch.uint32),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        kwargs = {}
        if active_mask is not None:
            kwargs["active_mask"] = _to_device(
                active_mask.reshape(1, 1, 1, batch), mesh_device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        out = decoder.decode_forward(
            hidden_states=_to_device(token.reshape(1, 1, batch, -1), mesh_device=mesh),
            page_table=page_table,
            current_positions=positions,
            **kwargs,
        )
        ttnn.synchronize_device(mesh)
        outputs.append(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).squeeze(0).float())
    return outputs


def tt_prefill(mesh, config, state, candidate, hidden, batch, multichip=False):
    decoder = build(mesh, config, state, candidate, batch, max(64, hidden.shape[1]), multichip)
    page_table = _to_device(
        torch.arange(batch, dtype=torch.int32).reshape(batch, 1),
        mesh_device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    positions = _to_device(
        torch.arange(hidden.shape[1], dtype=torch.int64).to(torch.uint32).reshape(1, -1).expand(batch, -1),
        mesh_device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    out = decoder.prefill_forward(
        hidden_states=_to_device(hidden.unsqueeze(0), mesh_device=mesh),
        page_table=page_table,
        current_positions=positions,
    )
    ttnn.synchronize_device(mesh)
    return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).squeeze(0).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--bar", type=float, default=0.995)
    parser.add_argument("--multichip", action="store_true", help="run the TP4 path instead of single chip")
    parser.add_argument("--active-mask", action="store_true", help="deactivate every other batch row")
    args = parser.parse_args()

    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260831)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    state = dense_state(config)
    hf_layer = _hf_layer(config, state)

    if args.multichip:
        ttnn.set_fabric_config(TARGET_FABRIC)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    else:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    failures = []
    try:
        if args.mode == "decode":
            batch = args.batch
            tokens = [(torch.randn(batch, 1, config.hidden_size) * 0.2).bfloat16() for _ in range(args.steps)]
            cache = DynamicCache(config=config)
            reference = []
            with torch.no_grad():
                for token in tokens:
                    reference.append(
                        hf_layer(
                            token, position_embeddings=(None, None), attention_mask=None, past_key_values=cache
                        ).float()
                    )
            mask = None
            if args.active_mask:
                mask = torch.zeros(batch, dtype=torch.bfloat16)
                mask[::2] = 1.0
            runs = {
                c: tt_decode_steps(mesh, config, state, c, tokens, batch, args.multichip, mask)
                for c in ("linear_final", "linear_kda_conv")
            }
            for step in range(args.steps):
                ref = reference[step]
                comp, fused = runs["linear_final"][step], runs["linear_kda_conv"][step]
                pcc_c = comp_pcc(ref, comp, args.bar)
                pcc_f = comp_pcc(ref, fused, args.bar)
                pcc_cf = comp_pcc(comp, fused, args.bar)
                print(
                    f"DENSE_TAPS decode step={step} composite_vs_hf={pcc_c[1]} fused_vs_hf={pcc_f[1]} fused_vs_composite={pcc_cf[1]}"
                )
                if not pcc_f[0]:
                    failures.append(f"decode step {step}: fused vs HF {pcc_f[1]}")
        else:
            batch = args.batch
            hidden = (torch.randn(batch, args.sequence, config.hidden_size) * 0.2).bfloat16()
            with torch.no_grad():
                reference = hf_layer(
                    hidden,
                    position_embeddings=(None, None),
                    attention_mask=None,
                    past_key_values=DynamicCache(config=config),
                ).float()
            outs = {
                c: tt_prefill(mesh, config, state, c, hidden, batch, args.multichip)
                for c in ("linear_final", "linear_kda_conv")
            }
            pcc_c = comp_pcc(reference, outs["linear_final"], args.bar)
            pcc_f = comp_pcc(reference, outs["linear_kda_conv"], args.bar)
            pcc_cf = comp_pcc(outs["linear_final"], outs["linear_kda_conv"], args.bar)
            print(
                f"DENSE_TAPS prefill sequence={args.sequence} composite_vs_hf={pcc_c[1]} fused_vs_hf={pcc_f[1]} fused_vs_composite={pcc_cf[1]}"
            )
            if not pcc_f[0]:
                failures.append(f"prefill: fused vs HF {pcc_f[1]}")
    finally:
        ttnn.close_mesh_device(mesh)
        if args.multichip:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    if failures:
        raise SystemExit("DENSE_TAPS FAILED:\n  " + "\n  ".join(failures))
    print("DENSE_TAPS OK")


if __name__ == "__main__":
    main()
