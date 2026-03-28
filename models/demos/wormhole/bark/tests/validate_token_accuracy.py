# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Token-level Accuracy Validation for Bark Small TTNN vs HuggingFace Reference.

Compares the TTNN implementation's forward-pass output against the HuggingFace
PyTorch reference implementation (suno/bark-small) at each pipeline stage.

Metrics:
  - PCC (Pearson Correlation Coefficient) for logits
  - Top-1 token agreement percentage
  - Max absolute error

Usage:
    python models/demos/wormhole/bark/tests/validate_token_accuracy.py
"""

import torch
from transformers import BarkModel

import ttnn
from models.demos.wormhole.bark.tt.bark_gpt import BarkConfig, TtBarkGPT, preprocess_model_parameters
from models.demos.wormhole.bark.tt.bark_fine import TtBarkFineModel, preprocess_fine_model_parameters


def pcc(a, b):
    """Compute Pearson Correlation Coefficient between two tensors."""
    a, b = a.float().flatten(), b.float().flatten()
    valid = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) == 0:
        return 0.0
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = torch.sqrt((a_c**2).sum() * (b_c**2).sum())
    if denom == 0:
        return 1.0 if (a_c * b_c).sum() == 0 else 0.0
    return ((a_c * b_c).sum() / denom).item()


def top1_agreement(logits_a, logits_b):
    """Percentage of positions where top-1 token matches."""
    tokens_a = logits_a.argmax(dim=-1)
    tokens_b = logits_b.argmax(dim=-1)
    return (tokens_a == tokens_b).float().mean().item() * 100


def validate_stage(stage_name, tt_model, ref_model, config, device, seq_len=64):
    """Validate one stage's forward pass against PyTorch reference."""
    print(f"\n{'='*60}")
    print(f"Validating: {stage_name}")
    print(f"{'='*60}")

    vocab_size = config.input_vocab_size
    input_ids = torch.randint(0, vocab_size, (1, seq_len))

    # Reference forward
    with torch.no_grad():
        ref_input_embeds = ref_model.input_embeds_layer(input_ids)
        ref_pos_ids = torch.arange(seq_len).unsqueeze(0)
        ref_pos_embeds = ref_model.position_embeds_layer(ref_pos_ids)
        ref_hidden = ref_input_embeds + ref_pos_embeds

        for layer in ref_model.layers:
            out = layer(ref_hidden)
            ref_hidden = out[0] if isinstance(out, tuple) else out

        ref_hidden = ref_model.layernorm_final(ref_hidden)
        ref_logits = ref_model.lm_head(ref_hidden)

    # TTNN forward
    tt_logits, _ = tt_model(input_ids=input_ids, use_cache=False, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_logits_torch = ttnn.to_torch(tt_logits)
    ttnn.deallocate(tt_logits)

    # Reshape for comparison
    if tt_logits_torch.dim() != ref_logits.dim():
        tt_logits_torch = tt_logits_torch.view_as(ref_logits)

    # Metrics
    pcc_score = pcc(tt_logits_torch, ref_logits)
    top1_pct = top1_agreement(tt_logits_torch, ref_logits)
    max_abs_err = (tt_logits_torch - ref_logits).abs().max().item()

    pcc_pass = pcc_score >= 0.95
    top1_pass = top1_pct >= 95.0

    print(f"  PCC:                {pcc_score:.6f}  {'✓' if pcc_pass else '✗ FAIL'} (target ≥ 0.95)")
    print(f"  Top-1 Agreement:    {top1_pct:.2f}%  {'✓' if top1_pass else '✗ FAIL'} (target ≥ 95%)")
    print(f"  Max Abs Error:      {max_abs_err:.6f}")

    return {
        "stage": stage_name,
        "pcc": pcc_score,
        "top1_pct": top1_pct,
        "max_abs_err": max_abs_err,
        "pcc_pass": pcc_pass,
        "top1_pass": top1_pass,
    }


def validate_fine_stage(tt_fine_model, ref_fine_model, device, seq_len=32):
    """Validate fine stage (non-causal, multi-codebook)."""
    print(f"\n{'='*60}")
    print(f"Validating: Fine (Coarse-to-Fine)")
    print(f"{'='*60}")

    n_codes_total = ref_fine_model.config.n_codes_total
    n_codes_given = ref_fine_model.config.n_codes_given

    # Create dummy input: [batch, seq, n_codes_total]
    input_tokens = torch.randint(0, 1024, (1, seq_len, n_codes_total), dtype=torch.long)

    results = []
    for codebook_idx in range(n_codes_given, n_codes_total):
        # Reference forward
        with torch.no_grad():
            ref_hidden = None
            for i in range(codebook_idx + 1):
                emb = ref_fine_model.input_embeds_layers[i](input_tokens[:, :, i])
                if ref_hidden is None:
                    ref_hidden = emb
                else:
                    ref_hidden = ref_hidden + emb

            pos_ids = torch.arange(seq_len).unsqueeze(0)
            ref_hidden = ref_hidden + ref_fine_model.position_embeds_layer(pos_ids)

            for layer in ref_fine_model.layers:
                out = layer(ref_hidden)
                ref_hidden = out[0] if isinstance(out, tuple) else out

            ref_hidden = ref_fine_model.layernorm_final(ref_hidden)
            head_idx = codebook_idx - n_codes_given
            ref_logits = ref_fine_model.lm_heads[head_idx](ref_hidden)

        # TTNN forward
        tt_logits = tt_fine_model(
            codebook_idx=codebook_idx,
            input_ids=input_tokens,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_logits_torch = ttnn.to_torch(tt_logits)
        ttnn.deallocate(tt_logits)

        if tt_logits_torch.dim() != ref_logits.dim():
            tt_logits_torch = tt_logits_torch.view_as(ref_logits)

        pcc_score = pcc(tt_logits_torch, ref_logits)
        top1_pct = top1_agreement(tt_logits_torch, ref_logits)
        pass_ok = pcc_score >= 0.95 and top1_pct >= 95.0
        print(f"  Codebook {codebook_idx}: PCC={pcc_score:.6f}, Top1={top1_pct:.1f}%  {'✓' if pass_ok else '✗'}")
        results.append({"codebook": codebook_idx, "pcc": pcc_score, "top1_pct": top1_pct})

    return results


def main():
    print("Loading HuggingFace Bark Small reference...")
    hf_model = BarkModel.from_pretrained("suno/bark-small")
    hf_model.eval()

    device = ttnn.open_device(device_id=0)

    try:
        compute_grid = device.compute_with_storage_grid_size()
        all_results = []

        # --- Semantic ---
        sem_config = BarkConfig(
            hidden_size=hf_model.semantic.config.hidden_size,
            num_heads=hf_model.semantic.config.num_heads,
            num_layers=hf_model.semantic.config.num_layers,
            block_size=hf_model.semantic.config.block_size,
            input_vocab_size=hf_model.semantic.config.input_vocab_size,
            output_vocab_size=hf_model.semantic.config.output_vocab_size,
            bias=getattr(hf_model.semantic.config, "bias", False),
            grid_size=compute_grid,
        )
        sem_params = preprocess_model_parameters(hf_model.semantic, device)
        tt_semantic = TtBarkGPT(device, sem_params, sem_config, is_causal=True)
        r = validate_stage("Semantic (Text-to-Semantic)", tt_semantic, hf_model.semantic, sem_config, device)
        all_results.append(r)

        # --- Coarse ---
        coarse_config = BarkConfig(
            hidden_size=hf_model.coarse_acoustics.config.hidden_size,
            num_heads=hf_model.coarse_acoustics.config.num_heads,
            num_layers=hf_model.coarse_acoustics.config.num_layers,
            block_size=hf_model.coarse_acoustics.config.block_size,
            input_vocab_size=hf_model.coarse_acoustics.config.input_vocab_size,
            output_vocab_size=hf_model.coarse_acoustics.config.output_vocab_size,
            bias=getattr(hf_model.coarse_acoustics.config, "bias", False),
            grid_size=compute_grid,
        )
        coarse_params = preprocess_model_parameters(hf_model.coarse_acoustics, device)
        tt_coarse = TtBarkGPT(device, coarse_params, coarse_config, is_causal=True)
        r = validate_stage("Coarse (Semantic-to-Coarse)", tt_coarse, hf_model.coarse_acoustics, coarse_config, device)
        all_results.append(r)

        # --- Fine ---
        fine_config = BarkConfig(
            hidden_size=hf_model.fine_acoustics.config.hidden_size,
            num_heads=hf_model.fine_acoustics.config.num_heads,
            num_layers=hf_model.fine_acoustics.config.num_layers,
            block_size=hf_model.fine_acoustics.config.block_size,
            input_vocab_size=hf_model.fine_acoustics.config.input_vocab_size,
            output_vocab_size=hf_model.fine_acoustics.config.output_vocab_size,
            bias=True,
            grid_size=compute_grid,
        )
        fine_params = preprocess_fine_model_parameters(hf_model.fine_acoustics, device)
        tt_fine = TtBarkFineModel(
            device, fine_params, fine_config,
            n_codes_total=hf_model.fine_acoustics.config.n_codes_total,
            n_codes_given=hf_model.fine_acoustics.config.n_codes_given,
        )
        fine_results = validate_fine_stage(tt_fine, hf_model.fine_acoustics, device)

        # Summary
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"{'Stage':<30} {'PCC':>10} {'Top-1 %':>10} {'Status':>8}")
        print("-" * 60)

        overall_pass = True
        for r in all_results:
            status = "PASS" if (r["pcc_pass"] and r["top1_pass"]) else "FAIL"
            if status == "FAIL":
                overall_pass = False
            print(f"{r['stage']:<30} {r['pcc']:>10.6f} {r['top1_pct']:>10.2f} {status:>8}")

        for fr in fine_results:
            pass_ok = fr["pcc"] >= 0.95 and fr["top1_pct"] >= 95.0
            if not pass_ok:
                overall_pass = False
            status = "PASS" if pass_ok else "FAIL"
            print(f"{'Fine CB' + str(fr['codebook']):<30} {fr['pcc']:>10.6f} {fr['top1_pct']:>10.2f} {status:>8}")

        print(f"\nTarget: PCC ≥ 0.95, Top-1 Agreement ≥ 95%")
        print(f"Result: {'ALL PASS ✓' if overall_pass else 'SOME FAILURES ✗'}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
