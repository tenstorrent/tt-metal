# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Language Model decode-mode PCC vs HuggingFace Qwen2 (single and multi-step decode after prefill)."""

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.tests.pcc.lm_pcc_common import (
    DECODE_GENERATION_LENGTH,
    DECODE_LAYERWISE_FAIL_STEP,
    PCC_THRESHOLD,
    SEQ_LEN,
    as_layer_probe,
    assert_tt_decode_positions_monotonic,
    build_tt_lm,
    compare_decode_hidden_pcc,
    compare_layerwise_decode_pcc,
    compare_l0_attention_stage_pccs,
    compare_l0_sdpa_stage_pccs,
    force_manual_fp32_decode_sdpa,
    hf_eager_bf16_softmax_attention,
    hf_cache_seq_length,
    prepare_failing_decode_step_context,
    print_decode_pcc_summary,
    print_l0_attention_stage_pcc_table,
    print_l0_decode_attention_pcc,
    print_l0_sdpa_stage_pcc_table,
    print_layerwise_decode_pcc_table,
    print_hf_decode_reference_comparison,
    print_manual_fp32_vs_fused_sdpa_comparison,
    capture_fused_vs_manual_l0_decode_at_step,
    print_fused_vs_manual_sdpa_investigation_report,
    reference_lm_decode_hidden,
    reference_lm_decode_l0_attention,
    reference_lm_decode_l0_attention_stages,
    reference_lm_decode_l0_sdpa_stages,
    reference_lm_decode_layer_hiddens,
    reference_lm_forward,
    reference_lm_prefill_cache,
    run_multi_step_decode_pcc_sweep,
    run_multi_step_decode_pcc_sweep_with_hf_attn,
    tt_decode_hidden,
    tt_prefill_hidden,
    verify_decode_cache_positions,
)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_decode_hidden_state_pcc(mesh_device, vv_config, lm_state):
    torch.manual_seed(0)
    cfg = vv_config.decoder

    # Same token stream as test_lm_pcc: SEQ_LEN prefill tokens + one decode token.
    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN + 1), dtype=torch.long)

    ref_hidden_full = reference_lm_forward(lm_state, input_ids, vv_config)
    ref_decode = ref_hidden_full[:, SEQ_LEN:]  # [1, 1, hidden]

    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    kv_cache = lm_tt.alloc_kv_cache(SEQ_LEN + 8)
    tt_prefill_hidden(lm_tt, input_ids[:, :SEQ_LEN], kv_cache)
    tt_decode = tt_decode_hidden(lm_tt, input_ids[:, SEQ_LEN : SEQ_LEN + 1], SEQ_LEN, kv_cache)

    passed_d, pcc_d = compare_decode_hidden_pcc(ref_decode, tt_decode)
    print(f"[test_lm_decode_pcc] decode step=0  PCC={pcc_d:.6f}")
    assert passed_d, f"LM decode last_hidden PCC {pcc_d:.6f} < {PCC_THRESHOLD} (fixed-cache + sdpa_decode)"


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_decode_multi_step_hidden_state_pcc(mesh_device, vv_config, lm_state):
    """Ten consecutive decode steps after a 32-token prefill (full-LM integration sweep)."""
    torch.manual_seed(0)
    cfg = vv_config.decoder

    prompt = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), dtype=torch.long)
    decode_tokens = torch.randint(0, cfg.vocab_size, (1, DECODE_GENERATION_LENGTH), dtype=torch.long)

    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    kv_cache = lm_tt.alloc_kv_cache(SEQ_LEN + DECODE_GENERATION_LENGTH + 8)
    tt_prefill_hidden(lm_tt, prompt, kv_cache)

    hf_cache = reference_lm_prefill_cache(lm_state, prompt, vv_config)
    prefill_hf_len = hf_cache_seq_length(hf_cache)
    assert prefill_hf_len == SEQ_LEN, f"HF cache length after prefill={prefill_hf_len}, expected {SEQ_LEN}"

    failures = []
    step_pccs: list[float] = []
    tt_positions: list[int] = []
    print("[test_lm_decode_pcc] multi-step decode after prefill")

    for step in range(DECODE_GENERATION_LENGTH):
        token = decode_tokens[:, step : step + 1]
        position = SEQ_LEN + step

        verify_decode_cache_positions(step, SEQ_LEN, position, hf_cache, tt_positions)

        ref_decode, hf_cache = reference_lm_decode_hidden(lm_state, token, vv_config, hf_cache)
        tt_decode = tt_decode_hidden(lm_tt, token, position, kv_cache)
        passed_d, pcc_d = compare_decode_hidden_pcc(ref_decode, tt_decode)
        step_pccs.append(pcc_d)

        print(f"Decode step {step}  PCC={pcc_d:.5f}")

        if not passed_d:
            failures.append(
                f"decode step={step} position={position} measured_pcc={pcc_d:.6f} threshold={PCC_THRESHOLD}"
            )

    assert_tt_decode_positions_monotonic(tt_positions, SEQ_LEN, DECODE_GENERATION_LENGTH)
    print_decode_pcc_summary(step_pccs)

    if failures:
        assert False, "LM multi-step decode PCC below threshold:\n" + "\n".join(failures)

    print("PASS")


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_decode_layerwise_pcc_at_failing_step(mesh_device, vv_config, lm_state):
    """Layer-wise PCC at decode step 7 (first step below 0.99 in the 10-step sweep).

    Diagnostic test: prints per-layer PCC to locate where TT vs HF bf16 diverges.
    """
    ctx = prepare_failing_decode_step_context(mesh_device, lm_state, vv_config)
    token = ctx["token"]
    position = ctx["position"]
    decode_step = ctx["decode_step"]

    ref_layers, ref_final, _ = reference_lm_decode_layer_hiddens(lm_state, token, vv_config, ctx["hf_cache"])
    probe = as_layer_probe(ctx["lm_tt"])
    tt_layers, tt_final = probe.forward_with_layer_hiddens(
        probe._embed(token), start_pos=position, kv_cache=ctx["kv_cache"]
    )

    layer_pccs = compare_layerwise_decode_pcc(ref_layers, ref_final, tt_layers, tt_final)
    print_layerwise_decode_pcc_table(decode_step, position, layer_pccs)

    _, overall_pcc = compare_decode_hidden_pcc(ref_final, tt_final)
    print(f"[layerwise decode] overall (final hidden) PCC={overall_pcc:.5f}")


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_decode_l0_attention_pcc_at_failing_step(mesh_device, vv_config, lm_state):
    """L0 decode attention PCC at step 7: TT sdpa_decode vs HF bf16 self-attn (40-token prefix)."""
    ctx = prepare_failing_decode_step_context(mesh_device, lm_state, vv_config)
    token = ctx["token"]
    position = ctx["position"]
    decode_step = ctx["decode_step"]

    ref_attn_in, ref_attn_out, _ = reference_lm_decode_l0_attention(
        lm_state, token, vv_config, ctx["hf_cache"], layer_idx=0
    )
    probe = as_layer_probe(ctx["lm_tt"])
    tt_attn_in, tt_attn_out = probe.l0_decode_attention(token, position, ctx["kv_cache"], layer_idx=0)

    in_pcc, out_pcc = print_l0_decode_attention_pcc(
        decode_step,
        position,
        ctx["cache_prefix_len"],
        ref_attn_in,
        ref_attn_out,
        tt_attn_in,
        tt_attn_out,
    )
    print(f"[L0 decode attention] matches layerwise L0 block output PCC ~0.987: attn_out={out_pcc:.5f}")


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_decode_l0_attention_stage_pcc_at_failing_step(mesh_device, vv_config, lm_state):
    """Stage-by-stage L0 decode attention PCC at step 7 to localize first divergence."""
    ctx = prepare_failing_decode_step_context(mesh_device, lm_state, vv_config)
    token = ctx["token"]
    position = ctx["position"]
    decode_step = ctx["decode_step"]

    ref_stages, _ = reference_lm_decode_l0_attention_stages(lm_state, token, vv_config, ctx["hf_cache"], layer_idx=0)
    probe = as_layer_probe(ctx["lm_tt"])
    tt_stages = probe.l0_decode_attention_stages(token, position, ctx["kv_cache"], layer_idx=0)

    stage_pccs = compare_l0_attention_stage_pccs(ref_stages, tt_stages)
    print_l0_attention_stage_pcc_table(
        decode_step,
        position,
        ctx["cache_prefix_len"],
        stage_pccs,
    )


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_decode_l0_sdpa_stage_pcc_at_failing_step(mesh_device, vv_config, lm_state):
    """Stage-by-stage L0 decode SDPA PCC at step 7 to localize first divergence inside attention."""
    ctx = prepare_failing_decode_step_context(mesh_device, lm_state, vv_config)
    token = ctx["token"]
    position = ctx["position"]
    decode_step = ctx["decode_step"]

    ref_stages, _ = reference_lm_decode_l0_sdpa_stages(lm_state, token, vv_config, ctx["hf_cache"], layer_idx=0)
    probe = as_layer_probe(ctx["lm_tt"])
    tt_manual_stages = probe.l0_decode_sdpa_stages(token, position, ctx["kv_cache"], layer_idx=0)
    tt_fused_stages = probe.l0_decode_sdpa_stages_fused(token, position, ctx["kv_cache"], layer_idx=0)

    stage_pccs = compare_l0_sdpa_stage_pccs(ref_stages, tt_manual_stages)
    print_l0_sdpa_stage_pcc_table(
        decode_step,
        position,
        ctx["cache_prefix_len"],
        stage_pccs,
        tt_path_note="manual fp32 matmul/softmax (test probe)",
    )

    _, fused_pcc = comp_pcc(ref_stages["sdpa_out"], tt_fused_stages["sdpa_out"], pcc=0.0)
    fused_err = (ref_stages["sdpa_out"] - tt_fused_stages["sdpa_out"]).abs().max().item()
    print(
        f"[L0 decode SDPA stages] production fused sdpa_out vs HF: " f"PCC={fused_pcc:.5f}  max_abs_err={fused_err:.6f}"
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_decode_multi_step_pcc_manual_fp32_sdpa_diagnostic(mesh_device, vv_config, lm_state):
    """Diagnostic: force existing manual fp32 SDPA branch; compare multi-step PCC vs fused baseline."""
    with force_manual_fp32_decode_sdpa():
        manual_pccs = run_multi_step_decode_pcc_sweep(
            mesh_device,
            lm_state,
            vv_config,
            label="manual fp32 SDPA (monkeypatched _fused_sdpa_decode_safe=False)",
        )
    print_manual_fp32_vs_fused_sdpa_comparison(manual_pccs)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_decode_hf_reference_attn_comparison_diagnostic(mesh_device, vv_config, lm_state):
    """Diagnostic: compare HF decode reference modes (sdpa / eager / eager bf16-softmax) vs TT fused."""
    runs = {
        "sdpa": run_multi_step_decode_pcc_sweep_with_hf_attn(
            mesh_device,
            lm_state,
            vv_config,
            hf_attn_implementation="sdpa",
            label="HF sdpa reference",
        ),
        "eager_fp32sm": run_multi_step_decode_pcc_sweep_with_hf_attn(
            mesh_device,
            lm_state,
            vv_config,
            hf_attn_implementation="eager",
            label="HF eager (fp32 softmax) reference",
        ),
        "eager_bf16sm": run_multi_step_decode_pcc_sweep_with_hf_attn(
            mesh_device,
            lm_state,
            vv_config,
            hf_attn_implementation="eager",
            label="HF eager (bf16 softmax patch) reference",
            hf_softmax_patch=hf_eager_bf16_softmax_attention(),
        ),
    }
    print_hf_decode_reference_comparison(runs)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_decode_fused_vs_manual_sdpa_report_at_step_7(mesh_device, vv_config, lm_state):
    """Diagnostic: stage-wise fused vs manual fp32 SDPA comparison at failing decode step 7 (position 39)."""
    investigation = capture_fused_vs_manual_l0_decode_at_step(
        mesh_device,
        lm_state,
        vv_config,
        decode_step=DECODE_LAYERWISE_FAIL_STEP,
    )
    report = print_fused_vs_manual_sdpa_investigation_report(investigation)

    assert investigation["decode_step"] == DECODE_LAYERWISE_FAIL_STEP
    assert investigation["position"] == SEQ_LEN + DECODE_LAYERWISE_FAIL_STEP
    assert investigation["input_embed_pcc"] >= 0.9999, "twin contexts should have identical inputs"

    sdpa_metrics = report["stage_metrics"]["sdpa_out"]
    assert sdpa_metrics["comparable"]
    print(
        f"\n[summary] step={investigation['decode_step']} position={investigation['position']} "
        f"sdpa_out PCC={sdpa_metrics['pcc']:.6f}  "
        f"hidden PCC={report['stage_metrics']['hidden_after_residual']['pcc']:.6f}"
    )
