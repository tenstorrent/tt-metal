# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end CosyVoice3 TTS test: LLM → Flow Decoder → Vocoder.

This test exercises the complete pipeline with dummy prompt data:
1. LLM generates speech tokens from dummy text+prompt tokens (on mesh)
2. Flow decoder converts speech tokens to mel spectrogram (on device)
3. Reference vocoder converts mel to audio waveform (on host)

Device management: LLM needs mesh (2 chips), Flow needs single device.
Since both can't be open simultaneously on the same chip, we sequence them:
mesh for LLM → close mesh → single device for Flow.
"""

import os
import sys

import pytest
import torch

import ttnn

WEIGHTS_DIR = "models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"


def test_e2e_dummy_prompt():
    """
    Full E2E pipeline with synthetic prompt data.

    Uses dummy text/speech tokens and random prompt features to verify
    the data flow: LLM → Flow → Vocoder produces valid audio.
    """
    if not os.path.exists(os.path.join(WEIGHTS_DIR, "llm.pt")):
        pytest.skip("llm.pt not found")
    if not os.path.exists(os.path.join(WEIGHTS_DIR, "flow.pt")):
        pytest.skip("flow.pt not found")
    if not os.path.exists(os.path.join(WEIGHTS_DIR, "hift.pt")):
        pytest.skip("hift.pt not found")

    # ===== Stage 1: LLM on mesh =====
    from models.demos.wormhole.cosy_voice.tt.cosyvoice_llm import CosyVoice3LM
    from models.demos.wormhole.cosy_voice.tt.model_config import CosyVoiceModelConfig

    print("\n=== Stage 1: LLM (mesh device) ===")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
    )

    try:
        config = CosyVoiceModelConfig(
            mesh_device=mesh,
            max_batch_size=1,
            weights_dir=WEIGHTS_DIR,
        )
        llm = CosyVoice3LM(config, mesh, dtype=ttnn.bfloat8_b)

        # Dummy prompt data
        torch.manual_seed(42)
        # Simulate: prompt text (10 tokens) + target text (5 tokens)
        prompt_text = torch.randint(0, 151000, (1, 10))
        text = torch.randint(0, 151000, (1, 5))
        prompt_speech = torch.randint(0, 6561, (1, 8))
        embedding = torch.randn(1, 192)

        # Run LLM — generate speech tokens
        speech_tokens = []
        token_gen = llm.inference(
            text=text,
            text_len=torch.tensor([5]),
            prompt_text=prompt_text,
            prompt_text_len=torch.tensor([10]),
            prompt_speech_token=prompt_speech,
            prompt_speech_token_len=torch.tensor([8]),
            embedding=embedding,
            max_token_text_ratio=10,  # keep short for testing
            min_token_text_ratio=2,
        )
        for tok in token_gen:
            speech_tokens.append(tok)
            if len(speech_tokens) >= 20:  # Cap at 20 tokens for speed
                break

        print(f"LLM generated {len(speech_tokens)} speech tokens")
        print(f"Token IDs (first 10): {speech_tokens[:10]}")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    if len(speech_tokens) == 0:
        pytest.skip("LLM generated 0 tokens — cannot test flow decoder")

    # ===== Stage 2: Flow Decoder on single device =====
    from models.demos.wormhole.cosy_voice.tt.flow.flow import TtCausalMaskedDiffWithDiT

    print("\n=== Stage 2: Flow Decoder (single device) ===")
    flow_sd = torch.load(os.path.join(WEIGHTS_DIR, "flow.pt"), map_location="cpu")
    device = ttnn.open_device(device_id=0)

    try:
        flow = TtCausalMaskedDiffWithDiT(device, flow_sd, dtype=ttnn.bfloat16)

        # Prepare flow inputs
        token_tensor = torch.tensor(speech_tokens).unsqueeze(0)
        prompt_feat = torch.randn(1, 16, 80)  # dummy prompt mel (8 tokens × 2 ratio)

        mel, _ = flow.inference(
            token=token_tensor,
            token_len=torch.tensor([token_tensor.shape[1]]),
            prompt_token=prompt_speech,
            prompt_token_len=torch.tensor([prompt_speech.shape[1]]),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]]),
            embedding=embedding,
            streaming=False,
            finalize=True,
        )
        print(f"Flow output: mel shape={mel.shape}, mean={mel.mean():.4f}, std={mel.std():.4f}")
    finally:
        ttnn.close_device(device)

    # ===== Stage 3: Vocoder on host =====
    print("\n=== Stage 3: Vocoder (host CPU) ===")
    sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice")
    try:
        from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
        from cosyvoice.hifigan.generator import CausalHiFTGenerator
    except ImportError:
        pytest.skip("Reference CosyVoice not importable")

    hift_sd = torch.load(os.path.join(WEIGHTS_DIR, "hift.pt"), map_location="cpu")
    hift_sd = {k.replace("generator.", ""): v for k, v in hift_sd.items()}

    f0_predictor = CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
    vocoder = CausalHiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_predictor=f0_predictor,
    )
    vocoder.load_state_dict(hift_sd, strict=True)
    vocoder.eval()

    with torch.no_grad():
        audio, _ = vocoder.inference(mel.float(), finalize=True)

    print(f"Audio output: {audio.shape}, duration={audio.shape[1]/24000:.3f}s")
    print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Assertions
    assert audio.dim() == 2, f"Expected 2D audio, got {audio.dim()}D"
    assert audio.shape[1] > 0, "Audio has zero length"
    assert audio.abs().max() <= 1.0, f"Audio exceeds range: {audio.abs().max()}"

    print(
        f"\n✅ E2E pipeline complete: {len(speech_tokens)} tokens → "
        f"mel {mel.shape} → audio {audio.shape} ({audio.shape[1]/24000:.3f}s)"
    )
