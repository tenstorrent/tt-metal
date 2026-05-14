# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reference CosyVoice3 Flow Decoder baseline — runs entirely on CPU/PyTorch.

This script uses the SAME frontend inputs as infer.py but runs the reference
PyTorch flow decoder instead of the TT version. If this produces intelligible
audio, the weights and inputs are correct and the bug is in our TT translation.
If this also produces unintelligible audio, the problem is upstream.

Usage:
    python3 models/demos/wormhole/cosy_voice/tests/test_reference_flow.py
"""

import os
import sys

import soundfile as sf


def custom_torchaudio_load(filepath, **kwargs):
    audio, sr = sf.read(filepath)
    if len(audio.shape) == 1:
        tensor = torch.tensor(audio).unsqueeze(0).float()
    else:
        tensor = torch.tensor(audio).transpose(0, 1).float()
    return tensor, sr


def custom_torchaudio_save(filepath, tensor, sample_rate, **kwargs):
    audio = tensor.transpose(0, 1).cpu().numpy()
    sf.write(filepath, audio, sample_rate)


import torchaudio

torchaudio.load = custom_torchaudio_load
torchaudio.save = custom_torchaudio_save

import torch

# Ensure reference code is in path
sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice")
sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice/third_party/Matcha-TTS")


def main():
    weights_dir = "models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"

    # ── 1. Load frontend (same as infer.py) ──
    print("Loading frontend...")
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from hyperpyyaml import load_hyperpyyaml

    with open(os.path.join(weights_dir, "cosyvoice3.yaml"), "r") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "llm": None,
                "flow": None,
                "hift": None,
                "qwen_pretrain_path": os.path.join(weights_dir, "CosyVoice-BlankEN"),
            },
        )

    frontend = CosyVoiceFrontEnd(
        configs["get_tokenizer"],
        configs["feat_extractor"],
        os.path.join(weights_dir, "campplus.onnx"),
        os.path.join(weights_dir, "speech_tokenizer_v3.onnx"),
        os.path.join(weights_dir, "spk2info.pt"),
        configs["allowed_special"],
    )

    # ── 2. Process inputs (same as infer.py) ──
    text = "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"
    prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    prompt_wav = "models/demos/wormhole/cosy_voice/ref/CosyVoice/asset/zero_shot_prompt.wav"

    print(f"Processing: text='{text[:20]}...' prompt_wav='{prompt_wav}'")
    model_input = frontend.frontend_zero_shot(text, prompt_text, prompt_wav, 16000, "")

    prompt_speech_token = model_input["llm_prompt_speech_token"]
    prompt_feat = model_input["prompt_speech_feat"]
    embedding = model_input["llm_embedding"]
    flow_prompt_token = model_input["flow_prompt_speech_token"]
    flow_embedding = model_input["flow_embedding"]

    print(f"Prompt speech tokens: {prompt_speech_token.shape}")
    print(f"Prompt feat: {prompt_feat.shape}")
    print(f"Embedding: {embedding.shape}")
    print(f"Flow prompt token: {flow_prompt_token.shape}")
    print(f"Flow embedding: {flow_embedding.shape}")

    # ── 3. Load reference flow decoder on CPU ──
    print("\nLoading reference flow decoder...")
    with open(os.path.join(weights_dir, "cosyvoice3.yaml"), "r") as f:
        flow_configs = load_hyperpyyaml(
            f,
            overrides={
                "llm": None,
                "hift": None,
                "qwen_pretrain_path": os.path.join(weights_dir, "CosyVoice-BlankEN"),
            },
        )
    ref_flow = flow_configs["flow"]
    # CRITICAL: The YAML only creates the model structure with random weights.
    # We must load the actual pretrained weights from flow.pt.
    flow_sd = torch.load(os.path.join(weights_dir, "flow.pt"), map_location="cpu", weights_only=True)
    ref_flow.load_state_dict(flow_sd)
    ref_flow.eval()
    print(f"Reference flow decoder loaded with weights: {type(ref_flow)}")

    # ── 4. Use FIXED speech tokens (simulate LLM output) ──
    # Use the prompt tokens repeated as a simple test, then also test with
    # the actual prompt_speech_token to verify the flow decoder works at all.
    # First test: just use prompt tokens as "generated" tokens to see if
    # the reference flow produces intelligible audio with known-good tokens.
    print("\n=== Test 1: Reference flow with prompt tokens as target ===")
    test_token = prompt_speech_token.clone().to(torch.int32)
    print(f"Test token shape: {test_token.shape}, first 10: {test_token.flatten()[:10].tolist()}")

    with torch.inference_mode():
        ref_mel, _ = ref_flow.inference(
            token=test_token,
            token_len=torch.tensor([test_token.shape[1]], dtype=torch.int32),
            prompt_token=flow_prompt_token.to(torch.int32),
            prompt_token_len=torch.tensor([flow_prompt_token.shape[1]], dtype=torch.int32),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32),
            embedding=flow_embedding,
            streaming=False,
            finalize=True,
        )
    print(f"Reference mel shape: {ref_mel.shape}")
    print(f"Reference mel stats: min={ref_mel.min():.4f}, max={ref_mel.max():.4f}, mean={ref_mel.mean():.4f}")

    # ── 5. Load vocoder ──
    print("\nLoading vocoder...")
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
    from cosyvoice.hifigan.generator import CausalHiFTGenerator

    hift_sd = torch.load(os.path.join(weights_dir, "hift.pt"), map_location="cpu")
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

    # ── 6. Synthesize audio ──
    with torch.no_grad():
        audio_ref, _ = vocoder.inference(ref_mel.float(), finalize=True)
        audio_ref = audio_ref.squeeze(1)
        print(f"Reference audio shape: {audio_ref.shape}")
        print(f"Reference audio stats: min={audio_ref.min():.4f}, max={audio_ref.max():.4f}")
        torchaudio.save("reference_output.wav", audio_ref, 24000)
        print(f"✅ Reference output saved to reference_output.wav")
        print(f"Audio length: {audio_ref.shape[1] / 24000:.2f}s")

    # ── 7. Also test with flow_prompt_speech_token vs llm_prompt_speech_token ──
    print(f"\n=== Diagnostic: Token comparison ===")
    print(f"llm_prompt_speech_token shape: {prompt_speech_token.shape}")
    print(f"flow_prompt_speech_token shape: {flow_prompt_token.shape}")
    if prompt_speech_token.shape == flow_prompt_token.shape:
        match = (prompt_speech_token == flow_prompt_token).all().item()
        print(f"Tokens match: {match}")
    else:
        print("Token shapes differ!")
    print(f"llm_embedding shape: {embedding.shape}")
    print(f"flow_embedding shape: {flow_embedding.shape}")
    if embedding.shape == flow_embedding.shape:
        diff = (embedding - flow_embedding).abs().max().item()
        print(f"Embedding max diff: {diff:.6f}")
    else:
        print("Embedding shapes differ!")


if __name__ == "__main__":
    main()
