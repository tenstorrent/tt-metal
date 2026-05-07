# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice3 Zero-Shot TTS Inference Script using Tenstorrent Wormhole.

This script uses the reference CosyVoice frontend to process text and prompt audio,
then uses our TTNN pipeline (LLM on mesh + Flow on device + Vocoder on host)
to generate the cloned speech.
"""

import argparse
import os
import sys

import soundfile as sf
import torchaudio


def custom_torchaudio_load(filepath, **kwargs):
    audio, sr = sf.read(filepath)
    if len(audio.shape) == 1:
        tensor = torch.tensor(audio).unsqueeze(0).float()
    else:
        tensor = torch.tensor(audio).transpose(0, 1).float()
    return tensor, sr


def custom_torchaudio_save(filepath, tensor, sample_rate, **kwargs):
    # tensor shape [C, T], soundfile expects [T, C]
    audio = tensor.transpose(0, 1).cpu().numpy()
    sf.write(filepath, audio, sample_rate)


torchaudio.load = custom_torchaudio_load
torchaudio.save = custom_torchaudio_save
import torch

import ttnn

# Ensure reference code is in path
sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice")
sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot TTS with CosyVoice3 on Wormhole")
    parser.add_argument(
        "--text",
        type=str,
        default="八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
        help="Target text to synthesize",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        help="Text spoken in the prompt audio",
    )
    parser.add_argument(
        "--prompt_wav",
        type=str,
        default="models/demos/wormhole/cosy_voice/ref/CosyVoice/asset/zero_shot_prompt.wav",
        help="Path to the prompt audio file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path",
    )
    args = parser.parse_args()

    weights_dir = "models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"

    # 1. Initialize the reference frontend (for text processing & audio feature extraction)
    print("Loading frontend...")
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

    # 2. Process inputs
    print(f"Processing prompt audio: {args.prompt_wav}")
    prompt_speech_16k = load_wav(args.prompt_wav, 16000)

    print("Running frontend processing...")
    # The frontend expects the path to the wav file, not the loaded tensor
    model_input = frontend.frontend_zero_shot(args.text, args.prompt_text, args.prompt_wav, 16000, "")

    text_token = model_input["text"]
    prompt_text_token = model_input["prompt_text"]
    prompt_speech_token = model_input["llm_prompt_speech_token"]
    prompt_feat = model_input["prompt_speech_feat"]
    embedding = model_input["llm_embedding"]

    print(f"Target text tokens: {text_token.shape[1]}")
    print(f"Prompt text tokens: {prompt_text_token.shape[1]}")
    print(f"Prompt speech tokens: {prompt_speech_token.shape[1]}")

    # 3. Load Vocoder (host CPU)
    print("Loading vocoder on host CPU...")
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

    # 4. Run LLM Stage (mesh device)
    print("\n=== Stage 1: LLM (mesh device) ===")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
    )

    try:
        from models.demos.wormhole.cosy_voice.tt.cosyvoice_llm import CosyVoice3LM
        from models.demos.wormhole.cosy_voice.tt.model_config import CosyVoiceModelConfig

        config = CosyVoiceModelConfig(
            mesh_device=mesh,
            max_batch_size=1,
            weights_dir=weights_dir,
        )
        llm = CosyVoice3LM(config, mesh, dtype=ttnn.bfloat8_b)

        speech_token_list = []
        token_gen = llm.inference(
            text=text_token,
            text_len=torch.tensor([text_token.shape[1]], dtype=torch.int32),
            prompt_text=prompt_text_token,
            prompt_text_len=torch.tensor([prompt_text_token.shape[1]], dtype=torch.int32),
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=torch.tensor([prompt_speech_token.shape[1]], dtype=torch.int32),
            embedding=embedding,
        )
        for tok in token_gen:
            speech_token_list.append(tok)
            # Optional: print tokens as they generate
            # print(f" {tok}", end="", flush=True)

        print(f"LLM generated {len(speech_token_list)} speech tokens")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    if not speech_token_list:
        print("Error: No speech tokens generated!")
        return

    llm_tokens = speech_token_list
    token = torch.tensor([llm_tokens], dtype=torch.int32)
    print(f"LLM generated {token.shape[1]} speech tokens: {llm_tokens}")

    # 5. Run Flow Decoder Stage (single device)
    print("\n=== Stage 2: Flow Decoder (single device) ===")
    from models.demos.wormhole.cosy_voice.tt.flow.flow import TtCausalMaskedDiffWithDiT

    device = ttnn.open_device(device_id=0)
    try:
        flow_sd = torch.load(os.path.join(weights_dir, "flow.pt"), map_location="cpu")
        flow = TtCausalMaskedDiffWithDiT(device, flow_sd, dtype=ttnn.bfloat16)

        speech_tokens = torch.tensor(speech_token_list).unsqueeze(0).to(torch.int32)

        mel, _ = flow.inference(
            token=speech_tokens,
            token_len=torch.tensor([speech_tokens.shape[1]], dtype=torch.int32),
            prompt_token=prompt_speech_token.to(torch.int32),
            prompt_token_len=torch.tensor([prompt_speech_token.shape[1]], dtype=torch.int32),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32),
            embedding=embedding,
            streaming=False,
            finalize=True,
        )
        print(f"Flow output: mel shape={mel.shape}")
        print(f"Mel stats: min={mel.min().item():.4f}, max={mel.max().item():.4f}, mean={mel.mean().item():.4f}")

        # 6. Run Vocoder Stage
        print("\n=== Stage 3: Vocoder (host CPU) ===")

        with torch.no_grad():
            audio, s = vocoder.inference(mel.float(), finalize=True)
            audio = audio.squeeze(1)

            print(f"Generated audio shape: {audio.shape}")
            print(
                f"Audio stats: min={audio.min().item():.4f}, max={audio.max().item():.4f}, mean={audio.mean().item():.4f}"
            )
            print(f"Source stats: min={s.min().item():.4f}, max={s.max().item():.4f}, mean={s.mean().item():.4f}")
        print(f"Audio length: {audio.shape[1] / 24000:.2f} seconds")

        # 7. Save output
        torchaudio.save(args.output, audio, 24000)
        print(f"\n✅ Synthesized speech saved to {args.output}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
