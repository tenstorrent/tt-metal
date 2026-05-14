import os
import sys

import soundfile as sf
import torch
import torchaudio


def custom_torchaudio_load(filepath, **kwargs):
    audio, sr = sf.read(filepath)
    if len(audio.shape) == 1:
        tensor = torch.tensor(audio).unsqueeze(0).float()
    else:
        tensor = torch.tensor(audio).transpose(0, 1).float()
    return tensor, sr


torchaudio.load = custom_torchaudio_load
sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice")
sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from hyperpyyaml import load_hyperpyyaml

weights_dir = "models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"
with open(os.path.join(weights_dir, "cosyvoice3.yaml"), "r") as f:
    fe_configs = load_hyperpyyaml(
        f,
        overrides={
            "llm": None,
            "flow": None,
            "hift": None,
            "qwen_pretrain_path": os.path.join(weights_dir, "CosyVoice-BlankEN"),
        },
    )
frontend = CosyVoiceFrontEnd(
    fe_configs["get_tokenizer"],
    fe_configs["feat_extractor"],
    os.path.join(weights_dir, "campplus.onnx"),
    os.path.join(weights_dir, "speech_tokenizer_v3.onnx"),
    os.path.join(weights_dir, "spk2info.pt"),
    fe_configs["allowed_special"],
)

model_input = frontend.frontend_zero_shot(
    "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    "models/demos/wormhole/cosy_voice/ref/CosyVoice/asset/zero_shot_prompt.wav",
    16000,
    "",
)

with open(os.path.join(weights_dir, "cosyvoice3.yaml"), "r") as f:
    llm_configs = load_hyperpyyaml(
        f,
        overrides={
            "flow": None,
            "hift": None,
            "qwen_pretrain_path": os.path.join(weights_dir, "CosyVoice-BlankEN"),
        },
    )
ref_llm = llm_configs["llm"]
llm_sd = torch.load(os.path.join(weights_dir, "llm.pt"), map_location="cpu", weights_only=True)
ref_llm.load_state_dict(llm_sd)
ref_llm.eval()

with torch.no_grad():
    token_gen = ref_llm.inference(
        text=model_input["text"],
        text_len=model_input["text_len"],
        prompt_text=model_input["prompt_text"],
        prompt_text_len=model_input["prompt_text_len"],
        prompt_speech_token=model_input["llm_prompt_speech_token"],
        prompt_speech_token_len=model_input["llm_prompt_speech_token_len"],
        embedding=model_input["llm_embedding"],
    )
    tokens = list(token_gen)
    print(f"Ref LLM generated {len(tokens)} tokens:")
    print(tokens)
