"""
Generate and save golden tensors for all blocks.
Used by TTNN PCC tests to compare against reference.

Run:
  cd tt-metal
  export VOXTRAL_MODEL_DIR=/path/to/model
  export PYTHONPATH=$(pwd):$(pwd)/models
  source python_env/bin/activate
  python3 models/demos/voxtral_tts/reference/save_goldens.py
"""

import os
import sys
from pathlib import Path

import torch

MODEL_DIR = Path(
    os.environ.get(
        "VOXTRAL_MODEL_DIR",
        "/home/ttuser/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/b81be46c3777f88621676791b512bb01dc1cb970",
    )
)
GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parents[3]))

from models.demos.voxtral_tts.reference.functional import (
    acoustic_transformer_step,
    build_rope_cache,
    codec_decoder_forward,
    ode_solve,
    text_decoder_forward,
    text_decoder_layer,
)
from models.demos.voxtral_tts.tt.load_checkpoint import (
    get_acoustic_transformer_state,
    get_codec_decoder_state,
    get_text_decoder_state,
    load_state_dict,
    load_voice_embeddings,
)

torch.manual_seed(42)


def main():
    print("Loading weights...")
    sd = load_state_dict(MODEL_DIR / "consolidated.safetensors")
    sd_text = get_text_decoder_state(sd)
    sd_acoustic = get_acoustic_transformer_state(sd)
    sd_codec = get_codec_decoder_state(sd)
    voices = load_voice_embeddings(MODEL_DIR)
    print("Weights loaded.")

    # ── Text decoder: per-layer golden inputs/outputs ──
    print("Generating text decoder goldens...")
    torch.manual_seed(42)
    B, S, D = 1, 64, 3072
    x = torch.randn(B, S, D, dtype=torch.bfloat16)
    cos, sin = build_rope_cache(S, 128, 1e6, "cpu")

    # Save layer 0 input and output
    for layer_idx in [0, 12, 25]:
        inp = x.clone()
        out, caps = text_decoder_layer(x, sd_text, layer_idx, cos, sin, capture_intermediates=True)

        torch.save(inp, GOLDEN_DIR / f"text_layer{layer_idx}_input.pt")
        torch.save(out, GOLDEN_DIR / f"text_layer{layer_idx}_output.pt")
        torch.save(caps["attn_attn_out"], GOLDEN_DIR / f"text_layer{layer_idx}_attn_out.pt")
        torch.save(caps["mlp_mlp_out"], GOLDEN_DIR / f"text_layer{layer_idx}_mlp_out.pt")

        x = out
        print(f"  Layer {layer_idx}: output shape {out.shape}")

    # ── Full text decoder forward ──
    print("Generating full text decoder golden...")
    torch.manual_seed(42)
    input_ids = torch.randint(0, 131072, (1, 32))
    h, _ = text_decoder_forward(input_ids, sd_text, capture_intermediates=False)
    torch.save(input_ids, GOLDEN_DIR / "text_input_ids.pt")
    torch.save(h, GOLDEN_DIR / "text_decoder_output.pt")
    print(f"  Full decoder output shape: {h.shape}")

    # ── Acoustic transformer step golden ──
    print("Generating acoustic transformer goldens...")
    torch.manual_seed(42)
    B, N = 1, 32
    h_ac = torch.randn(B, N, 3072, dtype=torch.bfloat16)
    x_t = torch.randn(B, N, 36, dtype=torch.bfloat16)
    t = 0.5

    v, sem_logits, caps = acoustic_transformer_step(h_ac, x_t, t, sd_acoustic, capture_intermediates=True)
    torch.save(h_ac, GOLDEN_DIR / "acoustic_h_input.pt")
    torch.save(x_t, GOLDEN_DIR / "acoustic_x_t_input.pt")
    torch.save(v, GOLDEN_DIR / "acoustic_velocity_output.pt")
    torch.save(sem_logits, GOLDEN_DIR / "acoustic_semantic_logits.pt")
    print(f"  Velocity shape: {v.shape}, sem_logits: {sem_logits.shape}")

    # ── ODE solve golden ──
    print("Generating ODE solve goldens...")
    torch.manual_seed(42)
    h_ode = torch.zeros(1, 10, 3072, dtype=torch.bfloat16)
    acoustic_codes, x_continuous, _ = ode_solve(h_ode, sd_acoustic, n_steps=8)
    torch.save(h_ode, GOLDEN_DIR / "ode_h_input.pt")
    torch.save(acoustic_codes, GOLDEN_DIR / "ode_acoustic_codes.pt")
    torch.save(x_continuous, GOLDEN_DIR / "ode_x_continuous.pt")
    print(f"  Acoustic codes shape: {acoustic_codes.shape}, x: {x_continuous.shape}")

    # ── Codec decoder golden ──
    print("Generating codec decoder goldens...")
    torch.manual_seed(42)
    B, N = 1, 50
    semantic_codes = torch.randint(0, 8192, (B, N))
    acoustic_codes_in = torch.randint(0, 21, (B, N, 36))
    waveform, caps = codec_decoder_forward(semantic_codes, acoustic_codes_in, sd_codec, capture_intermediates=True)

    torch.save(semantic_codes, GOLDEN_DIR / "codec_semantic_codes_input.pt")
    torch.save(acoustic_codes_in, GOLDEN_DIR / "codec_acoustic_codes_input.pt")
    torch.save(caps["block0_out"], GOLDEN_DIR / "codec_block0_out.pt")
    torch.save(waveform, GOLDEN_DIR / "codec_waveform_output.pt")
    print(f"  Waveform shape: {waveform.shape}, amplitude max: {waveform.abs().max():.4f}")

    # ── Voice embedding golden ──
    print("Generating voice embedding golden...")
    voice_emb = voices["casual_male"]
    torch.save(voice_emb, GOLDEN_DIR / "voice_casual_male.pt")
    print(f"  Voice embedding shape: {voice_emb.shape}")

    print(f"\nAll goldens saved to {GOLDEN_DIR}")
    print("Golden files:")
    for f in sorted(GOLDEN_DIR.glob("*.pt")):
        size = f.stat().st_size / 1024
        print(f"  {f.name}: {size:.1f} KB")


if __name__ == "__main__":
    main()
