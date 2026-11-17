# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal golden: the fp32 reference over all layers for prompts with an image or an audio clip.

Like make_golden.py, but the prompts go through vLLM's MiMo-V2-Omni processor (placeholder expansion, pixels, mel)
and start from ``inputs_embeds``: the token embeddings with each item's placeholder rows replaced by the item's
embeddings from the fp32 reference encoders (reference/vision.py with vLLM's RMSNorm merger, reference/audio.py),
rounded to bf16 as the device receives them. A text-only prompt of similar length is the control: it shows what the
same metrics look like without multimodal rows. Layers are streamed one at a time, every prompt through a layer
before the next loads.

Saved per prompt: the processor's token ids and fields, the multimodal rows, ``inputs_embeds`` (bf16), the hidden
state after every layer (bf16, [num_layers + 1, S, hidden]), the logits of the last ``--tail`` positions (fp32) and the
greedy next token. The device test is tests/test_model_mm_tp8.py.

    python -m models.experimental.mimo_v2_6_flash.reference.make_golden_mm --audio speech_5s.wav --out golden_mm.pt
"""

import argparse
import time

import torch
from loguru import logger

from models.experimental.mimo_v2_6_flash.reference import model as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig

IMAGE_TOKEN, AUDIO_TOKEN = 151655, 151669
PLACEHOLDER = {
    "image": "<|vision_start|><|image_pad|><|vision_end|>",
    "audio": "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>",
}


def chat(placeholder, question):
    """One user turn and the non-thinking assistant prefix, as the checkpoint's chat template renders them."""
    return f"<|im_start|>user\n{placeholder}{question}<|im_end|><|im_start|>assistant\n<think></think>"


# The control prompt: plain text of about the image prompt's length (no multimodal rows).
CONTROL_TEXT = (
    "Read the passage and answer the question.\n\n"
    "The water cycle describes how water moves through the Earth system. Heat from the Sun evaporates water from "
    "oceans, lakes and rivers, and plants release more water vapour through their leaves in a process called "
    "transpiration. As warm, moist air rises it cools, and the vapour condenses on tiny particles of dust and salt to "
    "form the droplets that make up clouds. When the droplets merge and grow heavy enough, they fall back to the "
    "surface as rain, snow, sleet or hail. Some of this precipitation soaks into the ground and recharges aquifers, "
    "some is stored for months or years in glaciers and snow packs, and the rest runs off the land into streams that "
    "carry it back to the sea. Over long periods the cycle also shapes the landscape: rivers carve valleys, glaciers "
    "grind mountains into sediment, and groundwater slowly dissolves limestone to create caves. Because the total "
    "amount of water on the planet stays nearly constant, the same molecules have been recycled countless times "
    "since the oceans first formed, which is why the water in a glass today may once have fallen as rain on a "
    "dinosaur.\n\nQuestion: What process releases water vapour from plants? Answer in one sentence."
)


class ReferenceEncoders:
    """The fp32 reference encoders as an EncoderBackend (tt/multimodal.py)."""

    def __init__(self, model_path):
        from models.experimental.mimo_v2_6_flash.reference.audio import MiMoAudioReference
        from models.experimental.mimo_v2_6_flash.reference.vision import MiMoVisionReference

        self.vision = MiMoVisionReference.from_model_path(model_path, merger_norm="rms")
        self.audio = MiMoAudioReference.from_model_path(model_path)

    @torch.no_grad()
    def embed_image(self, pixel_values, grid_thw):
        return self.vision.forward(pixel_values.float(), torch.as_tensor(grid_thw).reshape(-1, 3))

    embed_video = embed_image

    @torch.no_grad()
    def embed_audio(self, mel):
        mel = torch.as_tensor(mel).float()
        return self.audio.embed([mel[0] if mel.dim() == 3 else mel])[0]


def test_image():
    """A synthetic 448 x 448 picture (196 image tokens): a red disc and a yellow square on a blue background."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), (30, 60, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 120, 260, 320), fill=(220, 20, 20))
    draw.rectangle((290, 160, 400, 270), fill=(240, 220, 30))
    return img


def load_wav(path):
    import soundfile as sf

    wave, sr = sf.read(path, dtype="float32")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    return torch.from_numpy(wave), sr


def build_prompts(model_path, image_path=None, audio_path=None):
    """[(name, text, token ids, {field: tensor}, modality)] through vLLM's processor."""
    from PIL import Image
    from transformers import AutoConfig, AutoTokenizer
    from vllm.transformers_utils.processors.mimo_v2_omni import MiMoOmniProcessor

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    proc = MiMoOmniProcessor.from_hf_config(tok, AutoConfig.from_pretrained(model_path, trust_remote_code=True))
    out = []
    text = chat("", CONTROL_TEXT)
    out.append(("text", text, torch.tensor(tok(text)["input_ids"]), None, None))
    img = Image.open(image_path).convert("RGB") if image_path else test_image()
    text = chat(PLACEHOLDER["image"], "What is in this image? Answer in one sentence.")
    p = proc(text=text, images=[img], return_tensors="pt")
    fields = {"pixel_values": p["pixel_values"], "image_grid_thw": p["image_grid_thw"][0]}
    out.append(("image", text, p["input_ids"][0], fields, "image"))
    if audio_path:
        text = chat(PLACEHOLDER["audio"], "Transcribe the audio.")
        p = proc(text=text, audio=[load_wav(audio_path)], return_tensors="pt")
        out.append(("audio", text, p["input_ids"][0], {"audio_features": p["audio_features"][0]}, "audio"))
    return out


def main():
    from models.experimental.mimo_v2_6_flash.tt import multimodal as mm

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--out", required=True)
    parser.add_argument("--image", default=None, help="an image file (default: a synthetic picture)")
    parser.add_argument("--audio", default=None, help="a wav file (speech) for the audio prompt")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--tail", type=int, default=8, help="positions whose logits are saved")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    cfg = MiMoConfig.from_model_path(args.model_path)
    ckpt = MiMoCheckpoint(args.model_path, cfg)
    num_layers = args.num_layers or cfg.num_hidden_layers
    prompts = build_prompts(args.model_path, args.image, args.audio)

    backend = ReferenceEncoders(args.model_path)
    embedding = ckpt.embedding()
    records = []
    for name, text, ids, fields, modality in prompts:
        if modality is None:  # the text-only control
            embeds = embedding[ids].to(torch.bfloat16)
            mask = torch.zeros(ids.numel(), dtype=torch.bool)
            logger.info(f"{name}: {ids.numel()} tokens, text only")
            records.append(
                dict(
                    name=name,
                    text=text,
                    token_ids=ids,
                    modality=None,
                    offset=0,
                    length=0,
                    fields=None,
                    mm_rows=mask,
                    inputs_embeds=embeds,
                )
            )
            continue
        token = IMAGE_TOKEN if modality == "image" else AUDIO_TOKEN
        (offset, length), *rest = mm.placeholder_rows(ids, token)
        assert not rest, f"{name}: one item per prompt"
        item = mm.MMItem(modality, name, offset, length, None, fields)
        t0 = time.time()
        rows = mm.encode_item(backend, item)  # bf16, as the device gets them
        logger.info(
            f"{name}: {ids.numel()} tokens, {modality} rows [{offset}, {offset + length}) in {time.time() - t0:.1f}s"
        )
        embeds = embedding[ids].to(torch.bfloat16)
        embeds[offset : offset + length] = rows
        mask = torch.zeros(ids.numel(), dtype=torch.bool)
        mask[offset : offset + length] = True
        records.append(
            dict(
                name=name,
                text=text,
                token_ids=ids,
                modality=modality,
                offset=offset,
                length=length,
                fields=fields,
                mm_rows=mask,
                inputs_embeds=embeds,
            )
        )
    del embedding

    hidden = [r["inputs_embeds"].float() for r in records]
    per_layer = [[h.to(torch.bfloat16)] for h in hidden]
    for layer_idx in range(num_layers):
        t0 = time.time()
        layer = ref.ReferenceDecoderLayer(cfg, layer_idx, ckpt=ckpt)
        t1 = time.time()
        with torch.no_grad():
            for i, r in enumerate(records):
                hidden[i] = layer(hidden[i], torch.arange(r["token_ids"].numel()))
                per_layer[i].append(hidden[i].to(torch.bfloat16))
        del layer
        logger.info(f"layer {layer_idx}: load {t1 - t0:.1f}s, compute {time.time() - t1:.1f}s")

    norm = ckpt.final_norm()
    lm_head = ckpt.lm_head().float()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    with torch.no_grad():
        for i, r in enumerate(records):
            lg = ref.rms_norm(hidden[i][-args.tail :], norm, cfg.layernorm_epsilon) @ lm_head.T
            r["logits_tail"] = lg
            r["hidden"] = torch.stack(per_layer[i])
            r["next_token"] = int(lg[-1].argmax())
            top = lg[-1].topk(5)
            logger.info(
                f"{r['name']}: next token {tok.decode([r['next_token']])!r}; top-5 "
                + ", ".join(f"{tok.decode([int(t)])!r} {float(v):.2f}" for v, t in zip(top.values, top.indices))
            )
    torch.save(
        {
            "prompts": records,
            "num_layers": num_layers,
            "tail": args.tail,
            "encoder": "reference/vision.py (merger rms) and reference/audio.py, float32",
        },
        args.out,
    )
    logger.info(f"saved {args.out}")


if __name__ == "__main__":
    main()
