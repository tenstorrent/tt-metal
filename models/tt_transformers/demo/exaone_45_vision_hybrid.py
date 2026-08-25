# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""EXAONE-4.5 multimodal hybrid demo: host vision encoder + device text decoder.

Runs the EXAONE-4.5 vision tower (Qwen2.5-VL-style ViT) on the host CPU via HF
transformers (~0.5 s per image; lazy safetensors mmap loads only the vision
weights), then splices the merged image embeddings over the ``<|image_pad|>``
token positions of the chat-templated prompt and runs prefill + decode on the
TT text model (TP=8 across the mesh).

Mechanism: ``Generator.prefill_forward_text`` forwards ``**kwargs`` untouched
into ``Transformer.prepare_inputs_prefill``; the ``VisionHybridTransformer``
subclass masked-scatters the host embeddings right after the device embedding
lookup (the Gemma3/Mistral-3.1 fuse pattern, minus the on-device tower). Trace
must stay off for the vision prefill: the traced-prefill path whitelists
kwargs and would silently drop ``vision_embeddings``.

Validated 2026-08-25 on P150x8: prefill-logit PCC vs the full HF CPU forward
(image included) = 0.9956; generated description matches HF greedy output.

Usage:
    export HF_MODEL=LGAI-EXAONE/EXAONE-4.5-33B MESH_DEVICE=P150x8
    python models/tt_transformers/demo/exaone_45_vision_hybrid.py \
        [--image path.jpg] [--prompt "What is in this image?"] \
        [--max-new-tokens 200] [--enable-thinking]
"""
import argparse
import time

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs


class VisionHybridTransformer(Transformer):
    """Text Transformer that accepts host-encoded vision embeddings at prefill.

    ``vision_embeddings`` is a host torch tensor [n_image_tokens, hidden] whose
    rows replace the embeddings at the positions where the (padded) prompt has
    ``image_token_id``. Decode is unchanged.
    """

    def prepare_inputs_prefill(self, tokens, **kwargs):
        vision_embeddings = kwargs.pop("vision_embeddings", None)
        out = super().prepare_inputs_prefill(tokens, **kwargs)
        if vision_embeddings is None or kwargs.get("trace_enabled", False):
            return out

        image_token_id = self.args.hf_config.image_token_id
        tokens_embd = out[0]
        emb = ttnn.to_torch(
            tokens_embd, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        )  # [1, 1, S_padded, dim]
        mask = (tokens == image_token_id).view(1, 1, tokens.shape[-1], 1).expand_as(emb)
        n_slots, n_feats = mask[..., 0].sum().item(), vision_embeddings.shape[0]
        assert n_slots == n_feats, f"image token/feature mismatch: {n_slots} slots vs {n_feats} embeddings"
        emb = emb.masked_scatter(mask, vision_embeddings.to(emb.dtype))
        new_embd = self.args.prepare_residual_tensor_prefill(emb.squeeze(0))
        return (new_embd,) + out[1:]


def preprocess_image_prompt(hf_model_name, image, prompt_text, enable_thinking):
    """Processor on host. Returns the chat-templated inputs dict."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(hf_model_name)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}],
        }
    ]
    return proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )


def encode_image_host(hf_model_name, inputs):
    """Vision tower on host CPU. Returns merged image embeddings [n, out_hidden]."""
    from transformers import AutoModelForImageTextToText

    hf = AutoModelForImageTextToText.from_pretrained(hf_model_name, torch_dtype="auto")
    t0 = time.time()
    with torch.no_grad():
        vout = hf.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
    feats = torch.cat(vout.pooler_output, dim=0)
    logger.info(f"host vision encode: {time.time() - t0:.2f}s -> {tuple(feats.shape)}")
    del hf
    return feats


def encode_image_tt(mesh_device, inputs, debug=False):
    """Vision tower on device (models/demos/exaone45_vl). Returns [n, out_hidden]."""
    from models.demos.exaone45_vl.tt.model import DropInVisionTransformer
    from models.demos.exaone45_vl.tt.model_config import VisionModelArgs

    vargs = VisionModelArgs(mesh_device, max_batch_size=1, max_seq_len=2048)
    vision_ref = vargs.reference_vision_model()
    visual = DropInVisionTransformer(vision_ref, vargs, debug=debug)
    t0 = time.time()
    feats = visual(inputs["pixel_values"], inputs["image_grid_thw"])
    logger.info(f"TT vision encode (incl. weight load): {time.time() - t0:.2f}s -> {tuple(feats.shape)}")
    return feats.to(torch.bfloat16)


def main():
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="image path (default: synthetic shapes test image)")
    parser.add_argument("--prompt", default="What shapes and colors do you see in this image?")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--vision-device",
        choices=["host", "tt"],
        default="host",
        help="where the vision tower runs: host CPU (HF) or on-device (models/demos/exaone45_vl)",
    )
    parser.add_argument("--vision-debug", action="store_true", help="log TT-vision PCC vs the HF reference")
    cli = parser.parse_args()

    hf_model_name = os.environ["HF_MODEL"]

    if cli.image:
        from PIL import Image

        image = Image.open(cli.image).convert("RGB")
    else:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (448, 336), "white")
        d = ImageDraw.Draw(image)
        d.ellipse([40, 60, 200, 220], fill="red", outline="black", width=4)
        d.rectangle([260, 80, 400, 220], fill="blue", outline="black", width=4)

    inputs = preprocess_image_prompt(hf_model_name, image, cli.prompt, cli.enable_thinking)
    input_ids = inputs["input_ids"]

    feats = None
    if cli.vision_device == "host":
        feats = encode_image_host(hf_model_name, inputs)
        logger.info(f"prompt: {input_ids.shape[-1]} tokens ({feats.shape[0]} image tokens)")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, ttnn.GetNumAvailableDevices()),
        num_command_queues=1,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.device.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL),
    )
    try:
        if cli.vision_device == "tt":
            feats = encode_image_tt(mesh_device, inputs, debug=cli.vision_debug)
            logger.info(f"prompt: {input_ids.shape[-1]} tokens ({feats.shape[0]} image tokens)")
        paged_cfg = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
        args = ModelArgs(
            mesh_device,
            instruct=True,
            max_batch_size=1,
            optimizations=lambda ma: DecodersPrecision.performance(ma.n_layers, ma.model_name),
            max_seq_len=cli.max_seq_len,
        )
        model = VisionHybridTransformer(
            args=args,
            mesh_device=mesh_device,
            dtype=ttnn.bfloat8_b,
            state_dict=args.load_state_dict(),
            weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b),
            paged_attention_config=paged_cfg,
        )
        tt_kv_cache = [l.attention.layer_past for l in model.layers]
        page_table = torch.argsort(torch.randperm(paged_cfg.max_num_blocks)).reshape(1, paged_cfg.max_num_blocks)
        generator = Generator([model], [args], mesh_device, tokenizer=args.tokenizer)

        S = input_ids.shape[-1]
        tt_logits = generator.prefill_forward_text(
            input_ids,
            page_table=page_table,
            kv_cache=[tt_kv_cache],
            prompt_lens=[S],
            enable_trace=False,
            vision_embeddings=feats,
        )
        tt_logits = tt_logits.view(1, -1)[:, : args.vocab_size]

        cur_tok = torch.argmax(tt_logits, dim=-1, keepdim=True)
        current_pos = torch.tensor([S])
        generated = [cur_tok.item()]
        eos_id = args.tokenizer.eos_token_id
        t0 = time.time()
        for i in range(cli.max_new_tokens - 1):
            logits, _ = generator.decode_forward(
                cur_tok,
                current_pos,
                enable_trace=False,
                page_table=page_table,
                kv_cache=[tt_kv_cache],
                reset_batch=(i == 0),
                sampling_params=None,
            )
            cur_tok = torch.argmax(logits.view(1, -1)[:, : args.vocab_size], dim=-1, keepdim=True)
            if cur_tok.item() == eos_id:
                break
            generated.append(cur_tok.item())
            current_pos = current_pos + 1
        dt = time.time() - t0

        text = args.tokenizer.decode(generated, skip_special_tokens=True)
        print("=" * 70)
        print(text)
        print("=" * 70)
        logger.info(f"{len(generated)} tokens in {dt:.1f}s ({len(generated) / dt:.1f} tok/s)")
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
