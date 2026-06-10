# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Demo CLI for the dots.ocr ``ocr`` use case: TTNN and HF side-by-side.

Runs the TTNN pipeline (tt/ocr_model.py on a 1x4 mesh) and, by default,
the HF reference (DotsOCRForCausalLM, bf16 CPU — the checkpoint's
intended precision; its vision tower hardcodes a bf16 cast) on the same
image, printing both outputs for eyeball parity.

Example:
    python demo/demo_ocr.py --image demo/inputs/invoice_total.png
"""

import importlib.util
import json
import sys
from pathlib import Path

import typer

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
MODEL_ID = "rednote-hilab/dots.ocr"
DEFAULT_PROMPT = "Extract the text content from the image."

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_by_path(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def snapshot_dir() -> Path:
    """Local HF snapshot (shared with tt/weight_loader.py, plus the remote code)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_ID))


def load_host_processors(snap: Path | None = None):
    """(tokenizer, image_processor, chat_template) for the checkpoint.

    The checkpoint's DotsVLProcessor subclass predates transformers'
    mandatory video_processor argument, so the tokenizer + Qwen2VL image
    processor are loaded directly and the single ``<|imgpad|>`` emitted
    by the chat template is expanded manually (one per merged patch) —
    byte-identical input_ids to the original processor.
    """
    from transformers import AutoImageProcessor, AutoTokenizer

    snap = snap or snapshot_dir()
    tokenizer = AutoTokenizer.from_pretrained(snap)
    image_processor = AutoImageProcessor.from_pretrained(snap)
    chat_template = json.load(open(snap / "chat_template.json"))["chat_template"]
    return tokenizer, image_processor, chat_template


def run_hf_reference(image_paths: list[str], prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 32) -> list[str]:
    """Greedy DotsOCRForCausalLM outputs for each image (bf16 CPU, num_beams=1).

    Loads and frees the HF model inside the call so the ~6GB resident
    does not overlap the TTNN run. The e2e test imports this helper.
    """
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM

    snap = snapshot_dir()
    tokenizer, image_processor, chat_template = load_host_processors(snap)
    model = AutoModelForCausalLM.from_pretrained(
        snap, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    assert type(model).__name__ == "DotsOCRForCausalLM", type(model).__name__
    model.eval()

    outputs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = tokenizer.apply_chat_template(
            messages, chat_template=chat_template, add_generation_prompt=True, tokenize=False
        )
        vis = image_processor(images=[image], return_tensors="pt")
        t, h, w = vis["image_grid_thw"][0].tolist()
        text = text.replace("<|imgpad|>", "<|imgpad|>" * ((t * h * w) // 4))
        enc = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                pixel_values=vis["pixel_values"].bfloat16(),
                image_grid_thw=vis["image_grid_thw"],
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
            )
        gen = out[0][enc["input_ids"].shape[1] :]
        outputs.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    del model
    return outputs


@app.command()
def main(
    image: str = typer.Option(..., "--image", help="Path to the document image"),
    prompt: str = typer.Option(DEFAULT_PROMPT, "--prompt"),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens"),
    device: str = typer.Option("qb", "--device", help="Informational; the 1x4 mesh is opened locally"),
    skip_hf: bool = typer.Option(False, "--skip-hf", help="Skip the side-by-side HF reference run"),
):
    from PIL import Image

    import ttnn

    ocr_mod = _load_by_path("dots_ocr_tt_ocr_model", MODEL_DIR / "tt" / "ocr_model.py")
    tokenizer, image_processor, chat_template = load_host_processors()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    try:
        model = ocr_mod.TtOCRModel(mesh_device, tokenizer, image_processor, chat_template)
        ttnn_text = model.ocr(Image.open(image).convert("RGB"), prompt=prompt, max_new_tokens=max_new_tokens)
    finally:
        ttnn.close_mesh_device(mesh_device)

    typer.echo(f"Image:  {image}  (device: {device})")
    typer.echo(f"TTNN:   {ttnn_text}")
    if not skip_hf:
        hf_text = run_hf_reference([image], prompt=prompt, max_new_tokens=max_new_tokens)[0]
        typer.echo(f"HF:     {hf_text}")


if __name__ == "__main__":
    app()
