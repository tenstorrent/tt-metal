# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SeamlessM4T-v2 T2TT (text-to-text translation) demo CLI.

Translates a single source sentence with the TTNN model and prints the
output, alongside the HuggingFace reference translation for comparison.

Usage::

    python models/demos/facebook_seamless_m4t_v2_large/demo/demo_t2tt.py \\
        --src "Hello world." \\
        --src-lang eng \\
        --tgt-lang fra

Args:
    --src         Source sentence to translate.
    --src-lang    Source language code (e.g. ``eng``).
    --tgt-lang    Target language code (e.g. ``fra``).
    --device      Informational (we always open device 0 on p150).
    --max-new-tokens   AR generation budget (incl. 2-token prefix).

Output::

    Source: Hello world.
    Translation: Salut à vous, monde.
    HF: Salut à vous, monde.
"""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _run_hf_reference(src: str, src_lang: str, tgt_lang: str, max_new_tokens: int) -> str:
    """Run HF ``SeamlessM4Tv2ForTextToText.generate`` and return the decoded string."""
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl

    proc = AutoProcessor.from_pretrained(wl.HF_PATH)
    toks = proc(text=src, src_lang=src_lang, return_tensors="pt")
    model = SeamlessM4Tv2ForTextToText.from_pretrained(wl.HF_PATH, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            tgt_lang=tgt_lang,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            num_beams=1,
        )
    if hasattr(out, "sequences"):
        out = out.sequences
    text = proc.decode(out[0].tolist(), skip_special_tokens=True)
    # Free HF model so we don't keep ~8GB resident alongside the TTNN model.
    del model
    return text


@app.command()
def main(
    src: str = typer.Option(..., "--src", help="Source sentence to translate."),
    src_lang: str = typer.Option(..., "--src-lang", help="Source language code (e.g. eng)."),
    tgt_lang: str = typer.Option(..., "--tgt-lang", help="Target language code (e.g. fra)."),
    device: str = typer.Option(
        "p150",
        "--device",
        help="Informational; we always open device 0 on the local p150.",
    ),
    max_new_tokens: int = typer.Option(
        128, "--max-new-tokens", help="AR generation budget (includes the 2-token prefix)."
    ),
    skip_hf: bool = typer.Option(False, "--skip-hf", help="Skip the HF reference run (TTNN only)."),
):
    """Translate ``--src`` with the TTNN SeamlessM4T-v2 model."""
    from transformers import AutoProcessor

    import ttnn
    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_to_text_model import TextToTextModel

    typer.echo(f"[demo] device-arg={device} (always uses device 0 on local p150)")
    typer.echo("[demo] loading HF checkpoint into host memory ...")
    hf_sd = wl.load_hf_state_dict()
    processor = AutoProcessor.from_pretrained(wl.HF_PATH)

    typer.echo("[demo] opening TTNN device 0 ...")
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        typer.echo("[demo] building TextToTextModel (24-layer encoder + 24-layer decoder + LM head) ...")
        model = TextToTextModel(
            device=dev,
            hf_state_dict=hf_sd,
            processor=processor,
        )
        typer.echo("[demo] running TTNN translate() ...")
        ttnn_out = model.translate(
            src_text=src,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_new_tokens=max_new_tokens,
        )
    finally:
        ttnn.close_device(dev)

    typer.echo("")
    typer.echo(f"Source: {src}")
    typer.echo(f"Translation: {ttnn_out}")

    if not skip_hf:
        typer.echo("[demo] running HF reference for comparison ...")
        hf_out = _run_hf_reference(src, src_lang, tgt_lang, max_new_tokens)
        typer.echo(f"HF: {hf_out}")


if __name__ == "__main__":
    app()
