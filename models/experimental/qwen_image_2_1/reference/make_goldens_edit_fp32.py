# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Independent fp32 CPU reference for the image-conditioned text encoder.

Reads the exact processor inputs saved by `make_goldens_edit.py`, then records encoder hidden states
and vision features. Compare both precision policies when the pipeline goldens use bf16: rounding in
the vision tower can accumulate through the language decoder. Run in the reference environment:

    python -m models.experimental.qwen_image_2_1.reference.make_goldens_edit_fp32
"""
import argparse
import os
import time

import torch

from models.experimental.qwen_image_2_1.common.config import GOLDENS_DIR, snapshot_dir


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=os.path.join(GOLDENS_DIR, "edit", "text_encoder_edit.pt"))
    ap.add_argument("--out", default=os.path.join(GOLDENS_DIR, "edit", "text_encoder_edit_fp32.pt"))
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))
    g = torch.load(args.golden, weights_only=False, map_location="cpu")

    t0 = time.time()
    m = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(snapshot_dir(), "text_encoder"), dtype=torch.float32, device_map="cpu"
    ).eval()
    print(f"loaded the fp32 model in {time.time()-t0:.0f}s")

    vis = {}
    hooks = [
        # the pipeline's identity hook: hidden_states[-1] is then the last layer BEFORE the final norm
        m.model.language_model.norm.register_forward_hook(lambda mod, a, o: a[0]),
        m.model.visual.register_forward_hook(lambda mod, a, o: vis.__setitem__("out", o)),
    ]
    t0 = time.time()
    with torch.no_grad():
        out = m(
            input_ids=g["input_ids"],
            attention_mask=g["attention_mask"],
            pixel_values=g["pixel_values"].float(),
            image_grid_thw=g["image_grid_thw"],
            mm_token_type_ids=g["mm_token_type_ids"],
            output_hidden_states=True,
        )
    for h in hooks:
        h.remove()
    print(f"fp32 forward {time.time()-t0:.0f}s")

    v = vis["out"]
    hs = [h.detach().cpu() for h in out.hidden_states]
    # `get_image_features` splits `pooler_output` per image AFTER the tower returns, and the hook holds
    # the same object, so by now it is a tuple of per-image blocks (one entry for a single image).
    pooler = v.pooler_output
    pooler = torch.cat(list(pooler), dim=0) if isinstance(pooler, (list, tuple)) else pooler
    torch.save(
        {
            "hidden_states": hs,
            "pooler_output": pooler.detach().cpu(),
            "deepstack_features": [f.detach().cpu() for f in v.deepstack_features],
            "last_hidden_state": v.last_hidden_state.detach().cpu(),
        },
        args.out,
    )

    gh = g["hidden_states"]
    print("\nthe pipeline golden measured against this fp32 reference:")
    print(f"  vision merged tokens    pcc = {pcc(g['visual_out']['pooler_output'][0], pooler):.5f}")
    for i in (0, 12, 24, 30, 35, 36):
        print(f"  hidden_states[{i:2d}]       pcc = {pcc(gh[i][0], hs[i][0]):.5f}")
    print(f"  prompt_embeds           pcc = {pcc(g['prompt_embeds'][0], hs[-1][0][14:]):.5f}")
    print("\nsaved ->", args.out)


if __name__ == "__main__":
    main()
