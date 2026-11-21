# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Reference action tokens for tests/test_openvla_accuracy.py::test_predict_action_tokens.

Runs the Hugging Face OpenVLA model greedily on the test's solid-color images in bfloat16 on CPU and prints, for each
case, the action tokens and the top-5 token ids at every step. Run it in OpenVLA's pinned environment
(transformers 4.40.1, timm 0.9.10, tokenizers 0.19.1, torch 2.2.2) on a snapshot of openvla/openvla-7b revision
47a0ec7f, then paste the printed dictionary into REFERENCE_ACTIONS. One case takes about 8 minutes.

    python generate_reference_actions.py <path to the openvla-7b snapshot>
"""

import pprint
import sys

import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer

CASES = {
    "orange_red_block": {"rgb": (255, 100, 50), "instruction": "pick up the red block"},
    "blue_put_cup": {"rgb": (40, 90, 200), "instruction": "put the cup on the plate"},
}


def normalize_image(image):
    """The test's preprocessing: DINOv2 then SigLIP normalization, stacked on channels."""
    dinov2_mean = torch.tensor([0.484375, 0.455078125, 0.40625]).view(3, 1, 1)
    dinov2_std = torch.tensor([0.228515625, 0.2236328125, 0.224609375]).view(3, 1, 1)
    return torch.cat([(image - dinov2_mean) / dinov2_std, (image - 0.5) / 0.5]).unsqueeze(0)


def main(snapshot):
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForVision2Seq.from_pretrained(
        snapshot, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).eval()
    references = {}
    with torch.no_grad():
        for name, case in CASES.items():
            prompt = f"In: What action should the robot take to {case['instruction']}?\nOut:"
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            input_ids = torch.cat([input_ids, torch.tensor([[29871]])], dim=1)
            image = (torch.tensor(case["rgb"], dtype=torch.float32) / 255).view(3, 1, 1).expand(3, 224, 224)
            output = model.generate(
                input_ids,
                pixel_values=normalize_image(image).to(torch.bfloat16),
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=7,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            references[name] = {
                **case,
                "tokens": output.sequences[0, -7:].tolist(),
                "top5": [scores[0].float().topk(5).indices.tolist() for scores in output.scores],
            }
            print(f"{name}: {references[name]}", flush=True)
    pprint.pprint(references, width=120, sort_dicts=False)


if __name__ == "__main__":
    main(sys.argv[1])
