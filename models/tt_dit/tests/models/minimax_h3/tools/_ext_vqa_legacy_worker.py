# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Worker for DOVER, FAST-VQA, MaxVQA, and Flow Score. Runs under ~/ext_vqa_legacy_env
(torch~=1.13, numpy<2) -- DOVER's Swin backbone and MaxVQA's DOVER-derived visual
encoder are pinned to that generation; see ext_vqa_score.py's docstring for env setup.

Invoked by ext_vqa_score.py as a subprocess; not meant to be run standalone except
for debugging one metric in isolation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.expanduser("~/ext_vqa_repos")
DOVER_DIR = os.path.join(REPO_ROOT, "DOVER")
FASTVQA_DIR = os.path.join(REPO_ROOT, "FAST-VQA-and-FasterVQA")
MAXVQA_DIR = os.path.join(REPO_ROOT, "ExplainableVQA")


def score_dover(video_path: str, device: str) -> dict:
    """DOVER's disentangled aesthetic/technical heads -- this split is exactly VQA_A/VQA_T."""
    import numpy as np
    import torch
    import yaml
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
    from dover.models import DOVER

    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])

    with open(os.path.join(DOVER_DIR, "dover.yml")) as f:
        opt = yaml.safe_load(f)

    evaluator = DOVER(**opt["model"]["args"]).to(device)
    evaluator.load_state_dict(
        torch.load(os.path.join(DOVER_DIR, "pretrained_weights", "DOVER.pth"), map_location=device)
    )
    evaluator.eval()

    dopt = opt["data"]["val-l1080p"]["args"]
    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            temporal_samplers[stype] = UnifiedFrameSampler(sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])
        else:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"], sopt["t_frag"], sopt["frame_interval"], sopt["num_clips"]
            )

    views, _ = spatial_temporal_view_decomposition(video_path, dopt["sample_types"], temporal_samplers)
    for k, v in views.items():
        num_clips = dopt["sample_types"][k].get("num_clips", 1)
        views[k] = (
            ((v.permute(1, 2, 3, 0) - mean) / std)
            .permute(3, 0, 1, 2)
            .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
            .transpose(0, 1)
        ).to(device)

    with torch.no_grad():
        raw = evaluator(views)
    scores = dict(zip(views.keys(), (r.mean().item() for r in raw)))
    technical, aesthetic = scores["technical"], scores["aesthetic"]

    # Score-level fusion constants published alongside the DOVER checkpoint (evaluate_one_video.py::fuse_results).
    x = (technical - 0.1107) / 0.07355 * 0.6104 + (aesthetic + 0.08285) / 0.03774 * 0.3896
    return {"vqa_t": technical, "vqa_a": aesthetic, "fused_0_1": float(1 / (1 + np.exp(-x)))}


def score_fastvqa(video_path: str, device: str, model_name: str = "FAST-VQA") -> dict:
    import decord
    import numpy as np
    import torch
    import yaml
    from fastvqa.datasets import FragmentSampleFrames, SampleFrames, get_spatial_fragments
    from fastvqa.models import DiViDeAddEvaluator

    # Published per-model sigmoid-rescale constants (timothyhtimothy/FAST-VQA-and-FasterVQA :: vqa.py).
    mean_stds = {
        "FasterVQA": (0.14759505, 0.03613452),
        "FasterVQA-MS": (0.15218826, 0.03230298),
        "FasterVQA-MT": (0.14699507, 0.036453716),
        "FAST-VQA": (-0.110198185, 0.04178565),
        "FAST-VQA-M": (0.023889644, 0.030781006),
    }
    opts = {
        "FasterVQA": "options/fast/f3dvqa-b.yml",
        "FasterVQA-MS": "options/fast/fastervqa-ms.yml",
        "FasterVQA-MT": "options/fast/fastervqa-mt.yml",
        "FAST-VQA": "options/fast/fast-b.yml",
        "FAST-VQA-M": "options/fast/fast-m.yml",
    }
    with open(os.path.join(FASTVQA_DIR, opts[model_name])) as f:
        opt = yaml.safe_load(f)

    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)
    evaluator.load_state_dict(
        torch.load(os.path.join(FASTVQA_DIR, opt["test_load_path"]), map_location=device)["state_dict"]
    )
    evaluator.eval()

    video_reader = decord.VideoReader(video_path)
    s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]

    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])
    vsamples = {}
    for sample_type, sample_args in s_data_opt.items():
        if sample_args.get("t_frag", 1) > 1:
            sampler = FragmentSampleFrames(
                fsize_t=sample_args["clip_len"] // sample_args["t_frag"],
                fragments_t=sample_args["t_frag"],
                num_clips=sample_args.get("num_clips", 1),
            )
        else:
            sampler = SampleFrames(clip_len=sample_args["clip_len"], num_clips=sample_args["num_clips"])

        num_clips = sample_args.get("num_clips", 1)
        frames = sampler(len(video_reader))
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
        video = torch.stack([frame_dict[idx] for idx in frames], 0).permute(3, 0, 1, 2)

        sampled = get_spatial_fragments(video, **sample_args)
        sampled = ((sampled.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        sampled = sampled.reshape(sampled.shape[0], num_clips, -1, *sampled.shape[2:]).transpose(0, 1)
        vsamples[sample_type] = sampled.to(device)

    with torch.no_grad():
        raw = evaluator(vsamples).mean().item()
    m, s = mean_stds[model_name]
    return {"raw": raw, "score_0_1": float(1 / (1 + np.exp(-(raw - m) / s))), "model": model_name}


def score_maxvqa(video_path: str, device: str) -> dict:
    """MaxVQA's 16 antonym-pair dimensions (VQAssessment/ExplainableVQA), incl. clear-motion/blurry-motion."""
    sys.path.insert(0, MAXVQA_DIR)
    cwd = os.getcwd()
    # maxvqa.yml / maxvqa_maxwell.pt / the "../DOVER" backbone path are all cwd-relative in this repo.
    os.chdir(MAXVQA_DIR)
    try:
        import open_clip
        import torch
        import yaml
        from dover import DOVER
        from dover import datasets as dover_datasets
        from model import EnhancedVisualEncoder, MaxVQA, TextEncoder

        with open("maxvqa.yml") as f:
            opt = yaml.safe_load(f)
        dopt = opt["data"]["val-ytugc"]["args"]

        positive_descs = [
            "high quality",
            "good content",
            "organized composition",
            "vibrant color",
            "contrastive lighting",
            "consistent trajectory",
            "good aesthetics",
            "sharp",
            "in-focus",
            "noiseless",
            "clear-motion",
            "stable",
            "well-exposed",
            "original",
            "fluent",
            "clear",
        ]
        negative_descs = [
            "low quality",
            "bad content",
            "chaotic composition",
            "faded color",
            "gloomy lighting",
            "incoherent trajectory",
            "bad aesthetics",
            "fuzzy",
            "out-of-focus",
            "noisy",
            "blurry-motion",
            "shaky",
            "poorly-exposed",
            "compressed",
            "choppy",
            "severely degraded",
        ]
        prompts = [f"a X {desc} photo" for desc in positive_descs + negative_descs]

        clip_model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
        clip_model = clip_model.to(device)
        tokenizer = open_clip.get_tokenizer("RN50")
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(text_tokens)

        dover_backbone = DOVER(**opt["model"]["args"]).to(device)
        dover_backbone.load_state_dict(
            torch.load("../DOVER/pretrained_weights/DOVER.pth", map_location=device), strict=False
        )

        text_encoder = TextEncoder(clip_model)
        visual_encoder = EnhancedVisualEncoder(clip_model, dover_backbone)
        maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)
        maxvqa.load_state_dict(torch.load("maxvqa_maxwell.pt", map_location=device))
        maxvqa.initialize_inference(text_encoder)
        maxvqa.eval()

        # anno_file accepts a Python list directly, bypassing the labels-file/data_prefix path used for benchmark sets.
        dataset = dover_datasets.ViewDecompositionDataset(
            dict(dopt, anno_file=[dict(filename=os.path.abspath(video_path), label=-1)])
        )
        data = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))

        with torch.no_grad():
            vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
            res = maxvqa(vis_feats, text_encoder, train=False).cpu().numpy()[0]

        return {
            f"{pos}_vs_{neg}".replace(" ", "-"): float(v) for pos, neg, v in zip(positive_descs, negative_descs, res)
        }
    finally:
        os.chdir(cwd)


def score_flow(video_path: str, device: str, max_pairs: int = 32) -> dict:
    """Mean RAFT optical-flow magnitude across a subsample of consecutive frame pairs (motion strength)."""
    import decord
    import numpy as np
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device)
    model.eval()
    transforms = weights.transforms()

    # DOVER/FAST-VQA leave decord's global bridge set to "torch" in this same process; force
    # it back so vr[i] gives ndarrays, not tensors missing .asnumpy().
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    n = len(vr)
    if n < 2:
        return {"error": "not enough frames for optical flow"}
    stride = max(1, (n - 1) // max_pairs)
    idxs = list(range(0, n - 1, stride))[:max_pairs]

    magnitudes = []
    with torch.no_grad():
        for i in idxs:
            f1 = torch.from_numpy(vr[i].asnumpy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            f2 = torch.from_numpy(vr[i + 1].asnumpy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            f1, f2 = transforms(f1, f2)
            flow = model(f1.to(device), f2.to(device))[-1]
            magnitudes.append(flow.norm(dim=1).mean().item())

    return {"mean_flow_magnitude": float(np.mean(magnitudes)), "num_frame_pairs": len(magnitudes)}


DISPATCH = {"dover": score_dover, "fastvqa": score_fastvqa, "maxvqa": score_maxvqa, "flow": score_flow}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    results = {}
    for name in (n.strip() for n in args.metrics.split(",")):
        if not name:
            continue
        try:
            results[name] = DISPATCH[name](args.video, args.device)
        except Exception as exc:  # noqa: BLE001 - one metric's model/checkpoint failure shouldn't drop the rest
            results[name] = {"error": repr(exc)}

    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
