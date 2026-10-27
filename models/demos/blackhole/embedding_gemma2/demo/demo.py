# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""EmbeddingGemma 2 demos on one Blackhole P150a (traced device forwards).

    pytest models/demos/blackhole/embedding_gemma2/demo/demo.py -s
    python models/demos/blackhole/embedding_gemma2/demo/demo.py            (same, as a script)
    python models/demos/blackhole/embedding_gemma2/demo/demo.py --no-hf    (skip the HF comparison)

test_demo             text retrieval with the text tower (`TextEncoder`)
test_multimodal_demo  the full model (`EmbeddingGemma2`, public API) on in-tree media:
                      text retrieval, image <-> text, audio <-> text, video <-> text.
                      Every embedding is also computed by HF fp32 (`EmbeddingGemma2Model` +
                      sentence-transformers pooling); the TT vs HF cosine is printed per input.

Prompts follow the model card (`config_sentence_transformers.json`): queries get
"task: search result | query: ", documents "title: none | text: ". Images, audio and video get no
prefix (model card, Best Practices 1).
"""

import sys

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.embedding_gemma2.tt.text_encoder import TextEncoder
from models.demos.blackhole.embedding_gemma2.tt.weights import resolve_snapshot

QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "
TRACE_REGION = 256 * 1024 * 1024

QUERIES = [
    "What causes the northern lights?",
    "How do I bake sourdough bread?",
    "Which planet is the largest in the solar system?",
]
DOCUMENTS = [
    "The northern lights are caused by charged particles from the sun hitting the upper atmosphere.",
    "Jupiter is the largest planet in our solar system, more than twice as massive as all others combined.",
    "Sourdough bread uses a fermented starter of flour and water instead of commercial yeast.",
    "The stock market closed higher on Tuesday after a strong earnings season.",
]


def run_demo(device, model_id="google/embeddinggemma-2"):
    from transformers import AutoTokenizer

    snapshot = resolve_snapshot(model_id)
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    enc = TextEncoder.from_pretrained(device, str(snapshot), max_seq_len=2048)

    batches = [
        tok([QUERY_PREFIX + t for t in QUERIES], padding=True, return_tensors="pt"),
        tok([DOC_PREFIX + t for t in DOCUMENTS], padding=True, return_tensors="pt"),
    ]
    # one trace per (batch, bucket), all captured before the first replay
    shapes = [(b["input_ids"].shape[0], enc.prepare_inputs(b["input_ids"]).seq_len) for b in batches]
    enc.capture_all(shapes)
    q, d = (enc.encode(b["input_ids"], b["attention_mask"], traced=True) for b in batches)
    scores = q @ d.T
    for i, query in enumerate(QUERIES):
        best = int(scores[i].argmax())
        logger.info(f"query: {query!r}\n    best doc ({scores[i, best]:.3f}): {DOCUMENTS[best]!r}")
    enc.release_traces()
    return scores


# ---------------------------------------------------------------------- multimodal demo
VIDEO_TEXTS = [
    "a screen recording of a software dashboard",
    "a dog running on a beach",
    "a person cooking pasta in a kitchen",
]


def multimodal_inputs():
    """name -> (rows, columns): rows are text queries, columns the inputs they are matched against."""
    from models.demos.blackhole.embedding_gemma2.reference import full_model_inputs as F

    images = {k: {"image": F.image(k)} for k in F.IMAGE_FILES}
    audios = {k: {"audio": {"array": F.audio(k), "sampling_rate": F.SR}} for k in F.AUDIO_CAPTIONS}
    video = {"dashboard_video": {"video": F.video()}}
    return {
        "text retrieval": (
            {q: QUERY_PREFIX + q for q in QUERIES},
            {d[:48]: DOC_PREFIX + d for d in DOCUMENTS},
        ),
        "text -> image": ({c: c for c in F.IMAGE_CAPTIONS.values()}, images),
        "text -> audio": ({c: c for c in F.AUDIO_CAPTIONS.values()}, audios),
        "text -> video": ({t: t for t in VIDEO_TEXTS}, {**video, **{k: images[k] for k in ("house", "dog")}}),
    }


def run_multimodal_demo(device, model_id="google/embeddinggemma-2", compare_hf=True):
    from models.demos.blackhole.embedding_gemma2.tt.embedding_model import EmbeddingGemma2

    snapshot = resolve_snapshot(model_id)
    model = EmbeddingGemma2.from_pretrained(device, str(snapshot), max_seq_len=2048)
    tasks = multimodal_inputs()
    # one trace per ShapeKey of every encode() call below, all captured before the first replay
    keys = {k for rows, cols in tasks.values() for part in (rows, cols) for k in model.shape_keys(list(part.values()))}
    logger.info(f"capturing {len(keys)} traces: {[str(k) for k in keys]}")
    model.capture_all(keys)

    hf = None
    if compare_hf:
        from models.demos.blackhole.embedding_gemma2.reference.hf_full_reference import embed, load_reference_full_model

        hf_model = load_reference_full_model(snapshot)

        def hf(inputs):
            return torch.cat([embed(hf_model, model.process(x).features) for x in inputs])

    results = {}
    for task, (rows, cols) in tasks.items():
        r = model.encode(list(rows.values()), traced=True)
        c = model.encode(list(cols.values()), traced=True)
        scores = r @ c.T
        lines = [f"== {task}: TT (traced) similarities, rows = queries, columns = {list(cols)}"]
        for i, name in enumerate(rows):
            best = int(scores[i].argmax())
            vals = " ".join(f"{v:+.3f}" for v in scores[i].tolist())
            lines.append(f"  {name[:52]:52s} [{vals}] -> {list(cols)[best]}")
        out = {"scores": scores, "argmax": scores.argmax(-1).tolist(), "columns": list(cols)}
        if hf is not None:
            hr, hc = hf(list(rows.values())), hf(list(cols.values()))
            cos_r = torch.nn.functional.cosine_similarity(r, hr, dim=-1)
            cos_c = torch.nn.functional.cosine_similarity(c, hc, dim=-1)
            hf_scores = hr @ hc.T
            out.update(cos=torch.cat([cos_r, cos_c]), hf_argmax=hf_scores.argmax(-1).tolist())
            lines.append(
                "  TT vs HF fp32 embedding cosine: "
                + ", ".join(f"{n[:24]}={v:.5f}" for n, v in zip([*rows, *cols], out["cos"].tolist()))
            )
            lines.append(f"  argmax TT {out['argmax']}  HF {out['hf_argmax']}")
        logger.info("\n".join(lines))
        results[task] = out
    model.release_traces()
    return results


def test_demo(eg2_device):
    scores = run_demo(eg2_device)
    # query i -> its matching document
    assert scores.argmax(dim=-1).tolist() == [0, 2, 1]
    assert torch.isfinite(scores).all()


def test_multimodal_demo(eg2_device):
    res = run_multimodal_demo(eg2_device)
    assert res["text retrieval"]["argmax"] == [0, 2, 1]
    assert res["text -> image"]["argmax"] == [0, 1, 2, 3]  # each caption finds its image
    assert res["text -> video"]["argmax"][0] == 0  # the dashboard text finds the dashboard video
    for task, out in res.items():
        assert out["argmax"] == out["hf_argmax"], task
        assert float(out["cos"].min()) >= 0.99, task


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION)
    try:
        run_demo(dev)
        run_multimodal_demo(dev, compare_hf="--no-hf" not in sys.argv)
    finally:
        ttnn.close_device(dev)
