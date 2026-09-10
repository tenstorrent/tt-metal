"""Prompt contract parity against the diffusers encoders module (reference venv) and transformers' tokenizer."""
import torch

from models.autoports.minimaxai_minimax_music3.reference import prompt as P

CASES = [
    ("Genre: acoustic pop. BPM: 96.", "[verse]\nMorning light\n[chorus]\nSoftly"),
    (
        "## Global Metadata\n- **bpm** is 92, <|key E minor|>\n\n\n• Warm and *intimate*\n----\n",
        "[Verse] Walking down the street\n[Chorus]\nla la ^ la [Bridge] x",
    ),
    ("plain", "no tags at all\nline two"),
]


def test_clean_and_normalize_match_diffusers(diffusers_mm3):
    for cap, lyr in CASES:
        assert P.clean_caption(cap) == diffusers_mm3._clean_caption(cap)
        assert P.normalize_lyrics(lyr) == diffusers_mm3._normalize_lyrics(lyr)


def test_text_ids_match_transformers(snapshot):
    from transformers import Qwen2Tokenizer

    hf = Qwen2Tokenizer.from_pretrained(str(snapshot / "tokenizer"))
    tok = P.Music3Tokenizer(snapshot)
    for cap, lyr in CASES:
        text = P.prompt_text(cap, lyr)
        ours = tok.encode(text)
        ref = hf(text, return_tensors="pt")["input_ids"][0].tolist()
        assert ours == ref, (text, ours[:10], ref[:10])
        ids = P.build_text_ids(tok, cap, lyr)
        assert (
            ids.shape[0] == 2
            and (ids[1, 1:-2] == 151654).all()
            and ids[1, 0] == ids[0, 0]
            and torch.equal(ids[1, -2:], ids[0, -2:])
        )
        assert ids[0, -1] == 151669  # <|audio_start|>
