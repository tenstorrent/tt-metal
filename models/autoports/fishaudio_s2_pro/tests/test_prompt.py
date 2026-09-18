"""Prompt tensor parity with upstream fish-speech (Conversation.encode_for_inference). Reference venv only."""
import pytest
import torch

from models.autoports.fishaudio_s2_pro.tt.prompt import S2Tokenizer, build_prompt

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world! 12345 — naïve café.",
    "多语言 test: こんにちは <|speaker:1|>hi",
]


def _upstream_prompt(snapshot, text, ref_codes=None, ref_text=None):
    from fish_speech.content_sequence import TextPart, VQPart
    from fish_speech.conversation import Conversation, Message
    from fish_speech.tokenizer import FishTokenizer

    tok = FishTokenizer.from_pretrained(str(snapshot))
    conv = Conversation()
    if ref_codes is not None:
        parts = [
            TextPart(text="convert the provided text to speech reference to the following:\n\nText:\n", cal_loss=False),
            TextPart(text=f"<|speaker:0|>{ref_text}", cal_loss=False),
            TextPart(text="\n\nSpeech:\n", cal_loss=False),
            VQPart(codes=ref_codes, cal_loss=False),
        ]
    else:
        parts = [TextPart(text="convert the provided text to speech", cal_loss=False)]
    conv.append(Message(role="system", parts=parts, cal_loss=False, add_im_start=True, add_im_end=True))
    conv.append(
        Message(
            role="user", parts=[TextPart(text=text, cal_loss=False)], cal_loss=False, add_im_start=True, add_im_end=True
        )
    )
    conv.append(
        Message(role="assistant", parts=[], cal_loss=False, modality="voice", add_im_start=True, add_im_end=False)
    )
    enc, _, _ = conv.encode_for_inference(tok, num_codebooks=10)
    return enc


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("with_ref", [False, True])
def test_prompt_matches_upstream(snapshot, upstream, text, with_ref):
    tok = S2Tokenizer(snapshot)
    g = torch.Generator().manual_seed(1)
    ref = (
        torch.cat([torch.randint(0, 4096, (1, 37), generator=g), torch.randint(0, 1024, (9, 37), generator=g)])
        if with_ref
        else None
    )
    ours = build_prompt(tok, text, [ref] if with_ref else None, ["reference transcript"] if with_ref else None)
    theirs = _upstream_prompt(snapshot, text, ref, "reference transcript" if with_ref else None)
    assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
    assert torch.equal(ours, theirs)


def test_prompt_structure(snapshot):
    tok = S2Tokenizer(snapshot)
    p = build_prompt(tok, "hi")
    assert p.shape[0] == 11 and p[0, 0] == 151644 and p[0, -1] == 151673  # <|im_start|> ... <|voice|>
    assert (p[1:] == 0).all()
