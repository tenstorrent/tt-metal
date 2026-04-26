from __future__ import annotations

import pytest

from models.demos.audio.cosy_voice.demo.demo import build_argparser, validate_mode_args


def test_build_argparser_accepts_public_modes():
    parser = build_argparser()
    args = parser.parse_args(["--mode", "sft", "--text", "hello", "--output", "/tmp/out.wav", "--speaker-id", "中文女"])
    assert args.mode == "sft"


@pytest.mark.parametrize(
    ("argv", "expected_message"),
    [
        (["--mode", "sft", "--text", "hello", "--output", "/tmp/out.wav"], "--speaker-id"),
        (
            ["--mode", "zero_shot", "--text", "hello", "--output", "/tmp/out.wav", "--prompt-audio", "a.wav"],
            "--prompt-text",
        ),
        (["--mode", "cross_lingual", "--text", "hello", "--output", "/tmp/out.wav"], "--prompt-audio"),
        (
            ["--mode", "instruct", "--text", "hello", "--output", "/tmp/out.wav", "--speaker-id", "中文男"],
            "--instruction",
        ),
    ],
)
def test_validate_mode_args_rejects_incomplete_cases(argv, expected_message):
    parser = build_argparser()
    args = parser.parse_args(argv)
    with pytest.raises(ValueError, match=expected_message):
        validate_mode_args(args)
