# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The fixed input set every timing, listening and quality run uses.

Committed here so a reviewer can rerun exactly what was run. It is pure Python with no imports beyond the
standard library, so both the reference venv (scripts/prepare_inputs.py, scripts/run_reference.py) and
tt-metal's python_env (the demo and the tests) import this one definition.

**Primary set: LibriSpeech test-clean**, the set #54104's baseline targets name. Two speakers, one male and one
female. Each gets one prompt utterance (about 7 s) and three targets of about 3 s, 8 s and 15 s, all with real
transcripts.

Selection rule (deterministic, reproducible from the data alone): for each gender in SPEAKERS.TXT, take the
lowest-numbered test-clean speaker that has an utterance within 20% of each nominal length (7 s prompt, then
3 s / 8 s / 15 s targets). Within a speaker, pick the closest, ties broken by utterance ID, without reusing one.
Applied to LibriSpeech test-clean (openslr resources/12, test-clean.tar.gz md5
32fa31d27d2e1cad72775fee3f4849a9) on 2026-09-27, this gives the IDs below.

**Text:** LibriSpeech transcripts are upper case without punctuation. The text the model reads (`tts_text`) is
the transcript lower-cased, first letter capitalized, with a final period. The raw transcript is kept for WER.

**Secondary set: CosyVoice1 parity.** Upstream's own demo prompt (`asset/zero_shot_prompt.wav` from the
upstream checkout) with the English sentence CosyVoice1 scored. It is only a cross-check against the CosyVoice1
bring-up's numbers, never the primary result.
"""
from __future__ import annotations

import os

CORPUS_VERSION = 1
SUBSET = "test-clean"

# (utterance id, duration s, raw transcript), per speaker: prompt first, then targets (~3 s, ~8 s, ~15 s).
SPEAKERS = {
    "260": {
        "gender": "M",
        "prompt": (
            "260-123286-0016",
            7.0,
            "THESE THOUGHTS AGITATED ME ALL DAY AND MY IMAGINATION SCARCELY CALMED DOWN AFTER SEVERAL HOURS SLEEP",
        ),
        "targets": [
            ("260-123286-0014", 2.98, "TRULY THIS SEA IS OF INFINITE WIDTH"),
            (
                "260-123440-0010",
                8.315,
                "HOW CHEERFULLY HE SEEMS TO GRIN HOW NEATLY SPREAD HIS CLAWS AND WELCOME LITTLE FISHES IN WITH "
                "GENTLY SMILING JAWS",
            ),
            (
                "260-123440-0002",
                14.715,
                "IT WAS THE WHITE RABBIT RETURNING SPLENDIDLY DRESSED WITH A PAIR OF WHITE KID GLOVES IN ONE HAND AND "
                "A LARGE FAN IN THE OTHER HE CAME TROTTING ALONG IN A GREAT HURRY MUTTERING TO HIMSELF AS HE CAME OH "
                "THE DUCHESS THE DUCHESS",
            ),
        ],
    },
    "121": {
        "gender": "F",
        "prompt": ("121-121726-0003", 6.755, "HAY FEVER A HEART TROUBLE CAUSED BY FALLING IN LOVE WITH A GRASS WIDOW"),
        "targets": [
            ("121-127105-0015", 2.96, "HE QUITTED THE FIRE AND DROPPED BACK INTO HIS CHAIR"),
            (
                "121-127105-0003",
                7.725,
                "THERE WAS A UNANIMOUS GROAN AT THIS AND MUCH REPROACH AFTER WHICH IN HIS PREOCCUPIED WAY HE EXPLAINED",
            ),
            (
                "121-127105-0024",
                14.45,
                "POOR DOUGLAS BEFORE HIS DEATH WHEN IT WAS IN SIGHT COMMITTED TO ME THE MANUSCRIPT THAT REACHED HIM ON "
                "THE THIRD OF THESE DAYS AND THAT ON THE SAME SPOT WITH IMMENSE EFFECT HE BEGAN TO READ TO OUR HUSHED "
                "LITTLE CIRCLE ON THE NIGHT OF THE FOURTH",
            ),
        ],
    },
}

PARITY_PROMPT_WAV = "asset/zero_shot_prompt.wav"  # relative to the upstream checkout (COSYVOICE2_REPO)
PARITY_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
PARITY_TEXT = "The quick brown fox jumps over the lazy dog while the morning sun rises slowly."


def tts_text(transcript: str) -> str:
    """LibriSpeech transcript -> the sentence the model reads."""
    text = transcript.strip().lower()
    return text[:1].upper() + text[1:] + "."


def flac_relpath(utt_id: str) -> str:
    """Path relative to LIBRISPEECH_ROOT (the directory containing LibriSpeech/)."""
    spk, chapter, _ = utt_id.split("-")
    return os.path.join("LibriSpeech", SUBSET, spk, chapter, f"{utt_id}.flac")


def cases(include_parity: bool = False) -> list[dict]:
    """Every (prompt, target) pair, zero-shot, in a fixed order: speaker 260 then 121, targets short to long."""
    out = []
    for spk, s in SPEAKERS.items():
        p_id, p_dur, p_raw = s["prompt"]
        for t_id, t_dur, t_raw in s["targets"]:
            out.append(
                {
                    "case_id": f"zero_shot_{t_id}",
                    "set": "librispeech",
                    "mode": "zero_shot",
                    "lang": "en",
                    "speaker": spk,
                    "gender": s["gender"],
                    "prompt_utt": p_id,
                    "prompt_duration_s": p_dur,
                    "prompt_text": tts_text(p_raw),
                    "prompt_wav": flac_relpath(p_id),
                    "target_utt": t_id,
                    "target_duration_s": t_dur,
                    "text": tts_text(t_raw),
                    "text_raw": t_raw,
                }
            )
    if include_parity:
        out.append(
            {
                "case_id": "zero_shot_parity_cosyvoice1_en",
                "set": "cosyvoice1_parity",
                "mode": "zero_shot",
                "lang": "en",
                "speaker": "upstream_zero_shot_prompt",
                "gender": None,
                "prompt_utt": None,
                "prompt_duration_s": None,
                "prompt_text": PARITY_PROMPT_TEXT,
                "prompt_wav": PARITY_PROMPT_WAV,
                "target_utt": None,
                "target_duration_s": None,
                "text": PARITY_TEXT,
                "text_raw": PARITY_TEXT,
            }
        )
    return out
