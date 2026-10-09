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

**Token-accuracy extension** (`extension_cases()`, set `librispeech_tf`): twenty more teacher-forced sequences, so
token accuracy does not rest on the primary six. Only `tests/e2e/test_token_accuracy.py` uses it; RTF, WER and SIM
stay on the primary set. The primary rule, continued:
- the two primary speakers get six more targets each, nominal 4, 5, 6, 10, 11 and 12 s, with their primary prompts;
- per gender, the next speaker after the primary one with an utterance within 20 % of 7, 3, 6, 9 and 12 s gets its
  ~7 s prompt and four targets at 3, 6, 9 and 12 s.
Within a speaker: the closest utterance within 20 %, ties by utterance id, never reusing one (the primary set's
included). Applied to the same test-clean on 2026-09-28.
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

# (utterance id, duration s, raw transcript). The primary speakers' extra targets use their primary prompts.
EXTENSION_TARGETS = {
    "260": [
        ("260-123286-0017", 3.975, "I SHUDDER AS I RECALL THESE MONSTERS TO MY REMEMBRANCE"),
        ("260-123440-0020", 4.995, "WE WON'T TALK ABOUT HER ANY MORE IF YOU'D RATHER NOT WE INDEED"),
        (
            "260-123286-0023",
            5.875,
            "THE RAFT WAS HEAVED UP ON A WATERY MOUNTAIN AND PITCHED DOWN AGAIN AT A DISTANCE OF TWENTY FATHOMS",
        ),
        (
            "260-123288-0010",
            9.995,
            "ON THE MAST ALREADY I SEE THE LIGHT PLAY OF A LAMBENT SAINT ELMO'S FIRE THE OUTSTRETCHED SAIL CATCHES "
            "NOT A BREATH OF WIND AND HANGS LIKE A SHEET OF LEAD",
        ),
        (
            "260-123288-0007",
            11.2,
            "THE WIND NEVER LULLS BUT TO ACQUIRE INCREASED STRENGTH THE VAST BANK OF HEAVY CLOUDS IS A HUGE "
            "RESERVOIR OF FEARFUL WINDY GUSTS AND RUSHING STORMS",
        ),
        (
            "260-123440-0004",
            12.02,
            "ALICE TOOK UP THE FAN AND GLOVES AND AS THE HALL WAS VERY HOT SHE KEPT FANNING HERSELF ALL THE TIME "
            "SHE WENT ON TALKING DEAR DEAR HOW QUEER EVERYTHING IS TO DAY",
        ),
    ],
    "121": [
        ("121-121726-0004", 4.02, "HEAVEN A GOOD PLACE TO BE RAISED TO"),
        ("121-121726-0008", 4.99, "HOSE MAN'S EXCUSE FOR WETTING THE WALK"),
        ("121-121726-0001", 5.925, "HARANGUE THE TIRESOME PRODUCT OF A TIRELESS TONGUE"),
        (
            "121-127105-0000",
            9.875,
            "IT WAS THIS OBSERVATION THAT DREW FROM DOUGLAS NOT IMMEDIATELY BUT LATER IN THE EVENING A REPLY THAT "
            "HAD THE INTERESTING CONSEQUENCE TO WHICH I CALL ATTENTION",
        ),
        (
            "121-127105-0023",
            10.91,
            "LET ME SAY HERE DISTINCTLY TO HAVE DONE WITH IT THAT THIS NARRATIVE FROM AN EXACT TRANSCRIPT OF MY OWN "
            "MADE MUCH LATER IS WHAT I SHALL PRESENTLY GIVE",
        ),
        (
            "121-123859-0003",
            10.825,
            "LOVE IS A BABE THEN MIGHT I NOT SAY SO TO GIVE FULL GROWTH TO THAT WHICH STILL DOTH GROW",
        ),
    ],
}
EXTENSION_SPEAKERS = {
    "672": {
        "gender": "M",
        "prompt": ("672-122797-0057", 6.56, "YES IN REALITY THOSE WERE HAPPY TIMES"),
        "targets": [
            ("672-122797-0065", 3.03, "NOW THAT TOO IS OVER"),
            (
                "672-122797-0070",
                6.27,
                "THE GOLDEN STAR OF TINSEL WAS STILL ON THE TOP OF THE TREE AND GLITTERED IN THE SUNSHINE",
            ),
            (
                "672-122797-0071",
                8.875,
                "IN THE COURT YARD SOME OF THE MERRY CHILDREN WERE PLAYING WHO HAD DANCED AT CHRISTMAS ROUND THE FIR "
                "TREE AND WERE SO GLAD AT THE SIGHT OF HIM",
            ),
            (
                "672-122797-0067",
                13.035,
                "THE TRUNKS WERE MOVED THE TREE WAS PULLED OUT AND THROWN RATHER HARD IT IS TRUE DOWN ON THE FLOOR BUT "
                "A MAN DREW HIM TOWARDS THE STAIRS WHERE THE DAYLIGHT SHONE",
            ),
        ],
    },
    "237": {
        "gender": "F",
        "prompt": (
            "237-134500-0037",
            7.105,
            "BUT EMIL IF I UNDERSTAND THEN ALL OUR GOOD TIMES ARE OVER WE CAN NEVER DO NICE THINGS TOGETHER ANY MORE",
        ),
        "targets": [
            ("237-134493-0009", 2.975, "PLEASE WAIT FOR ME MARIE EMIL COAXED"),
            (
                "237-134500-0000",
                6.13,
                "FRANK READ ENGLISH SLOWLY AND THE MORE HE READ ABOUT THIS DIVORCE CASE THE ANGRIER HE GREW",
            ),
            (
                "237-126133-0002",
                8.92,
                "THEN DEAR SAID MISSUS WHITNEY YOU MUST BE KINDER TO HER THAN EVER THINK WHAT IT WOULD BE FOR ONE OF "
                "YOU TO BE AWAY FROM HOME EVEN AMONG FRIENDS",
            ),
            (
                "237-126133-0001",
                11.965,
                "EVERY CHANCE SHE COULD STEAL AFTER PRACTICE HOURS WERE OVER AND AFTER THE CLAMOROUS DEMANDS OF THE "
                "BOYS UPON HER TIME WERE FULLY SATISFIED WAS SEIZED TO FLY ON THE WINGS OF THE WIND TO THE FLOWERS",
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


def _case(set_name: str, spk: str, gender: str, prompt: tuple, target: tuple) -> dict:
    p_id, p_dur, p_raw = prompt
    t_id, t_dur, t_raw = target
    return {
        "case_id": f"zero_shot_{t_id}",
        "set": set_name,
        "mode": "zero_shot",
        "lang": "en",
        "speaker": spk,
        "gender": gender,
        "prompt_utt": p_id,
        "prompt_duration_s": p_dur,
        "prompt_text": tts_text(p_raw),
        "prompt_wav": flac_relpath(p_id),
        "target_utt": t_id,
        "target_duration_s": t_dur,
        "text": tts_text(t_raw),
        "text_raw": t_raw,
    }


def extension_cases() -> list[dict]:
    """The token-accuracy extension, in a fixed order: the primary speakers' extra targets, then the new speakers."""
    out = [
        _case("librispeech_tf", spk, SPEAKERS[spk]["gender"], SPEAKERS[spk]["prompt"], t)
        for spk, targets in EXTENSION_TARGETS.items()
        for t in targets
    ]
    for spk, s in EXTENSION_SPEAKERS.items():
        out.extend(_case("librispeech_tf", spk, s["gender"], s["prompt"], t) for t in s["targets"])
    return out


def cases(include_parity: bool = False) -> list[dict]:
    """Every (prompt, target) pair, zero-shot, in a fixed order: speaker 260 then 121, targets short to long."""
    out = [_case("librispeech", spk, s["gender"], s["prompt"], t) for spk, s in SPEAKERS.items() for t in s["targets"]]
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
