# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Constants for the vendored ACE-Step 5 Hz LM stack (subset of ACE-Step ``constants``)."""

# Supported languages for constrained LM decoding
VALID_LANGUAGES = [
    "ar",
    "az",
    "bg",
    "bn",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "id",
    "is",
    "it",
    "ja",
    "ko",
    "la",
    "lt",
    "ms",
    "ne",
    "nl",
    "no",
    "pa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sa",
    "sk",
    "sr",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "vi",
    "yue",
    "zh",
    "unknown",
]

KEYSCALE_NOTES = ["A", "B", "C", "D", "E", "F", "G"]
KEYSCALE_ACCIDENTALS = ["", "#", "b", "♯", "♭"]
KEYSCALE_MODES = ["major", "minor"]

VALID_KEYSCALES = set()
for note in KEYSCALE_NOTES:
    for acc in KEYSCALE_ACCIDENTALS:
        for mode in KEYSCALE_MODES:
            VALID_KEYSCALES.add(f"{note}{acc} {mode}")

BPM_MIN = 30
BPM_MAX = 300
DURATION_MIN = 10
DURATION_MAX = 600
VALID_TIME_SIGNATURES = [2, 3, 4, 6]

DEFAULT_LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"
DEFAULT_LM_UNDERSTAND_INSTRUCTION = (
    "Understand the given musical conditions and describe the audio semantics accordingly:"
)
DEFAULT_LM_INSPIRED_INSTRUCTION = "Expand the user's input into a more detailed and specific musical description:"
DEFAULT_LM_REWRITE_INSTRUCTION = "Format the user's input into a more detailed and specific musical description:"
