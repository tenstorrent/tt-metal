# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
T5Tool: Translation and text generation using T5 on TTNN.

Supports translation between languages using the T5-base model.
Runs on full N300 mesh (1,2) - utility functions handle mesh tensor conversion.
"""

from loguru import logger
from transformers import AutoTokenizer, T5ForConditionalGeneration

import ttnn
from models.experimental.t5.t5_utils import run_generate
from models.experimental.t5.tt.t5_for_conditional_generation import t5_small_for_conditional_generation

# Supported language pairs (T5-base supports these)
SUPPORTED_LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "ro": "Romanian",
}


class T5Tool:
    """
    TTNN-accelerated T5 translation tool.

    Translates text between supported languages using the T5 model.
    """

    def __init__(self, mesh_device):
        self.device = mesh_device
        self._init_model()

    def _init_model(self):
        """Load T5 model."""
        logger.info("Loading T5-small translation model...")

        # Load TTNN model - use t5-small instead of t5-base to fit in memory
        # alongside other models (LLM, Whisper, etc.)
        self.tt_model, _ = t5_small_for_conditional_generation(self.device)

        # Load tokenizer and reference model for generation
        self.model_name = "t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=128)
        self.hf_model = T5ForConditionalGeneration.from_pretrained(self.model_name)

        logger.info("T5-small ready.")

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "de",
    ) -> str:
        """
        Translate text from source language to target language.

        Args:
            text: Text to translate.
            source_lang: Source language code (en, de, fr, ro).
            target_lang: Target language code (en, de, fr, ro).

        Returns:
            Translated text.
        """
        source_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
        target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)

        logger.info(f"Translating from {source_name} to {target_name}: '{text[:50]}...'")

        # Synchronize device before T5 operations
        ttnn.synchronize_device(self.device)

        # Format input for T5
        input_text = f"translate {source_name} to {target_name}: {text}"

        # Run generation
        output = run_generate(
            input_text,
            self.tokenizer,
            self.hf_model,
            self.tt_model,
            self.device,
            run_tt_model=True,
            log=False,
        )

        logger.info(f"Translation: '{output}'")
        return output

    def summarize(self, text: str, max_length: int = 128) -> str:
        """
        Summarize text.

        Args:
            text: Text to summarize.
            max_length: Maximum length of summary.

        Returns:
            Summarized text.
        """
        logger.info(f"Summarizing: '{text[:50]}...'")

        input_text = f"summarize: {text}"

        # Synchronize device
        ttnn.synchronize_device(self.device)

        output = run_generate(
            input_text,
            self.tokenizer,
            self.hf_model,
            self.tt_model,
            self.device,
            run_tt_model=True,
            log=False,
        )

        logger.info(f"Summary: '{output}'")
        return output

    def close(self):
        """Release resources."""
        self.tt_model = None
        self.hf_model = None
        self.tokenizer = None
        self.device = None
        logger.info("T5Tool closed.")
