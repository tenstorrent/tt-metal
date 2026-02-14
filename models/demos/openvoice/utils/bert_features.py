# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
BERT Feature Extraction for MeloTTS.

Handles BERT feature extraction for prosody modeling.
Supports:
1. Preprocessed features (fastest)
2. CPU-based BERT extraction (fallback)
3. Dummy features for testing

BERT models used by MeloTTS:
- English/most languages: chinese-roberta-wwm-ext-large (1024 dim)
- Japanese: tohoku-nlp/bert-base-japanese-v3 (768 dim)
"""

from pathlib import Path
from typing import List, Optional, Tuple

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optional BERT dependencies
try:
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BERTFeatureExtractor:
    """
    Extract BERT features for MeloTTS.

    Can operate in three modes:
    1. 'dummy' - Return zero features (for testing)
    2. 'precomputed' - Load from cache
    3. 'live' - Run BERT model on CPU
    """

    # Model configurations
    BERT_MODELS = {
        "default": {
            "model_name": "hfl/chinese-roberta-wwm-ext-large",
            "hidden_size": 1024,
        },
        "japanese": {
            "model_name": "tohoku-nlp/bert-base-japanese-v3",
            "hidden_size": 768,
        },
    }

    def __init__(
        self,
        mode: str = "dummy",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize BERT feature extractor.

        Args:
            mode: 'dummy', 'precomputed', or 'live'
            cache_dir: Directory for precomputed features
            device: Device for BERT model ('cpu' or 'cuda')
        """
        self.mode = mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device

        self.tokenizer = None
        self.model = None
        self.ja_tokenizer = None
        self.ja_model = None

        if mode == "live":
            self._load_models()

    def _load_models(self):
        """Load BERT models for live extraction."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required for live BERT extraction. " "Install with: pip install transformers"
            )

        print("Loading BERT models...")

        # Main BERT (Chinese RoBERTa)
        cfg = self.BERT_MODELS["default"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
        self.model = AutoModel.from_pretrained(cfg["model_name"])
        self.model.eval()
        self.model.to(self.device)

        # Japanese BERT
        ja_cfg = self.BERT_MODELS["japanese"]
        self.ja_tokenizer = AutoTokenizer.from_pretrained(ja_cfg["model_name"])
        self.ja_model = AutoModel.from_pretrained(ja_cfg["model_name"])
        self.ja_model.eval()
        self.ja_model.to(self.device)

        print("BERT models loaded")

    def extract(
        self,
        text: str,
        language: str = "EN",
        seq_len: Optional[int] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Extract BERT features for text.

        Args:
            text: Input text
            language: Language code (EN, ZH, JA, etc.)
            seq_len: Expected sequence length (for alignment)

        Returns:
            Tuple of (bert_features [1, 1024, T], ja_bert_features [1, 768, T])
        """
        if self.mode == "dummy":
            return self._dummy_features(seq_len or 100)
        elif self.mode == "precomputed":
            return self._load_precomputed(text)
        else:
            return self._extract_live(text, language, seq_len)

    def _dummy_features(self, seq_len: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Return zero features for testing."""
        bert = torch.zeros(1, 1024, seq_len)
        ja_bert = torch.zeros(1, 768, seq_len)
        return bert, ja_bert

    def _load_precomputed(self, text: str) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Load precomputed features from cache."""
        if self.cache_dir is None:
            raise ValueError("cache_dir required for precomputed mode")

        # Hash text for cache key
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        cache_path = self.cache_dir / f"{text_hash}.pt"

        if cache_path.exists():
            data = torch.load(cache_path)
            return data["bert"], data["ja_bert"]
        else:
            # Fall back to dummy
            print(f"Warning: Precomputed features not found for text, using dummy")
            return self._dummy_features(100)

    def _extract_live(
        self,
        text: str,
        language: str,
        seq_len: Optional[int],
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Extract features using BERT models."""
        if self.model is None:
            self._load_models()

        with torch.no_grad():
            # Main BERT
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            bert_features = outputs.last_hidden_state.transpose(1, 2)  # [B, 1024, T]

            # Japanese BERT (always extract, will be zeros for non-JA)
            if language == "JA":
                ja_inputs = self.ja_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                ja_inputs = {k: v.to(self.device) for k, v in ja_inputs.items()}
                ja_outputs = self.ja_model(**ja_inputs)
                ja_bert_features = ja_outputs.last_hidden_state.transpose(1, 2)
            else:
                ja_bert_features = torch.zeros(1, 768, bert_features.size(2))

        # Resize to match seq_len if provided
        if seq_len is not None:
            bert_features = self._resize_features(bert_features, seq_len)
            ja_bert_features = self._resize_features(ja_bert_features, seq_len)

        return bert_features.cpu(), ja_bert_features.cpu()

    def _resize_features(self, features: "torch.Tensor", target_len: int) -> "torch.Tensor":
        """Resize features to target length using interpolation."""
        if features.size(2) == target_len:
            return features

        return torch.nn.functional.interpolate(
            features,
            size=target_len,
            mode="linear",
            align_corners=False,
        )

    def precompute_and_save(
        self,
        texts: List[str],
        languages: List[str],
        output_dir: str,
    ):
        """
        Precompute and save BERT features for a list of texts.

        Args:
            texts: List of input texts
            languages: List of language codes
            output_dir: Directory to save features
        """
        if self.mode != "live":
            raise ValueError("Must be in 'live' mode to precompute features")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import hashlib

        for text, lang in zip(texts, languages):
            bert, ja_bert = self._extract_live(text, lang, None)

            text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
            cache_path = output_dir / f"{text_hash}.pt"

            torch.save(
                {
                    "text": text,
                    "language": lang,
                    "bert": bert,
                    "ja_bert": ja_bert,
                },
                cache_path,
            )

            print(f"Saved: {cache_path}")


def get_bert_extractor(
    mode: str = "dummy",
    cache_dir: Optional[str] = None,
) -> BERTFeatureExtractor:
    """
    Get BERT feature extractor.

    Args:
        mode: 'dummy', 'precomputed', or 'live'
        cache_dir: Cache directory for precomputed features

    Returns:
        BERTFeatureExtractor instance
    """
    return BERTFeatureExtractor(mode=mode, cache_dir=cache_dir)


# Language code mapping
LANGUAGE_CODES = {
    "EN": 0,  # English
    "ES": 1,  # Spanish
    "FR": 2,  # French
    "ZH": 3,  # Chinese
    "JA": 4,  # Japanese
    "KO": 5,  # Korean
}

# Tone codes (simplified)
TONE_CODES = {
    "neutral": 0,
    "tone1": 1,
    "tone2": 2,
    "tone3": 3,
    "tone4": 4,
    "tone5": 5,  # Neutral tone in Chinese
}
