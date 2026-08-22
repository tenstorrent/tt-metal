# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reference loader for hexgrad/Kokoro-82M.

Kokoro is a StyleTTS2 + ISTFTNet TTS model. Its `config.json` has no
`model_type`, so `transformers.AutoModel.{from_pretrained,from_config}` cannot
construct it. The real module is `kokoro.KModel` (a plain `torch.nn.Module`),
installed via `pip install --no-deps kokoro`.

`kokoro.__init__` eagerly imports `kokoro.pipeline`, which imports the heavy
`misaki` grapheme-to-phoneme stack (spacy, phonemizer, ...). That stack is only
needed for text->phoneme conversion at inference time, NOT to construct or walk
the acoustic model. To keep the tt-metal env pristine we register a lightweight
stand-in for `misaki` (mirroring scripts/tt_hw_planner/cpu_compat.py's approach)
so `import kokoro` succeeds without pulling the spacy chain.

Exposes `load_reference_model(model_id) -> torch.nn.Module` (eval mode). The
returned module's children are the real components discovery walks:
`bert` (CustomAlbert), `bert_encoder`, `predictor` (ProsodyPredictor),
`text_encoder` (TextEncoder), `decoder` (ISTFTNet Decoder).
"""
from __future__ import annotations

import sys
import types


def _install_misaki_stub() -> None:
    """Register import-safe stand-ins for `misaki`, `misaki.en`, `misaki.espeak`
    so `import kokoro` (which imports `kokoro.pipeline`) does not require the real
    misaki/spacy stack. Only installed when misaki is genuinely absent; never
    shadows a real install."""
    try:
        import misaki  # noqa: F401  (real install present -> use it)

        return
    except Exception:
        pass

    class _Stub(types.ModuleType):
        def __getattr__(self, name: str):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            # Return a dummy type so uses like `List[en.MToken]` in annotations resolve.
            return type(name, (), {})

    misaki = _Stub("misaki")
    misaki.__path__ = []  # mark as package
    en = _Stub("misaki.en")
    espeak = _Stub("misaki.espeak")
    misaki.en = en
    misaki.espeak = espeak
    sys.modules.setdefault("misaki", misaki)
    sys.modules.setdefault("misaki.en", en)
    sys.modules.setdefault("misaki.espeak", espeak)


def load_reference_model(model_id: str):
    """Return an eval-mode `torch.nn.Module` equivalent to the HF reference for
    Kokoro, loaded from the real `kokoro-v1_0.pth` checkpoint the repo ships."""

    _install_misaki_stub()

    from kokoro import KModel  # noqa: E402  (import after stub is registered)

    repo_id = model_id or "hexgrad/Kokoro-82M"
    # `disable_complex=True` avoids the complex-STFT path (CPU-friendlier and not
    # needed to construct / structurally walk the module tree).
    model = KModel(repo_id=repo_id, disable_complex=True)
    model.eval()
    return model


if __name__ == "__main__":
    m = load_reference_model("hexgrad/Kokoro-82M")
    print("loaded", type(m).__name__)
    for n, ch in m.named_children():
        print(" ", n, type(ch).__name__)
