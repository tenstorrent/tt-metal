"""Reference-model loader for ``coqui/XTTS-v2``.

``coqui/XTTS-v2`` is **not** a transformers checkpoint: its ``config.json`` has no
``model_type`` key (``AutoConfig``/``AutoModel`` cannot construct it) and the repo
ships a native Coqui checkpoint (``model.pth`` plus ``vocab.json``,
``speakers_xtts.pth``, ``mel_stats.pth``, ``dvae.pth``) rather than HF-native
weights.

The faithful reference is therefore the native Coqui runtime: we build the real
``TTS.tts.models.xtts.Xtts`` module and load the real ``model.pth`` weights via its
own ``load_checkpoint``. The returned module is the exact same PyTorch module the
ttnn port mirrors, so per-component PCC comparisons read genuine trained weights
(not a random-init structural stand-in).

Compatibility note: the installed ``TTS`` wheel imports
``transformers.pytorch_utils.isin_mps_friendly``, which was removed in
transformers >= 5. We restore that symbol (a thin wrapper over ``torch.isin``)
*before* importing ``TTS`` so the real library loads unchanged. This shim is
applied lazily inside ``load_reference_model`` — importing this module has no
side effects.
"""

from __future__ import annotations

import os

import torch


def _install_transformers_compat_shim() -> None:
    """Restore ``isin_mps_friendly`` removed from transformers >= 5.

    The Coqui ``TTS`` package (via ``TTS.tts.layers.tortoise.autoregressive``)
    does ``from transformers.pytorch_utils import isin_mps_friendly``. On newer
    transformers this raises ``ImportError`` and breaks the whole TTS import
    chain. The original helper is just an MPS-safe wrapper around ``torch.isin``.
    """
    import transformers.pytorch_utils as ptu

    if not hasattr(ptu, "isin_mps_friendly"):

        def isin_mps_friendly(elements, test_elements):
            return torch.isin(elements, test_elements)

        ptu.isin_mps_friendly = isin_mps_friendly


def _resolve_checkpoint_dir(model_id: str) -> str:
    """Return a local directory holding the Coqui XTTS checkpoint files.

    Uses the HF hub cache; falls back to an offline cache lookup so a fully
    populated cache works without network access.
    """
    from huggingface_hub import snapshot_download

    patterns = ["config.json", "vocab.json", "model.pth", "speakers_xtts.pth", "mel_stats.pth"]
    try:
        return snapshot_download(model_id, allow_patterns=patterns)
    except Exception:
        # Network unavailable: reuse whatever is already cached locally.
        return snapshot_download(model_id, allow_patterns=patterns, local_files_only=True)


def load_reference_model(model_id: str):
    """Return the real XTTS-v2 module (``nn.Module``) in eval mode with trained weights.

    Loaded from the native Coqui ``model.pth`` checkpoint via the Coqui ``TTS``
    runtime — the same module the ttnn port mirrors. Deterministic and
    import-safe (all work happens here, not at import time).
    """
    torch.manual_seed(0)

    _install_transformers_compat_shim()

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    checkpoint_dir = _resolve_checkpoint_dir(model_id)

    config = XttsConfig()
    config.load_json(os.path.join(checkpoint_dir, "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=checkpoint_dir,
        eval=True,
        use_deepspeed=False,
    )
    model.eval()

    return model


if __name__ == "__main__":
    # Quick self-check: real module loads and a real (weight-backed) forward runs.
    torch.manual_seed(0)
    m = load_reference_model("coqui/XTTS-v2")
    assert isinstance(m, torch.nn.Module), f"expected nn.Module, got {type(m)}"
    assert not m.training, "model must be in eval mode"

    # Exercise the trained transformer stack: conditioning encoder + perceiver
    # resampler produce the GPT conditioning latent. Deterministic under the seed.
    mel = torch.randn(1, 80, 200)
    with torch.no_grad():
        style_emb = m.gpt.get_style_emb(mel)
    assert style_emb.shape == (1, 1024, 32), style_emb.shape
    print("OK:", type(m).__name__, "children=", [n for n, _ in m.named_children()])
    print("forward style_emb shape:", tuple(style_emb.shape))
