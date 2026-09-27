# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Real CosyVoice2-0.5B checkpoint access: `FunAudioLLM/CosyVoice2-0.5B` on the
HuggingFace Hub, confirmed against the actual repo listing (not guessed from a
naming convention) -- `llm.pt` (2.02 GB), `flow.pt` (450 MB), `hift.pt` (83 MB),
plus `cosyvoice2.yaml`/`config.json`. `hf_hub_download` caches under
`~/.cache/huggingface/hub` like every other HF download this package already
does (`test_qwen2lm.py`'s `Qwen/Qwen2-0.5B-Instruct` pull), so a second call in
the same environment is a cache hit, not a re-download.

Each `.pt` file is a flat `state_dict`-shaped file (confirmed by loading and
inspecting real keys, not assumed): `hift.pt` is `HiFTGenerator.state_dict()`
directly (`conv_pre`/`ups`/`source_downs`/`source_resblocks`/`resblocks`/
`conv_post`/`f0_predictor`/`m_source`, all top-level, no wrapping prefix, and
those attribute names match this package's own `TorchHiFTDecodeRef`/
`TorchConvRNNF0PredictorRef` construction exactly -- see each class's
`from_checkpoint` classmethod). `flow.pt` and `llm.pt` are the analogous
`state_dict()`s for `CausalMaskedDiffWithXvec` and `Qwen2LM` respectively.
"""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download

REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"


def load_checkpoint_file(filename: str) -> dict:
    """Download (or reuse the HF cache for) one real file from the real
    CosyVoice2-0.5B repo and `torch.load` it. `filename` is one of
    `llm.pt`/`flow.pt`/`hift.pt`/`cosyvoice2.yaml`/etc, exactly as it appears
    in the real repo listing."""
    path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    return torch.load(path, map_location="cpu")


def sub_state_dict(state_dict: dict, prefix: str) -> dict:
    """`{k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}`
    -- pulls one submodule's keys out of a flat checkpoint dict and strips the
    prefix, so the result loads directly into that submodule's own
    `load_state_dict`."""
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def build_local_qwen2_checkpoint_dir(llm_state_dict: dict, out_dir: str) -> str:
    """`llm.pt`'s `llm.*`-prefixed keys ARE a real `Qwen2ForCausalLM.state_dict()`
    once the outer `llm.` prefix is stripped -- confirmed directly: real
    upstream `Qwen2Encoder.model = Qwen2ForCausalLM(pretrain_path)`, so
    `Qwen2LM.llm.model.*` IS that model's own state dict, keys and all
    (`llm.model.model.embed_tokens.weight` -> `model.embed_tokens.weight`,
    `llm.model.model.layers.N.*` -> `model.layers.N.*`,
    `llm.model.lm_head.weight` -> `lm_head.weight`). Writes that (plus the real
    `CosyVoice-BlankEN/config.json`, which has `tie_word_embeddings: true`, so
    `lm_head.weight` need not even be included) as a local HF-format checkpoint
    directory, so `models.tt_transformers.tt.model_config.ModelArgs`'
    existing, already-validated `HF_MODEL=<local_dir>` -> `AutoModelForCausalLM.
    from_pretrained(<local_dir>)` loading path can be reused unchanged --
    rather than writing a second, new low-level loader for the 24-layer
    backbone itself. Returns `out_dir` for convenience (set `HF_MODEL=` to it).
    """
    import json
    import os
    import shutil

    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    llm_prefix = "llm."
    qwen_sd = {
        k[len(llm_prefix) :]: v.contiguous().clone()
        for k, v in llm_state_dict.items()
        if k.startswith(llm_prefix) and k != "llm.lm_head.weight"
    }
    assert qwen_sd, "no 'llm.*' keys found -- was the wrong state dict passed?"

    config_path = hf_hub_download(repo_id=REPO_ID, filename="CosyVoice-BlankEN/config.json")
    shutil.copy(config_path, f"{out_dir}/config.json")
    with open(f"{out_dir}/config.json") as f:
        assert json.load(f)["tie_word_embeddings"] is True

    # `lm_head.weight` dropped above: `tie_word_embeddings: true` means real
    # upstream's `lm_head.weight` shares storage with `model.embed_tokens.weight`
    # (tied embedding), which `safetensors.save_file` refuses to serialize
    # (no aliased storage across tensors) -- and this package's `TtQwen2LM` never
    # builds/uses `lm_head` at all (see qwen2lm.py's module docstring), so
    # dropping it loses nothing. `.clone()` on every tensor for the same reason:
    # guards against any OTHER incidental storage sharing `torch.load` preserved.
    save_file(qwen_sd, f"{out_dir}/model.safetensors")
    return out_dir
