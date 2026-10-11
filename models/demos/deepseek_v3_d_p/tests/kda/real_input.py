# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real Kimi-K3 inputs to the first decoder layer (``model.layers.0``, a KDA layer).

The input to ``layers.0`` is exactly ``input_layernorm(embed_tokens[token_ids])`` (pinned against the vLLM
trace by ``tests/kimi_k3/test_golden_contract.py``). Deeper layers' inputs follow earlier layers and AttnRes,
so this module deliberately builds only the first layer's input: applying a later layer's norm to raw
embeddings would be a proxy no real run produces.

Token ids are the caller's: the Gutenberg prose here only checks the pipeline; accuracy evidence uses code
tokens from the vLLM trace's ``metadata.json``. Building inputs needs only local checkpoint shards
(the embedding table and ``layers.0``); fetching the tokenizer and corpus needs network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

KIMI_K3_REPO = "moonshotai/Kimi-K3"
KIMI_K3_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"

# Project Gutenberg #1342, Pride and Prejudice, as pinned by the linear-layers real-text inputs.
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
GUTENBERG_SHA256 = "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"

# Kimi-K3 tokenizer split pattern (tokenization_kimi.py TikTokenTokenizer.pat_str at KIMI_K3_REVISION).
_KIMI_PATTERN = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)


def gutenberg_token_ids(cache_dir: Path) -> list[int]:
    """Kimi-K3 token ids of the pinned Gutenberg corpus body (after its ``*** START OF`` line)."""
    import tiktoken
    from huggingface_hub import get_session, hf_hub_download
    from tiktoken.load import load_tiktoken_bpe

    path = Path(cache_dir) / Path(GUTENBERG_URL).name
    if not path.is_file():
        response = get_session().get(GUTENBERG_URL, follow_redirects=True)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != GUTENBERG_SHA256:
        raise ValueError(f"{path} sha256 {digest} != pinned {GUTENBERG_SHA256}")
    text = content.decode("utf-8")
    body = text[text.index("\n", text.index("*** START OF")) + 1 :]
    ranks = load_tiktoken_bpe(hf_hub_download(KIMI_K3_REPO, "tiktoken.model", revision=KIMI_K3_REVISION))
    encoding = tiktoken.Encoding(name=KIMI_K3_REPO, pat_str=_KIMI_PATTERN, mergeable_ranks=ranks, special_tokens={})
    return encoding.encode_ordinary(body)


class FirstLayerInput:
    """``input_layernorm(embed_tokens[ids])`` for ``model.layers.0``, as the BF16 tensor the layer receives.

    Loads the embedding table (2.3 GB BF16) and the norm weight once, so a long sequence can be built segment
    by segment without holding all of it.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
        self.eps = config.get("text_config", config)["rms_norm_eps"]
        with (checkpoint_dir / "model.safetensors.index.json").open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        root = next(r for r in ("language_model.model.", "model.") if f"{r}embed_tokens.weight" in weight_map)
        self.embedding = self._tensor(checkpoint_dir, weight_map, f"{root}embed_tokens.weight")
        self.norm_weight = self._tensor(checkpoint_dir, weight_map, f"{root}layers.0.input_layernorm.weight")

    @staticmethod
    def _tensor(checkpoint_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
        shard = checkpoint_dir / weight_map[key]
        if not shard.is_file():
            raise FileNotFoundError(f"{key} lives in {shard.name}, which is not in {checkpoint_dir}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def __call__(self, token_ids: torch.Tensor | list[int]) -> torch.Tensor:
        """``[1, T, hidden]`` in the embedding dtype (BF16), for 1-D ``token_ids`` of length T."""
        embeddings = self.embedding[torch.as_tensor(token_ids, dtype=torch.long)]
        # KimiRMSNorm: normalize in FP32, cast back to the input dtype, then scale (BF16 result).
        normalized = embeddings.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        return (self.norm_weight * normalized.to(embeddings.dtype)).unsqueeze(0)


def chunk_decay_statistics(gate: torch.Tensor, chunk: int = 32, ceiling: float = 160.0) -> dict[str, float]:
    """Per-chunk cumulative log decay ``|G_last|`` over every (chunk, head, channel) of a ``[T, H, K]`` gate.

    Reports quantiles, how many chunk-channels sit at the K3 bound's ceiling (``-5`` per token, so 160 per
    32 tokens), and how many are weak enough that ``exp(-|G_last|)`` rounds to 1 in BF16 (below 2^-9).
    Trailing tokens that do not fill a chunk are ignored; with no full chunk only ``chunk_channels`` (0) is reported.
    """
    usable = gate.shape[0] // chunk * chunk
    if not usable:
        return {"chunk_channels": 0}
    g_last = gate[:usable].double().reshape(-1, chunk, *gate.shape[1:]).sum(1).abs().flatten()
    # torch.quantile accepts at most 2^24 values; a fixed stride keeps larger inputs deterministic.
    sample = g_last[:: max(1, -(-g_last.numel() // 2**24))]
    quantiles = torch.quantile(sample, torch.tensor([0.5, 0.99], dtype=torch.float64))
    return {
        "chunk_channels": g_last.numel(),
        "p50": float(quantiles[0]),
        "p99": float(quantiles[1]),
        "max": float(g_last.max()),
        "at_ceiling": int((g_last >= ceiling * (1 - 1e-6)).sum()),
        "above_144": int((g_last > 144).sum()),
        "weak_below_2^-9": int((g_last < 2.0**-9).sum()),
    }
