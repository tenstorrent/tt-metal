# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Case registry, local model files and the CPU-reference cache of the GDN baseline.

A case is (model, tokens per chunk, weights, inputs, plan). ``chain`` runs three full chunks with the layer's
carried state; ``ragged`` runs one full chunk and then a partial last chunk of ``tokens - RAGGED_TAIL`` valid rows
(masked ``valid_len``), whose padding rows hold large finite poison values.

Local files live under ``<checkout>/.weights/<org>--<name>/<revision>/`` (``GDN_BASELINE_WEIGHTS`` overrides the
root): the tokenizer for text cases; for real weights also ``layer0.safetensors`` (the first GDN
layer's ``linear_attn.*`` tensors plus ``input_layernorm.weight``), ``embed_rows.safetensors`` (only the embedding
rows the text needs) and ``manifest.json``. ``prepare.py`` fetches them and fills the reference cache; the device
tests only read and fail on a miss. Model configs are the pinned in-tree copies (``reference/gdn/qwen_models.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.layer import gdn_forward_reference
from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import QWEN_GDN_MODELS, qwen_gdn_config, qwen_model_config
from models.demos.deepseek_v3_d_p.reference.gdn.weights import GDN_WEIGHT_NAMES

REPOSITORY_ROOT = Path(__file__).resolve().parents[6]

# Covers: reference/gdn/layer.py math, the input builders below (randn seeds, text slicing, layer-0 input norm), the
# synthetic weight generator and the stored fields. Bump with any change to what a cache entry holds.
REFERENCE_CACHE_VERSION = 1

CHAINED_CHUNKS = 3
RAGGED_TAIL = 40  # last-chunk padding rows; valid_len = tokens - 40 is not tile aligned
RANDN_SEED = 1234
SYNTHETIC_WEIGHT_SEED = 0  # random_gdn_state_dict(seed=layer index 0), as test_gdn_tp's "random" variant
POISON_SCALE = 8.0
GDN_LAYER = 0  # first linear-attention layer of every target model
PREFIX = (
    "linear_attn."  # checkpoint module of the GDN weights; the qwen36 device loader keeps it, the reference does not
)

TEXT_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"  # Pride and Prejudice, public domain


MODELS = {name: QWEN_GDN_MODELS[name] for name in ("qwen38_27b", "qwen36_35b")}  # pinned repo + revision
TOKENS = (640, 1280)


@dataclass(frozen=True)
class GdnCase:
    model: str
    tokens: int
    weights: str  # "synthetic" | "real"
    inputs: str  # "randn" | "text"
    plan: str  # "chain" | "ragged"

    @property
    def name(self) -> str:
        return f"{self.model}-T{self.tokens}-{self.weights}-{self.inputs}-{self.plan}"

    @property
    def valid_lengths(self) -> list[int]:
        if self.plan == "chain":
            return [self.tokens] * CHAINED_CHUNKS
        return [self.tokens, self.tokens - RAGGED_TAIL]


SMOKE_TOKENS = 128  # harness smoke cases on the smaller model before production shapes


def _cases() -> dict[str, GdnCase]:
    cases = [
        GdnCase("qwen36_35b", SMOKE_TOKENS, "synthetic", "randn", "chain"),
        GdnCase("qwen36_35b", SMOKE_TOKENS, "synthetic", "randn", "ragged"),
    ]
    for model in MODELS:
        for tokens in TOKENS:
            cases += [
                GdnCase(model, tokens, "synthetic", "randn", "chain"),
                GdnCase(model, tokens, "synthetic", "randn", "ragged"),
                GdnCase(model, tokens, "real", "text", "chain"),
                GdnCase(model, tokens, "real", "text", "ragged"),
                GdnCase(model, tokens, "real", "randn", "chain"),
            ]
    return {case.name: case for case in cases}


CASES = _cases()


# --------------------------------------------------------------------------------------------------------------
# Local model files
# --------------------------------------------------------------------------------------------------------------
def weights_root() -> Path:
    return Path(os.environ.get("GDN_BASELINE_WEIGHTS", REPOSITORY_ROOT / ".weights"))


def model_dir(model: str) -> Path:
    source = MODELS[model]
    return weights_root() / source.repo.replace("/", "--") / source.revision


def corpus_path() -> Path:
    return weights_root() / "corpus" / Path(TEXT_URL).name


def text_config(model: str) -> dict:
    """The pinned config.json's text tower (in-tree copy, ``reference/gdn/model_configs``)."""
    config = qwen_model_config(model)
    return config.get("text_config", config)


def gdn_config(model: str) -> GDNConfig:
    return qwen_gdn_config(model)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _synthetic_args(model: str) -> SimpleNamespace:
    """The fields random_gdn_state_dict reads, from config.json (same derivation as Qwen36ModelArgs)."""
    tc = text_config(model)
    q = tc["linear_num_key_heads"] * tc["linear_key_head_dim"]
    v = tc["linear_num_value_heads"] * tc["linear_value_head_dim"]
    return SimpleNamespace(
        dim=tc["hidden_size"],
        gdn_nv=tc["linear_num_value_heads"],
        gdn_dv=tc["linear_value_head_dim"],
        gdn_qkv_dim=2 * q + v,
        gdn_z_dim=v,
        gdn_value_dim=v,
        gdn_conv_kernel_size=tc["linear_conv_kernel_dim"],
    )


def load_layer_weights(model: str, weights: str) -> dict[str, torch.Tensor]:
    """``linear_attn.*`` tensors of the first GDN layer (bf16, checkpoint keys)."""
    if weights == "synthetic":
        from models.demos.blackhole.qwen36.tests.test_factory import random_gdn_state_dict

        return random_gdn_state_dict(_synthetic_args(model), seed=SYNTHETIC_WEIGHT_SEED)
    from safetensors.torch import load_file

    path = model_dir(model) / "layer0.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run: python -m {__package__}.prepare --fetch-weights {model}")
    tensors = load_file(path)
    return {k: v for k, v in tensors.items() if k.startswith(PREFIX)}


def weights_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in GDN_WEIGHT_NAMES:
        tensor = state_dict[PREFIX + name].contiguous()
        digest.update(name.encode())
        digest.update(tensor.view(torch.uint8 if tensor.dtype != torch.bfloat16 else torch.int16).numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------------------------
def _text_token_ids(model: str, count: int) -> list[int]:
    from tokenizers import Tokenizer

    raw = corpus_path().read_text(encoding="utf-8")
    body = raw[raw.index("\n", raw.index("*** START OF")) + 1 :]
    tokenizer = Tokenizer.from_file(str(model_dir(model) / "tokenizer.json"))
    ids = tokenizer.encode(body).ids[:count]
    if len(ids) < count:
        raise ValueError(f"corpus yields {len(ids)} tokens, need {count}")
    return ids


def _layer0_input(model: str, token_ids: list[int]) -> torch.Tensor:
    """Embedding -> Qwen3_5RMSNorm (``* (1 + w)``, eps from config) -> bf16, the first layer's GDN input."""
    from safetensors.torch import load_file

    directory = model_dir(model)
    rows = load_file(directory / "embed_rows.safetensors")
    position = {int(i): r for r, i in enumerate(rows["ids"].tolist())}
    missing = sorted(set(token_ids) - set(position))
    if missing:
        raise ValueError(f"embed_rows.safetensors lacks {len(missing)} token rows; refetch the weights")
    emb = rows["rows"][[position[i] for i in token_ids]].float()
    norm_w = load_file(directory / "layer0.safetensors")["input_layernorm.weight"].float()
    eps = text_config(model)["rms_norm_eps"]
    normed = emb * torch.rsqrt(emb.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * (1.0 + norm_w)).to(torch.bfloat16)


def build_inputs(case: GdnCase) -> torch.Tensor:
    """[chunks * tokens, hidden] bf16 layer input; ragged padding rows hold poison."""
    config = gdn_config(case.model)
    total = len(case.valid_lengths) * case.tokens
    valid_total = sum(case.valid_lengths)
    if case.inputs == "randn":
        generator = torch.Generator().manual_seed(RANDN_SEED)
        x = torch.randn(total, config.hidden_size, generator=generator).to(torch.bfloat16)
    else:
        ids = _text_token_ids(case.model, valid_total)
        valid = _layer0_input(case.model, ids)
        x = torch.empty(total, config.hidden_size, dtype=torch.bfloat16)
        offset = 0
        for c, n in enumerate(case.valid_lengths):
            x[c * case.tokens : c * case.tokens + n] = valid[offset : offset + n]
            offset += n
    for c, n in enumerate(case.valid_lengths):
        if n < case.tokens:
            generator = torch.Generator().manual_seed(RANDN_SEED + 1 + c)
            pad = torch.randn(case.tokens - n, config.hidden_size, generator=generator) * POISON_SCALE
            x[c * case.tokens + n : (c + 1) * case.tokens] = pad.to(torch.bfloat16)
    return x


# --------------------------------------------------------------------------------------------------------------
# Reference cache
# --------------------------------------------------------------------------------------------------------------
def _model_cache_root() -> Path:
    import ttnn

    return Path(ttnn.CONFIG.model_cache_path) / "gdn_baseline"


def case_identity(case: GdnCase, fingerprint: str) -> dict:
    identity = {
        "case": asdict(case),
        "version": REFERENCE_CACHE_VERSION,
        "model": {"repo": MODELS[case.model].repo, "revision": MODELS[case.model].revision},
        "weights_fingerprint": fingerprint,
        "valid_lengths": case.valid_lengths,
    }
    if case.inputs == "text":
        directory = model_dir(case.model)
        identity["corpus_sha256"] = _file_sha256(corpus_path())
        identity["tokenizer_sha256"] = _file_sha256(directory / "tokenizer.json")
        identity["embed_rows_sha256"] = _file_sha256(directory / "embed_rows.safetensors")
        identity["layer0_sha256"] = _file_sha256(directory / "layer0.safetensors")
    else:
        identity["randn_seed"] = RANDN_SEED
    return identity


def reference_cache_path(case: GdnCase, identity: dict) -> Path:
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return _model_cache_root() / f"{case.name}-{key}.pt"


def compute_reference(case: GdnCase, state_dict: dict[str, torch.Tensor], x: torch.Tensor) -> dict:
    config = gdn_config(case.model)
    weights = {name: state_dict[PREFIX + name] for name in GDN_WEIGHT_NAMES}
    state = None
    chunks = []
    for c, n in enumerate(case.valid_lengths):
        out, state = gdn_forward_reference(x[c * case.tokens : c * case.tokens + n], weights, config, state)
        chunks.append({"output": out, "recurrent": state.recurrent.clone(), "conv": state.conv.clone()})
    return {"chunks": chunks}


def load_case(case: GdnCase, state_dict: dict[str, torch.Tensor]) -> dict:
    """Cached inputs and reference for a case; raises on a miss (fill it with prepare.py)."""
    identity = case_identity(case, weights_fingerprint(state_dict))
    path = reference_cache_path(case, identity)
    if not path.is_file():
        raise FileNotFoundError(
            f"GDN baseline reference cache miss for {case.name} ({path}); run CPU-only first: "
            f"python -m {__package__}.prepare --case {case.name}"
        )
    entry = torch.load(path, weights_only=False)
    if entry["identity"] != identity:
        raise AssertionError(f"{path}: stored identity differs from the requested one")
    return entry


def per_device_conv_columns(conv: torch.Tensor, config: GDNConfig, tp: int) -> torch.Tensor:
    """Reorder conv-state columns from HF order [q | k | v] to the TP order [q_0 k_0 v_0 | q_1 k_1 v_1 | ...]
    that ``tp_common.prepare_gdn_qkv`` gives the device (each rank's K heads with its contiguous V heads)."""
    qs, ks, vs = conv.split([config.q_dim, config.k_dim, config.v_dim], dim=-1)
    parts = []
    for rank in range(tp):
        for t, width in ((qs, config.q_dim // tp), (ks, config.k_dim // tp), (vs, config.v_dim // tp)):
            parts.append(t[..., rank * width : (rank + 1) * width])
    return torch.cat(parts, dim=-1)
