# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reference-model loader for Voxtral-TTS (``mistralai/Voxtral-4B-TTS-2603``).

WHY THIS FILE EXISTS
--------------------
``AutoModelForCausalLM/AutoModel.from_pretrained`` cannot load this repo, and that is not a
packaging accident — the architecture is genuinely not in ``transformers``:

  * the repo ships no ``config.json`` and no ``auto_map``; the config is Mistral-native
    ``params.json`` (which *does* carry ``"model_type": "voxtral_tts"``, but AutoConfig only
    ever looks at ``config.json``, hence "Unrecognized model ... should have a `model_type`
    key in its config.json");
  * the weights are a single Mistral-native ``consolidated.safetensors`` (8.0 GB, 386
    tensors), not a sharded HF ``model.safetensors`` with HF key names;
  * the model card declares ``library_name: vllm`` / ``tags: [mistral-common]`` /
    ``pipeline_tag: text-to-speech``. Upstream is vLLM-Omni
    (``vllm_omni/model_executor/models/voxtral_tts/*``), which is a live part of vLLM and
    pulls in ``vllm`` + ``mistral_common`` + ``flash_attn`` + ``apex``.

Voxtral-TTS is also not one network: it is an AR backbone (Ministral-derived, 3.4B), a
flow-matching acoustic transformer (390M) and a codec decoder (~150M). There is no single
transformers class that could express it even with the right config.

STRATEGY (option 4 — the architecture lives outside transformers)
-----------------------------------------------------------------
Rather than install vLLM-Omni (a GPU stack: ``flash_attn``, ``apex``, ``vllm``) just to run a
CPU golden, this loader uses the model's own torch-only reference implementation that already
lives in this tree at ``models/experimental/voxtral_tts/reference/`` — an op-for-op port of
upstream, written against the pinned vLLM-Omni commit recorded in that directory's
``PROVENANCE.md``. This module wraps that reference in a real ``nn.Module`` tree and fills it
with the **real weights the repo ships** (``consolidated.safetensors``).

Two properties are deliberate:

* **The module tree mirrors the checkpoint's key names.** ``model.state_dict()`` keys are the
  checkpoint's keys, so ``model.get_submodule("layers.0.attention.wq")``,
  ``model.acoustic_transformer.layers[2].feed_forward.w3`` and
  ``model.audio_tokenizer.decoder_blocks[7].layers[1].attention.q_norm`` all resolve, and a
  per-component PCC test can address exactly the tensor the TTNN port reads.
* **Every leaf is callable and every ``forward`` delegates to the audited reference
  functions**, so a submodule captured by the bring-up harness computes the same thing the
  block-level golden does. Nothing is re-derived here.

  The one place a leaf is *not* a stock ``nn`` module is the codec's convolutions: upstream's
  ``CausalConv1d``/``CausalConvTranspose1d`` do asymmetric causal padding and right-trimming,
  so a stock ``nn.Conv1d`` with the same weight would silently produce non-causal output.
  They are wrapped (same ``weight`` name, same shape) to call the reference's padding.

DEVIATIONS FROM THE RAW CHECKPOINT KEYS (both are what the reference loaders already do, and
what the TTNN port consumes):

* the codec's ``weight_norm`` parametrizations (``...conv.parametrizations.weight.original0``
  / ``original1``) are **folded** into a plain ``...conv.weight``;
* the semantic codebook is stored as EMA running sums, so
  ``quantizer.semantic_codebook.{embedding_sum,cluster_usage}`` are kept *and* the derived
  ``audio_tokenizer.semantic_embedding`` buffer is added.

Two config values are absent from ``params.json`` and come from upstream fallbacks
(``n_decoding_steps=7``, flow ``norm_eps=1e-5``); see ``voxtral_common_ref.py``.

DETERMINISM
-----------
Import has no side effects and loading takes no seed: weights are read from disk, the flow
block's ``time_embedding.inv_freq`` is recomputed analytically (it is absent from the release),
and the semantic codebook is a division. The model is returned in ``eval()`` mode with
``requires_grad_(False)``. The **only** stochastic path in the whole model is the flow
matching solver's initial noise ``x_0``; ``AcousticTransformer.decode_frame`` therefore
requires an explicit ``x_0`` unless a ``generator`` is passed, so a PCC test cannot
accidentally golden against fresh noise.

Loaded models are cached per ``(checkpoint realpath, dtype)`` — the fp32 model is ~16 GB and
takes tens of seconds to read, and a PCC session builds it many times.

USAGE
-----
    from _reference_loader import load_reference_model
    model = load_reference_model("/path/to/voxtral-tts-native")

    hidden = model(inputs_embeds=torch.randn(1, 8, 3072))          # [1, 8, 3072]
    hidden = model(input_ids=torch.tensor([1, 25, 24, 36]))        # text ids -> hidden
    codes  = model.acoustic_transformer(hidden[:, -1], x_0=x0)     # [B, 37]
    wav    = model.audio_tokenizer(codes_bt)                       # [B, 1, T*1920]
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
from typing import Optional

import torch
import torch.nn as nn

__all__ = ["load_reference_model"]


# =======================================================================================
# Locating the torch-only reference implementation
# =======================================================================================
_REF_PKG = "models.experimental.voxtral_tts.reference"
_REF_RELPATH = os.path.join("models", "experimental", "voxtral_tts", "reference")


def _candidate_repo_roots(model_dir: str):
    """Directories that might be a tt-metal checkout containing the reference package.

    The model directory is typically a tree of symlinks into the checkout (that is how the
    weights are shared), so resolving any of its files and walking up finds the repo even when
    this file lives somewhere else entirely.
    """
    seen, out = set(), []

    def add(p):
        if p and p not in seen:
            seen.add(p)
            out.append(p)

    add(os.environ.get("TT_METAL_HOME"))
    add(os.environ.get("VOXTRAL_REPO_ROOT"))

    # Walk up from every real path reachable from the model dir (symlinks included), and from
    # this file's own location.
    starts = [os.path.realpath(model_dir), os.path.abspath(__file__)]
    try:
        for name in sorted(os.listdir(model_dir)):
            starts.append(os.path.realpath(os.path.join(model_dir, name)))
    except OSError:
        pass
    for start in starts:
        d = start if os.path.isdir(start) else os.path.dirname(start)
        while d and d != os.path.dirname(d):
            add(d)
            d = os.path.dirname(d)

    for p in sys.path:
        add(os.path.abspath(p) if p else os.getcwd())
    return out


def _import_reference(model_dir: str):
    """Import the four reference modules, putting the repo root on ``sys.path`` if needed.

    Returns ``(common, backbone, flow, codec)``.
    """
    if _REF_PKG + ".voxtral_common_ref" not in sys.modules:
        root = next(
            (r for r in _candidate_repo_roots(model_dir)
             if os.path.isfile(os.path.join(r, _REF_RELPATH, "voxtral_common_ref.py"))),
            None,
        )
        if root is None:
            raise ImportError(
                "Could not locate the Voxtral-TTS torch reference implementation "
                f"({_REF_RELPATH}/voxtral_common_ref.py).\n"
                "It is the only CPU-runnable definition of this architecture — Voxtral-TTS is "
                "not a transformers model (no config.json / auto_map; upstream is vLLM-Omni).\n"
                "Point TT_METAL_HOME (or VOXTRAL_REPO_ROOT) at the tt-metal checkout that "
                "contains models/experimental/voxtral_tts/reference/."
            )
        if root not in sys.path:
            sys.path.insert(0, root)

    import importlib

    return tuple(
        importlib.import_module(f"{_REF_PKG}.{m}")
        for m in ("voxtral_common_ref", "voxtral_backbone_ref", "voxtral_flow_ref", "voxtral_codec_ref")
    )


# =======================================================================================
# Generic container: a module whose children are named exactly as the checkpoint keys them
# =======================================================================================
class _Container(nn.Module):
    """Holds submodules under arbitrary (possibly numeric) names.

    ``nn.ModuleList`` would give integer indexing but forces contiguous 0..N-1 children; the
    codec's ``decoder_blocks`` are 0..7 with convs only at {0,2,4,6} and transformers only at
    {1,3,5,7}, and both must keep their checkpoint index. This keeps the names *and* supports
    ``[i]`` indexing.
    """

    def __getitem__(self, key):
        name = str(key)
        if name not in self._modules:
            raise KeyError(f"{type(self).__name__} has no child {name!r} "
                           f"(have: {sorted(self._modules)})")
        return self._modules[name]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def keys(self):
        return list(self._modules)


class _RMSNorm(nn.Module):
    """``x * rsqrt(mean(x^2) + eps) * weight`` — delegates to the reference's ``rms_norm`` so
    the op order is bit-identical to the block-level golden (``torch.nn.RMSNorm`` upcasts and
    reduces differently)."""

    def __init__(self, dim: int, eps: float, rms_norm_fn):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps
        self._rms_norm = rms_norm_fn

    def forward(self, x):
        return self._rms_norm(x, self.weight, self.eps)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class _CausalConv1d(nn.Module):
    """Upstream ``CausalConv1d``: left-pad ``k - stride`` (plus length round-up) then conv."""

    def __init__(self, weight: torch.Tensor, kernel: int, stride: int, pad_mode: str, fn):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.kernel, self.stride, self.pad_mode = kernel, stride, pad_mode
        self._fn = fn

    def forward(self, x):
        return self._fn(x, self.weight, self.kernel, self.stride, self.pad_mode)

    def extra_repr(self):
        out, inp, k = self.weight.shape
        return f"{inp}, {out}, kernel_size={k}, stride={self.stride}, pad_mode={self.pad_mode}, causal=True"


class _CausalConvTranspose1d(nn.Module):
    """Upstream ``CausalConvTranspose1d``: full transposed conv, then trim ``k - stride`` on
    the right (``trim_ratio=1.0``)."""

    def __init__(self, weight: torch.Tensor, kernel: int, stride: int, fn):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.kernel, self.stride = kernel, stride
        self._fn = fn

    def forward(self, x):
        return self._fn(x, self.weight, self.kernel, self.stride)

    def extra_repr(self):
        inp, out, k = self.weight.shape
        return f"{inp}, {out}, kernel_size={k}, stride={self.stride}, causal=True"


def _linear(weight: torch.Tensor) -> nn.Linear:
    out_f, in_f = weight.shape
    m = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        m.weight.copy_(weight)
    m.weight.data = weight  # share storage; avoids a second 8 GB of copies
    return m


def _embedding(weight: torch.Tensor) -> nn.Embedding:
    n, d = weight.shape
    m = nn.Embedding(n, d)
    m.weight.data = weight
    return m


# =======================================================================================
# Grafting a flat {checkpoint key -> tensor} dict onto a module tree
# =======================================================================================
def _graft(root: nn.Module, path: str, leaf: nn.Module):
    """Attach ``leaf`` at dotted ``path``, creating ``_Container``s for missing intermediates."""
    parts = path.split(".")
    node = root
    for p in parts[:-1]:
        child = node._modules.get(p)
        if child is None:
            child = _Container()
            node.add_module(p, child)
        node = child
    node.add_module(parts[-1], leaf)


def _graft_tensor(root: nn.Module, path: str, tensor: torch.Tensor, buffer: bool = False):
    """Attach a bare Parameter/buffer (no wrapping module) at dotted ``path``."""
    parts = path.split(".")
    node = root
    for p in parts[:-1]:
        child = node._modules.get(p)
        if child is None:
            child = _Container()
            node.add_module(p, child)
        node = child
    if buffer:
        node.register_buffer(parts[-1], tensor)
    else:
        node.register_parameter(parts[-1], nn.Parameter(tensor))


# Leaf names that are plain tensors on their parent, not a ``<name>.weight`` submodule.
_BARE_PARAMS = ("attention_scale", "ffn_scale")
_BARE_BUFFERS = ("inv_freq", "embedding_sum", "cluster_usage", "semantic_embedding")


def _build_subtree(root: nn.Module, state: dict, C, codec_conv_spec: Optional[dict] = None):
    """Populate ``root`` from ``{relative key -> tensor}``, choosing a real module per leaf.

    ``codec_conv_spec`` maps a conv's module path to ``(kind, kernel, stride, pad_mode)``;
    only the codec passes it.
    """
    codec_conv_spec = codec_conv_spec or {}
    for key in sorted(state):
        t = state[key]
        parts = key.split(".")
        leaf = parts[-1]

        if leaf in _BARE_BUFFERS:
            _graft_tensor(root, key, t, buffer=True)
            continue
        if leaf in _BARE_PARAMS:
            _graft_tensor(root, key, t, buffer=False)
            continue
        if leaf != "weight":  # nothing else in this checkpoint; be loud rather than guess
            raise KeyError(f"unexpected checkpoint leaf {key!r}")

        mod_path = ".".join(parts[:-1])
        name = parts[-2]

        if mod_path in codec_conv_spec:
            kind, k, s, pad_mode = codec_conv_spec[mod_path]
            leaf_mod = (_CausalConv1d(t, k, s, pad_mode, C["causal_conv1d"]) if kind == "conv"
                        else _CausalConvTranspose1d(t, k, s, C["causal_conv_transpose1d"]))
        elif name in ("tok_embeddings", "embeddings"):
            leaf_mod = _embedding(t)
        elif t.ndim == 1:
            leaf_mod = _RMSNorm(t.shape[0], C["eps_for"](key), C["rms_norm"])
            leaf_mod.weight.data = t
        elif t.ndim == 2:
            leaf_mod = _linear(t)
        else:
            raise KeyError(f"cannot type checkpoint leaf {key!r} with shape {tuple(t.shape)}")
        _graft(root, mod_path, leaf_mod)


def _weights_of(module: nn.Module, strip_weight_suffix: bool = False) -> dict:
    """Rebuild the flat ``{key -> tensor}`` dict the reference functions take.

    Cheap: a dict of references to the parameters already held by the tree, so the reference
    functions see exactly the tensors a PCC test would read off the submodules.
    """
    w = {}
    for name, p in module.named_parameters(recurse=True):
        if strip_weight_suffix and name.endswith(".weight"):
            name = name[: -len(".weight")]
        w[name] = p
    for name, b in module.named_buffers(recurse=True):
        if b is not None:
            w[name] = b
    return w


# =======================================================================================
# Block 2 — flow-matching acoustic transformer (390M)
# =======================================================================================
class AcousticTransformer(_Container):
    """``h [B, 3072]`` -> ``audio_codes [B, 37]``. Delegates to ``voxtral_flow_ref``."""

    def __init__(self, flow_mod, common_mod):
        super().__init__()
        self._flow, self._common = flow_mod, common_mod

    def _w(self):
        return _weights_of(self)

    def predict_velocity(self, x_t, llm_output, t_emb):
        return self._flow.predict_velocity(x_t, llm_output, t_emb, self._w())

    def semantic_code(self, llm_hidden):
        """``h [B, 3072]`` -> greedy semantic code ``[B, 1]``."""
        return self._flow.semantic_code(llm_hidden, self._w())

    def time_embed(self, t):
        """``t [B, 1]`` -> ``cat(cos, sin) [B, 3072]``.

        Named ``time_embed``, not ``time_embedding``: the checkpoint puts ``inv_freq`` under a
        ``time_embedding`` submodule, and that name is reserved for it here so the tree keeps
        mirroring the checkpoint.
        """
        return self._flow.time_embedding(t, self.inv_freq)

    @property
    def inv_freq(self):
        """Recomputed at load: registered persistent upstream but ABSENT from the release."""
        return self.get_submodule("time_embedding").inv_freq

    def decode_frame(self, sem_code, llm_hidden, x_0=None, generator=None, **kw):
        """Euler-integrate the velocity field to acoustic codes ``[B, 36]``.

        ``x_0`` is the solver's initial noise and is the model's only stochastic input. It is
        required here (or a ``generator``) so a golden can never be taken against fresh,
        unreproducible noise — upstream draws it from the ambient RNG.
        """
        if x_0 is None:
            if generator is None:
                raise ValueError(
                    "decode_frame is stochastic in x_0; pass x_0=... (or generator=...) so the "
                    "result is reproducible. e.g. x_0=torch.randn(B, 36, "
                    "generator=torch.Generator().manual_seed(0))"
                )
            x_0 = torch.randn(sem_code.shape[0], self._common.N_ACOUSTIC_CODEBOOK, generator=generator)
        return self._flow.decode_frame(sem_code, llm_hidden, self._w(), x_0=x_0, **kw)

    def forward(self, llm_hidden, x_0=None, generator=None, **kw):
        """Full Block 2: ``h [B, 3072]`` -> ``[B, 37]`` (semantic ++ acoustic, offset applied)."""
        sem = self.semantic_code(llm_hidden)
        ac = self.decode_frame(sem, llm_hidden, x_0=x_0, generator=generator, **kw)
        return torch.cat([sem, ac], dim=1)


# =======================================================================================
# Block 3 — codec decoder (~150M)
# =======================================================================================
class CodecDecoder(_Container):
    """``codes [B, 37, T]`` (offset stripped) -> ``waveform [B, 1, T*1920]`` @ 24 kHz.

    The codec *encoder*'s weights are absent from the public release, so this is decode-only —
    reference-audio voice cloning is not reproducible from the shipped checkpoint (preset voice
    embeddings are)."""

    def __init__(self, codec_mod):
        super().__init__()
        self._codec = codec_mod

    def _w(self):
        return _weights_of(self)

    def quantizer_decode(self, codes):
        return self._codec.quantizer_decode(codes, self._w())

    def strip_offset_and_trim(self, codes):
        """Block 2's emitted frames ``[T, 37]`` -> this block's input ``[1, 37, T']``."""
        return self._codec.strip_offset_and_trim(codes)

    def forward(self, codes):
        return self._codec.reference_decode(codes, self._w())


# =======================================================================================
# Block 1 (+ the whole model) — Ministral-derived AR backbone, 3.4B
# =======================================================================================
class VoxtralTTSReferenceModel(nn.Module):
    """The full Voxtral-TTS reference as one ``nn.Module``, keyed like the checkpoint.

    The backbone's own tensors sit at the top level (``layers.*``, ``norm``,
    ``mm_audio_embeddings.*``) exactly as ``consolidated.safetensors`` stores them; the other
    two networks hang off ``acoustic_transformer`` and ``audio_tokenizer``.

    ``forward`` is the AR backbone's block boundary — ``inputs_embeds [1, S, 3072]`` ->
    ``hidden_states [1, S, 3072]`` — i.e. what a ``*ForCausalLM`` would expose, and what
    Block 2 consumes one position at a time.
    """

    def __init__(self, model_id: str, params: dict, refs, dtype: torch.dtype):
        super().__init__()
        common, backbone, flow, codec = refs
        self._common, self._backbone, self._flow, self._codec = common, backbone, flow, codec
        self.model_id = model_id
        self.params = params
        self.config = params  # convenience alias; this is Mistral params.json, not an HF config
        self.model_type = params.get("model_type", "voxtral_tts")
        self.dtype_ = dtype

    # -- the reference's weight-dict conventions ----------------------------------------
    def _backbone_w(self):
        """``voxtral_backbone_ref`` keys: no ``.weight`` suffix, and the two embedding tables
        renamed to ``tok_embeddings`` / ``audio_embeddings``."""
        w = {}
        for name, p in self.named_parameters(recurse=True):
            if name.startswith(("acoustic_transformer.", "audio_tokenizer.")):
                continue
            if name.endswith(".weight"):
                name = name[: -len(".weight")]
            w[name] = p
        w["tok_embeddings"] = w.pop("mm_audio_embeddings.tok_embeddings")
        w["audio_embeddings"] = w.pop("mm_audio_embeddings.audio_codebook_embeddings.embeddings")
        return w

    # -- input side ---------------------------------------------------------------------
    def embed_text(self, token_ids):
        """text token ids ``[S]`` / ``[1, S]`` -> ``inputs_embeds [1, S, 3072]``."""
        return self._backbone.embed_text(self._backbone_w(), token_ids)

    def embed_frame(self, codes):
        """one frame's 37 codes -> ``[1, 1, 3072]`` (sum over the 37 offset codebooks)."""
        return self._backbone.embed_frame(self._backbone_w(), codes)

    def embed_frames(self, codes):
        """``[T, 37]`` -> ``[1, T, 3072]``."""
        return self._backbone.embed_frames(self._backbone_w(), codes)

    # -- the block ----------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, inputs_embeds=None, input_ids=None, n_layers: Optional[int] = None):
        """``inputs_embeds [1, S, 3072]`` -> ``hidden_states [1, S, 3072]`` (causal, no cache).

        ``input_ids`` is accepted as a convenience: it is embedded with ``tok_embeddings``
        first. ``n_layers`` shortens the stack for wiring-only tests; it defaults to all 26.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("pass inputs_embeds [1, S, 3072] or input_ids [S]")
            inputs_embeds = self.embed_text(input_ids)
        n = self._common.N_LAYERS if n_layers is None else n_layers
        return self._backbone.reference_forward(inputs_embeds, self._backbone_w(), n_layers=n)

    @torch.no_grad()
    def text_logits(self, hidden):
        """Tied text head (``tok_embeddings``), for the text/EOS path only."""
        return self._backbone.text_logits(hidden, self._backbone_w())

    def incremental(self, n_layers: Optional[int] = None):
        """A stateful prefill + single-step decoder over a KV-cache (shaped like a TTNN traced
        decoder: build once -> prefill -> step per frame)."""
        n = self._common.N_LAYERS if n_layers is None else n_layers
        return self._backbone.IncrementalBackbone(self._backbone_w(), n_layers=n)


# =======================================================================================
# Public entry point
# =======================================================================================
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _resolve_repo(model_id: str):
    """-> (model_dir, params.json path, consolidated.safetensors path)."""
    model_dir = os.path.abspath(os.path.expanduser(model_id))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"not a directory: {model_id}")

    params_path = os.path.join(model_dir, "params.json")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(
            f"{model_id} has neither config.json nor params.json — cannot identify the "
            f"architecture. Contents: {sorted(os.listdir(model_dir))}"
        )

    ckpt = os.path.join(model_dir, "consolidated.safetensors")
    if not os.path.isfile(ckpt):
        shards = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
        raise FileNotFoundError(
            f"no consolidated.safetensors in {model_id} (found: {shards or 'no safetensors'}).\n"
            "The Voxtral-TTS reference needs the real (CC BY-NC 4.0, non-commercial) weights:\n"
            "    hf download mistralai/Voxtral-4B-TTS-2603 consolidated.safetensors "
            "params.json tekken.json --local-dir <dir>"
        )
    return model_dir, params_path, os.path.realpath(ckpt)


def load_reference_model(model_id: str):
    """Return an ``nn.Module`` (in eval mode) equivalent to the HF reference for this model,
    loaded from whatever real format the repo actually ships.

    For Voxtral-TTS that is the Mistral-native ``consolidated.safetensors`` + ``params.json``
    pair driving the model's own (non-transformers) architecture — see the module docstring.
    The returned module's ``state_dict()`` keys are the checkpoint's keys.
    """
    model_dir, params_path, ckpt = _resolve_repo(model_id)
    dtype = torch.float32  # the reference math is fp32; the checkpoint is bf16

    key = (ckpt, str(dtype))
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached

    with open(params_path) as f:
        params = json.load(f)
    declared = params.get("model_type")
    if declared not in (None, "voxtral_tts"):
        raise ValueError(
            f"{model_id} declares model_type={declared!r}; this loader implements 'voxtral_tts'."
        )

    common, backbone, flow, codec = refs = _import_reference(model_dir)

    # Shared helpers the leaf builders need, plus the per-block RMSNorm epsilon. The epsilons
    # genuinely differ (backbone/flow 1e-5, codec 1e-2, codec QK-norm 1e-6) and the codec's
    # 1e-2 is what the weights were trained with, unusual as it looks.
    def eps_for(key_):
        if ".q_norm." in key_ or ".k_norm." in key_:
            return common.CODEC_QK_NORM_EPS
        return None  # filled in per block below

    helpers = {
        "rms_norm": common.rms_norm,
        "causal_conv1d": codec.causal_conv1d,
        "causal_conv_transpose1d": codec.causal_conv_transpose1d,
    }

    model = VoxtralTTSReferenceModel(model_dir, params, refs, dtype)

    # ---- Block 1: AR backbone (top level, checkpoint-native names) ---------------------
    b_state = backbone.load_backbone_state(ckpt, dtype)
    b_flat = {}
    for k, v in b_state.items():
        if k == "tok_embeddings":
            b_flat["mm_audio_embeddings.tok_embeddings.weight"] = v
        elif k == "audio_embeddings":
            b_flat["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"] = v
        else:
            b_flat[k + ".weight"] = v
    _build_subtree(model, b_flat, {**helpers, "eps_for": lambda k: eps_for(k) or common.NORM_EPS})

    # ---- Block 2: flow-matching acoustic transformer -----------------------------------
    acoustic = AcousticTransformer(flow, common)
    _build_subtree(acoustic, flow.load_flow_state(ckpt, dtype),
                   {**helpers, "eps_for": lambda k: eps_for(k) or common.FM_NORM_EPS})
    model.add_module("acoustic_transformer", acoustic)

    # ---- Block 3: codec decoder --------------------------------------------------------
    # Which decoder_blocks index is a conv vs a transformer, and each conv's geometry, is
    # config-derived rather than inferable from the weight shapes (a [1024,1024,4] tensor is
    # equally valid as Conv1d or ConvTranspose1d, and getting it wrong is silent).
    conv_spec = {"output_proj.conv": ("conv", common.PATCH_PROJ_KERNEL, 1, "reflect")}
    for stage, ci in enumerate(common.DEC_CONV_BLOCKS):
        k, s = common.DEC_CONV_KERNELS[stage], common.DEC_CONV_STRIDES[stage]
        conv_spec[f"decoder_blocks.{ci}.conv"] = (
            ("conv", k, s, "replicate") if s == 1 else ("conv_transpose", k, s, None)
        )
    audio_tokenizer = CodecDecoder(codec)
    _build_subtree(audio_tokenizer, codec.load_codec_state(ckpt, dtype),
                   {**helpers, "eps_for": lambda k: eps_for(k) or common.CODEC_NORM_EPS},
                   codec_conv_spec=conv_spec)
    model.add_module("audio_tokenizer", audio_tokenizer)

    model.eval()
    model.requires_grad_(False)

    with _CACHE_LOCK:
        _CACHE.setdefault(key, model)
        model = _CACHE[key]
    return model


# =======================================================================================
# Self-check
# =======================================================================================
def _selfcheck(model_id: str) -> int:
    torch.manual_seed(0)
    m = load_reference_model(model_id)
    common = m._common

    n_par = sum(p.numel() for p in m.parameters())
    print(f"[loader] {type(m).__name__}  model_type={m.model_type}  "
          f"training={m.training}  params={n_par/1e9:.3f}B")

    # The tree is addressable by checkpoint key.
    for path in ("layers.0.attention.wq", "layers.25.feed_forward.w3", "norm",
                 "mm_audio_embeddings.tok_embeddings",
                 "acoustic_transformer.layers.2.attention_norm",
                 "acoustic_transformer.semantic_codebook_output",
                 "audio_tokenizer.decoder_blocks.0.conv",
                 "audio_tokenizer.decoder_blocks.7.layers.1.attention.q_norm",
                 "audio_tokenizer.output_proj.conv"):
        sub = m.get_submodule(path)
        print(f"[loader]   {path:58s} -> {type(sub).__name__} {sub.extra_repr()[:52]}")

    # Block 1 forward, from real token ids.
    text_ids, frames = m._backbone.make_synthetic_inputs(n_text=6, n_frames=4)
    embeds = torch.cat([m.embed_text(text_ids), m.embed_frames(frames)], dim=1)
    hidden = m(inputs_embeds=embeds)
    print(f"[loader] backbone {tuple(embeds.shape)} -> {tuple(hidden.shape)}  "
          f"mean {hidden.mean():+.4f} std {hidden.std():.4f}")
    assert hidden.shape == embeds.shape and torch.isfinite(hidden).all()

    # Determinism: a second load is the same object and the same numbers.
    assert load_reference_model(model_id) is m
    assert torch.equal(m(inputs_embeds=embeds), hidden), "forward is not deterministic"

    # KV-cache path must reproduce the causal prefill (proves RoPE offsets + cache wiring).
    inc = m.incremental()
    p = embeds.shape[1] - 3
    pre = inc.prefill(embeds[:, :p])
    steps = torch.cat([inc.step(embeds[:, t:t + 1]) for t in range(p, embeds.shape[1])], dim=1)
    print(f"[loader] kv-cache PCC: prefill {common.pcc(pre, hidden[:, p-1:p]):.6f}  "
          f"steps {common.pcc(steps, hidden[:, p:]):.6f}")

    # Block 2: real hidden state -> 37 codes (deterministic x_0).
    h = hidden[:, -1]
    x0 = torch.randn(1, common.N_ACOUSTIC_CODEBOOK, generator=torch.Generator().manual_seed(0))
    codes = m.acoustic_transformer(h, x_0=x0)
    assert torch.equal(m.acoustic_transformer(h, x_0=x0), codes)
    print(f"[loader] flow h{tuple(h.shape)} -> codes {tuple(codes.shape)}  "
          f"semantic={int(codes[0,0])} acoustic[:6]={codes[0,1:7].tolist()}")

    # Block 3: codes -> waveform at 24 kHz.
    T = 24
    syn = m._codec.make_synthetic_codes(n_frames=T)
    wav = m.audio_tokenizer(syn)
    exp = T * common.PATCH_SIZE * 8
    print(f"[loader] codec {tuple(syn.shape)} -> wav {tuple(wav.shape)} "
          f"({wav.shape[-1]/common.SAMPLING_RATE:.2f}s @ {common.SAMPLING_RATE} Hz) "
          f"peak {wav.abs().max():.4f}")
    assert wav.shape == (1, 1, exp), f"expected {(1, 1, exp)}, got {tuple(wav.shape)}"
    assert torch.isfinite(wav).all()

    # The wrapped modules must agree with the raw reference functions they delegate to.
    wb = m._backbone.load_backbone_state(os.path.realpath(
        os.path.join(m.model_id, "consolidated.safetensors")), torch.float32)
    ref_hidden = m._backbone.reference_forward(embeds, wb)
    print(f"[loader] wrapper vs raw reference_forward: PCC {common.pcc(hidden, ref_hidden):.8f}, "
          f"max|d| {(hidden - ref_hidden).abs().max():.3e}")
    assert torch.equal(hidden, ref_hidden), "module tree diverges from the raw reference"

    print("[loader] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck(
        sys.argv[1] if len(sys.argv) > 1 else "/localdev/lserbedzija/resolver_test/voxtral-tts-native"
    ))
