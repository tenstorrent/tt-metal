# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device HunyuanVideo-1.5 byT5 (glyph) encoder built on the shared tt_dit T5.

HunyuanVideo-1.5 conditions on two text towers: Qwen2.5-VL (`pipe.text_encoder`,
see `qwen_encoder.py`) and a byT5 *glyph* encoder (`pipe.text_encoder_2`). The
byT5 tower only ever sees the quoted spans the pipeline extracts from the prompt
(`extract_glyph_texts`), tokenized by `ByT5Tokenizer` to a fixed 256-token
padded window. When a prompt contains no quoted text the diffusers pipeline
never calls the encoder at all -- it emits zeros and an all-zero mask -- so this
adapter is on the critical path only for glyph prompts.

Checkpoint shape (verified against `text_encoder_2/config.json`): a standard
12-layer gated-GELU T5 *encoder* with `d_model=1472`, `d_ff=3584`, 6 heads and
`d_kv=64`. That makes the attention inner width `6*64 = 384`, which is **not**
`d_model`; the shared `models/tt_dit/encoders/t5/model_t5.py` was extended to
carry an independent q/k/v width with an output projection back to `d_model`
(`T5Config.attention_inner_dim`), so this is a straight reuse rather than a fork.
The vocabulary is 1510: byT5's 259 byte-level/special ids plus 1251 added
`<color-*>`/`<*-font-*>` glyph-style tokens. There is no sentencepiece model --
the embedding table is a plain 1510x1472 lookup, and it is replicated (never
sharded) on the mesh.

Placement. Tensor parallelism has to divide both `num_heads` (6) and `d_model`
(1472 = 2^6 * 23), so the only legal factors are 1 and 2. Neither axis of the
production 8x4 DiT mesh can express that, and carving an overlapping submesh out
of a live parent deadlocks TTNN, so this module never creates a submesh: it
accepts a genuinely disjoint 1-device (TP1) or 2-device (TP2) mesh supplied by
the caller (`HY_BYT5_SUBMESH`, set by the test fixture) and otherwise fails
closed back to host byT5.

Gating. `HY_TT_BYT5=1` opts in; the default is host byT5 until the hardware PCC
gate in `tests/pcc/test_byt5_encoder_pcc.py` passes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

import ttnn
from models.tt_dit.encoders.t5.model_t5 import T5Config, T5Encoder
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache

# A test fixture or application may provide a genuinely disjoint TP1/TP2 mesh.
# Never create a submesh here: doing so can overlap a live parent/DiT context.
HY_BYT5_SUBMESH = None

# TTNN tile height. The token window is fixed at 256 in the diffusers pipeline,
# but the adapter pads any shorter window up to a tile so TILE_LAYOUT and the
# relative-position bias stay well formed.
_TILE = 32

_EXPECTED_CONFIG = {
    "d_model": 1472,
    "d_ff": 3584,
    "d_kv": 64,
    "num_heads": 6,
    "num_layers": 12,
    "vocab_size": 1510,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "gated-gelu",
    "dense_act_fn": "gelu_new",
    "is_encoder_decoder": False,
    "is_gated_act": True,
}

# `tie_word_embeddings` is deliberately absent from the strict contract above.
# The checkpoint's own `text_encoder_2/config.json` stores `false`, but
# `T5Config.from_pretrained` hands back `True`: HuggingFace does not round-trip
# this field. Enforcing it therefore rejected the very checkpoint this port was
# written against. It is also the one field in the set that cannot matter --
# it governs only whether a language-modelling head shares the input embedding
# table, and `T5EncoderModel` has no LM head, so no value of it can move an
# encoder activation. (The real risk it might have guarded, a checkpoint that
# actually carries an LM head, is already caught: `load_torch_state_dict` is
# strict about unexpected keys.) Every other check stays fail-closed; the
# observed value is surfaced through `ByT5Support.reason` so the quirk stays
# visible rather than silently ignored.
_TIE_WORD_EMBEDDINGS_IN_CHECKPOINT = False

# byT5's token window in `HunyuanVideo15Pipeline.tokenizer_2_max_length`.
DEFAULT_PROMPT_LENGTH = 256

# ByT5Tokenizer pads with id 0 (`<pad>`); used when padding a short window up to
# a tile. The synthesized positions are always masked out, so the id only has to
# be inside the vocabulary.
_PAD_TOKEN_ID = 0


@dataclass(frozen=True)
class ByT5Support:
    supported: bool
    strategy: str
    reason: str

    @property
    def tensor_parallel(self) -> int:
        """Tensor-parallel factor implied by `strategy` (1 when unsupported)."""
        if not self.supported:
            return 1
        return int(self.strategy.split("-", 1)[0].removeprefix("TP"))

    @property
    def mesh_axis(self) -> int:
        """Mesh axis the attention/FFN widths are fractured over."""
        return int(self.strategy.rsplit("axis", 1)[1])


def _config_mismatches(config) -> list[str]:
    mismatches = []
    for name, expected in _EXPECTED_CONFIG.items():
        actual = getattr(config, name, None)
        if actual != expected:
            mismatches.append(f"{name}={actual!r} (expected {expected!r})")
    architectures = getattr(config, "architectures", None)
    if architectures is not None and list(architectures) != ["T5EncoderModel"]:
        mismatches.append(f"architectures={architectures!r} (expected ['T5EncoderModel'])")
    return mismatches


def _tie_word_embeddings_note(config) -> str:
    """Report, without rejecting, a `tie_word_embeddings` HF did not round-trip."""
    actual = getattr(config, "tie_word_embeddings", _TIE_WORD_EMBEDDINGS_IN_CHECKPOINT)
    if bool(actual) == _TIE_WORD_EMBEDDINGS_IN_CHECKPOINT:
        return ""
    return (
        f"; note: tie_word_embeddings={actual!r} while the checkpoint stores "
        f"{_TIE_WORD_EMBEDDINGS_IN_CHECKPOINT!r} (HuggingFace does not round-trip this field). "
        "It only ties an LM head, which an encoder does not have, so it is not enforced"
    )


def analyze_byt5_support(config, mesh_shape) -> ByT5Support:
    """Validate the exact checkpoint contract and a non-overlapping TP1/TP2 placement."""
    mismatches = _config_mismatches(config)
    if mismatches:
        return ByT5Support(
            False,
            "host",
            "unsupported Hunyuan byT5 config: " + "; ".join(mismatches),
        )

    shape = tuple(int(size) for size in mesh_shape)
    if len(shape) != 2:
        return ByT5Support(False, "host", f"expected a 2D mesh shape, got {shape}")

    num_devices = shape[0] * shape[1]
    attention_width = int(config.num_heads) * int(config.d_kv)
    widths = (
        ("num_heads", int(config.num_heads)),
        ("attention inner width", attention_width),
        ("d_model", int(config.d_model)),
        ("d_ff", int(config.d_ff)),
    )
    legal = [factor for factor in (1, 2) if all(width % factor == 0 for _, width in widths)]

    if num_devices not in legal:
        return ByT5Support(
            False,
            "host",
            f"safe placement requires a dedicated mesh of {' or '.join(str(f) for f in legal)} device(s), "
            f"got {shape} ({num_devices} devices); tensor parallelism must divide both num_heads "
            f"({config.num_heads}) and d_model ({config.d_model}), so no factor above {max(legal)} is legal "
            "and the 8x4 DiT mesh cannot express it -- overlapping submeshes are forbidden",
        )

    # A 1-device mesh has no axis to fracture; keep axis 1 so the parallel config
    # and the prepared-weight cache key stay identical in shape to the TP2 case.
    axis = shape.index(2) if num_devices == 2 else 1
    note = _tie_word_embeddings_note(config)
    if num_devices == 1:
        return ByT5Support(
            True,
            "TP1-axis1",
            f"replicated single-device placement; q/k/v use independent width {attention_width}, "
            f"output projection returns to d_model={config.d_model}{note}",
        )
    return ByT5Support(
        True,
        f"TP2-axis{axis}",
        f"q/k/v use independent width {attention_width}; output projection returns to "
        f"d_model={config.d_model}{note}",
    )


def require_byt5_support(config, mesh_shape) -> ByT5Support:
    support = analyze_byt5_support(config, mesh_shape)
    if not support.supported:
        raise RuntimeError(
            "HY_TT_BYT5 was requested, but the byT5 checkpoint/placement is unsupported: "
            f"{support.reason}. Host byT5 remains the correct default."
        )
    return support


def select_byt5_device(config, dit_device=None):
    """Pick a byT5 mesh without ever carving one out of a live parent.

    Returns `(device_or_None, support)`. A `None` device means the caller must
    keep host byT5; `support.reason` explains why.
    """
    reserved = HY_BYT5_SUBMESH
    if reserved is not None:
        support = analyze_byt5_support(config, tuple(reserved.shape))
        return (reserved if support.supported else None), support
    if dit_device is None:
        return None, ByT5Support(False, "host", "no dedicated byT5 mesh was reserved (HY_BYT5_SUBMESH is unset)")
    support = analyze_byt5_support(config, tuple(dit_device.shape))
    return (dit_device if support.supported else None), support


def byt5_tt_config(config, *, max_prompt_length: int = DEFAULT_PROMPT_LENGTH) -> T5Config:
    """Translate the HF byT5 config into the shared tt_dit `T5Config`."""
    return T5Config(
        vocab_size=config.vocab_size,
        embed_dim=config.d_model,
        ff_dim=config.d_ff,
        kv_dim=config.d_kv,
        num_heads=config.num_heads,
        num_hidden_layers=config.num_layers,
        max_prompt_length=max_prompt_length,
        layer_norm_eps=config.layer_norm_epsilon,
        relative_attention_num_buckets=config.relative_attention_num_buckets,
        relative_attention_max_distance=config.relative_attention_max_distance,
    )


def byt5_cache_name(config) -> str:
    """Prepared-weight cache identity. Mesh/TP live in the key `cache` derives."""
    return (
        f"HunyuanVideo-1.5-byT5-v{config.vocab_size}-h{config.d_model}"
        f"-f{config.d_ff}-a{int(config.num_heads) * int(config.d_kv)}-l{config.num_layers}"
    )


def plan_byt5_inputs(input_ids, attention_mask, *, vocab_size, tile: int = _TILE):
    """Host-side preprocessing shared by the adapter and its host-only tests.

    Returns `(ids_int32, mask_bf16_or_None, logical_length)`. The token window is
    padded up to a tile multiple so TILE_LAYOUT and the `seq x seq` relative
    position bias are well formed. Padding is exact, not an approximation: T5's
    relative bias for position pair `(i, j)` depends only on `j - i`, so the
    bias over the original prefix is unchanged, and the synthesized positions are
    masked out of every softmax. When the caller supplies no mask, one is
    synthesized so the real tokens cannot attend to the synthesized tail.
    """
    ids = torch.as_tensor(input_ids)
    if ids.dim() != 2:
        raise ValueError(f"byT5 input_ids must be rank 2 (batch, sequence), got shape {tuple(ids.shape)}")
    ids = ids.detach().to(device="cpu", dtype=torch.int64)
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= int(vocab_size)):
        raise ValueError(
            f"byT5 input_ids out of range for vocab_size={vocab_size}: "
            f"[{int(ids.min())}, {int(ids.max())}]; the tokenizer and checkpoint disagree"
        )

    batch, length = ids.shape
    mask = None
    if attention_mask is not None:
        mask = torch.as_tensor(attention_mask).detach().to(device="cpu", dtype=torch.float32)
        if tuple(mask.shape) != (batch, length):
            raise ValueError(
                f"byT5 attention_mask shape {tuple(mask.shape)} does not match input_ids shape {(batch, length)}"
            )
        if not torch.isin(mask, torch.tensor([0.0, 1.0])).all():
            raise ValueError("byT5 attention_mask must contain only 0 and 1")

    padded_length = -(-length // tile) * tile
    if padded_length != length:
        pad = padded_length - length
        if mask is None:
            mask = torch.ones(batch, length, dtype=torch.float32)
        ids = torch.cat([ids, torch.full((batch, pad), _PAD_TOKEN_ID, dtype=ids.dtype)], dim=1)
        mask = torch.cat([mask, torch.zeros(batch, pad, dtype=mask.dtype)], dim=1)

    return (
        ids.to(torch.int32).contiguous(),
        None if mask is None else mask.to(torch.bfloat16).contiguous(),
        length,
    )


def finalize_byt5_output(embedding, *, length: int, attention_mask=None, zero_padding: bool = True):
    """Crop the tile padding back off and optionally neutralize masked positions.

    HuggingFace leaves padded positions at whatever the residual stream produced;
    those values are semantically undefined because the DiT consumes the mask
    alongside the embeddings. Zeroing them matches what the Wan on-device T5 and
    this repository's Qwen adapter already do and keeps any padding that survives
    `HunyuanVideo15Pipeline._trim_to_valid` neutral instead of arbitrary.
    """
    embedding = embedding[:, :length]
    if zero_padding and attention_mask is not None:
        mask = torch.as_tensor(attention_mask).detach().to(device="cpu")[:, :length]
        embedding = embedding * mask.unsqueeze(-1).to(embedding.dtype)
    return embedding


def build_tt_byt5_encoder(text_encoder, device, *, max_prompt_length: int = DEFAULT_PROMPT_LENGTH) -> T5Encoder:
    """Build and strictly load the exact Hunyuan byT5 checkpoint on a TP1/TP2 mesh."""
    support = require_byt5_support(text_encoder.config, tuple(device.shape))
    tensor_parallel = support.tensor_parallel
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=tensor_parallel, mesh_axis=support.mesh_axis)
    )
    tt_config = byt5_tt_config(text_encoder.config, max_prompt_length=max_prompt_length)
    model = T5Encoder(
        tt_config,
        device,
        CCLManager(mesh_device=device, num_links=1, topology=ttnn.Topology.Linear) if tensor_parallel > 1 else None,
        parallel_config,
    )
    # `load_torch_state_dict` runs strict, so a checkpoint whose key set drifts
    # from the T5 naming this port renames raises instead of silently loading a
    # partially initialized encoder. `TT_DIT_CACHE_DIR` enables the serialized
    # prepared-weight cache; without it this is the direct state-dict load.
    cache.load_model(
        model,
        model_name=byt5_cache_name(text_encoder.config),
        subfolder="text_encoder_2",
        parallel_config=parallel_config,
        mesh_shape=tuple(device.shape),
        get_torch_state_dict=text_encoder.state_dict,
    )
    return model


def _pcc(a, b) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() == 0:
        return 1.0
    centered_a = a - a.mean()
    centered_b = b - b.mean()
    denominator = centered_a.norm() * centered_b.norm()
    if denominator == 0:
        # One side is constant (e.g. an all-zero on-device result). Correlation is
        # undefined, so fall back to an equality check rather than reporting 1.0
        # and letting a dead output pass the gate.
        return 1.0 if torch.allclose(a, b, atol=1e-3, rtol=1e-2) else 0.0
    return float(torch.dot(centered_a, centered_b) / denominator)


class TTByT5EncoderAdapter:
    """Drop-in `T5EncoderModel` adapter returning host embeddings to diffusers.

    `HunyuanVideo15Pipeline._get_byt5_prompt_embeds` calls the encoder as
    `text_encoder(input_ids=..., attention_mask=<float>)[0]`, so `__call__`
    returns a 1-tuple whose element is the final encoder hidden state. Everything
    else (`.config`, `.dtype`, ...) proxies to the wrapped host encoder, which is
    also what the optional first-call self-check compares against.
    """

    def __init__(
        self,
        real_text_encoder,
        device,
        *,
        max_prompt_length: int = DEFAULT_PROMPT_LENGTH,
        zero_padding: bool | None = None,
        verify: bool | None = None,
        pcc_threshold: float | None = None,
    ):
        self.__dict__["_real"] = real_text_encoder
        self.__dict__["_device"] = device
        self.__dict__["_support"] = require_byt5_support(real_text_encoder.config, tuple(device.shape))
        self.__dict__["_tt"] = build_tt_byt5_encoder(real_text_encoder, device, max_prompt_length=max_prompt_length)
        self.__dict__["_zero_padding"] = (
            os.environ.get("HY_BYT5_ZERO_PAD", "1") == "1" if zero_padding is None else bool(zero_padding)
        )
        self.__dict__["_verify"] = os.environ.get("HY_BYT5_VERIFY", "1") == "1" if verify is None else bool(verify)
        self.__dict__["_pcc_threshold"] = (
            float(os.environ.get("HY_BYT5_PCC", "0.99")) if pcc_threshold is None else float(pcc_threshold)
        )
        self.__dict__["_checked"] = False

    def __getattr__(self, name):
        return getattr(self.__dict__["_real"], name)

    @torch.no_grad()
    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        if kwargs.get("output_hidden_states") or kwargs.get("output_attentions"):
            raise ValueError("Hunyuan TT byT5 supports final encoder hidden state only")
        if input_ids is None:
            raise ValueError("Hunyuan TT byT5 requires input_ids; inputs_embeds are not supported")

        real = self.__dict__["_real"]
        device = self.__dict__["_device"]
        ids, mask, length = plan_byt5_inputs(input_ids, attention_mask, vocab_size=int(real.config.vocab_size))

        tt_ids = ttnn.from_torch(
            ids,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        tt_mask = (
            ttnn.from_torch(
                mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
            if mask is not None
            else None
        )

        output = self.__dict__["_tt"](tt_ids, attention_mask=tt_mask)[-1]
        # Every parameter except the TP-fractured attention/FFN widths is
        # replicated and both replicas all-gather back to the full width, so
        # device 0 already holds the complete embedding.
        embedding = ttnn.to_torch(ttnn.get_device_tensors(output)[0])
        embedding = finalize_byt5_output(
            embedding,
            length=length,
            attention_mask=mask,
            zero_padding=self.__dict__["_zero_padding"],
        ).to(real.dtype)

        if self.__dict__["_verify"] and not self.__dict__["_checked"]:
            self.__dict__["_checked"] = True
            self._verify_against_host(input_ids, attention_mask, embedding)
        return (embedding,)

    def _verify_against_host(self, input_ids, attention_mask, embedding) -> None:
        """One-shot fail-closed check against the wrapped host encoder.

        byT5 is 12 layers over at most 256 tokens, so the host reference costs a
        fraction of a second once per generation. Only the masked-in tokens are
        compared: padded positions are deliberately zeroed on the TT side and are
        undefined on the host side.
        """
        real = self.__dict__["_real"]
        threshold = self.__dict__["_pcc_threshold"]
        ids = torch.as_tensor(input_ids).to(torch.int64)
        mask = None if attention_mask is None else torch.as_tensor(attention_mask)
        with torch.no_grad():
            reference = real(
                input_ids=ids,
                attention_mask=None if mask is None else mask.float(),
            )[
                0
            ].to(embedding.dtype)

        selector = (
            torch.ones_like(reference, dtype=torch.bool)
            if mask is None
            else mask.bool().unsqueeze(-1).expand_as(reference)
        )
        valid_pcc = _pcc(reference[selector], embedding[selector])
        print(
            f"[HY_TT_BYT5] on-device byT5 valid-token PCC vs host = {valid_pcc:.6f} "
            f"(threshold {threshold}, placement {self.__dict__['_support'].strategy})",
            flush=True,
        )
        if valid_pcc < threshold:
            raise RuntimeError(
                f"on-device byT5 valid-token PCC {valid_pcc:.6f} < {threshold}; "
                "unset HY_TT_BYT5 to fall back to the host encoder"
            )

    def deallocate_weights(self):
        model = self.__dict__["_tt"]
        if model.is_loaded():
            model.deallocate_weights()
            ttnn.synchronize_device(self.__dict__["_device"])
