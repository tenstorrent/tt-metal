# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein prompt encoding (``diffusers.Flux2KleinPipeline._get_qwen3_prompt_embeds``) on the mesh.

Host side, the reference recipe (``pipeline_flux2_klein.py:226-239``; the BFL ``text_encoder.py:389-403``
``Qwen3Embedder.forward`` is identical): the prompt goes through the Qwen3 chat template as a single user turn (no
system message, generation prompt appended, thinking disabled), is right-padded / truncated to ``sequence_length``
(512, ``P:204 tokenizer_max_length``) tokens with pad id ``<|endoftext|>``, and ``input_ids`` + ``attention_mask``
go to the encoder. Device side (``P:249-261``): the hidden states AFTER layers ``text_encoder_out_layers`` (9, 18, 27
of 36; HF ``hidden_states[k]`` is the input of layer ``k``, i.e. the output of layer ``k - 1``, no final norm) are
concatenated per token in increasing layer order: ``[B, L, 3 * hidden]`` (7680 for Qwen3-4B, 12288 for Qwen3-8B).
Padded positions are NOT zeroed and carry no mask downstream: the DiT receives all 512 rows (``T:1281``).

Only the decoder layers below the deepest tap are built and run (``num_layers = max(out_layers)`` = 27 of 36, no
final norm; ``FLUX2_KLEIN_TEXT_ENCODER_LAYERS``): the deepest tap, ``hidden_states[27]``, is the output of layer 26,
and layers 27-35 plus the final norm only feed the discarded logits. The generic encoder appends the raw output of
its last layer for index ``len(layers)`` when it has no final norm (``encoders/transformer.py``), so the reduced
build returns the same three taps as the full one; the weights load through their own cache entry
(``text_encoder_L27``). Mirrors the FLUX 3 Klein text encoder (32 of 36 layers). Encode is dispatch-bound (~5 ms per
layer), so 9 fewer layers are ~45 ms per request and 25 % fewer FSDP weight gathers.

RoPE tables are built on the HOST (:func:`qwen3_rope_tables`) with HF ``Qwen3RotaryEmbedding``'s arithmetic (fp32
``1 / theta ** (arange(0, 128, 2) / 128)``, fp32 outer product with ``arange(L)``, ``cat(freqs, freqs)``, one bf16
rounding of cos / sin) and passed to the generic encoder as ``pos_embeds``: its on-device tables (SFPU pow / cos / sin)
differ from HF's by bf16 ulps at the larger positions (the FLUX 3 Klein M5 measurement), and in a 512-row prompt the
positions ``[R, 512)`` are the pad rows the DiT reads, so the tables are kept bit-equal to the reference instead.

The per-request constants stay on the device (:meth:`Flux2KleinPromptEncoder.constants_info`): the RoPE tables of
``arange(512)`` are identical every call and are uploaded once per sequence length, and the causal + padding attention
bias, which the generic encoder would rebuild from the mask with a dozen small ops per call (``_prepare_attn_bias``),
is built on the host (``encoders.transformer.attention_bias``: ``[1, 1, 512, 512]`` bf16, exactly ``{0, MASK_VALUE}``)
and kept per real length ``R`` (right padding: the mask is ``[1] * R + [0] * (512 - R)``) in a bounded LRU
(``FLUX2_KLEIN_ENCODER_BIAS_CACHE_SIZE`` entries x 512 KB per chip), so a repeated length uploads nothing but the
tokens. The rows are bit-equal to the mask path (``test_constants_cache``); FLUX 3 Klein keeps its bias through the
mask-cached ``tensor.tril`` and rebuilds the tables per request. The encoder is never traced (a trace would read the
cached addresses; the LRU may free them).

Mesh placement mirrors ``pipelines/flux3_klein``: the encoder is co-resident with the DiT, tensor parallel over the
DiT's TP axis and FSDP (weight shards gathered per layer) over the other mesh axis, so Qwen3-4B's 8 GB bf16 costs
250 MB per chip on TP4 x FSDP8 (:func:`build_flux2_klein_text_encoder`). Its output ``[1, 512, joint]`` bf16 is
replicated on every chip (TP all-gathered, FSDP replicated), so the pipeline hands it to the DiT WITHOUT a host round
trip (:meth:`Flux2KleinTextEncoding.encode_device`): ``ttnn.mesh_partition(dim=1, cluster_axis=sp_axis)`` keeps the
``512 / SP`` rows each device owns (a per-device slice, exact; the FLUX 3 Klein ``output_sp_axis`` handoff) in the
layout the denoiser's ``embed_prompt`` consumes. ``encode_host`` (one replica read of the ``[512, joint]`` rows,
re-uploaded SP-sharded by the denoiser) stays for the empty prompt of the base CFG, the tests and the recorded-text
overrides; the two paths feed bit-identical rows (``test_dit_handoff``). Classifier-free guidance (the base models)
is two encodings, the cached ``encode_host("")`` and the request's.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import transformers
from loguru import logger

import ttnn

from ...encoders.qwen3 import Qwen3Checkpoint, Qwen3Encoder
from ...encoders.qwen3.model_qwen3 import _rope_theta
from ...encoders.transformer import attention_bias
from ...parallel.config import EncoderParallelConfig, ParallelFactor
from ...utils import tensor
from . import packing

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...parallel.manager import CCLManager

FLUX2_KLEIN_TEXT_SEQUENCE_LENGTH = packing.FLUX2_KLEIN_TEXT_LENGTH
FLUX2_KLEIN_TEXT_OUT_LAYERS = (9, 18, 27)
# Decoder layers 0..26 only: the deepest tap, hidden_states[27], is the output of layer 26 (no final norm).
FLUX2_KLEIN_TEXT_ENCODER_LAYERS = max(FLUX2_KLEIN_TEXT_OUT_LAYERS)
FLUX2_KLEIN_TEXT_SUBFOLDER = "text_encoder"
# Device attention biases kept per (batch, sequence length, real lengths): 512 KB per chip and entry at 512 tokens.
FLUX2_KLEIN_ENCODER_BIAS_CACHE_SIZE = 32
_CACHE_NAME_PREFIX = "flux2_klein_text_encoder_"


def format_prompts(tokenizer: transformers.PreTrainedTokenizerBase, prompts: Sequence[str]) -> list[str]:
    """The chat-templated text of every prompt, exactly as the diffusers pipeline builds it."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]


def tokenize(
    tokenizer: transformers.PreTrainedTokenizerBase, prompts: Sequence[str], *, sequence_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(input_ids, attention_mask)`` ``[B, sequence_length]`` int64, right-padded, truncated with a warning."""
    texts = format_prompts(tokenizer, prompts)
    out = tokenizer(texts, padding="max_length", max_length=sequence_length, truncation=True, return_tensors="pt")
    longest = tokenizer(texts, padding="longest", return_tensors="pt").input_ids
    if longest.shape[-1] > sequence_length:
        logger.warning(f"prompt truncated to {sequence_length} tokens (needed {longest.shape[-1]})")
    return out.input_ids, out.attention_mask


def qwen3_rope_tables(
    positions: torch.Tensor, *, head_dim: int, rope_theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF ``Qwen3RotaryEmbedding``'s bf16 tables: ``positions`` ``(L,)`` or ``(1, L)`` int64 -> ``(cos, sin)`` each
    ``(1, L, head_dim)`` bf16 (the generic encoder's ``pos_embeds`` layout). The arithmetic is HF's: fp32 inverse
    frequencies ``1 / theta ** (arange(0, head_dim, 2) / head_dim)``, an fp32 outer product (HF's K=1 matmul is a
    single fp32 product per element), ``cat(freqs, freqs)``, ``attention_scaling`` 1 (default rope), one bf16
    rounding of cos / sin."""
    positions = positions.reshape(-1)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :].float()
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16)[None], emb.sin().to(torch.bfloat16)[None]


def real_lengths(mask: torch.Tensor) -> tuple[int, ...] | None:
    """The real length of every row of a right-padded ``(B, L)`` attention mask (``[1] * R + [0] * (L - R)``), or
    ``None`` when a row is not of that form (the bias is then not cached)."""
    lengths = tuple(int(r) for r in mask.to(torch.int64).sum(dim=1))
    positions = torch.arange(mask.shape[1])[None, :]
    expected = positions < torch.tensor(lengths)[:, None]
    return lengths if torch.equal(mask.to(torch.bool), expected) else None


def text_encoder_blob(snapshot: str | os.PathLike) -> str:
    """The HuggingFace blob name of the first ``text_encoder/model*.safetensors`` shard: the basename of the
    symlink's OWN target (in a hub cache the file's sha256), or ``name:size`` for a plain file. The base and
    distilled snapshots of one size link the SAME text encoder blobs, so one cache entry per mesh layout serves
    both."""
    directory = Path(snapshot) / FLUX2_KLEIN_TEXT_SUBFOLDER
    shards = sorted(directory.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no model*.safetensors under {directory}")
    path = shards[0]
    if path.is_symlink():
        return Path(os.readlink(path)).name
    return f"{path.name}:{path.stat().st_size}"


def text_encoder_cache_name(snapshot: str | os.PathLike) -> str:
    """The weight-cache entry of a snapshot's text encoder (``flux2_klein_text_encoder_<blob[:16]>``)."""
    return f"{_CACHE_NAME_PREFIX}{text_encoder_blob(snapshot)[:16]}"


def fsdp_parallel_config(
    parallel_config: EncoderParallelConfig, mesh_shape: Sequence[int], *, is_fsdp: bool
) -> EncoderParallelConfig:
    """``parallel_config`` with FSDP placed over the mesh axis the tensor parallelism does not use.

    With ``is_fsdp`` the ``fsdp`` factor is added when the caller left it unset (a config that already names it is
    returned as is); it self-disables when the other axis has one device. Without ``is_fsdp`` a config that shards
    weights is a contradiction and raises.
    """
    tp = parallel_config.tensor_parallel
    fsdp = parallel_config.fsdp
    if not is_fsdp:
        if fsdp is not None and fsdp.factor > 1:
            raise ValueError(f"is_fsdp=False but parallel_config shards weights over mesh axis {fsdp.mesh_axis}")
        return parallel_config
    if fsdp is not None:
        return parallel_config
    shape = tuple(int(s) for s in mesh_shape)
    if len(shape) != 2:
        raise ValueError(f"FSDP placement needs a 2-D mesh, got shape {shape}")
    other = 1 - tp.mesh_axis
    if shape[other] == 1:
        logger.info(f"text encoder FSDP disabled: mesh axis {other} of shape {shape} holds one device")
        return parallel_config
    return parallel_config._replace(fsdp=ParallelFactor(factor=shape[other], mesh_axis=other))


class Flux2KleinPromptEncoder:
    """Qwen3 on the mesh (or the torch model for ``use_torch_encoder``) behind one ``encode`` call."""

    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice | None,
        ccl_manager: CCLManager | None,
        parallel_config: EncoderParallelConfig | None,
        use_torch_encoder: bool = False,
        out_layers: Sequence[int] = FLUX2_KLEIN_TEXT_OUT_LAYERS,
        cache_name: str | None = None,
        linear_math_fidelity: ttnn.MathFidelity | None = None,
        num_layers: int | None = None,
        final_norm: bool = False,
    ) -> None:
        """``checkpoint_name`` is a snapshot directory or repository id holding ``tokenizer/`` and ``text_encoder/``;
        ``cache_name`` names the weight-cache entry (required for a local PATH, see ``Qwen3Checkpoint.build``);
        ``linear_math_fidelity`` is the matmul fidelity of the encoder's linears (``None``: the generic HiFi2).
        ``num_layers`` is the device encoder's depth, default ``max(out_layers)`` (the layers the taps need, 27 of
        36); ``final_norm`` (default off) keeps the checkpoint's final RMSNorm, which must stay OFF for the deepest
        tap to be the raw output of layer ``num_layers - 1`` (with it on, the generic forward would hand back the
        normed state for that index). ``num_layers=<num_hidden_layers>, final_norm=True`` is the full-depth
        checkpoint (the reference configuration of ``test_prompt_encoder_flux2_klein.py``)."""
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._out_layers = tuple(sorted(out_layers))
        if not self._out_layers or self._out_layers[0] < 1:
            raise ValueError(f"out_layers must be hidden_states indices >= 1, got {out_layers}")
        self._num_layers = self._out_layers[-1] if num_layers is None else int(num_layers)
        if self._num_layers < self._out_layers[-1]:
            raise ValueError(f"num_layers {self._num_layers} < the deepest tap {self._out_layers[-1]}")
        if final_norm and self._num_layers == self._out_layers[-1]:
            raise ValueError(
                f"final_norm with num_layers == deepest tap {self._num_layers}: hidden_states[{self._num_layers}] "
                "would be the normed state, not the raw output of the last layer"
            )
        self._final_norm = bool(final_norm)
        self._checkpoint_name = checkpoint_name
        self._cache_name = cache_name
        # Per-request constants kept on the device (module docstring): RoPE tables per sequence length, attention
        # biases per (batch, sequence length, real lengths) in an LRU.
        self._pos_embeds: dict[int, tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        self._attn_biases: OrderedDict[tuple[int, int, tuple[int, ...]], ttnn.Tensor] = OrderedDict()
        self._bias_cache_size = FLUX2_KLEIN_ENCODER_BIAS_CACHE_SIZE
        self._constants_hits = 0
        self._constants_misses = 0
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")
        if use_torch_encoder:
            self._encoder = transformers.AutoModelForCausalLM.from_pretrained(
                checkpoint_name, subfolder=FLUX2_KLEIN_TEXT_SUBFOLDER, torch_dtype=torch.bfloat16
            ).eval()
            self._checkpoint = None
        else:
            self._checkpoint = Qwen3Checkpoint(
                checkpoint_name,
                subfolder=FLUX2_KLEIN_TEXT_SUBFOLDER,
                linear_math_fidelity=linear_math_fidelity,
                num_layers=self._num_layers,
                final_norm=self._final_norm,
            )
            self._encoder = None

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def out_layers(self) -> tuple[int, ...]:
        return self._out_layers

    @property
    def parallel_config(self) -> EncoderParallelConfig | None:
        return self._parallel_config

    @property
    def num_layers(self) -> int:
        """Decoder layers the device encoder runs (the torch encoder always runs the whole checkpoint)."""
        return self._num_layers

    @property
    def final_norm(self) -> bool:
        return self._final_norm

    @property
    def cache_subfolder(self) -> str | None:
        """The weight-cache subfolder of the device encoder (``text_encoder_L27`` at the default depth)."""
        return None if self._checkpoint is None else self._checkpoint.cache_subfolder

    @property
    def hidden_size(self) -> int:
        if self._checkpoint is not None:
            return self._checkpoint.config.embed_size
        return self._encoder.config.hidden_size

    @property
    def joint_dim(self) -> int:
        return self.hidden_size * len(self._out_layers)

    @property
    def head_dim(self) -> int:
        if self._checkpoint is not None:
            return self._checkpoint.config.head_size
        return self._encoder.config.head_dim

    @property
    def rope_theta(self) -> float:
        if self._checkpoint is not None:
            return self._checkpoint.config.rope_config.theta
        return _rope_theta(self._encoder.config)

    def rope_tables(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The host-built HF-exact ``(cos, sin)`` of ``positions`` (:func:`qwen3_rope_tables`)."""
        return qwen3_rope_tables(positions, head_dim=self.head_dim, rope_theta=self.rope_theta)

    def load_weights(self) -> None:
        if self._checkpoint is not None and self._encoder is None:
            assert self._device is not None
            self._encoder = self._checkpoint.build(
                device=self._device,
                parallel_config=self._parallel_config,
                ccl_manager=self._ccl_manager,
                cache_name=self._cache_name,
            )

    def deallocate_weights(self) -> None:
        """Free the device encoder's weights and cached constants (the pipeline's ``close``); the tokenizer and
        config stay."""
        self.clear_constants()
        if isinstance(self._encoder, Qwen3Encoder):
            self._encoder.deallocate_weights()

    # --- per-request constants on the device (module docstring) -------------------------------------------------------

    def constants_info(self) -> dict:
        """``{'pos_embeds': [sequence lengths cached], 'attn_biases': [(batch, L, real lengths)...], 'bias_cache_size',
        'hits', 'misses'}``: one hit or miss per encoded batch for the bias (the tables are a one-off per length)."""
        return {
            "pos_embeds": sorted(self._pos_embeds),
            "attn_biases": list(self._attn_biases),
            "bias_cache_size": self._bias_cache_size,
            "hits": self._constants_hits,
            "misses": self._constants_misses,
        }

    def clear_constants(self) -> None:
        """Free the cached device RoPE tables and attention biases (they are rebuilt on the next encode)."""
        for cos, sin in self._pos_embeds.values():
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        self._pos_embeds.clear()
        for bias in self._attn_biases.values():
            ttnn.deallocate(bias)
        self._attn_biases.clear()

    def _device_pos_embeds(self, sequence_length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The HF-exact ``(cos, sin)`` of ``arange(sequence_length)`` on the device, uploaded once per length (HF
        positions are ``arange(L)`` whatever the mask: right padding)."""
        cached = self._pos_embeds.get(sequence_length)
        if cached is None:
            cos, sin = self.rope_tables(torch.arange(sequence_length))
            cached = (tensor.from_torch(cos, device=self._device), tensor.from_torch(sin, device=self._device))
            self._pos_embeds[sequence_length] = cached
        return cached

    def _device_attn_bias(self, mask: torch.Tensor) -> tuple[ttnn.Tensor, bool]:
        """``(bias, cached)``: the ``[B, 1, L, L]`` bf16 causal + padding bias of ``mask`` ``[B, L]`` on the device
        (``encoders.transformer.attention_bias``, the values the generic encoder builds from the mask), from the LRU
        when the right-padded lengths were seen before; ``cached`` False means the caller frees it (a mask that is not
        right padded is uploaded uncached)."""
        lengths = real_lengths(mask)
        key = None if lengths is None else (int(mask.shape[0]), int(mask.shape[1]), lengths)
        if key is not None and key in self._attn_biases:
            self._attn_biases.move_to_end(key)
            self._constants_hits += 1
            return self._attn_biases[key], True
        self._constants_misses += 1
        bias = tensor.from_torch(attention_bias(mask), device=self._device)
        if key is None:
            return bias, False
        self._attn_biases[key] = bias
        while len(self._attn_biases) > self._bias_cache_size:
            _, evicted = self._attn_biases.popitem(last=False)
            ttnn.deallocate(evicted)
        return bias, True

    def _forward_device(
        self, tokens: torch.Tensor, mask: torch.Tensor, *, sequence_length: int, traced: bool
    ) -> ttnn.Tensor:
        """The device encoder on ``(input_ids, attention_mask)`` ``[B, L]``: the taps concatenated per token,
        ``[B, L, joint_dim]`` bf16 replicated on every chip (TP all-gathered, FSDP replicated); the tokens and (eager)
        the taps are freed before the return, the RoPE tables and the attention bias stay cached on the device."""
        assert isinstance(self._encoder, Qwen3Encoder) and self._device is not None
        tt_tokens = tensor.from_torch(tokens, device=self._device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos_embeds = self._device_pos_embeds(sequence_length)
        attn_bias, bias_cached = self._device_attn_bias(mask)
        # A container of layer indices returns ONLY those states, in layer order (the generic encoder
        # compacts the list), so index by position; ``_out_layers`` is kept sorted for that. Index
        # ``num_layers`` (the deepest tap at the default depth) is the raw output of the last layer: no final norm.
        hidden = self._encoder.forward(
            tt_tokens,
            attn_bias=attn_bias,
            pos_embeds=pos_embeds,
            skip_final_linear=True,
            output_hidden_states=set(self._out_layers),
            traced=traced,
        )
        tt_embeds = ttnn.concat(list(hidden), dim=-1)
        ttnn.deallocate(tt_tokens)
        if not bias_cached:
            ttnn.deallocate(attn_bias)
        if not traced:  # a trace's outputs are its persistent buffers and stay allocated
            for tap in hidden:
                ttnn.deallocate(tap)
        return tt_embeds

    def encode(
        self,
        prompts: Sequence[str],
        *,
        sequence_length: int = FLUX2_KLEIN_TEXT_SEQUENCE_LENGTH,
        traced: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(prompt_embeds [B, L, joint_dim] bf16, attention_mask [B, L])`` on the host."""
        tokens, mask = tokenize(self._tokenizer, prompts, sequence_length=sequence_length)
        if isinstance(self._encoder, Qwen3Encoder):
            tt_embeds = self._forward_device(tokens, mask, sequence_length=sequence_length, traced=traced)
            # Replicated over TP (and FSDP): one device holds the full rows.
            embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_embeds)[0])
            ttnn.deallocate(tt_embeds)
        else:
            self.load_weights()
            with torch.no_grad():
                out = self._encoder(tokens, attention_mask=mask, output_hidden_states=True)
            embeds = torch.concat([out.hidden_states[k] for k in self._out_layers], dim=-1)
        return embeds.to(torch.bfloat16), mask

    def encode_device(
        self,
        prompts: Sequence[str],
        *,
        sequence_length: int = FLUX2_KLEIN_TEXT_SEQUENCE_LENGTH,
        sp_axis: int | None = None,
        with_host: bool = False,
        traced: bool = False,
    ) -> tuple[ttnn.Tensor, torch.Tensor | None, torch.Tensor]:
        """``(prompt_embeds, host_rows, attention_mask)`` with the embeddings LEFT ON THE DEVICE: ``[B, L, joint_dim]``
        bf16 replicated on every chip, or, with ``sp_axis``, ``ttnn.mesh_partition(dim=1, cluster_axis=sp_axis)`` of
        it (each device keeps the ``L / SP`` rows of its SP position, replicated over the other axis: the
        ``[None, sp, None]`` placement ``Flux2KleinTransformer.embed_prompt`` takes under ``shard_prompt``). A
        per-device slice: the values are those of :meth:`encode` bit for bit. ``with_host`` also reads the
        ``[B, L, joint_dim]`` rows from one replica BEFORE the partition (the recorded text of a result); ``None``
        otherwise. The caller owns the returned device tensor. Requires the device encoder."""
        if not isinstance(self._encoder, Qwen3Encoder):
            raise ValueError("encode_device needs the device encoder (use_torch_encoder=False)")
        if sp_axis is not None:
            shape = tuple(int(s) for s in self._device.shape)
            if sp_axis not in range(len(shape)):
                raise ValueError(f"sp_axis {sp_axis} is not an axis of the mesh {shape}")
            if sequence_length % (32 * shape[sp_axis]):
                raise ValueError(
                    f"{sequence_length} prompt rows do not partition over SP {shape[sp_axis]} (mesh axis {sp_axis}) "
                    "in tile rows"
                )
        tokens, mask = tokenize(self._tokenizer, prompts, sequence_length=sequence_length)
        tt_embeds = self._forward_device(tokens, mask, sequence_length=sequence_length, traced=traced)
        host = ttnn.to_torch(ttnn.get_device_tensors(tt_embeds)[0]).to(torch.bfloat16) if with_host else None
        if sp_axis is not None:
            sharded = ttnn.mesh_partition(tt_embeds, dim=1, cluster_axis=sp_axis)
            ttnn.deallocate(tt_embeds)
            tt_embeds = sharded
        return tt_embeds, host, mask


def build_flux2_klein_text_encoder(
    snapshot: str | os.PathLike,
    *,
    mesh_device: ttnn.MeshDevice,
    parallel_config: EncoderParallelConfig,
    ccl_manager: CCLManager,
    is_fsdp: bool = True,
    load_weights: bool = True,
    linear_math_fidelity: ttnn.MathFidelity | None = None,
    num_layers: int | None = None,
    final_norm: bool = False,
) -> Flux2KleinPromptEncoder:
    """The snapshot's Qwen3 text encoder on ``mesh_device``: tensor parallel as ``parallel_config`` says and, with
    ``is_fsdp``, weight-sharded over the other mesh axis (:func:`fsdp_parallel_config`). Weights go through
    ``utils.cache.load_model`` under :func:`text_encoder_cache_name` / the depth's cache subfolder
    (``text_encoder_L27`` by default, ``text_encoder`` for the full checkpoint) with the ``_FSDP`` key suffix when
    sharded. ``num_layers`` / ``final_norm`` as :class:`Flux2KleinPromptEncoder` (default: the 27 layers the taps
    need, no final norm). ``load_weights=False`` returns the host half alone (tokenizer, chat template, config);
    ``load_weights()`` later uploads the encoder."""
    snapshot = Path(snapshot)
    parallel_config = fsdp_parallel_config(parallel_config, tuple(mesh_device.shape), is_fsdp=is_fsdp)
    encoder = Flux2KleinPromptEncoder(
        checkpoint_name=str(snapshot),
        device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        cache_name=text_encoder_cache_name(snapshot),
        linear_math_fidelity=linear_math_fidelity,
        num_layers=num_layers,
        final_norm=final_norm,
    )
    if load_weights:
        encoder.load_weights()
        logger.info(
            f"FLUX.2 Klein text encoder: {encoder.hidden_size} hidden, taps {encoder.out_layers}, joint "
            f"{encoder.joint_dim}, {encoder.num_layers} layers{' + final norm' if encoder.final_norm else ''} "
            f"(cache {encoder.cache_subfolder}), parallel {parallel_config}, linear fidelity {linear_math_fidelity}"
        )
    return encoder


@dataclass(frozen=True)
class Flux2KleinDeviceText:
    """One prompt's encoding left on the device (:meth:`Flux2KleinTextEncoding.encode_device`): ``tokens`` ``[1, L,
    joint_dim]`` bf16 replicated, or SP-partitioned on dim 1 over mesh axis ``sp_axis`` (``[1, L / SP, joint_dim]`` per
    device); ``ids`` the text ids; ``rows`` the ``[L, joint_dim]`` host rows when they were read (``None``
    otherwise). ``pair`` is the denoiser's ``text`` input; the caller owns ``tokens`` until the denoiser consumes
    it."""

    tokens: ttnn.Tensor
    ids: torch.Tensor
    rows: torch.Tensor | None
    sp_axis: int | None

    @property
    def pair(self) -> tuple[ttnn.Tensor, torch.Tensor]:
        return self.tokens, self.ids

    @property
    def host(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(rows, ids)`` as ``encode_host`` returns them, or ``None`` when the rows were not read."""
        return None if self.rows is None else (self.rows, self.ids)


class Flux2KleinTextEncoding:
    """One layout's prompt front: a prompt string in, the DiT's prompt rows and ids out.

    ``encode_device(prompt)`` leaves the rows on the device (:class:`Flux2KleinDeviceText`): with ``sp_axis`` (the
    layout's sequence-parallel mesh axis; the pipeline passes it under ``shard_prompt``) the ``[1, L, joint_dim]``
    bf16 output is ``ttnn.mesh_partition`` ed over SP on dim 1, exactly the placement the denoiser's hoisted
    ``embed_prompt`` reads, so no host round trip (a 7.9 MB replica read, host tilize and 32 shard writes per request
    on the 4B) sits between the encoder and the DiT. ``encode_host(prompt)`` returns ``(tokens, ids)``: ``tokens``
    ``[sequence_length, joint_dim]`` torch.bfloat16 (the reference's ``prompt_embeds[0]``, read from replica 0) and
    ``ids = packing.text_ids(sequence_length)`` ``[sequence_length, 4]`` int64 (``P:267-283``, ``(0, 0, 0, l)``), the
    same tensor on every call. The pipeline holds one per layout next to its cached ``""`` host encoding for
    classifier-free guidance.
    """

    def __init__(
        self,
        encoder: Flux2KleinPromptEncoder,
        *,
        text_ids: torch.Tensor | None = None,
        sequence_length: int = FLUX2_KLEIN_TEXT_SEQUENCE_LENGTH,
        sp_axis: int | None = None,
    ) -> None:
        """``sp_axis``: the DiT's sequence-parallel mesh axis when ``encode_device`` is to partition the rows over it
        (``sequence_length`` must be a multiple of 32 x SP: 512 over SP 4 and 8); ``None`` leaves them replicated."""
        ids = packing.text_ids(sequence_length) if text_ids is None else text_ids
        if tuple(ids.shape) != (sequence_length, 4) or ids.dtype != torch.int64:
            raise ValueError(f"text_ids must be int64 [{sequence_length}, 4], got {ids.dtype} {tuple(ids.shape)}")
        if sp_axis is not None:
            device = encoder._device
            if device is None:
                raise ValueError("sp_axis needs the device encoder")
            shape = tuple(int(s) for s in device.shape)
            if sp_axis not in range(len(shape)):
                raise ValueError(f"sp_axis {sp_axis} is not an axis of the mesh {shape}")
            if sequence_length % (32 * shape[sp_axis]):
                raise ValueError(
                    f"{sequence_length} prompt rows do not partition over SP {shape[sp_axis]} (mesh axis {sp_axis}) "
                    "in tile rows"
                )
        self._encoder = encoder
        self._length = sequence_length
        self._ids = ids
        self._sp_axis = sp_axis

    @property
    def encoder(self) -> Flux2KleinPromptEncoder:
        return self._encoder

    @property
    def sp_axis(self) -> int | None:
        """The mesh axis ``encode_device`` partitions the rows over (``None``: replicated)."""
        return self._sp_axis

    @property
    def joint_dim(self) -> int:
        return self._encoder.joint_dim

    @property
    def sequence_length(self) -> int:
        return self._length

    @property
    def ids(self) -> torch.Tensor:
        return self._ids

    def encode_host(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(tokens [sequence_length, joint_dim] bf16, ids [sequence_length, 4] int64)`` of one prompt."""
        embeds, _ = self._encoder.encode([prompt], sequence_length=self._length)
        return embeds[0].contiguous(), self._ids

    def encode_device(self, prompt: str, *, with_host: bool = False) -> Flux2KleinDeviceText:
        """One prompt's rows on the device, partitioned over ``sp_axis`` when set (class docstring); ``with_host``
        also reads the ``[sequence_length, joint_dim]`` host rows (one replica, before the partition)."""
        tokens, rows, _ = self._encoder.encode_device(
            [prompt], sequence_length=self._length, sp_axis=self._sp_axis, with_host=with_host
        )
        return Flux2KleinDeviceText(
            tokens=tokens, ids=self._ids, rows=None if rows is None else rows[0].contiguous(), sp_axis=self._sp_axis
        )


__all__ = [
    "FLUX2_KLEIN_ENCODER_BIAS_CACHE_SIZE",
    "FLUX2_KLEIN_TEXT_ENCODER_LAYERS",
    "FLUX2_KLEIN_TEXT_OUT_LAYERS",
    "FLUX2_KLEIN_TEXT_SEQUENCE_LENGTH",
    "FLUX2_KLEIN_TEXT_SUBFOLDER",
    "Flux2KleinDeviceText",
    "Flux2KleinPromptEncoder",
    "Flux2KleinTextEncoding",
    "build_flux2_klein_text_encoder",
    "format_prompts",
    "fsdp_parallel_config",
    "qwen3_rope_tables",
    "real_lengths",
    "text_encoder_blob",
    "text_encoder_cache_name",
    "tokenize",
]
