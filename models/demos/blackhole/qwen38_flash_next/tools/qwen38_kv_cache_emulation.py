# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The device's sparse-attention KV cache formats, applied to the CPU oracle's K and V at cache-write time (outside
``tt/``, which is never edited).

``ttnn.transformer.sparse_sdpa`` stores one ``[V | K]`` row of 512 values per token per device.  ``bf16`` is what the
oracle already does.  ``scaled_fp8`` is the format the device builds today (``per_token_cast_to_fp8`` with power-of-two
scales + ``pack_scaled_fp8_kv_cache``, the whole row as the 512-wide latent field): per 128-element block one fp32
scale ``2 ** ceil(log2(clamp(max|x|, 1e-4) * fp32(1/448)))`` and E4M3FN codes of ``x / scale`` as the Blackhole packer
writes them, measured bitwise by the device half on 134M real elements: the mantissa TRUNCATED toward zero to three
bits, a value exactly on a code falling one code toward zero (the SFPU reciprocal sits a little below 1 / scale), one
subnormal binade (a magnitude in [2^-7, 2^-6) keeps its three truncated mantissa bits under a zero exponent, a code
``m * 2^-9``) and a signed zero below 2^-7.  The kernel then tilizes the codes into bfp8_b tiles (16 consecutive
elements share the block's largest exponent, 7 mantissa bits), multiplies by the scale and packs the product to bfp8_b
again before the two matmuls.  ``scaled_fp8_rne`` is the same layout with a hypothetical nearest-even packer
(tt-metal's host conversion ``float8.cpp``: round to nearest even, saturate to +-448, flush below 2^-6).  ``fp8_e4m3``
is the plain 512-byte e4m3 row the kernel also reads (nearest-even storage, then the bfp8_b tilize) and
``fp8_e4m3_storage`` its storage rounding alone; no in-tree Blackhole op writes that row today, so both are what-ifs.
The bfp8_b rounding follows the host packer (round to nearest even, clamp at 127) unless ``truncate`` is asked for.
Every emulated value has at most seven significant bits, so it is exact in the oracle's bf16 cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.reference import qsa_selected_token_mask, zero_centered_rms_norm
from models.demos.blackhole.qwen38_flash_next.tt.qsa import Qwen38QSA, Qwen38QSAState, Qwen38QSAWeights, _partial_rope

E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0**-6
E4M3_SUBNORMAL_STEP = 2.0**-9
E4M3_MAX_CODE = 0x7E
E4M3_NAN_CODE = 0x7F
SCALE_BLOCK = 128
SCALE_CLAMP_MIN = 1.0e-4
SCALE_CLAMP_MAX = 3.0e38
INV_448 = torch.tensor(1.0 / E4M3_MAX, dtype=torch.float32)  # the kernel's fp32 constant, 0x3B124925
PACKER_FP8_MANTISSA_BITS = 10  # PCK_DEST_RD_CTRL_Round_10b_mant is set for an e4m3 pack destination
BFP8_BLOCK = 16
BFP8_MANTISSA_BITS = 7
KV_FORMATS = ("bf16", "scaled_fp8", "scaled_fp8_rne", "fp8_e4m3", "fp8_e4m3_storage")
BFP8_ROUNDINGS = ("rne", "truncate")


def _float32_from_bits(bits: torch.Tensor) -> torch.Tensor:
    """int64 holding an unsigned 32-bit pattern -> float32 with that pattern."""
    signed = torch.where(bits >= 2**31, bits - 2**32, bits)
    return signed.to(torch.int32).view(torch.float32)


def _bits_from_float32(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def round_mantissa(values: torch.Tensor, bits: int) -> torch.Tensor:
    """fp32 with the mantissa rounded to nearest even at ``bits`` fraction bits (a carry moves into the exponent, as
    the packer's 10-bit dest read does); NaN and infinities pass through."""
    x = values.detach().to(torch.float32)
    pattern = _bits_from_float32(x)
    sign = pattern & 0x80000000
    magnitude = pattern & 0x7FFFFFFF
    shift = 23 - bits
    lower = magnitude & ((1 << shift) - 1)
    kept = magnitude >> shift
    half = 1 << (shift - 1)
    round_up = (lower > half) | ((lower == half) & (kept & 1 == 1))
    kept = kept + round_up.to(torch.int64)
    rounded = _float32_from_bits(sign | (kept << shift))
    return torch.where(torch.isfinite(x), rounded, x)


def fp8_e4m3_codes(values: torch.Tensor) -> torch.Tensor:
    """``float8.cpp`` fp32 -> E4M3FN bits: NaN -> 0x7F; +-inf and every |x| >= 464 -> +-0x7E (448); |x| < 2^-6 (an fp8
    exponent field <= 0, decided before rounding) -> the signed zero; otherwise round to nearest even on three
    mantissa bits, a carry landing on E=15/M=7 saturates to 448."""
    x = values.detach().to(torch.float32)
    pattern = _bits_from_float32(x)
    sign = (pattern >> 31) << 7
    exponent = (pattern >> 23) & 0xFF
    mantissa = pattern & 0x7FFFFF
    fp8_exponent = exponent - 127 + 7
    top = mantissa >> 20
    round_bit = (mantissa >> 19) & 1
    sticky = mantissa & 0x7FFFF
    round_up = (round_bit == 1) & ((sticky != 0) | (top & 1 == 1))
    top = top + round_up.to(torch.int64)
    carry = top == 8
    top = torch.where(carry, torch.zeros_like(top), top)
    rounded_exponent = fp8_exponent + carry.to(torch.int64)
    saturate = (rounded_exponent > 15) | ((rounded_exponent == 15) & (top == 7))
    code = torch.where(saturate, torch.full_like(top, E4M3_MAX_CODE), (rounded_exponent << 3) | top)
    code = torch.where(fp8_exponent <= 0, torch.zeros_like(code), code)
    code = torch.where(torch.isnan(x), torch.full_like(code, E4M3_NAN_CODE), sign | code)
    return code.to(torch.uint8)


def fp8_e4m3_values(codes: torch.Tensor, *, subnormals: bool = False) -> torch.Tensor:
    """E4M3FN bits -> fp32: 0x7F/0xFF -> NaN; else (-1)^s 2^(E-7) (1 + M/8); an E=0 code is the signed zero (the host
    converter's flush) or, with ``subnormals``, the value ``(-1)^s M * 2^-9`` (torch's float8_e4m3fn)."""
    code = codes.to(torch.int64)
    sign = (code >> 7) & 1
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    pattern = (sign << 31) | ((exponent - 7 + 127) << 23) | (mantissa << 20)
    pattern = torch.where(exponent == 0, sign << 31, pattern)
    values = _float32_from_bits(pattern)
    if subnormals:
        small = (1.0 - 2.0 * sign.to(torch.float32)) * mantissa.to(torch.float32) * E4M3_SUBNORMAL_STEP
        values = torch.where(exponent == 0, small, values)
    return torch.where((exponent == 15) & (mantissa == 7), torch.full_like(values, float("nan")), values)


def fp8_e4m3_codes_device(scaled: torch.Tensor) -> torch.Tensor:
    """The e4m3 byte the Blackhole packer writes for the fp32 product ``x * recip(scale)``, as measured bitwise by the
    device half (every bf16 mantissa of 39 binades at two scales, 262,144 real rows): the mantissa is TRUNCATED toward
    zero to three bits, a value exactly on a code falls one code toward zero (the SFPU reciprocal of the power-of-two
    scale is a little below 1 / scale), a magnitude then in [2^-7, 2^-6) keeps its three truncated mantissa bits under
    a zero exponent field (a subnormal code ``m * 2^-9``), a magnitude below 2^-7 is a signed zero, zero stays a signed
    zero, magnitudes above 448 saturate to 448.  The arithmetic is ``ttnn/qsa.py::e4m3_codes_device`` of the device
    branch, kept in lockstep."""
    value = scaled.detach().to(torch.float32)
    sign = torch.signbit(value).to(torch.uint8) << 7
    magnitude = value.abs()
    saturated = magnitude > E4M3_MAX
    mantissa, exponent = torch.frexp(torch.clamp(magnitude, max=E4M3_MAX))
    power = exponent - 1  # magnitude = (2 * mantissa) * 2**power, 2 * mantissa in [1, 2)
    eighths = (2.0 * mantissa - 1.0) * 8.0
    code_mantissa = torch.floor(eighths)
    exact = (eighths == code_mantissa) & (magnitude > 0)
    code_mantissa = torch.where(exact, code_mantissa - 1, code_mantissa)
    borrow = code_mantissa < 0
    code_mantissa = torch.where(borrow, torch.full_like(code_mantissa, 7.0), code_mantissa)
    power = torch.where(borrow, power - 1, power)
    codes = sign | ((power + 7).clamp(min=0).to(torch.uint8) << 3) | code_mantissa.to(torch.uint8)
    codes = torch.where((magnitude == 0) | (power < -7), sign, codes)
    return torch.where(saturated, sign | 0x7E, codes)


def fp8_e4m3_round(values: torch.Tensor) -> torch.Tensor:
    return fp8_e4m3_values(fp8_e4m3_codes(values))


def bfp8_b_round(values: torch.Tensor, *, rounding: str = "rne") -> torch.Tensor:
    """The Bfp8_b tile value of every element: blocks of 16 consecutive elements along the last dim share the block's
    largest exponent; each keeps a sign and a 7-bit mantissa with an explicit leading one (q * 2^(shared - 6), q in
    0..127), rounded to nearest even and clamped at 127 like the host packer (``rne``) or truncated (``truncate``)."""
    if rounding not in BFP8_ROUNDINGS:
        raise ValueError(f"bfp8 rounding must be one of {BFP8_ROUNDINGS}, got {rounding!r}")
    x = values.detach().to(torch.float32)
    width = x.shape[-1]
    if width % BFP8_BLOCK:
        raise ValueError(f"bfp8 blocks need {BFP8_BLOCK} | width, got {width}")
    blocks = x.reshape(*x.shape[:-1], width // BFP8_BLOCK, BFP8_BLOCK)
    magnitude = blocks.abs()
    _, exponent = torch.frexp(magnitude)  # |x| = m * 2^e, m in [0.5, 1): the value's exponent is e - 1
    exponent = torch.where(magnitude > 0, exponent - 1, torch.full_like(exponent, -140))
    shared = exponent.amax(dim=-1, keepdim=True).clamp(min=-140)
    scale = torch.exp2((shared - (BFP8_MANTISSA_BITS - 1)).float())
    q = magnitude / scale
    q = torch.round(q) if rounding == "rne" else torch.floor(q)
    q = q.clamp(max=float(2**BFP8_MANTISSA_BITS - 1))
    return (torch.sign(blocks) * q * scale).reshape(x.shape)


def scaled_fp8_scales(values: torch.Tensor, *, power_of_two: bool) -> torch.Tensor:
    """``per_token_cast_to_fp8``'s scale per 128-element block along the last dim: ``fp32(clamp(max|x|, 1e-4, 3e38) *
    fp32(1/448))`` (one fp32 multiply, as the kernel); with ``power_of_two`` (``round_scale_to_power_of_two=True``, what
    the device build passes) an exact power of two is kept and anything else becomes the next power of two up.
    Returns ``[..., H/128]`` fp32."""
    x = values.detach().to(torch.float32)
    width = x.shape[-1]
    if width % SCALE_BLOCK:
        raise ValueError(f"scaled fp8 blocks need {SCALE_BLOCK} | width, got {width}")
    blocks = x.reshape(*x.shape[:-1], width // SCALE_BLOCK, SCALE_BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp(min=SCALE_CLAMP_MIN, max=SCALE_CLAMP_MAX)
    scale = amax * INV_448
    if power_of_two:
        m, e = torch.frexp(scale)  # scale = m * 2^e, m in [0.5, 1): exactly 2^(e-1) when m == 0.5
        scale = torch.ldexp(torch.ones_like(scale), torch.where(m == 0.5, e - 1, e))
    return scale


def scaled_fp8_quantize_device(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The device's quantisation of K/V rows (``emulate_scaled_fp8_rows`` of the device branch, bitwise the packed
    bytes): power-of-two scales and :func:`fp8_e4m3_codes_device` of ``x / scale`` (exact: the scale is a power of
    two).  Returns ``(codes uint8 [..., H], scales fp32 [..., H/128])``."""
    x = values.detach().to(torch.float32)
    scale = scaled_fp8_scales(x, power_of_two=True)
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // SCALE_BLOCK, SCALE_BLOCK)
    return fp8_e4m3_codes_device(blocks / scale.unsqueeze(-1)).reshape(x.shape), scale


def scaled_fp8_quantize(values: torch.Tensor, *, power_of_two_scale: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """The hypothetical nearest-even packer on the same layout: ``codes = e4m3_rne(x * (1/scale))`` with the packer's
    10-bit mantissa rounding of the fp32 product first (``float8.cpp`` semantics: saturate, flush below 2^-6).  The
    device does not round this way (see :func:`fp8_e4m3_codes_device`); this is the "what a nearest-rounding packer
    would give" row.  Returns ``(codes uint8 [..., H], scales fp32 [..., H/128])``."""
    x = values.detach().to(torch.float32)
    scale = scaled_fp8_scales(x, power_of_two=power_of_two_scale)
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // SCALE_BLOCK, SCALE_BLOCK)
    scaled = round_mantissa(blocks * (1.0 / scale.unsqueeze(-1)), PACKER_FP8_MANTISSA_BITS)
    return fp8_e4m3_codes(scaled).reshape(x.shape), scale


def scaled_fp8_dequantize(codes: torch.Tensor, scales: torch.Tensor, *, subnormals: bool = True) -> torch.Tensor:
    """``per_token_cast_back`` / the device branch's ``unpack_scaled_fp8_rows``: decode(codes) * scale, fp32 (subnormal
    codes as their float8_e4m3fn values unless ``subnormals=False``)."""
    decoded = fp8_e4m3_values(codes, subnormals=subnormals)
    blocks = decoded.reshape(*codes.shape[:-1], codes.shape[-1] // SCALE_BLOCK, SCALE_BLOCK)
    return (blocks * scales.unsqueeze(-1).to(torch.float32)).reshape(codes.shape)


@dataclass(frozen=True)
class KVCacheRounding:
    """K and V of one token as the device's cache and kernel present them (``head_dim`` along the last axis)."""

    kv_format: str
    bfp8_rounding: str = "rne"

    def __post_init__(self) -> None:
        if self.kv_format not in KV_FORMATS:
            raise ValueError(f"kv format must be one of {KV_FORMATS}, got {self.kv_format!r}")
        if self.bfp8_rounding not in BFP8_ROUNDINGS:
            raise ValueError(f"bfp8 rounding must be one of {BFP8_ROUNDINGS}, got {self.bfp8_rounding!r}")

    @property
    def identity(self) -> bool:
        return self.kv_format == "bf16"

    def key(self, key: torch.Tensor) -> torch.Tensor:
        return self._round(key)

    def value(self, value: torch.Tensor) -> torch.Tensor:
        return self._round(value)

    def _round(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return x
        values = x.detach().to(torch.float32)
        tilize = lambda v: bfp8_b_round(v, rounding=self.bfp8_rounding)  # noqa: E731
        if self.kv_format == "fp8_e4m3_storage":
            rounded = fp8_e4m3_round(values)
        elif self.kv_format == "fp8_e4m3":
            rounded = tilize(fp8_e4m3_round(values))
        else:
            if self.kv_format == "scaled_fp8":
                codes, scales = scaled_fp8_quantize_device(values)
            else:
                codes, scales = scaled_fp8_quantize(values)
            latent = tilize(fp8_e4m3_values(codes, subnormals=self.kv_format == "scaled_fp8"))
            rounded = tilize(latent * scales.repeat_interleave(SCALE_BLOCK, dim=-1))
        return rounded.to(x.dtype)

    def describe(self) -> str:
        if self.identity:
            return "bf16"
        detail = {
            "scaled_fp8": "the device's cache: e4m3 codes truncated by the Blackhole packer (exact values fall one "
            "code, one subnormal binade), one fp32 power-of-two scale per 128-block, dequantised in bfp8_b",
            "scaled_fp8_rne": "hypothetical nearest-even packer on the scaled layout (float8.cpp rounding), fp32 "
            "power-of-two scale per 128-block, dequantised in bfp8_b",
            "fp8_e4m3": "what-if plain e4m3 row (nearest-even per element), then the kernel's bfp8_b tilize",
            "fp8_e4m3_storage": "what-if plain e4m3 row (nearest-even, saturate 448, flush below 2^-6), no bfp8 tilize",
        }[self.kv_format]
        return f"{self.kv_format}: {detail}; bfp8 rounding {self.bfp8_rounding}"


class Qwen38QSAKVRounded(Qwen38QSA):
    """``Qwen38QSA`` whose current K and V pass through ``rounding`` before they are attended and cached.  The forward
    is the oracle's forward with the two rounding calls after the RoPE and an optional ``capture`` hook (layer index
    and the tensors of the current pass); at the identity it is bitwise the oracle."""

    def __init__(
        self,
        weights: Qwen38QSAWeights,
        rounding: KVCacheRounding,
        capture: Callable[[int, dict[str, Any]], None] | None = None,
    ):
        super().__init__(weights)
        self.rounding = rounding
        self.capture = capture

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        state: Qwen38QSAState | None = None,
    ) -> tuple[torch.Tensor, Qwen38QSAState, torch.Tensor]:
        dims = self.weights.dimensions
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != dims.hidden_size:
            raise ValueError(f"hidden_states must be [batch, sequence, {dims.hidden_size}]")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(f"QSA oracle requires BF16 activations, got {hidden_states.dtype}")
        batch, sequence, _ = hidden_states.shape
        previous_length = 0 if state is None else state.length
        key_length = previous_length + sequence
        if state is not None:
            if state.raw_index_keys.shape != (batch, previous_length, dims.index_head_dim):
                raise ValueError("cached raw QSA index keys do not match the current batch/dimensions")
            if state.keys.shape != (batch, dims.kv_heads, previous_length, dims.head_dim):
                raise ValueError("cached QSA keys do not match the pinned KV dimensions")
        if attention_mask.shape != (batch, 1, sequence, key_length):
            raise ValueError(f"QSA mask must be [B,1,{sequence},{key_length}] for the supplied cache")
        cos, sin = position_embeddings
        if cos.shape != (batch, key_length, dims.rope_dim) or sin.shape != cos.shape:
            raise ValueError(f"position embeddings must both cover the full cache as [B,{key_length},{dims.rope_dim}]")
        current_cos = cos[:, -sequence:]
        current_sin = sin[:, -sequence:]

        index_qk = F.linear(hidden_states, self.weights.index_qk)
        index_query, raw_keys = torch.split(
            index_qk,
            (dims.index_query_heads * dims.index_head_dim, dims.index_kv_heads * dims.index_head_dim),
            dim=-1,
        )
        index_query = index_query.reshape(batch, sequence, dims.index_query_heads, dims.index_head_dim)
        current_raw_keys = raw_keys.reshape(batch, sequence, dims.index_head_dim)
        raw_keys = current_raw_keys if state is None else torch.cat((state.raw_index_keys, current_raw_keys), dim=1)
        selected = qsa_selected_token_mask(
            index_query,
            raw_keys,
            cos,
            sin,
            attention_mask,
            q_norm_weight=self.weights.index_q_norm,
            k_norm_weight=self.weights.index_k_norm,
            token_budget=dims.token_budget,
            compress_ratio=dims.compress_ratio,
            eps=self.weights.rms_norm_eps,
        )
        if attention_mask.is_floating_point():
            combined_mask = attention_mask + selected
            selected_visible = selected == 0
        else:
            combined_mask = attention_mask & selected
            selected_visible = selected

        qg = F.linear(hidden_states, self.weights.qg).view(batch, sequence, dims.query_heads, 2 * dims.head_dim)
        query, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(batch, sequence, dims.query_width)
        query = zero_centered_rms_norm(query, self.weights.q_norm, self.weights.rms_norm_eps)
        key = F.linear(hidden_states, self.weights.k).view(batch, sequence, dims.kv_heads, dims.head_dim)
        key = zero_centered_rms_norm(key, self.weights.k_norm, self.weights.rms_norm_eps)
        value = F.linear(hidden_states, self.weights.v).view(batch, sequence, dims.kv_heads, dims.head_dim)

        query = _partial_rope(query.transpose(1, 2), current_cos, current_sin, unsqueeze_dim=1)
        current_key = _partial_rope(key.transpose(1, 2), current_cos, current_sin, unsqueeze_dim=1)
        current_value = value.transpose(1, 2)
        # The device writes the row to its cache and attends the cached bytes, the current token's own row included.
        exact_key, exact_value = current_key, current_value
        current_key = self.rounding.key(current_key)
        current_value = self.rounding.value(current_value)
        if state is None:
            key, value = current_key, current_value
        else:
            key = torch.cat((state.keys, current_key), dim=-2)
            value = torch.cat((state.values, current_value), dim=-2)
        if self.capture is not None:
            self.capture(
                self.weights.layer_idx,
                {
                    "query": query,
                    "exact_key": exact_key,
                    "exact_value": exact_value,
                    "current_key": current_key,
                    "current_value": current_value,
                    "combined_mask": combined_mask,
                    "previous_length": previous_length,
                },
            )
        repeated_key = key.repeat_interleave(dims.kv_repeat, dim=1)
        repeated_value = value.repeat_interleave(dims.kv_repeat, dim=1)
        scores = torch.matmul(query, repeated_key.transpose(2, 3)) * (dims.head_dim**-0.5)
        if combined_mask.is_floating_point():
            scores = scores + combined_mask
        else:
            scores = scores.masked_fill(~combined_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = (
            torch.matmul(probabilities, repeated_value)
            .transpose(1, 2)
            .contiguous()
            .reshape(batch, sequence, dims.query_width)
        )
        output = output * torch.sigmoid(gate)
        next_state = Qwen38QSAState(raw_index_keys=raw_keys, keys=key, values=value)
        return F.linear(output, self.weights.out), next_state, selected_visible


def install_kv_rounding(
    oracle: Any, rounding: KVCacheRounding, capture: Callable[[int, dict[str, Any]], None] | None = None
) -> None:
    """Route ``oracle.layer`` (``Qwen38TextModelOracle``) through a wrapper that gives every QSA layer the rounding
    subclass over the layer's own weights; composes with other ``layer`` wrappers (the BF4 expert emulation)."""

    original_layer = oracle.layer

    def layer(layer_index: int):
        created = original_layer(layer_index)
        attention = created.attention
        if isinstance(attention, Qwen38QSA) and not isinstance(attention, Qwen38QSAKVRounded):
            created.attention = Qwen38QSAKVRounded(attention.weights, rounding, capture=capture)
        return created

    oracle.layer = layer  # type: ignore[method-assign]
