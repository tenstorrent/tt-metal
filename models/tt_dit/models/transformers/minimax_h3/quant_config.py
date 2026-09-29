# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Weight-dtype policy for the MiniMax-H3 DiT linears.

Same shape as ``models/tt_dit/models/transformers/ltx/quant_config.py``: one frozen profile holds
the whole policy and hands it to the modules as construction kwargs, so the transformer, the block
and the attention stay precision-agnostic and carry no dtype literals of their own.

Why construction time and not a state-dict pass: there is no Torch bfloat8, so ``ttnn.from_torch``
is the only quantizer and ``Parameter`` enforces the dtype it was declared with. Setting the dtype
at build time therefore means a cache miss loads each shard *direct to quant* and the cache write
holds quantized tensorbins -- the bf16 copy of a 38.5 GB block stack is never resident on device,
which is the whole point on a 32 GB part.

The arithmetic this exists for (per block, 50 blocks): to_qkv 5376x21504, to_out 7168x5376,
ff1 5376x28672 (packed [gate|up]), ff2 14336x5376 -- 385.3 M parameters, 19.27 B over the stack,
38.5 GB at bf16 against 31.831 GiB of p150 DRAM. ``bf8_weights`` puts that at 22.3 GB.

No profile means no change: every mesh that does not ask for one builds exactly the bf16 model it
built before.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn

_DTYPE_TAGS = {
    ttnn.bfloat16: "bf16",
    ttnn.bfloat8_b: "bf8",
    ttnn.bfloat4_b: "bf4",
    ttnn.float32: "fp32",
}

# Bytes per element including the block-float exponent overhead (one fp8 exponent per 16-element
# block, i.e. 1/16 byte per element on top of the mantissa). Used only by `stack_bytes`, which is a
# planning aid for the memory budget, not something the model reads.
_DTYPE_BYTES = {
    ttnn.bfloat16: 2.0,
    ttnn.bfloat8_b: 1.0 + 1.0 / 16,
    ttnn.bfloat4_b: 0.5 + 1.0 / 16,
    ttnn.float32: 4.0,
}


@dataclass(frozen=True)
class MiniMaxH3QuantProfile:
    """Weight dtypes for one DiT block's four linears, plus the optional activation cast.

    ``out_dtype`` is a carve-out rather than a free knob: ``to_out`` feeds the fused
    ``addcmul_residual + to_out(...) * addcmul_gate`` epilogue, whose bf16 ternary inputs have to
    match the weight tile format (the same carve-out LTX makes, for the same reason).

    ``bf16_blocks`` names block indices that keep the bf16 model wholesale -- Python indexing, so
    ``(-1,)`` is the last block. It is the escape hatch for the datatype sweep: if quantizing costs
    more accuracy than the budget allows, pinning the first and/or last block back to bf16 buys it
    at 385 MB a block.

    ``activation_dtype`` / ``pin_output_bf16`` are off in every shipped profile here and exist so a
    TP>1 mesh can reuse this class. The cast only pays for itself when the activation is the payload
    of a fused all-gather; at TP=1 there is no gather, so narrowing the activation buys no traffic
    and costs a full typecast pass plus precision.
    """

    name: str
    qkv_dtype: ttnn.DataType
    out_dtype: ttnn.DataType
    ff_dtype: ttnn.DataType
    activation_dtype: ttnn.DataType | None = None
    pin_output_bf16: bool = False
    bf16_blocks: tuple[int, ...] = ()

    @property
    def cache_tag(self) -> str:
        """Short, stable string for the device-weight cache path.

        Derived from the dtypes themselves and not from ``name``, so two profiles cannot share a
        cache directory just because someone reused a name -- the cached tensorbins ARE the
        quantized weights, and reading bf8 bins into a bf16 Parameter is a silent wrong answer.
        """
        parts = [
            f"q{_DTYPE_TAGS[self.qkv_dtype]}",
            f"o{_DTYPE_TAGS[self.out_dtype]}",
            f"f{_DTYPE_TAGS[self.ff_dtype]}",
        ]
        if self.activation_dtype is not None:
            parts.append(f"a{_DTYPE_TAGS[self.activation_dtype]}")
        if self.pin_output_bf16:
            parts.append("pin")
        if self.bf16_blocks:
            parts.append("keep" + "_".join(str(i) for i in self.bf16_blocks))
        return "-".join(parts)

    def keeps_block_bf16(self, index: int, num_layers: int) -> bool:
        """Is block ``index`` pinned to bf16? Negative entries count from the end."""
        return any(i % num_layers == index for i in self.bf16_blocks)

    def for_block(self, index: int, num_layers: int) -> MiniMaxH3QuantProfile:
        """This profile as block ``index`` should see it -- all-bf16 when the block is pinned."""
        if not self.keeps_block_bf16(index, num_layers):
            return self
        return replace(
            self,
            qkv_dtype=ttnn.bfloat16,
            out_dtype=ttnn.bfloat16,
            ff_dtype=ttnn.bfloat16,
            activation_dtype=None,
            pin_output_bf16=False,
        )

    def attention_kwargs(self) -> dict:
        """Construction kwargs for ``MiniMaxH3Attention``."""
        return {
            "qkv_dtype": self.qkv_dtype,
            "out_dtype": self.out_dtype,
            "activation_dtype": self.activation_dtype,
            "pin_output_bf16": self.pin_output_bf16,
        }

    def ffn_kwargs(self) -> dict:
        """Construction kwargs for ``ParallelFeedForward``."""
        return {
            "ff1_dtype": self.ff_dtype,
            "ff2_dtype": self.ff_dtype,
            "activation_dtype": self.activation_dtype,
            "pin_output_bf16": self.pin_output_bf16,
        }

    def stack_bytes(self, *, hidden_size: int, inner_dim: int, ffn_dim: int, num_layers: int) -> float:
        """Device bytes the block stack's four linears take under this profile.

        A planning aid for the DRAM budget -- nothing in the model reads it -- but it is the number
        that decides whether a canvas fits, so it lives next to the policy that sets it.
        """
        total = 0.0
        for index in range(num_layers):
            p = self.for_block(index, num_layers)
            total += hidden_size * 3 * inner_dim * _DTYPE_BYTES[p.qkv_dtype]
            total += inner_dim * hidden_size * _DTYPE_BYTES[p.out_dtype]
            total += hidden_size * 2 * ffn_dim * _DTYPE_BYTES[p.ff_dtype]
            total += ffn_dim * hidden_size * _DTYPE_BYTES[p.ff_dtype]
        return total

    @staticmethod
    def bf8_weights(bf16_blocks: tuple[int, ...] = ()) -> MiniMaxH3QuantProfile:
        """The p150 policy: bf8 qkv/ff1/ff2, bf16 ``to_out``, activations untouched.

        22.3 GB over the 50-block stack, or 22.7 GB with one block pinned back to bf16.
        """
        return MiniMaxH3QuantProfile(
            name="bf8_weights",
            qkv_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat16,
            ff_dtype=ttnn.bfloat8_b,
            bf16_blocks=bf16_blocks,
        )

    @staticmethod
    def bf8_weights_bf8_out() -> MiniMaxH3QuantProfile:
        """``bf8_weights`` with the ``to_out`` carve-out given up too: 20.5 GB.

        1.8 GB cheaper and the fallback if the activation high-water mark does not leave room at
        the canvas we want to serve. It narrows one input of the fused addcmul epilogue, so it has
        to be PCC-checked on its own rather than assumed to behave like ``bf8_weights``.
        """
        return MiniMaxH3QuantProfile(
            name="bf8_weights_bf8_out",
            qkv_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat8_b,
            ff_dtype=ttnn.bfloat8_b,
        )

    @staticmethod
    def bf16() -> MiniMaxH3QuantProfile:
        """The unquantized policy, spelled out. Only useful as an explicit A/B baseline on a mesh
        with the DRAM for it -- passing no profile at all gives the same model."""
        return MiniMaxH3QuantProfile(
            name="bf16",
            qkv_dtype=ttnn.bfloat16,
            out_dtype=ttnn.bfloat16,
            ff_dtype=ttnn.bfloat16,
        )


# Explicit registry rather than getattr on the class: the staticmethods share a namespace with the
# instance methods, so a typo in a preset name must not be able to reach `attention_kwargs`.
PRESETS = {
    "bf8_weights": MiniMaxH3QuantProfile.bf8_weights,
    "bf8_weights_bf8_out": MiniMaxH3QuantProfile.bf8_weights_bf8_out,
    "bf16": MiniMaxH3QuantProfile.bf16,
}


def resolve_quant_profile(spec) -> MiniMaxH3QuantProfile | None:
    """``None`` / a preset name / a profile -> a profile or ``None`` (meaning "build bf16")."""
    if spec is None or isinstance(spec, MiniMaxH3QuantProfile):
        return spec
    if spec in PRESETS:
        return PRESETS[spec]()
    raise ValueError(f"unknown MiniMax-H3 quant profile {spec!r}; known: {sorted(PRESETS)}")
