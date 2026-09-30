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
    # Arithmetic precision, as opposed to storage precision. `None` on both means "whatever the
    # block has always done" -- HiFi2 with fp32 destination accumulation -- so every profile that
    # does not mention them, and the no-profile bf16 path, builds exactly the model it built before.
    #
    # These are a POLICY knob and not a free one: they decide how many mantissa passes the matmul
    # unit makes over a bfloat8_b tile and whether the destination register accumulates in fp32, so
    # they belong beside the weight dtypes rather than hardcoded in the block. Stage 05 listed the
    # fidelity sweep as not-measured precisely because there was no knob here to sweep.
    mm_math_fidelity: ttnn.MathFidelity | None = None
    mm_fp32_dest_acc_en: bool | None = None

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
            # The arithmetic too. A pinned block exists to buy accuracy back, and LoFi on a bf16
            # weight is a real loss (unlike on bfloat8_b, where the tile's 8-bit mantissa is already
            # the bound), so leaving the reduced fidelity in place would spend the 385 MB and keep
            # most of the error it was spent on.
            mm_math_fidelity=None,
            mm_fp32_dest_acc_en=None,
        )

    def attention_kwargs(self) -> dict:
        """Construction kwargs for ``MiniMaxH3Attention``.

        ``mm_compute_kernel_overrides`` is here because the attention builds its OWN matmul
        compute-kernel config for to_qkv and to_out rather than taking the block's, so a policy
        wired only into the block reaches ff1/ff2 and silently leaves half the block's matmul time
        on the old arithmetic.
        """
        return {
            "qkv_dtype": self.qkv_dtype,
            "out_dtype": self.out_dtype,
            "activation_dtype": self.activation_dtype,
            "pin_output_bf16": self.pin_output_bf16,
            "mm_compute_kernel_overrides": self.compute_kernel_kwargs(),
        }

    def ffn_kwargs(self) -> dict:
        """Construction kwargs for ``ParallelFeedForward``."""
        return {
            "ff1_dtype": self.ff_dtype,
            "ff2_dtype": self.ff_dtype,
            "activation_dtype": self.activation_dtype,
            "pin_output_bf16": self.pin_output_bf16,
        }

    def compute_kernel_kwargs(self) -> dict:
        """``init_device_compute_kernel_config`` overrides for the block's four matmuls.

        Only the keys this profile actually sets, so the block keeps its own defaults for the rest
        and an unset profile is indistinguishable from no profile at all.
        """
        kwargs = {}
        if self.mm_math_fidelity is not None:
            kwargs["math_fidelity"] = self.mm_math_fidelity
        if self.mm_fp32_dest_acc_en is not None:
            kwargs["fp32_dest_acc_en"] = self.mm_fp32_dest_acc_en
        return kwargs

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
    def bf8_weights_bf8_out_nofp32acc() -> MiniMaxH3QuantProfile:
        """``bf8_weights_bf8_out`` with the block matmuls' fp32 destination accumulate turned off.

        Same 20.5 GB and the SAME cached tensorbins -- this changes the arithmetic, not the storage,
        so ``cache_tag`` is deliberately unchanged and a device-weight cache written under
        ``bf8_weights_bf8_out`` is reused as-is.

        fp32 destination accumulation halves the usable destination-register tiles, which costs
        blocking freedom on the four large block matmuls, and it buys precision the INPUTS cannot
        carry: a bfloat8_b tile is an 8-bit mantissa against one shared exponent per 16 values, so
        the accumulator is not what bounds the error.

        The policy reaches all four block matmuls -- to_qkv and to_out through
        ``attention_kwargs``, ff1 and ff2 through the block -- and that is deliberate. The attention
        builds its OWN compute-kernel config, so a policy wired only into the block reaches ff1/ff2
        and silently leaves the two attention projections (26.7 ms of a 221.7 ms block) on the old
        arithmetic. Measured on one real block at the served p150 shape (1344x768 x73, 22464 padded
        rows), three warm samples, medians rather than minima:

            HiFi2 + fp32_dest_acc   261.64 / 261.92 / 261.64 ms   (the default)
            HiFi2                   252.68 / 255.05 / 257.26 ms   (-2.5%, this profile)
            LoFi                    226.43 / 228.81 / 230.17 ms   (-12.5%)

        LoFi is the bigger win and is deliberately NOT offered: at full depth on the real checkpoint
        it scores video PCC 0.9892 against the 0.99 bar every quantized row here holds (audio
        0.9721, which passes its 0.95). It misses by 0.0008 and the bar was not moved.

        ``bf16_blocks=(0, -1)`` -- the escape hatch this class documents, 770 MB of DRAM and a
        re-quantized cache to take the first and last block out of the reduced regime entirely --
        was measured on top of LoFi and made it WORSE, not better: video PCC 0.9858 against plain
        LoFi's 0.9892. So the error LoFi adds is not concentrated in the ends, and pinning two of
        fifty blocks is not the lever for it. Both rows are in stage 06's results.json; neither
        preset is shipped, so neither has a PCC row of its own here.
        """
        return replace(
            MiniMaxH3QuantProfile.bf8_weights_bf8_out(),
            name="bf8_weights_bf8_out_nofp32acc",
            mm_fp32_dest_acc_en=False,
        )

    @staticmethod
    def bf4_ff(bf16_blocks: tuple[int, ...] = ()) -> MiniMaxH3QuantProfile:
        """``bf8_weights_bf8_out`` with the two feed-forward linears dropped to bf4: 12.0 GB.

        The feed-forward pair is where the bytes are -- ff1 is 5376x28672 packed [gate|up] and ff2
        14336x5376, so together they are 71 % of a block's 385.3 M parameters and dropping them from
        bf8 to bf4 takes 8.5 GB off the stack. Attention stays bf8: to_qkv and to_out are the
        remaining 29 % (2.5 GB to be had) and they feed the SDPA and the fused addcmul epilogue,
        where stage 02 already had to add a pad-row window to get the attention numerics right.

        This is the datatype sweep's lever, not a default: a bf4 tile carries a 4-bit mantissa
        against a shared exponent per 16 values, so it has to earn its place on measured component
        PCC and end-to-end CLIP before anything serves it.
        """
        return MiniMaxH3QuantProfile(
            name="bf4_ff",
            qkv_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat8_b,
            ff_dtype=ttnn.bfloat4_b,
            bf16_blocks=bf16_blocks,
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
    "bf8_weights_bf8_out_nofp32acc": MiniMaxH3QuantProfile.bf8_weights_bf8_out_nofp32acc,
    "bf4_ff": MiniMaxH3QuantProfile.bf4_ff,
    # bf4 feed-forward with the first and last block pinned back to bf16 -- the sweep's accuracy
    # escape hatch, 385 MB a block. Named here so it is reachable from a command line.
    #
    # The `replace` is not cosmetic: `bf4_ff()` names itself "bf4_ff" whatever `bf16_blocks` it is
    # given, so without it this preset resolves to a profile that calls itself by the other preset's
    # name, and `name` is what every run's RESULT json records as the policy it ran under.
    "bf4_ff_keep_ends": lambda: replace(MiniMaxH3QuantProfile.bf4_ff(bf16_blocks=(0, -1)), name="bf4_ff_keep_ends"),
    "bf16": MiniMaxH3QuantProfile.bf16,
}


def resolve_quant_profile(spec) -> MiniMaxH3QuantProfile | None:
    """``None`` / a preset name / a profile -> a profile or ``None`` (meaning "build bf16")."""
    if spec is None or isinstance(spec, MiniMaxH3QuantProfile):
        return spec
    if spec in PRESETS:
        return PRESETS[spec]()
    raise ValueError(f"unknown MiniMax-H3 quant profile {spec!r}; known: {sorted(PRESETS)}")
