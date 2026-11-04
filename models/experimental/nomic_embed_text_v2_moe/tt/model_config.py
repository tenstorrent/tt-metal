# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-side settings every TTNN op in this port runs under.

Each op group has its own compute kernel config. Measured on a p300c, against exact float64 math
on real weights and real-text activations:

  destination accum    fp32 everywhere. A bf16 accumulator costs 6x to 15x max-abs on every
                       matmul (fc2, a 3072-deep reduction, degrades from 1.35e-1 to 2.08e+00;
                       the router reroutes 28 tokens in 1024 rather than 1). Re-blocked for the
                       8-tile destination it frees, it makes the dense minimal_matmuls 2% to 9%
                       faster at 8x512, 0.3% of device time, while the steady 8x512 last-hidden
                       PCC falls from 0.9965 to 0.9940.
  packer L1 accum      on for every matmul. Without it each K block reloads the fp32 partials
                       into the destination register; on its own it halves the dense matmuls.
                       It changes nothing numerically, the partials being fp32 either way.
  math fidelity        HiFi2 for QKV, fc1, fc2 and the expert w1, LoFi for the expert w2, HiFi3 for
                       out_proj, HiFi4 for the router. HiFi3 on bf16 x bf16 skips only the
                       low-bits-by-low-bits partial product and matches HiFi4's error to three
                       digits. fc1 at HiFi2 is 13% faster at 8x512. The router, bound by reading
                       its input, runs the same at every fidelity. HiFi2 drops the low mantissa
                       bits of the activations (in0): on QKV and fc2 it saves 1.7% to 4% of a
                       forward at 1k to 8k tokens, for a pooled direction error 14% higher and mean
                       last-hidden PCC 0.99556 against 0.99581 over 39 inputs of 1k to 16k tokens
                       (2026-10-27). out_proj at HiFi2 saved at most 0.6% and stays at HiFi3.
                       The routed experts (grouped_matmul) take the activations as their first
                       operand, whose low mantissa bits the low fidelities drop. With w1 at HiFi2
                       that left an error shared by the tokens (pooled error 1.26x that of the
                       dense pass the routed one replaced), and w1 ran at HiFi3 while its GELU
                       bound it anyway. With the polynomial GELU (routed GELU below) the FPU
                       bounds it, and HiFi2 saves 1.4% to 3% of a forward at 1k to 8k tokens for
                       a pooled direction error 2% higher on average (p99 2.2e-3 -> 3.0e-3) and
                       mean PCC 0.99577 against 0.99579 over 39 inputs (2026-10-27). w2 at LoFi makes the routed
                       outputs about 1.1% smaller than at HiFi2, which covers the bfloat8_b
                       intermediate's mantissa, and saves 0.36 ms of an 8x512 forward. With it no
                       end-to-end draw over seeds 0..7 falls below PCC 0.98 and mean errors grow
                       about 1.4x; the 512-draw envelopes (default shapes; 4x512 and 8x384) count
                       29 and 4 draws over 5e-3 against HiFi2's 32 and 10, routing reshuffled
                       rather than a gain.
  weights              bf16, but bfloat8_b for both expert weights, which bound the small passes
                       by streaming them: the old token-major w1 was 25% faster at 128 tokens, w2
                       5% at 128 tokens and 11% at 74.
  expert intermediate  bfloat8_b for the (1, 1, R, F) tensor that w1 writes and w2 reads. On the
                       transposed dense pass the routed one replaced, w1 was 17% and w2 11% faster
                       with it at 3520 tokens, 3.6% of device time at 8x512.
  router softmax       HiFi4 with fp32 accumulation, both halves required: the router's scores
                       (router_front_end, its matmul and softmax in one op) measure 1.6e-3 to
                       2.4e-3 max-abs, 9.6e-3 to 1.1e-2 at HiFi2, against a 5e-3 budget; on its
                       own ttnn.softmax measured 5.4e-3 to 6.8e-3 with HiFi4 alone, and
                       router_front_end refuses a bfloat16 destination.
  SDPA                 HiFi2 on the bfloat16-destination (streaming) kernel (2026-10-29): 14% to 33%
                       faster a call than the fp32 kernel at HiFi2 and 0.7% to 5% of a forward at
                       2x512 to 16x512, with retrieval within noise (2026-10-28). It hung the device
                       until SDPA's reader waited for its last write acks (#63304): the next op's
                       writer on core (0,3) waited forever in its write barrier, 4 hangs in about
                       18,000 2x512 forwards; with the barrier, none in 20,000. HiFi2 drops the low
                       mantissa bits of the softmax probabilities, the in0 of SDPA's second matmul:
                       on the fp32 kernel 9% to 12.6% faster a call from S = 128, for 11% more
                       per-text error over 22,562 retrieval texts with nDCG within noise, and about
                       50% more on random ids (one 2x512 input at PCC 0.977). HiFi3 matches HiFi4's
                       error to every digit measured. LoFi makes every attention output about 4%
                       small: all 560 random-id rows of 39 inputs of 1k to 16k tokens fall under PCC
                       0.99. Most of the error is the approximate exp (8x the bfloat16 floor), which
                       the streaming kernel uses regardless of exp_approx_mode.
  attention mask       none for a batch without padding: a dense mask doubled SDPA's time at
                       8x512, read once per query chunk of every head. A padded batch's is
                       bfloat4_b, which holds 0 and dtype-min exactly: the same output as bfloat16
                       to the bit, 131 against 203 us on a padded 8x512 batch on the streaming
                       kernel. On the fp32 one a bfloat16 mask does not fit beside a 256 x 512
                       score block.
  GELU                 tanh, for the dense FFN and the experts. It is within 4.7e-4 of exact erf,
                       a thirtieth of bfloat16's rounding, and 18% faster: 1353 against 1645 us
                       for the dense experts at 8x512, 171 against 208 for the dense FFN. On fc1
                       above 32 tile rows of M it is fused into the matmul, whose 2D multicast
                       program applies it from the packer beside the math (tt/matmul_config.py,
                       gelu_on_packer): fc1 and its GELU take 260 us at 8x512 against 132 + 171,
                       the transposed dense expert w1 took 1630 against 786 + 1353.
                       grouped_matmul applies it to the routed w1 (routed GELU below). The LUT,
                       fused into the transposed w1's minimal_matmul, cost 75
                       us over the w1 where the tanh form cost 844, 4.6 ms a forward at 8x512, but
                       it tripled the random-id tail (108 of 512 draws over 5e-3 against 37) and
                       failed five tests, two of test_ttnn_model.py at 2x37 and three of
                       test_ttnn_encoder.py at 2x128. Re-gated on 1k to 16k token inputs
                       (2026-10-27): 4% to 5.5% faster, but mean last-hidden PCC 0.9958 -> 0.9903
                       and the pooled direction error 3.2x on random ids, 2x on real text.
  dense GELU form      the tanh GELU's fast form, x / (1 + exp(-2u)) (dense_fast_gelu), as the
                       routed w1 computes it, fused into fc1 at every M (from the packer). 1.6% to
                       2.5% of a forward at 1k to 8k tokens, the gain at 1024 tokens from fusing
                       it into the small-M program; on 39 inputs of 1k to 16k tokens mean PCC
                       0.99581 against 0.99580 and the same error tail (2026-10-27).
  routed GELU          the routed experts' w1 (grouped_matmul) computes GELU as x * Phi(t),
                       t = clamp(x, -3.5, 3.5), Phi a degree-4 polynomial in t^2 (grouped_matmul's
                       GELU_TANH parameter 2, routed_gelu_form 2): |error| <= 1.3e-3 against exact
                       erf, under the rounding of its bfloat8_b output, in about 9 SFPU instructions
                       a row against about 22 for the tanh form's fast evaluation (form 1, within 1
                       bf16 ULP of the accurate tanh form, which took w1 from 370 to 272 us at 4096
                       tokens). The polynomial saved 1.5% to 1.7% of a forward at 1k to 8k tokens,
                       with mean PCC 0.99570 against 0.99556 and a pooled direction error 7% lower
                       over 39 inputs of 1k to 16k tokens (2026-10-27).
  routed experts       every MoE layer runs each token through its two experts only
                       (routing_metadata, grouped_matmul and routed_combine, tt/experts.py)
                       instead of through all eight, gated. An 8x512 forward takes 12.8 ms against
                       25.1 with the transposed dense pass it replaced. On 22,562 retrieval texts
                       the median 1 - cos per split is 0.65x to 1.07x that pass's, with nDCG
                       within run-to-run noise. It also replaced a stacked dense pass up to 320
                       tokens, which took 22 us a MoE layer less device time at 32 tokens and 18 us
                       more at 128 (tt/experts.py), and more host time at every size.

None of the reduced-precision choices above moved retrieval nDCG beyond run-to-run noise. These
were measured and left off:
- HiFi2 on out_proj: at most 0.6% of a forward at 1k to 16k tokens. Before inputs below 1024
  tokens left the scope, HiFi2 on QKV, out_proj and fc2 was left off for short batches: on QKV
  it shifted the pooled embedding of 2x128 with a quarter padded (6 of 8 seeds over 1e-3
  against 2), on out_proj and fc2 it widened the random-id tail (14 of 192 draws over 1e-2
  against 1).
- bfloat16 accumulation in the dense matmuls and bfloat8_b dense weights at 1k to 16k tokens:
  within 0.4% either way on device, so no reason to take their error.
- bfloat8_b dense weights: 1.6% at 1x128, for the largest error rise of any option. On the routed
  code, 6.8% at 1x32 and 2.3% at 1x128, but one random-id draw at last-hidden PCC 0.906 (3x40 seed
  1, against 0.992) and twice the random-id tail (58 of 512 draws over 5e-3 against 29).

dst_full_sync_en stays off. Where it is correct it is slower, since math and pack stop
overlapping, and in two cases it returns wrong results: a 1x8 subblock with fp32 accumulation
(PCC 0.47 on the expert w1 matmul) and a 16-tile subblock with bf16 accumulation (non-finite).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import ttnn

ACTIVATION_DTYPE = ttnn.bfloat16
WEIGHT_DTYPE = ttnn.bfloat16

# The one path kept at higher precision: the router's softmax feeds a top-2 selection, so a
# near-tie decided by rounding changes which experts a token visits. Rounding the probabilities
# to bfloat16 before the selection reroutes 0.34% to 0.59% of tokens. routing_metadata selects from
# the fp32 scores and copies the two selected weights as they are, so nothing on the path is cast.
ROUTER_DTYPE = ttnn.float32

# Both expert weights, w1 and w2, which every expert pass streams. See the module docstring.
EXPERT_WEIGHT_DTYPE = ttnn.bfloat8_b

# The expert intermediate, the (1, 1, R, F) w1 output with its GELU. See the module docstring.
EXPERT_INTERMEDIATE_DTYPE = ttnn.bfloat8_b

# The dense FFN's intermediate, fc1's output with its GELU, which fc2 reads. See the module docstring.
DENSE_INTERMEDIATE_DTYPE = ttnn.bfloat8_b

# The additive attention mask of a padded batch. Its values are 0 and dtype-min, which bfloat4_b
# holds exactly, and SDPA reads one mask chunk per query chunk of every head. See the module
# docstring.
ATTENTION_MASK_DTYPE = ttnn.bfloat4_b

# The GELU of the dense FFN and of the experts. See the module docstring.
DENSE_GELU = ttnn.GeluVariant.Tanh
DENSE_FAST_GELU = True
# grouped_matmul's GELU_TANH parameter for the routed w1: 0 the accurate tanh form, 1 its fast
# evaluation, 2 the op's polynomial. See the module docstring.
ROUTED_GELU_FORM = 2

# Above this many tile rows of M the dense fc1 runs as one expert of grouped_matmul, with its GELU in
# DENSE_GELU_FORM (the same parameter as ROUTED_GELU_FORM); at or below, as a ttnn.linear with the
# tanh GELU fused (dense_gelu, dense_fast_gelu).
DENSE_FC1_GROUPED_MIN_TILES = 0
DENSE_GELU_FORM = 2

LAYOUT = ttnn.TILE_LAYOUT
MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG

# The most computed tokens (B x ceil32(S)) one backbone pass takes; a larger input runs as chunks of whole
# sequences (tt/chunking.py). Past it the activations leave L1 and a token costs more: on the code of
# 2026-10-26 a token cost 2.73 to 2.88 us at 8192 tokens a pass and 2.85 to 2.99 at 16384.
MAX_CHUNK_TOKENS = 8192


class OpGroup(Enum):
    """The sets of ops that share one compute kernel config.

    Different kernel families never share a config: the same field changes different arithmetic
    in each, and they carry separate measured budgets, the router needing both halves of HiFi4 with
    fp32 accumulation. Within the matmuls every role has its own program config, and
    packer_l1_acc only pays off against a given K blocking, so every role has its own group.
    """

    QKV = "qkv"
    ATTN_OUT = "attn_out"
    FC1 = "fc1"
    FC2 = "fc2"
    ROUTER = "router"  # router_front_end, the router's matmul and softmax
    ROUTED_W1 = "routed_w1"  # the experts, tt/experts.py
    ROUTED_W2 = "routed_w2"
    SDPA = "sdpa"
    NORM = "norm"  # emb_ln and both block norms
    REDUCE = "reduce"  # routed_combine and the pooling sums


MATMUL_GROUPS = (
    OpGroup.QKV,
    OpGroup.ATTN_OUT,
    OpGroup.FC1,
    OpGroup.FC2,
    OpGroup.ROUTER,
    OpGroup.ROUTED_W1,
    OpGroup.ROUTED_W2,
)


_MATMUL_FIDELITY = {
    OpGroup.QKV: ttnn.MathFidelity.HiFi2,
    OpGroup.FC1: ttnn.MathFidelity.HiFi2,
    OpGroup.FC2: ttnn.MathFidelity.HiFi2,
    OpGroup.ROUTER: ttnn.MathFidelity.HiFi4,
    OpGroup.ROUTED_W1: ttnn.MathFidelity.HiFi2,
    OpGroup.ROUTED_W2: ttnn.MathFidelity.LoFi,
}

# fp32 accumulation selects SDPA's older kernel; a bfloat16 destination its streaming one. See the
# module docstring.
_SDPA_FIDELITY = ttnn.MathFidelity.HiFi2
_SDPA_FP32_DEST_ACC = False
_SDPA_MATH_APPROX = False


def _compute_config(arch, group: OpGroup) -> ttnn.DeviceComputeKernelConfig:
    """math_approx_mode is off everywhere because it selects cheaper SFPU polynomials."""
    if group is OpGroup.SDPA:
        return ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=_SDPA_FIDELITY,
            math_approx_mode=_SDPA_MATH_APPROX,
            fp32_dest_acc_en=_SDPA_FP32_DEST_ACC,
            packer_l1_acc=False,
        )
    is_matmul = group in MATMUL_GROUPS
    fidelity = _MATMUL_FIDELITY.get(group, ttnn.MathFidelity.HiFi3) if is_matmul else ttnn.MathFidelity.HiFi4
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=is_matmul,
    )


@dataclass(frozen=True)
class TtModelConfig:
    """The settings that depend on the device, plus the dtypes and layout they go with.

    Model dimensions are not here; they come from the vendored config.json via
    reference.configuration_nomic_moe, the one source both sides read.
    """

    core_grid: ttnn.CoreCoord
    l1_cb_bytes: int  # per-core L1 left for circular buffers, which bounds a matmul's blocks
    l1_banks: int  # the banks an L1-interleaved tensor spreads over
    l1_top: int  # the end of that L1: circular buffers grow up from its start, tensors down from here
    # Out of the hash, which a mapping proxy does not support; equality still compares it.
    compute_kernel_configs: Mapping[OpGroup, ttnn.DeviceComputeKernelConfig] = field(hash=False)

    activation_dtype: ttnn.DataType = ACTIVATION_DTYPE
    weight_dtype: ttnn.DataType = WEIGHT_DTYPE
    router_dtype: ttnn.DataType = ROUTER_DTYPE
    expert_weight_dtype: ttnn.DataType = EXPERT_WEIGHT_DTYPE
    expert_intermediate_dtype: ttnn.DataType = EXPERT_INTERMEDIATE_DTYPE
    dense_intermediate_dtype: ttnn.DataType = DENSE_INTERMEDIATE_DTYPE
    attention_mask_dtype: ttnn.DataType = ATTENTION_MASK_DTYPE
    dense_gelu: ttnn.GeluVariant = DENSE_GELU
    dense_fast_gelu: bool = DENSE_FAST_GELU
    routed_gelu_form: int = ROUTED_GELU_FORM
    dense_fc1_grouped_min_tiles: int = DENSE_FC1_GROUPED_MIN_TILES
    dense_gelu_form: int = DENSE_GELU_FORM
    attention_l1: bool = True  # the head tensors and SDPA's output in L1 where they fit, tt/attention.py
    # QKV, the head split, q * 1/8 and rotary as one generated op (ttnn.operations.nomic_qkv_rotary_heads) wherever it
    # takes x as it comes: 32 to 128 tile rows of M in the split norms' layout (tt/attention.py). Off: the four-op
    # chain (QKV, nlp_create_qkv_heads, two rotary_embedding_llama).
    fused_attention_front: bool = True
    max_chunk_tokens: int = MAX_CHUNK_TOKENS  # the largest chunk a forward runs at once, tt/chunking.py
    layout: ttnn.Layout = LAYOUT

    def compute_kernel_config(self, group: OpGroup) -> ttnn.DeviceComputeKernelConfig:
        return self.compute_kernel_configs[group]

    def matmul_weight_dtype(self, group: OpGroup) -> ttnn.DataType:
        return self.router_dtype if group is OpGroup.ROUTER else self.weight_dtype

    @classmethod
    def from_device(cls, device) -> "TtModelConfig":
        """Build the config from an open ttnn device or single-device mesh.

        The grid is queried, never hardcoded: this p300c reports 11x10, not the (8, 10) that
        models/ttt_compat/tt/model_config.py implies.

        Each group gets its own config object: ttnn's are mutable, so a shared one would let a
        change made for one group reach all of them.
        """
        device_info = ttnn._ttnn.reports.get_device_info(device)
        return cls(
            core_grid=device.compute_with_storage_grid_size(),
            l1_cb_bytes=device_info.cb_limit,
            l1_banks=device_info.l1_num_banks,
            l1_top=device_info.address_at_first_l1_cb_buffer + device_info.cb_limit,
            compute_kernel_configs=MappingProxyType(
                {group: _compute_config(device.arch(), group) for group in OpGroup}
            ),
        )
