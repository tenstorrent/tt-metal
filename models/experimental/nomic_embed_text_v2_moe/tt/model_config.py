# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-side settings every TTNN op in this port runs under.

Each op group has its own compute kernel config. Measured on a p300c, against exact float64 math
on real weights and real-text activations:

  destination accum    fp32 everywhere. A bf16 accumulator costs 6x to 15x max-abs on every
                       matmul (fc2, a 3072-deep reduction, degrades from 1.35e-1 to 2.08e+00;
                       the router reroutes 28 tokens in 1024 rather than 1). Re-blocked for the
                       8-tile destination it frees, it makes the dense minimal_matmuls 2% to 9%
                       faster at 8x512, 0.3% of device time, and the transposed w1 no faster,
                       while the steady 8x512 last-hidden PCC falls from 0.9965 to 0.9940.
  packer L1 accum      on for every matmul. Without it each K block reloads the fp32 partials
                       into the destination register; on its own it halves the dense matmuls.
                       It changes nothing numerically, the partials being fp32 either way.
  math fidelity        HiFi2 for fc1 and expert w1, HiFi3 for QKV, out_proj and fc2, LoFi for
                       expert w2, HiFi4 for the router. HiFi3 on bf16 x bf16 skips only the
                       low-bits-by-low-bits partial product and matches HiFi4's error to three
                       digits. fc1 at HiFi2 is 13% faster at 8x512. The transposed w1, bfloat8_b
                       weights in in0, is bit-identical at HiFi2 and HiFi3, and its small-pass
                       sparse_matmul is 16% faster at 128 tokens. w2 at LoFi is 25% faster on
                       a 4096-token transposed pass and 5% to 7% on the token-major passes of
                       1x128 and 2x37. The router, bound by reading its input, runs the same at
                       every fidelity.
  weights              bf16, but bfloat8_b for both expert weights. w1's sparse_matmul is bound by
                       streaming them, 25% faster at 128 tokens; w2 gains 5% at 128 tokens and
                       11% at 74.
  expert intermediate  bfloat8_b for the (1, E, T, F) tensor that w1 writes, the GELU rewrites and
                       w2 reads. w1 is 17% and w2 11% faster at 3520 tokens, 3.6% of device time
                       at 8x512. The GELU runs the same in either dtype.
  expert output        bfloat8_b on a transposed pass for the w2 output, the gate that weights it
                       and their product (see tt/experts.py). w2 is 11% and the reduce 43% faster
                       at 4096 tokens, 3.1% of device time at 8x512.
  softmax              HiFi4 with fp32 accumulation, both halves required: max-abs is 2.7e-2 to
                       3.0e-2 stock, 5.4e-3 to 6.8e-3 with HiFi4 alone, 1.4e-3 to 1.9e-3 with
                       both, against the router's 5e-3 budget.
  SDPA                 HiFi3 with fp32 accumulation. With the chunks and placement of
                       tt/attention.py a call takes 115 us at 8x512 against 1098 us before. HiFi3
                       matches HiFi4's error to every digit measured and is 9% faster. A bfloat16
                       destination selects SDPA's streaming kernel, 30% faster a call, 0.4 ms a
                       forward at 8x512; with the tanh GELU it moved 5 of 32 encoder draws past the
                       pooled gate of test_ttnn_encoder.py against 1. Most of the error is the
                       approximate exp (8x the bfloat16 floor); exp_approx_mode=False takes it to
                       1.5x for 3x the time.
  attention mask       none for a batch without padding: a dense mask doubled SDPA's time at
                       8x512, read once per query chunk of every head. A padded batch's is
                       bfloat4_b, which holds 0 and dtype-min exactly: the same output as bfloat16
                       to the bit, 131 against 203 us on a padded 8x512 batch on the streaming
                       kernel. On the fp32 one a bfloat16 mask does not fit beside a 256 x 512
                       score block.
  GELU                 tanh, for the dense FFN and the experts. It is within 4.7e-4 of exact erf,
                       a thirtieth of bfloat16's rounding, and 18% faster: 1353 against 1645 us
                       for the experts at 8x512, 171 against 208 for the dense FFN. On fc1 above
                       32 tile rows of M and on the transposed expert w1 it is fused into the
                       matmul, whose 2D multicast program applies it from the packer beside the
                       math (tt/matmul_config.py, gelu_on_packer): w1 and its GELU take 1630 us at
                       8x512 against 786 + 1353, fc1 and its GELU 260 against 132 + 171. The LUT,
                       fused into minimal_matmul, costs 75 us over the w1 where the tanh form now
                       costs 844, 4.6 ms a forward at 8x512, but it triples the random-id tail (108
                       of 512 draws over 5e-3 against 37) and fails five tests, two of
                       test_ttnn_model.py at 2x37 and three of test_ttnn_encoder.py at 2x128; it is
                       a TtModelConfig switch away (expert_gelu).

None of the reduced-precision choices above moved retrieval nDCG beyond run-to-run noise. These
were measured and left off:
- HiFi2 on QKV, out_proj and fc2. On QKV it shifts the pooled embedding of short padded batches
  (2x128 with a quarter padded: 6 of 8 seeds over 1e-3 against 2). On out_proj and fc2 it widens
  the random-id tail (14 of 192 draws over 1e-2 against 1).
- LoFi on expert w1: slower at 4096 tokens (827 to 907 us), and about 0.003 off the steady
  last-hidden PCC.
- bfloat8_b dense weights: 1.6% at 1x128, for the largest error rise of any option.
- bfloat4_b weights for the small-pass w1: 1.5% at 2x37, for last-hidden PCC 0.964 at 1x128
  against 0.992.

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
# to bfloat16 before the topk reroutes 0.34% to 0.59% of tokens. ttnn.topk takes fp32, so only the
# two selected weights are cast, after the selection, for the bfloat16 gate the experts consume.
ROUTER_DTYPE = ttnn.float32

# The (1, E, T, F) expert intermediate: the w1 output, the GELU and the w2 input. See the module
# docstring.
EXPERT_INTERMEDIATE_DTYPE = ttnn.bfloat8_b

# The w2 output of a transposed expert pass, the gate that weights it and their product. See the
# module docstring.
EXPERT_OUTPUT_DTYPE = ttnn.bfloat8_b

# The additive attention mask of a padded batch. Its values are 0 and dtype-min, which bfloat4_b
# holds exactly, and SDPA reads one mask chunk per query chunk of every head. See the module
# docstring.
ATTENTION_MASK_DTYPE = ttnn.bfloat4_b

# The GELU of the dense FFN and of the experts. See the module docstring.
DENSE_GELU = ttnn.GeluVariant.Tanh
EXPERT_GELU = ttnn.GeluVariant.Tanh

LAYOUT = ttnn.TILE_LAYOUT
MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG


class OpGroup(Enum):
    """The sets of ops that share one compute kernel config.

    Different kernel families never share a config: the same field changes different arithmetic
    in each, and they carry separate measured budgets, the router softmax needing both halves of
    HiFi4 with fp32 accumulation. Within the matmuls every role has its own program config, and
    packer_l1_acc only pays off against a given K blocking, so every role has its own group.
    """

    QKV = "qkv"
    ATTN_OUT = "attn_out"
    FC1 = "fc1"
    FC2 = "fc2"
    ROUTER = "router"
    EXPERT_W1 = "expert_w1"
    EXPERT_W2 = "expert_w2"
    SDPA = "sdpa"
    SOFTMAX = "softmax"  # the router's
    NORM = "norm"  # emb_ln and both block norms
    REDUCE = "reduce"  # the expert fast_reduce_nc and the pooling sums


MATMUL_GROUPS = (
    OpGroup.QKV,
    OpGroup.ATTN_OUT,
    OpGroup.FC1,
    OpGroup.FC2,
    OpGroup.ROUTER,
    OpGroup.EXPERT_W1,
    OpGroup.EXPERT_W2,
)


_MATMUL_FIDELITY = {
    OpGroup.FC1: ttnn.MathFidelity.HiFi2,
    OpGroup.ROUTER: ttnn.MathFidelity.HiFi4,
    OpGroup.EXPERT_W1: ttnn.MathFidelity.HiFi2,
    OpGroup.EXPERT_W2: ttnn.MathFidelity.LoFi,
}

# bfloat8_b halves the expert weight streams that bound the small passes. Every other matmul
# weight stays at its full width: see the module docstring.
_MATMUL_WEIGHT_DTYPE = {
    OpGroup.EXPERT_W1: ttnn.bfloat8_b,
    OpGroup.EXPERT_W2: ttnn.bfloat8_b,
}


# fp32 accumulation selects SDPA's older kernel; a bfloat16 destination its streaming one. See the
# module docstring.
_SDPA_FIDELITY = ttnn.MathFidelity.HiFi3
_SDPA_FP32_DEST_ACC = True
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
    # matmul_weight_dtypes holds the overrides of weight_dtype and router_dtype. Both mappings stay
    # out of the hash, which a mapping proxy does not support; equality still compares them.
    compute_kernel_configs: Mapping[OpGroup, ttnn.DeviceComputeKernelConfig] = field(hash=False)
    matmul_weight_dtypes: Mapping[OpGroup, ttnn.DataType] = field(hash=False)

    activation_dtype: ttnn.DataType = ACTIVATION_DTYPE
    weight_dtype: ttnn.DataType = WEIGHT_DTYPE
    router_dtype: ttnn.DataType = ROUTER_DTYPE
    expert_intermediate_dtype: ttnn.DataType = EXPERT_INTERMEDIATE_DTYPE
    expert_output_dtype: ttnn.DataType = EXPERT_OUTPUT_DTYPE
    attention_mask_dtype: ttnn.DataType = ATTENTION_MASK_DTYPE
    dense_gelu: ttnn.GeluVariant = DENSE_GELU
    expert_gelu: ttnn.GeluVariant = EXPERT_GELU
    attention_l1: bool = True  # the head tensors and SDPA's output in L1 where they fit, tt/attention.py
    layout: ttnn.Layout = LAYOUT

    def compute_kernel_config(self, group: OpGroup) -> ttnn.DeviceComputeKernelConfig:
        return self.compute_kernel_configs[group]

    def matmul_weight_dtype(self, group: OpGroup) -> ttnn.DataType:
        default = self.router_dtype if group is OpGroup.ROUTER else self.weight_dtype
        return self.matmul_weight_dtypes.get(group, default)

    @classmethod
    def from_device(cls, device) -> "TtModelConfig":
        """Build the config from an open ttnn device or single-device mesh.

        The grid is queried, never hardcoded: this p300c reports 11x10, not the (8, 10) that
        models/tt_transformers/tt/model_config.py implies.

        Each group gets its own config object: ttnn's are mutable, so a shared one would let a
        change made for one group reach all of them.
        """
        device_info = ttnn._ttnn.reports.get_device_info(device)
        return cls(
            core_grid=device.compute_with_storage_grid_size(),
            l1_cb_bytes=device_info.cb_limit,
            l1_banks=device_info.l1_num_banks,
            compute_kernel_configs=MappingProxyType(
                {group: _compute_config(device.arch(), group) for group in OpGroup}
            ),
            matmul_weight_dtypes=MappingProxyType(dict(_MATMUL_WEIGHT_DTYPE)),
        )
