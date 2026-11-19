# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experimental BF16 baseline partition selection with direct FP32 DRAM reads.

This deliberately rounds logits BEFORE selection. It is distinct from the
exact-FP32 candidate, whose module is untouched. The original InlineSelector
partition functions are copied verbatim, preserving their tie permutation.
Selected values are widened BF16 bits in FP32 tiles; the ordinary mesh softmax
and BF16 score/index tail remain intact. Full-decoder accuracy gates still apply.

Finite conversion uses integer round-to-nearest-even, including signed zeros
and subnormals. NaNs canonicalize to 0x7fc0, matching the repository host BF16
conversion; parity with device typecast for nonfinite inputs remains a device
probe requirement. This module does not claim that CPU testing proves it.
"""

import types

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_topk_probe import SELECTOR_CPP, MeshFP32TopK

ROUND_BF16_CPP = r"""
inline uint32_t probe_bf16_rne(uint32_t bits) {
    if ((bits & 0x7fffffffU) > 0x7f800000U) return 0x7fc0U;
    return ((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16) & 0xffffU;
}
"""

PARTITION_BODY_CPP = r"""
    // The baseline packs one rounded BF16 key and its original expert ID.
    // Work in the same 256-word L1 array and preserve its original order.
    auto* queue = reinterpret_cast<uint32_t*>(get_write_ptr(0) + bias);
    for (uint32_t i = 0; i < 256; ++i) {
        queue[i] = (probe_bf16_rne(data[i]) << 16) | i;
    }
    v6_select_top8(queue);
    for (uint32_t i = 0; i < 8; ++i) {
        data[512+i] = queue[i] & 0xffff0000U;
        data[528+i] = queue[i] & 0xffffU;
    }
"""


def make_kernel_source(selector_cpp):
    # Accept either the full InlineSelector kernel or its portable prefix.
    prefix = selector_cpp.split("void kernel_main", 1)[0]
    if "v6_select_top8" not in prefix or "v6_greater" not in prefix:
        raise ValueError("BF16 partition probe requires the original InlineSelector selection functions")
    marker = "// IEEE bitwise numeric order"
    main_marker = "void kernel_main()"
    if SELECTOR_CPP.count(marker) != 1 or SELECTOR_CPP.count(main_marker) != 1:
        raise ValueError("FP32 DRAM reader source changed; adapt the probe explicitly")
    includes, remainder = SELECTOR_CPP.split(marker, 1)
    body = main_marker + remainder.split(main_marker, 1)[1]
    first, last = "    uint32_t best[8], ids[8];", "    scratch.push_back(1);"
    if body.count(first) != 1 or body.count(last) != 1:
        raise ValueError("FP32 selector body changed; adapt the probe explicitly")
    body = body[: body.index(first)] + PARTITION_BODY_CPP + body[body.index(last) :]
    return includes + ROUND_BF16_CPP + prefix + body


class MeshBF16PartitionTopK(MeshFP32TopK):
    """Identical setup-owned buffers and trace lifetime to MeshFP32TopK.

    The inherited call reads [1,1,1,256] FP32 DRAM logits and returns FP32
    values/UINT32 IDs, both logical [1,1,1,8]. Outputs are overwritten on each
    call, consumed serially, and kept alive until all using traces are released.
    """

    def __init__(self, mesh, selector_cpp):
        self.kernel_source = make_kernel_source(selector_cpp)
        source = MeshFP32TopK.__init__
        constructor = types.FunctionType(
            source.__code__,
            dict(source.__globals__, SELECTOR_CPP=self.kernel_source),
            source.__name__,
            source.__defaults__,
            source.__closure__,
        )
        constructor.__kwdefaults__ = source.__kwdefaults__
        constructor(self, mesh)


class _PartitionProxy:
    def __init__(self, selector):
        self.selector = selector

    def __getattr__(self, name):
        return getattr(ttnn, name)

    def topk(self, logits, *args, **kwargs):
        if logits.shape[-2] != 1:
            return ttnn.topk(logits, *args, **kwargs)
        if args or kwargs != {"k": 8, "dim": -1, "sorted": True}:
            raise ValueError("BF16 partition probe implements only the mesh sorted 256-to-8 routing call")
        return self.selector(logits)


def install(dec):
    """Install on a fresh linear decoder during setup; return previous _moe."""
    if dec.kind != "linear_attention" or not hasattr(dec, "inline_selector"):
        raise ValueError("BF16 partition probe requires the original linear-attention InlineSelector")
    if hasattr(dec, "experimental_bf16_partition_selector"):
        raise ValueError("BF16 partition selector is already installed")
    original = dec._moe
    source = original.__func__
    if not {"topk", "softmax"}.issubset(source.__code__.co_names):
        raise ValueError("BF16 partition probe requires the direct MoE topk/softmax body")
    selector = MeshBF16PartitionTopK(dec.mesh_device, dec.inline_selector.kernel_source)
    replacement = types.FunctionType(
        source.__code__,
        dict(source.__globals__, ttnn=_PartitionProxy(selector)),
        source.__name__,
        source.__defaults__,
        source.__closure__,
    )
    replacement.__kwdefaults__ = source.__kwdefaults__
    dec._moe = types.MethodType(replacement, dec)
    dec.experimental_bf16_partition_selector = selector
    dec.precision_policy["experimental_bf16_partition_selector"] = {
        "scope": "singleton linear-attention MoE; multirow native FP32 topk retained",
        "selection_precision": "FP32 input rounded to BF16 RNE before comparison",
        "tie_order": "verbatim inherited InlineSelector v6_select_top8 partition functions",
        "values": "BF16 keys widened to FP32; original IDs retained as UINT32",
        "score_path": "unchanged mesh FP32 softmax followed by BF16 score cast",
        "nan_policy": "canonical BF16 0x7fc0; device-typecast parity unverified",
        "status": "experimental diagnostic; quality gates unchanged",
    }
    return original
