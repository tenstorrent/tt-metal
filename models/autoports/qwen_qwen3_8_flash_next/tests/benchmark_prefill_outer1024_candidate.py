# SPDX-License-Identifier: Apache-2.0
"""Isolated serial-owner view1024 experiment. No production monkeypatches.

Preserve <=512, owner/native B1 and decode. Keep production DRAM placement
above512; extend only compact norm/RM-QK shape guards and MoE group capacity.
Copied experimental functions are checksum-pinned, with one literal guard edit.
"""

import hashlib
import inspect
import textwrap

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_moe1024_candidate as moe
from models.autoports.qwen_qwen3_8_flash_next.tt import gdn_prefill_qk, prefill_view
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import ParallelMoE


def extend_guard(function, sha256):
    source = textwrap.dedent(inspect.getsource(function))
    if hashlib.sha256(source.encode()).hexdigest() != sha256:
        raise ValueError(f"Qualified baseline changed: {function.__qualname__}")
    if source.count("(128, 256, 512)") != 1:
        raise ValueError("Expected exactly one qualified length guard")
    changed = source.replace("(128, 256, 512)", "(128, 256, 512, 1024)")
    namespace = dict(function.__globals__)
    exec(compile(changed, __file__ + ":isolated_length_guard", "exec"), namespace)
    return namespace[function.__name__]


class Outer1024Ops(prefill_view._PrefillOps):
    norm = extend_guard(
        prefill_view._PrefillOps.norm, "20b9a8efcf4484b73e1cad8b03c8f4fac90ea043bf30d022fcd7508445b79372"
    )


rm_recurrence1024 = extend_guard(
    gdn_prefill_qk._rm_recurrence, "fd1a83609474ca26df35d55e90487b9d72a38103ea713ba3e2226d934eb4163c"
)


class Outer1024GDN(gdn_prefill_qk.PrefillQKRowMajorGDN):
    def _recurrence(self, q, k, v, g, beta):
        b, s, _ = q.shape
        if not (self.prefill_qk_rm_enabled and self.recurrence_broadcast and b == 1 and s == 1024):
            return super()._recurrence(q, k, v, g, beta)
        q = self._heads(q, b, s, 4)
        k = self._heads(k, b, s, 4)
        q = ttnn.rms_norm(
            ttnn.typecast(q, ttnn.float32), weight=self.qscale, epsilon=1e-06 / 128, compute_kernel_config=self.compute
        )
        k = ttnn.rms_norm(
            ttnn.typecast(k, ttnn.float32), weight=self.kscale, epsilon=1e-06 / 128, compute_kernel_config=self.compute
        )
        v = ttnn.typecast(self._heads(v, b, s, 12), ttnn.float32)
        return rm_recurrence1024(self, q, k, v, g, beta)


class Outer1024MoE(ParallelMoE):
    def __call__(self, value):
        batch, sequence = tuple(value.shape)[1:3]
        if batch != 1 or sequence > 1024:
            raise ValueError("Experimental MoE is serial prefill only through1024")
        if sequence <= 512:
            # Preserve the exact production group selection for all short tails.
            return super().__call__(value)
        return self.ops.scatter_reduce(moe.group1024(self, value))


def make_candidate(owner, lengths):
    if owner.batch_size != 32 or max(lengths) != 1024:
        raise ValueError("Expected original32-slot owner and prepared1024 view")
    result = prefill_view._decoder(owner, {}, lengths)
    result.ops.__class__ = Outer1024Ops
    result.mlp.__class__ = Outer1024MoE
    # Existing prefill_group_size=512 remains correct for unchanged <=512 calls.
    result.mlp.offsets1024 = result.ops.upload(
        torch.arange(128, dtype=torch.int32) * 1024, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    )
    if owner.kind == "linear_attention":
        result.attention.__class__ = Outer1024GDN
    return result
