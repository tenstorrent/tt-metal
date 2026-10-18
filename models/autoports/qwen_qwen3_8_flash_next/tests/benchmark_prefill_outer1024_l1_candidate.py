# SPDX-License-Identifier: Apache-2.0
"""Isolated outer1024: L1 projection/residual family with HC norm in DRAM.

Only the copied GDN serial view changes. QSA keeps its original512 internal
chunks. S<=512, other tails, owner and native B1 retain their original paths.
This is a fit/performance hypothesis, not adoption.
"""

import hashlib
import inspect
import textwrap
from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_outer1024_candidate as base
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder, prefill_view

BASE_SHA256 = "e1e4f5c31e46614be42b963695e8dd3e37015d95703e12534ee7ec238a6c950b"
CHUNK_SHA256 = "c5afac8cc355d5626e2c0170d86465aa96fd9c8aa849ae5994fac565572ec35a"
assert hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() == BASE_SHA256
moe = base.moe  # Preserve the frozen probe's explicit source-provenance chain.


def extended_chunk():
    source = textwrap.dedent(inspect.getsource(multichip_decoder.MultichipDecoder._chunk))
    assert hashlib.sha256(source.encode()).hexdigest() == CHUNK_SHA256
    old = "self.ops.l1_prefill = not decode and x.shape[1] == 1 and x.shape[2] <= 512"
    new = "self.ops.l1_prefill = not decode and x.shape[1] == 1 and (x.shape[2] <= 512 or x.shape[2] == 1024)"
    assert source.count(old) == 1
    namespace = dict(multichip_decoder.__dict__)
    exec(compile(source.replace(old, new), __file__ + ":isolated_l1_1024_guard", "exec"), namespace)
    return namespace["_chunk"]


class Outer1024L1Ops(base.Outer1024Ops):
    def norm(self, value, weight, eps, *, reuse_output=False):
        if tuple(value.shape) != (1, 1024, 4, 640):
            return super().norm(value, weight, eps, reuse_output=reuse_output)
        original_memory = str(value.memory_config())
        # The padded HC shape occupies 40 MiB, despite only 5 MiB of logical rows.
        # Keep its compact normalization/restoration chain in DRAM. Matmul's
        # existing l1_prefill flag independently selects L1 inputs/outputs.
        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        result = super().norm(value, weight, eps, reuse_output=reuse_output)
        assert result.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        signature = self.norm_compaction["1024"]
        signature.update(
            original_input_memory_config=original_memory,
            weighted_stream_memory_config=str(result.memory_config()),
            weighted_stream_shape=list(result.shape),
            weighted_stream_padded_shape=list(result.padded_shape),
            projection_family_l1=bool(self.l1_prefill),
            candidate_scope="S1024 serial-view only; padded norm/restored streams DRAM",
        )
        return result


class Outer1024L1Decoder(prefill_view._PrefillDecoder):
    _chunk = extended_chunk()


def make_candidate(owner, lengths):
    if owner.batch_size != 32 or max(lengths) != 1024:
        raise ValueError("Expected original32-slot owner and prepared1024 view")
    if owner.kind != "linear_attention":
        # Whole1024 QSA was not numerically qualified. Keep the complete QSA
        # decoder, including HC/PLE/MoE, on the original internal512 geometry.
        return prefill_view._decoder(owner, {}, tuple(length for length in lengths if length <= 512))
    result = base.make_candidate(owner, lengths)
    result.__class__ = Outer1024L1Decoder
    result.ops.__class__ = Outer1024L1Ops
    return result
