# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated GDN tail graph rewrites; set TAIL_FUSION=concat|activation|both|gated."""

import inspect
import json
import os
import sys
import textwrap
from pathlib import Path

import probe

original_configure = probe.configure


def configure(candidate):
    original_configure(candidate)
    cls = probe.md.MultichipDecoder
    original = cls._linear_attention_prefill_chunk_tail
    source = textwrap.dedent(inspect.getsource(original))
    mode = os.environ.get("TAIL_FUSION", "concat")
    if mode not in ("concat", "activation", "both", "gated"):
        raise ValueError(mode)
    if mode == "gated":
        start = source.index("    output = ttnn.reshape(attended,")
        stop = source.index("    return self._tp_linear(", start)
        source = (
            source[:start]
            + """    attended = ttnn.reshape(attended, (self.batch * value_heads, sequence, value_dim))
    z = ttnn.reshape(z, (self.batch, sequence, value_width))
    output = ttnn.experimental.kda.sigmoid_gated_rms_norm(
        attended, z, ttnn.reshape(self.weights["gated_norm"], (value_dim,)), value_heads,
        epsilon=self.eps, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16,
    )
    output = ttnn.reshape(ttnn.multiply(output, z), (1, self.batch, sequence, value_width))
"""
            + source[stop:]
        )
        # Native output already has the fused norm's exact head-major contract.
        # Remove the padded rank4 detour only for this consuming-graph rewrite.
        probe.patch_function(
            cls,
            "_linear_attention_native_prefill",
            [("    attended = ttnn.reshape(attended, (self.batch * 12, sequence, 1, 128))\n", "")],
        )
    if mode in ("concat", "both"):
        old = "ttnn.permute(ttnn.multiply(output, ttnn.silu(z)), (0, 2, 1, 3))"
        new = "ttnn.experimental.nlp_concat_heads(ttnn.multiply(output, ttnn.silu(z)), memory_config=ttnn.DRAM_MEMORY_CONFIG)"
        if old not in source:
            raise ValueError("Tail concat source changed")
        source = source.replace(old, new)
    if mode in ("activation", "both"):
        old = "ttnn.multiply(output, ttnn.silu(z))"
        new = "ttnn.multiply(output, z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])"
        if old not in source:
            raise ValueError("Tail activation source changed")
        source = source.replace(old, new)
    namespace = dict(original.__globals__)
    exec(compile(source, "<native GDN tail fusion experiment>", "exec"), namespace)
    cls._linear_attention_prefill_chunk_tail = namespace[original.__name__]


if __name__ == "__main__":
    probe.configure = configure
    probe.main()
    path = Path(sys.argv[sys.argv.index("--result") + 1])
    result = json.loads(path.read_text())
    result["candidate"] = "native_tail_" + os.environ.get("TAIL_FUSION", "concat")
    path.write_text(json.dumps(result, indent=2) + "\n")
