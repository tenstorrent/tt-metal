# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native flat-QKV and direct log-decay experiment; serving defaults untouched."""

import inspect
import json
import os
import sys
import textwrap
from pathlib import Path

import probe
import scan_followup

import ttnn


def flat_tail(self, mixed, z, beta, decay, sequence, key_width, value_width, value_heads, value_dim):
    if isinstance(mixed, tuple):
        q, k, v = mixed
    else:
        q = ttnn.reshape(mixed[..., :key_width], (self.batch, sequence, key_width))
        k = ttnn.reshape(mixed[..., key_width : 2 * key_width], (self.batch, sequence, key_width))
        v = ttnn.reshape(mixed[..., 2 * key_width :], (self.batch, sequence, value_width))
    beta = ttnn.reshape(ttnn.typecast(ttnn.sigmoid(beta), ttnn.float32), (self.batch, sequence, value_heads))
    g = ttnn.multiply(self.weights["a"], ttnn.softplus(ttnn.add(decay, self.weights["dt_bias"])))
    g = ttnn.reshape(ttnn.typecast(g, ttnn.float32), (self.batch, sequence, value_heads))
    mask = getattr(self, "_sequence_mask", None)
    if mask is not None:
        mask = ttnn.typecast(ttnn.reshape(mask, (self.batch, sequence, 1)), ttnn.float32)
        beta = ttnn.multiply(beta, mask)
        g = ttnn.multiply(g, mask)
    constants = scan_followup.native_recurrence.constants
    attended, state = ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=value_dim**-0.5,
        initial_state=ttnn.typecast(self.caches["recurrent"], ttnn.float32),
        output_final_state=True,
        chunk_size=32,
        use_qk_l2norm=False,
        output_head_major=True,
        eye=constants[0],
        tril=constants[1],
        ones=constants[2],
        masks=constants[3],
    )
    ttnn.copy(ttnn.typecast(state, self.policy.linear_recurrent_state_dtype), self.caches["recurrent"])
    attended = ttnn.reshape(attended, (self.batch * value_heads, sequence, 1, value_dim))
    return self._linear_attention_prefill_chunk_tail(attended, z, sequence, value_heads, value_dim, value_width)


def fused_conv(self, mixed, sequence, key_width, value_width):
    chunk = ttnn.to_layout(ttnn.reshape(ttnn.permute(mixed, (0, 1, 3, 2)), (1, sequence, -1)), ttnn.ROW_MAJOR_LAYOUT)
    history = ttnn.permute(self.caches["conv"], (0, 1, 3, 2))
    history = ttnn.to_layout(ttnn.reshape(history[:, :, 1:, :], (1, 3, -1)), ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.experimental.kda.qkv_causal_conv1d_silu(
        chunk,
        history,
        *[self.weights[f"conv_tap{i}"] for i in range(4)],
        key_width,
        key_width,
        value_width,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def configure(candidate):
    scan_followup.configure(candidate)
    cls = probe.md.MultichipDecoder
    original = cls._linear_attention_prefill_chunk_impl
    source = textwrap.dedent(inspect.getsource(original))
    marker = "    query = ttnn.reshape(mixed[..., :key_width]"
    offset = source.index(marker)
    source = (
        source[:offset]
        + """    return flat_tail(self, mixed, z, beta, decay, sequence, key_width, value_width, value_heads, value_dim)
"""
    )
    lines = source.splitlines()
    guard = "hidden_states.shape[2] % 32"
    if os.environ.get("FLAT_FUSED_CONV") == "1":
        guard += " or self.batch != 1"
    lines[1:1] = [f"    if {guard}:", "        return original(self, hidden_states)"]
    source = "\n".join(lines)
    if os.environ.get("FLAT_FUSED_CONV") == "1":
        start = source.index("    convolved = ttnn.multiply(")
        end = source.index("    return flat_tail(", start)
        source = (
            source[:start]
            + "    mixed = fused_conv(self, mixed, sequence, key_width, value_width)\n    ttnn.copy(next_conv_state, self.caches['conv'])\n\n"
            + source[end:]
        )
    namespace = dict(original.__globals__, flat_tail=flat_tail, original=original, fused_conv=fused_conv)
    exec(compile(source, "<native flat prefill experiment>", "exec"), namespace)
    cls._linear_attention_prefill_chunk_impl = namespace[original.__name__]


if __name__ == "__main__":
    os.environ["QWEN_GDN_PHASED"] = "1"
    scan_followup.native_recurrence.captured = []
    probe.configure = configure
    probe.build_generator = scan_followup.build
    probe.main()
    path = Path(sys.argv[sys.argv.index("--result") + 1])
    result = json.loads(path.read_text())
    result["candidate"] = "native_flat_fusedconv" if os.environ.get("FLAT_FUSED_CONV") == "1" else "native_flat"
    result["timing_caveat"] = os.environ.get("TIMING_CAVEAT", "No concurrent HF CPU job during measured runs")
    path.write_text(json.dumps(result, indent=2) + "\n")
