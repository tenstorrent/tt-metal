# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tap singleton prefill intermediates in the existing immediate-twice oracle.

Run with the same arguments as probe_vllm_state_determinism. The added persistent
device copies and blocking host snapshots can mask timing bugs. This diagnostic
does not establish serving, async, or performance acceptance. Production source
is unchanged; ordinary _chunk math and operand ordering are preserved.
"""


def main():
    import argparse
    import sys

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests import probe_vllm_state_determinism as diagnostic
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_vllm_async import read_ranks
    from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM
    from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import Collectives, MultichipDecoder

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--collective-layer", type=int, help="Also tap this actual layer's singleton MLP gather/reducer"
    )
    options, remainder = parser.parse_known_args()
    original_argv = sys.argv
    sys.argv = [sys.argv[0], *remainder]

    original_init = QwenFlashNextForCausalLM.__init__
    original_chunk = MultichipDecoder._chunk
    original_sum, original_gather = Collectives.sum, Collectives.gather
    original_exact_sum = multichip_decoder.exact_rank_sum
    owners, buffers = {}, {}
    collective_buffers = {}
    current = dict(mlp_layer=None, sum_layer=None, exact_seen=False)
    names = (
        "00_input",
        "01_post_attention_residual",
        "02_mlp_mixed",
        "03_mlp_injection",
        "04_mlp_block",
        "05_final_residual",
    )

    def initialize(adapter, generator):
        original_init(adapter, generator)
        for layer in generator.model.layers:
            owners[id(layer)] = generator

    def selected(layer):
        return layer is not None and layer.layer_idx == options.collective_layer

    def sum_with_taps(ccl, value):
        layer = current["mlp_layer"]
        if not selected(layer) or ccl is not layer.ccl:
            return original_sum(ccl, value)
        taps = collective_buffers[id(layer)]
        if value.dtype != ttnn.bfloat16 or tuple(value.shape) != (1, 1, 1, 2560):
            raise AssertionError("Expected exact BF16 singleton MLP collective input")
        ttnn.copy(value, taps["03b_collective_input_bf16"])
        previous = current["sum_layer"], current["exact_seen"]
        current["sum_layer"] = layer
        current["exact_seen"] = False
        try:
            # Preserve the actual selected branch and its custom reducer.
            result = original_sum(ccl, value)
            if not current["exact_seen"]:
                raise AssertionError("Selected collective did not exercise the actual custom reducer")
            if result.dtype != ttnn.float32:
                raise AssertionError("Expected FP32 exact-rank-sum result before caller BF16 cast")
            return result
        finally:
            current["sum_layer"], current["exact_seen"] = previous

    def gather_with_taps(ccl, value, dim=-1):
        result = original_gather(ccl, value, dim=dim)
        layer = current["sum_layer"]
        if selected(layer) and ccl is layer.ccl:
            if result.dtype != ttnn.bfloat16 or tuple(result.shape) != (1, 4, 1, 2560):
                raise AssertionError("Expected BF16 four-rank gathered singleton")
            ttnn.copy(result, collective_buffers[id(layer)]["03c_gathered_bf16"])
        return result

    def exact_sum_with_taps(gathered):
        result = original_exact_sum(gathered)
        layer = current["sum_layer"]
        if selected(layer):
            if result.dtype != ttnn.float32 or tuple(result.shape) != (1, 1, 1, 2560):
                raise AssertionError("Expected actual FP32 custom-reducer output")
            ttnn.copy(result, collective_buffers[id(layer)]["03d_reducer_fp32"])
            current["exact_seen"] = True
        return result

    def chunk(layer, x, state, token_ids=None):
        if x.shape[-2] != 1:
            return original_chunk(layer, x, state, token_ids)
        owner = owners[id(layer)]
        key = id(layer)
        if key not in buffers:
            # First singleton prefill warmup precedes all generator captures.
            # Never permit a missing tap to allocate inside a later capture.
            if owner.prepared or owner.traces or owner.prefill_traces or owner.sampling_traces:
                raise AssertionError("Singleton taps must be allocated before every live trace")
            buffers[key] = {
                name: ttnn.empty(
                    (1, 1, 1, 1 if name == "03_mlp_injection" else x.shape[-1]),
                    dtype=x.dtype,
                    layout=x.layout,
                    device=owner.mesh,
                    memory_config=x.memory_config(),
                )
                for name in names
            }
            if selected(layer):
                collective_buffers[key] = {
                    name: ttnn.empty(
                        shape,
                        dtype=dtype,
                        layout=x.layout,
                        device=owner.mesh,
                        memory_config=x.memory_config(),
                    )
                    for name, shape, dtype in (
                        ("03b_collective_input_bf16", (1, 1, 1, 2560), ttnn.bfloat16),
                        ("03c_gathered_bf16", (1, 4, 1, 2560), ttnn.bfloat16),
                        ("03d_reducer_fp32", (1, 1, 1, 2560), ttnn.float32),
                    )
                }
        taps = buffers[key]
        ttnn.copy(x, taps["00_input"])
        # Same two expressions as MultichipDecoder._chunk, split only to name
        # the original MLP result. Every tap is a copy into existing storage.
        x, mixed, injection = layer._attention_chunk(x, state, token_ids)
        ttnn.copy(x, taps["01_post_attention_residual"])
        ttnn.copy(mixed, taps["02_mlp_mixed"])
        ttnn.copy(injection, taps["03_mlp_injection"])
        previous = current["mlp_layer"]
        current["mlp_layer"] = layer
        try:
            block = layer.mlp(mixed)
        finally:
            current["mlp_layer"] = previous
        ttnn.copy(block, taps["04_mlp_block"])
        output = layer.mlp_mix.write(x, block, injection)
        ttnn.copy(output, taps["05_final_residual"])
        return output

    def extra_fields(generator, kind, kwargs):
        """Persistent singleton layer/optional BF16 gather and FP32 reducer taps add copies that can mask timing bugs; diagnostic only."""
        if kind != "prefill":
            return {}
        ends = kwargs["prompt_lens"]
        starts = kwargs.get("start_pos", [0] * len(ends))
        # A single shared tap set is meaningful only for a singleton submission
        # whose complete prefill call consists of exactly this logical token.
        if len(ends) != 1 or int(ends[0]) - int(starts[0]) != 1:
            return {}
        fields = {}
        for ordinal, layer in enumerate(generator.model.layers):
            if id(layer) not in buffers:
                raise AssertionError("Singleton prefill did not initialize its diagnostic taps")
            actual_layer = generator.model.layer_indices[ordinal]
            for name in names:
                fields[f"intermediate.layer{actual_layer:02d}.{name}"] = read_ranks(buffers[id(layer)][name])
            if selected(layer):
                for name, tensor in collective_buffers[id(layer)].items():
                    fields[f"intermediate.layer{actual_layer:02d}.{name}"] = read_ranks(tensor)
        return fields

    QwenFlashNextForCausalLM.__init__ = initialize
    MultichipDecoder._chunk = chunk
    Collectives.sum, Collectives.gather = sum_with_taps, gather_with_taps
    multichip_decoder.exact_rank_sum = exact_sum_with_taps
    try:
        diagnostic.main(extra_fields=extra_fields)
    finally:
        QwenFlashNextForCausalLM.__init__ = original_init
        MultichipDecoder._chunk = original_chunk
        Collectives.sum, Collectives.gather = original_sum, original_gather
        multichip_decoder.exact_rank_sum = original_exact_sum
        sys.argv = original_argv


if __name__ == "__main__":
    main()
