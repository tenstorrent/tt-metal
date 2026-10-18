"""Exact HF parity for uint64 hashing, EOS boundaries and request slot history."""
import json
from pathlib import Path

import pytest
import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.ple_hash import ngram_indices


class ReturnIndices(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.empty(0)

    def forward(self, ids):
        return ids.unsqueeze(-1)


@pytest.mark.parametrize("physical", [1, 14, 32, 33])
def test_hash_trace(physical):
    stats = Path(__file__).resolve().parents[1] / "doc/functional_decoder/weight_stats/layer_1.json"
    config = Qwen4ExpTextConfig(**json.loads(stats.read_text())["text_config"])
    # Tiny placeholder storage on meta; hash constants are independent of weight values.
    with torch.device("meta"):
        ref = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, 1)
    from transformers.models.qwen4_exp.modeling_qwen4_exp import _build_layer_multipliers

    ref.layer_multipliers = _build_layer_multipliers(config.vocab_size, config.ngram_size, 0, config.seed)
    ref.ngram_heads_vocab_sizes = torch.tensor(ref.head_vocab_sizes, dtype=torch.int64)
    ref.ngram_heads_offsets = torch.tensor(ref.head_offsets, dtype=torch.int64)
    ref.ngram_embedding = ReturnIndices()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:

        def device(x):
            return ttnn.from_torch(x, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

        def lane(values):
            x = torch.zeros(2, 32, dtype=torch.int32)
            x[:, 0] = torch.tensor(values)
            return x

        tokens = device(torch.zeros(2, physical, dtype=torch.int32))
        positions = device(lane([0, 0]))
        slots = device(lane([3, 1]))
        valid = device(lane([1, 1]))
        state = device(torch.zeros(4, 32, dtype=torch.int32))
        output = device(torch.zeros(2 * physical, 16, dtype=torch.int32))

        def forward():
            return ngram_indices(
                tokens,
                positions,
                slots,
                state,
                valid_lengths=valid,
                output=output,
                multipliers=ref.layer_multipliers.tolist(),
                primes=ref.head_vocab_sizes,
                offsets=ref.head_offsets,
                eos=config.eos_token_id,
            )

        forward()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        histories = {1: [], 3: []}
        for step, lengths in enumerate(([31, 29], [1, 3], [2, 1], [13, 17])):
            lengths = [min(x, physical) for x in lengths]
            order = [3, 1] if step % 2 == 0 else [1, 3]
            torch.manual_seed(5 + step)
            x = torch.randint(0, config.vocab_size, (2, physical), dtype=torch.int32)
            x[:, min(1, physical - 1)] = config.eos_token_id
            if step == 3:
                histories = {1: [], 3: []}
            pos = [len(histories[s]) for s in order]
            expected = torch.zeros(2, physical, 16, dtype=torch.int64)
            for b, s in enumerate(order):
                histories[s].extend(x[b, : lengths[b]].tolist())
                expected[b, : lengths[b]] = ref(torch.tensor([histories[s]]), None)[0, -lengths[b] :]
            for dst, src in ((tokens, x), (positions, lane(pos)), (slots, lane(order)), (valid, lane(lengths))):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(src, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), dst
                )
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(output).reshape(2, physical, 16).long()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print("PLE_HASH_HF_PARITY_SLOT_SWAP_TAIL_EOS_TRACE_PASS")
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
