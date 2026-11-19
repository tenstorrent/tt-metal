# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exact hash probe; CPU mode never imports TTNN or opens devices.

Device mode prepares all buckets before capture, then reuses traces while
changing tokens, masks, lengths and reset flags, sharing explicit token history
across prefill/decode buckets. Host conversion is only at test boundaries.
This proves the hash submodule, not complete PLE or decoder acceptance.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tt.ngram_hash import (
    _INTEGER_SOURCE,
    EOS,
    MULTIPLIERS,
    OFFSETS,
    PRIMES,
    VOCAB_SIZE,
    DeviceNGramHash,
)


class _CaptureIDs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.empty(0))

    def forward(self, ids):
        return ids.unsqueeze(-1)


class _TokenCache:
    """The exact two-token append/tail policy, without unrelated KV state."""

    def __init__(self, state=None):
        self.state = state

    def get_conv_state(self, layer_idx, state_idx):
        assert layer_idx == 1 and state_idx == 2
        return self.state

    def update_conv_state(self, ids, layer_idx, state_idx, conv_kernel_size):
        assert layer_idx == 1 and state_idx == 2 and conv_kernel_size == 2
        self.state = (torch.cat([self.state, ids], dim=-1) if self.state is not None else ids)[:, -2:].clone()
        return self.state


def make_reference(snapshot):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    assert importlib.metadata.version("transformers") == "5.22.2"
    config = Qwen4ExpTextConfig(**json.loads((snapshot / "config.json").read_text())["text_config"]).to_dict()
    assert (config["vocab_size"], config["ngram_size"], config["heads_per_ngram"]) == (VOCAB_SIZE, 3, 8)
    eos = config["eos_token_id"]
    assert (eos[0] if isinstance(eos, list) else eos) == EOS
    assert tuple(hf._build_layer_multipliers(VOCAB_SIZE, 3, 0, config["seed"]).tolist()) == MULTIPLIERS
    assert tuple(hf._find_nth_prime_after(config["ngram_vocab_size_base"] - 1, i + 1) for i in range(16)) == PRIMES
    # Instantiate only the hash method owner, replacing the enormous embedding
    # with an ID recorder. The actual installed HF forward and shift code runs.
    reference = hf.Qwen4ExpTextNGramEmbedding.__new__(hf.Qwen4ExpTextNGramEmbedding)
    torch.nn.Module.__init__(reference)
    reference.layer_idx = 1
    reference.context_len = 2
    reference.ngram_size = 3
    reference.heads_per_ngram = 8
    reference.eos_token_id = EOS
    reference.layer_multipliers = torch.tensor(MULTIPLIERS, dtype=torch.int64)
    reference.ngram_heads_vocab_sizes = torch.tensor(PRIMES, dtype=torch.int64)
    reference.ngram_heads_offsets = torch.tensor(OFFSETS, dtype=torch.int64)
    reference.ngram_embedding = _CaptureIDs()
    provenance = {
        "transformers": importlib.metadata.version("transformers"),
        "reference_path": hf.__file__,
        "reference_sha256": hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
        "multipliers": MULTIPLIERS,
        "primes": PRIMES,
        "offsets": OFFSETS,
    }
    return reference, provenance


def cpu_probe(reference):
    """Compile the *same* limb/remainder C++ and compare with HF I64 IDs."""
    generator = torch.Generator().manual_seed(601)
    ids = torch.randint(0, VOCAB_SIZE, (4, 1025), generator=generator)
    ids[0, :12] = torch.tensor([0, 1, 2, EOS, 3, 4, EOS, EOS, 248319, 248318, 0, 7])
    ids[:, 30:34] = EOS
    history = torch.tensor([[EOS, EOS], [1, 2], [EOS, 9], [9, EOS]])
    expected = reference(ids, _TokenCache(history.clone()))
    all_ids = torch.cat([history, ids], dim=1)
    triples = (
        torch.stack([reference._shift_right_ignore_eos(all_ids, shift)[:, 2:] for shift in range(3)], dim=-1)
        .contiguous()
        .to(torch.int64)
    )
    source = (
        "#include <cstdint>\n"
        + _INTEGER_SOURCE
        + r"""
extern "C" void hash_many(const uint32_t* tokens, const uint32_t* lo, const uint32_t* hi,
                          const uint32_t* primes, const uint32_t* offsets, uint32_t* out, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        Limbs a = product(tokens[i*3], lo[0], hi[0]);
        Limbs b = product(tokens[i*3+1], lo[1], hi[1]);
        Limbs c = product(tokens[i*3+2], lo[2], hi[2]);
        Limbs mixed2 = {a.lo ^ b.lo, a.hi ^ b.hi};
        Limbs mixed3 = {mixed2.lo ^ c.lo, mixed2.hi ^ c.hi};
        for (uint32_t h = 0; h < 16; ++h)
            out[i*16+h] = remainder(h < 8 ? mixed2 : mixed3, primes[h]) + offsets[h];
    }
}
extern "C" uint32_t check_all_products(const uint32_t* lo, const uint32_t* hi) {
    for (uint32_t token = 0; token < 248320; ++token) {
        for (uint32_t j = 0; j < 3; ++j) {
            const Limbs actual = product(token, lo[j], hi[j]);
            const uint64_t expected = uint64_t(token) * ((uint64_t(hi[j]) << 32) | lo[j]);
            if (actual.lo != uint32_t(expected) || actual.hi != uint32_t(expected >> 32)) return 1;
        }
    }
    return 0;
}
"""
    )
    array = lambda values: (ctypes.c_uint32 * len(values))(*values)
    lo = array([x & 0xFFFFFFFF for x in MULTIPLIERS])
    hi = array([x >> 32 for x in MULTIPLIERS])
    output = array([0] * expected.numel())
    with tempfile.TemporaryDirectory(prefix="qwen-ngram-cpu-") as folder:
        cpp, so = Path(folder) / "hash.cpp", Path(folder) / "hash.so"
        cpp.write_text(source)
        subprocess.run(["c++", "-std=c++17", "-O2", "-shared", "-fPIC", str(cpp), "-o", str(so)], check=True)
        library = ctypes.CDLL(str(so))
        assert library.check_all_products(lo, hi) == 0
        library.hash_many(array(triples.flatten().tolist()), lo, hi, array(PRIMES), array(OFFSETS), output, ids.numel())
    actual = torch.tensor(list(output), dtype=torch.int64).reshape(expected.shape)
    assert torch.equal(actual, expected), "Exact limb hash differs from installed HF forward"
    # Stateful HF split control exercises initial single token and EOS boundaries.
    cache = _TokenCache(history.clone())
    chunks = [reference(piece, cache) for piece in ids.split([1, 31, 1, 32, 255, 1, 704], dim=1)]
    assert torch.equal(torch.cat(chunks, dim=1), expected)
    return {"all_valid_token_products": VOCAB_SIZE * 3, "hf_exact_hash_ids": expected.numel(), "hf_chunk_control": True}


def device_probe(reference, *, batch, max_sequence):
    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    traces = {}
    report = []
    try:

        def upload(value):
            return ttnn.from_torch(value.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

        def refresh(value, target):
            host = ttnn.from_torch(value.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, target)

        history = torch.zeros(batch, 16, dtype=torch.int64)
        history[:, :2] = EOS
        device_history = upload(history)
        buckets = sorted(set([1, 33, max_sequence]))
        operations = {}
        for sequence in buckets:
            tokens = torch.zeros(batch, sequence, 16, dtype=torch.int64)
            control = torch.zeros(batch, 16, dtype=torch.int64)
            control[:, 1] = 1
            operations[sequence] = DeviceNGramHash(
                tokens=upload(tokens),
                control=upload(control),
                history=device_history,
                output=upload(tokens),
            )
            operations[sequence].prepare()
            operations[sequence].forward()
        ttnn.synchronize_device(mesh)
        for sequence, operation in operations.items():
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            operation.forward()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[sequence] = trace
        refresh(history, device_history)
        generator = torch.Generator().manual_seed(602)
        scenarios = [
            (33, 33, True),
            (1, 1, False),
            (max_sequence, max_sequence, False),
            (33, 7, False),
            (1, 1, True),
            (33, 32, False),
            (33, 31, True),
        ]
        for ordinal, (sequence, logical_length, reset) in enumerate(scenarios):
            operation = operations[sequence]
            packed = torch.zeros(batch, sequence, 16, dtype=torch.int64)
            packed[:, :, 0] = torch.randint(0, VOCAB_SIZE, (batch, sequence), generator=generator)
            packed[:, :, 1] = 1
            packed[:, 0, 0] = EOS if ordinal % 2 == 0 else VOCAB_SIZE - 1
            if sequence > 4:
                packed[:, 2:4, 0] = EOS
                packed[:, 4, 1] = 0
            control = torch.zeros(batch, 16, dtype=torch.int64)
            control[:, 0] = torch.tensor([max(0, logical_length - row % 4) for row in range(batch)])
            # Include independent resets and a true zero-length inactive row.
            control[:, 1] = int(reset)
            if batch > 1:
                control[1, 1] = int(not reset)
            expected = torch.zeros(batch, sequence, 16, dtype=torch.int64)
            for row in range(batch):
                length = int(control[row, 0])
                prior = torch.full((1, 2), EOS) if control[row, 1] else history[row : row + 1, :2].clone()
                if length:
                    ids = torch.where(packed[row : row + 1, :length, 1].bool(), packed[row : row + 1, :length, 0], EOS)
                    cache = _TokenCache(prior)
                    expected[row : row + 1, :length] = reference(ids, cache)
                    history[row, :2] = cache.state[0]
                else:
                    history[row, :2] = prior[0]
            refresh(packed, operation.tensors[0])
            refresh(control, operation.tensors[1])
            ttnn.execute_trace(mesh, traces[sequence], cq_id=0, blocking=True)
            actual = ttnn.to_torch(operation.tensors[-1]).to(torch.int64)
            actual_history = ttnn.to_torch(device_history).to(torch.int64)
            assert torch.equal(actual, expected), f"Hash mismatch scenario {ordinal}, S={sequence}"
            assert torch.equal(actual_history, history), f"History mismatch scenario {ordinal}"
            report.append(
                {
                    "scenario": ordinal,
                    "physical_sequence": sequence,
                    "lengths": control[:, 0].tolist(),
                    "reset": control[:, 1].tolist(),
                    "exact_match": True,
                    "trace_id": str(traces[sequence]),
                    "compared_hash_ids": expected.numel(),
                }
            )
        return {"batch": batch, "scenarios": report, "capture_count": len(traces), "trace_reuse": True}
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--max-sequence", type=int, default=257)
    args = parser.parse_args()
    reference, provenance = make_reference(args.snapshot)
    report = {"provenance": provenance, "cpu": cpu_probe(reference)}
    if args.device:
        report["device"] = device_probe(reference, batch=args.batch, max_sequence=args.max_sequence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "cpu": report["cpu"], "device": args.device, "output": str(args.output)}))


if __name__ == "__main__":
    main()
