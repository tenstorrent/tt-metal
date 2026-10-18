"""Device seed derivation, deterministic streams, and fixed-logit distribution."""

import json
from collections import Counter
from pathlib import Path
import torch
import ttnn
from models.common.sampling.generator import _hash_request_seed_to_device_seed
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_flash_next.tt.state_ops import advance_positions, prepare_sample_seeds


def host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def main():
    torch.set_num_threads(8)
    root = Path("models/autoports/qwen_qwen3_8_flash_next")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=160000000)
    gen = None
    advance_trace = None
    try:
        gen = build_generator(
            root,
            mesh,
            layer_indices=[],
            batch_size=32,
            max_seq_len=2048,
            checkpoint="/home/vkovacevic/qwen38-flash/hf-local/de4b8e4d43b917e7706784d8bb445c9af86a3540",
        )
        gen.prepare()
        checks = []
        for seed in (0, 123, 0xFFFFFFFF):
            gen.set_sampling_params(seed=seed)
            for position in (0, 7, 2047, 262144):
                positions = torch.zeros(32, 32, dtype=torch.int32)
                positions[:, 0] = position + torch.arange(32)
                gen._copy(positions, gen.tt_pos)
                prepare_sample_seeds(gen.seed_init, gen.tt_pos, gen.sample_seeds)
                actual = host(gen.sample_seeds).reshape(-1).tolist()
                expected = [
                    _hash_request_seed_to_device_seed((seed + i) & 0xFFFFFFFF, position + i, salt=i) for i in range(32)
                ]
                assert actual == expected, (seed, position, actual, expected)
                checks.append(dict(seed=seed, position=position, exact=True))
        # Warm this control operation before its separate test-only capture.
        gen._positions([0] * 32)
        advance_positions(gen.tt_pos, gen.tt_active)
        ttnn.synchronize_device(mesh)
        advance_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        advance_positions(gen.tt_pos, gen.tt_active)
        ttnn.end_trace_capture(mesh, advance_trace, cq_id=0)
        logits = torch.full(tuple(gen.tt_logits.shape), -100.0, dtype=torch.bfloat16)
        logits[..., :2] = 0
        gen._copy(logits, gen.tt_logits)
        trials = []
        for seed in (123, 123, 987):
            gen.set_sampling_params(top_k=8, top_p=1.0, temperature=1.0, seed=seed)
            gen.reset()
            before = dict(gen.counters)
            for _ in range(256):
                ttnn.execute_trace(mesh, advance_trace, cq_id=0, blocking=False)
                gen._replay("sample")
            trial = host(gen.tt_history)[:, 1:257].long()
            trials.append(trial)
            for name in ("token_refreshes", "position_refreshes", "page_table_refreshes", "token_readbacks"):
                assert gen.counters[name] == before.get(name, 0)
        assert torch.equal(trials[0], trials[1])
        changed_fraction = float((trials[0] != trials[2]).float().mean())
        assert changed_fraction > 0.5
        allowed = {r * 62080 + i for r in range(4) for i in range(2)}
        counts = Counter(trials[0].reshape(-1).tolist())
        assert set(counts) == allowed, counts
        expected = 32 * 256 / 8
        max_relative_error = max(abs(n / expected - 1) for n in counts.values())
        assert max_relative_error < 0.2, (counts, max_relative_error)
        report = dict(
            status="pass",
            batch=32,
            hash_checks=checks,
            hash_reference="models/common/sampling/generator.py::_hash_request_seed_to_device_seed",
            seed_counter="absolute device position",
            same_seed_exact=True,
            different_seed_changed_fraction=changed_fraction,
            fixed_logits="Two equal local logits per rank; eight equal global candidates; k8,p1,temp1",
            counts=dict(counts),
            samples=32 * 256,
            max_relative_count_error=max_relative_error,
            no_per_token_host_refresh=True,
            production_traces=4,
            test_only_advance_traces=1,
            scope="Native seed metadata and common Sampling1D; no decoder layers; real sampler shapes and mesh",
        )
        (root / "doc/full_model/seed_contract.json").write_text(json.dumps(report, indent=2) + "\n")
        print("SEED_CONTRACT_PASS", json.dumps(report), flush=True)
    finally:
        if advance_trace is not None:
            ttnn.release_trace(mesh, advance_trace)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
