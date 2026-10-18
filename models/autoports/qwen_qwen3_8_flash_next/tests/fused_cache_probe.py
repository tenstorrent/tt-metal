# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unmeasured real-QSA cache transport diagnosis; run only with a device lease.

Stops each QSA call directly after paired K/V writes and checks physical indices
and entire caches against exact BF16 row copies. Host reads are intentional
probe instrumentation, outside every benchmark and production runtime.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="1,33")
    parser.add_argument("--positions", default="0,31,511")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument(
        "--legacy-adapter", type=Path, help="Source snapshot from the failed candidate, for causal control"
    )
    args = parser.parse_args()
    import torch
    from transformers import AutoConfig

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests import fused_cache_candidate as candidate
    from models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_decoder import CHECKPOINT
    from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState

    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    if args.legacy_adapter:
        tree = ast.parse(args.legacy_adapter.read_text())
        assignment = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_PHYSICAL_INDICES" for t in n.targets)
        )
        candidate._PHYSICAL_INDICES = ast.literal_eval(assignment.value)
    candidate.snapshot_provenance(args.evidence_dir)
    report = {
        "legacy_adapter": str(args.legacy_adapter) if args.legacy_adapter else None,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": [],
    }
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    state = CheckpointState(CHECKPOINT, "model.language_model.layers.3.self_attn.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)

    class ProbeComplete(Exception):
        pass

    try:
        model = candidate.IndexedCacheQSA.from_state_dict(
            state, hf_config=cfg, layer_idx=3, mesh_device=mesh, batch_size=1, max_context=args.context
        )
        rng = torch.Generator().manual_seed(2026)
        page_host = torch.randperm(args.context // model.page_size, generator=rng).reshape(1, -1).int()

        def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout, device=mesh)

        page = upload(page_host, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        original = candidate.IndexedCacheQSA._update_kv_indexed

        def inspect_update(self, k, v, positions, page_table):
            key, value = ttnn.to_torch(k), ttnn.to_torch(v)
            start = int(ttnn.to_torch(positions)[0, 0])
            seq = key.shape[2]
            expected_indices = torch.full((1, 128), -1, dtype=torch.int32)
            before = [ttnn.to_torch(cache).clone() for cache in (self.k_cache, self.v_cache)]
            for token in range(seq):
                absolute = start + token
                physical_page = int(page_host[0, absolute // self.page_size])
                row = absolute % self.page_size
                expected_indices[0, token] = physical_page * self.page_size + row
                before[0][physical_page, :, row, :] = key[0, :, token, :]
                before[1][physical_page, :, row, :] = value[0, :, token, :]
            original(self, k, v, positions, page_table)
            actual_indices = ttnn.to_torch(self._physical_update_indices)
            actual = [ttnn.to_torch(cache) for cache in (self.k_cache, self.v_cache)]
            check = {
                "seq": seq,
                "position": start,
                "indices_exact": torch.equal(expected_indices, actual_indices),
                "expected_indices": expected_indices[0, :seq].tolist(),
                "actual_indices": actual_indices[0, :seq].tolist(),
                "cache_mismatches": [int(torch.count_nonzero(a != e)) for a, e in zip(actual, before)],
                "input_padded_shapes": [list(k.padded_shape), list(v.padded_shape)],
            }
            report["checks"].append(check)
            (args.evidence_dir / "cache_probe.json").write_text(json.dumps(report, indent=2) + "\n")
            print("CACHE_TRANSPORT_CHECK " + json.dumps(check), flush=True)
            if not check["indices_exact"] or any(check["cache_mismatches"]):
                torch.save(
                    {
                        "indices": actual_indices,
                        "expected_indices": expected_indices,
                        "key": key,
                        "value": value,
                        "actual": actual,
                        "expected": before,
                    },
                    args.evidence_dir / f"failure_s{seq}_p{start}.pt",
                )
            raise ProbeComplete

        with patch.object(candidate.IndexedCacheQSA, "_update_kv_indexed", inspect_update):
            for seq in map(int, args.lengths.split(",")):
                for start in map(int, args.positions.split(",")):
                    if not 1 <= seq <= 128 or not 0 <= start <= args.context - seq:
                        raise ValueError("Probe range exceeds candidate contract")
                    hidden = upload((torch.randn(1, 1, seq, 2560, generator=rng) * 0.05).bfloat16())
                    position = torch.zeros(1, 32, dtype=torch.int32)
                    position[0, 0] = start
                    position = upload(position, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                    try:
                        model._chunk_forward(hidden, current_pos=position, page_table=page)
                    except ProbeComplete:
                        pass
                    else:
                        raise AssertionError("Native cache hook was not exercised")
        assert all(c["indices_exact"] and not any(c["cache_mismatches"]) for c in report["checks"]), report
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
