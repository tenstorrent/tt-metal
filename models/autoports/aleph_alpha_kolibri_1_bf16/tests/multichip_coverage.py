# SPDX-License-Identifier: Apache-2.0
"""Logical tails, circular pages, batches and changed-input traced replay on TP4."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

import ttnn

from ..tt.multichip_decoder import MeshPolicy, MultichipDecoder
from ..tt.optimized_decoder import OptimizedDecoder
from .optimized_coverage import runtime_audit
from .reference import config, load_weights
from .run_decoder import pcc

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "doc/multichip_decoder"
OUT = Path(os.environ.get("MC_ARTIFACT_DIR", str(REFERENCE)))


class Harness:
    def __init__(
        self, layer, baseline=False, capacity=16384, ring=False, ring_tokens=8192, reference_name="coverage_baseline"
    ):
        self.baseline, self.layer, self.capacity = baseline, layer, capacity
        self.cfg = config()
        self.cfg.max_position_embeddings = 1048576
        self.sliding = self.cfg.layer_types[layer] == "sliding_attention"
        if not baseline:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        self.mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1 if baseline else 4),
            physical_device_ids=[0] if baseline else [],
            trace_region_size=10000000,
        )
        weights = load_weights(layer)
        opts = {} if baseline else {"policy": MeshPolicy(**json.loads(os.environ.get("MC_POLICY", "{}")))}
        if not baseline and ring and self.sliding:
            opts["sliding_cache_tokens"] = ring_tokens
        self.model = (OptimizedDecoder if baseline else MultichipDecoder).from_state_dict(
            weights, hf_config=self.cfg, layer_idx=layer, mesh_device=self.mesh, **opts
        )
        del weights
        self.sharded = not baseline and self.model.hidden_width == 640
        self.blocks = (ring_tokens if ring and self.sliding else capacity) // 32
        self.cache = tuple(
            self.tt(torch.zeros(self.blocks, 4, 32, 128, dtype=torch.bfloat16), ttnn.bfloat8_b, dim=1) for _ in range(2)
        )
        self.recorded = torch.load(ROOT / f"doc/optimized_decoder/recorded_inputs/layer_{layer}.pt", weights_only=True)
        self.rows = []
        self.outputs = {}
        self.reference = None if baseline else torch.load(REFERENCE / f"{reference_name}_{layer}.pt", weights_only=True)
        extra_reference = OUT / f"{reference_name}_{layer}.pt"
        if not baseline and OUT != REFERENCE and extra_reference.exists():
            self.reference.update(torch.load(extra_reference, weights_only=True))

    def tt(self, x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, dim=None):
        mapper = (
            None
            if self.baseline
            else (ttnn.ReplicateTensorToMesh(self.mesh) if dim is None else ttnn.ShardTensorToMesh(self.mesh, dim=dim))
        )
        return ttnn.from_torch(
            x.contiguous(),
            device=self.mesh,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def integer(self, x):
        return self.tt(x, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def input(self, n, offset=0, batch=1):
        return torch.cat(
            [self.recorded[:, (torch.arange(n) + offset + r * 137) % self.recorded.shape[1]] for r in range(batch)],
            dim=0,
        )

    def inp(self, x):
        return self.tt(x, dim=3 if self.sharded else None)

    def read(self, x, cache=False):
        if self.baseline:
            return ttnn.to_torch(x)
        pieces = [ttnn.to_torch(v) for v in ttnn.get_device_tensors(x)]
        if cache:
            assert all(v.shape[1] == 1 for v in pieces)
            return torch.cat(pieces, dim=1)
        if self.sharded:
            return torch.cat(pieces, dim=-1)
        first = pieces[0]
        for other in pieces[1:]:
            assert pcc(first, other) > 0.99999
        return first

    def copy(self, value, dest, dim=None):
        mapper = (
            None
            if self.baseline
            else (ttnn.ReplicateTensorToMesh(self.mesh) if dim is None else ttnn.ShardTensorToMesh(self.mesh, dim=dim))
        )
        host = ttnn.from_torch(value.contiguous(), dtype=dest.dtype, layout=dest.layout, mesh_mapper=mapper)
        ttnn.copy_host_to_device_tensor(host, dest)

    def angles(self, positions):
        phase = positions.float()[:, None] / 10000.0 ** (torch.arange(0, 128, 2).float() / 128)
        phase = torch.cat([phase, phase], -1)[None, None]
        return phase.cos().bfloat16(), phase.sin().bfloat16()

    def pages(self, seed, batch=1):
        gen = torch.Generator().manual_seed(seed)
        if batch == 1:
            order = torch.randperm(self.blocks, generator=gen, dtype=torch.int32)
            return order[torch.arange(self.capacity // 32) % self.blocks][None]
        # Each request owns8 physical pages; logical128-token checks stay within them.
        order = torch.randperm(self.blocks, generator=gen, dtype=torch.int32)
        return order[: batch * 8].reshape(batch, 8)

    def check(self, label, actual):
        value = actual.detach().cpu()
        if self.baseline:
            self.outputs[label] = value
        else:
            score = pcc(self.reference[label], value)
            row = dict(case=label, pcc=score)
            if label.startswith("trace_b"):
                row["per_request_pcc"] = [
                    pcc(a, b) for a, b in zip(self.reference[label].reshape(-1, 2560), value.reshape(-1, 2560))
                ]
                assert min(row["per_request_pcc"]) >= 0.995, row
            elif value.ndim == 4 and value.shape[1] > 1:
                row["per_request_pcc"] = [pcc(a, b) for a, b in zip(self.reference[label].unbind(1), value.unbind(1))]
                assert min(row["per_request_pcc"]) >= 0.995, row
            self.rows.append(row)
            print(json.dumps(row), flush=True)
            assert score >= 0.995, row

    def cache_check(self, label, pages, end):
        pos = torch.arange(max(0, end - 513), end)
        for role, cache in zip(("k", "v"), self.cache):
            cpu = self.read(cache, cache=True)
            logical = cpu[pages[0, pos // 32].long(), :, pos % 32, :]
            self.check(label + "_" + role, logical)

    def close(self):
        ttnn.close_mesh_device(self.mesh)


def run(a):
    torch.set_num_threads(8)
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") != "1"
    h = Harness(a.layer, a.baseline, ring=a.ring, ring_tokens=a.ring_tokens)
    try:
        lengths = [int(v) for v in a.lengths.split(",")]
        for length in lengths:
            pages = h.pages(length)
            plan = h.model.prepare_prefill(page_table_host=pages, seq_len=length)
            inp = h.inp(h.input(length)[None])
            with runtime_audit():
                out = h.model.prefill_forward(inp, kv_cache=h.cache, plan=plan)
            h.check(f"prefill_{length}", h.read(out))
            del out, inp, plan
            h.cache_check(f"cache_{length}", pages, length)
        # Nonaligned continuation crosses both page and chunk boundaries.
        pages = h.pages(701)
        first = h.model.prepare_prefill(page_table_host=pages, seq_len=31)
        inp = h.inp(h.input(31)[None])
        out = h.model.prefill_forward(inp, kv_cache=h.cache, plan=first)
        del out, inp, first
        for start, count in ((31, 35), (66, 17)):
            plan = h.model.prepare_prefill(page_table_host=pages, seq_len=count, start_pos=start)
            inp = h.inp(h.input(count, start)[None])
            with runtime_audit():
                out = h.model.prefill_forward(inp, kv_cache=h.cache, plan=plan)
            h.check(f"continuation_{start}_{count}", h.read(out))
            del out, inp, plan
            h.cache_check(f"continuation_cache_{start}", pages, start + count)
        # Warm all batch/program signatures before capturing any live trace.
        variants = []
        for batch in (1, 3, 17, 32):
            pages = h.pages(812 + batch, batch=batch)
            inp = h.inp(h.input(1, 100, batch).reshape(1, 1, batch, 2560))
            positions = torch.arange(batch, dtype=torch.int32) % 7
            c, s = h.angles(positions)
            kw = dict(
                kv_cache=h.cache,
                page_table=h.integer(pages),
                current_pos=h.integer(positions),
                cos=h.tt(c),
                sin=h.tt(s),
            )
            for _ in range(2):
                with runtime_audit():
                    out = h.model.decode_forward(inp, **kw)
                del out
            variants.append((batch, inp, kw, pages))
        gc.collect()
        ttnn.synchronize_device(h.mesh)
        traces = []
        for batch, inp, kw, pages in variants:
            tid = ttnn.begin_trace_capture(h.mesh, cq_id=0)
            with runtime_audit():
                out = h.model.decode_forward(inp, **kw)
            ttnn.end_trace_capture(h.mesh, tid, cq_id=0)
            # Later captures may share backing allocations with earlier traces.
            # Only these exact outputs are corruptible: no output is consumed
            # until its own trace has rewritten it; readback finishes before
            # any other trace runs. Inputs/KV/scratch precede every capture.
            from ttnn.tools import trace_allocation_tracker

            trace_allocation_tracker.acknowledge_corruptible(out)
            traces.append((batch, inp, kw, pages, tid, out))
        try:
            for repeat in range(4):
                for batch, inp, kw, pages, tid, out in traces:
                    # Start at0 each repeat so every cached prefix row is initialized
                    # by this same request. Alternate page ownership and independent rows.
                    changed = pages.flip(-1) if repeat % 2 else pages
                    h.copy(changed, kw["page_table"])
                    for step in range(3):
                        h.copy(
                            h.input(1, 200 + repeat * 19 + step, batch).reshape(1, 1, batch, 2560),
                            inp,
                            dim=3 if h.sharded else None,
                        )
                        pos = torch.full((batch,), step, dtype=torch.int32)
                        h.copy(pos, kw["current_pos"])
                        c, s = h.angles(pos)
                        h.copy(c, kw["cos"])
                        h.copy(s, kw["sin"])
                        ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
                        actual = h.read(out)
                        h.check(f"trace_b{batch}_r{repeat}_s{step}", actual)
                        ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
                        assert torch.equal(actual, h.read(out)), "Replay nondeterminism"
        finally:
            for *_, tid, out in traces:
                ttnn.release_trace(h.mesh, tid)
        if a.baseline:
            target = OUT / f"coverage_baseline_{a.layer}.pt"
            previous = torch.load(target, weights_only=True) if target.exists() else {}
            torch.save(previous | h.outputs, target)
        result = dict(
            layer=a.layer,
            baseline=a.baseline,
            ring=a.ring,
            ring_tokens=a.ring_tokens if a.ring and h.sliding else None,
            lengths=lengths,
            tracking=1,
            skip_program_cache=0,
            trace_variants=[1, 3, 17, 32],
            repeated_replays=96,
            changed_inputs=True,
            changed_pages=True,
            changed_positions=True,
            runtime_audit=True,
            watcher=os.environ.get("TT_METAL_WATCHER"),
            watcher_disable_eth=os.environ.get("TT_METAL_WATCHER_DISABLE_ETH"),
            rows=h.rows,
            source_sha256=hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
        )
        (OUT / f"{a.tag}_{a.layer}.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        h.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--ring", action="store_true")
    p.add_argument("--ring-tokens", type=int, default=8192)
    p.add_argument("--tag", default="coverage")
    p.add_argument("--lengths", default="1,31,32,33,127,128,129,511,512,513,777,4095,4096,4097,8193")
    run(p.parse_args())
