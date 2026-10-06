# SPDX-License-Identifier: Apache-2.0
"""Real-shape logical API, continuation, cache ownership and trace lifetime tests."""

import argparse
import gc
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder
from .optimized_provenance import provenance
from .reference import ReferenceDecoder, config, load_weights
from .run_decoder import pcc

ROOT = Path(__file__).resolve().parents[1]
STD = 0.48495227098464966


@contextmanager
def runtime_audit():
    """Reject Python torch execution and explicit TTNN host transfers in forward."""
    seen = []
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call":
            module = frame.f_globals.get("__name__", "")
            name = frame.f_code.co_name
            if module == "torch" or module.startswith("torch."):
                seen.append(module + "." + name)
        elif event == "c_call":
            module = getattr(arg, "__module__", "") or ""
            if module == "torch" or module.startswith("torch."):
                seen.append(module + "." + getattr(arg, "__name__", "?"))

    forbidden = {
        name: getattr(ttnn, name)
        for name in ("from_torch", "to_torch", "as_tensor", "to_device", "copy_host_to_device_tensor", "from_device")
    }

    def fail(*a, **kw):
        raise AssertionError("Host transfer inside forward")

    for name in forbidden:
        setattr(ttnn, name, fail)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)
        for name, value in forbidden.items():
            setattr(ttnn, name, value)
    assert not seen, seen[:10]


def decoder_options():
    from ..tt.optimized_decoder import DecoderPolicy

    return {"policy": DecoderPolicy(**json.loads(os.environ.get("OPT_POLICY", "{}")))}


class Harness:
    def __init__(self, layer, capacity=1024, synthetic=False, allocation_tracking=True):
        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == ("1" if allocation_tracking else "0")
        assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") != "1"
        self.provenance = provenance()
        print("PROVENANCE", json.dumps(self.provenance), flush=True)
        self.layer = layer
        self.capacity = capacity
        self.blocks = (capacity + 31) // 32
        self.weights = load_weights(
            layer, stats_path=ROOT / f"doc/functional_decoder/weight_stats_{layer}.json", synthetic=synthetic
        )
        self.ref = ReferenceDecoder(self.weights, layer)
        mesh_options = {}
        if os.environ.get("OPT_WORKER_L1_SIZE"):
            mesh_options["worker_l1_size"] = int(os.environ["OPT_WORKER_L1_SIZE"])
        self.mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0, **mesh_options
        )
        decoder_class = OptimizedDecoder
        if os.environ.get("OPT_PROJECTIONS"):
            from .optimized_projection_candidates import ProjectionCandidate

            decoder_class = ProjectionCandidate
        if os.environ.get("OPT_LAYOUT"):
            from .optimized_layout_candidates import LayoutCandidate

            decoder_class = LayoutCandidate
        if os.environ.get("OPT_PREFILL"):
            from .optimized_prefill_candidates import PrefillCandidate

            decoder_class = PrefillCandidate
        if os.environ.get("OPT_SPLIT"):
            from .optimized_split_candidates import SplitCandidate

            decoder_class = SplitCandidate
        if os.environ.get("OPT_RUNTIME"):
            from .optimized_runtime_candidates import RuntimeCandidate

            decoder_class = RuntimeCandidate
        self.model = decoder_class.from_state_dict(
            self.weights, hf_config=config(), layer_idx=layer, mesh_device=self.mesh, **decoder_options()
        )
        self.cache = tuple(
            self.tt(
                torch.zeros(self.blocks, 4, 32, 128, dtype=torch.bfloat16),
                dtype=getattr(ttnn, os.environ.get("OPT_CACHE", "bfloat8_b")),
            )
            for _ in range(2)
        )
        self.provenance["cache_dtype"] = str(self.cache[0].dtype)
        self.provenance["decoder_class"] = type(self.model).__module__ + "." + type(self.model).__name__
        if hasattr(self.model, "policy"):
            self.provenance["resolved_policy"] = asdict(self.model.policy)
        print("DECODER_CLASS", self.provenance["decoder_class"], flush=True)
        self.rows = []
        self.input_offset = 0
        self.recorded = None
        if os.environ.get("OPT_REAL_INPUT") == "1":
            self.recorded = torch.load(
                ROOT / f"doc/optimized_decoder/recorded_inputs/layer_{layer}.pt", weights_only=True
            )

    def inputs(self, length):
        if self.recorded is None:
            return (torch.randn(1, length, 2560) * STD).bfloat16()
        indices = (torch.arange(length) + self.input_offset) % self.recorded.shape[1]
        self.input_offset += length
        return self.recorded[:, indices].contiguous()

    def tt(self, v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            v.contiguous(), device=self.mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def integer(self, v):
        return self.tt(v, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def copy(self, v, dest):
        host = ttnn.from_torch(v.contiguous(), dtype=dest.dtype, layout=dest.layout)
        ttnn.copy_host_to_device_tensor(host, dest)

    def angles(self, start, n):
        p = torch.arange(start, start + n).float()[:, None] / (10000.0 ** (torch.arange(0, 128, 2).float() / 128))
        p = torch.cat([p, p], -1)[None, None]
        return p.cos().bfloat16(), p.sin().bfloat16()

    def check(self, name, expected, actual, **metadata):
        value = pcc(expected, actual)
        row = dict(case=name, pcc=value, **metadata)
        self.rows.append(row)
        print(json.dumps(row), flush=True)
        assert value >= 0.995, row

    def close(self):
        ttnn.close_mesh_device(self.mesh)


def run(layer, synthetic=False, output=None):
    torch.set_num_threads(8)
    torch.manual_seed(9100 + layer)
    h = Harness(layer, capacity=16384, synthetic=synthetic)
    cache_checks = []
    try:
        # Increase lengths on one loaded instance, then reuse a short request.
        for length in [1, 31, 32, 33, 127, 128, 129, 511, 512, 513, 777, 1025, 8193, 17]:
            h.ref.reset()
            x = h.inputs(length)
            pages = torch.randperm(h.blocks, dtype=torch.int32)[None]
            plan = h.model.prepare_prefill(page_table_host=pages, seq_len=length)
            inp = h.tt(x[None])
            expected = h.ref(x)
            before_cache = [ttnn.to_torch(t) for t in h.cache] if length in (33, 777) else None
            with runtime_audit():
                out = h.model.prefill_forward(inp, kv_cache=h.cache, plan=plan)
            actual = ttnn.to_torch(out)[0]
            h.check("logical_prefill", expected, actual, length=length)
            if before_cache is not None:
                untouched = pages[0, (length + 31) // 32 :].long()
                for before, tensor in zip(before_cache, h.cache):
                    assert torch.equal(before[untouched], ttnn.to_torch(tensor)[untouched])
                cache_checks.append(
                    dict(mode="prefill", length=length, untouched_pages=int(untouched.numel()), exact=True)
                )
            if length == 129:
                continuation = h.inputs(173)
                next_plan = h.model.prepare_prefill(page_table_host=pages, seq_len=173, start_pos=length)
                next_inp = h.tt(continuation[None])
                next_expected = h.ref(continuation, start=length)
                with runtime_audit():
                    next_out = h.model.prefill_forward(next_inp, kv_cache=h.cache, plan=next_plan)
                h.check("nonaligned_prefix_continuation", next_expected, ttnn.to_torch(next_out), start=129, length=173)
                follow = h.inputs(1)
                follow_expected = h.ref(follow, start=302)
                c, s = h.angles(302, 1)
                follow_in = h.tt(follow.reshape(1, 1, 1, 2560))
                follow_kw = dict(
                    kv_cache=h.cache,
                    page_table=h.integer(pages),
                    current_pos=h.integer(torch.tensor([302], dtype=torch.int32)),
                    cos=h.tt(c),
                    sin=h.tt(s),
                )
                with runtime_audit():
                    follow_out = h.model.decode_forward(follow_in, **follow_kw)
                h.check(
                    "decode_after_nonaligned_continuation", follow_expected, ttnn.to_torch(follow_out), position=302
                )
                del next_plan, next_inp, next_out, follow_in, follow_kw, follow_out
                h.ref.reset()
                control_plan = h.model.prepare_prefill(page_table_host=pages, seq_len=length, chunk_size=160)
                control = h.model.prefill_forward(inp, kv_cache=h.cache, plan=control_plan)
                h.check("unchunked_control", expected, ttnn.to_torch(control), length=length)
                h.check("chunk_vs_unchunked", actual, ttnn.to_torch(control), length=length)
                del control_plan, control
            del plan, inp, out
        # Stable streaming chunk buffers: all shapes and modes prepared first.
        pages = torch.arange(h.blocks, dtype=torch.int32)[None]
        table = h.integer(pages)
        buckets = {}
        for n in [32, 64, 96, 128, 4096]:
            c, s = h.angles(0, n)
            inp = h.tt(torch.zeros(1, 1, n, 2560, dtype=torch.bfloat16))
            kw = dict(
                kv_cache=h.cache,
                page_table=table,
                chunk_page_table=h.integer(pages[:, : n // 32]),
                chunk_start=h.integer(torch.tensor([0], dtype=torch.int32)),
                cos=h.tt(c),
                sin=h.tt(s),
            )
            if n == 4096:
                kw["chunk_start_alignment"] = 0
            buckets[n] = (inp, kw)
            out = h.model.prefill_chunk_forward(inp, **kw)
            del out
        dec_in = h.tt(torch.zeros(1, 1, 1, 2560, dtype=torch.bfloat16))
        c, s = h.angles(0, 1)
        dec_kw = dict(
            kv_cache=h.cache,
            page_table=table,
            current_pos=h.integer(torch.tensor([0], dtype=torch.int32)),
            cos=h.tt(c),
            sin=h.tt(s),
        )
        out = h.model.decode_forward(dec_in, **dec_kw)
        del out
        gc.collect()
        ttnn.synchronize_device(h.mesh)
        tid = ttnn.begin_trace_capture(h.mesh, cq_id=0)
        try:
            with runtime_audit():
                dec_out = h.model.decode_forward(dec_in, **dec_kw)
        finally:
            ttnn.end_trace_capture(h.mesh, tid, cq_id=0)
        captures = 1
        replays = 0
        deterministic = True
        try:
            for length in [31, 32, 33, 127, 128, 129, 513, 17, 255, 256, 257, 8193]:
                h.ref.reset()
                x = h.inputs(length + 1)
                pages = torch.randperm(h.blocks, dtype=torch.int32)[None]
                h.copy(pages, table)
                expected = h.ref(x[:, :length])
                outputs = []
                chunk_size = 4096 if length == 8193 else 128
                for off in range(0, length, chunk_size):
                    count = min(chunk_size, length - off)
                    n = (count + 31) // 32 * 32
                    inp, kw = buckets[n]
                    value = torch.zeros(1, 1, n, 2560, dtype=torch.bfloat16)
                    value[0, 0, :count] = x[0, off : off + count]
                    c, s = h.angles(off, n)
                    h.copy(value, inp)
                    h.copy(c, kw["cos"])
                    h.copy(s, kw["sin"])
                    h.copy(pages[:, off // 32 : (off + n) // 32], kw["chunk_page_table"])
                    h.copy(torch.tensor([off], dtype=torch.int32), kw["chunk_start"])
                    with runtime_audit():
                        out = h.model.prefill_chunk_forward(inp, **kw)
                    outputs.append(ttnn.to_torch(out)[0, 0, :count])
                    del out
                h.check(
                    "prefill_with_live_decode_trace", expected, torch.cat(outputs), length=length, trace_id=int(tid)
                )
                h.copy(x[:, length:].reshape(1, 1, 1, 2560), dec_in)
                h.copy(torch.tensor([length], dtype=torch.int32), dec_kw["current_pos"])
                c, s = h.angles(length, 1)
                h.copy(c, dec_kw["cos"])
                h.copy(s, dec_kw["sin"])
                expected_decode = h.ref(x[:, length:], start=length)
                before_cache = [ttnn.to_torch(t) for t in h.cache] if length in (17, 513) else None
                repeats = []
                for _ in range(3):
                    ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
                    replays += 1
                    repeats.append(ttnn.to_torch(dec_out))
                deterministic &= all(torch.equal(repeats[0], r) for r in repeats[1:])
                if before_cache is not None:
                    physical = int(pages[0, length // 32])
                    untouched = torch.ones(h.blocks, 32, dtype=torch.bool)
                    untouched[physical, length % 32] = False
                    for before, tensor in zip(before_cache, h.cache):
                        after = ttnn.to_torch(tensor)
                        assert torch.equal(before.permute(0, 2, 1, 3)[untouched], after.permute(0, 2, 1, 3)[untouched])
                    cache_checks.append(dict(mode="decode", position=length, physical_page=physical, exact=True))
                h.check(
                    "changed_input_page_position_replay",
                    expected_decode,
                    repeats[-1],
                    position=length,
                    trace_id=int(tid),
                )
            assert deterministic
        finally:
            ttnn.release_trace(h.mesh, tid)
        result = dict(
            provenance=h.provenance,
            layer=layer,
            real_weights=not synthetic,
            cache_ownership_checks=cache_checks,
            rows=h.rows,
            runtime_audit="clean",
            deterministic=deterministic,
            trace=dict(
                captures=captures,
                replays=replays,
                request_releases=0,
                teardown_releases=1,
                allocation_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
                skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "1",
            ),
        )
        path = ROOT / "doc/optimized_decoder" / (output or f"coverage_{layer}.json")
        path.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        h.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(args.layer, args.synthetic, args.output)
