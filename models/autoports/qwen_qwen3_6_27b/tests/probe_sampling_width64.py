# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused W=64 sampler contract; no model weights or serving stack required.

Run in an otherwise idle hardware session, from the tt-metal root:
  TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
    python -m models.autoports.qwen_qwen3_6_27b.tests.probe_sampling_width64 \
    --output /tmp/sampling-width64.json

The distribution sweep is eager (256 draws/user by default). Short traced sequences
cover every parameter setting without thousands of allocation-tracker GC scans.
--legacy runs only the old default-width k=40 diagnostic, even on an old binding
without max_top_k. It reports whether the defect was reproduced, without requiring
the new API or treating the expected invalid samples as an exception.
"""

import argparse
import json
import math
import os
from pathlib import Path

# Tracking configuration must precede importing TTNN.
os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")

import torch

import ttnn

USERS, WIDTH = 32, 64
KS = (1, 16, 31, 32, 33, 40, 63, 64)
PS = (0.0, 0.5, 0.95, 1.0)
TEMPS = (0.5, 1.0, 2.0)  # reciprocal temperatures, as accepted by ttnn.sampling


def inputs():
    rng = torch.Generator().manual_seed(91827)
    # Exactly representable BF16 strict logits, with appreciable mass at rank64.
    ranked = -torch.arange(WIDTH, dtype=torch.float32) / 128
    values, ids = [], []
    for row in range(USERS):
        order = torch.randperm(WIDTH, generator=rng)
        values.append(ranked[order])
        ids.append((100003 + row * 1009 + 7 * torch.randperm(WIDTH, generator=rng)).int())
    return torch.stack(values).view(1, 1, USERS, WIDTH), torch.stack(ids).view(1, 1, USERS, WIDTH)


def reference(k, p, temp):
    logits = (-torch.arange(k, dtype=torch.float32) / 128).to(torch.bfloat16).float() * temp
    probs = logits.softmax(0).to(torch.bfloat16).float()
    # Mirror the writer's descending cumulative > p, including p=0 => greedy.
    p = float(torch.tensor(p, dtype=torch.bfloat16))
    crossings = (probs.cumsum(0) > p).nonzero().flatten()
    kept = int(crossings[0]) + 1 if crossings.numel() else k
    result = torch.zeros(WIDTH)
    result[:kept] = probs[:kept] / probs[:kept].sum()
    return result, kept


def rank_samples(samples, values, ids):
    ranked_ids = ids.reshape(USERS, WIDTH).gather(1, values.reshape(USERS, WIDTH).argsort(descending=True))
    matches = samples[..., None] == ranked_ids[None, :, :]
    valid = matches.any(-1)
    return matches.int().argmax(-1), valid


def statistics(samples, values, ids, kvals, p, temp):
    ranks, valid = rank_samples(samples, values, ids)
    assert valid.all(), f"{int((~valid).sum())} off-candidate token IDs (p={p}, temp={temp})"
    assert (ranks < kvals[None, :]).all(), "Sampled a rank beyond the requested top-k"
    summary = []
    for k in sorted(set(kvals.tolist())):
        selected = ranks[:, kvals == k].flatten()
        count = torch.bincount(selected, minlength=WIDTH)
        n = selected.numel()
        expected, kept = reference(k, p, temp)
        empirical = count.float() / n
        # BF16 exp/reduction can move a nucleus boundary by one token. Only
        # p=.5/.95 receives this allowance; top-k and greedy stay strict.
        edge = float(expected[kept - 1]) if 0 < p < 1 else 0.0
        allowed = min(k, kept + (1 if edge else 0))
        assert not count[allowed:].any(), f"k={k}, p={p}: samples outside nucleus + one edge token"
        tv = float((empirical - expected).abs().sum() / 2)
        # E[TV] <= .5 sum_i sqrt(q_i(1-q_i)/n), plus a bounded-difference
        # concentration term (failure probability <= 1e-5 per histogram).
        # 0.02 covers approximate exp/BF16 error; one edge mass is explicit.
        limit = float(0.5 * (expected * (1 - expected) / n).sqrt().sum())
        limit += math.sqrt(math.log(100000) / (2 * n)) + 0.02 + edge
        assert tv <= limit, f"k={k}, p={p}, temp={temp}: TV={tv:.4f} > {limit:.4f}"
        sigma = (expected * (1 - expected) / n).sqrt()
        error = (empirical - expected).abs()
        per_rank_limit = 6 * sigma + 0.005
        if edge:
            # Only the two potentially disputed boundary bins get extra slack.
            per_rank_limit[max(0, kept - 1) : min(WIDTH, kept + 1)] += edge
        assert (error <= per_rank_limit).all(), f"k={k}, p={p}, temp={temp}: rank histogram mismatch"
        if p == 1.0 and k > 32:
            required = list(range(32, min(40, k))) + ([63] if k == 64 else [])
            assert (
                count[required] > 0
            ).all(), f"k={k}: missing boundary ranks {[i + 1 for i in required if count[i] == 0]}"
        summary.append(
            dict(k=k, samples=n, kept=kept, tv=round(tv, 5), tv_limit=round(limit, 5), rank_counts=count[:k].tolist())
        )
    return summary


class Probe:
    def __init__(self, device):
        self.device = device
        self.values, self.ids = inputs()
        self.k = torch.tensor(KS * (USERS // len(KS)), dtype=torch.int32)
        self.seeds = torch.arange(1, USERS + 1, dtype=torch.int32) * 104729
        self.perm = torch.arange(USERS - 1, -1, -1)
        self.host = {}  # Own immutable torch + TT host copies until all async writes finish.
        self.device_tensors = {}
        specs = {
            "values": (self.values, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "ids": (self.ids, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            "k": (self.k, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            "p": (torch.ones(USERS), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            "temp": (torch.ones(USERS), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            "seeds": (self.seeds, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "users": (torch.arange(USERS).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "out": (torch.zeros(1, 1, 1, USERS).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        }
        self.specs = specs
        for name, (value, dtype, layout) in specs.items():
            self.device_tensors[name] = ttnn.from_torch(value.clone(), dtype=dtype, layout=layout, device=device)
            self.prepare(name, "base", value)
        for p in PS:
            self.prepare("p", p, torch.full((USERS,), p))
        for temp in TEMPS:
            self.prepare("temp", temp, torch.full((USERS,), temp))
        self.prepare("k", "default", self.k.clamp(max=32))
        self.prepare("k", "legacy", torch.full((USERS,), 40, dtype=torch.int32))
        self.prepare("seeds", "changed", self.seeds + 7919)
        for name, tensor in (
            ("values", self.values[:, :, self.perm]),
            ("ids", self.ids[:, :, self.perm]),
            ("k", self.k[self.perm]),
            ("seeds", self.seeds[self.perm]),
        ):
            self.prepare(name, "remap", tensor)
        self.prepare("values", "sequential", (-torch.arange(WIDTH).float() / 128).expand(1, 1, USERS, WIDTH))
        self.prepare("ids", "sequential", torch.arange(WIDTH).int().expand(1, 1, USERS, WIDTH))
        self.traces = {}
        self.output_address = self.device_tensors["out"].buffer_address()

    def prepare(self, name, key, value):
        dtype, layout = self.specs[name][1:]
        owned = value.contiguous().clone()
        self.host[name, key] = (owned, ttnn.from_torch(owned, dtype=dtype, layout=layout))

    def upload(self, name, key="base"):
        ttnn.copy_host_to_device_tensor(self.host[name, key][1], self.device_tensors[name])

    def configure(self, p, temp, mode="base", default=False):
        for name in ("values", "ids"):
            self.upload(name, mode)
        self.upload("k", "default" if default else ("remap" if mode == "remap" else "base"))
        self.upload("p", p)
        self.upload("temp", temp)

    def reset(self, key="base"):
        self.upload("seeds", key)
        ttnn.manual_seed(seeds=self.device_tensors["seeds"], user_ids=self.device_tensors["users"])

    def sample(self, width):
        t = self.device_tensors
        kwargs = {} if width == 32 else {"max_top_k": 64}
        result = ttnn.sampling(t["values"], t["ids"], t["k"], t["p"], t["temp"], output_tensor=t["out"], **kwargs)
        assert result.buffer_address() == self.output_address, "Sampling replaced the persistent token-out buffer"

    def draw(self, width, n, traced=False):
        outputs = []
        for _ in range(n):
            if traced:
                ttnn.execute_trace(self.device, self.traces[width], cq_id=0, blocking=True)
            else:
                self.sample(width)
            outputs.append(ttnn.to_torch(self.device_tensors["out"]).reshape(USERS).long().clone())
        return torch.stack(outputs)

    def warm(self):
        # Exercise both compile variants, all runtime params, both index forms,
        # and manual_seed BEFORE either trace is captured.
        for width in (32, 64):
            for p in PS:
                for temp in TEMPS:
                    self.configure(p, temp, default=width == 32)
                    self.reset()
                    self.sample(width)
            self.configure(1.0, 1.0, mode="sequential", default=width == 32)
            self.reset()
            self.sample(width)
        ttnn.synchronize_device(self.device)

    def capture(self):
        for width in (32, 64):
            self.configure(1.0, 1.0, default=width == 32)
            self.reset()
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            self.sample(width)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            self.traces[width] = tid

    def close(self):
        for tid in self.traces.values():
            ttnn.release_trace(self.device, tid)


def run(probe, args, result):
    if args.legacy:
        probe.configure(1.0, 1.0)
        probe.upload("k", "legacy")
        probe.reset()
        samples = probe.draw(32, args.draws)
        ranks, valid = rank_samples(samples, probe.values, probe.ids)
        tail = torch.bincount(ranks[valid], minlength=WIDTH)[32:40]
        reproduced = bool((~valid).any() or (tail == 0).any())
        result.update(
            passed=reproduced,
            legacy_defect_reproduced=reproduced,
            off_candidate_count=int((~valid).sum()),
            ranks33_through40_counts=tail.tolist(),
        )
        return

    probe.warm()
    # All large empirical sweeps precede capture; no late program compilation.
    result["distributions"] = []
    for width in (64, 32):
        for p in PS:
            for temp in TEMPS:
                probe.configure(p, temp, default=width == 32)
                probe.reset()
                samples = probe.draw(width, args.draws)
                kvals = probe.k if width == 64 else probe.k.clamp(max=32)
                stats = statistics(samples, probe.values, probe.ids, kvals, p, temp)
                result["distributions"].append(dict(max_top_k=width, p=p, reciprocal_temperature=temp, rows=stats))
        print(json.dumps(dict(progress="distribution_complete", max_top_k=width)), flush=True)

    probe.capture()
    result["trace_checks"] = []
    for p in PS:
        for temp in TEMPS:
            # Alternate both traces, whose persistent input/output buffers are shared.
            for width in (64, 32):
                probe.configure(p, temp, default=width == 32)
                probe.reset()
                eager = probe.draw(width, args.trace_draws)
                probe.reset()
                replay = probe.draw(width, args.trace_draws, traced=True)
                probe.reset()
                repeat = probe.draw(width, args.trace_draws, traced=True)
                assert torch.equal(eager, replay), "Eager/trace sequence changed"
                assert torch.equal(replay, repeat), "Repeated seed changed the trace sequence"
                probe.reset("changed")
                changed = probe.draw(width, args.trace_draws, traced=True)
                if p > 0:
                    assert not torch.equal(replay, changed), "Changed seeds did not change stochastic output"
                result["trace_checks"].append(
                    dict(
                        max_top_k=width,
                        p=p,
                        reciprocal_temperature=temp,
                        eager_equal=True,
                        repeated_seed_equal=True,
                        changed_seed_equal=bool(torch.equal(replay, changed)),
                    )
                )

    probe.configure(1.0, 1.0)
    probe.reset()
    canonical = probe.draw(64, args.trace_draws, traced=True)
    probe.configure(1.0, 1.0, mode="remap")
    probe.reset("remap")
    remapped = probe.draw(64, args.trace_draws, traced=True)
    assert torch.equal(remapped[:, probe.perm.argsort()], canonical), "Request identity changed after row remap"
    for width in (64, 32):
        probe.configure(1.0, 1.0, mode="sequential", default=width == 32)
        probe.reset()
        seq = probe.draw(width, args.trace_draws, traced=True)
        kvals = probe.k if width == 64 else probe.k.clamp(max=32)
        assert ((seq >= 0) & (seq < kvals[None, :])).all(), "Sequential-ID control failed"
        probe.configure(0.0, 1.0, mode="sequential", default=width == 32)
        probe.reset()
        assert not probe.draw(width, 1, traced=True).any(), "Known greedy sequential-ID answer is zero"

    # Tracker positive control: exact unsafe allocation, then retire and replay.
    unsafe = ttnn.empty_like(probe.device_tensors["out"])
    caught = False
    try:
        ttnn.execute_trace(probe.device, probe.traces[64], cq_id=0, blocking=True)
    except RuntimeError as error:
        if "corrupted on replay" not in str(error).lower():
            raise
        caught = True
    finally:
        del unsafe
    assert caught, "Trace allocation tracking positive control did not fire"
    probe.configure(1.0, 1.0)
    probe.reset()
    final = probe.draw(64, args.trace_draws, traced=True)
    assert torch.equal(final, canonical), "Final replay changed after interleaved variants and tracker control"
    result.update(
        passed=True,
        row_remap_identity=True,
        sequential_ids=True,
        persistent_output=True,
        allocation_positive_control=True,
        trace_ids={str(k): int(v) for k, v in probe.traces.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--trace-draws", type=int, default=4)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    if args.draws < 256 or args.trace_draws < 2:
        parser.error("Require --draws >= 256 and --trace-draws >= 2 for statistical/sequence checks")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    torch.set_num_threads(4)
    result = dict(
        passed=False,
        legacy=args.legacy,
        draws_per_user=args.draws,
        trace_draws=args.trace_draws,
        users=USERS,
        width=WIDTH,
        trace_allocation_tracking=True,
        program_cache_tracking=True,
        distribution_path="eager",
        trace_path="sampling into persistent RM token-out",
        p_zero_semantics="greedy, matching existing cumulative > p writer",
    )
    device = None
    probe = None
    try:
        device = ttnn.open_device(device_id=args.device_id, trace_region_size=8 * 1024 * 1024)
        device.enable_program_cache()
        probe = Probe(device)
        run(probe, args, result)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("distributions", "trace_checks")}), flush=True)
        if probe is not None:
            probe.close()
        if device is not None:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
