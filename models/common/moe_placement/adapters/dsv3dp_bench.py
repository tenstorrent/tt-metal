# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device driver for the deepseek_v3_d_p prefill family (run in its own process by the adapter).

Wraps one parametrization of a test in models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py
via a pytest plugin, overriding the weight cache and the PCC/perf knobs of ``run_chunked_transformer_updated``.
The token sequence is PREFILL_TRACE_DIR (metadata.json token_ids).

  capture   untraced, 1 iteration, PREFILL_DUMP_ROUTING_DIR on -> routing.npz
  perf      traced, N iterations, golden PCC on, exact digests of every KV cache slot (cache-digests.jsonl)
  nopcc     traced, N iterations, timing only
  profile   untraced, 2 iterations, device sync at TtMoe stage boundaries (profile_summary.json)

Every mode records the transport TtMoe actually selected (moe_transport) so the caller can check that the
calibration cost model matched the run.

Usage: python -m models.common.moe_placement.adapters.dsv3dp_bench --config run.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

# TtMoe stage intervals are named "<previous mark>-><mark>"; the routed-expert stage is the interval that ENDS at
# one of these marks (the interval ending at "pre_routed_expert" is TP gather + dispatch + shared expert).
ROUTED_EXPERT_MARKS = ("routed_expert", "routed_expert_combine")


def stage_shares(stage_ms: dict, iteration_ms: float) -> dict:
    """MoE and routed-expert shares of one profiled iteration from {"a->b": ms}."""
    moe_ms = sum(stage_ms.values())
    expert_ms = sum(v for k, v in stage_ms.items() if k.split("->")[1] in ROUTED_EXPERT_MARKS)
    return dict(
        stage_ms=dict(sorted(stage_ms.items())),
        iteration_ms=iteration_ms,
        moe_share=round(moe_ms / iteration_ms, 4),
        routed_expert_share=round(expert_ms / iteration_ms, 4),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    cfg = json.loads(p.parse_args().config.read_text())
    mode, out = cfg["mode"], Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(os.environ["TT_METAL_HOME"])
    sys.path.insert(0, str(repo))
    if mode == "capture":
        os.environ["PREFILL_DUMP_ROUTING_DIR"] = str(out / "routing")
    elif os.getenv("PREFILL_DUMP_ROUTING_DIR"):
        raise SystemExit("PREFILL_DUMP_ROUTING_DIR must not be set for timed runs")
    import pytest
    from loguru import logger

    layers, chunks, chunk = cfg["num_layers"], cfg["n_chunks"], cfg["chunk_tokens"]
    traced = mode in ("perf", "nopcc")
    iters = {"capture": 1, "profile": 2}.get(mode, cfg.get("num_iters", 10))
    trace = Path(os.environ["PREFILL_TRACE_DIR"])
    token_ids = json.loads((trace / "metadata.json").read_text())["token_ids"]
    if len(token_ids) < chunks * chunk:
        raise SystemExit(f"{trace}: {len(token_ids)} tokens, need {chunks * chunk}")
    records, iteration_totals, prof, transports = [], [], [], set()

    def sink(message):
        text = message.record["message"]
        m = re.fullmatch(r"\[chunk timing\] iter=(\d+) chunk=(\d+) ([\d.]+) ms", text)
        if m:
            row = dict(iteration=int(m[1]), chunk=int(m[2]), replay_ms=float(m[3]))
            records.append(row)
            with (out / "samples.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        m = re.fullmatch(r"iter (\d+) done \((\d+) chunks\) in ([\d.]+) seconds", text)
        if m:
            iteration_totals.append(dict(iteration=int(m[1]), chunks=int(m[2]), whole_seconds=float(m[3])))

    logger.add(sink, level="INFO")
    state = dict(
        mode=mode,
        model=cfg["model"],
        layers=layers,
        chunks=chunks,
        chunk_tokens=chunk,
        num_iters=iters,
        trace=traced,
        cache=cfg["cache"],
        trace_directory=str(trace),
        tokens_sha256=hashlib.sha256(
            json.dumps(token_ids[: chunks * chunk], separators=(",", ":")).encode()
        ).hexdigest(),
        host=os.uname().nodename,
        job=os.environ.get("SLURM_JOB_ID"),
        status="running",
        started=time.time(),
    )

    def save():
        (out / "measurement.json").write_text(json.dumps(state, indent=2) + "\n")

    def resolve(v, module):
        return getattr(module, v[1:]) if isinstance(v, str) and v.startswith("@") else v

    def install_transport_probe():
        from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe

        init = TtMoe.__init__

        def probed(self, *a, **k):
            init(self, *a, **k)
            transports.add(self.moe_transport)

        TtMoe.__init__ = probed

    def install_profile():
        """Sync-bracketed TtMoe stage timing via wrappers. Syncs sit only at module boundaries outside the
        shared-expert sub-device region. Syncs serialize the pipeline: read shares, not absolutes."""
        import ttnn
        from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe

        moe_fwd = TtMoe.forward

        class Proxy:
            def __init__(self, obj, moe, name):
                object.__setattr__(self, "_o", obj)
                object.__setattr__(self, "_m", moe)
                object.__setattr__(self, "_n", name)

            def __call__(self, *x, **k):
                self._m._mp_mark("pre_" + self._n)
                r = self._o(*x, **k)
                self._m._mp_mark(self._n)
                return r

            def __getattr__(self, n):
                return getattr(self._o, n)

        def moe_forward(self, *x, **k):
            dev = self.mesh_device
            if not getattr(self, "_mp_wrapped", False):
                self._mp_wrapped = True
                for n in ("gate", "routing_setup", "combine_module", "reduce_module"):
                    if getattr(self, n, None) is not None:
                        setattr(self, n, Proxy(getattr(self, n), self, n))
                if self.overlap_routed_expert_with_combine:
                    inner = self._routed_expert_and_combine

                    def expert_and_combine(*xx, **kk):
                        self._mp_mark("pre_routed_expert")
                        r = inner(*xx, **kk)
                        self._mp_mark("routed_expert_combine")
                        return r

                    self._routed_expert_and_combine = expert_and_combine
                else:
                    self.routed_expert = Proxy(self.routed_expert, self, "routed_expert")
            marks = []

            def mark(name):
                ttnn.synchronize_device(dev)
                marks.append((name, time.perf_counter()))

            self._mp_mark = mark
            mark("start")
            r = moe_fwd(self, *x, **k)
            mark("end")
            for (n0, t0), (n1, t1) in zip(marks, marks[1:]):
                prof.append((len(iteration_totals), f"{n0}->{n1}", (t1 - t0) * 1000.0))
            return r

        TtMoe.forward = moe_forward

    class Plugin:
        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, session, config, items):
            if len(items) != 1:
                raise pytest.UsageError(f"selector matched {len(items)} tests: {[x.nodeid for x in items]}")
            item = items[0]
            want = dict(num_layers=layers, n_chunks=chunks, num_iters=iters, use_trace=traced)
            for k, v in want.items():
                if k in item.callspec.params and item.callspec.params[k] != v:
                    raise pytest.UsageError(f"{item.nodeid}: {k}={item.callspec.params[k]}, wanted {v}")
            module = inspect.getmodule(item.obj)
            original = module.run_chunked_transformer_updated

            def run(*args, **kwargs):
                bound = inspect.signature(original).bind(*args, **kwargs)
                bound.apply_defaults()
                if bound.arguments["variant"].name != cfg["variant"]:
                    raise RuntimeError(
                        f"test runs variant {bound.arguments['variant'].name}, expected {cfg['variant']}"
                    )
                overrides = dict(
                    weight_cache_path=Path(cfg["cache"]),
                    baseline_chunk_times_s=None,
                    perf_margin=None,
                    check_pcc=mode == "perf",
                )
                for k, v in cfg.get("run_overrides", {}).items():
                    overrides[k] = resolve(v, module)
                if mode == "perf":
                    for k, v in cfg.get("perf_overrides", {}).items():
                        overrides[k] = resolve(v, module)
                bound.arguments.update({k: v for k, v in overrides.items() if k in bound.arguments})
                state.update(nodeid=item.nodeid, check_pcc=mode == "perf")
                save()
                install_transport_probe()
                if mode == "profile":
                    install_profile()
                if mode != "perf":
                    return original(*bound.args, **bound.kwargs)
                gather = module.gather_cache_natural

                def checked_gather(*gargs, **gkwargs):  # exact digests of every valid KV slot
                    cache, stripes = gather(*gargs, **gkwargs)
                    positions = module.blockcyclic_positions(stripes, module.CHUNK, cache.shape[1])
                    rows = []
                    for slot in range(cache.shape[0]):
                        natural = module.unrotate_cache_layer(
                            cache[slot], positions, chunks * module.CHUNK
                        ).contiguous()
                        rows.append(
                            dict(
                                slot=slot,
                                shape=list(natural.shape),
                                sha256=hashlib.sha256(natural.float().numpy().tobytes()).hexdigest(),
                            )
                        )
                    with (out / "cache-digests.jsonl").open("a") as f:
                        f.write(json.dumps(dict(shape=list(cache.shape), stripes=stripes, layers=rows)) + "\n")
                    return cache, stripes

                module.gather_cache_natural = checked_gather
                try:
                    return original(*bound.args, **bound.kwargs)
                finally:
                    module.gather_cache_natural = gather

            module.run_chunked_transformer_updated = run

        def pytest_runtest_logreport(self, report):
            if report.skipped:
                state["skipped"] = report.longreprtext

    iter_id = cfg["iter_ids"][str(iters)]
    selector = " and ".join(
        [
            cfg["mesh_id"],
            f"L{layers}",
            *cfg.get("extra_selector", []),
            cfg["chunks_id"],
            iter_id,
            "traced" if traced else "notrace",
        ]
    )
    state["selector"] = selector
    save()
    rc = pytest.main(
        ["-q", "-s", "--tb=short", f"{cfg['test_file']}::{cfg['test']}", "-k", selector], plugins=[Plugin()]
    )
    state.update(
        pytest_returncode=int(rc),
        finished=time.time(),
        iteration_totals=iteration_totals,
        sample_count=len(records),
        moe_transport=sorted(transports),
    )
    ok = rc == 0 and not state.get("skipped") and len(iteration_totals) == iters
    if ok and traced:
        ok = len(records) == iters * chunks
    if ok and mode == "capture":
        from models.common.moe_placement import routing

        state.update(
            routing.compact_torch_dumps(
                out / "routing", out / "routing.npz", cfg["num_experts"], cfg["top_k"], chunk, chunks
            )
        )
        if not cfg.get("keep_raw"):
            shutil.rmtree(out / "routing")
    if ok and mode == "profile":
        tot, calls = defaultdict(float), defaultdict(int)
        for it, stage, ms in prof:
            if it == iters - 1:  # the measured (non-compile) iteration
                tot[stage] += ms
                calls[stage] += 1
        state["profile"] = dict(stage_shares(tot, iteration_totals[-1]["whole_seconds"] * 1000.0), calls=dict(calls))
        with (out / "profile.jsonl").open("w") as f:
            for it, stage, ms in prof:
                f.write(json.dumps(dict(iteration=it, stage=stage, ms=round(ms, 4))) + "\n")
    state["status"] = "complete" if ok else "failed"
    save()
    raise SystemExit(0 if ok else (int(rc) or 1))


if __name__ == "__main__":
    main()
