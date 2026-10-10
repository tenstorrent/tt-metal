# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0a: host reference runtime at real Kimi-K3 dims (no device).

Decides whether the single-layer suite's reference fits a CI budget. Times every component
reference cold at each token count, full-row and row-sampled, plus the module reference the suite
would otherwise use for MLA (``cpu_mla_reference``, bf16 on CPU). Run it on the CI runner hosts,
not only a dev box:

    python models/demos/deepseek_v3_d_p/tests/kimi_k3/single_layer/bench_reference.py \
        --tokens 1280 2560 5120 10240 --rows 512 --json /tmp/k3_ref_bench.json

T is the request length a scenario's reference covers (LB: 1280 per chunk, GH: 5120 per chunk).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import tempfile
import time
from collections.abc import Callable

import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.reference_ops import (
    MlaDims,
    PooledExperts,
    dense_mlp,
    kda_forward,
    mla_forward,
    moe_forward,
)

H = KimiK3Config.EMB_SIZE


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _sample_rows(tokens: int, rows: int, generator: torch.Generator) -> torch.Tensor:
    """First and last rows plus a uniform random spread. The suite stratifies (per SP shard, chunk
    boundaries, heavy-tailed rows), but the cost depends only on the count, which this matches."""
    if rows >= tokens:
        return torch.arange(tokens)
    if rows == 1:
        return torch.tensor([tokens - 1])
    interior = 1 + torch.randperm(tokens - 2, generator=generator)[: rows - 2]
    return torch.cat([torch.tensor([0, tokens - 1]), interior]).sort().values


def _bf16_normal(generator, *shape, std):
    return (torch.randn(*shape, generator=generator) * std).bfloat16()


def _kda_weights(generator):
    from models.demos.deepseek_v3_d_p.tests.kda.utils import random_weights

    return random_weights(kimi_k3_kda_config())


def _mla_weights(generator):
    cfg = kimi_k3_hf_config()
    heads = cfg.num_attention_heads
    std = 0.02
    return {
        "q_a_proj.weight": _bf16_normal(generator, cfg.q_lora_rank, H, std=std),
        "q_a_layernorm.weight": torch.ones(cfg.q_lora_rank).bfloat16(),
        "q_b_proj.weight": _bf16_normal(
            generator, heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), cfg.q_lora_rank, std=std
        ),
        "kv_a_proj_with_mqa.weight": _bf16_normal(generator, cfg.kv_lora_rank + cfg.qk_rope_head_dim, H, std=std),
        "kv_a_layernorm.weight": torch.ones(cfg.kv_lora_rank).bfloat16(),
        "kv_b_proj.weight": _bf16_normal(
            generator, heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank, std=std
        ),
        "o_proj.weight": _bf16_normal(generator, H, heads * cfg.v_head_dim, std=std),
        "g_proj.weight": _bf16_normal(generator, heads * cfg.v_head_dim, H, std=std),
    }


def _moe_weights(generator, pools: int):
    latent, inter, experts = KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE, KimiK3Config.MOE_INTERMEDIATE_SIZE, 896
    shared = KimiK3Config.SHARED_EXPERT_INTERMEDIATE_SIZE
    w = {
        "gate_weights": {
            "weight": _bf16_normal(generator, experts, H, std=0.02),
            "e_score_correction_bias": _bf16_normal(generator, experts, std=0.01),
        },
        "latent_weights": {
            "down_proj": _bf16_normal(generator, latent, H, std=0.02),
            "up_proj": _bf16_normal(generator, H, latent, std=0.02),
            "norm": torch.ones(latent).bfloat16(),
        },
        "shared_expert_weights": {
            "gate_proj": _bf16_normal(generator, shared, H, std=0.02),
            "up_proj": _bf16_normal(generator, shared, H, std=0.02),
            "down_proj": _bf16_normal(generator, H, shared, std=0.02),
        },
    }
    pooled = PooledExperts(
        gate=_bf16_normal(generator, pools, inter, latent, std=0.02),
        up=_bf16_normal(generator, pools, inter, latent, std=0.02),
        down=_bf16_normal(generator, pools, latent, inter, std=0.02),
        pool_of=torch.arange(experts) % pools,
        sign=torch.where(torch.rand(experts, inter, generator=generator) < 0.5, -1.0, 1.0),
    )
    return w, pooled


def _dense_weights(generator):
    inter = KimiK3Config.INTERMEDIATE_SIZE
    return {
        "gate_proj": _bf16_normal(generator, inter, H, std=0.02),
        "up_proj": _bf16_normal(generator, inter, H, std=0.02),
        "down_proj": _bf16_normal(generator, H, inter, std=0.02),
    }


def _fp32(tree):
    """Upcast once, outside the timed region. MLA/MoE/dense weights are drawn in bf16 (the device
    format); the KDA ones come from `random_weights` (fp32, its own fixed seed)."""
    if isinstance(tree, dict):
        return {k: _fp32(v) for k, v in tree.items()}
    if isinstance(tree, PooledExperts):
        return PooledExperts(tree.gate.float(), tree.up.float(), tree.down.float(), tree.pool_of, tree.sign)
    return tree.float() if tree.is_floating_point() else tree


def _time(fn: Callable[[], object], repeats: int) -> tuple[float, float]:
    """``(first, best)`` wall seconds: the first call is the one CI pays; best is the fastest repeat
    (warmer allocator/threads -- though the bf16 module reference is cache-cold every time)."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.no_grad():
            fn()
        times.append(time.perf_counter() - start)
    return times[0], min(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, nargs="+", default=[640, 1280, 2560, 5120, 10240])
    parser.add_argument("--rows", type=int, default=512, help="sampled output rows for the sampled variants")
    parser.add_argument("--pools", type=int, default=16, help="routed-expert weight pools (896 experts share them)")
    parser.add_argument("--threads", type=int, default=None, help="torch threads (default: physical cores)")
    parser.add_argument(
        "--components",
        nargs="+",
        default=["kda", "mla", "mla_module_bf16", "moe", "dense"],
        choices=["kda", "mla", "mla_module_bf16", "moe", "dense"],
    )
    parser.add_argument("--module-bf16-max-tokens", type=int, default=10240)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    threads = args.threads or max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(threads)
    generator = torch.Generator().manual_seed(2026)
    host = {
        "node": platform.node(),
        "cpu": next(
            (l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "unknown"
        ),
        "logical_cpus": os.cpu_count(),
        "torch_threads": threads,
        "torch": torch.__version__,
    }
    print(f"host: {host}")

    results = []

    def record(component, variant, tokens, rows, timing):
        first, best = timing
        entry = {
            "component": component,
            "variant": variant,
            "tokens": tokens,
            "rows": rows,
            "first_seconds": round(first, 3),
            "best_seconds": round(best, 3),
            # A process-lifetime high-water mark, so it bounds memory for everything run so far
            # rather than measuring this record; run one --components at a time to isolate it.
            "process_peak_rss_gb": round(_peak_rss_gb(), 2),
        }
        results.append(entry)
        print(f"{component:16s} {variant:10s} T={tokens:6d} rows={rows:6d}  first {first:8.2f} s  best {best:8.2f} s")

    w_bf16 = pooled = None
    kda_config = kimi_k3_kda_config()
    mla_dims = MlaDims.from_hf(kimi_k3_hf_config())

    for component in args.components:
        if component == "kda":
            w = _fp32(_kda_weights(generator))
        elif component in ("mla", "mla_module_bf16"):
            w_bf16 = _mla_weights(generator)
            w = _fp32(w_bf16)
        elif component == "moe":
            w, pooled = (_fp32(t) for t in _moe_weights(generator, args.pools))
        elif component == "dense":
            w = _fp32(_dense_weights(generator))
        for tokens in args.tokens:
            x = torch.randn(tokens, H, generator=generator)
            rows = _sample_rows(tokens, args.rows, generator)
            if component == "kda":
                record("kda", "full", tokens, tokens, _time(lambda: kda_forward(x, w, kda_config), args.repeats))
                record(
                    "kda",
                    "sampled",
                    tokens,
                    len(rows),
                    _time(lambda: kda_forward(x, w, kda_config, rows=rows), args.repeats),
                )
            elif component == "mla":
                record("mla", "full", tokens, tokens, _time(lambda: mla_forward(x, w, mla_dims), args.repeats))
                record(
                    "mla",
                    "sampled",
                    tokens,
                    len(rows),
                    _time(lambda: mla_forward(x, w, mla_dims, rows=rows), args.repeats),
                )
            elif component == "mla_module_bf16":
                if tokens > args.module_bf16_max_tokens:
                    continue
                from models.demos.deepseek_v3_d_p.utils import chunked_prefill_utils

                # Cold: a fresh empty cache dir, and CI=true so nothing is written either. Both
                # variables are restored after, so later components see the caller's environment.
                hf_config = kimi_k3_hf_config(max_seq=max(8192, tokens))
                saved = {k: os.environ.get(k) for k in (chunked_prefill_utils._REF_CACHE_ENV, "CI")}
                with tempfile.TemporaryDirectory() as cache_dir:
                    os.environ[chunked_prefill_utils._REF_CACHE_ENV] = cache_dir
                    os.environ["CI"] = "true"
                    try:
                        record(
                            "mla_module",
                            "bf16",
                            tokens,
                            tokens,
                            _time(
                                lambda: chunked_prefill_utils.cpu_mla_reference(hf_config, w_bf16, x.bfloat16()),
                                args.repeats,
                            ),
                        )
                    finally:
                        for key, value in saved.items():
                            if value is None:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = value
            elif component == "moe":
                record("moe", "full", tokens, tokens, _time(lambda: moe_forward(x, w, pooled), args.repeats))
                record(
                    "moe",
                    "sampled",
                    tokens,
                    len(rows),
                    _time(lambda: moe_forward(x, w, pooled, rows=rows), args.repeats),
                )
            elif component == "dense":
                record("dense", "full", tokens, tokens, _time(lambda: dense_mlp(x, w), args.repeats))
                record("dense", "sampled", tokens, len(rows), _time(lambda: dense_mlp(x, w, rows=rows), args.repeats))
        w = w_bf16 = pooled = None  # drop this component's weights before the next one's

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"host": host, "results": results}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
