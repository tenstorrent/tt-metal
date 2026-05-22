# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Section-level timing benchmark for TTML DeepSeek MLA.

This script mirrors ``MultiHeadLatentAttention.forward`` and synchronizes after
each stage so we can see where time is spent before choosing a fusion target.

Example:
    python tt-train/tests/benchmark/mla_profile_benchmark.py --cases smoke,nano,tiny-short
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ttnn
import ttml
from ttml.models.deepseek import DeepSeekConfig
from ttml.models.deepseek.autograd_ops import autograd_concat, autograd_slice, split_heads
from ttml.models.deepseek.mla import MultiHeadLatentAttention


@dataclass(frozen=True)
class MLACase:
    name: str
    batch_size: int
    seq_len: int
    dim: int
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int


@dataclass(frozen=True)
class StageEstimate:
    read_bytes: int = 0
    write_bytes: int = 0
    flops: int = 0

    @property
    def traffic_bytes(self) -> int:
        return self.read_bytes + self.write_bytes


CASES: dict[str, MLACase] = {
    # Existing test_deepseek.py parity shape. Fast smoke for validating the script.
    "smoke": MLACase("smoke", 2, 64, 64, 2, 32, 32, 32, 32, 32),
    # tt-train/configs/model_configs/moe/nano_deepseek_char.yaml
    "nano-s128": MLACase("nano-s128", 1, 128, 512, 8, 256, 128, 64, 32, 64),
    "nano": MLACase("nano", 1, 256, 512, 8, 256, 128, 64, 32, 64),
    "nano-b2": MLACase("nano-b2", 2, 256, 512, 8, 256, 128, 64, 32, 64),
    # tt-train/configs/model_configs/moe/tiny_deepseek_char.yaml dims, shorter S.
    "tiny-short": MLACase("tiny-short", 1, 512, 1536, 12, 512, 256, 96, 32, 128),
    "tiny-s1024": MLACase("tiny-s1024", 1, 1024, 1536, 12, 512, 256, 96, 32, 128),
    # Same dims as tiny_deepseek_char.yaml. Opt in because full SDPA can be slow.
    "tiny-full": MLACase("tiny-full", 1, 2048, 1536, 12, 512, 256, 96, 32, 128),
}


STAGE_ORDER = [
    "q_projection",
    "q_split",
    "q_rope",
    "kv_down_split",
    "k_rope_broadcast",
    "kv_up_split",
    "qk_assemble",
    "sdpa",
    "heads_fusion",
    "output_projection",
]


def _make_config(case: MLACase) -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=64,
        dim=case.dim,
        inter_dim=case.dim * 3,
        moe_inter_dim=256,
        n_layers=1,
        n_dense_layers=1,
        n_heads=case.n_heads,
        q_lora_rank=case.q_lora_rank,
        kv_lora_rank=case.kv_lora_rank,
        qk_nope_head_dim=case.qk_nope_head_dim,
        qk_rope_head_dim=case.qk_rope_head_dim,
        v_head_dim=case.v_head_dim,
        max_seq_len=case.seq_len,
        rope_theta=10000.0,
    )


def _bf16_bytes(elements: int) -> int:
    return elements * 2


def _matmul_flops(m: int, k: int, n: int, batch: int = 1) -> int:
    return 2 * batch * m * k * n


def _estimate_stages(case: MLACase) -> dict[str, StageEstimate]:
    B = case.batch_size
    S = case.seq_len
    D = case.dim
    H = case.n_heads
    q_rank = case.q_lora_rank
    kv_rank = case.kv_lora_rank
    qk_nope = case.qk_nope_head_dim
    qk_rope = case.qk_rope_head_dim
    qk_head = qk_nope + qk_rope
    v_dim = case.v_head_dim

    estimates: dict[str, StageEstimate] = {}

    # Matmul traffic is a lower-bound tensor-size estimate; actual kernel traffic
    # depends on tiling/reuse. FLOPs use dense matmul convention: 2*M*K*N.
    if q_rank == 0:
        q_proj_flops = _matmul_flops(B * S, D, H * qk_head)
        q_proj_read = _bf16_bytes(B * S * D + D * H * qk_head)
    else:
        q_proj_flops = _matmul_flops(B * S, D, q_rank) + _matmul_flops(B * S, q_rank, H * qk_head)
        q_proj_read = _bf16_bytes(B * S * D + D * q_rank + B * S * q_rank + q_rank * H * qk_head)
    estimates["q_projection"] = StageEstimate(q_proj_read, _bf16_bytes(B * S * H * qk_head), q_proj_flops)

    estimates["q_split"] = StageEstimate(
        read_bytes=_bf16_bytes(B * S * H * qk_head),
        write_bytes=_bf16_bytes(B * S * H * qk_head),
    )
    estimates["q_rope"] = StageEstimate(
        read_bytes=_bf16_bytes(B * S * H * qk_head),
        write_bytes=_bf16_bytes(B * S * H * qk_head),
        flops=B * S * H * qk_rope * 6,  # rough: sin/cos rotation as mul/sub/adds, caches excluded
    )

    kv_down_flops = _matmul_flops(B * S, D, kv_rank + qk_rope)
    estimates["kv_down_split"] = StageEstimate(
        read_bytes=_bf16_bytes(B * S * D + D * (kv_rank + qk_rope)),
        write_bytes=_bf16_bytes(B * S * (kv_rank + qk_rope)),
        flops=kv_down_flops,
    )
    estimates["k_rope_broadcast"] = StageEstimate(
        read_bytes=_bf16_bytes(B * S * qk_rope),
        write_bytes=_bf16_bytes(B * S * H * qk_rope),
        flops=B * S * qk_rope * 6,
    )

    kv_up_flops = _matmul_flops(B * S, kv_rank, H * (qk_nope + v_dim))
    estimates["kv_up_split"] = StageEstimate(
        read_bytes=_bf16_bytes(B * S * kv_rank + kv_rank * H * (qk_nope + v_dim)),
        write_bytes=_bf16_bytes(B * S * H * (qk_nope + v_dim)),
        flops=kv_up_flops,
    )
    estimates["qk_assemble"] = StageEstimate(
        read_bytes=_bf16_bytes(B * S * H * (qk_nope + qk_rope + qk_nope + qk_rope)),
        write_bytes=_bf16_bytes(B * S * H * (qk_head + qk_head)),
    )

    # SDPA lower-bound FLOPs:
    # QK^T and P@V are dense batched matmuls. Softmax FLOPs are approximate.
    sdpa_flops = _matmul_flops(S, qk_head, S, batch=B * H) + _matmul_flops(S, S, v_dim, batch=B * H)
    sdpa_flops += B * H * S * S * 5
    estimates["sdpa"] = StageEstimate(
        read_bytes=_bf16_bytes(B * H * S * (qk_head + qk_head + v_dim)),
        write_bytes=_bf16_bytes(B * H * S * v_dim),
        flops=sdpa_flops,
    )
    estimates["heads_fusion"] = StageEstimate(
        read_bytes=_bf16_bytes(B * H * S * v_dim),
        write_bytes=_bf16_bytes(B * S * H * v_dim),
    )
    estimates["output_projection"] = StageEstimate(
        read_bytes=_bf16_bytes(B * S * H * v_dim + H * v_dim * D),
        write_bytes=_bf16_bytes(B * S * D),
        flops=_matmul_flops(B * S, H * v_dim, D),
    )
    return estimates


def _make_inputs(case: MLACase, rng: np.random.Generator):
    x_np = rng.standard_normal((case.batch_size, 1, case.seq_len, case.dim), dtype=np.float32)
    x = ttml.autograd.Tensor.from_numpy(x_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)

    mask_np = np.tril(np.ones((case.seq_len, case.seq_len), dtype=np.float32)).reshape(1, 1, case.seq_len, case.seq_len)
    mask = ttml.autograd.Tensor.from_numpy(mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    return x, mask


def _sync(device) -> None:
    ttnn.synchronize_device(device)


def _time_stage(device, timings_us: dict[str, list[float]], name: str, fn):
    _sync(device)
    start = time.perf_counter()
    result = fn()
    _sync(device)
    timings_us[name].append((time.perf_counter() - start) * 1_000_000.0)
    return result


def _run_profiled_forward(module: MultiHeadLatentAttention, x, mask, device, timings_us: dict[str, list[float]]):
    B, _, S, _ = list(x.get_value().shape)
    n_heads = module.n_heads
    qk_nope = module.qk_nope_head_dim
    qk_head = module.qk_head_dim
    kv_lora = module.kv_lora_rank

    def q_projection():
        if module.q_lora_rank == 0:
            return module.wq(x)
        return module.wq_b(module.q_norm(module.wq_a(x)))

    q = _time_stage(device, timings_us, "q_projection", q_projection)
    q = _time_stage(device, timings_us, "q_split", lambda: split_heads(q, n_heads))

    def q_rope():
        q_nope = autograd_slice(q, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
        q_pe = autograd_slice(q, [0, 0, 0, qk_nope], [B, n_heads, S, qk_head])
        q_pe = ttml.ops.rope.rope(q_pe, module.rope_params)
        return q_nope, q_pe

    q_nope, q_pe = _time_stage(device, timings_us, "q_rope", q_rope)

    def kv_down_split():
        kv_full = module.wkv_a(x)
        kv = autograd_slice(kv_full, [0, 0, 0, 0], [B, 1, S, kv_lora])
        k_pe = autograd_slice(kv_full, [0, 0, 0, kv_lora], [B, 1, S, kv_lora + module.qk_rope_head_dim])
        return kv, k_pe

    kv, k_pe = _time_stage(device, timings_us, "kv_down_split", kv_down_split)

    def k_rope_broadcast():
        k_pe_rot = ttml.ops.rope.rope(k_pe, module.rope_params)
        return autograd_concat([k_pe_rot] * n_heads, dim=1)

    k_pe = _time_stage(device, timings_us, "k_rope_broadcast", k_rope_broadcast)

    def kv_up_split():
        kv_up = module.wkv_b(module.kv_norm(kv))
        kv_up = split_heads(kv_up, n_heads)
        k_nope = autograd_slice(kv_up, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
        v = autograd_slice(kv_up, [0, 0, 0, qk_nope], [B, n_heads, S, qk_nope + module.v_head_dim])
        return k_nope, v

    k_nope, v = _time_stage(device, timings_us, "kv_up_split", kv_up_split)

    def qk_assemble():
        q_full = autograd_concat([q_nope, q_pe], dim=3)
        k_full = autograd_concat([k_nope, k_pe], dim=3)
        return q_full, k_full

    q_full, k_full = _time_stage(device, timings_us, "qk_assemble", qk_assemble)

    attn = _time_stage(
        device,
        timings_us,
        "sdpa",
        lambda: ttml.ops.attention.scaled_dot_product_attention_composite(q_full, k_full, v, mask),
    )
    attn = _time_stage(device, timings_us, "heads_fusion", lambda: ttml.ops.multi_head_utils.heads_fusion(attn))
    return _time_stage(device, timings_us, "output_projection", lambda: module.wo(attn))


def _run_plain_forward(module: MultiHeadLatentAttention, x, mask, device) -> float:
    _sync(device)
    start = time.perf_counter()
    module(x, mask)
    _sync(device)
    return (time.perf_counter() - start) * 1_000_000.0


def _stats(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        "avg_us": statistics.fmean(values),
        "min_us": min(values),
        "max_us": max(values),
        "p50_us": sorted_values[len(sorted_values) // 2],
    }


def _print_case(
    case: MLACase,
    timings_us: dict[str, list[float]],
    end_to_end_us: list[float] | None = None,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    total_avg = sum(_stats(timings_us[name])["avg_us"] for name in STAGE_ORDER)
    estimates = _estimate_stages(case)

    print(f"\nMLA case: {case.name}")
    print(
        "shape: "
        f"B={case.batch_size} S={case.seq_len} dim={case.dim} H={case.n_heads} "
        f"q_lora={case.q_lora_rank} kv_lora={case.kv_lora_rank} "
        f"qk=({case.qk_nope_head_dim}+{case.qk_rope_head_dim}) v={case.v_head_dim}"
    )
    print("+-------------------+----------+----------+----------+--------+------------+-----------+")
    print("| stage             | avg us   | p50 us   | min us   | pct    | est MiB    | est TF/s  |")
    print("+-------------------+----------+----------+----------+--------+------------+-----------+")
    for name in STAGE_ORDER:
        s = _stats(timings_us[name])
        pct = 100.0 * s["avg_us"] / total_avg if total_avg > 0 else 0.0
        estimate = estimates[name]
        mib = estimate.traffic_bytes / (1024 * 1024)
        tflops = estimate.flops / (s["avg_us"] * 1e-6) / 1e12 if estimate.flops > 0 and s["avg_us"] > 0 else 0.0
        print(
            f"| {name:<17} | {s['avg_us']:>8.0f} | {s['p50_us']:>8.0f} | {s['min_us']:>8.0f} | "
            f"{pct:>5.1f}% | {mib:>10.2f} | {tflops:>9.2f} |"
        )
        rows.append(
            {
                "case": case.name,
                "stage": name,
                "avg_us": s["avg_us"],
                "p50_us": s["p50_us"],
                "min_us": s["min_us"],
                "max_us": s["max_us"],
                "pct_of_section_sum": pct,
                "est_traffic_bytes": estimate.traffic_bytes,
                "est_flops": estimate.flops,
                "est_tflops_per_s": tflops,
            }
        )
    print("+-------------------+----------+----------+----------+--------+------------+-----------+")
    print(f"section_sum_avg_us={total_avg:.0f}")
    if end_to_end_us:
        e2e = _stats(end_to_end_us)
        print(f"end_to_end_avg_us={e2e['avg_us']:.0f} p50_us={e2e['p50_us']:.0f} min_us={e2e['min_us']:.0f}")
    return rows


def run_case(
    case: MLACase,
    warmup: int,
    iterations: int,
    grad_mode: str,
    seed: int,
    end_to_end: bool,
) -> list[dict[str, str | float]]:
    ctx = ttml.autograd.AutoContext.get_instance()
    device = ctx.get_device()
    rng = np.random.default_rng(seed)

    config = _make_config(case)
    rope_params = ttml.ops.rope.build_rope_params(config.max_seq_len, config.qk_rope_head_dim, config.rope_theta)
    module = MultiHeadLatentAttention(config, rope_params)

    timings_us = {name: [] for name in STAGE_ORDER}
    end_to_end_us: list[float] = []
    previous_grad_mode = ctx.get_gradient_mode()
    ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED if grad_mode == "enabled" else ttml.autograd.GradMode.DISABLED)
    try:
        for iteration in range(warmup + iterations):
            x, mask = _make_inputs(case, rng)
            _sync(device)
            target_timings = timings_us if iteration >= warmup else {name: [] for name in STAGE_ORDER}
            _run_profiled_forward(module, x, mask, device, target_timings)
            ctx.reset_graph()
        if end_to_end:
            for iteration in range(warmup + iterations):
                x, mask = _make_inputs(case, rng)
                elapsed_us = _run_plain_forward(module, x, mask, device)
                if iteration >= warmup:
                    end_to_end_us.append(elapsed_us)
                ctx.reset_graph()
    finally:
        ctx.set_gradient_mode(previous_grad_mode)
        ctx.reset_graph()

    return _print_case(case, timings_us, end_to_end_us)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="smoke,nano,tiny-short",
        help=f"Comma-separated case names. Available: {', '.join(CASES)}",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--grad-mode",
        choices=("disabled", "enabled"),
        default="disabled",
        help="Forward kernels are the same; disabled avoids autograd graph overhead in the timing loop.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional path to write raw summary rows.")
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help="Also time an unsplit MLA forward with a single synchronize before/after the whole call.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_cases = [name.strip() for name in args.cases.split(",") if name.strip()]
    unknown = [name for name in selected_cases if name not in CASES]
    if unknown:
        raise SystemExit(f"Unknown cases: {unknown}. Available: {', '.join(CASES)}")

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    all_rows: list[dict[str, str | float]] = []
    try:
        for idx, name in enumerate(selected_cases):
            all_rows.extend(
                run_case(CASES[name], args.warmup, args.iterations, args.grad_mode, args.seed + idx, args.end_to_end)
            )
    finally:
        ctx.close_device()

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "case",
                    "stage",
                    "avg_us",
                    "p50_us",
                    "min_us",
                    "max_us",
                    "pct_of_section_sum",
                    "est_traffic_bytes",
                    "est_flops",
                    "est_tflops_per_s",
                ],
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
