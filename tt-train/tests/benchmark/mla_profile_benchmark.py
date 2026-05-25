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
    "kv_down_split",
    "k_rope",
    "kv_up_projection",
    "qkv_assemble",
    "q_rope",
    "sdpa",
    "heads_fusion",
    "output_projection",
]

BACKWARD_STAGE_ORDER = [
    "autograd_forward",
    "backward",
    "forward_backward",
]

_FLOAT_DTYPES = frozenset(
    {
        ttnn.DataType.BFLOAT16,
        ttnn.DataType.FLOAT32,
        ttnn.DataType.BFLOAT8_B,
        ttnn.DataType.BFLOAT4_B,
    }
)


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


def _mark_backward_boundary(tensor, stage: str, boundary: str, events: list[tuple[str, str, float]]):
    import _ttml as cpp

    ctx = ttml.autograd.AutoContext.get_instance()
    if ctx.get_gradient_mode() != ttml.autograd.GradMode.ENABLED:
        return tensor
    if tensor.get_value().dtype not in _FLOAT_DTYPES:
        return tensor

    output = cpp.autograd.create_tensor(tensor.get_value(), requires_grad=True)
    input_tensor = tensor

    def backward_fn():
        _sync(ctx.get_device())
        events.append((stage, boundary, time.perf_counter()))
        grad = output.get_grad() if output.is_grad_initialized() else cpp.core.zeros_like(output.get_value())
        if input_tensor.get_requires_grad():
            input_tensor.add_grad(grad)

    links = []
    input_node = tensor.get_node()
    if input_node is not None:
        links.append(input_node)

    node_id = ctx.add_backward_node(backward_fn, links)
    if node_id is not None:
        output.set_node(node_id)
    return output


def _mark_backward_boundaries(tensors, stage: str, boundary: str, events: list[tuple[str, str, float]]):
    return tuple(_mark_backward_boundary(tensor, stage, boundary, events) for tensor in tensors)


def _backward_section_times_us(events: list[tuple[str, str, float]]) -> dict[str, float]:
    timings_us: dict[str, float] = {}
    for stage in STAGE_ORDER:
        starts = [
            timestamp for event_stage, boundary, timestamp in events if event_stage == stage and boundary == "start"
        ]
        ends = [timestamp for event_stage, boundary, timestamp in events if event_stage == stage and boundary == "end"]
        if starts and ends:
            timings_us[stage] = max(0.0, (max(starts) - min(ends)) * 1_000_000.0)
    return timings_us


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

    q_pre = _time_stage(device, timings_us, "q_projection", q_projection)

    def kv_down_split():
        kv_full = module.wkv_a(x)
        kv = autograd_slice(kv_full, [0, 0, 0, 0], [B, 1, S, kv_lora])
        k_pe = autograd_slice(kv_full, [0, 0, 0, kv_lora], [B, 1, S, kv_lora + module.qk_rope_head_dim])
        return kv, k_pe

    kv, k_pe = _time_stage(device, timings_us, "kv_down_split", kv_down_split)

    k_pe = _time_stage(device, timings_us, "k_rope", lambda: ttml.ops.rope.rope(k_pe, module.rope_params))

    kv_up = _time_stage(device, timings_us, "kv_up_projection", lambda: module.wkv_b(module.kv_norm(kv)))

    def qkv_assemble():
        q = split_heads(q_pre, n_heads)
        kv_up_heads = split_heads(kv_up, n_heads)
        k_nope = autograd_slice(kv_up_heads, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
        v = autograd_slice(kv_up_heads, [0, 0, 0, qk_nope], [B, n_heads, S, qk_nope + module.v_head_dim])
        k_pe_broadcast = autograd_concat([k_pe] * n_heads, dim=1)
        k_full = autograd_concat([k_nope, k_pe_broadcast], dim=3)
        return q, k_full, v

    q_full, k_full, v = _time_stage(device, timings_us, "qkv_assemble", qkv_assemble)

    def q_rope():
        q_nope = autograd_slice(q_full, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
        q_pe = autograd_slice(q_full, [0, 0, 0, qk_nope], [B, n_heads, S, qk_head])
        q_pe = ttml.ops.rope.rope(q_pe, module.rope_params)
        return autograd_concat([q_nope, q_pe], dim=3)

    q_full = _time_stage(device, timings_us, "q_rope", q_rope)

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


def _run_marked_forward(module: MultiHeadLatentAttention, x, mask, events: list[tuple[str, str, float]]):
    B, _, S, _ = list(x.get_value().shape)
    n_heads = module.n_heads
    qk_nope = module.qk_nope_head_dim
    qk_head = module.qk_head_dim
    kv_lora = module.kv_lora_rank

    x_q = _mark_backward_boundary(x, "q_projection", "start", events)
    if module.q_lora_rank == 0:
        q_pre = module.wq(x_q)
    else:
        q_pre = module.wq_b(module.q_norm(module.wq_a(x_q)))
    q_pre = _mark_backward_boundary(q_pre, "q_projection", "end", events)

    x_kv = _mark_backward_boundary(x, "kv_down_split", "start", events)
    kv_full = module.wkv_a(x_kv)
    kv = autograd_slice(kv_full, [0, 0, 0, 0], [B, 1, S, kv_lora])
    k_pe = autograd_slice(kv_full, [0, 0, 0, kv_lora], [B, 1, S, kv_lora + module.qk_rope_head_dim])
    kv, k_pe = _mark_backward_boundaries((kv, k_pe), "kv_down_split", "end", events)

    k_pe = _mark_backward_boundary(k_pe, "k_rope", "start", events)
    k_pe = ttml.ops.rope.rope(k_pe, module.rope_params)
    k_pe = _mark_backward_boundary(k_pe, "k_rope", "end", events)

    kv = _mark_backward_boundary(kv, "kv_up_projection", "start", events)
    kv_up = module.wkv_b(module.kv_norm(kv))
    kv_up = _mark_backward_boundary(kv_up, "kv_up_projection", "end", events)

    q_pre, kv_up, k_pe = _mark_backward_boundaries((q_pre, kv_up, k_pe), "qkv_assemble", "start", events)
    q_full = split_heads(q_pre, n_heads)
    kv_up_heads = split_heads(kv_up, n_heads)
    k_nope = autograd_slice(kv_up_heads, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
    v = autograd_slice(kv_up_heads, [0, 0, 0, qk_nope], [B, n_heads, S, qk_nope + module.v_head_dim])
    k_pe_broadcast = autograd_concat([k_pe] * n_heads, dim=1)
    k_full = autograd_concat([k_nope, k_pe_broadcast], dim=3)
    q_full, k_full, v = _mark_backward_boundaries((q_full, k_full, v), "qkv_assemble", "end", events)

    q_full = _mark_backward_boundary(q_full, "q_rope", "start", events)
    q_nope = autograd_slice(q_full, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
    q_pe = autograd_slice(q_full, [0, 0, 0, qk_nope], [B, n_heads, S, qk_head])
    q_pe = ttml.ops.rope.rope(q_pe, module.rope_params)
    q_full = autograd_concat([q_nope, q_pe], dim=3)
    q_full = _mark_backward_boundary(q_full, "q_rope", "end", events)

    q_full, k_full, v = _mark_backward_boundaries((q_full, k_full, v), "sdpa", "start", events)
    attn = ttml.ops.attention.scaled_dot_product_attention_composite(q_full, k_full, v, mask)
    attn = _mark_backward_boundary(attn, "sdpa", "end", events)

    attn = _mark_backward_boundary(attn, "heads_fusion", "start", events)
    attn = ttml.ops.multi_head_utils.heads_fusion(attn)
    attn = _mark_backward_boundary(attn, "heads_fusion", "end", events)

    attn = _mark_backward_boundary(attn, "output_projection", "start", events)
    output = module.wo(attn)
    return _mark_backward_boundary(output, "output_projection", "end", events)


def _run_forward_backward(
    module: MultiHeadLatentAttention, x, mask, device
) -> tuple[float, float, float, dict[str, float]]:
    events: list[tuple[str, str, float]] = []
    _sync(device)
    total_start = time.perf_counter()
    forward_start = total_start
    output = _run_marked_forward(module, x, mask, events)
    loss = ttml.ops.unary.mean(output)
    _sync(device)
    forward_us = (time.perf_counter() - forward_start) * 1_000_000.0

    backward_start = time.perf_counter()
    loss.backward(False)
    _sync(device)
    backward_us = (time.perf_counter() - backward_start) * 1_000_000.0
    forward_backward_us = (time.perf_counter() - total_start) * 1_000_000.0
    return forward_us, backward_us, forward_backward_us, _backward_section_times_us(events)


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
    backward_timings_us: dict[str, list[float]] | None = None,
    backward_section_timings_us: dict[str, list[float]] | None = None,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    total_avg = sum(_stats(timings_us[name])["avg_us"] for name in STAGE_ORDER)

    print(f"\nMLA case: {case.name}")
    print(
        "shape: "
        f"B={case.batch_size} S={case.seq_len} dim={case.dim} H={case.n_heads} "
        f"q_lora={case.q_lora_rank} kv_lora={case.kv_lora_rank} "
        f"qk=({case.qk_nope_head_dim}+{case.qk_rope_head_dim}) v={case.v_head_dim}"
    )
    print("+-------------------+----------+----------+----------+--------+")
    print("| stage             | avg us   | p50 us   | min us   | pct    |")
    print("+-------------------+----------+----------+----------+--------+")
    for name in STAGE_ORDER:
        s = _stats(timings_us[name])
        pct = 100.0 * s["avg_us"] / total_avg if total_avg > 0 else 0.0
        print(f"| {name:<17} | {s['avg_us']:>8.0f} | {s['p50_us']:>8.0f} | {s['min_us']:>8.0f} | " f"{pct:>5.1f}% |")
        rows.append(
            {
                "case": case.name,
                "stage": name,
                "avg_us": s["avg_us"],
                "p50_us": s["p50_us"],
                "min_us": s["min_us"],
                "max_us": s["max_us"],
                "pct_of_section_sum": pct,
            }
        )
    print("+-------------------+----------+----------+----------+--------+")
    print(f"section_sum_avg_us={total_avg:.0f}")
    if end_to_end_us:
        e2e = _stats(end_to_end_us)
        print(f"end_to_end_avg_us={e2e['avg_us']:.0f} p50_us={e2e['p50_us']:.0f} min_us={e2e['min_us']:.0f}")
        rows.append(
            {
                "case": case.name,
                "stage": "end_to_end_forward",
                "avg_us": e2e["avg_us"],
                "p50_us": e2e["p50_us"],
                "min_us": e2e["min_us"],
                "max_us": e2e["max_us"],
                "pct_of_section_sum": 0.0,
            }
        )
    if backward_timings_us:
        print("backward measurements:")
        for name in BACKWARD_STAGE_ORDER:
            s = _stats(backward_timings_us[name])
            print(f"  {name}: avg_us={s['avg_us']:.0f} p50_us={s['p50_us']:.0f} min_us={s['min_us']:.0f}")
            rows.append(
                {
                    "case": case.name,
                    "stage": name,
                    "avg_us": s["avg_us"],
                    "p50_us": s["p50_us"],
                    "min_us": s["min_us"],
                    "max_us": s["max_us"],
                    "pct_of_section_sum": 0.0,
                }
            )
    if backward_section_timings_us:
        total_backward_avg = sum(_stats(backward_section_timings_us[name])["avg_us"] for name in STAGE_ORDER)
        print("backward section measurements:")
        print("+-------------------+----------+----------+----------+--------+")
        print("| stage             | avg us   | p50 us   | min us   | pct    |")
        print("+-------------------+----------+----------+----------+--------+")
        for name in STAGE_ORDER:
            s = _stats(backward_section_timings_us[name])
            pct = 100.0 * s["avg_us"] / total_backward_avg if total_backward_avg > 0 else 0.0
            print(
                f"| {name:<17} | {s['avg_us']:>8.0f} | {s['p50_us']:>8.0f} | {s['min_us']:>8.0f} | " f"{pct:>5.1f}% |"
            )
            rows.append(
                {
                    "case": case.name,
                    "stage": f"backward_{name}",
                    "avg_us": s["avg_us"],
                    "p50_us": s["p50_us"],
                    "min_us": s["min_us"],
                    "max_us": s["max_us"],
                    "pct_of_section_sum": pct,
                }
            )
        print("+-------------------+----------+----------+----------+--------+")
        print(f"backward_section_sum_avg_us={total_backward_avg:.0f}")
    return rows


def run_case(
    case: MLACase,
    warmup: int,
    iterations: int,
    grad_mode: str,
    seed: int,
    end_to_end: bool,
    backward: bool,
) -> list[dict[str, str | float]]:
    ctx = ttml.autograd.AutoContext.get_instance()
    device = ctx.get_device()
    rng = np.random.default_rng(seed)

    config = _make_config(case)
    rope_params = ttml.ops.rope.build_rope_params(config.max_seq_len, config.qk_rope_head_dim, config.rope_theta)
    module = MultiHeadLatentAttention(config, rope_params)
    opt_cfg = ttml.optimizers.SGDConfig.make(0.0, 0.0, 0.0, 0.0, False)
    zero_grad_optimizer = ttml.optimizers.SGD(module.parameters(), opt_cfg)

    timings_us = {name: [] for name in STAGE_ORDER}
    end_to_end_us: list[float] = []
    backward_timings_us = {name: [] for name in BACKWARD_STAGE_ORDER}
    backward_section_timings_us = {name: [] for name in STAGE_ORDER}
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
        if backward:
            ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
            for iteration in range(warmup + iterations):
                zero_grad_optimizer.zero_grad()
                x, mask = _make_inputs(case, rng)
                forward_us, backward_us, forward_backward_us, section_times_us = _run_forward_backward(
                    module, x, mask, device
                )
                if iteration >= warmup:
                    backward_timings_us["autograd_forward"].append(forward_us)
                    backward_timings_us["backward"].append(backward_us)
                    backward_timings_us["forward_backward"].append(forward_backward_us)
                    for name in STAGE_ORDER:
                        backward_section_timings_us[name].append(section_times_us[name])
                ctx.reset_graph()
    finally:
        ctx.set_gradient_mode(previous_grad_mode)
        ctx.reset_graph()

    return _print_case(
        case,
        timings_us,
        end_to_end_us,
        backward_timings_us if backward else None,
        backward_section_timings_us if backward else None,
    )


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
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Also time normal forward, backward-only, and forward+backward with autograd enabled.",
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
                run_case(
                    CASES[name],
                    args.warmup,
                    args.iterations,
                    args.grad_mode,
                    args.seed + idx,
                    args.end_to_end,
                    args.backward,
                )
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
                ],
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
