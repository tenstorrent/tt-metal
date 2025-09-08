# model_profile.py
import argparse, json, yaml
from dataclasses import dataclass
from typing import Optional, Dict


# ----------------------------
# FLOPs (Chinchilla-style)
# ----------------------------
def chinchilla_flops(
    seq_len: int,
    vocab_size: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    ffw_size: int,
    gqa_groups: Optional[int] = None,
    forward_only: bool = False,
) -> int:
    key_size = d_model // num_heads
    kv_heads = num_heads if not gqa_groups or gqa_groups < 2 else max(1, num_heads // gqa_groups)

    # Q/K/V projections
    q_proj = 2 * seq_len * d_model * (key_size * num_heads)
    k_proj = 2 * seq_len * d_model * (key_size * kv_heads)
    v_proj = 2 * seq_len * d_model * (key_size * kv_heads)

    # Attention math
    att_logits = 2 * num_heads * seq_len * seq_len * key_size
    att_softmax = 3 * num_heads * seq_len * seq_len
    att_value = 2 * num_heads * seq_len * seq_len * key_size
    att_out = 2 * seq_len * (key_size * num_heads) * d_model

    att = q_proj + k_proj + v_proj + att_logits + att_softmax + att_value + att_out
    ffw = 2 * seq_len * (d_model * ffw_size + ffw_size * d_model)

    forward = num_layers * (att + ffw)
    return forward if forward_only else forward * 3  # +2x backward


# ----------------------------
# Param count
# ----------------------------
def gpt_params(
    d_model: int, num_heads: int, num_layers: int, seq_len: int, vocab_size: int, include_embed: bool
) -> int:
    ffw_size = 4 * d_model
    attention = 3 * d_model**2 + 3 * d_model
    attproj = d_model**2 + d_model
    ffw = d_model * ffw_size + ffw_size
    ffwproj = ffw_size * d_model + d_model
    layernorms = 2 * 2 * d_model
    ln_f = 2 * d_model
    lm_head = d_model * vocab_size  # no bias

    total = num_layers * (attention + attproj + ffw + ffwproj + layernorms) + ln_f + lm_head
    if include_embed:
        total += d_model * vocab_size + d_model * seq_len  # token + pos
    return total


def llama_params(
    d_model: int, num_heads: int, num_layers: int, vocab_size: int, num_groups: Optional[int], include_embed: bool
) -> int:
    kv_heads = num_heads if not num_groups or num_groups < 2 else max(1, num_heads // num_groups)
    head_dim = d_model // num_heads
    kv_dim = kv_heads * head_dim

    # attention (no biases)
    att = (d_model * d_model) + (d_model * kv_dim) + (d_model * kv_dim) + (d_model * d_model)

    # SwiGLU MLP (~8/3 * d_model)
    n_ff = int(round((8.0 / 3.0) * d_model))
    mlp = (d_model * n_ff) + (d_model * n_ff) + (n_ff * d_model)

    norms = 2 * d_model
    total = num_layers * (att + mlp + norms) + d_model  # final RMSNorm

    if include_embed:
        # Count token embeddings + lm_head (tied weights treated as one matrix here)
        total += d_model * vocab_size
    return total


# ----------------------------
# Helpers
# ----------------------------
DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}

# Device peak TFLOPS (public specs; pick the closest variant)
# precision keys: 'fp16' (use for bf16 too), 'fp8'
DEVICE_PEAK_TFLOPS: Dict[str, Dict[str, float]] = {
    # Wormhole n150 (12 GB) vs n300 (24 GB)
    "wormhole-n150": {"fp16": 74.0, "fp8": 262.0},
    "wormhole-n300": {"fp16": 131.0, "fp8": 466.0},
    # Blackhole variants (range seen 745–774 FP8; FP16 ~372 TFLOPS)
    "blackhole": {"fp16": 372.0, "fp8": 774.0},
}


def pick_peak(device: str, precision: str) -> float:
    device = device.lower()
    precision = precision.lower()
    if device not in DEVICE_PEAK_TFLOPS:
        raise ValueError(f"Unknown device '{device}'. Known: {list(DEVICE_PEAK_TFLOPS.keys())}")
    if precision not in ("fp16", "bf16", "fp8"):
        raise ValueError("precision must be one of: fp16, bf16, fp8")
    key = "fp16" if precision in ("fp16", "bf16") else "fp8"
    return DEVICE_PEAK_TFLOPS[device][key]


@dataclass
class RunSpec:
    model_type: str
    d_model: int
    num_heads: int
    num_layers: int
    seq_len: int
    vocab_size: int
    num_groups: Optional[int]
    batch_size: int
    grad_accum: int


def estimate(config: dict, dtype: str, include_embed: bool, forward_only: bool):
    tcfg = config["training_config"]
    mtype = tcfg["model_type"].lower()
    tr = tcfg["transformer_config"]

    spec = RunSpec(
        model_type=mtype,
        d_model=int(tr["embedding_dim"]),
        num_heads=int(tr["num_heads"]),
        num_layers=int(tr["num_blocks"]),
        seq_len=int(tr["max_sequence_length"]),
        vocab_size=int(tr["vocab_size"]),
        num_groups=int(tr.get("num_groups", 1)) if "num_groups" in tr else None,
        batch_size=int(tcfg.get("batch_size", 1)),
        grad_accum=int(tcfg.get("gradient_accumulation_steps", 1)),
    )

    if mtype == "gpt2":
        params = gpt_params(spec.d_model, spec.num_heads, spec.num_layers, spec.seq_len, spec.vocab_size, include_embed)
        ffw_size = 4 * spec.d_model
        flops_seq = chinchilla_flops(
            spec.seq_len,
            spec.vocab_size,
            spec.d_model,
            spec.num_heads,
            spec.num_layers,
            ffw_size,
            forward_only=forward_only,
        )
    elif mtype == "llama":
        params = llama_params(
            spec.d_model, spec.num_heads, spec.num_layers, spec.vocab_size, spec.num_groups, include_embed
        )
        ffw_size = int(round((8.0 / 3.0) * spec.d_model))
        flops_seq = chinchilla_flops(
            spec.seq_len,
            spec.vocab_size,
            spec.d_model,
            spec.num_heads,
            spec.num_layers,
            ffw_size,
            gqa_groups=spec.num_groups,
            forward_only=forward_only,
        )
    else:
        raise ValueError(f"Unsupported model_type {mtype}")

    microbatches = spec.batch_size * spec.grad_accum
    flops_step = flops_seq * microbatches

    bytes_per_param = DTYPE_BYTES[dtype.lower()]
    size_gb = (params * bytes_per_param) / (1024**3)

    return {
        "spec": spec,
        "params": params,
        "model_size_gb": round(size_gb, 4),
        "dtype": dtype.lower(),
        "ffw_size_used": ffw_size,
        "flops_per_sequence": int(flops_seq),
        "flops_per_step": int(flops_step),
        "forward_only": forward_only,
        "include_embed": include_embed,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Estimate params/FLOPs and device efficiency.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--dtype", default="bf16", choices=DTYPE_BYTES.keys())
    ap.add_argument("--include-embed", action="store_true", help="Include embeddings in param count.")
    ap.add_argument("--forward-only", action="store_true", help="FLOPs forward only (no backward).")

    # New: device/throughput options
    ap.add_argument(
        "--device",
        choices=["wormhole-n150", "wormhole-n300", "blackhole"],
        help="Tenstorrent device to compare against.",
    )
    ap.add_argument(
        "--peak-precision",
        choices=["fp16", "bf16", "fp8"],
        default="bf16",
        help="Precision for device peak comparison (bf16≈fp16).",
    )
    ap.add_argument(
        "--measured-time", type=float, help="Measured time in seconds for one step (or one fwd pass if --forward-only)."
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out = estimate(cfg, args.dtype, args.include_embed, args.forward_only)

    # If device + time provided, compute achieved TFLOPS and % of peak
    efficiency = None
    if args.device and args.measured_time and args.measured_time > 0:
        # choose whether to use per-sequence or per-step based on what was timed
        # Convention: user timings are per-step (microbatches included).
        work_flops = out["flops_per_step"]
        achieved_tflops = (work_flops / args.measured_time) / 1e12
        peak = pick_peak(args.device, args.peak_precision)
        percent = 100.0 * achieved_tflops / peak if peak > 0 else 0.0
        efficiency = {
            "device": args.device,
            "peak_precision": args.peak_precision,
            "peak_tflops": peak,
            "measured_time_s": args.measured_time,
            "achieved_tflops": round(achieved_tflops, 3),
            "percent_of_peak": round(percent, 2),
        }

    if args.json:
        payload = {"profile": out, "efficiency": efficiency}
        print(json.dumps(payload, indent=2, default=str))
        return

    # Pretty print
    print("\n=== Model Profile ===")
    print(f"model_type              : {out['spec'].model_type}")
    print(f"d_model / heads / layers: {out['spec'].d_model} / {out['spec'].num_heads} / {out['spec'].num_layers}")
    print(f"seq_len / vocab         : {out['spec'].seq_len} / {out['spec'].vocab_size}")
    print(f"num_groups (GQA)        : {out['spec'].num_groups}")
    print(f"dtype                   : {out['dtype']}")
    print(f"FFW size used           : {out['ffw_size_used']}")
    print(f"Params (incl_embed={out['include_embed']}): {out['params']:,}")
    print(f"Model size GB           : {out['model_size_gb']}")
    print(f"FLOPs / sequence        : {out['flops_per_sequence']:,} (forward_only={out['forward_only']})")
    print(f"FLOPs / step            : {out['flops_per_step']:,} (uses batch_size * grad_accum)")
    if efficiency:
        print("\n=== Device Efficiency ===")
        print(f"Device                  : {efficiency['device']} ({efficiency['peak_precision']})")
        print(f"Peak TFLOPS             : {efficiency['peak_tflops']}")
        print(f"Measured time (s)       : {efficiency['measured_time_s']}")
        print(f"Achieved TFLOPS         : {efficiency['achieved_tflops']}")
        print(f"% of peak               : {efficiency['percent_of_peak']}%")
    print()


if __name__ == "__main__":
    main()
