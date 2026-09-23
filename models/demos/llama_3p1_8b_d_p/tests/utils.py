# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only test utilities: checkpoint reads, metrics and independent shard coordinates."""

import json
import math
import os
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

FULL_LIMITS = (0.99, 0.15)


def read_raw_weights(checkpoint_path, names):
    """Read raw HF tensors; a mapping optionally gives each tensor a local test name."""
    path = Path(checkpoint_path)
    index_path = path / "model.safetensors.index.json"
    assert index_path.is_file(), f"required checkpoint index is unavailable: {index_path}"
    index = json.loads(index_path.read_text())["weight_map"]
    names = names if isinstance(names, dict) else {name: name for name in names}
    result = {}
    for local_name, checkpoint_name in names.items():
        shard_path = path / index[checkpoint_name]
        assert shard_path.is_file(), f"required checkpoint shard is unavailable: {shard_path}"
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            result[local_name] = shard.get_tensor(checkpoint_name)
    return result


def metrics(expected, actual, *, dtype=torch.float64, error_type=AssertionError):
    """PCC and normalized L2; retain explicit float32 accumulation for the indexed-RoPE test."""
    # Large repeated SP rows need float64 reductions to avoid understating PCC.
    expected, actual = expected.to(dtype).flatten(), actual.to(dtype).flatten()
    if not expected.numel() or expected.shape != actual.shape:
        raise error_type("metrics require equal nonempty tensors")
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise error_type("metrics require finite expected and actual tensors")
    e, a = expected - expected.mean(), actual - actual.mean()
    denominator = torch.linalg.vector_norm(e) * torch.linalg.vector_norm(a)
    norm = torch.linalg.vector_norm(expected)
    if denominator == 0 or norm == 0:
        raise error_type("metrics require nonzero centered and reference norms")
    pcc = (torch.dot(e, a) / denominator).item()
    nl2 = (torch.linalg.vector_norm(actual - expected) / norm).item()
    if not math.isfinite(pcc) or not math.isfinite(nl2):
        raise error_type(f"metrics must be finite, got PCC={pcc}, NL2={nl2}")
    return pcc, nl2


def hidden_limits(num_layers, is_bf16):
    return ((0.999, 0.025) if is_bf16 else (0.999, 0.05)) if num_layers == 1 else FULL_LIMITS


def kv_limits(layer_idx, is_bf16):
    # Layer zero has no preceding device error accumulation.
    if layer_idx == 0:
        return (0.9999, 0.01) if is_bf16 else (0.999, 0.02)
    return FULL_LIMITS


def selected_logit_positions(length):
    """Sample the prompt plus tile/chunk boundaries and every short continuation row."""
    positions = set(range(length)) if length <= 65 else set(range(0, length, 8))
    positions.update(p for p in [31, 32, 255, 256, 511, 512, 1023, 1024, 1535, 1536, length - 1] if p < length)
    positions.update(range(1024, min(1033, length)))
    return sorted(positions)


def positions(start, sp):
    return torch.tensor([p for p in range(start, start + 1024) if (p // 256) % 4 == sp])


def check_metric(expected, actual, limits, label, records, *, enforce=True):
    pcc, nl2 = metrics(expected, actual)
    within_limits = pcc >= limits[0] and nl2 <= limits[1]
    records.append(
        {"label": label, "pcc": pcc, "nl2": nl2, "limits": limits, "within_limits": within_limits, "enforced": enforce}
    )
    logger.info(f"{label}: PCC={pcc:.9f}, NL2={nl2:.9f}")
    if enforce:
        assert within_limits, (label, pcc, nl2, limits)


def check_logits_rows(expected, actual, selected, *, start, end, limits, records):
    check_metric(expected, actual, limits, f"logits range=[{start},{end})", records, enforce=True)
    wanted = expected.argmax(dim=-1)
    top1 = (actual.argmax(dim=-1) == wanted).float().mean().item()
    top5 = (actual.topk(5, dim=-1).indices == wanted[:, None]).any(dim=-1).float().mean().item()
    records.append(
        {"label": "teacher-forced token agreement", "positions": selected.tolist(), "top1": top1, "top5": top5}
    )
    logger.info(f"teacher-forced {len(selected)} positions: top1={top1:.6f}, top5={top5:.6f}")
    assert top1 >= 0.90 and top5 >= 0.99
    return actual


def assemble_head(shards, plane, head, *, stripe_tokens=256):
    """Join four SP shards in natural token order for one plane and one TP head."""
    if len(shards) != 32 or type(stripe_tokens) is not int or stripe_tokens <= 0:
        raise ValueError("Expected 32 shards and a positive stripe length")
    shape = tuple(shards[0].shape)
    if len(shape) != 4 or shape[1] != 1 or not shape[2] or shape[2] % stripe_tokens:
        raise ValueError("Each shard must contain complete SP stripes and one local KV head")
    if not 0 <= plane < shape[0] or not 0 <= head < 8:
        raise ValueError("Plane or TP head is out of range")
    if any(tuple(s.shape) != shape or s.dtype != shards[0].dtype or s.device.type != "cpu" for s in shards):
        raise ValueError("All shards must have matching CPU shape and dtype")
    # Local stripe 0 from SP0..3 precedes local stripe 1 from SP0..3.
    pieces = [shards[sp * 8 + head][plane, 0].reshape(-1, stripe_tokens, shape[-1]) for sp in range(4)]
    return torch.stack(pieces, dim=1).reshape(-1, shape[-1])


CHUNKS = ((0, 0, 1024), (1, 0, 1024), (0, 1024, 2048), (1, 1024, 2048))
LAYER_NAMES = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def join_hidden_tp_replicas(shards):
    """Validate all 32 chip replicas before retaining one 256-token stripe from each SP."""
    if len(shards) != 32:
        raise ValueError("Expected 32 hidden shards")
    shape = tuple(shards[0].shape)
    if len(shape) != 4 or shape[:3] != (1, 1, 256):
        raise ValueError("Expected hidden shard [1,1,256,features]")
    if any(tuple(x.shape) != shape or x.dtype != shards[0].dtype or x.device.type != "cpu" for x in shards):
        raise ValueError("Hidden shards must share CPU shape and dtype")
    pieces = []
    for sp in range(4):
        first = shards[sp * 8]
        for tp in range(8):
            if not torch.equal(first, shards[sp * 8 + tp]):
                raise AssertionError(f"TP replica mismatch at SP={sp}, TP={tp}")
        pieces.append(first[0, 0])
    return torch.cat(pieces, dim=0).contiguous()


def full_hidden(chunks):
    """Only two complete, ordered chunks can form one native 2K layer output."""
    if set(chunks) != {0, 1024}:
        raise ValueError("Expected chunks beginning at 0 and 1024")
    left, right = chunks[0], chunks[1024]
    if left.ndim != 2 or left.shape[0] != 1024 or right.shape != left.shape or right.dtype != left.dtype:
        raise ValueError("Expected two equal [1024,features] hidden chunks")
    return torch.cat((left, right), dim=0)


def layer_input(embedding_chunks, hidden_layers, layer_idx):
    """Layer zero consumes native embedding; later layers consume the previous native output."""
    if type(layer_idx) is not int or not 0 <= layer_idx < 32:
        raise ValueError("Layer index outside [0,32)")
    return full_hidden(embedding_chunks if layer_idx == 0 else hidden_layers[layer_idx - 1])


def local_limits(cache_dtype, kind):
    """Use the published decoder bounds without an accumulated-error allowance."""
    if cache_dtype not in ("bfloat16", "bfloat8_b") or kind not in ("hidden", "k", "v"):
        raise ValueError("Unknown dtype or comparison kind")
    if kind == "hidden":
        return (0.999, 0.025) if cache_dtype == "bfloat16" else (0.999, 0.05)
    return (0.9999, 0.01) if cache_dtype == "bfloat16" else (0.999, 0.02)


def windows():
    """Each interval is one natural 256-token SP stripe inside one of the two chunks."""
    return [(chunk, sp, chunk + sp * 256, chunk + (sp + 1) * 256) for chunk in (0, 1024) for sp in range(4)]


def validate_observer_order(observed):
    if observed != list(range(32)):
        raise AssertionError("Every real layer must complete once in checkpoint order")


def score_row(metrics, expected, actual, limits, **coordinates):
    assert expected.shape == actual.shape and expected.numel()
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    pcc, nl2 = metrics(expected, actual)
    assert math.isfinite(pcc) and math.isfinite(nl2)
    return dict(coordinates, pcc=pcc, nl2=nl2, limits=limits, within_limits=pcc >= limits[0] and nl2 <= limits[1])


def write_pcc_summary(report, root):
    """Retain per-layer worst cases in CI without uploading captured tensors or checkpoint data."""
    groups = {}
    for row in report["local_rows"]:
        key = (row["slot"], row["layer"], row["kind"])
        group = groups.setdefault(
            key,
            dict(
                slot=key[0],
                layer=key[1],
                kind=key[2],
                checks=0,
                min_pcc=1.0,
                max_nl2=0.0,
                limits=row["limits"],
                passed=True,
            ),
        )
        group["checks"] += 1
        group["min_pcc"] = min(group["min_pcc"], row["pcc"])
        group["max_nl2"] = max(group["max_nl2"], row["nl2"])
        group["passed"] &= row["within_limits"]
    rows = [groups[key] for key in sorted(groups)]
    summary = {key: report[key] for key in ("status", "case_passed", "prompt_case", "cache_dtype", "counts")}
    summary.update(
        reference="Each layer uses its actual device input; extrema include every head and SP stripe.",
        layers=rows,
        failures=report["local_misses"],
        final_outputs=report["final_global_rows"],
        token_agreement=[r for r in report["records"] if r["label"] == "teacher-forced token agreement"],
    )
    directory = Path(root) / "pcc"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"llama31-2k-{report['prompt_case']}-{report['cache_dtype']}"
    (directory / f"{name}.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    text = [
        f"# Llama-3.1-8B 2K: {report['prompt_case']}, {report['cache_dtype']}",
        "",
        f"Case: {'PASS' if report['case_passed'] else 'FAIL / INCOMPLETE'}",
        "",
        summary["reference"],
        "",
        "| Slot | Layer | Tensor | Checks | Min PCC | Max NL2 | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    text.extend(
        f"| {r['slot']} | {r['layer']} | {r['kind']} | {r['checks']} | {r['min_pcc']:.9f} | "
        f"{r['max_nl2']:.9f} | {'PASS' if r['passed'] else 'FAIL'} |"
        for r in rows
    )
    (directory / f"{name}.md").write_text("\n".join(text) + "\n")


def prefill_runner_scenario():
    """Resolve the common runner's deterministic two-slot acceptance workload from the model manifest."""
    manifest = Path(__file__).resolve().parents[1] / "tt/runners/manifests/llama_3p1_8b.json"
    env = json.loads(manifest.read_text())["env"]
    capacity = int(os.environ.get("PREFILL_MAX_SEQ_LEN", env["PREFILL_MAX_SEQ_LEN"]))
    if capacity not in (2048, 4096, 8192, 16384, 32768, 65536):
        raise ValueError("runner acceptance capacity must be 2K, 4K, 8K, 16K, 32K or 64K")
    users, layers, chunk_size = (
        int(env[key]) for key in ("PREFILL_NUM_USERS", "PREFILL_NUM_LAYERS", "PREFILL_CHUNK_SIZE")
    )
    if int(os.environ.get("PREFILL_NUM_USERS", users)) != users:
        raise ValueError(f"runner acceptance requires the model's {users} slots")
    env.update(
        PREFILL_MAX_SEQ_LEN=str(capacity),
        PREFILL_LAYER_ACK_D2H="0",
        PREFILL_USE_TRACE="0",
        PREFILL_KV_ONLY_LAST_LAYER="0",
        PREFILL_STANDALONE_CHUNKED_PCC="0.99",
    )
    return {
        "users": users,
        "layers": layers,
        "max_seq_len": capacity,
        "layer_ack_d2h": "0",
        "env": env,
        "producer": {
            "PREFILL_PRODUCER_CHUNKS": str(capacity // chunk_size),
            "PREFILL_PRODUCER_MAX_REQUESTS": str(users),
            "PREFILL_PRODUCER_DURATION_S": "inf",
            "PREFILL_PRODUCER_WARMUP_CHUNKS": "0",
            "PREFILL_PRODUCER_MULTI_TURN_PROB": "0",
            "PREFILL_PCC_GOLDEN_LEN": str(capacity),
            "PREFILL_PRODUCER_INTERLEAVE": "round_robin",
            "PREFILL_PRODUCER_P_GAP": "0",
            "PREFILL_PRODUCER_P_BURST": "0",
            "PREFILL_SEND_SHUTDOWN": "1",
        },
    }


def validate_prefill_slot_traces(spec, scenario):
    """Require distinct complete goldens so a crossed slot mapping cannot pass on identical data."""
    paths = [Path(path.strip()) for path in spec.split(",") if path.strip()]
    if len(paths) != scenario["users"]:
        raise ValueError(f"set {scenario['users']} distinct golden trace directories in PREFILL_PRODUCER_SLOT_TRACES")
    ids = [json.loads((path / "metadata.json").read_text())["token_ids"] for path in paths]
    if any(len(tokens) != scenario["max_seq_len"] for tokens in ids) or len({tuple(tokens) for tokens in ids}) != len(
        ids
    ):
        raise ValueError(f"goldens must contain distinct {scenario['max_seq_len']}-token prompts")
    for path in paths:
        for layer in range(scenario["layers"]):
            if not (path / "kv_cache" / f"layer_{layer}.safetensors").is_file():
                raise ValueError(f"missing layer {layer} under {path}")
