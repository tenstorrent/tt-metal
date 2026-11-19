# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Uniform CPU FP64-rounded indexer projection control over a saved QSA context.

Only index_qk_proj accumulation changes: real BF16 operands are accumulated in
FP64 and rounded to BF16. All HF norm/pooling/RoPE, selection and attention
operations stay unchanged. This is an independent backend diagnostic, never a
replacement for literal-HF acceptance. No TTNN import or device access occurs.

Literal attention outputs may be reused only when sorted selected-token sets
are identical: the indexer does not alter attention query/gate/K/V operands.
Every changed-set row recomputes both literal and alternate attention using
original 32-row projection shapes and requires literal output bit equality.
The original selected-ID digest must match the full saved reference digest.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .probe_qsa import load_reference


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_tensors(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def comparison(expected, actual):
    """FP64 per-row PCC and stable merged centered global covariance."""
    expected, actual = expected.reshape(-1, 2560), actual.reshape(-1, 2560)
    values, total = [], 0
    mean_x = mean_y = xx = yy = xy = 0.0
    for start in range(0, len(expected), 1024):
        x, y = expected[start : start + 1024].double(), actual[start : start + 1024].double()
        assert bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        xc, yc = x - x.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
        denominator = (xc.square().sum(-1) * yc.square().sum(-1)).sqrt()
        row = (xc * yc).sum(-1) / denominator
        row = torch.where(denominator == 0, (x == y).all(-1).double(), row).clamp(-1, 1)
        assert bool(torch.isfinite(row).all())
        values.append(row)
        n = x.numel()
        mx, my = float(x.mean()), float(y.mean())
        dx, dy = mx - mean_x, my - mean_y
        factor = total * n / (total + n)
        xc, yc = x - mx, y - my
        xx += float(xc.square().sum()) + dx * dx * factor
        yy += float(yc.square().sum()) + dy * dy * factor
        xy += float((xc * yc).sum()) + dx * dy * factor
        mean_x += dx * n / (total + n)
        mean_y += dy * n / (total + n)
        total += n
    rows = torch.cat(values)
    order = rows.argsort()[:10]
    failed = torch.where(rows < 0.995)[0].tolist()
    return {
        "global_pcc_fp64": xy / math.sqrt(xx * yy),
        "minimum_row_pcc_fp64": float(rows.min()),
        "failed_row_count": len(failed),
        "failed_rows": failed,
        "rows": len(rows),
        "worst_rows": [{"row": int(index), "pcc": float(rows[index])} for index in order],
    }, rows


@torch.inference_mode()
def prepare_projection_cache(args, model, rope, source, identity):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    if args.projection_cache.exists():
        saved = torch.load(args.projection_cache, weights_only=True, mmap=True, map_location="cpu")
        assert saved["identity"] == identity
        print(json.dumps({"projection_cache_hit": str(args.projection_cache)}), flush=True)
        return saved
    length = source["key"]["length"]
    saved = {
        "identity": identity,
        "queries": {name: torch.empty(1, 4, length, 128, dtype=torch.bfloat16) for name in ("literal", "fp64")},
        "compressed": {name: torch.empty(1, length // 4, 128, dtype=torch.bfloat16) for name in ("literal", "fp64")},
        "keys": torch.empty(1, 2, length, 256, dtype=torch.bfloat16),
        "values": torch.empty(1, 2, length, 256, dtype=torch.bfloat16),
        "projection_changed_elements": 0,
        "projection_changed_rows": 0,
    }
    histories = {name: torch.zeros(1, 3, 128, dtype=torch.bfloat16) for name in ("literal", "fp64")}
    weight64 = model.indexer.index_qk_proj.weight.double()
    begin = time.monotonic()
    for start in range(0, length, 32):
        end = start + 32
        hidden = source["hidden"][:, start:end]
        positions = torch.arange(start, end).reshape(1, -1)
        cos, sin = rope(hidden, positions)
        gc, gs = rope(hidden, positions.div(4, rounding_mode="floor") * 4)
        literal = model.indexer.index_qk_proj(hidden)
        alternate = F.linear(hidden.double(), weight64).bfloat16()
        different = literal != alternate
        saved["projection_changed_elements"] += int(different.sum())
        saved["projection_changed_rows"] += int(different.any(-1).sum())
        for name, projection in (("literal", literal), ("fp64", alternate)):
            query = model.indexer.q_layernorm(projection[..., :512].reshape(1, 32, 4, 128))
            query = apply_rotary_pos_emb(query.transpose(1, 2), cos=cos, sin=sin)
            saved["queries"][name][:, :, start:end] = query
            history = torch.cat((histories[name], projection[..., 512:]), 1)
            pooled = history.float().unfold(1, 4, 1).mean(-1).bfloat16()
            pooled = model.indexer.k_layernorm(pooled)
            pooled = apply_rotary_pos_emb(pooled.unsqueeze(1), cos=gc, sin=gs).squeeze(1)
            saved["compressed"][name][:, start // 4 : end // 4] = pooled[:, 3::4]
            histories[name] = history[:, -3:].clone()
        key = model.k_norm(model.k_proj(hidden).reshape(1, 32, 2, 256)).transpose(1, 2)
        saved["keys"][:, :, start:end] = apply_rotary_pos_emb(key, cos=cos, sin=sin)
        saved["values"][:, :, start:end] = model.v_proj(hidden).reshape(1, 32, 2, 256).transpose(1, 2)
        if end % 16384 == 0 or end == length:
            print(json.dumps({"projection_rows": end, "seconds": time.monotonic() - begin}), flush=True)
    save_tensors(args.projection_cache, saved)
    return saved


@torch.inference_mode()
def main():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb, repeat_kv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--device-cache", type=Path, required=True)
    parser.add_argument("--projection-cache", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert "ttnn" not in sys.modules
    torch.set_num_threads(8)
    start_time = time.monotonic()
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    device = torch.load(args.device_cache, weights_only=True, mmap=True, map_location="cpu")
    device_fixture = device["reference_fixture"]
    assert source["key"] == device_fixture.get("source_key", device_fixture["key"])
    if device_fixture["key"] != source["key"]:
        assert device_fixture["boundary_split_reuse"] is True
        assert device_fixture["key"] == {**source["key"], "length": 262143, "decode_steps": 1}
    length = source["key"]["length"]
    assert length == source["key"]["capacity"] == device["length"] == 262144
    assert source["key"]["decode_steps"] == 0 and source["key"]["seed"] == 1979
    _, model, rope, _, provenance = load_reference(args.snapshot)
    assert source["key"]["reference_sha256"] == provenance["reference_sha256"]
    assert source["key"]["config_sha256"] == provenance["config_sha256"]
    assert source["key"]["snapshot"] == str(args.snapshot.resolve())
    identity = {
        "policy": "uniform index_qk_proj FP64 accumulation then BF16, all other HF operations unchanged",
        "source_key": source["key"],
        "device_fixture_metadata": device_fixture,
        "reference_provenance": provenance,
        "harness_sha256": digest(Path(__file__)),
        "reference_fixture_sha256": digest(args.source_cache),
        "device_output_sha256": digest(args.device_cache),
    }
    report = {
        "scope": __doc__,
        "identity": identity,
        "device_executed": False,
        "accepted": False,
        "status": "running",
        "rows": length,
        "changed_set_rows": [],
        "literal_recomputed_rows": 0,
        "literal_recomputed_bitexact": True,
    }
    write_json(args.output, report)
    cache = prepare_projection_cache(args, model, rope, source, identity)
    report.update({key: cache[key] for key in ("projection_changed_elements", "projection_changed_rows")})
    alternate_output = source["expected"].clone()
    selected_digest = hashlib.sha256()
    alternate_digest = hashlib.sha256()
    selection_begin = time.monotonic()

    def attention(query, gate, selected):
        keys = repeat_kv(cache["keys"][:, :, selected], 12)
        values = repeat_kv(cache["values"][:, :, selected], 12)
        logits = torch.matmul(query, keys.transpose(-1, -2)) * (256**-0.5)
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        result = torch.matmul(probabilities, values)
        result *= torch.sigmoid(gate)
        return model.o_proj(result.reshape(1, 1, 6144))

    for start in range(0, length, 32):
        end = start + 32
        scores = {}
        for name in ("literal", "fp64"):
            query = cache["queries"][name][:, :, start:end].float()
            keys = cache["compressed"][name][:, : end // 4].float().transpose(-1, -2).unsqueeze(1)
            score = torch.matmul(query, keys)
            scores[name] = score.relu().sum(1)[0] / math.sqrt(128)
        query = gate = None
        for row, position in enumerate(range(start, end)):
            complete = (position + 1) // 4
            selected = {}
            for name, digest_object in (("literal", selected_digest), ("fp64", alternate_digest)):
                if complete:
                    blocks = scores[name][row, :complete].topk(min(512, complete)).indices
                    ids = (blocks[:, None] * 4 + torch.arange(4)).flatten()
                else:
                    ids = torch.empty(0, dtype=torch.int64)
                ids = torch.cat((ids, torch.arange(complete * 4, position + 1))).sort().values
                digest_object.update(ids.numpy().tobytes())
                selected[name] = ids
            if torch.equal(selected["literal"], selected["fp64"]):
                continue
            report["changed_set_rows"].append(position)
            if query is None:
                hidden = source["hidden"][:, start:end]
                cos, sin = rope(hidden, torch.arange(start, end).reshape(1, -1))
                q, g = model.q_proj(hidden).reshape(1, 32, 24, 512).chunk(2, -1)
                query = apply_rotary_pos_emb(model.q_norm(q).transpose(1, 2), cos=cos, sin=sin)
                gate = g.transpose(1, 2)
            literal = attention(query[:, :, row : row + 1], gate[:, :, row : row + 1], selected["literal"])
            assert torch.equal(
                literal, source["expected"][:, position : position + 1]
            ), f"Unchanged literal attention reconstruction differs at row{position}; output reuse is invalid"
            report["literal_recomputed_rows"] += 1
            alternate_output[:, position : position + 1] = attention(
                query[:, :, row : row + 1], gate[:, :, row : row + 1], selected["fp64"]
            )
        if end % 16384 == 0 or end == length:
            report["completed_rows"] = end
            report["changed_set_count"] = len(report["changed_set_rows"])
            write_json(args.output, report)
            print(
                json.dumps(
                    {
                        "selection_rows": end,
                        "changed_sets": report["changed_set_count"],
                        "selection_seconds": time.monotonic() - selection_begin,
                    }
                ),
                flush=True,
            )
    report["literal_selected_ids_sha256"] = selected_digest.hexdigest()
    report["expected_literal_selected_ids_sha256"] = source["reference_report"]["selected_ids_sha256"]
    report["fp64_selected_ids_sha256"] = alternate_digest.hexdigest()
    report["literal_selection_hash_exact"] = (
        report["literal_selected_ids_sha256"] == report["expected_literal_selected_ids_sha256"]
    )
    assert report["literal_selection_hash_exact"], "Original selected-ID digest does not bind reused outputs"
    report["TT_vs_literal"], literal_rows = comparison(source["expected"], device["actual"])
    report["TT_vs_fp64_indexer"], alternate_rows = comparison(alternate_output, device["actual"])
    report["literal_vs_fp64_indexer"], policy_rows = comparison(source["expected"], alternate_output)
    report["new_TT_failures_under_control"] = torch.where((alternate_rows < 0.995) & (literal_rows >= 0.995))[
        0
    ].tolist()
    report["resolved_literal_failures_under_control"] = torch.where((alternate_rows >= 0.995) & (literal_rows < 0.995))[
        0
    ].tolist()
    report["status"] = "completed_diagnostic"
    report["elapsed_seconds"] = time.monotonic() - start_time
    assert "ttnn" not in sys.modules
    save_tensors(
        args.tensor_output,
        {
            "identity": identity,
            "report": report,
            "fp64_indexer_output": alternate_output,
            "TT_vs_literal_row_pcc": literal_rows,
            "TT_vs_fp64_indexer_row_pcc": alternate_rows,
            "literal_vs_fp64_indexer_row_pcc": policy_rows,
        },
    )
    report["tensor_output"] = str(args.tensor_output)
    write_json(args.output, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("changed_set_rows", "identity", "scope")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
