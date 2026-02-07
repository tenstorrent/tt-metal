# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any

import torch

import ttnn

# Make `import models.*` work when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite.tt.layer0_tt import _alloc_contiguous_page_table, _alloc_paged_kvpe_cache, _round_up
from models.demos.glm4_moe_lite.tt.model_tt import Glm4MoeLiteDenseOnlyTT
from models.demos.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir


def _default_golden_path(snapshot_dir: Path, *, num_layers: int, max_new_tokens: int) -> Path:
    snap_name = Path(snapshot_dir).name
    root = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/golden"))
    return root / f"golden_tokens_{snap_name}_layers{num_layers}_new{max_new_tokens}.json"


def _load_hparams(snapshot_dir: Path) -> Glm4MoeLiteHParams:
    cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()
    return hparams


@dataclass
class _StepStat:
    prompt: str
    step_label: str
    expected_token: int
    pred_token: int
    expected_logit: float
    pred_logit: float
    gap: float
    expected_rank: int  # 1-based


def _analyze_step(*, logits: torch.Tensor, expected_token_id: int, topk: int) -> tuple[int, float, int, float, float]:
    # Some paths return [1,1,1,V] (extra singleton sequence axis). Normalize.
    if logits.ndim == 4 and int(logits.shape[2]) == 1:
        logits = logits.squeeze(2)
    if logits.ndim != 3 or int(logits.shape[0]) != 1 or int(logits.shape[1]) != 1:
        raise ValueError(f"expected logits shape [1,1,V], got {tuple(logits.shape)}")

    log = logits[0, 0].to(dtype=torch.float32)
    pred_token_id = int(log.argmax().item())

    expected_token_id = int(expected_token_id)
    expected_logit = float(log[expected_token_id].item())
    pred_logit = float(log[pred_token_id].item())
    gap = float(pred_logit - expected_logit)

    # Rank of expected token (1-based). For long vocab this is expensive; approximate
    # rank by comparing to top-k, then fall back to full sort if not in top-k.
    k = min(int(topk), int(log.numel()))
    topv, topi = torch.topk(log, k=k)
    topi_list = [int(x) for x in topi.tolist()]
    if expected_token_id in set(topi_list):
        expected_rank = topi_list.index(expected_token_id) + 1
    else:
        # Exact rank: count how many logits are strictly greater.
        expected_rank = int((log > expected_logit).sum().item()) + 1

    return pred_token_id, expected_logit, expected_rank, pred_logit, gap


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze TT vs offline golden next-token near-ties for truncated GLM-4.7-Flash.")
    p.add_argument("--model-id", default="zai-org/GLM-4.7-Flash")
    p.add_argument("--snapshot-dir", type=Path, default=None)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--golden", type=Path, default=None, help="Golden JSON path (optional).")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--topk", type=int, default=10, help="Top-k used for rank approximation and summary bins.")
    p.add_argument("--max-records", type=int, default=8)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="Optional JSON output path for detailed stats.")
    p.add_argument("--moe-fp32-acc", choices=["0", "1"], default="1")
    args = p.parse_args()

    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir is not None else Path(resolve_best_effort_snapshot_dir(args.model_id))
    missing = find_missing_shards(snapshot_dir)
    if missing:
        print(f"ERROR: snapshot missing {len(missing)} shards (example: {missing[0]}).", file=sys.stderr)
        return 2

    num_layers = int(args.num_layers)
    max_new_tokens = int(args.max_new_tokens)
    golden_path = Path(args.golden) if args.golden is not None else _default_golden_path(
        snapshot_dir, num_layers=num_layers, max_new_tokens=max_new_tokens
    )
    if not golden_path.is_file():
        print(f"ERROR: golden file not found: {golden_path}", file=sys.stderr)
        print("Generate it with:", file=sys.stderr)
        print(
            f"  python {Path(__file__).resolve().with_name('generate_truncated_golden_tokens.py')} "
            f"--snapshot-dir {snapshot_dir} --num-layers {num_layers} --max-new-tokens {max_new_tokens}",
            file=sys.stderr,
        )
        return 2

    golden = json.loads(golden_path.read_text())
    records = list(golden.get("records", []))[: max(0, int(args.max_records))]
    if not records:
        print("ERROR: golden file has no records", file=sys.stderr)
        return 2

    os.environ["GLM4_MOE_LITE_ENABLE_MOE"] = "1"
    os.environ["GLM4_MOE_LITE_NUM_LAYERS"] = str(num_layers)
    os.environ["GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS"] = "1"
    os.environ["GLM4_MOE_LITE_EXPERTS_TT_DTYPE"] = "bf16"
    os.environ["GLM4_MOE_LITE_MOE_FP32_ACC"] = str(args.moe_fp32_acc)

    hparams = _load_hparams(snapshot_dir)
    kvpe_dim = int(hparams.kv_lora_rank + hparams.qk_rope_head_dim)

    topk = max(1, int(args.topk))
    out_path = Path(args.out) if args.out is not None else None
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir is not None
        else Path(os.path.expanduser(f"~/.cache/ttnn/models/glm4_moe_lite/near_tie_analysis_cache_layers{num_layers}"))
    )

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[int(args.device_id)],
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )

    step_stats: list[_StepStat] = []
    try:
        runner = Glm4MoeLiteDenseOnlyTT.create(
            device=mesh_device,
            snapshot_dir=snapshot_dir,
            cache_dir=cache_dir,
            max_seq_len=2048,
            hparams=hparams,
        )

        for rec_i, rec in enumerate(records):
            prompt = str(rec.get("prompt", f"<record{rec_i}>"))
            prompt_ids = torch.tensor([rec["prompt_input_ids"]], dtype=torch.int32)
            expected_ids = list(rec["generated_ids"])
            if len(expected_ids) < max_new_tokens:
                raise ValueError(f"golden record {rec_i} has only {len(expected_ids)} generated ids; expected {max_new_tokens}")

            block_size = 64
            prompt_len = int(prompt_ids.shape[1])
            total_len = prompt_len + int(max_new_tokens)
            blocks_per_seq = max(1, _round_up(total_len, block_size) // block_size)
            blocks_per_seq = max(blocks_per_seq, _round_up(128, block_size) // block_size)

            kv_cache = [
                _alloc_paged_kvpe_cache(
                    device=mesh_device,
                    max_num_blocks=int(1 * blocks_per_seq),
                    block_size=block_size,
                    kvpe_dim=kvpe_dim,
                    dtype=ttnn.bfloat16,
                )
                for _ in range(num_layers)
            ]
            page_table = _alloc_contiguous_page_table(batch=1, blocks_per_seq=blocks_per_seq)

            try:
                logits = runner.prefill(
                    tokens=prompt_ids,
                    prompt_lens=[prompt_len],
                    page_table=page_table,
                    kv_cache=kv_cache,
                    seq_pad_multiple=block_size,
                )
                pred, exp_log, exp_rank, pred_log, gap = _analyze_step(
                    logits=logits, expected_token_id=int(expected_ids[0]), topk=topk
                )
                step_stats.append(
                    _StepStat(
                        prompt=prompt,
                        step_label="prefill",
                        expected_token=int(expected_ids[0]),
                        pred_token=int(pred),
                        expected_logit=float(exp_log),
                        pred_logit=float(pred_log),
                        gap=float(gap),
                        expected_rank=int(exp_rank),
                    )
                )

                token_in = int(expected_ids[0])
                for step in range(max_new_tokens - 1):
                    start_pos = torch.tensor([prompt_len + step], dtype=torch.int32)
                    tokens = torch.tensor([[token_in]], dtype=torch.int32)
                    logits = runner.decode(tokens=tokens, start_pos=start_pos, page_table=page_table, kv_cache=kv_cache)
                    pred, exp_log, exp_rank, pred_log, gap = _analyze_step(
                        logits=logits, expected_token_id=int(expected_ids[step + 1]), topk=topk
                    )
                    step_stats.append(
                        _StepStat(
                            prompt=prompt,
                            step_label=f"decode_step{step}",
                            expected_token=int(expected_ids[step + 1]),
                            pred_token=int(pred),
                            expected_logit=float(exp_log),
                            pred_logit=float(pred_log),
                            gap=float(gap),
                            expected_rank=int(exp_rank),
                        )
                    )
                    token_in = int(expected_ids[step + 1])
            finally:
                try:
                    for t in kv_cache:
                        ttnn.deallocate(t)
                except Exception:
                    pass
    finally:
        ttnn.close_mesh_device(mesh_device)

    total = len(step_stats)
    exact = sum(1 for s in step_stats if int(s.pred_token) == int(s.expected_token))
    mism = total - exact

    mism_gaps = [float(s.gap) for s in step_stats if int(s.pred_token) != int(s.expected_token)]
    mism_ranks = [int(s.expected_rank) for s in step_stats if int(s.pred_token) != int(s.expected_token)]
    max_gap = max(mism_gaps) if mism_gaps else 0.0
    p50_gap = statistics.median(mism_gaps) if mism_gaps else 0.0
    p90_gap = statistics.quantiles(mism_gaps, n=10)[8] if len(mism_gaps) >= 10 else (max_gap if mism_gaps else 0.0)

    print("")
    print("=== Near-tie analysis summary ===")
    print(f"model:        {args.model_id}")
    print(f"snapshot_dir: {snapshot_dir}")
    print(f"golden:       {golden_path}")
    print(f"layers:       {num_layers}")
    print(f"new_tokens:   {max_new_tokens}")
    print(f"records:      {len(records)}")
    print(f"steps:        {total}")
    print(f"exact_match:  {exact}/{total} ({(exact / total) * 100.0:.2f}%)")
    print(f"mismatches:   {mism}")
    if mism:
        print(f"gap(p50):     {p50_gap:.4f}")
        print(f"gap(p90):     {p90_gap:.4f}")
        print(f"gap(max):     {max_gap:.4f}")
        print(f"rank(min):    {min(mism_ranks)}")
        print(f"rank(max):    {max(mism_ranks)}")

        # Small actionable list of the worst offenders.
        worst = sorted([s for s in step_stats if s.pred_token != s.expected_token], key=lambda s: s.gap, reverse=True)[:10]
        print("")
        print("Top mismatches by logit gap (pred - expected):")
        for s in worst:
            print(
                f"- {s.step_label}: gap={s.gap:.4f} rank={s.expected_rank:4d} "
                f"pred={s.pred_token} expected={s.expected_token} prompt={s.prompt[:80]!r}"
            )

        print("")
        print("Suggested guardrail for tests (empirical):")
        print(f"- max_logit_gap >= {max_gap:.4f} (consider rounding up to 0.25/0.5/0.75)")
        print(f"- expected_rank <= {max(mism_ranks)} (use topk >= {max(mism_ranks)})")
    else:
        print("All steps exactly match the offline greedy tokens.")

    if out_path is not None:
        payload: dict[str, Any] = {
            "model_id": args.model_id,
            "snapshot_dir": str(snapshot_dir),
            "golden_path": str(golden_path),
            "num_layers": num_layers,
            "max_new_tokens": max_new_tokens,
            "records": len(records),
            "total_steps": total,
            "exact_matches": exact,
            "mismatches": mism,
            "mismatch_gap_p50": p50_gap,
            "mismatch_gap_p90": p90_gap,
            "mismatch_gap_max": max_gap,
            "mismatch_rank_min": (min(mism_ranks) if mism_ranks else None),
            "mismatch_rank_max": (max(mism_ranks) if mism_ranks else None),
            "steps": [s.__dict__ for s in step_stats],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nWrote detailed stats: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

