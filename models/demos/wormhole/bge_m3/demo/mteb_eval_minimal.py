# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal MTEB evaluation: HF reference vs TT create_tt_model side-by-side.

Runs STSBenchmark from the official MTEB(eng, v1) benchmark and compares:
  1. HuggingFace model (via mteb.get_model / sentence-transformers) — gold reference
  2. TT model via create_tt_model + trace capture — our on-device implementation

Usage (from tt-metal root):
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/mteb_eval_minimal.py

    # Custom batch/seq:
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/mteb_eval_minimal.py \
        --batch-size 1 --max-seq-len 512
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    import mteb
    from mteb.models import ModelMeta
except ImportError:
    raise ImportError(
        "mteb is required for this evaluation script.\n"
        "Install with: uv pip install -r models/demos/wormhole/bge_m3/demo/requirements_mteb.txt"
    )
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

import ttnn

MODEL_NAME = "BAAI/bge-m3"
ALL_TASKS = ["STSBenchmark", "SICK-R"]
DEFAULT_TASKS = ALL_TASKS


# ─── TT Embedder using trace capture for repeated forwards ───────────────────


def _pool_and_normalize(last_hidden_state: torch.Tensor) -> torch.Tensor:
    if last_hidden_state.dim() == 4 and last_hidden_state.shape[1] == 1:
        last_hidden_state = last_hidden_state.squeeze(1)
    cls_embeddings = last_hidden_state[:, 0, :].to(torch.float32)
    return F.normalize(cls_embeddings, p=2, dim=-1)


class TTEmbedder:
    """MTEB adapter for the BGE-M3 TT model."""

    def __init__(self, device, batch_size: int = 32, max_seq_len: int = 512):
        from models.common.auto_compose import to_torch_auto_compose
        from models.demos.wormhole.bge_m3.tt.common import create_tt_model

        logger.info(f"Loading TT model: {MODEL_NAME} (batch={batch_size}, seq={max_seq_len})")
        self.device = device
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.to_torch = to_torch_auto_compose

        self.model_args, self.model, _ = create_tt_model(
            mesh_device=device,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            dtype=ttnn.bfloat8_b,
            hf_model_name=MODEL_NAME,
        )

        self._mteb_meta = ModelMeta.create_empty(overwrites={"name": f"tt-{MODEL_NAME}", "revision": None})
        self._build_inputs_and_capture()
        logger.info("TT model ready (trace captured).")

    @property
    def mteb_model_meta(self):
        return self._mteb_meta

    def _build_inputs_and_capture(self):
        warmup_texts = ["warmup sentence"] * self.batch_size
        encoded = self.model_args.encode_prompts(warmup_texts, prompt_length=self.max_seq_len)
        staged = encoded["model_inputs"]
        self.input_ids_dev = staged["input_ids"]
        self.attention_mask_dev = staged["attention_mask"]
        self.token_type_ids_dev = staged["token_type_ids"]
        self.position_ids_dev = staged["position_ids"]

        logger.info("Running warmup forward...")
        warmup_output = self.model(**staged)
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(warmup_output)

        logger.info("Capturing trace...")
        self.output_dev = self.model.capture_trace(
            input_ids=self.input_ids_dev,
            attention_mask=self.attention_mask_dev,
            token_type_ids=self.token_type_ids_dev,
            position_ids=self.position_ids_dev,
            mesh_device=self.device,
            cq_id=0,
        )

    def _update_inputs(self, texts: list[str]):
        encoded = self.model_args.encode_prompts(texts, prompt_length=self.max_seq_len)

        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(encoded["input_ids"].int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            self.input_ids_dev,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                encoded["attention_mask"].bfloat16(),
                dtype=self.model_args.attention_mask_dtype,
                layout=ttnn.TILE_LAYOUT,
            ),
            self.attention_mask_dev,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(encoded["token_type_ids"].int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            self.token_type_ids_dev,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(encoded["position_ids"].int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            self.position_ids_dev,
        )

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
        all_texts = []
        for batch in inputs:
            all_texts.extend(batch["text"])

        num_batches = math.ceil(len(all_texts) / self.batch_size)
        all_embeddings = []
        for start in tqdm(range(0, len(all_texts), self.batch_size), total=num_batches, desc="TT encode"):
            batch_texts = all_texts[start : start + self.batch_size]
            actual_batch_size = len(batch_texts)

            while len(batch_texts) < self.batch_size:
                batch_texts.append("")

            self._update_inputs(batch_texts)
            self.model.execute_trace(blocking=True)

            hidden_states = self.to_torch(self.output_dev, device=self.device)
            hidden_states = hidden_states[:actual_batch_size]
            normalized = _pool_and_normalize(hidden_states)
            all_embeddings.append(normalized.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def similarity(self, embeddings1, embeddings2):
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        return torch.mm(embeddings1, embeddings2.t())

    def similarity_pairwise(self, embeddings1, embeddings2):
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        return (embeddings1 * embeddings2).sum(dim=1)

    def release(self):
        self.model.release_trace()


# ─── Evaluation runner ────────────────────────────────────────────────────────


def run_eval(model, task_names: list[str], output_dir: str, label: str) -> dict:
    tasks = mteb.get_tasks(tasks=task_names)
    if not tasks:
        logger.error(f"No tasks found for: {task_names}")
        return {}

    logger.info(f"[{label}] Running MTEB on {len(tasks)} tasks: {[t.metadata.name for t in tasks]}")
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=output_dir, eval_splits=["test"], overwrite_results=True)

    parsed = {}
    for task_result in results:
        task_name = task_result.task_name
        main_score = None
        for split_name, split_scores in task_result.scores.items():
            for subset_scores in split_scores:
                if main_score is None:
                    main_score = subset_scores.get("main_score")
        parsed[task_name] = main_score
        logger.info(f"[{label}] {task_name}: {main_score:.4f}" if main_score else f"[{label}] {task_name}: N/A")

    return parsed


def print_comparison(hf_results: dict, tt_results: dict):
    all_tasks = sorted(set(list(hf_results.keys()) + list(tt_results.keys())))
    if not all_tasks:
        return

    header = f"{'Task':<30s} {'HF Score':>10s} {'TT Score':>10s} {'Delta':>10s}"
    sep = "-" * len(header)
    lines = [sep, "MTEB Evaluation: HF vs TT Comparison", sep, header, sep]

    hf_scores, tt_scores, deltas = [], [], []
    for task in all_tasks:
        hf_s = hf_results.get(task)
        tt_s = tt_results.get(task)
        hf_str = f"{hf_s:.4f}" if hf_s is not None else "N/A"
        tt_str = f"{tt_s:.4f}" if tt_s is not None else "N/A"
        if hf_s is not None and tt_s is not None:
            d = tt_s - hf_s
            delta_str = f"{d:+.4f}"
            deltas.append(d)
            hf_scores.append(hf_s)
            tt_scores.append(tt_s)
        else:
            delta_str = "N/A"
        lines.append(f"{task:<30s} {hf_str:>10s} {tt_str:>10s} {delta_str:>10s}")

    lines.append(sep)
    if hf_scores and tt_scores:
        hf_avg = sum(hf_scores) / len(hf_scores)
        tt_avg = sum(tt_scores) / len(tt_scores)
        avg_d = sum(deltas) / len(deltas)
        lines.append(f"{'Average':<30s} {hf_avg:>10.4f} {tt_avg:>10.4f} {avg_d:>+10.4f}")
    lines.append(sep)

    for line in lines:
        logger.info(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task",
        choices=["all", "STSBenchmark", "SICK-R"],
        default="STSBenchmark",
        help="Which MTEB task to run (default: STSBenchmark)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="TT model batch size")
    parser.add_argument("--max-seq-len", type=int, default=512, help="TT model max sequence length")
    parser.add_argument("--output-dir", default="./mteb_eval_results", help="Output directory")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID")
    args = parser.parse_args()

    tasks = ALL_TASKS if args.task == "all" else [args.task]

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # ── HF ──
    logger.info("=" * 60)
    logger.info("Running HF reference evaluation")
    logger.info("=" * 60)
    hf_model = mteb.get_model(MODEL_NAME)
    hf_results = run_eval(hf_model, tasks, str(output_base / "hf"), "HF")
    del hf_model

    logger.info("=" * 60)
    logger.info("Running TT evaluation")
    logger.info("=" * 60)
    device = ttnn.open_device(
        device_id=args.device_id,
        trace_region_size=50_000_000,
        num_command_queues=1,
    )
    try:
        tt_model = TTEmbedder(device=device, batch_size=args.batch_size, max_seq_len=args.max_seq_len)
        tt_results = run_eval(tt_model, tasks, str(output_base / "tt"), "TT")
        tt_model.release()
    finally:
        ttnn.close_device(device)

    # ── Compare ──
    print_comparison(hf_results, tt_results)

    results_file = output_base / "comparison.json"
    with open(results_file, "w") as f:
        json.dump({"hf": hf_results, "tt": tt_results}, f, indent=2)
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
