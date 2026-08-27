# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MTEB STS evaluation for HF/CPU and the optimized BGE-M3 DP2 path.

The TT model always runs B12/S8192 on one N300 (DP=2, B6/device), captures one
trace, and replays it for every batch. Short final batches are padded with empty
requests; those synthetic rows are removed before MTEB scoring. The model's CLS
slice is captured in the trace, so only [B, 1, 1, D] is copied back to the host.

The script needs the mteb package. Install it once into the tt-metal
python_env from the tt-metal root:

    uv pip install --python python_env/bin/python mteb

Examples from the tt-metal root:

    # Ten examples per dataset, both HF and TT:
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/mteb_eval_minimal.py \
        --smoke-samples 10 --output-dir mteb_eval_results/smoke

    # Full STSBenchmark and SICK-R evaluation:
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/mteb_eval_minimal.py \
        --output-dir mteb_eval_results/full
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

try:
    import mteb
    from mteb.models import ModelMeta
except ImportError as exc:  # pragma: no cover - depends on the local install
    raise ImportError(
        "This script needs the mteb package, and the import failed.\n"
        "Install it into the tt-metal python_env from the tt-metal root:\n"
        "    uv pip install --python python_env/bin/python mteb"
    ) from exc

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

import ttnn

MODEL_NAME = "BAAI/bge-m3"
TASKS = ["STSBenchmark", "SICK-R"]
BATCH_SIZE = 12
SEQ_LEN = 8192
MESH_SHAPE = (2, 1)


def _prepare_torch_inputs(tokenizer, texts: list[str], pad_token_id: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))
    nonpad = input_ids.ne(pad_token_id).to(torch.long)
    position_ids = torch.cumsum(nonpad, dim=1) * nonpad + pad_token_id
    return {
        "input_ids": input_ids,
        # Compact uint32 [B,1] valid lengths; custom SDPA expands only mask tiles in L1.
        "attention_mask": encoded["attention_mask"].sum(dim=1, keepdim=True),
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
    }


def _to_batch_sharded(inputs: dict[str, torch.Tensor], mesh_device, *, device: bool) -> dict[str, ttnn.Tensor]:
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    kwargs = {"mesh_mapper": mapper}
    if device:
        kwargs.update(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return {
        key: ttnn.from_torch(value.int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, **kwargs)
        for key, value in inputs.items()
    }


def _copy_inputs(host_inputs: dict[str, ttnn.Tensor], device_inputs: dict[str, ttnn.Tensor]) -> None:
    for key in host_inputs:
        ttnn.copy_host_to_device_tensor(host_inputs[key], device_inputs[key])


def _normalize_cls(cls_output: torch.Tensor) -> torch.Tensor:
    cls_output = cls_output.reshape(cls_output.shape[0], -1, cls_output.shape[-1])[:, 0, :].to(torch.float32)
    return F.normalize(cls_output, p=2, dim=-1)


class TTDP2Embedder:
    """MTEB adapter for fixed B12/S8192 DP2 trace replay."""

    def __init__(self, mesh_device):
        from models.demos.wormhole.bge_m3.tt.common import create_tt_model

        self.mesh_device = mesh_device
        logger.info(f"Loading TT model: {MODEL_NAME} (DP=2, B{BATCH_SIZE}, S{SEQ_LEN})")
        self.model_args, self.model, _ = create_tt_model(
            mesh_device=mesh_device,
            max_batch_size=BATCH_SIZE,
            max_seq_len=SEQ_LEN,
            dtype=ttnn.bfloat8_b,
            hf_model_name=MODEL_NAME,
            data_parallel=True,
            pooling="cls",
        )
        assert self.model._data_parallel, "DP=2 model was not activated"
        self.pad_token_id = int(self.model_args.pad_token_id)
        self.composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=MESH_SHAPE)
        self._mteb_meta = ModelMeta.create_empty(overwrites={"name": f"tt-dp2-{MODEL_NAME}", "revision": None})
        self._build_inputs_and_capture()
        logger.info("TT model ready; B12/S8192 trace captured")

    @property
    def mteb_model_meta(self):
        return self._mteb_meta

    def _build_inputs_and_capture(self) -> None:
        torch_inputs = _prepare_torch_inputs(
            self.model_args.tokenizer,
            ["warmup sentence"] * BATCH_SIZE,
            self.pad_token_id,
        )
        self.host_inputs = _to_batch_sharded(torch_inputs, self.mesh_device, device=False)
        self.device_inputs = _to_batch_sharded(torch_inputs, self.mesh_device, device=True)

        logger.info("Compiling DP2 forward")
        warmup_output = self.model.forward(**self.device_inputs)
        ttnn.synchronize_device(self.mesh_device)
        ttnn.deallocate(warmup_output)

        logger.info("Capturing DP2 trace")
        self.output_dev = self.model.capture_trace(**self.device_inputs, mesh_device=self.mesh_device, cq_id=0)

    def _update_inputs(self, texts: list[str]) -> None:
        torch_inputs = _prepare_torch_inputs(self.model_args.tokenizer, texts, self.pad_token_id)
        host_inputs = _to_batch_sharded(torch_inputs, self.mesh_device, device=False)
        _copy_inputs(host_inputs, self.device_inputs)

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
        all_texts: list[str] = []
        for batch in inputs:
            all_texts.extend(batch["text"])

        all_embeddings = []
        num_batches = math.ceil(len(all_texts) / BATCH_SIZE)
        for start in tqdm(range(0, len(all_texts), BATCH_SIZE), total=num_batches, desc="TT DP2 encode"):
            batch_texts = list(all_texts[start : start + BATCH_SIZE])
            actual_batch_size = len(batch_texts)
            batch_texts.extend([""] * (BATCH_SIZE - actual_batch_size))

            self._update_inputs(batch_texts)
            self.model.execute_trace(blocking=True)
            cls_output = ttnn.to_torch(self.output_dev, mesh_composer=self.composer)
            cls_output = cls_output[:actual_batch_size]
            if not torch.isfinite(cls_output).all():
                raise RuntimeError("TT embedding output contains non-finite values")
            all_embeddings.append(_normalize_cls(cls_output).cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def similarity(self, embeddings1, embeddings2):
        embeddings1 = torch.from_numpy(embeddings1) if isinstance(embeddings1, np.ndarray) else embeddings1
        embeddings2 = torch.from_numpy(embeddings2) if isinstance(embeddings2, np.ndarray) else embeddings2
        return torch.mm(embeddings1, embeddings2.t())

    def similarity_pairwise(self, embeddings1, embeddings2):
        embeddings1 = torch.from_numpy(embeddings1) if isinstance(embeddings1, np.ndarray) else embeddings1
        embeddings2 = torch.from_numpy(embeddings2) if isinstance(embeddings2, np.ndarray) else embeddings2
        return (embeddings1 * embeddings2).sum(dim=1)

    def release(self) -> None:
        self.model.release_trace()


def _limit_tasks(tasks, sample_limit: int | None) -> None:
    if sample_limit is None:
        return
    for task in tasks:
        task.load_data()
        if "test" not in task.dataset:
            raise RuntimeError(f"{task.metadata.name} has no test split")
        count = min(sample_limit, len(task.dataset["test"]))
        task.dataset["test"] = task.dataset["test"].select(range(count))
        logger.info(f"Smoke subset: {task.metadata.name} test={count}")


def run_eval(model, task_names: list[str], output_dir: Path, label: str, sample_limit: int | None) -> dict[str, float]:
    tasks = mteb.get_tasks(tasks=task_names)
    _limit_tasks(tasks, sample_limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{label}] Running: {[task.metadata.name for task in tasks]}")

    results = mteb.MTEB(tasks=tasks).run(
        model,
        output_folder=str(output_dir),
        eval_splits=["test"],
        overwrite_results=True,
        encode_kwargs={"batch_size": BATCH_SIZE, "show_progress_bar": True},
    )

    parsed: dict[str, float] = {}
    for task_result in results:
        main_score = None
        for split_scores in task_result.scores.values():
            for subset_scores in split_scores:
                if main_score is None:
                    main_score = subset_scores.get("main_score")
        if main_score is not None:
            parsed[task_result.task_name] = float(main_score)
            logger.info(f"[{label}] {task_result.task_name}: {main_score:.6f}")

    with open(output_dir / "scores_summary.json", "w") as f:
        json.dump(parsed, f, indent=2)
    return parsed


def _save_comparison(output_base: Path, hf_results: dict[str, float], tt_results: dict[str, float]) -> None:
    comparison = {}
    for task in sorted(set(hf_results) | set(tt_results)):
        hf_score = hf_results.get(task)
        tt_score = tt_results.get(task)
        delta = tt_score - hf_score if hf_score is not None and tt_score is not None else None
        relative_delta_pct = 100.0 * delta / hf_score if delta is not None and hf_score else None
        comparison[task] = {
            "hf": hf_score,
            "tt": tt_score,
            "delta": delta,
            "relative_delta_pct": relative_delta_pct,
        }
        if hf_score is not None and tt_score is not None:
            logger.info(
                f"[COMPARE] {task}: HF={hf_score:.6f} TT={tt_score:.6f} "
                f"delta={delta:+.6f} ({relative_delta_pct:+.2f}%)"
            )
        else:
            logger.info(f"[COMPARE] {task}: HF={hf_score} TT={tt_score}")

    with open(output_base / "comparison.json", "w") as f:
        json.dump({"hf": hf_results, "tt": tt_results, "comparison": comparison}, f, indent=2)
    logger.info(f"Scoring saved under {output_base}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["all"] + TASKS, default="all")
    parser.add_argument("--mode", choices=["both", "hf", "tt"], default="both")
    parser.add_argument("--smoke-samples", type=int, default=None, help="Limit each test split for a smoke run")
    parser.add_argument("--output-dir", default="./mteb_eval_results")
    args = parser.parse_args()

    if args.smoke_samples is not None and args.smoke_samples <= 0:
        parser.error("--smoke-samples must be positive")
    task_names = TASKS if args.task == "all" else [args.task]
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    hf_results: dict[str, float] = {}
    tt_results: dict[str, float] = {}

    if args.mode in ("both", "hf"):
        logger.info("Loading HF reference model on CPU")
        hf_model = mteb.get_model(MODEL_NAME, device="cpu")
        hf_results = run_eval(hf_model, task_names, output_base / "hf", "HF/CPU", args.smoke_samples)
        del hf_model
        gc.collect()

    if args.mode in ("both", "tt"):
        logger.info("Opening one N300 as a 2x1 mesh")
        mesh_device = None
        try:
            mesh_device = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
                trace_region_size=50_000_000,
                num_command_queues=1,
            )
            tt_model = TTDP2Embedder(mesh_device)
            tt_results = run_eval(tt_model, task_names, output_base / "tt", "TT/DP2", args.smoke_samples)
            tt_model.release()
        finally:
            if mesh_device is not None:
                ttnn.close_mesh_device(mesh_device)

    _save_comparison(output_base, hf_results, tt_results)


if __name__ == "__main__":
    main()
