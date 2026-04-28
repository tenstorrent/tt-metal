# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced inference: run prompt + reference tokens through the pod pipeline and compare predictions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.deepseek_v3_b1.demo.mesh_device_context import open_mesh_device
from models.demos.deepseek_v3_b1.demo.model_pipeline import ModelPipeline

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-0528"


@dataclass
class TeacherForcedResult:
    """Serializable result of teacher-forced inference (next-token prediction vs reference).

    ``predicted_token_ids`` and ``per_position_match`` have the same length (one per step run).
    Each ``predicted_token_ids[i]`` is compared to ``reference_token_ids[i]``. If the run stops
    early because the model predicted EOS, those lists may be shorter than the full effective
    ``reference_token_ids`` (remaining reference positions were not evaluated).
    """

    prompt_text: str
    reference_text: str
    prompt_token_ids: list[int]
    reference_token_ids: list[int]
    predicted_token_ids: list[int]
    per_position_match: list[bool]
    num_correct: int
    num_total: int
    accuracy: float
    eos_token_id: int | None = None
    stopped_early_predicted_eos: bool = False
    #: When set, accuracy over fixed-size slices of ``per_position_match`` (reference positions).
    per_chunk_accuracy: list[dict[str, Any]] | None = None
    chunk_accuracy_token_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict (bools and ints are native JSON types)."""
        d = asdict(self)
        if d.get("per_chunk_accuracy") is None:
            del d["per_chunk_accuracy"]
        if d.get("chunk_accuracy_token_size") is None:
            del d["chunk_accuracy_token_size"]
        return d

    def save_json(self, path: Path | str) -> None:
        """Write result to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def _truncate_reference_at_first_eos(reference_token_ids: list[int], eos_token_id: int | None) -> list[int]:
    """If ``eos_token_id`` is set, keep tokens through the first EOS inclusive; else full list."""
    if eos_token_id is None:
        return list(reference_token_ids)
    for i, tid in enumerate(reference_token_ids):
        if tid == eos_token_id:
            return list(reference_token_ids[: i + 1])
    return list(reference_token_ids)


def compute_per_chunk_accuracy(
    per_position_match: list[bool],
    chunk_size: int,
) -> list[dict[str, Any]]:
    """Slice ``per_position_match`` into chunks of ``chunk_size``; return accuracy per slice."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    out: list[dict[str, Any]] = []
    n = len(per_position_match)
    idx = 0
    chunk_index = 0
    while idx < n:
        end = min(idx + chunk_size, n)
        chunk = per_position_match[idx:end]
        num_c = sum(chunk)
        num_t = len(chunk)
        out.append(
            {
                "chunk_index": chunk_index,
                "start": idx,
                "end_exclusive": end,
                "num_correct": num_c,
                "num_total": num_t,
                "accuracy": float(num_c) / float(num_t) if num_t else 0.0,
            }
        )
        idx = end
        chunk_index += 1
    return out


def run_teacher_forced(
    model_pipeline: ModelPipeline,
    prompt_token_ids: list[int],
    reference_token_ids: list[int],
    *,
    prompt_text: str = "",
    reference_text: str = "",
    eos_token_id: int | None = None,
    chunk_accuracy_token_size: int | None = None,
) -> TeacherForcedResult:
    """
    Prefill on prompt tokens, then teacher-force reference tokens: at each step compare
    the model's predicted next token to the reference token at the same position.

    - After prefill, prediction is compared to ``reference_token_ids[0]``.
    - For ``i = 0 .. len(reference)-2``, ``decode_forward(reference_token_ids[i])``;
      compare output to ``reference_token_ids[i + 1]``.

    If ``eos_token_id`` is not ``None``:
    - Reference is truncated after the first EOS token in the file (inclusive).
    - Stops early when the model predicts EOS (same idea as ``ModelPipeline.run_inference``):
      no further ``decode_forward`` calls after an EOS prediction.

    If ``chunk_accuracy_token_size`` is set (e.g. 256), ``TeacherForcedResult.per_chunk_accuracy``
    lists accuracy over consecutive slices of ``per_position_match`` of that length.
    """
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must be non-empty")
    if not reference_token_ids:
        raise ValueError("reference_token_ids must be non-empty")

    effective_ref = _truncate_reference_at_first_eos(reference_token_ids, eos_token_id)

    predicted_token_ids: list[int] = []
    per_position_match: list[bool] = []
    stopped_early_predicted_eos = False

    prefill_results = model_pipeline.prefill_forward(prompt_token_ids)
    if not prefill_results:
        raise RuntimeError("prefill_forward() returned no DecodeResults")

    pred = prefill_results[0].token_0
    predicted_token_ids.append(pred)
    per_position_match.append(pred == effective_ref[0])
    if eos_token_id is not None and pred == eos_token_id:
        stopped_early_predicted_eos = len(effective_ref) > 1
    else:
        for i in range(len(effective_ref) - 1):
            pred = model_pipeline.decode_forward(effective_ref[i])
            predicted_token_ids.append(pred)
            per_position_match.append(pred == effective_ref[i + 1])
            if eos_token_id is not None and pred == eos_token_id:
                stopped_early_predicted_eos = i + 2 < len(effective_ref)
                break

    num_correct = sum(per_position_match)
    num_total = len(per_position_match)
    accuracy = float(num_correct) / float(num_total) if num_total else 0.0

    per_chunk: list[dict[str, Any]] | None = None
    chunk_sz: int | None = None
    if chunk_accuracy_token_size is not None:
        chunk_sz = chunk_accuracy_token_size
        per_chunk = compute_per_chunk_accuracy(per_position_match, chunk_sz)

    return TeacherForcedResult(
        prompt_text=prompt_text,
        reference_text=reference_text,
        prompt_token_ids=list(prompt_token_ids),
        reference_token_ids=effective_ref,
        predicted_token_ids=predicted_token_ids,
        per_position_match=per_position_match,
        num_correct=num_correct,
        num_total=num_total,
        accuracy=accuracy,
        eos_token_id=eos_token_id,
        stopped_early_predicted_eos=stopped_early_predicted_eos,
        per_chunk_accuracy=per_chunk,
        chunk_accuracy_token_size=chunk_sz,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "DeepSeek-V3-B1 teacher-forced inference on TT-NN (pod pipeline)",
    )
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="User prompt text (chat template applied)")
    parser.add_argument(
        "--reference-file",
        type=Path,
        required=True,
        help="Text file with reference generation (GPU/other); tokenized without chat template",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Write TeacherForcedResult JSON to this path (optional)",
    )
    parser.add_argument(
        "--no-eos-stop",
        action="store_true",
        help="Do not stop at EOS: use full reference text and ignore tokenizer eos_token_id",
    )
    parser.add_argument(
        "--chunk-accuracy",
        type=int,
        nargs="?",
        const=256,
        default=None,
        metavar="N",
        help=(
            "Include per-chunk accuracy in JSON: slice reference positions every N tokens "
            "(default N=256 when flag is passed with no value)"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="HF tokenizer id or local tokenizer path",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Path to the weight cache directory (required for --weights real)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local HuggingFace model dir with model.safetensors.index.json (required for --weights real/state_dict)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=("synthetic", "real", "state_dict"),
        default="real",
        help="synthetic: random prepare path; real: TensorCache + HF safetensors; state_dict: HF safetensors + prepare path (no cache)",
    )
    parser.add_argument(
        "--fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP32 destination accumulator for LMHead sampling",
    )
    parser.add_argument(
        "--persistent-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use persistent mode for LMHead sampling kernel",
    )
    parser.add_argument(
        "--dense-layer-id-override",
        type=int,
        default=None,
        metavar="ID",
        help="Force all dense stages to use this layer id (e.g. 0); default: use 0,1,2",
    )
    parser.add_argument(
        "--moe-layer-id-override",
        type=int,
        default=None,
        metavar="ID",
        help="Force all MoE stages to use this layer id (e.g. 3); default: use stage-dependent layer ids",
    )
    return parser


def load_tokenizer(tokenizer_name_or_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)


def run_teacher_forced_demo(
    *,
    prompt: str,
    reference_text: str,
    tokenizer_name_or_path: str,
    weights_mode: Literal["synthetic", "real", "state_dict"] = "real",
    cache_path: Path | None = None,
    model_path: Path | None = None,
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
    output_file: Path | None = None,
    no_eos_stop: bool = False,
    chunk_accuracy_token_size: int | None = None,
) -> TeacherForcedResult | None:
    """Run teacher-forced inference on mesh id 0; returns result only on mesh 0."""
    logger.info("Starting DeepSeek V3 B1 teacher-forced demo")

    result: TeacherForcedResult | None = None
    with open_mesh_device() as mesh_device:
        model_pipeline = ModelPipeline(
            mesh_device=mesh_device,
            weights_mode=weights_mode,
            cache_path=cache_path,
            model_path=model_path,
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
        )

        my_mesh_id = mesh_device.get_system_mesh_id()
        if my_mesh_id == 0:
            tokenizer = load_tokenizer(tokenizer_name_or_path)
            messages = [{"role": "user", "content": prompt}]
            prompt_with_template = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.debug("Prompt with chat template: {}", prompt_with_template)

            prompt_ids = tokenizer.encode(prompt_with_template, add_special_tokens=False)
            if not prompt_ids:
                raise RuntimeError("Chat template produced an empty prompt")

            ref_ids = tokenizer.encode(reference_text, add_special_tokens=False)
            if not ref_ids:
                raise RuntimeError("Reference text tokenized to zero tokens")

            eos_token_id: int | None = None if no_eos_stop else tokenizer.eos_token_id

            logger.info(
                "Teacher-forced run: {} prompt tokens, {} reference tokens (eos_stop={})",
                len(prompt_ids),
                len(ref_ids),
                eos_token_id is not None,
            )
            result = run_teacher_forced(
                model_pipeline,
                prompt_ids,
                ref_ids,
                prompt_text=prompt,
                reference_text=reference_text,
                eos_token_id=eos_token_id,
                chunk_accuracy_token_size=chunk_accuracy_token_size,
            )
            logger.info(
                "Accuracy: {}/{} = {:.4f}",
                result.num_correct,
                result.num_total,
                result.accuracy,
            )
            if output_file is not None:
                result.save_json(output_file)
                logger.info("Wrote JSON to {}", output_file)

        model_pipeline.barrier()
    logger.info("Pod pipeline complete")
    return result


def main(argv: list[str] | None = None) -> int:
    ttnn.init_distributed_context()
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.weights == "real":
        if args.cache_path is None:
            parser.error("--cache-path is required when --weights real")
        if args.model_path is None:
            parser.error("--model-path is required when --weights real")
    if args.weights in ("real", "state_dict"):
        if args.model_path is None:
            parser.error(f"--model-path is required when --weights {args.weights}")
        index_path = args.model_path / "model.safetensors.index.json"
        if not index_path.is_file():
            parser.error(f"--model-path must contain model.safetensors.index.json (missing {index_path})")

    if not args.reference_file.is_file():
        parser.error(f"--reference-file must exist: {args.reference_file}")
    if args.chunk_accuracy is not None and args.chunk_accuracy < 1:
        parser.error("--chunk-accuracy N requires N >= 1")

    reference_text = args.reference_file.read_text(encoding="utf-8")

    result = run_teacher_forced_demo(
        prompt=args.prompt,
        reference_text=reference_text,
        tokenizer_name_or_path=args.tokenizer,
        weights_mode=args.weights,
        cache_path=args.cache_path,
        model_path=args.model_path,
        lm_head_fp32_dest_acc_en=args.fp32,
        lm_head_persistent_mode=args.persistent_mode,
        dense_layer_id_override=args.dense_layer_id_override,
        moe_layer_id_override=args.moe_layer_id_override,
        output_file=args.output_file,
        no_eos_stop=args.no_eos_stop,
        chunk_accuracy_token_size=args.chunk_accuracy,
    )

    # Mesh 0 only has result; others print nothing extra
    if result is not None and args.output_file is None:
        print(json.dumps(result.to_dict(), indent=2), file=sys.stdout, flush=True)
    else:
        print(file=sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
