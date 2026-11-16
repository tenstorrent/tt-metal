"""Run the standard resume gate with a diagnostic prefill-Q=32 geometry override.

This wrapper does not edit production sources or policy. It changes only the
runtime ``geometry['prefill_q']`` value after model construction and before the
readiness runner begins its cases, leaving P19 weights, activation/KV/CCL policy,
absolute positions, and K32 source configuration unchanged.
"""

import ast
import hashlib
import sys
from pathlib import Path

import torch

from readiness_check import run_prefill_resume_check as runner

POLICY_ID = "stage12_bf16_hifi4_kv_down_bfp8_p19"
MODEL_REL = Path("models/autoports/google_gemma_4_31b_it")
ORIGINAL_IMPORT_BUILDER = runner._import_build_generator
ORIGINAL_RUN = runner.run_prefill_resume_check
ORIGINAL_COMPARE = runner.compare_logits
RUN_METADATA = {}


def _verify_sdpa_source(model_dir):
    source = model_dir / "tt/optimized_decoder.py"
    tree = ast.parse(source.read_text())
    sdpa_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "SDPAProgramConfig"
    ]
    configs = [
        {keyword.arg: keyword.value for keyword in call.keywords}
        for call in sdpa_calls
        if {keyword.arg for keyword in call.keywords} >= {"q_chunk_size", "k_chunk_size"}
        and isinstance(
            next((keyword.value for keyword in call.keywords if keyword.arg == "q_chunk_size"), None), ast.IfExp
        )
    ]
    if len(configs) != 1:
        raise RuntimeError(f"Expected one prefill SDPA config with Q/K chunks; found {len(configs)}")
    config = configs[0]
    if not isinstance(config["k_chunk_size"], ast.Constant) or config["k_chunk_size"].value != 32:
        raise RuntimeError("Pinned prefill K chunk is no longer literal K32")
    q = config["q_chunk_size"]
    if not (
        isinstance(q, ast.IfExp)
        and isinstance(q.body, ast.Constant)
        and q.body.value == 32
        and isinstance(q.orelse, ast.Subscript)
        and isinstance(q.orelse.value, ast.Attribute)
        and q.orelse.value.attr == "geometry"
        and isinstance(q.orelse.slice, ast.Constant)
        and q.orelse.slice.value == "prefill_q"
    ):
        raise RuntimeError("Unexpected production Q chunk selection; review this diagnostic override")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _import_build_generator(model_dir):
    original_builder = ORIGINAL_IMPORT_BUILDER(model_dir)
    model_dir = Path(model_dir).resolve()
    optimized_decoder_sha = _verify_sdpa_source(model_dir)

    def build_generator(**kwargs):
        generator = original_builder(**kwargs)
        policy = generator.model.precision_config
        if policy["config_id"] != POLICY_ID:
            raise RuntimeError(f"Expected P19, got {policy['config_id']}")
        if policy["ccl_dtype"]["prefill"] != "bfloat16":
            raise RuntimeError("This experiment must retain P19 BF16 prefill CCL")
        original_q = []
        for layer in generator.model.layers:
            geometry = layer.geometry
            if geometry["prefill_q"] != 128:
                raise RuntimeError(f"Unexpected original P19 Q geometry: {geometry['prefill_q']}")
            original_q.append(geometry["prefill_q"])
            geometry["prefill_q"] = 32
        RUN_METADATA.update(
            {
                "policy": policy,
                "original_prefill_q_per_layer": original_q,
                "override_prefill_q": 32,
                "forced_actual_prefill_q": 32,
                "prefill_k_chunk_size": 32,
                "prefill_k_source": "optimized_decoder.py SDPAProgramConfig literal, AST-verified",
                "optimized_decoder_sha256": optimized_decoder_sha,
                "override_method": "set every layer.geometry['prefill_q']=32 after construction, before runner cases/warmup",
            }
        )
        return generator

    return build_generator


def _compare_with_positions(reference, candidate):
    result = ORIGINAL_COMPARE(reference, candidate)
    ref_values, ref_ids = torch.topk(reference[0].float(), 2, dim=-1)
    cand_values, cand_ids = torch.topk(candidate[0].float(), 2, dim=-1)
    result["per_position"] = [
        {
            "window_index": i,
            "pcc": runner._pcc(reference[0, i], candidate[0, i]),
            "reference_top1": int(ref_ids[i, 0]),
            "candidate_top1": int(cand_ids[i, 0]),
            "reference_top1_top2_gap": float(ref_values[i, 0] - ref_values[i, 1]),
            "candidate_top1_top2_gap": float(cand_values[i, 0] - cand_values[i, 1]),
        }
        for i in range(reference.shape[1])
    ]
    return result


def _run_with_metadata(*args, **kwargs):
    record = ORIGINAL_RUN(*args, **kwargs)
    record["diagnostic_override"] = dict(RUN_METADATA)
    record["diagnostic_override"][
        "scope_limit"
    ] = "One full-model diagnostic run only. Q32 everywhere is not a selected production policy."
    return record


runner._import_build_generator = _import_build_generator
runner.compare_logits = _compare_with_positions
runner.run_prefill_resume_check = _run_with_metadata

if __name__ == "__main__":
    sys.exit(runner._main())
