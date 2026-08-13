# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check production-trace logits against repeats, batch positions, and eager baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.generator import build_generator
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.generator_vllm import TTMistralSmall24BForCausalLM
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device

MODEL_DIR = Path(__file__).resolve().parents[2]
OUTPUT = MODEL_DIR / "readiness_vllm/logit_determinism.json"
PROMPT_TOKENS = 32
MAX_SEQ_LEN = 256
NUM_BLOCKS = MAX_SEQ_LEN // 32 + 32


def _page_table(batch: int) -> torch.Tensor:
    table = torch.full((batch, MAX_SEQ_LEN // 32), -1, dtype=torch.int32)
    table[:, 0] = torch.arange(batch, dtype=torch.int32)
    # vLLM reserves the next physical block before scheduling decode at the
    # first position after this exactly-one-page prompt.
    table[:, 1] = batch + torch.arange(batch, dtype=torch.int32)
    return table


def _prompt(generator, batch: int) -> torch.Tensor:
    ids = generator.tokenizer.encode(
        "Explain why the sky is blue in one concise paragraph for a curious student. " * 8,
        add_special_tokens=False,
    )[:PROMPT_TOKENS]
    if len(ids) != PROMPT_TOKENS:
        raise RuntimeError("failed to build exact 32-token prompt")
    return torch.tensor([ids] * batch, dtype=torch.long)


def _params(batch: int):
    return type(
        "DeterminismSampling",
        (),
        {"top_k": [1] * batch, "top_p": [0.0] * batch, "temperature": [0.0] * batch},
    )()


def _trace_logits(adapter, cache, batch: int) -> torch.Tensor:
    tokens = _prompt(adapter.generator, batch)
    page_table = _page_table(batch)
    adapter.model.reset_kv_cache(cache)
    sampled = adapter.prefill_forward(
        tokens=tokens,
        page_table=page_table,
        kv_cache=cache,
        prompt_lens=[PROMPT_TOKENS] * batch,
        sampling_params=_params(batch),
        empty_slots=list(range(batch)),
    )
    adapter.decode_forward(
        tokens=sampled,
        start_pos=torch.full((batch,), PROMPT_TOKENS, dtype=torch.long),
        page_table=page_table,
        kv_cache=cache,
        enable_trace=True,
        sampling_params=_params(batch),
        reset_batch=True,
    )
    adapter.generator._synchronize()
    host = ttnn.to_torch(
        adapter.generator._trace_logits,
        mesh_composer=ttnn.ConcatMeshToTensor(adapter.mesh_device, dim=3),
    )
    return host[0, 0, :batch, : adapter.model.vocab_size].clone()


def _eager_logits(adapter, cache) -> torch.Tensor:
    adapter.generator.prepare_for_prefill()
    tokens = _prompt(adapter.generator, 1)
    page_table = _page_table(1)
    adapter.model.reset_kv_cache(cache)
    sampled = adapter.generator.prefill_forward_device_sample(
        tokens,
        page_table=page_table,
        kv_cache=cache,
        prompt_lens=[PROMPT_TOKENS],
        top_k=[1],
        top_p=[0.0],
        temperature=[1.0],
    )
    sampled_host = adapter.generator._sampled_tokens_to_torch(sampled)[:1].reshape(1, 1)
    return adapter.generator.decode_forward(
        sampled_host,
        torch.tensor([PROMPT_TOKENS]),
        page_table=page_table,
        kv_cache=cache,
        sampling_mode="host",
        enable_trace=False,
    )[0].clone()


def _summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(torch.max(torch.abs(reference - candidate)).item()),
        "pcc": float(torch.corrcoef(torch.stack((reference, candidate)))[0, 1].item()),
        "reference_argmax": int(torch.argmax(reference).item()),
        "candidate_argmax": int(torch.argmax(candidate).item()),
    }


def _digest(logits: torch.Tensor) -> str:
    return hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    mesh = open_readiness_mesh_device("P300_QUAD", "FABRIC_1D", 200_000_000)
    generator = None
    try:
        generator = build_generator(
            MODEL_DIR,
            mesh,
            max_batch_size=32,
            max_seq_len=MAX_SEQ_LEN,
            num_blocks=NUM_BLOCKS,
            pooled_kv_cache=True,
        )
        adapter = TTMistralSmall24BForCausalLM(generator)
        cache = adapter.allocate_kv_cache(
            (NUM_BLOCKS, generator.model.num_kv_heads, 32, generator.model.head_dim),
            generator.model.config.kv_cache_dtype,
            generator.model.num_layers,
        )
        trace_1 = _trace_logits(adapter, cache, 1)[0]
        trace_2 = _trace_logits(adapter, cache, 1)[0]
        trace_batch = _trace_logits(adapter, cache, 2)
        eager = _eager_logits(adapter, cache)
        checks = {
            "vllm_trace_run_to_run": _summary(trace_1, trace_2),
            "vllm_trace_batch_position_0": _summary(trace_1, trace_batch[0]),
            "vllm_trace_batch_position_1": _summary(trace_1, trace_batch[1]),
            "standalone_eager_baseline": _summary(eager, trace_1),
        }
        vllm_checks = [checks[name] for name in checks if name.startswith("vllm_trace_")]
        baseline = checks["standalone_eager_baseline"]
        if not (
            all(check["exact"] for check in vllm_checks)
            and baseline["pcc"] >= 0.999
            and baseline["reference_argmax"] == baseline["candidate_argmax"]
        ):
            raise AssertionError(f"logit determinism failed: {checks}")
        result = {
            "status": "pass",
            "prompt_tokens": PROMPT_TOKENS,
            "selected_precision_config": "bfp4_lofi_bfp8kv_bf16ccl",
            "vllm_path": "generator_vllm adapter with traced model+split-sampler decode",
            "standalone_path": "full-model generator eager decode on the same caller-owned cache geometry",
            "checks": checks,
            "digests": {
                "trace_run_1": _digest(trace_1),
                "trace_run_2": _digest(trace_2),
                "trace_batch_position_0": _digest(trace_batch[0]),
                "trace_batch_position_1": _digest(trace_batch[1]),
                "standalone_eager": _digest(eager),
            },
        }
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")


if __name__ == "__main__":
    main()
