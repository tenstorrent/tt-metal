# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-owner helper: matched-state public submission order invariance.

No mesh creation or standalone device execution at import. Invoked by the
prefill signature harness only with --check-decode-submission-order.
"""
import hashlib

import torch


def _hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def set_matched_greedy_params(gen, params_type):
    # Seed scalars retain B1 semantics in the common formatter, even with slots.
    # Explicit seed rows avoid leaving slot1 entropy-driven.
    gen.set_sampling_params(params_type(temperature=0.0, top_k=1, top_p=1.0, seed=[0, 0]), slots=(0, 1))
    assert gen.request_seeds[:2] == [0, 0]


def check(gen, prefill, pool, tables):
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.slot_state import _state_tensors
    from models.common.sampling.generator import SamplingParams

    def snapshot():
        values = {}
        for slot, cache in enumerate(gen.caches):
            values[f"slot{slot}.host_position"] = cache.position
            for name, tensor in _state_tensors(cache):
                for rank, part in enumerate(ttnn.get_device_tensors(tensor)):
                    values[f"slot{slot}.{name}.rank{rank}"] = _hash(ttnn.to_torch(part))
            if pool is not None:
                attention = [layer.attention for layer in cache.layers if layer.attention is not None]
                for index, state in enumerate(attention):
                    for field in ("k_cache", "v_cache"):
                        pages = list(ttnn.get_device_tensors(state.page_table))
                        for rank, part in enumerate(ttnn.get_device_tensors(getattr(state, field))):
                            ids = ttnn.to_torch(pages[rank]).flatten().long()[: (cache.position + 31) // 32]
                            logical = ttnn.to_torch(part)[ids].permute(1, 0, 2, 3).reshape(1, -1, 256)
                            values[f"slot{slot}.qsa{index}.{field}.rank{rank}"] = _hash(logical[:, : cache.position])
        for name, tensor in (("feedback", gen.decode_tokens), ("seeds", gen.seed_state.state)):
            for rank, part in enumerate(ttnn.get_device_tensors(tensor)):
                value = ttnn.to_torch(part)
                value = value.flatten() if name == "feedback" else value
                values[f"{name}.rank{rank}"] = _hash(value[:2])
        return values

    def traces():
        return {
            name: {repr(key): str(value) for key, value in getattr(gen, name).items()}
            for name in ("traces", "sampling_traces")
        }

    runs = []
    for submitted_slots in ((0, 1), (1, 0)):
        gen.reset()
        set_matched_greedy_params(gen, SamplingParams)
        prefill((33, 47), all_logits=False)
        output = gen.decode_forward(
            torch.tensor([42 + slot for slot in submitted_slots]),
            [33 if slot == 0 else 47 for slot in submitted_slots],
            slots=submitted_slots,
            sample_on_device=True,
            page_table=tables[list(submitted_slots)]
            if pool is not None
            else torch.arange(tables.shape[-1], dtype=torch.int32).repeat(2, 1),
            kv_cache=pool if pool is not None else [gen.caches[slot] for slot in submitted_slots],
        )
        physical_tokens = {slot: int(output[row]) for row, slot in enumerate(submitted_slots)}
        logits = gen._read_logits(gen.sample_logits)
        logit_hashes = {str(slot): _hash(logits[slot]) for slot in (0, 1)}
        assert bool(torch.isfinite(logits[:2]).all())
        feedback_exact = True
        for part in ttnn.get_device_tensors(gen.decode_tokens):
            actual = ttnn.to_torch(part).flatten().long()
            feedback_exact = feedback_exact and all(int(actual[slot]) == physical_tokens[slot] for slot in (0, 1))
        runs.append(
            {
                "submitted_slots": list(submitted_slots),
                "tokens_by_physical_slot": physical_tokens,
                "logit_hashes_by_physical_slot": logit_hashes,
                "feedback_matches_output": feedback_exact,
                "state_hashes": snapshot(),
                "trace_ids": traces(),
            }
        )
    baseline, reversed_run = runs
    checks = {
        "tokens_exact": baseline["tokens_by_physical_slot"] == reversed_run["tokens_by_physical_slot"],
        "logits_exact": baseline["logit_hashes_by_physical_slot"] == reversed_run["logit_hashes_by_physical_slot"],
        "all_request_state_exact": baseline["state_hashes"] == reversed_run["state_hashes"],
        "token_feedback_exact": all(run["feedback_matches_output"] for run in runs),
        "model_and_sampling_trace_ids_reused": baseline["trace_ids"] == reversed_run["trace_ids"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "runs": runs,
        "same_history": "Reset; physical slot0 prefix33/token42, slot1 prefix47/token43; submit rows in either order",
        "readback_boundary": "Diagnostic full logits/state readback; no host argmax or token-out performance claim",
    }
