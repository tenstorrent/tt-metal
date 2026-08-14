# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reduced full-wrapper B2 mixed-prompt and public token-out contract probe."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator
from models.common.sampling import SamplingParams


def _device_tensor(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def _slot_kv_snapshot(generator, slot):
    blocks = generator.page_table_host[slot].tolist()
    snapshots = []
    for layer in generator.model.layers:
        if layer.layer_kind == "full_attention":
            snapshots.extend(_device_tensor(layer.caches[name])[blocks].clone() for name in ("key", "value"))
    return snapshots


def _linear_slot_snapshot(generator, slot):
    snapshots = []
    for layer in generator.model.layers:
        if layer.layer_kind == "linear_attention":
            snapshots.append(_device_tensor(layer.caches["conv"])[:, slot].clone())
            snapshots.append(_device_tensor(layer.caches["recurrent"])[slot].clone())
    return snapshots


def _assert_exact(lhs, rhs, label):
    assert len(lhs) == len(rhs)
    for index, (before, after) in enumerate(zip(lhs, rhs)):
        assert torch.equal(before, after), f"{label}[{index}] changed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    try:
        generator = build_generator(
            Path("models/autoports/qwen_qwen3_6_27b"),
            mesh,
            num_layers=4,
            max_context=128,
            batch=2,
        )
        tokens = torch.zeros((2, 65), dtype=torch.long)
        tokens[0] = torch.arange(65) % generator.model.vocab_size
        tokens[1, :63] = torch.arange(63) % generator.model.vocab_size
        logits = generator.prefill_forward(
            tokens,
            page_table=generator._page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=[65, 63],
        )
        first = torch.argmax(logits[:, 0], dim=-1).tolist()
        inactive_kv_before = _slot_kv_snapshot(generator, 1)
        state = generator.setup_token_out_decode(
            first,
            [65, 63],
            page_table=generator._page_table,
            kv_cache=generator.kv_cache,
            active_mask=[1, 0],
            # Exercise the public non-greedy common-sampler path; the measured
            # production benchmark uses the same split candidate path with
            # greedy top-k=1 parameters.
            sampling_params=SamplingParams(temperature=0.8, top_k=5, top_p=0.9),
        )
        assert not generator.sampling.tt_sampling.force_argmax_sampling
        assert state["kv_cache"] is generator.kv_cache
        for _ in range(3):
            sampled = generator.token_out_decode_step(readback=True)
        inactive_kv_after = _slot_kv_snapshot(generator, 1)
        _assert_exact(inactive_kv_before, inactive_kv_after, "inactive KV")
        positions = ttnn.to_torch(ttnn.get_device_tensors(state["position"])[0]).reshape(-1)[:2].tolist()
        assert len(sampled) == 2
        assert positions[1] == 63, positions

        # Reuse slot 1 while slot 0 remains live.  Reset must preserve slot 0,
        # clear slot 1 linear state, and reject decode until slot 1 is refilled.
        live_kv_before = _slot_kv_snapshot(generator, 0)
        live_linear_before = _linear_slot_snapshot(generator, 0)
        generator.reset_slots([1])
        _assert_exact(live_linear_before, _linear_slot_snapshot(generator, 0), "live linear state")
        for tensor in _linear_slot_snapshot(generator, 1):
            assert torch.count_nonzero(tensor) == 0, "reset slot linear state is nonzero"
        try:
            generator.decode_forward(
                torch.tensor([[0], [1]]),
                torch.tensor([68, 0]),
                page_table=generator._page_table,
                kv_cache=generator.kv_cache,
                active_mask=[0, 1],
            )
        except RuntimeError as error:
            assert "require prefill" in str(error)
        else:
            raise AssertionError("reset slot decoded before refill")

        reuse_tokens = torch.zeros((2, 65), dtype=torch.long)
        reuse_tokens[1, :63] = torch.arange(63) % generator.model.vocab_size
        generator.prefill_forward(
            reuse_tokens,
            page_table=generator._page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=[0, 63],
        )
        _assert_exact(live_kv_before, _slot_kv_snapshot(generator, 0), "live KV during peer refill")
        _assert_exact(live_linear_before, _linear_slot_snapshot(generator, 0), "live linear during peer refill")
        result = {
            "status": "FULL_MODEL_MIXED_SLOTS_OK",
            "mesh": [1, 4],
            "layer_indices": [0, 1, 2, 3],
            "prompt_lengths": [65, 63],
            "active_mask": [1, 0],
            "sampling": {"temperature": 0.8, "top_k": 5, "top_p": 0.9, "force_argmax": False},
            "sampled_tokens": sampled,
            "positions": positions,
            "inactive_kv_exact": True,
            "reset_reuse_ok": True,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("FULL_MODEL_MIXED_SLOTS_OK", sampled, positions, "INACTIVE_KV_EXACT RESET_REUSE_OK", flush=True)
    finally:
        if generator is not None:
            generator.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
