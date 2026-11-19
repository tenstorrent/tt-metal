# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real layer-3 QSA control and a traced paged-device component probe.

CPU mode imports no TTNN and opens no device. A host-built prefix is an
explicit decode-cache fixture, not a claim of full prefill validation. Run
prefix=0 for sequential fresh-state token steps, prefix=2048 for the first
pruning boundary, and prefix near 262144 for the advertised decode context.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


class ReferenceCache:
    def __init__(self, raw=None, k=None, v=None):
        self.raw, self.k, self.v = raw, k, v

    def update_indexer(self, raw, layer_idx):
        assert layer_idx == 3
        self.raw = raw if self.raw is None else torch.cat([self.raw, raw], dim=1)
        return self.raw

    def update(self, k, v, layer_idx):
        assert layer_idx == 3
        self.k = k if self.k is None else torch.cat([self.k, k], dim=-2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=-2)
        return self.k, self.v


def load_reference(snapshot):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(**json.loads((snapshot / "config.json").read_text())["text_config"])
    config._attn_implementation = "eager"
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.3."
    state = {}
    shard_keys = {}
    for key, filename in weight_map.items():
        if key.startswith(prefix + "self_attn."):
            shard_keys.setdefault(filename, []).append(key)
    for filename, keys in shard_keys.items():
        with safe_open(snapshot / filename, framework="pt") as shard:
            for key in keys:
                state[key.removeprefix(prefix)] = shard.get_tensor(key)
    with torch.device("meta"):
        model = hf.Qwen4ExpTextAttention(config, 3)
    model.load_state_dict({key.removeprefix("self_attn."): value for key, value in state.items()}, assign=True)
    model.eval()
    rope = hf.Qwen4ExpTextRotaryEmbedding(config)
    provenance = {
        "transformers": importlib.metadata.version("transformers"),
        "reference_path": hf.__file__,
        "reference_sha256": hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
        "real_weight_keys": sorted(state),
        "checkpoint_shards": sorted(shard_keys),
        "advertised_context": config.max_position_embeddings,
    }
    return config, model, rope, state, provenance


@torch.no_grad()
def fixtures(model, rope, *, prefix, steps):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    gen = torch.Generator().manual_seed(1703)
    hidden = (torch.randn(1, prefix + steps, 2560, generator=gen) * 0.1).bfloat16()
    cos, sin = rope(hidden, torch.arange(prefix + steps).reshape(1, -1))
    raw = model.indexer.index_qk_proj(hidden)[..., 512:].reshape(1, prefix + steps, 128)
    if prefix:
        keys = model.k_norm(model.k_proj(hidden[:, :prefix]).reshape(1, prefix, 2, 256)).transpose(1, 2)
        keys = apply_rotary_pos_emb(keys, cos=cos[:, :prefix], sin=sin[:, :prefix])
        values = model.v_proj(hidden[:, :prefix]).reshape(1, prefix, 2, 256).transpose(1, 2)
    else:
        keys = values = None
    full_blocks = prefix // 4
    pooled = raw[:, : full_blocks * 4].float().reshape(1, full_blocks, 4, 128).mean(dim=2).bfloat16()
    pooled = model.indexer.k_layernorm(pooled)
    pooled = apply_rotary_pos_emb(
        pooled.unsqueeze(1), cos=cos[:, : full_blocks * 4 : 4], sin=sin[:, : full_blocks * 4 : 4]
    ).squeeze(1)
    cache = ReferenceCache(raw[:, :prefix] if prefix else None, keys, values)
    expected_outputs, expected_masks, rows = [], [], []
    for position in range(prefix, prefix + steps):
        query = hidden[:, position : position + 1]
        position_embeddings = (cos[:, : position + 1], sin[:, : position + 1])
        causal = torch.zeros(1, 1, 1, position + 1, dtype=torch.bfloat16)
        mask_cache = ReferenceCache(cache.raw)
        mask = model.indexer(query, position_embeddings, causal, mask_cache)
        output, _ = model(query, position_embeddings, causal, cache)

        # Independent compressed-cache formulation, using actual weights and
        # the installed reference's rotary/norm functions, not its selection loop.
        visible = position + 1
        complete = visible // 4
        q = model.indexer.index_qk_proj(query)[..., :512].reshape(1, 1, 4, 128)
        q = model.indexer.q_layernorm(q)
        q = apply_rotary_pos_emb(
            q, cos=cos[:, position : position + 1], sin=sin[:, position : position + 1], unsqueeze_dim=2
        )
        if complete:
            compressed = raw[:, : complete * 4].float().reshape(1, complete, 4, 128).mean(dim=2).bfloat16()
            compressed = model.indexer.k_layernorm(compressed)
            compressed = apply_rotary_pos_emb(
                compressed.unsqueeze(1), cos=cos[:, : complete * 4 : 4], sin=sin[:, : complete * 4 : 4]
            ).squeeze(1)
            scores = (q[0, 0].float() @ compressed[0].float().T).relu().sum(0) / math.sqrt(128)
            selected = scores.topk(min(512, complete)).indices
        else:
            selected = torch.empty(0, dtype=torch.int64)
        independent = torch.zeros(visible, dtype=torch.bool)
        if complete:
            independent[(selected[:, None] * 4 + torch.arange(4)).flatten()] = True
        independent[complete * 4 :] = True
        assert torch.equal(independent, mask[0, 0, 0] == 0), f"CPU compressed selection mismatch at {position}"
        expected_outputs.append(output)
        expected_masks.append(independent)
        rows.append(
            {
                "position": position,
                "visible_tokens": visible,
                "complete_blocks": complete,
                "selected_tokens": int(independent.sum()),
                "dropped_tokens": int((~independent).sum()),
                "compressed_control_exact": True,
            }
        )
    raw_history = torch.zeros(1, 1, 3, 128, dtype=torch.bfloat16)
    if prefix:
        count = min(3, prefix)
        raw_history[:, :, -count:] = raw[:, prefix - count : prefix].unsqueeze(1)
    return {
        "hidden": hidden[:, prefix:],
        "keys": keys,
        "values": values,
        "compressed": pooled,
        "raw_history": raw_history,
        "outputs": expected_outputs,
        "masks": expected_masks,
        "rows": rows,
    }


def pcc(reference, actual):
    return torch.corrcoef(torch.stack([reference.float().flatten(), actual.float().flatten()]))[0, 1].item()


def device_probe(config, weights, fixture, *, prefix, capacity):
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa import QSA

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        loader = WeightLoader(weights, mesh)
        model = QSA(loader, config)
        permutation = torch.randperm(capacity // 32, generator=torch.Generator().manual_seed(1704)).reshape(1, -1).int()
        state = model.make_state(capacity=capacity, page_table=permutation)
        x = loader.tensor(fixture["hidden"][:, :1].reshape(1, 1, 1, 2560))
        seeded_keys = torch.zeros(capacity // 32, 2, 32, 256, dtype=torch.bfloat16)
        seeded_values = torch.zeros_like(seeded_keys)
        if prefix:
            for position in range(prefix):
                page = int(permutation[0, position // 32])
                seeded_keys[page, :, position % 32] = fixture["keys"][0, :, position]
                seeded_values[page, :, position % 32] = fixture["values"][0, :, position]
        seeded_compressed = torch.zeros(1, 1, capacity // 4, 128, dtype=torch.bfloat16)
        seeded_compressed[:, :, : prefix // 4] = fixture["compressed"].unsqueeze(1)

        def refresh(value, destination):
            host = ttnn.from_torch(value.contiguous(), dtype=destination.dtype, layout=destination.layout)
            ttnn.copy_host_to_device_tensor(host, destination)

        def reset():
            refresh(seeded_keys, state.k_cache)
            refresh(seeded_values, state.v_cache)
            refresh(seeded_compressed, state.compressed_keys)
            refresh(fixture["raw_history"], state.raw_history)
            refresh(torch.tensor([prefix], dtype=torch.int32), state.position)

        reset()
        model.step(x, state=state, return_mask=True)
        ttnn.synchronize_device(mesh)
        reset()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output, mask = model.step(x, state=state, return_mask=True)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        reset()
        measurements = []
        for ordinal, expected in enumerate(fixture["outputs"]):
            position = prefix + ordinal
            refresh(fixture["hidden"][:, ordinal : ordinal + 1].reshape(1, 1, 1, 2560), x)
            refresh(torch.tensor([position], dtype=torch.int32), state.position)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(output).reshape(expected.shape)
            actual_mask = ttnn.to_torch(mask)[0, 0]
            expected_mask = torch.zeros(capacity, dtype=torch.bool)
            expected_mask[: position + 1] = fixture["masks"][ordinal]
            actual_allowed = actual_mask == 0
            disagreements = int((actual_allowed != expected_mask.unsqueeze(0)).sum())
            row = {
                "position": position,
                "output_pcc": pcc(expected, actual),
                "mask_bit_disagreements": disagreements,
                "selected_tokens": int(actual_allowed[0].sum()),
                "trace_id": str(trace),
            }
            measurements.append(row)
            print(json.dumps(row), flush=True)
            assert row["output_pcc"] >= 0.995, row
            assert disagreements == 0, row
        return {
            "rows": measurements,
            "page_permutation": permutation.tolist(),
            "physical_capacity": capacity,
            "capture_count": 1,
            "trace_reuse": True,
            "fixture_scope": "host-seeded prefix cache followed by device token steps",
        }
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--device", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    config, reference, rope, weights, provenance = load_reference(args.snapshot)
    if not (0 <= args.prefix and 1 <= args.steps and args.prefix + args.steps <= config.max_position_embeddings):
        raise ValueError("Requested fixture exceeds the unmodified HF context")
    capacity = args.capacity or max(128, math.ceil((args.prefix + args.steps) / 128) * 128)
    if capacity < args.prefix + args.steps:
        raise ValueError("Physical cache capacity cannot truncate the fixture")
    fixture = fixtures(reference, rope, prefix=args.prefix, steps=args.steps)
    report = {"provenance": provenance, "prefix": args.prefix, "steps": args.steps, "cpu": fixture["rows"]}
    if args.device:
        report["device"] = device_probe(config, weights, fixture, prefix=args.prefix, capacity=capacity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "device": args.device, "cpu": report["cpu"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
