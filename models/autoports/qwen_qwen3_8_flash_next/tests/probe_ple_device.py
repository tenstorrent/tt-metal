# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate real-weight PLE arithmetic, with setup-prepared real embedding rows.

This diagnostic explicitly excludes device hashing and embedding lookup and is
NOT full decoder acceptance. Only setup, HF reference, input refresh and final
comparison use torch. Captured PLE arithmetic has device-only hidden/embedding
inputs and an explicit persistent convolution state.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_quantization import SelectedRows
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
from models.autoports.qwen_qwen3_8_flash_next.tt.ple import PLE


def pcc(a, b):
    return torch.corrcoef(torch.stack((a.float().flatten(), b.float().flatten())))[0, 1].item()


def metric(expected, actual):
    actual = actual.reshape(expected.shape)
    return {
        "pcc": pcc(expected, actual),
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs_error": (expected.float() - actual.float()).abs().max().item(),
    }


def load_reference(snapshot, config):
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.1."
    state = {}
    for name, filename in index.items():
        if name.startswith(prefix + "ple.") and ".ngram_embedding.shard_" not in name:
            with safe_open(snapshot / filename, framework="pt") as f:
                state[name.removeprefix(prefix)] = f.get_tensor(name)
    with torch.device("meta"):
        reference = Qwen4ExpTextPLELayer(config, 1, 0)
    reference.load_state_dict(
        {key.removeprefix("ple."): value for key, value in state.items()}, strict=False, assign=True
    )
    reference.ple_embedding.ngram_embedding = SelectedRows(
        snapshot, index, prefix + "ple.ple_embedding.ngram_embedding.", "absmax"
    )
    reference.eval()
    return reference, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="1,7,31,32,33,65")
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    torch.manual_seed(731)
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    reference, state = load_reference(args.snapshot, config)
    width = config.hidden_size * config.hc_count
    eos = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
    cases = []
    with torch.inference_mode():
        for length in lengths:
            hidden = (torch.randn(1, length + 3, width) * 0.02).bfloat16()
            tokens = torch.randint(0, config.vocab_size, (1, length + 3))
            tokens[0, max(0, length - 2)] = eos
            embeddings = reference.ple_embedding(tokens, None)
            cache = DynamicCache(config=config)
            expected = [reference(hidden[:, :length], tokens[:, :length], cache)]
            states = [cache.get_conv_state(1, state_idx=1).clone()]
            for position in range(length, length + 3):
                expected.append(
                    reference(hidden[:, position : position + 1], tokens[:, position : position + 1], cache)
                )
                states.append(cache.get_conv_state(1, state_idx=1).clone())
            cases.append((length, hidden, embeddings, expected, states))
    print("PLE_CPU_REFERENCES_READY", flush=True)
    report = {
        "scope": "real-weight PLE arithmetic diagnostic; embedding rows prepared at setup; not complete decoder acceptance",
        "snapshot": str(args.snapshot),
        "cases": [],
    }
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        module = PLE(WeightLoader(state, mesh), config)

        def host(value):
            return ttnn.from_torch(value.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        def device(value):
            return ttnn.from_torch(value.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

        for length, hidden, embeddings, expected, reference_states in cases:
            zero = host(torch.zeros(1, 1, module.state_len, width, dtype=torch.bfloat16))
            conv_state = device(torch.zeros(1, 1, module.state_len, width, dtype=torch.bfloat16))
            prefill_x = device(hidden[:, :length].unsqueeze(0))
            prefill_e = device(embeddings[:, :length].unsqueeze(0))
            decode_x = device(hidden[:, length : length + 1].unsqueeze(0))
            decode_e = device(embeddings[:, length : length + 1].unsqueeze(0))
            refresh = [
                (host(hidden[:, p : p + 1].unsqueeze(0)), host(embeddings[:, p : p + 1].unsqueeze(0)))
                for p in range(length, length + 3)
            ]
            prefill = module(prefill_x, prefill_e, conv_state=conv_state)
            ttnn.synchronize_device(mesh)
            actual_prefill = ttnn.to_torch(prefill)
            entry = {
                "prefill_length": length,
                "prefill": metric(expected[0], actual_prefill),
                "prefill_state": metric(reference_states[0].transpose(1, 2).unsqueeze(0), ttnn.to_torch(conv_state)),
                "decode": [],
            }
            # Store the TT-produced prefix state, preserving its own accumulated
            # numerical error. All persistent input/state allocations predate trace.
            prefix_state = host(ttnn.to_torch(conv_state))
            warm = module(decode_x, decode_e, conv_state=conv_state)
            ttnn.synchronize_device(mesh)
            del warm
            ttnn.copy_host_to_device_tensor(prefix_state, conv_state)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            output = module(decode_x, decode_e, conv_state=conv_state)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            # This exact capture output may be overwritten by the next replay.
            # It has no persistent aliases, and is always read back before replay.
            # No inputs, states, scopes or program-cache allocations are exempted.
            trace_allocation_tracker.acknowledge_corruptible(output)
            try:
                first = None
                for repeat in range(2):
                    ttnn.copy_host_to_device_tensor(prefix_state, conv_state)
                    ttnn.copy_host_to_device_tensor(refresh[0][0], decode_x)
                    ttnn.copy_host_to_device_tensor(refresh[0][1], decode_e)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    actual = ttnn.to_torch(output)
                    if first is None:
                        first = actual.clone()
                        entry["decode"].append(metric(expected[1], actual))
                    else:
                        entry["reset_replay_equal"] = torch.equal(first, actual)
                for step in (1, 2):
                    ttnn.copy_host_to_device_tensor(refresh[step][0], decode_x)
                    ttnn.copy_host_to_device_tensor(refresh[step][1], decode_e)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    actual = ttnn.to_torch(output)
                    entry["decode"].append(metric(expected[step + 1], actual))
                entry["final_state"] = metric(
                    reference_states[-1].transpose(1, 2).unsqueeze(0), ttnn.to_torch(conv_state)
                )
            finally:
                ttnn.release_trace(mesh, trace)
            # A fresh request on the same loaded module must reproduce prefill.
            ttnn.copy_host_to_device_tensor(zero, conv_state)
            repeated = module(prefill_x, prefill_e, conv_state=conv_state)
            entry["prefill_reset_equal"] = torch.equal(actual_prefill, ttnn.to_torch(repeated))
            report["cases"].append(entry)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(entry), flush=True)
            assert entry["prefill"]["pcc"] >= 0.995 and entry["prefill"]["finite"]
            assert all(item["pcc"] >= 0.995 and item["finite"] for item in entry["decode"])
            assert entry["reset_replay_equal"] and entry["prefill_reset_equal"]
    except Exception as exc:
        report["error"] = str(exc).split("backtrace:")[0]
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
