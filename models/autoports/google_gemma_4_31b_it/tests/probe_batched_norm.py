# SPDX-License-Identifier: Apache-2.0
"""Discriminate wide interleaved and bounded sharded hidden RMSNorm.

--patch-prefill delegates remaining arguments to probe_batched_prefill, replacing
only four named prefill norms in process. Production dispatch remains unchanged.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

import ttnn

from ..tt.model import Gemma4Model
from ..tt.multichip_decoder import MultichipDecoder

NAMES = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")
ORIGINAL_NORM = MultichipDecoder._norm


def bounded_norm(self, x, name=None):
    rows = math.prod(tuple(x.padded_shape)[:-1])
    if self.decoding or name not in NAMES or rows <= 1024 or x.shape[-1] != 5376:
        return ORIGINAL_NORM(self, x, name)
    per_batch = math.prod(tuple(x.padded_shape)[1:-1])
    count = 1024 // per_batch
    if count < 1:
        raise ValueError("Experimental bounded norm requires each request <=1024 physical rows")
    outputs = []
    for start in range(0, x.shape[0], count):
        end = min(x.shape[0], start + count)
        part = ttnn.slice(x, (start, 0, 0, 0), (end, x.shape[1], x.shape[2], x.shape[3]))
        outputs.append(ttnn.to_memory_config(ORIGINAL_NORM(self, part, name), ttnn.DRAM_MEMORY_CONFIG))
    return ttnn.concat(outputs, dim=0)


def host(tensor):
    return torch.stack([ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)])


def difference(actual, reference):
    delta = actual.float() - reference.float()
    unequal = actual != reference
    counts = unequal.reshape(actual.shape[0], -1).sum(dim=1).tolist()
    result = {
        "equal_per_device": [count == 0 for count in counts],
        "different_elements_per_device": counts,
        "max_abs_error": delta.abs().max().item(),
        "rms_error": delta.square().mean().sqrt().item(),
    }
    indices = unequal.nonzero()
    if len(indices):
        coordinate = tuple(indices[0].tolist())
        result["first_difference"] = {
            "coordinate": list(coordinate),
            "actual": actual[coordinate].item(),
            "reference": reference[coordinate].item(),
        }
    return result


def main():
    if "--patch-prefill" in sys.argv:
        sys.argv.remove("--patch-prefill")
        MultichipDecoder._norm = bounded_norm
        from .probe_batched_prefill import main as prefill_main

        print(
            json.dumps(
                {
                    "experiment": "bounded_hidden_norms",
                    "names": NAMES,
                    "max_norm_rows": 1024,
                    "production_unchanged": True,
                }
            ),
            flush=True,
        )
        prefill_main()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.length % 32 or args.batch * args.length <= 1024:
        raise ValueError("Use aligned per-row length and >1024 total rows for the norm discriminator")
    torch.set_num_threads(8)
    torch.manual_seed(729)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    report = {
        "scope": "real embeddings and layer0 hidden norm weights; no full-model accuracy/performance claim",
        "batch": args.batch,
        "length": args.length,
        "norms": {},
        "coordinate_order": ["device", "batch", "singleton", "token", "channel"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = Gemma4Model(mesh, layer_indices=(0, 5))
        layer = model.layers[0]
        layer.prefill_l1 = False
        ids = torch.randint(3, model.config.vocab_size, (args.batch, args.length), dtype=torch.int32)
        device_ids = model.upload(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        embedded = model.embed(device_ids)
        wide = ttnn.reshape(embedded, (args.batch, 1, args.length, model.config.hidden_size))
        wide_host = host(wide)
        solo_embeddings = []
        for row in range(args.batch):
            row_ids = model.upload(ids[row : row + 1], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            value = model.embed(row_ids)
            solo_embeddings.append(host(ttnn.reshape(value, (1, 1, args.length, model.config.hidden_size))))
            del row_ids, value
        report["embedding"] = difference(wide_host, torch.cat(solo_embeddings, dim=1))
        del wide_host, solo_embeddings
        for name in NAMES:
            generic = ORIGINAL_NORM(layer, wide, name)
            bounded = bounded_norm(layer, wide, name)
            solo_norms = []
            for row in range(args.batch):
                part = ttnn.slice(wide, (row, 0, 0, 0), (row + 1, 1, args.length, model.config.hidden_size))
                value = ORIGINAL_NORM(layer, part, name)
                solo_norms.append(host(value))
                del part, value
            reference = torch.cat(solo_norms, dim=1)
            report["norms"][name] = {
                "generic_vs_solo": difference(host(generic), reference),
                "bounded_vs_solo": difference(host(bounded), reference),
            }
            del generic, bounded, solo_norms, reference
            output.write_text(json.dumps(report, indent=2) + "\n")
        report["program_count"] = mesh.num_program_cache_entries()
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
