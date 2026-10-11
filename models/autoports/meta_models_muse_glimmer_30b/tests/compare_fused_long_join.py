# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact full-context equivalence for the output tile-join rewrite.

The preserved logical-join method uses the same layer/weights and attention/cache
math as the final decoder. Prior full-context HF runs establish that control's
accuracy. This check changes only how its chunk outputs are assembled.
"""

import argparse
import ast
import gc
import gzip
import hashlib
import json
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.probe_physical_join import LogicalJoinDecoder
from models.autoports.meta_models_muse_glimmer_30b.tests.run_fused import (
    CONFIG,
    ROOT,
    AutoConfig,
    FusedDecoder,
    device_only,
    read,
    to_device,
    weights,
)


def main(layer_idx):
    torch.set_num_threads(8)
    torch.manual_seed(791)
    cfg = AutoConfig.from_pretrained(CONFIG).text_config
    archived = gzip.decompress((ROOT / "doc/fused_decoder/pre_join_decoder_source.py.gz").read_bytes())
    namespace = {}
    exec(compile(archived, "<preserved-fused-control>", "exec"), namespace)
    control_class = namespace["FusedDecoder"]

    def methods(source, cls):
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls)
        return {n.name: ast.dump(n, include_attributes=False) for n in node.body if isinstance(n, ast.FunctionDef)}

    old_methods = methods(archived, "FusedDecoder")
    current_methods = methods((ROOT / "tt/fused_decoder.py").read_text(), "FusedDecoder")
    unchanged = ["from_state_dict", "allocate_cache", "_linear", "_norm", "_rope", "_finish", "_prefill_chunk"]
    assert all(old_methods[name] == current_methods[name] for name in unchanged)
    assert (
        old_methods["prefill_forward"]
        == methods((ROOT / "tests/probe_physical_join.py").read_text(), "LogicalJoinDecoder")["prefill_forward"]
    )
    ref, state = weights(cfg, layer_idx, True)
    del ref
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    rows = []
    try:
        layer = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh)
        del state
        kv = layer.allocate_cache()
        pt = to_device(torch.randperm(4096)[None].int(), mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        for length in (131071, 131072):
            print("JOIN_CASE", length, flush=True)
            x = to_device(torch.randn(1, 1, length, 6656).bfloat16(), mesh)
            with device_only():
                old = LogicalJoinDecoder.prefill_forward(layer, x, kv_cache=kv, page_table=pt)
            expected = read(old).clone()
            old_cache = [read(t).clone() for t in kv]
            del old
            gc.collect()
            with device_only():
                new = layer.prefill_forward(x, kv_cache=kv, page_table=pt)
            got = read(new)
            assert torch.isfinite(got).all()
            assert torch.equal(expected, got), "Output tile-join must be bitwise exact"
            assert all(torch.equal(a, read(b)) for a, b in zip(old_cache, kv)), "Cache contents changed"
            position = length - 1
            token = to_device(torch.randn(1, 1, 1, 6656).bfloat16(), mesh)
            pos = to_device(torch.tensor([position], dtype=torch.int32), mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            cos = ttnn.slice(layer.cos, [0, 0, position, 0], [1, 1, position + 1, 128]) if layer.use_rope else None
            sin = ttnn.slice(layer.sin, [0, 0, position, 0], [1, 1, position + 1, 128]) if layer.use_rope else None
            inputs = dict(kv_cache=kv, page_table=pt, current_pos=pos, rope_cos=cos, rope_sin=sin)
            with device_only():
                old_decode = control_class.decode_forward(layer, token, **inputs)
                new_decode = layer.decode_forward(token, **inputs)
            assert torch.equal(read(old_decode), read(new_decode)), "Maximum-context decode changed"
            rows.append(
                {
                    "length": length,
                    "shape": list(got.shape),
                    "output_bitwise_equal": True,
                    "kv_bitwise_equal": True,
                    "decode_bitwise_equal": True,
                    "decode_position": position,
                    "pcc": 1.0,
                }
            )
            print("JOIN_PASS", rows[-1], flush=True)
            del x, new, got, expected, old_cache
            del token, pos, cos, sin, inputs, old_decode, new_decode
            gc.collect()
        paths = [ROOT / "tt/fused_decoder.py", Path(__file__), ROOT / "tests/probe_physical_join.py"]
        report = {
            "command": sys.argv,
            "layer": layer_idx,
            "kind": layer.kind,
            "unchanged_prefill_math_methods_ast": unchanged,
            "logical_join_control_ast_matches_archive": True,
            "rows": rows,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "control_hf_evidence": f"long_context_{layer_idx}.json",
            "control_source_sha256": hashlib.sha256(
                gzip.decompress((ROOT / "doc/fused_decoder/pre_join_decoder_source.py.gz").read_bytes())
            ).hexdigest(),
        }
        (ROOT / f"doc/fused_decoder/long_join_{layer_idx}.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    main(parser.parse_args().layer)
