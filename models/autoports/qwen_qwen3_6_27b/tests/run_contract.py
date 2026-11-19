# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Request reuse, boundaries and prefix-state controls on real HF layer weights."""

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_class, decoder_kwargs, provenance
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import MODEL, REVISION, load_layer, pcc


def row_pcc(a, b):
    a, b = a.double().reshape(-1, a.shape[-1]), b.double().reshape(-1, b.shape[-1])
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    values = (a * b).sum(-1) / (a.square().sum(-1) * b.square().sum(-1)).sqrt()
    assert torch.isfinite(values).all(), "Non-finite per-row PCC"
    return values.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(183)
    c = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    c._attn_implementation = "sdpa"
    weights = load_layer(args.layer)
    hf = Qwen3_5DecoderLayer(c, args.layer).eval().to(torch.bfloat16)
    hf.load_state_dict(weights)
    rot = Qwen3_5TextRotaryEmbedding(c)
    results = []

    def save():
        args.output.write_text(
            json.dumps({"provenance": provenance(), "layer": args.layer, "checks": results}, indent=2) + "\n"
        )

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        layer = decoder_class().from_state_dict(
            weights, **decoder_kwargs(), hf_config=c, layer_idx=args.layer, mesh_device=mesh
        )
        linear = layer.kind == "linear_attention"
        pages = torch.randperm(64, dtype=torch.int32).reshape(1, 64)
        state = layer.create_recurrent_state() if linear else layer.create_paged_state(pages)
        buffers = [state.conv, state.recurrent] if linear else [state.key, state.value]
        zero = [ttnn.from_torch(torch.zeros(tuple(t.shape)), dtype=t.dtype, layout=t.layout) for t in buffers]

        def reset():
            for z, t in zip(zero, buffers):
                ttnn.copy_host_to_device_tensor(z, t)

        def read(t):
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

        def inputs(x, start=0):
            r = (
                None
                if linear
                else tuple(
                    layer._upload(t.unsqueeze(1))
                    for t in rot(x, torch.arange(start, start + x.shape[1]).reshape(1, -1))
                )
            )
            return layer._upload(x), r

        def forward(x, r, start=0):
            with (
                patch.object(ttnn, "from_torch", side_effect=AssertionError("runtime from_torch")),
                patch.object(ttnn, "to_torch", side_effect=AssertionError("runtime to_torch")),
            ):
                return layer.prefill_forward(x, state=state, start_pos=start, rope=r)

        for length in [1, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1025, 17]:
            reset()
            if not linear:
                pages = torch.randperm(64, dtype=torch.int32).reshape(1, 64)
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(pages, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), state.page_table
                )
            x = torch.randn(1, length, 5120, dtype=torch.bfloat16) * 0.1
            with torch.no_grad():
                expected = hf(
                    x,
                    position_embeddings=rot(x, torch.arange(length).reshape(1, -1)),
                    past_key_values=DynamicCache(config=c),
                )
            tx, r = inputs(x)
            actual = read(forward(tx, r))
            rows = row_pcc(expected.float() - x.float(), actual.float() - x.float())
            entry = {
                "case": "length",
                "length": length,
                "pcc": pcc(expected, actual),
                "min_contribution_pcc": min(rows),
                "worst_row": min(range(len(rows)), key=rows.__getitem__),
                "contribution_pcc": rows,
                "all_finite": bool(torch.isfinite(actual).all()),
            }
            results.append(entry)
            save()
            print({k: v for k, v in entry.items() if k != "contribution_pcc"}, flush=True)
            assert entry["pcc"] >= 0.995 and min(rows) >= 0.995 and entry["all_finite"]
        # Prefix continuation compared per row against one full reference call.
        length = 545
        x = torch.randn(1, length, 5120, dtype=torch.bfloat16) * 0.1
        reset()
        tx, r = inputs(x)
        full = read(forward(tx, r))
        for start in [1, 32, 33, 128, 511, 512, 544]:
            reset()
            prefix, pr = inputs(x[:, :start])
            read(forward(prefix, pr))
            n = min(8, length - start)
            suffix, sr = inputs(x[:, start : start + n], start)
            resumed = read(forward(suffix, sr, start))
            residual = x[:, start : start + n].float()
            expected = full[:, start : start + n].float() - residual
            positive = row_pcc(expected, resumed.float() - residual)
            reset()
            negative = read(forward(suffix, sr, start))
            control = row_pcc(expected, negative.float() - residual)
            entry = {"case": "resume", "start": start, "rows": n, "positive": positive, "dropped_history": control}
            results.append(entry)
            save()
            print(entry, flush=True)
            assert min(positive) >= 0.995, entry
            assert min(control) < 0.995, entry
        mixed_x, mixed_state, mixed_rope, mixed_expected = [], [], [], []
        starts = [32, 33]
        for start, tail in zip(starts, [33, 17]):
            xx = torch.randn(1, start + tail, 5120, dtype=torch.bfloat16) * 0.1
            with torch.no_grad():
                yy = hf(
                    xx,
                    position_embeddings=rot(xx, torch.arange(start + tail).reshape(1, -1)),
                    past_key_values=DynamicCache(config=c),
                )
            ss = (
                layer.create_recurrent_state()
                if linear
                else layer.create_paged_state(torch.randperm(64).reshape(1, 64))
            )
            prefix, pr = inputs(xx[:, :start])
            out = layer.prefill_forward(prefix, state=ss, rope=pr)
            del out
            suffix, sr = inputs(xx[:, start:], start)
            mixed_x.append(suffix)
            mixed_state.append(ss)
            mixed_rope.append(sr)
            mixed_expected.append(yy[:, start:])
        mixed = layer.prefill_forward(mixed_x, state=mixed_state, start_pos=starts, rope=mixed_rope)
        mixed_pcc = [pcc(e, read(a)) for e, a in zip(mixed_expected, mixed)]
        results.append({"case": "mixed_rows", "starts": starts, "lengths": [33, 17], "pcc": mixed_pcc})
        save()
        assert all(math.isfinite(p) and p >= 0.995 for p in mixed_pcc)
        print("CONTRACT_PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
