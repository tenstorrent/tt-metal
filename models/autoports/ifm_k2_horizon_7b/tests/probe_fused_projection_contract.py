"""Fused AGMM/SwiGLU against actual gathered BFP8 and packed BFP4 operands."""

import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import K2Generator
from .probe_attention_contract import metric


def main():
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    results = []
    rows = [0, 13, 26, 117, 247, 256]
    try:
        gen = K2Generator(mesh, override_num_layers=3)
        gen._ensure_owned_cache(1, 384)
        gen.reset()
        for index, layer in enumerate(gen.model.layers):
            original = layer._fused_prefill_projection

            def inspected(x, w, *, swiglu=False, original=original, layer=layer, index=index):
                cast = ttnn.typecast
                quantized = cast(x, ttnn.bfloat8_b)

                def retained_cast(value, dtype, *args, **kw):
                    return quantized if value is x and dtype == ttnn.bfloat8_b else cast(value, dtype, *args, **kw)

                with patch.object(ttnn, "typecast", side_effect=retained_cast):
                    out = original(x, w, swiglu=swiglu)
                # The native fused op can consume its local shard directly;
                # persistent remote-gather scratch is not a public full gather.
                parts = [ttnn.to_torch(t).reshape(-1, 1024)[rows].double() for t in ttnn.get_device_tensors(quantized)]
                inputs = torch.cat(parts, dim=-1)
                reference, actual = [], []
                for rank in range(4):
                    weight = ttnn.to_torch(ttnn.get_device_tensors(w)[rank]).squeeze().double()
                    ref = inputs @ weight
                    if swiglu:
                        tiled = ref.reshape(len(rows), -1, 32)
                        gate = tiled[:, 0::2].reshape(len(rows), -1)
                        up = tiled[:, 1::2].reshape(len(rows), -1)
                        ref = torch.nn.functional.silu(gate) * up
                    reference.append(ref)
                    actual.append(
                        ttnn.to_torch(ttnn.get_device_tensors(out)[rank]).reshape(-1, out.shape[-1])[rows].double()
                    )
                ref, value = torch.cat(reference, -1), torch.cat(actual, -1)
                record = {
                    "layer": index,
                    "role": "swiglu" if swiglu else "qkv",
                    "rows": rows,
                    "weight_dtype": str(w.dtype),
                    "actual_input_dtype": str(quantized.dtype),
                    "oracle": "FP64 concatenation of exact retained quantized native source inputs @ actual dequantized packed weight; exact SiLU for fused MLP",
                    "all": metric(ref, value),
                    "per_row": {str(row): metric(ref[i], value[i]) for i, row in enumerate(rows)},
                    "per_rank": [metric(a, b) for a, b in zip(reference, actual)],
                }
                results.append(record)
                Path("models/autoports/ifm_k2_horizon_7b/doc/full_model/fused_projection_fixed_input.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
                print(index, record["role"], record["all"], record["per_row"], flush=True)
                return out

            layer._fused_prefill_projection = inspected
        ids = (gen.tokenizer.encode("A careful scientist checks the evidence before drawing a conclusion. ") * 30)[:257]
        gen.prefill_forward(
            torch.tensor([ids]),
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=[257],
            sampling_mode="host",
        )
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
