"""Identical real operands: native N8/N16 vs exact dequantized matmul oracle."""

import json
from pathlib import Path

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
    selected = [0, 13, 26, 117, 247, 256]
    try:
        gen = K2Generator(mesh, override_num_layers=3)
        gen._ensure_owned_cache(1, 384)
        gen.reset()
        for index, layer in enumerate(gen.model.layers):
            original = layer._linear

            def inspected(x, w, *, index=index, layer=layer, original=original):
                out = original(x, w)
                if w is not layer.wo and w is not layer.wdown:
                    return out
                role = "o" if w is layer.wo else "down"
                outputs = {"LoFi_N16": out}
                for fidelity, n in [(ttnn.MathFidelity.LoFi, 8), (ttnn.MathFidelity.HiFi4, 16)]:
                    config = ttnn.MinimalMatmulConfig(
                        M_block_size=8,
                        K_block_size=8,
                        N_block_size=n,
                        subblock_h=2,
                        subblock_w=4,
                        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                    )
                    compute = ttnn.init_device_compute_kernel_config(
                        mesh.arch(),
                        math_fidelity=fidelity,
                        math_approx_mode=False,
                        fp32_dest_acc_en=False,
                        packer_l1_acc=True,
                    )
                    outputs[f"{fidelity.name}_N{n}"] = ttnn.experimental.minimal_matmul(
                        x, w, config=config, compute_kernel_config=compute, dtype=ttnn.bfloat16
                    )
                reference, actual = [], {name: [] for name in outputs}
                for rank in range(4):
                    xx = ttnn.to_torch(ttnn.get_device_tensors(x)[rank]).reshape(-1, x.shape[-1])[selected].double()
                    ww = ttnn.to_torch(ttnn.get_device_tensors(w)[rank]).squeeze().double()
                    reference.append(xx @ ww)
                    for name, value in outputs.items():
                        actual[name].append(
                            ttnn.to_torch(ttnn.get_device_tensors(value)[rank])
                            .reshape(-1, value.shape[-1])[selected]
                            .double()
                        )
                ref = sum(reference)
                record = {
                    "layer": index,
                    "role": role,
                    "rows": selected,
                    "weight_dtype": str(w.dtype),
                    "input_dtype": str(x.dtype),
                    "oracle": "FP64 dot of actual dequantized TT inputs and weights; partial ranks summed in FP64",
                    "variants": {},
                }
                for name, ranks in actual.items():
                    value = sum(ranks)
                    record["variants"][name] = {
                        "sum": metric(ref, value),
                        "per_row": {str(row): metric(ref[i], value[i]) for i, row in enumerate(selected)},
                        "per_rank": [metric(r, a) for r, a in zip(reference, ranks)],
                    }
                a, b = sum(actual["LoFi_N16"]), sum(actual["LoFi_N8"])
                record["N16_vs_N8"] = {"equal": bool(torch.equal(a, b)), "metrics": metric(a, b)}
                reverse = (torch.arange(a.shape[-1]) // 32) % 12 >= 8
                record["N16_vs_N8"]["forward_columns_equal_per_rank"] = [
                    bool(torch.equal(aa[:, ~reverse], bb[:, ~reverse]))
                    for aa, bb in zip(actual["LoFi_N16"], actual["LoFi_N8"])
                ]
                record["N16_vs_N8"]["reverse_columns"] = metric(a[:, reverse], b[:, reverse])
                assert all(record["N16_vs_N8"]["forward_columns_equal_per_rank"])
                results.append(record)
                Path("models/autoports/ifm_k2_horizon_7b/doc/full_model/projection_fixed_input.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
                print(
                    index, role, record["N16_vs_N8"], {k: v["sum"] for k, v in record["variants"].items()}, flush=True
                )
                return out

            layer._linear = inspected
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
