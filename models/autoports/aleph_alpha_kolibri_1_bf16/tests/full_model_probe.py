# SPDX-License-Identifier: Apache-2.0
"""Reduced real-weight wrapper probe; no full-stack acceptance claims."""
import time

import torch

import ttnn

from ..tt.checkpoint import load_weights
from ..tt.model import KolibriModel
from .reference import ReferenceDecoder, norm


def main():
    torch.set_num_threads(8)
    begin = time.monotonic()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    try:
        model = KolibriModel(mesh, layer_indices=[0, 4], rope_capacity=1024)
        tt = model.tensor
        pages = torch.arange(32, dtype=torch.int32)[None]
        tables = {key: tt(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for key in ("full", "sliding")}
        caches = [
            tuple(tt(torch.zeros(32, 1, 32, 128, dtype=torch.bfloat16), ttnn.bfloat8_b) for _ in range(2))
            for _ in model.layers
        ]
        token = tt(torch.tensor([[42]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        pos = tt(torch.tensor([0], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        print("PROBE_SETUP", time.monotonic() - begin, flush=True)
        y = model.decode_forward(token, current_pos=pos, page_tables=tables, kv_cache=caches)
        actual = torch.cat([ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(y)], -1).reshape(1, 1, -1)
        emb = load_weights("model.embed_tokens.")["model.embed_tokens.weight"]
        x = emb[torch.tensor([[42]])]
        for idx in [0, 4]:
            prefix = f"model.layers.{idx}."
            ref = ReferenceDecoder({k.removeprefix(prefix): v for k, v in load_weights(prefix).items()}, idx)
            x = ref(x)
        x = norm(x, load_weights("model.norm.")["model.norm.weight"])
        expected = torch.nn.functional.linear(x.float(), load_weights("lm_head.")["lm_head.weight"].float())
        pcc = torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1].item()
        print(
            "PROBE_RESULT",
            {
                "pcc": pcc,
                "tt_top5": actual[0, 0].topk(5).indices.tolist(),
                "hf_top5": expected[0, 0].topk(5).indices.tolist(),
                "seconds": time.monotonic() - begin,
            },
            flush=True,
        )
        assert pcc > 0.99
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
