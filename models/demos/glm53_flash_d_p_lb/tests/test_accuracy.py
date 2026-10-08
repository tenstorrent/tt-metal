# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Long-context accuracy of the whole model on the device, against the text itself (no golden needed): the canonical
prompt (spec target.seq tokens, the model reciting A Tale of Two Cities) in target.chunk chunks from position 0; for
every row, is the next token of the text the device's top-1 / in its top-5?

The LM head runs on the device for this test: lm_head [V, H] bf16 sharded by vocab over every chip (V / n rows each),
the final-norm output gathered to all rows, logits per chip [S, V / n] (HiFi4, fp32 acc) -> ttnn.topk(5) per chip,
the 8 x 5 candidates merged on the host. Checked once against the host LM head (fp32) on a sample of rows.

Per chunk: top1 / top5 vs the text (also over the chunk's first and second half), and the device argmax ids
(generated/glm53_flash_d_p_lb/accuracy_<tag>.pt) for comparison with the CPU golden's per-chunk numbers. Also the
per-chip DRAM use after load and after the first chunk.

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONPATH=$PWD \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml GLM_ACC_TAG=bfp4 \\
    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_accuracy.py -s
"""

import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
OUT = Path(__file__).resolve().parents[4] / "generated" / "glm53_flash_d_p_lb"
TOPK = 5
MC = ttnn.DRAM_MEMORY_CONFIG


def dram_gib(mesh) -> dict:
    """Per-chip DRAM allocated / free (GiB), from the allocator's memory view."""
    out = {}
    try:
        for k, dev in enumerate(mesh.get_devices()):
            v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
            out[k] = (
                round(v.num_banks * v.total_bytes_allocated_per_bank / 2**30, 2),
                round(v.num_banks * v.total_bytes_free_per_bank / 2**30, 2),
            )
    except Exception as e:  # mesh-level view only
        try:
            v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            out["mesh"] = (
                round(v.num_banks * v.total_bytes_allocated_per_bank / 2**30, 2),
                round(v.num_banks * v.total_bytes_free_per_bank / 2**30, 2),
            )
        except Exception as e2:
            out["error"] = f"{e} / {e2}"
    return out


class DeviceLmHead:
    """lm_head [V, H] sharded by vocab over the n chips; top-k per chip on the device, merged on the host."""

    def __init__(self, mesh, weight: torch.Tensor):
        self.mesh, self.n = mesh, mesh.get_num_devices()
        self.vocab, hidden = weight.shape
        assert self.vocab % (self.n * ttnn.TILE_SIZE) == 0, (self.vocab, self.n)
        self.per = self.vocab // self.n
        self.w = ttnn.from_torch(
            weight.to(torch.bfloat16).T.contiguous().reshape(1, 1, hidden, self.vocab),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=MC,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        self.cfg = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )

    def topk(self, hidden_all: ttnn.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden_all [1, 1, S, H] replicated -> (values [S, k], global ids [S, k]) on the host."""
        lg = ttnn.linear(hidden_all, self.w, dtype=ttnn.bfloat16, compute_kernel_config=self.cfg, memory_config=MC)
        vals, idx = ttnn.topk(lg, k=TOPK, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(lg)
        vs = [ttnn.to_torch(t).float().reshape(-1, TOPK) for t in ttnn.get_device_tensors(vals)]
        ix = [
            ttnn.to_torch(t).to(torch.int64).reshape(-1, TOPK) + d * self.per
            for d, t in enumerate(ttnn.get_device_tensors(idx))
        ]
        ttnn.deallocate(vals)
        ttnn.deallocate(idx)
        v, i = torch.cat(vs, dim=-1), torch.cat(ix, dim=-1)  # [S, n k]
        o = v.argsort(dim=-1, descending=True)[:, :TOPK]
        return torch.gather(v, 1, o), torch.gather(i, 1, o)


@mesh_parametrize
def test_accuracy(mesh_device):
    from models.demos.common.bringup.reference.prompt import tokens as prompt_tokens
    from models.demos.glm53_flash_d_p.reference.weights import WeightLoader
    from models.demos.glm53_flash_d_p.tt.common import gather_rows

    seq = int(os.environ.get("GLM_ACC_SEQ", S.get("target.seq")))
    chunk = int(os.environ.get("GLM_ACC_CHUNK", S.get("target.chunk")))
    tag = os.environ.get("GLM_ACC_TAG", "run")
    toks = prompt_tokens(S, seq).to(torch.long)
    layers = S.layers()

    t0 = time.time()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=True)
    host_head = model._lm_head  # fp32 [V, H], for the cross-check
    head = DeviceLmHead(mesh_device, WeightLoader(model.path).get("lm_head.weight"))
    mem_load = dram_gib(mesh_device)
    print(f"loaded in {time.time() - t0:.0f}s; DRAM per chip (alloc GiB, free GiB) after load: {mem_load}", flush=True)

    per_chunk, ids_all = [], []
    mem_run = None
    for c, start in enumerate(range(0, seq, chunk)):
        t = time.time()
        h = model.embed(toks[start : start + chunk])
        for i in layers:
            h2 = model.layer(i, h, start, None)
            model.free(h)
            h = h2
        hid = model.final_norm(h)
        model.free(h)
        if mem_run is None:
            mem_run = dram_gib(mesh_device)
        hall = gather_rows(hid) if model.model.layout == "split" else hid
        vals, ids = head.topk(hall)
        if c == 0:  # cross-check the device head against the host fp32 head on a sample of rows
            rows = list(range(0, chunk, 97))
            ref = torch.nn.functional.linear(model.to_host(hid)[rows], host_head).topk(TOPK, dim=-1).indices
            agree = float((ref[:, 0] == ids[rows, 0]).float().mean())
            print(f"device LM head vs host fp32 on {len(rows)} rows: top1 agree {agree:.3f}", flush=True)
            assert agree > 0.95, agree
        if hall is not hid:
            ttnn.deallocate(hall)
        model.free(hid)
        n = min(chunk, seq - 1 - start)  # rows with a next token in the text
        want = toks[start + 1 : start + 1 + n]
        hit1 = ids[:n, 0] == want
        hit5 = (ids[:n] == want[:, None]).any(-1)
        half = n // 2
        rec = {
            "start": start,
            "top1": round(float(hit1.float().mean()), 4),
            "top5": round(float(hit5.float().mean()), 4),
            "top1_first_half": round(float(hit1[:half].float().mean()), 4),
            "top1_second_half": round(float(hit1[half:].float().mean()), 4),
            "seconds": round(time.time() - t, 2),
        }
        per_chunk.append(rec)
        ids_all.append(ids[:, 0].clone())
        print(
            f"chunk [{start:6d},{start + chunk:6d}) top1 {rec['top1']:.4f} top5 {rec['top5']:.4f} "
            f"(halves {rec['top1_first_half']:.4f} / {rec['top1_second_half']:.4f})",
            flush=True,
        )

    hits = sum(r["top1"] * min(chunk, seq - 1 - r["start"]) for r in per_chunk) / (seq - 1)
    print(f"\nall {seq - 1} rows: top1 {hits:.4f}", flush=True)
    print(f"DRAM per chip after the first chunk (alloc GiB, free GiB): {mem_run}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"argmax": torch.cat(ids_all), "per_chunk": per_chunk}, OUT / f"accuracy_{tag}.pt")
    (OUT / f"accuracy_{tag}.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "seq": seq,
                "chunk": chunk,
                "top1_all": round(hits, 4),
                "per_chunk": per_chunk,
                "dram_after_load": mem_load,
                "dram_after_chunk": mem_run,
            },
            indent=1,
        )
    )
