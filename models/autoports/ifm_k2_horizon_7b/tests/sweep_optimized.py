"""Same-harness real-weight, real embedding-activation candidate comparisons."""

import argparse
import dataclasses
import gc
import hashlib
import inspect
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, DynamicCache

import ttnn
from models.autoports.ifm_k2_horizon_7b.tt.fused_decoder import FusedDecoder
from models.autoports.ifm_k2_horizon_7b.tt.optimized_decoder import MatmulGeometry, OptimizedDecoder, PrecisionPolicy

from .run_functional import MODEL, REVISION, load_reference, pcc, to_device


def real_activations(count):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    text = (
        "A research team studies how light travels through water. They record the measurements, "
        "compare the results with a mathematical model, and explain the uncertainty. "
        "Write a Python function that computes a running average. The history of cities "
        "includes migration, trade, science, and changing forms of government. "
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = torch.tensor((ids * math.ceil(count / len(ids)))[:count])
    index = json.loads(Path(hf_hub_download(MODEL, "model.safetensors.index.json", revision=REVISION)).read_text())
    path = hf_hub_download(MODEL, index["weight_map"]["model.embed_tokens.weight"], revision=REVISION)
    with safe_open(path, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight")[ids].bfloat16()
    return embeddings


def policy_for(name):
    if name == "default" or name.startswith(
        (
            "fastattention_",
            "prefillnorm",
            "l1prefill_",
            "swiglu_",
            "prefillcompute_",
            "separateqkv_",
            "fusedconfig_",
            "fusedl1_",
        )
    ):
        return PrecisionPolicy()
    if name.startswith("policy_"):
        p = PrecisionPolicy()
        variants = {
            "read3qkv": dict(qkv_geometry=MatmulGeometry(64, 8, 3, False)),
            "read3mlp": dict(mlp_geometry=MatmulGeometry(64, 4, 3, False)),
            "read3both": dict(
                qkv_geometry=MatmulGeometry(64, 8, 3, False), mlp_geometry=MatmulGeometry(64, 4, 3, False)
            ),
            "separateprefill": dict(prefill_fused=False),
            "accurateprefill": dict(fast_prefill=False),
            "fp32prefill": dict(prefill_fp32=True),
            "attnhifi": dict(attention_fidelity="HiFi2"),
            "mlphifi": dict(mlp_fidelity="HiFi2"),
            "downhifi": dict(down_fidelity="HiFi2"),
            "attn8": dict(attention="bfloat8_b"),
            "mlp8": dict(mlp="bfloat8_b"),
            "down8": dict(down="bfloat8_b"),
            "kv16": dict(kv="bfloat16"),
            "actattn8": dict(attention_activation="bfloat8_b"),
            "actmlp8": dict(mlp_activation="bfloat8_b"),
        }
        return dataclasses.replace(p, **variants[name.removeprefix("policy_")])
    if name.startswith("tuned"):
        p = policy_for("dram_64_8_2_split_b4")
        return dataclasses.replace(
            p,
            qkv_geometry=MatmulGeometry(64, 8, 2, False),
            o_geometry=MatmulGeometry(64, 8, 2, False),
            mlp_geometry=MatmulGeometry(64, 4, 3 if "packed3" in name else 2, False),
            down_geometry=MatmulGeometry(64, 12, 2, False),
            packed_mlp="packed" in name,
            fused_gate="fused" in name,
        )
    if name.startswith(("prefill_", "norm_", "sdpa_")):
        return policy_for("tuned_split")
    p = PrecisionPolicy(
        attention="bfloat8_b",
        mlp="bfloat8_b",
        down="bfloat8_b",
        attention_fidelity="HiFi2",
        mlp_fidelity="HiFi2",
        down_fidelity="HiFi2",
        kv="bfloat16",
        packed_mlp=True,
        dram=False,
        norm_layout="dram",
        prefill_fp32=True,
        prefill_fused=False,
        fast_prefill=False,
    )
    if name.startswith("dram"):
        parts = name.split("_")
        # dram_<cores>_<Kblock>_<readers>_<packed|split>_<b4|b8>
        g = MatmulGeometry(int(parts[1]), int(parts[2]), int(parts[3]))
        return dataclasses.replace(
            p,
            attention="bfloat4_b" if parts[5] == "b4" else "bfloat8_b",
            mlp="bfloat4_b" if parts[5] == "b4" else "bfloat8_b",
            down="bfloat4_b" if parts[5] == "b4" else "bfloat8_b",
            attention_fidelity="LoFi",
            mlp_fidelity="LoFi",
            down_fidelity="LoFi",
            kv="bfloat8_b",
            packed_mlp=parts[4] == "packed",
            dram=True,
            qkv_geometry=g,
            o_geometry=g,
            mlp_geometry=g,
            down_geometry=g,
            norm_layout="l1",
        )
    variants = {
        "b8": {},
        "b8_lofi": dict(attention_fidelity="LoFi", mlp_fidelity="LoFi", down_fidelity="LoFi"),
        "mlp4": dict(mlp="bfloat4_b", mlp_fidelity="LoFi"),
        "down4": dict(down="bfloat4_b", down_fidelity="LoFi"),
        "mlpdown4": dict(mlp="bfloat4_b", down="bfloat4_b", mlp_fidelity="LoFi", down_fidelity="LoFi"),
        "attn4": dict(attention="bfloat4_b", attention_fidelity="LoFi"),
        "attn4_hifi2": dict(attention="bfloat4_b"),
        "all4": dict(
            attention="bfloat4_b",
            mlp="bfloat4_b",
            down="bfloat4_b",
            attention_fidelity="LoFi",
            mlp_fidelity="LoFi",
            down_fidelity="LoFi",
        ),
        "kv8": dict(kv="bfloat8_b"),
    }
    return dataclasses.replace(p, **variants[name])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["fused", "b8", "b8_lofi", "mlp4", "down4", "mlpdown4", "attn4", "attn4_hifi2", "all4", "kv8"],
    )
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefill-trace", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    config, state, hf, rope_fn = load_reference()
    acts = real_activations(args.batch * (args.seq + 1)).reshape(args.batch, args.seq + 1, 4096)
    x, dx = acts[:, : args.seq].contiguous(), acts[:, -1:].contiguous()
    rope = rope_fn(x, torch.arange(args.seq)[None].expand(args.batch, -1))
    dr = rope_fn(dx, torch.full((args.batch, 1), args.seq, dtype=torch.int64))
    hc = DynamicCache(config=config)
    hp = hf(x, position_embeddings=rope, past_key_values=hc)
    hd = hf(dx, position_embeddings=dr, past_key_values=hc)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:
        tx = to_device(x[None], mesh)
        tr = tuple(to_device(r[:, None], mesh) for r in rope)
        td = to_device(dx.transpose(0, 1)[None], mesh)
        rr = tuple(to_device(r[None].repeat(1, 1, 32, 1), mesh) for r in dr)
        pos = to_device(torch.full((args.batch,), args.seq, dtype=torch.int32), mesh, True)
        pages = math.ceil((args.seq + 128) / 32)
        torch.manual_seed(813)
        page_host = torch.randperm(args.batch * pages).reshape(args.batch, pages).int()
        table = to_device(page_host, mesh, True)
        for name in args.candidates:
            row = {
                "candidate": name,
                "seq": args.seq,
                "batch": args.batch,
                "activation_source": "checkpoint token embeddings, fixed natural-language token sequence",
            }
            print("START", name, flush=True)
            candidate = None
            trace = None
            caches = None
            try:
                cls = FusedDecoder if name == "fused" else OptimizedDecoder
                if name.startswith(
                    (
                        "prefill_",
                        "norm_",
                        "sdpa_",
                        "fastattention_",
                        "prefillnorm",
                        "l1prefill_",
                        "swiglu_",
                        "prefillcompute_",
                        "separateqkv_",
                        "fusedconfig_",
                        "fusedl1_",
                    )
                ):
                    from .optimized_candidates import candidate_class

                    cls = candidate_class(name)
                policy = None if name == "fused" else policy_for(name)
                candidate = cls.from_state_dict(
                    state,
                    hf_config=config,
                    layer_idx=0,
                    mesh_device=mesh,
                    **({} if policy is None else {"policy": policy}),
                )
                row["policy"] = None if policy is None else dataclasses.asdict(policy)
                row["implementation"] = cls.__module__ + "." + cls.__name__
                row["runtime_sha256"] = hashlib.sha256(
                    Path(inspect.getfile(OptimizedDecoder if name != "fused" else cls)).read_bytes()
                ).hexdigest()
                caches = tuple(
                    ttnn.zeros(
                        (args.batch * pages, 8, 32, 128),
                        dtype=getattr(candidate, "kv_dtype", ttnn.bfloat16),
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                    )
                    for _ in range(2)
                )
                kw = dict(rope=tr, kv_cache=caches, page_table=table, plan=candidate.prepare_prefill(seq_len=args.seq))
                out = candidate.prefill_forward(tx, **kw)
                host = ttnn.to_torch(out).reshape_as(hp)
                row["prefill_pcc"] = pcc(host, hp)
                out.deallocate(True)
                times = []
                for _ in range(3):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    out = candidate.prefill_forward(tx, **kw)
                    ttnn.synchronize_device(mesh)
                    times.append((time.perf_counter() - start) * 1e6)
                    out.deallocate(True)
                row["prefill_host_us"] = statistics.median(times)
                if args.prefill_trace:
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    out = candidate.prefill_forward(tx, **kw)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    times = []
                    for _ in range(5):
                        start = time.perf_counter()
                        for _ in range(10):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        times.append((time.perf_counter() - start) * 1e6 / 10)
                    row["prefill_traced_host_us"] = statistics.median(times)
                    row["prefill_traced_pcc"] = pcc(ttnn.to_torch(out).reshape_as(hp), hp)
                    ttnn.release_trace(mesh, trace)
                    trace = None
                    out.deallocate(True)
                dkw = dict(rope=rr, kv_cache=caches, page_table=table, current_pos=pos)
                out = candidate.decode_forward(td, **dkw)
                host = ttnn.to_torch(out)
                row["decode_pcc"] = pcc(host, hd)
                out.deallocate(True)
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = candidate.decode_forward(td, **dkw)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    for _ in range(100):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times.append((time.perf_counter() - start) * 1e6 / 100)
                row["decode_traced_host_us"] = statistics.median(times)
                row["decode_traced_samples_us"] = times
                row["deterministic"] = torch.equal(host, ttnn.to_torch(out))
                row["passed"] = min(row["prefill_pcc"], row["decode_pcc"]) >= 0.995 and row["deterministic"]
                out.deallocate(True)
            except Exception as exc:
                row.update(passed=False, error=str(exc))
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                candidate = None
                caches = None
                gc.collect()
            rows.append(row)
            Path(args.output).write_text(
                json.dumps(
                    {
                        "invocation": sys.argv,
                        "records": rows,
                        "timing_basis": "host, 5 groups of 100 async trace replays + sync; prefill eager warmed; full real dimensions",
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(row), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
