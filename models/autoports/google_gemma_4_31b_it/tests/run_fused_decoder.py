"""Layer-only real-weight validation; all host conversions are harness boundaries."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.reference import selected_reference
from models.autoports.google_gemma_4_31b_it.tt.fused_decoder import HF_MODEL, HF_REVISION, FusedDecoder


def weights(layer):
    idx = json.loads(Path(hf_hub_download(HF_MODEL, "model.safetensors.index.json", revision=HF_REVISION)).read_text())
    prefix = f"model.language_model.layers.{layer}."
    keys = [k for k in idx["weight_map"] if k.startswith(prefix)]
    out = {}
    for shard in set(idx["weight_map"][k] for k in keys):
        with safe_open(hf_hub_download(HF_MODEL, shard, revision=HF_REVISION), framework="pt") as f:
            for k in keys:
                if idx["weight_map"][k] == shard:
                    out[k.removeprefix(prefix)] = f.get_tensor(k)
    return out


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run(args):
    torch.set_num_threads(16)
    torch.manual_seed(123)
    cfg = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    cfg._attn_implementation = "eager"
    if args.synthetic:
        stats = json.loads((Path(__file__).parents[1] / "doc/functional_decoder/weight_stats.json").read_text())
        rows = stats[cfg.layer_types[args.layer]]["tensors"]
        state = {r["name"]: (torch.randn(r["shape"]) * r["std"] + r["mean"]).to(torch.bfloat16) for r in rows}
    else:
        state = weights(args.layer)
    with torch.device("meta"):
        ref = Gemma4TextDecoderLayer(cfg, args.layer)
    ref.load_state_dict(state, assign=True)
    ref.eval()
    rope = Gemma4TextRotaryEmbedding(cfg)
    kind = cfg.layer_types[args.layer]
    b, s = args.batch, args.seq
    x = torch.randn(b, s + 1, cfg.hidden_size, dtype=torch.bfloat16)
    pos = torch.arange(s + 1).expand(b, -1)
    cos, sin = rope(x, pos, layer_type=kind)
    indices = list(range(max(0, s - 32), s + 1))
    selected = selected_reference(ref, x, cos, sin, indices)
    if s <= 4096:
        mask = torch.arange(s + 1)[None, :] <= torch.arange(s + 1)[:, None]
        if kind == "sliding_attention":
            mask &= torch.arange(s + 1)[None, :] > torch.arange(s + 1)[:, None] - cfg.sliding_window
        mask = torch.where(mask, 0.0, float("-inf")).to(torch.bfloat16)[None, None]
        with torch.no_grad():
            expected = ref(x, position_embeddings=(cos, sin), attention_mask=mask, shared_kv_states={})
        reference_pcc = pcc(selected, expected[:, indices])
        print("REDUCED_REFERENCE_CONTROL_PCC", reference_pcc, flush=True)
        assert reference_pcc >= 0.9999
    else:
        expected = None
    varying_positions = [s - 1 - (i % min(s, 7)) for i in range(b)]
    varied_expected = torch.cat(
        [
            selected_reference(
                ref, x[i : i + 1, : pos + 1], cos[i : i + 1, : pos + 1], sin[i : i + 1, : pos + 1], [pos]
            )
            for i, pos in enumerate(varying_positions)
        ],
        dim=0,
    ).flip(0)
    print("HF_REFERENCE_READY", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)

    def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh)

    try:
        model = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=args.layer, mesh_device=mesh)

        def audit(method):
            def call(*a, **kw):
                with (
                    patch("ttnn.from_torch", side_effect=AssertionError("runtime from_torch")),
                    patch("ttnn.to_torch", side_effect=AssertionError("runtime to_torch")),
                    patch("ttnn.as_tensor", side_effect=AssertionError("runtime as_tensor")),
                ):
                    return method(*a, **kw)

            return call

        model.prefill_forward = audit(model.prefill_forward)
        model.decode_forward = audit(model.decode_forward)
        blocks = (s + 32) // 32
        pages = (torch.randperm(b * blocks) + 3).reshape(b, blocks).to(torch.int32)
        page = tt(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cache = model.allocate_cache(num_blocks=b * blocks + 3)
        out = model.prefill_forward(
            tt(x[:, :s].unsqueeze(1)),
            cos=tt(cos[:, :s].unsqueeze(1)),
            sin=tt(sin[:, :s].unsqueeze(1)),
            kv_cache=cache,
            page_table=page,
        )
        actual = ttnn.to_torch(out).reshape(b, s, -1)
        prefill_pcc = (
            pcc(actual[:, indices[:-1]], selected[:, :-1]) if expected is None else pcc(actual, expected[:, :s])
        )
        print("PREFILL_PCC", prefill_pcc, flush=True)
        if args.extended:
            assert s >= 2049
            cache = model.allocate_cache(num_blocks=b * blocks + 3)
            first = model.prefill_forward(
                tt(x[:, :1024].unsqueeze(1)),
                cos=tt(cos[:, :1024].unsqueeze(1)),
                sin=tt(sin[:, :1024].unsqueeze(1)),
                kv_cache=cache,
                page_table=page,
            )
            tail = model.prefill_forward(
                tt(x[:, 1024:s].unsqueeze(1)),
                cos=tt(cos[:, 1024:s].unsqueeze(1)),
                sin=tt(sin[:, 1024:s].unsqueeze(1)),
                kv_cache=cache,
                page_table=page,
                chunk_start=1024,
            )
            resumed = torch.cat(
                [ttnn.to_torch(first).reshape(b, 1024, -1), ttnn.to_torch(tail).reshape(b, s - 1024, -1)], dim=1
            )
            resume_pcc = pcc(resumed, actual)
            print("RESUME_PCC", resume_pcc, flush=True)
            assert resume_pcc >= 0.995
            del first, tail
            repeated = model.prefill_forward(
                tt(x[:, :s].unsqueeze(1)),
                cos=tt(cos[:, :s].unsqueeze(1)),
                sin=tt(sin[:, :s].unsqueeze(1)),
                kv_cache=cache,
                page_table=page,
            )
            assert torch.equal(ttnn.to_torch(repeated).reshape(b, s, -1), actual), "Nondeterministic prefill"
            del repeated
            print("PREFILL_DETERMINISM_PASS", flush=True)
            boundary_results = []
            for case, length in enumerate([1, 31, 32, 33, 1023, 1024, 1025, 2049, 33]):
                fresh = torch.roll(x[:, :length], shifts=case + 1, dims=1) * (-1 if case % 2 == 0 else 1)
                changed_pages = torch.roll(pages.flatten(), case + 1).reshape_as(pages)
                fresh_page = tt(changed_pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                y = model.prefill_forward(
                    tt(fresh.unsqueeze(1)),
                    cos=tt(cos[:, :length].unsqueeze(1)),
                    sin=tt(sin[:, :length].unsqueeze(1)),
                    kv_cache=cache,
                    page_table=fresh_page,
                )
                reference = selected_reference(
                    ref, fresh, cos[:, :length], sin[:, :length], list(range(max(0, length - 32), length))
                )
                got = ttnn.to_torch(y).reshape(b, length, -1)[:, -reference.shape[1] :]
                score = pcc(got, reference)
                print("BOUNDARY", length, score, flush=True)
                assert score >= 0.995
                boundary_results.append(dict(length=length, pcc=score))
                del y, fresh_page
            # Restore original request after fresh-request/cache-ownership checks.
            out = model.prefill_forward(
                tt(x[:, :s].unsqueeze(1)),
                cos=tt(cos[:, :s].unsqueeze(1)),
                sin=tt(sin[:, :s].unsqueeze(1)),
                kv_cache=cache,
                page_table=page,
            )
        else:
            resume_pcc = None
            boundary_results = []
        if s == cfg.max_position_embeddings - 1 and not getattr(args, "skip_exact_maximum", False):
            # Exact maximum prefill plus a non-aligned near-maximum prompt on one instance.
            del out
            maximum = model.prefill_forward(
                tt(x.unsqueeze(1)), cos=tt(cos.unsqueeze(1)), sin=tt(sin.unsqueeze(1)), kv_cache=cache, page_table=page
            )
            last = ttnn.to_torch(maximum)[:, 0, -1:].reshape(b, 1, -1)
            maximum_pcc = pcc(last, selected[:, -1:])
            print("EXACT_MAXIMUM_PREFILL_PCC", maximum_pcc, flush=True)
            assert maximum_pcc >= 0.995
            del maximum
        else:
            maximum_pcc = None
        for cache_tensor in cache:
            sentinel = ttnn.slice(cache_tensor, (0, 0, 0, 0), (3, model.nkv, 32, model.head_dim))
            assert torch.count_nonzero(ttnn.to_torch(sentinel)) == 0, "Cache writes escaped owned pages"
            del sentinel
        dx = tt(x[:, s].reshape(1, 1, b, -1))
        dc = tt(cos[:, s].reshape(1, b, 1, -1))
        ds = tt(sin[:, s].reshape(1, b, 1, -1))
        cp = tt(torch.full((b,), s, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kw = dict(cos=dc, sin=ds, kv_cache=cache, page_table=page, current_pos=cp)
        interleaved = []
        if args.extended:
            other_cache = model.allocate_cache(num_blocks=b * blocks + 3)
            reuse_cases = [(31, 0), (32, 0), (33, 0), (1025, 0), (33, 1024)]
            if getattr(args, "new_bucket_case", False):
                reuse_cases.append((34, 0))
            for length, start in reuse_cases:
                inputs = tt(-x[:, start : start + length].unsqueeze(1))
                options = dict(
                    cos=tt(cos[:, start : start + length].unsqueeze(1)),
                    sin=tt(sin[:, start : start + length].unsqueeze(1)),
                    kv_cache=other_cache,
                    page_table=page,
                    chunk_start=start,
                )
                if length != 34:
                    warm = model.prefill_forward(inputs, **options)
                    del warm
                interleaved.append((inputs, options))
        out = model.decode_forward(dx, **kw)
        print("EAGER_DECODE_PCC", pcc(ttnn.to_torch(out).reshape(b, 1, -1), selected[:, -1:]), flush=True)
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = model.decode_forward(dx, **kw)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        previous = None
        for _ in range(3):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            decode_pcc = pcc(ttnn.to_torch(traced).reshape(b, 1, -1), selected[:, -1:])
            replay = ttnn.to_torch(traced).clone()
            if previous is not None:
                assert torch.equal(previous, replay), "Nondeterministic replay"
            previous = replay
            print("TRACED_DECODE_PCC", decode_pcc, flush=True)

        for inputs, options in interleaved:
            eager_output = model.prefill_forward(inputs, **options)
            # Consume the other request's output before replay can reuse temporary storage.
            ttnn.to_torch(eager_output)
            del eager_output
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            assert torch.equal(previous, ttnn.to_torch(traced)), "Interleaved prefill corrupted decode"
            print("INTERLEAVED_PREFILL_TRACE_REUSE", tid, inputs.shape[2], options["chunk_start"], flush=True)

        def refresh(value, dest, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(value, dtype=dtype, layout=layout), dest)

        rows = torch.arange(b)
        vp = torch.tensor(varying_positions)
        refresh(x[rows, vp].flip(0).reshape(1, 1, b, -1), dx)
        refresh(cos[rows, vp].flip(0).reshape(1, b, 1, -1), dc)
        refresh(sin[rows, vp].flip(0).reshape(1, b, 1, -1), ds)
        refresh(vp.flip(0).to(torch.int32), cp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        refresh(pages.flip(0), page, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        changed_pcc = pcc(ttnn.to_torch(traced).reshape(b, 1, -1), varied_expected)
        print("UPDATED_POSITIONS_AND_PAGE_ROWS_PCC", changed_pcc, flush=True)
        assert changed_pcc >= 0.995
        ttnn.release_trace(mesh, tid)
        assert min(prefill_pcc, decode_pcc) >= 0.995
        print("RUNTIME_CONVERSION_GUARDS_PASS", flush=True)
        if args.result:
            Path(args.result).write_text(
                json.dumps(
                    dict(
                        implementation=type(model).__module__ + "." + type(model).__name__,
                        layer_type=kind,
                        batch=b,
                        prefill_tokens=s,
                        decode_position=s,
                        prefill_pcc=prefill_pcc,
                        decode_pcc=decode_pcc,
                        real_weights=not args.synthetic,
                        reference_scope=(
                            "all_rows" if expected is not None else "last_32_prefill_rows_and_decode_with_all_prefix_KV"
                        ),
                        trace_replays=4 + len(interleaved),
                        interleaved_prefill_cases=len(interleaved),
                        trace_captures=1,
                        request_time_trace_releases=0,
                        updated_inputs_pcc=changed_pcc,
                        deterministic=True,
                        resume_pcc=resume_pcc,
                        boundaries=boundary_results,
                        maximum_prefill_pcc=maximum_pcc,
                    ),
                    indent=2,
                )
                + "\n"
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--result")
    p.add_argument("--extended", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--new-bucket-case", action="store_true")
    p.add_argument(
        "--skip-exact-maximum",
        action="store_true",
        help="Reuse unchanged exact-maximum prefill evidence during decode-only repair",
    )
    run(p.parse_args())
