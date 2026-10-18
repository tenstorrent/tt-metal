"""Full-context PLE component harness with bounded host/device activation storage.

Run from checkout root with the active TTNN environment:
  TT_METAL_TRACE_ALLOC_TRACKING=1 python -m \
    models.autoports.qwen_qwen3_8_flash_next.tests.ple_long_context

Default lengths are the actual HF 262144 limit and the non-aligned 262143 tail.
The component PCC is diagnostic; decoder-layer acceptance still requires .995
on complete layer 1. No full-context residual tensor or host runtime fallback is
used. HF comparisons and input refreshes are explicit harness boundaries.
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.test_ple import CHECKPOINT, LazyCheckpoint, pcc
from models.autoports.qwen_qwen3_8_flash_next.tt.ple import PLE


class MappedCheckpointTable(torch.nn.Module):
    """Independent HF-only reference; mmap shards, group lookups by storage shard."""

    def __init__(self, checkpoint):
        super().__init__()
        self.weight = torch.empty(0, dtype=torch.bfloat16)
        prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
        self.shards = [checkpoint[prefix + f"shard_{i}.weight"] for i in range(128)]

    def warm_pages(self):
        """CPU-reference-only sequential page faulting; no TT/device execution."""
        started = time.monotonic()
        checksum = 0.0
        touched = 0
        total = 0
        page_bytes = os.sysconf("SC_PAGE_SIZE")
        for index, shard in enumerate(self.shards):
            # One BF16 element per OS page. Materialize the sum so
            # all indexed pages are actually read before timing the reference.
            selected = shard.reshape(-1)[:: page_bytes // shard.element_size()]
            checksum += selected.double().sum().item()
            # Safetensors payload offsets need not be OS-page aligned; the
            # final element also touches a possible extra trailing page.
            checksum += shard.reshape(-1)[-1].double().item()
            touched += selected.numel() + 1
            total += shard.numel() * shard.element_size()
            if (index + 1) % 16 == 0:
                print(
                    json.dumps(
                        {
                            "event": "HF_REFERENCE_PAGE_WARMUP_PROGRESS",
                            "shards": index + 1,
                            "logical_bytes_scanned": total,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        return {
            "scope": "CPU reference only; not device execution/capacity evidence",
            "shards": len(self.shards),
            "page_stride_bytes": page_bytes,
            "elements_touched": touched,
            "logical_bytes_scanned": total,
            "elapsed_seconds": time.monotonic() - started,
            "checksum": checksum,
        }

    def forward(self, indices):
        flat = indices.reshape(-1)
        shards = torch.div(flat, 2500012, rounding_mode="floor")
        offsets = flat % 2500012
        output = torch.empty(flat.numel(), 160, dtype=torch.bfloat16)
        for shard in shards.unique().tolist():
            selected = torch.nonzero(shards == shard).flatten()
            output[selected] = self.shards[shard][offsets[selected]]
        return output.reshape(*indices.shape, 160)


class StreamingPCC:
    def __init__(self):
        self.n = 0
        self.sa = self.sb = self.saa = self.sbb = self.sab = 0.0
        self.minimum_chunk_pcc = 1.0

    def update(self, a, b):
        a, b = a.double().reshape(-1), b.double().reshape(-1)
        self.n += a.numel()
        self.sa += a.sum().item()
        self.sb += b.sum().item()
        self.saa += a.square().sum().item()
        self.sbb += b.square().sum().item()
        self.sab += (a * b).sum().item()
        score = pcc(a, b)
        assert math.isfinite(score), "Nonfinite component PCC"
        self.minimum_chunk_pcc = min(score, self.minimum_chunk_pcc)

    def result(self):
        numerator = self.sab - self.sa * self.sb / self.n
        denominator = math.sqrt((self.saa - self.sa**2 / self.n) * (self.sbb - self.sb**2 / self.n))
        return {
            "pcc": numerator / denominator,
            "minimum_chunk_pcc": self.minimum_chunk_pcc,
            "compared_elements": self.n,
        }


def device_buffer(host, dtype, mesh, *, tiled=False):
    return ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT if tiled else ttnn.ROW_MAJOR_LAYOUT, device=mesh)


def copy_input(host, target):
    source = ttnn.from_torch(host, dtype=target.dtype, layout=target.layout)
    ttnn.copy_host_to_device_tensor(source, target)


def lane(value):
    out = torch.zeros(1, 32, dtype=torch.int32)
    out[0, 0] = value
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[262144, 262143])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--decode-replays", type=int, default=3)
    parser.add_argument(
        "--warm-reference-pages",
        action="store_true",
        help="Sequentially fault all real HF table pages on the CPU before comparisons",
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--weight-cache",
        type=Path,
        default=Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ple_long_context.json"),
    )
    args = parser.parse_args()
    checkpoint = LazyCheckpoint(args.checkpoint)
    cfg = AutoConfig.from_pretrained(checkpoint.path).text_config
    assert cfg.max_position_embeddings == 262144
    assert args.chunk_size > 0 and args.decode_replays > 0
    assert all(0 < n <= cfg.max_position_embeddings for n in args.lengths)
    torch.set_num_threads(8)
    with torch.device("meta"):
        reference = Qwen4ExpTextPLELayer(cfg, 1, 0)
    prefix = "model.language_model.layers.1.ple."
    weights = {
        k[len(prefix) :]: checkpoint[k]
        for k in checkpoint.index
        if k.startswith(prefix) and "ngram_embedding.shard_" not in k
    }
    missing, extra = reference.load_state_dict(weights, strict=False, assign=True)
    assert missing == ["ple_embedding.ngram_embedding.weight"] and not extra
    reference.ple_embedding.ngram_embedding = MappedCheckpointTable(checkpoint)
    reference.eval()
    embedding = checkpoint["model.language_model.embed_tokens.weight"]
    generator = torch.Generator().manual_seed(731)
    tokens = torch.randint(0, cfg.vocab_size, (max(args.lengths) + 1,), generator=generator)
    eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
    tokens[100::101] = eos
    shapes = {1, args.chunk_size, *[n % args.chunk_size for n in args.lengths if n % args.chunk_size]}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    results = []
    try:
        module = PLE.from_state_dict(checkpoint, hf_config=cfg, mesh_device=mesh, cache_dir=args.weight_cache)
        page_warmup = reference.ple_embedding.ngram_embedding.warm_pages() if args.warm_reference_pages else None
        if page_warmup is not None:
            print(json.dumps({"event": "HF_REFERENCE_PAGE_WARMUP_COMPLETE", **page_warmup}), flush=True)
        cache = module.make_cache(max_requests=1)
        inputs = {}
        for seq in sorted(shapes):
            module.prepare(batch=1, physical_seq_len=seq)
            inputs[seq] = {
                "x": device_buffer(
                    torch.zeros(1, 1, seq, 10240, dtype=torch.bfloat16), ttnn.bfloat16, mesh, tiled=True
                ),
                "tokens": device_buffer(torch.zeros(1, seq, dtype=torch.int32), ttnn.uint32, mesh),
                "positions": device_buffer(lane(0), ttnn.uint32, mesh),
                "slots": device_buffer(lane(0), ttnn.uint32, mesh),
                "valid_lengths": device_buffer(lane(seq), ttnn.uint32, mesh),
            }

        def forward(buffers):
            return module.forward(
                buffers["x"],
                tokens=buffers["tokens"],
                positions=buffers["positions"],
                slots=buffers["slots"],
                valid_lengths=buffers["valid_lengths"],
                cache=cache,
            )

        def refresh(buffers, ids, position):
            x = embedding[ids].repeat(1, 1, 4)
            copy_input(x.unsqueeze(0), buffers["x"])
            copy_input(ids.to(torch.int32), buffers["tokens"])
            copy_input(lane(position), buffers["positions"])
            return x

        # Every supported shape and code path compiles before any live trace.
        for buffers in inputs.values():
            warm = forward(buffers)
            ttnn.synchronize_device(mesh)
            del warm
        for length in args.lengths:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
                trace = None
            hf_cache = DynamicCache(config=cfg)
            statistics = StreamingPCC()
            started = time.monotonic()
            for start in range(0, length, args.chunk_size):
                seq = min(args.chunk_size, length - start)
                ids = tokens[start : start + seq].reshape(1, seq)
                x = refresh(inputs[seq], ids, start)
                expected = reference(x, ids, hf_cache)
                output = forward(inputs[seq])
                actual = ttnn.to_torch(output).squeeze(0)
                statistics.update(actual, expected)
                del output
                if start % (args.chunk_size * 64) == 0 or start + seq == length:
                    print(
                        json.dumps(
                            {
                                "event": "PLE_CONTEXT_PROGRESS",
                                "length": length,
                                "processed": start + seq,
                                **statistics.result(),
                            }
                        ),
                        flush=True,
                    )
            row = {
                "component": "PLE",
                "kind": "prefill",
                "length": length,
                "physical_chunk": args.chunk_size,
                "full_real_table_shards": len(module.tables),
                "elapsed_seconds": time.monotonic() - started,
                **statistics.result(),
            }
            assert row["pcc"] >= 0.99, row  # Component diagnostic floor, not layer acceptance.
            results.append(row)
            if length < cfg.max_position_embeddings:
                # Snapshot TT state at the explicit prefill/decode harness boundary.
                # Restore after capture and between alternatives from the same prefix.
                saved_tokens = ttnn.to_torch(cache.token_history)
                saved_conv = ttnn.to_torch(cache.conv_history)
                decode = inputs[1]
                ids = tokens[length : length + 1].reshape(1, 1)
                refresh(decode, ids, length)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                traced_output = forward(decode)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                decode_scores = []
                first_actual = None
                for replay in range(args.decode_replays):
                    ids = torch.tensor([[int(tokens[length]) if replay % 2 == 0 else eos]])
                    x = refresh(decode, ids, length)
                    copy_input(saved_tokens, cache.token_history)
                    copy_input(saved_conv, cache.conv_history)
                    expected = reference(x, ids, copy.deepcopy(hf_cache))
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    actual = ttnn.to_torch(traced_output).squeeze(0)
                    score = pcc(actual, expected)
                    assert math.isfinite(score) and score >= 0.99, score
                    if replay == 0:
                        first_actual = actual.clone()
                    elif replay % 2 == 0:
                        torch.testing.assert_close(actual, first_actual, rtol=0, atol=0)
                    decode_scores.append(score)
                    print(
                        json.dumps(
                            {
                                "event": "PLE_LONG_DECODE_REPLAY",
                                "position": length,
                                "replay": replay,
                                "token": ids.item(),
                                "pcc": score,
                            }
                        ),
                        flush=True,
                    )
                results.append(
                    {
                        "component": "PLE",
                        "kind": "traced_decode",
                        "position": length,
                        "context_consumed": length + 1,
                        "trace_id": str(trace),
                        "replay_pcc": decode_scores,
                    }
                )
                ttnn.release_trace(mesh, trace)
                trace = None
                del traced_output
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "hf_advertised_context": cfg.max_position_embeddings,
                        "scope": "PLE component only; complete decoder .995 PCC gate remains independent",
                        "reference_page_warmup": page_warmup,
                        "input_distribution": "real token embedding repeated into4 hyper-streams; seeded valid vocabulary IDs and periodic EOS",
                        "results": results,
                    },
                    indent=2,
                )
                + "\n"
            )
        print(json.dumps({"event": "PLE_CONTEXT_RUN_COMPLETE", "artifact": str(args.output)}), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
