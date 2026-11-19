# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""H2 diagnostic: fixed physical full-attention prefill, never a production path.

Usage after loading a real ``MultichipDecoder`` and allocating its cache::

    adapter = NormalizedAttentionPrefill(layer, state=state, physical_bucket=64)
    adapter.refresh(start=32, valid_count=33, page_mapping=local_pages)
    output = adapter.forward(hidden_padded_to_64)

All device inputs are allocated in the constructor. ``refresh`` copies host data
into those same allocations, outside capture; ``forward`` returns the complete
physical bucket. The caller must discard dummy rows, without making a new
logical-shape device tensor in a sealed generator. Preparation must warm each
physical bucket before any capture. This adapter alone does not implement that
generator-wide preparation or residual/embedding/head staging.

Only aligned bodies/tails are implemented here. For an unaligned absolute start,
the existing layer-decode path must consume min(valid_count, (-start) % 32) real
tokens, using a persistent [1] position input refreshed outside capture. Never
replay cached prefix tokens. Then use this adapter for aligned bodies, retaining
the decoder's 16384 chunk cap. This is a diagnostic limitation, not a proposed
restriction on the public generator.

Suggested parent-owned device probe: use a real full-attention layer, private and
pooled states, and bucket 64. Warm starts 0/32 and counts 33/34 before sealing;
compare valid output rows and K/V with ``layer.prefill_forward`` on independent
identical caches. Then forbid cache misses and repeat 33 -> 34 -> 33 with runtime
start/page-map changes. Separately test bucket 128/count 65 (one wholly dummy
page), and max-context bucket 16384/start 250016/count 12000 under watcher. Compare
untouched pool slots and wholly dummy physical pages byte-for-byte. Those runs,
including coarse-bucket numerical equivalence, are deliberately NOT claimed here.
"""

from dataclasses import dataclass
from math import ceil

FULL_PREFILL_BUCKETS = (32, 64, 128, 480, 512, 768, 1024, 4096, 8192, 16384)


@dataclass(frozen=True)
class AttentionInputs:
    rope_indices: tuple[int, ...]
    fill_pages: tuple[int, ...]
    attention_pages: tuple[int, ...]
    start: int
    valid_count: int


def make_attention_inputs(
    *, physical_bucket, supported_context, local_page_count, start, valid_count, page_mapping, page_offset=0
):
    """Pure-host plan; page_mapping has local IDs, never pool-global IDs.

    Fill -1 is inserted AFTER adding the pool offset. SDPA instead receives safe
    aliases for unused virtual pages. Extra virtual capacity is context pages
    plus a whole physical bucket, rounded to eight pages: 8704 pages for the
    262144 context / 16384 bucket. It allocates no extra physical K/V storage.
    """
    if physical_bucket not in FULL_PREFILL_BUCKETS:
        raise ValueError(f"Unprepared full-attention bucket: {physical_bucket}")
    if supported_context <= 0 or local_page_count < ceil(supported_context / 32):
        raise ValueError("Cache region cannot cover supported_context")
    if start < 0 or start % 32 or not 0 < valid_count <= physical_bucket:
        raise ValueError("Aligned body required; consume the unaligned prologue with one-token decode")
    if start + valid_count > supported_context:
        raise ValueError("Logical interval exceeds supported context")
    mapping = tuple(int(page) for page in page_mapping)
    if len(mapping) != ceil(supported_context / 32):
        raise ValueError("Page mapping must have the configured fixed logical capacity")
    if page_offset < 0 or any(page < -1 or page >= local_page_count for page in mapping):
        raise ValueError("Page IDs must be local to the state's private/pool region, or -1 for unused pages")
    first_page, valid_end_page = start // 32, ceil((start + valid_count) / 32)
    if any(page < 0 for page in mapping[:valid_end_page]):
        raise ValueError("All causal prefix and newly written pages must have valid mappings")
    fill = tuple(
        mapping[first_page + index] + page_offset if index < ceil(valid_count / 32) else -1
        for index in range(physical_bucket // 32)
    )
    attention_capacity = ceil((ceil(supported_context / 32) + physical_bucket // 32) / 8) * 8
    # Local page zero belongs to this state, including for pooled cache regions.
    attention = tuple(page + page_offset if page >= 0 else page_offset for page in mapping)
    attention += (page_offset,) * (attention_capacity - len(attention))
    # Dummy query positions may exceed max context; embedding must never do so.
    rope = tuple(min(start + index, supported_context - 1) for index in range(physical_bucket))
    return AttentionInputs(rope, fill, attention, start, valid_count)


class NormalizedAttentionPrefill:
    """One stable binding and bucket; preserves the decoder's math/collectives.

    ``state`` remains caller-owned. This helper neither clears caches nor replaces
    their backing allocations. ``forward`` uses the decoder's existing norm,
    projection, rotary, reduction, residual and MLP implementations unchanged.
    """

    def __init__(self, layer, *, state, physical_bucket, supported_context=None):
        import torch

        import ttnn

        if layer.kind != "full_attention" or physical_bucket > layer.chunk_size:
            raise ValueError("Expected full-attention decoder with its existing chunk cap")
        self.layer, self.state, self.bucket = layer, state, physical_bucket
        self.context = supported_context or layer.config.max_position_embeddings
        if self.context > layer.config.max_position_embeddings:
            raise ValueError("Context exceeds the loaded RoPE table")
        self.page_count = state.page_count or state.key.shape[0]
        if state.page_offset + self.page_count > state.key.shape[0]:
            raise ValueError("State region extends outside physical K/V cache")
        if state.key.shape != state.value.shape:
            raise ValueError("K/V pool geometry differs")
        self._cache_binding = (id(state.key), id(state.value), state.page_offset, state.page_count)
        initial = make_attention_inputs(
            physical_bucket=self.bucket,
            supported_context=self.context,
            local_page_count=self.page_count,
            start=0,
            valid_count=1,
            page_mapping=range(ceil(self.context / 32)),
            page_offset=state.page_offset,
        )
        self._specs = {
            "rope_indices": (ttnn.uint32, lambda p: torch.tensor(p.rope_indices, dtype=torch.int32).reshape(1, -1)),
            "fill_pages": (ttnn.int32, lambda p: torch.tensor(p.fill_pages, dtype=torch.int32).reshape(1, -1)),
            "attention_pages": (
                ttnn.int32,
                lambda p: torch.tensor(p.attention_pages, dtype=torch.int32).reshape(1, -1),
            ),
            "start": (ttnn.int32, lambda p: torch.tensor([p.start], dtype=torch.int32)),
        }
        self.inputs = {
            name: ttnn.from_torch(
                pack(initial),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=layer.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(layer.device),
            )
            for name, (dtype, pack) in self._specs.items()
        }
        self.current = None

    def refresh(self, *, start, valid_count, page_mapping):
        """Host-side runtime update; no device allocation and no request slices."""
        import ttnn

        state = self.state
        if self._cache_binding != (id(state.key), id(state.value), state.page_offset, state.page_count):
            raise ValueError("Cache binding changed; construct and prepare a separate adapter before sealing")
        plan = make_attention_inputs(
            physical_bucket=self.bucket,
            supported_context=self.context,
            local_page_count=self.page_count,
            start=start,
            valid_count=valid_count,
            page_mapping=page_mapping,
            page_offset=state.page_offset,
        )
        for name, (dtype, pack) in self._specs.items():
            host = ttnn.from_torch(
                pack(plan),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.layer.device),
            )
            ttnn.copy_host_to_device_tensor(host, self.inputs[name])
        self.current = plan
        return plan

    def attention(self, normalized_hidden, *, project=True):
        """Device-only prefill attention with fixed-shape runtime position/maps."""
        import ttnn

        if self.current is None:
            raise RuntimeError("refresh must initialize the request before forward/warmup")
        layer, state, n = self.layer, self.state, self.bucket
        if tuple(normalized_hidden.shape) != (1, 1, n, layer.hidden_width):
            raise ValueError("Caller must supply the fixed physical hidden bucket")
        packed = layer.linear(normalized_hidden, "packed_input")
        qg, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.reshape(packed, (1, n, 2560)), num_heads=8, num_kv_heads=1, transpose_key=False
        )
        q = layer.norm(qg[:, :4, :, :], "self_attn.q_norm")
        k = layer.norm(k, "self_attn.k_norm")
        gate = ttnn.reshape(ttnn.permute(qg[:, 4:8, :, :], (0, 2, 1, 3)), (1, 1, n, 1024))
        rope = ttnn.reshape(
            ttnn.embedding(self.inputs["rope_indices"], layer.rope_decode, layout=ttnn.TILE_LAYOUT), (1, 1, n, 128)
        )
        q, k = (
            layer.rope(q, rope[:, :, :, :64], rope[:, :, :, 64:]),
            layer.rope(k, rope[:, :, :, :64], rope[:, :, :, 64:]),
        )
        # Cast to each actual cache dtype, preserving any paged-cache dtype split.
        ttnn.experimental.paged_fill_cache(state.key, ttnn.typecast(k, state.key.dtype), self.inputs["fill_pages"])
        ttnn.experimental.paged_fill_cache(state.value, ttnn.typecast(v, state.value.dtype), self.inputs["fill_pages"])
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            state.key,
            state.value,
            self.inputs["attention_pages"],
            chunk_start_idx_tensor=self.inputs["start"],
            scale=1 / 16,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=128
            ),
            compute_kernel_config=layer.compute,
        )
        out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, 1, n, 1024))
        out = ttnn.multiply(gate, out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return layer.reduce(layer.linear(out, "self_attn.o_proj")) if project else out

    def forward(self, hidden):
        """One complete layer, retaining physical output height including dummy rows."""
        import ttnn

        normalized = self.layer.norm(hidden, "input_layernorm")
        mixed = self.attention(normalized)
        return ttnn.to_memory_config(self.layer.finish(hidden, mixed), ttnn.DRAM_MEMORY_CONFIG)


def host_check():
    """Metadata/sentinel checks only: does not import TTNN or open devices."""
    plan = make_attention_inputs(
        physical_bucket=128,
        supported_context=256,
        local_page_count=8,
        start=32,
        valid_count=65,
        page_mapping=(7, 6, 5, 4, 3, 2, 1, 0),
        page_offset=16,
    )
    assert plan.fill_pages == (22, 21, 20, -1)
    assert min(plan.attention_pages) >= 16
    assert len(plan.rope_indices) == 128 and plan.rope_indices[0] == 32
    tail = make_attention_inputs(
        physical_bucket=16384,
        supported_context=262144,
        local_page_count=8192,
        start=250016,
        valid_count=12000,
        page_mapping=range(8192),
    )
    assert len(tail.attention_pages) == 8704
    assert max(tail.rope_indices) == 262143
    assert tail.fill_pages[375:] == (-1,) * 137
    assert (tail.start + len(tail.rope_indices) + 127) // 128 * 4 <= len(tail.attention_pages)
    print("HOST_PLAN_OK; hardware, numerical equivalence and program-cache invariance UNVERIFIED")


def main():
    """Real-weight diagnostic. The caller must wrap this command in timeout.

    Example: python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_normalized_attention
        --lengths 33,34,65 --output path/to/normalized_attention.json

    Use --host-check for metadata-only validation without importing TTNN.
    Numerical checks use the existing decoder-probe PCC bar, 0.995; untouched
    pages and unrelated pool slots must be bit equal. This is eager repair
    evidence, not trace, full-model accuracy, or physical-capacity evidence.
    """
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--host-check", action="store_true")
    parser.add_argument("--lengths", default="33,34,65")
    parser.add_argument("--starts", default="0,32")
    parser.add_argument("--ownership", choices=("private", "pooled", "both"), default="both")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.host_check:
        host_check()
        return
    if not args.output:
        parser.error("--output is required for the hardware diagnostic")
    lengths = [int(value) for value in args.lengths.split(",")]
    starts = [int(value) for value in args.starts.split(",")]
    if any(value <= 0 or value > FULL_PREFILL_BUCKETS[-1] for value in lengths):
        parser.error("Each --lengths entry must be between 1 and 16384")
    if any(value < 0 or value % 32 for value in starts):
        parser.error("Each --starts entry must be nonnegative and aligned to 32")

    import torch
    from transformers import AutoConfig

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import Checkpoint
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder

    torch.set_num_threads(8)
    torch.manual_seed(834)
    checkpoint = Checkpoint()
    config = AutoConfig.from_pretrained(checkpoint.path).text_config
    weights = checkpoint.tensors("model.language_model.layers.3.")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    try:
        decoder = MultichipDecoder.from_state_dict(weights, hf_config=config, layer_idx=3, mesh_device=mesh)

        def upload(value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def host_shards(value):
            return [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(value)]

        def compare(expected, actual):
            scores, errors, equals = [], [], []
            for a, b in zip(expected, actual, strict=True):
                if a.shape != b.shape:
                    raise AssertionError(f"Host comparison shape mismatch: {a.shape} vs {b.shape}")
                same = torch.equal(a, b)
                equals.append(same)
                scores.append(1.0 if same else torch.corrcoef(torch.stack((a.flatten(), b.flatten())))[0, 1].item())
                errors.append((a - b).abs().max().item())
            return {"pcc": scores, "max_abs": errors, "bit_equal": equals}

        ownerships = ("private", "pooled") if args.ownership == "both" else (args.ownership,)
        for valid in lengths:
            bucket = next(size for size in FULL_PREFILL_BUCKETS if size >= ceil(valid / 32) * 32)
            for start in starts:
                # Include safe physical cache padding for the reference and its
                # subsequent decode; this tiny test does not claim max context.
                context = ceil((start + bucket + 32) / 256) * 256
                if context > config.max_position_embeddings:
                    raise ValueError("This probe's padded reference allocation exceeds model context")
                pages = context // 32
                mapping = list(reversed(range(pages)))
                table = upload(
                    torch.tensor([mapping], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                for ownership in ownerships:
                    if ownership == "pooled":
                        reference_states = decoder.allocate_batch_state(batch=2, context=context)
                        normalized_states = decoder.allocate_batch_state(batch=2, context=context)
                        reference_state, normalized_state = reference_states[1], normalized_states[1]
                    else:
                        reference_state = decoder.allocate_state(context=context)
                        normalized_state = decoder.allocate_state(context=context)
                    adapter = NormalizedAttentionPrefill(
                        decoder, state=normalized_state, physical_bucket=bucket, supported_context=context
                    )
                    if start:
                        prefix = upload(torch.randn(1, 1, start, decoder.hidden_width, dtype=torch.bfloat16))
                        decoder.prefill_forward(prefix, state=reference_state, start_position=0, page_table=table)
                    # Copy only in this comparison setup. The adapter never
                    # resets, clears or substitutes caller-owned cache backing.
                    ttnn.copy(reference_state.key, normalized_state.key)
                    ttnn.copy(reference_state.value, normalized_state.value)
                    before = {name: host_shards(getattr(normalized_state, name)) for name in ("key", "value")}
                    real = torch.randn(1, 1, valid, decoder.hidden_width, dtype=torch.bfloat16)
                    physical = torch.randn(1, 1, bucket, decoder.hidden_width, dtype=torch.bfloat16) * 7
                    physical[:, :, :valid, :] = real
                    expected = decoder.prefill_forward(
                        upload(real), state=reference_state, start_position=start, page_table=table
                    )
                    plan = adapter.refresh(start=start, valid_count=valid, page_mapping=mapping)
                    got = adapter.forward(upload(physical))
                    record = {
                        "layer": 3,
                        "valid": valid,
                        "bucket": bucket,
                        "start": start,
                        "context": context,
                        "ownership": ownership,
                        "pool_offset": normalized_state.page_offset,
                        "key_dtype": str(normalized_state.key.dtype),
                        "value_dtype": str(normalized_state.value.dtype),
                        "fill_pages": list(plan.fill_pages),
                        "attention_pages": len(plan.attention_pages),
                        "output": compare(
                            host_shards(expected), [tensor[:, :, :valid, :] for tensor in host_shards(got)]
                        ),
                    }
                    wrote_pages = {page for page in plan.fill_pages if page >= 0}
                    untouched = [page for page in range(normalized_state.key.shape[0]) if page not in wrote_pages]
                    logical_pages = [page + normalized_state.page_offset for page in mapping]
                    logical_end = start + valid
                    for name in ("key", "value"):
                        actual_cache = host_shards(getattr(normalized_state, name))
                        expected_cache = host_shards(getattr(reference_state, name))
                        record[f"{name}_valid"] = compare(
                            [cache[logical_pages, 0].reshape(-1, 256)[:logical_end] for cache in expected_cache],
                            [cache[logical_pages, 0].reshape(-1, 256)[:logical_end] for cache in actual_cache],
                        )
                        record[f"{name}_untouched_pages"] = [
                            torch.equal(old[untouched], new[untouched])
                            for old, new in zip(before[name], actual_cache, strict=True)
                        ]
                    # The next real token overwrites the first dummy row. Its
                    # attention still must see the same valid prefix and cache.
                    next_row = upload(torch.randn(1, 1, 1, decoder.hidden_width, dtype=torch.bfloat16))
                    position = upload(
                        torch.tensor([logical_end], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    expected_next = decoder.decode_forward(
                        next_row, state=reference_state, page_table=table, current_position=position
                    )
                    actual_next = decoder.decode_forward(
                        next_row, state=normalized_state, page_table=table, current_position=position
                    )
                    record["decode_after"] = compare(host_shards(expected_next), host_shards(actual_next))
                    record["passed"] = all(
                        score >= 0.995
                        for key in ("output", "key_valid", "value_valid", "decode_after")
                        for score in record[key]["pcc"]
                    ) and all(record["key_untouched_pages"] + record["value_untouched_pages"])
                    records.append(record)
                    output_path.write_text(json.dumps(records, indent=2) + "\n")
                    print(json.dumps(record), flush=True)
        if not all(record["passed"] for record in records):
            raise AssertionError("Normalized attention diagnostic failed; inspect per-case comparisons")
    finally:
        output_path.write_text(json.dumps(records, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
