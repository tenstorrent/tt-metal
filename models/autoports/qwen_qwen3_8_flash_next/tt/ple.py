"""Full-domain PLE for the actual Qwen3.8-Flash-Next layer-1 contract.

Setup may read Torch checkpoint tensors. Runtime takes only device tensors.
All 128 shards remain resident as BFP4; no prompt-selected table or host misses.
Physical chunks are prepared before trace capture. Logical valid lengths and
request slots/positions are device inputs. EOS resets lexical history only.
"""
from dataclasses import dataclass
from pathlib import Path

import ttnn

from .ple_hash import ngram_indices
from .ple_lookup import lookup_rows


@dataclass
class PLECache:
    token_history: object  # UINT32 RM [requests,32]; older,newer,next_position
    conv_history: object  # BF16 RM [requests*9,10240]; oldest to newest


@dataclass
class PLEWorkspace:
    indices: object
    embeddings: object
    taps: object


class PLE:
    """Return the PLE injection, before the decoder's residual addition.

    ``forward(x, *, tokens, positions, slots, valid_lengths, cache)`` accepts
    BF16 TILE x [1,B,S,10240], UINT32 RM tokens [B,S], and UINT32 RM control
    tensors [B,32] (column zero is used). Logical lanes contain ``valid_lengths``
    tokens. Position zero clears that request's history; other positions must
    equal its saved next position. Slots must be unique within a call.
    Padding does not advance history. Returned padding is zero. ``prepare``
    must be called for every physical (B,S) before any live trace is captured.
    """

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, mesh_device, layer_idx=1, prefix=None, cache_dir=None):
        """Load a lazy checkpoint Mapping; only one table shard is read at once.

        Optional disk cache is keyed by the caller's checkpoint-specific
        ``cache_dir``; callers must not share that directory across revisions.
        """
        cfg = getattr(hf_config, "text_config", hf_config)
        if (
            cfg.hidden_size,
            cfg.hc_count,
            cfg.ple_embed_dim,
            cfg.ngram_size,
            cfg.heads_per_ngram,
            cfg.ple_conv_kernel_size,
        ) != (2560, 4, 2560, 3, 8, 4):
            raise ValueError("PLE requires the actual Qwen3.8-Flash-Next config")
        if layer_idx + 1 not in cfg.ple_layer_ids:
            raise ValueError("Selected decoder has no PLE")
        obj = cls()
        obj.mesh = mesh_device
        obj.eps = cfg.rms_norm_eps
        obj.eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
        obj.workspaces = {}
        prefix = prefix if prefix is not None else f"model.language_model.layers.{layer_idx}.ple."
        get = lambda key: state_dict[prefix + key]
        obj.multipliers = get("ple_embedding.layer_multipliers").tolist()
        obj.primes = get("ple_embedding.ngram_heads_vocab_sizes").tolist()
        obj.offsets = get("ple_embedding.ngram_heads_offsets").tolist()
        obj.tables = []
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
        for i in range(128):
            path = cache_dir / f"ple_shard_{i}.bfp4.tensorbin" if cache_dir else None
            if path is not None and path.is_file():
                table = ttnn.load_tensor(path, device=mesh_device)
            else:
                host = get(f"ple_embedding.ngram_embedding.shard_{i}.weight")
                if tuple(host.shape) != (2500012, 160):
                    raise ValueError(f"Unexpected checkpoint shard {i}: {tuple(host.shape)}")
                encoded = ttnn.from_torch(host, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
                if path is not None:
                    temporary = path.with_suffix(".partial.tensorbin")
                    ttnn.dump_tensor(temporary, encoded)
                    temporary.replace(path)
                table = ttnn.to_device(encoded, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                del host, encoded
            if tuple(table.shape) != (2500012, 160) or table.dtype != ttnn.bfloat4_b:
                raise ValueError(f"Invalid cached PLE shard {i}")
            obj.tables.append(table)
        obj.key = ttnn.from_torch(
            get("key_proj.weight").T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        obj.value = ttnn.from_torch(
            get("value_proj.weight").T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        for name in ("norm_key", "norm_query", "norm_conv"):
            # HF performs FP32 normalization and FP32 multiplication by (1+w).
            weight = (1.0 + get(name + ".weight").float()).reshape(1, 1, 4, 2560)
            setattr(obj, name, ttnn.from_torch(weight, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device))
        conv = get("conv1d.weight").squeeze(1).T.contiguous().reshape(4, 1, 10240)
        obj.conv = ttnn.from_torch(conv.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        obj.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        return obj

    def make_cache(self, *, max_requests):
        import torch  # Setup only; no calls in forward.

        token = torch.zeros((max_requests, 32), dtype=torch.int32)
        token[:, :2] = self.eos
        return PLECache(
            ttnn.from_torch(token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh),
            ttnn.from_torch(
                torch.zeros((max_requests * 9, 10240), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
            ),
        )

    def prepare(self, *, batch, physical_seq_len):
        """Allocate persistent lookup/tap outputs, before capture or warmup."""
        key = (batch, physical_seq_len)
        if key in self.workspaces:
            return self.workspaces[key]
        if batch < 1 or physical_seq_len < 1:
            raise ValueError("Physical batch and sequence must be positive")
        n = batch * physical_seq_len
        allocate = lambda shape, dtype: ttnn.allocate_tensor_on_device(
            shape, dtype, ttnn.ROW_MAJOR_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        workspace = PLEWorkspace(
            allocate((n, 16), ttnn.uint32), allocate((n, 2560), ttnn.bfloat16), allocate((4, n, 10240), ttnn.bfloat16)
        )
        self.workspaces[key] = workspace
        return workspace

    def _norm(self, x, weight, n):
        x = ttnn.typecast(ttnn.reshape(x, (1, n, 4, 2560)), ttnn.float32)
        variance = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        normalized = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(variance, self.eps)))
        return ttnn.typecast(ttnn.multiply(normalized, weight), ttnn.bfloat16)

    def forward(self, x, *, tokens, positions, slots, valid_lengths, cache):
        batch, seq = tokens.shape
        if tuple(x.shape) != (1, batch, seq, 10240):
            raise ValueError("PLE x must have shape [1,B,S,10240]")
        ws = self.workspaces[(batch, seq)]
        ngram_indices(
            tokens,
            positions,
            slots,
            cache.token_history,
            valid_lengths=valid_lengths,
            output=ws.indices,
            multipliers=self.multipliers,
            primes=self.primes,
            offsets=self.offsets,
            eos=self.eos,
        )
        lookup_rows(self.tables, ws.indices, output=ws.embeddings)
        n = batch * seq
        emb = ttnn.reshape(ttnn.to_layout(ws.embeddings, ttnn.TILE_LAYOUT), (1, 1, n, 2560))
        key = self._norm(ttnn.linear(emb, self.key, compute_kernel_config=self.compute), self.norm_key, n)
        query = self._norm(x, self.norm_query, n)
        # HF reduces a BF16 product, then rounds each gate transform to BF16.
        dot = ttnn.sum(ttnn.multiply(key, query), dim=-1, keepdim=True)
        gate = ttnn.multiply(dot, 2560**-0.5)
        gate = ttnn.multiply(ttnn.sqrt(ttnn.maximum(ttnn.abs(gate), 1e-6)), ttnn.sign(gate))
        gate = ttnn.sigmoid(gate)
        value = ttnn.linear(emb, self.value, compute_kernel_config=self.compute)
        value = ttnn.reshape(value, (1, n, 1, 2560))
        gated = ttnn.reshape(ttnn.multiply(gate, value), (1, 1, n, 10240))
        normalized = self._norm(gated, self.norm_conv, n)
        normalized = ttnn.to_layout(ttnn.reshape(normalized, (n, 10240)), ttnn.ROW_MAJOR_LAYOUT)
        # The gather zeros padding, resets only position-zero requests, and
        # updates all nine historical vectors after reading all old taps.
        _conv_taps(
            normalized, positions, slots, valid_lengths, cache.conv_history, output=ws.taps, batch=batch, seq=seq
        )
        taps = ttnn.to_layout(ws.taps, ttnn.TILE_LAYOUT)
        convolved = ttnn.sum(ttnn.multiply(ttnn.typecast(taps, ttnn.float32), self.conv), dim=0, keepdim=True)
        convolved = ttnn.silu(ttnn.typecast(convolved, ttnn.bfloat16))
        # Mask gated values via the same device controls; tap index 3 is the
        # normalized current token, so cannot be used as a numerical mask.
        gated_rm = ttnn.to_layout(ttnn.reshape(gated, (n, 10240)), ttnn.ROW_MAJOR_LAYOUT)
        _mask_padding(gated_rm, valid_lengths, batch=batch, seq=seq)
        result = ttnn.add(ttnn.to_layout(gated_rm, ttnn.TILE_LAYOUT), ttnn.reshape(convolved, (n, 10240)))
        return ttnn.reshape(result, (1, batch, seq, 10240))


def _conv_taps(x, positions, slots, valid_lengths, history, *, output, batch, seq):
    """Device gather/cache update, separated from tensor-core arithmetic."""
    grid = x.device().compute_with_storage_grid_size()
    if batch > grid.x * grid.y:
        raise ValueError("PLE convolution needs one worker per batch lane")
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(batch)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    tensors = [x, positions, slots, valid_lengths, history, output]
    args = [batch, seq]
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    cb = ttnn.CBDescriptor(
        total_size=20544,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=20544)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).with_name("kernels") / "ple_conv.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))


def _mask_padding(x, valid_lengths, *, batch, seq):
    grid = x.device().compute_with_storage_grid_size()
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(batch)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = [
        seq,
        *ttnn.TensorAccessorArgs(x).get_compile_time_args(),
        *ttnn.TensorAccessorArgs(valid_lengths).get_compile_time_args(),
    ]
    cb = ttnn.CBDescriptor(
        total_size=20544,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=20544)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).with_name("kernels") / "ple_mask.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[
            ((c.x, c.y), [x.buffer_address(), valid_lengths.buffer_address(), i]) for i, c in enumerate(coords)
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([x, valid_lengths], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
